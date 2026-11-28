# SPDX-License-Identifier: GPL-3.0-or-later
"""ZTF stage 2: staged detections -> analysis files for the photometric-calibration specs.

    python -I repo/replication/ztf/build.py --stage build/ztf/stage --out build/ztf/analysis [--workers 4]

Writes ``<out>/times/`` (small per-partition tables, step 1), ``<out>/exposures.parquet`` and
``<out>/data/<partition>.parquet``, one analysis file per staged partition. Read them back as the
glob ``<out>/data/*.parquet``.

Steps (all checks print; see README.md for the values on the dial-in partition):

1. Exposure IDs. The light curves carry no exposure ID, only hmjd (heliocentric MJD at the
   object's position). Within one CCD quadrant (fieldid, rcid, filterid) the heliocentric
   correction varies by < ~10 s, and exposures are >= ~45 s apart, so sorted distinct hmjd values
   separated by more than EXP_GAP start a new exposure: ``expid``. A quadrant's exposure spans
   partitions, so each partition contributes its distinct (quadrant, hmjd) values with counts
   (about 88k for 34M detections), and the clustering runs on their union. Checked: no object
   twice in one exposure; the largest hmjd spread within an exposure.
2. Visits. Quadrant-exposures of the same field and filter whose median times are within
   VISIT_GAP are one telescope exposure across the focal plane: ``visit``. The median is exact,
   taken from the counts of step 1 (the mean of the two middle values when the count is even).
3. Quality cuts: catflags == 0 (no flagged epochs, which includes clouds/moon, bit 15), and
   0 < magerr < MAX_MAGERR.
4. Colour. ``mag`` is calibrated for a source with g - r = 0; the colour correction is
   mag + clrcoeff * (g - r). Each object's g - r is the difference of the median good-epoch mags of
   a g and an r object in the same field and quadrant within MATCH_ARCSEC (one-to-one matches only).
   Objects without a partner are dropped; g - r is kept in GR_RANGE. Matching is within a
   partition (each object belongs to one), so objects at partition edges can lose their partner.
5. Airmass at Palomar from hmjd (treated as MJD; < 8 min off, fine for airmass) and object
   position; kept in AIRMASS_RANGE.
6. Instrument. ``clrcoeff`` is fitted per quadrant-exposure, so much of its variation over time is
   fitting noise. ``cc_loo`` is the mean clrcoeff of the *other* quadrants in the same visit (same
   atmosphere, independent fitting errors); ``z_gr = cc_loo * gr`` instruments ``cc_gr``. Rows
   whose visit has a single quadrant (no instrument; ~1.4% on the dial-in partition) are dropped,
   so OLS and IV specs share one sample.

Steps 1-2 need all partitions; steps 3-6 run one partition at a time (``--workers`` in parallel,
each holding one partition in memory: ~10 GB peak for a 34M-detection partition).

Output columns: objectid filterid fieldid rcid expid visit season hmjd mag magerr w clrcoeff gr
cc_gr cc_loo z_gr airmass mobj. ``w = 1/magerr^2``; ``season`` = whole years since 2018-01-01;
``mobj`` = the object's median mag in the sample.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

EXP_GAP = 2.5e-4  # days (~22 s)
VISIT_GAP = 2e-3  # days (~3 min)
MAX_MAGERR = 0.2
MATCH_ARCSEC = 1.5
GR_RANGE = (-0.5, 2.5)
AIRMASS_RANGE = (1.0, 3.0)
PALOMAR_LAT, PALOMAR_LON = 33.3563, -116.8648  # deg (east positive)
MJD_2018 = 58119.0
QUAD = ["fieldid", "rcid", "filterid"]
COLS = [
    "objectid", "filterid", "fieldid", "rcid", "expid", "visit", "season", "hmjd", "mag", "magerr",
    "w", "clrcoeff", "gr", "cc_gr", "cc_loo", "z_gr", "airmass", "mobj",
]  # fmt: skip


def times_one(src: str, dst: str) -> str:
    """Step 1, one partition: distinct (quadrant, hmjd) with counts and clrcoeff range."""
    (
        pl.scan_parquet(src)
        .group_by(*QUAD, "hmjd")
        .agg(
            pl.len().alias("n"),
            pl.col("clrcoeff").min().alias("cmin"),
            pl.col("clrcoeff").max().alias("cmax"),
        )
        .sink_parquet(dst)
    )
    return dst


def exposures(times: list[str]) -> pl.DataFrame:
    """Steps 1-2 on the union of the partitions' times: one row per (quadrant, hmjd) with expid,
    visit and cc_loo."""
    t = (
        pl.scan_parquet(times)
        .group_by(*QUAD, "hmjd")
        .agg(pl.col("n").sum(), pl.col("cmin").min(), pl.col("cmax").max())
        .sort(*QUAD, "hmjd")
        .collect()
        .with_columns(gap=pl.col("hmjd").diff().over(QUAD).fill_null(1.0))
    )
    amb = t.filter(pl.col("gap").is_between(5e-5, 1e-3)).height
    t = t.with_columns(expid=(pl.col("gap") > EXP_GAP).cum_sum().cast(pl.Int64))
    q = (
        t.with_columns(
            cum=pl.col("n").cum_sum().over("expid"),
            k1=(pl.col("n").sum().over("expid") - 1) // 2,
            k2=pl.col("n").sum().over("expid") // 2,
        )
        .group_by(*QUAD, "expid")
        .agg(
            t=(
                pl.col("hmjd").filter(pl.col("cum") > pl.col("k1")).first()
                + pl.col("hmjd").filter(pl.col("cum") > pl.col("k2")).first()
            )
            / 2,
            clrcoeff=pl.col("cmin").min(),  # Float32 as stored; summed in float64 below
            same=(pl.col("cmin").min() == pl.col("cmax").max()),
            span=pl.col("hmjd").max() - pl.col("hmjd").min(),
        )
        .sort("fieldid", "filterid", "t")
        .with_columns(gap=pl.col("t").diff().over("fieldid", "filterid").fill_null(1.0))
        .with_columns(visit=(pl.col("gap") > VISIT_GAP).cum_sum().cast(pl.Int64))
    )
    print(
        f"exposures {q.height:,}; ambiguous gaps (5e-5..1e-3 d) {amb}; "
        f"max hmjd spread in an exposure {q['span'].max() * 86400:.1f}s"
    )
    print(f"clrcoeff constant within every quadrant-exposure: {bool(q['same'].all())}")
    multi = q.group_by("visit", "rcid").len().filter(pl.col("len") > 1).height
    print(f"visits {q['visit'].n_unique():,}; (visit, rcid) with more than one exposure: {multi}")
    # float64 sum: a Float32 sum over the visit rounds at ~1e-7 and depends on row order
    q = q.with_columns(
        s=pl.col("clrcoeff").cast(pl.Float64).sum().over("visit"), n=pl.len().over("visit")
    ).with_columns(
        cc_loo=pl.when(pl.col("n") > 1).then((pl.col("s") - pl.col("clrcoeff")) / (pl.col("n") - 1))
    )
    w = q.filter(pl.col("cc_loo").is_not_null()).with_columns(
        a=pl.col("clrcoeff") - pl.col("clrcoeff").mean().over(QUAD),
        b=pl.col("cc_loo") - pl.col("cc_loo").mean().over(QUAD),
    )
    print(
        f"within-quadrant sd of clrcoeff {w['a'].std():.4f}; "
        f"corr(own, other quadrants') {w.select(pl.corr('a', 'b')).item():.3f}"
    )
    return t.select(*QUAD, "hmjd", "expid").join(q.select("expid", "visit", "cc_loo"), on="expid")


def colours(d: pl.DataFrame) -> pl.DataFrame:
    obj = d.group_by("objectid", *QUAD).agg(
        pl.col("objra").first(), pl.col("objdec").first(), pl.col("mag").median().alias("mmed")
    )
    g, r = obj.filter(pl.col("filterid") == 1), obj.filter(pl.col("filterid") == 2)

    def xyz(t: pl.DataFrame) -> np.ndarray:
        ra, de = np.radians(t["objra"].to_numpy()), np.radians(t["objdec"].to_numpy())
        return np.c_[np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)]

    rq = r.partition_by("fieldid", "rcid", as_dict=True)
    pairs = []
    for (fid, rc), gg in g.group_by("fieldid", "rcid"):
        rr = rq.get((fid, rc))
        if rr is None:
            continue
        dist, j = cKDTree(xyz(rr)).query(xyz(gg), distance_upper_bound=np.radians(MATCH_ARCSEC / 3600))
        ok = np.isfinite(dist)
        pairs.append(
            pl.DataFrame(
                {
                    "g": gg["objectid"].to_numpy()[ok],
                    "r": rr["objectid"].to_numpy()[j[ok]],
                    "gr": gg["mmed"].to_numpy()[ok] - rr["mmed"].to_numpy()[j[ok]],
                }
            )
        )
    if not pairs:  # no field-quadrant holds both a g and an r object
        return pl.DataFrame(schema={"objectid": pl.Int64, "gr": obj["mmed"].dtype}), 0
    p = pl.concat(pairs).unique("r", keep="none").unique("g", keep="none")
    both = [p.select(pl.col(f).alias("objectid"), "gr") for f in ("g", "r")]
    return pl.concat(both), p.height


def airmass(d: pl.DataFrame) -> pl.Series:
    lat = np.radians(PALOMAR_LAT)
    jd = d["hmjd"].to_numpy() + 2400000.5
    lst = np.radians((280.46061837 + 360.98564736629 * (jd - 2451545.0) + PALOMAR_LON) % 360)
    ha = lst - np.radians(d["objra"].to_numpy())
    de = np.radians(d["objdec"].to_numpy())
    return pl.Series("airmass", 1 / (np.sin(lat) * np.sin(de) + np.cos(lat) * np.cos(de) * np.cos(ha)))


def analysis_one(src: str, exp: str, dst: str, threads: int) -> dict:
    """Steps 3-6, one partition."""
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    d = pl.read_parquet(src)
    n_det = d.height
    d = d.join(pl.read_parquet(exp), on=[*QUAD, "hmjd"], how="left")
    assert d["expid"].null_count() == 0, f"{src}: detections without an exposure"
    dup = d.group_by("expid", "objectid").len().filter(pl.col("len") > 1).height
    d = d.filter((pl.col("catflags") == 0) & (pl.col("magerr") > 0) & (pl.col("magerr") < MAX_MAGERR))
    n_good = d.height
    col, n_pairs = colours(d)
    d = d.join(col, on="objectid").filter(pl.col("gr").is_between(*GR_RANGE))
    d = d.with_columns(airmass(d)).filter(pl.col("airmass").is_between(*AIRMASS_RANGE))
    d = (
        d.with_columns(
            w=1 / pl.col("magerr") ** 2,
            season=((pl.col("hmjd") - MJD_2018) / 365.25).floor().cast(pl.Int16),
            cc_gr=pl.col("clrcoeff") * pl.col("gr"),
            z_gr=pl.col("cc_loo") * pl.col("gr"),
        )
        .filter(pl.col("z_gr").is_not_null())
        .with_columns(mobj=pl.col("mag").median().over("objectid"))
    )
    tmp = Path(dst).with_suffix(".part")
    d.select(COLS).write_parquet(tmp)
    tmp.rename(dst)
    return dict(
        det=n_det, dup=dup, good=n_good, pairs=n_pairs, rows=d.height, objects=d["objectid"].n_unique()
    )


def run(pool: ProcessPoolExecutor, fn, jobs: list[tuple], label: str) -> list:
    t0, out = time.time(), []
    futs = [pool.submit(fn, *j) for j in jobs]
    for i, f in enumerate(as_completed(futs), 1):
        out.append(f.result())
        if i % 50 == 0 or i == len(futs):
            print(f"{label}: {i}/{len(futs)} partitions, {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8, help="POLARS_MAX_THREADS per worker")
    a = ap.parse_args()

    files = sorted(Path(a.stage).glob("Norder=*_Npix=*.parquet"))
    out = Path(a.out)
    for sub in ("times", "data"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} partition(s)", flush=True)
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as pool:
        times = run(pool, times_one, [(str(f), str(out / "times" / f.name)) for f in files], "times")
        exp = exposures(sorted(times))
        exp.write_parquet(out / "exposures.parquet")
        del exp
        jobs = [
            (str(f), str(out / "exposures.parquet"), str(out / "data" / f.name), a.threads) for f in files
        ]
        res = run(pool, analysis_one, jobs, "analysis")
    tot = {k: sum(r[k] for r in res) for k in res[0]}
    print(f"detections {tot['det']:,}; object twice in an exposure {tot['dup']}")
    print(f"good detections {tot['good']:,}; g/r matched pairs {tot['pairs']:,}")
    s = pl.scan_parquet(out / "data" / "*.parquet").select(
        pl.col("expid").n_unique(), pl.col("visit").n_unique(), pl.col("airmass").min().alias("amin"),
        pl.col("airmass").max().alias("amax"),
    ).collect(engine="streaming")  # fmt: skip
    print(
        f"analysis rows {tot['rows']:,}; objects {tot['objects']:,}; exposures {s['expid'].item():,}; "
        f"visits {s['visit'].item():,}; airmass {s['amin'].item():.2f}..{s['amax'].item():.2f}"
    )


if __name__ == "__main__":
    main()
