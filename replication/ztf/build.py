# SPDX-License-Identifier: GPL-3.0-or-later
"""ZTF stage 2: staged detections -> analysis file for the photometric-calibration specs.

    python -I repo/replication/ztf/build.py --stage build/ztf/stage --out build/ztf/analysis.parquet

Steps (all checks print; see README.md for the values on the dial-in partition):

1. Exposure IDs. The light curves carry no exposure ID, only hmjd (heliocentric MJD at the
   object's position). Within one CCD quadrant (fieldid, rcid, filterid) the heliocentric
   correction varies by < ~10 s, and exposures are >= ~45 s apart, so sorted distinct hmjd values
   separated by more than EXP_GAP start a new exposure: ``expid``. Checked: no object twice in one
   exposure; the largest hmjd spread within an exposure.
2. Visits. Quadrant-exposures of the same field and filter whose median times are within
   VISIT_GAP are one telescope exposure across the focal plane: ``visit``.
3. Quality cuts: catflags == 0 (no flagged epochs, which includes clouds/moon, bit 15), and
   0 < magerr < MAX_MAGERR.
4. Colour. ``mag`` is calibrated for a source with g - r = 0; the colour correction is
   mag + clrcoeff * (g - r). Each object's g - r is the difference of the median good-epoch mags of
   a g and an r object in the same field and quadrant within MATCH_ARCSEC (one-to-one matches only).
   Objects without a partner are dropped; g - r is kept in GR_RANGE.
5. Airmass at Palomar from hmjd (treated as MJD; < 8 min off, fine for airmass) and object
   position; kept in AIRMASS_RANGE.
6. Instrument. ``clrcoeff`` is fitted per quadrant-exposure, so much of its variation over time is
   fitting noise. ``cc_loo`` is the mean clrcoeff of the *other* quadrants in the same visit (same
   atmosphere, independent fitting errors); ``z_gr = cc_loo * gr`` instruments ``cc_gr``. Rows
   whose visit has a single quadrant (no instrument; ~1.4% on the dial-in partition) are dropped,
   so OLS and IV specs share one sample.

Output columns: objectid filterid fieldid rcid expid visit season hmjd mag magerr w clrcoeff gr
cc_gr cc_loo z_gr airmass mobj. ``w = 1/magerr^2``; ``season`` = whole years since 2018-01-01;
``mobj`` = the object's median mag in the sample.

This runs in memory (fine for tens of partitions). For the full release, steps 1-2 need the
distinct-time table built per quadrant group and the rest done lazily per partition.
"""

from __future__ import annotations

import argparse
import os
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


def exposure_ids(d: pl.DataFrame) -> pl.DataFrame:
    t = (
        d.select(*QUAD, "hmjd")
        .unique()
        .sort(*QUAD, "hmjd")
        .with_columns(gap=pl.col("hmjd").diff().over(QUAD).fill_null(1.0))
    )
    amb = t.filter(pl.col("gap").is_between(5e-5, 1e-3)).height
    t = t.with_columns(expid=(pl.col("gap") > EXP_GAP).cum_sum().cast(pl.Int64))
    d = d.join(t.select(*QUAD, "hmjd", "expid"), on=[*QUAD, "hmjd"])
    dup = d.group_by("expid", "objectid").len().filter(pl.col("len") > 1).height
    span = d.group_by("expid").agg((pl.col("hmjd").max() - pl.col("hmjd").min()).alias("s"))["s"].max()
    print(
        f"exposures {d['expid'].n_unique():,}; ambiguous gaps (5e-5..1e-3 d) {amb}; "
        f"object twice in an exposure {dup}; max hmjd spread in an exposure {span * 86400:.1f}s"
    )
    return d


def visits(d: pl.DataFrame) -> pl.DataFrame:
    """One row per quadrant-exposure with its visit and the leave-one-out clrcoeff of its visit."""
    q = (
        d.group_by(*QUAD, "expid")
        .agg(
            pl.col("hmjd").median().alias("t"),
            pl.col("clrcoeff").first(),
            pl.col("clrcoeff").n_unique().alias("ncc"),
        )
        .sort("fieldid", "filterid", "t")
        .with_columns(gap=pl.col("t").diff().over("fieldid", "filterid").fill_null(1.0))
        .with_columns(visit=(pl.col("gap") > VISIT_GAP).cum_sum().cast(pl.Int64))
    )
    print(f"clrcoeff constant within every quadrant-exposure: {bool((q['ncc'] == 1).all())}")
    multi = q.group_by("visit", "rcid").len().filter(pl.col("len") > 1).height
    print(f"visits {q['visit'].n_unique():,}; (visit, rcid) with more than one exposure: {multi}")
    q = q.with_columns(s=pl.col("clrcoeff").sum().over("visit"), n=pl.len().over("visit")).with_columns(
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
    return q.select("expid", "visit", "cc_loo")


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
    p = pl.concat(pairs).unique("r", keep="none").unique("g", keep="none")
    print(f"g/r matched pairs {p.height:,}")
    both = [p.select(pl.col(f).alias("objectid"), "gr") for f in ("g", "r")]
    return pl.concat(both)


def airmass(d: pl.DataFrame) -> pl.Series:
    lat = np.radians(PALOMAR_LAT)
    jd = d["hmjd"].to_numpy() + 2400000.5
    lst = np.radians((280.46061837 + 360.98564736629 * (jd - 2451545.0) + PALOMAR_LON) % 360)
    ha = lst - np.radians(d["objra"].to_numpy())
    de = np.radians(d["objdec"].to_numpy())
    return pl.Series("airmass", 1 / (np.sin(lat) * np.sin(de) + np.cos(lat) * np.cos(de) * np.cos(ha)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(str(p) for p in Path(a.stage).glob("Norder=*_Npix=*.parquet"))
    d = pl.read_parquet(files)
    print(f"{len(files)} partition(s), {d.height:,} detections")
    d = exposure_ids(d)
    vis = visits(d)  # all detections, before cuts
    d = d.filter((pl.col("catflags") == 0) & (pl.col("magerr") > 0) & (pl.col("magerr") < MAX_MAGERR))
    print(f"good detections {d.height:,}")
    d = d.join(colours(d), on="objectid").filter(pl.col("gr").is_between(*GR_RANGE))
    d = d.with_columns(airmass(d)).filter(pl.col("airmass").is_between(*AIRMASS_RANGE))
    d = (
        d.join(vis, on="expid")
        .with_columns(
            w=1 / pl.col("magerr") ** 2,
            season=((pl.col("hmjd") - MJD_2018) / 365.25).floor().cast(pl.Int16),
            cc_gr=pl.col("clrcoeff") * pl.col("gr"),
            z_gr=pl.col("cc_loo") * pl.col("gr"),
        )
        .filter(pl.col("z_gr").is_not_null())
    )
    d = d.with_columns(mobj=pl.col("mag").median().over("objectid"))
    am = d["airmass"]
    print(
        f"analysis rows {d.height:,}; objects {d['objectid'].n_unique():,}; exposures "
        f"{d['expid'].n_unique():,}; visits {d['visit'].n_unique():,}; "
        f"airmass {am.min():.2f}..{am.max():.2f}"
    )
    cols = [
        "objectid", "filterid", "fieldid", "rcid", "expid", "visit", "season", "hmjd", "mag", "magerr",
        "w", "clrcoeff", "gr", "cc_gr", "cc_loo", "z_gr", "airmass", "mobj",
    ]  # fmt: skip
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.select(cols).write_parquet(a.out)


if __name__ == "__main__":
    main()
