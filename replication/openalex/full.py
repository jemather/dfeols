# SPDX-License-Identifier: GPL-3.0-or-later
"""D14 stage 3: the full author x year panel (all fields, 1950-2025) with KMZ-style institutions.

The KMZ choices of `kmz.py` applied to every author with 2+ active years (Ted 2026-11-22):
  institution  main institution per author-year, lineage rollup, ALIAS (Wisconsin/UT systems), changes
               only at R+P+S moves, carried forward otherwise; gap years filled (zero output, move at the
               gap midpoint)
  variants     (cumulative, --variant)
                 affil   + author-level affiliation years where no paper lists an institution
                 edu     + prefer education-type institutions as the main one
                 lag2    + output credited 2 years before publication
  outcomes     works, frac_works, cites, frac_cites (all kept works, stage 2 rules), and from KMZ's 41
               journals impact_pages (pages x IF / coauthors; KMZ's 1970s IF before 1970, 1990s IF after
               2001) and articles3 (3+ page articles)
Output build/openalex/full/<variant>/b=XX.parquet (one file per author bucket) and summary.json.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/openalex/full.py --variant affil --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

VARIANTS = {"affil": dict(affil=True, edu=False, lag=0), "edu": dict(affil=True, edu=True, lag=0),
            "lag2": dict(affil=True, edu=True, lag=2)}
Y0, Y1 = 1950, 2025


def prep(oa: Path, out: Path, buckets: int) -> None:
    """Once: author_affil and the 41-journal outcomes, partitioned by author bucket."""
    import polars as pl

    from kmz import IF

    c = pl.col
    d = out / "_prep"
    if (d / "done").exists():
        return
    (d / "affil").mkdir(parents=True, exist_ok=True)
    (d / "kmz").mkdir(parents=True, exist_ok=True)
    aa = (pl.scan_parquet(oa / "stage" / "author_affil" / "*.parquet")
          .filter(c("year").is_between(Y0, Y1) & c("institution_id").is_not_null() & c("author_id").is_not_null())
          .unique().with_columns((c("author_id") % buckets).alias("_b")).collect(engine="streaming"))
    for (b,), part in aa.partition_by("_b", as_dict=True).items():
        part.drop("_b").write_parquet(d / "affil" / f"b={b:02d}.parquet")
    del aa
    jb = pl.read_parquet(oa / "kmz" / "journals.parquet")
    dec = pl.when(c("year") >= 1990).then(1990).when(c("year") < 1970).then(1970).otherwise(c("year") // 10 * 10)
    ifd = pl.DataFrame([{"abbr": k, "d": dd, "if": v[i]} for k, v in IF.items() for i, dd in enumerate((1970, 1980, 1990))])
    w = (pl.scan_parquet(oa / "stage" / "works" / "*.parquet").join(jb.lazy(), on="source_id", how="inner")
         .filter(c("type").is_in(["article", "review"]) & c("year").is_between(Y0, Y1)
                 & ~c("is_paratext").fill_null(False) & ~c("is_retracted").fill_null(False))
         .select("work_id", "source_id", "abbr", "year", "authors_count",
                 c("first_page").cast(pl.Int32, strict=False).alias("fp"), c("last_page").cast(pl.Int32, strict=False).alias("lp"))
         .collect(engine="streaming"))
    nxt = (w.filter(c("fp").is_not_null()).select("source_id", "year", "fp").unique().sort("source_id", "year", "fp")
           .with_columns(c("fp").shift(-1).over("source_id", "year").alias("nfp")))
    w = (w.join(nxt, on=["source_id", "year", "fp"], how="left")
         .with_columns(pl.when(c("lp") > c("fp")).then(c("lp") - c("fp") + 1).when(c("nfp") > c("fp")).then(c("nfp") - c("fp")).alias("pages"))
         .with_columns(pl.when(c("pages").is_between(1, 100)).then(c("pages")).alias("pages"), dec.alias("d"))
         .with_columns(c("pages").fill_null(c("pages").median().over("source_id", "d")).fill_null(10))
         .join(ifd, on=["abbr", "d"], how="left"))
    aw = (pl.scan_parquet(oa / "stage" / "authorships" / "*.parquet").select("work_id", "author_id")
          .filter(c("author_id").is_not_null()).join(w.lazy(), on="work_id", how="inner")
          .unique(["work_id", "author_id"]).collect(engine="streaming"))
    fr = c("if") / c("authors_count").clip(lower_bound=1)
    ok = c("pages") >= 3
    y = (aw.group_by("author_id", "year").agg((fr * c("pages")).filter(ok).sum().alias("impact_pages"),
                                              ok.sum().cast(pl.Float64).alias("articles3"))
         .with_columns((c("author_id") % buckets).alias("_b")))
    for (b,), part in y.partition_by("_b", as_dict=True).items():
        part.drop("_b").write_parquet(d / "kmz" / f"b={b:02d}.parquet")
    (d / "done").write_text(json.dumps({"affil_rows": None, "kmz_works": w.height, "kmz_author_years": y.height}))


def bucket(oa: str, out: str, variant: str, b: int) -> dict:
    import polars as pl

    from kmz import ALIAS, TOP25, fill_gaps, main_inst

    c = pl.col
    t0 = time.time()
    oa, out = Path(oa), Path(out)
    v = VARIANTS[variant]
    o = "author_id"
    inst = pl.read_parquet(oa / "stage" / "institutions" / "*.parquet", columns=["institution_id", "lineage", "type"])
    edu = inst.select(c("institution_id").alias("root"), (c("type") == "education").alias("edu"))
    tops = inst.filter(c("lineage").list.len() == 1).select(c("institution_id").alias("lineage"))
    root = (inst.select("institution_id", "lineage").explode("lineage").join(tops, on="lineage", how="semi")
            .sort("institution_id", "lineage").group_by("institution_id").agg(c("lineage").first().alias("root")))

    p = (pl.read_parquet(oa / "panel" / "panel" / f"b={b:02d}.parquet",
                         columns=["author_id", "year", "works", "frac_works", "cites", "frac_cites", "field_id"])
         .filter(c("year").is_between(Y0, Y1)))
    p = p.filter(pl.len().over(o) >= 2)
    span = p.group_by(o).agg(c("year").min().alias("first_year"), c("year").max().alias("last_year"), c("field_id").first())
    u = span.select(o)
    ayi = (pl.scan_parquet(oa / "panel" / "a" / "ayi" / f"b={b:02d}" / "*.parquet")
           .filter(c("year").is_between(Y0, Y1)).join(u.lazy(), on=o, how="semi").collect())
    if v["affil"]:
        fb = (pl.read_parquet(out / "_prep" / "affil" / f"b={b:02d}.parquet").join(u, on=o, how="semi")
              .join(ayi.select(o, "year").unique(), on=[o, "year"], how="anti").with_columns(pl.lit(1.0).alias("weight")))
        ayi = pl.concat([ayi, fb.select(ayi.columns)], how="vertical_relaxed")
    ayi = (ayi.join(root, on="institution_id", how="left").with_columns(c("root").fill_null(c("institution_id")))
           .with_columns(c("root").replace(ALIAS)).join(edu, on="root", how="left").with_columns(c("edu").fill_null(False)))
    m = main_inst(ayi, v["edu"]).sort(o, "year")
    m = m.with_columns(c("inst").shift(1).over(o).alias("p_inst"), c("share").shift(1).over(o).alias("p_share"),
                       c("inst").shift(-1).over(o).alias("n_inst"), c("year").shift(-1).over(o).alias("n_year"),
                       pl.int_range(pl.len()).over(o).alias("k"))
    mv = ((c("inst") != c("p_inst")) & (c("n_year") == c("year") + 1) & (c("n_inst") == c("inst"))
          & (c("share") >= 0.5) & (c("p_share") >= 0.5)).fill_null(False)
    m = m.with_columns(mv.alias("move"), pl.when((c("k") == 0) | mv).then(c("inst")).alias("anchor"))
    s = (p.select(o, "year").join(m.select(o, "year", "anchor", "move"), on=[o, "year"], how="left")
         .sort(o, "year").with_columns(c("anchor").forward_fill().over(o).alias("inst"))
         .filter(c("inst").is_not_null())
         .select(o, "year", "inst", c("move").fill_null(False), pl.lit(False).alias("filled")))
    s = fill_gaps(s, "midpoint", 0)

    lag = v["lag"]
    oc = p.select(o, (c("year") - lag).cast(pl.Int16).alias("year"), "works", "frac_works", "cites", "frac_cites")
    kz = (pl.read_parquet(out / "_prep" / "kmz" / f"b={b:02d}.parquet")
          .with_columns((c("year") - lag).cast(pl.Int16).alias("year")))
    s = (s.join(oc, on=[o, "year"], how="left").join(kz, on=[o, "year"], how="left")
         .with_columns(*[c(x).fill_null(0) for x in ("works", "frac_works", "cites", "frac_cites", "impact_pages", "articles3")])
         .join(span, on=o, how="left")
         .with_columns(c("inst").is_in(list(TOP25)).alias("top25"), (c("year") // 10 * 10).cast(pl.Int16).alias("decade")))
    d = out / variant
    d.mkdir(parents=True, exist_ok=True)
    s.write_parquet(d / f"b={b:02d}.parquet")
    return {"bucket": b, "rows": s.height, "authors": s[o].n_unique(), "filled": int(s["filled"].sum()),
            "moves": int(s["move"].sum()), "top25_rows": int(s["top25"].sum()), "seconds": round(time.time() - t0, 1)}


def main() -> None:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--openalex", type=Path, default=Path("build/openalex"))
    ap.add_argument("--out", type=Path, default=Path("build/openalex/full"))
    ap.add_argument("--variant", choices=list(VARIANTS), nargs="+", default=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--buckets", type=int, default=64)
    ap.add_argument("--only", type=int, nargs="*", help="run these buckets (testing)")
    a = ap.parse_args()
    os.environ["POLARS_MAX_THREADS"] = str(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{time.strftime('%H:%M:%S')} prep", flush=True)
    prep(a.openalex, a.out, a.buckets)
    todo = a.only if a.only else list(range(a.buckets))
    for var in a.variant:
        print(f"{time.strftime('%H:%M:%S')} variant {var}: {VARIANTS[var]}", flush=True)
        res = []
        with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(bucket, str(a.openalex), str(a.out), var, b): b for b in todo}
            for f in as_completed(futs):
                try:
                    r = f.result()
                    res.append(r)
                    print(f"{time.strftime('%H:%M:%S')} {var} {r}", flush=True)
                except Exception as e:
                    print(f"{time.strftime('%H:%M:%S')} FAIL {var} {futs[f]}: {type(e).__name__}: {e}", flush=True)
        tot = {k: sum(r[k] for r in res) for k in ("rows", "authors", "filled", "moves", "top25_rows")}
        (a.out / var / "summary.json").write_text(json.dumps({"variant": var, **VARIANTS[var], "totals": tot,
                                                               "buckets": sorted(res, key=lambda r: r["bucket"])}, indent=1))
        print(json.dumps({var: tot}), flush=True)


if __name__ == "__main__":
    main()
