# SPDX-License-Identifier: GPL-3.0-or-later
"""D14 stage 2: author x year panel with a main institution, from the stage-1 tables.

Sample rules are options (defaults PROPOSED, not yet agreed with Ted; docs/openalex.md):
  --types        work types kept (default article review)
  --years        publication years kept (default 1950 2025)
  retracted and paratext works dropped; works without a year dropped.

Stage A (per stage-1 file pair, parallel): works filtered by the rules, joined to that file's
authorships (a work's authorships come from the same input file), aggregated, and written in
--buckets buckets by author_id % buckets:
  ay/   author_id, year, works, frac_works (sum 1/authors_count), cites, frac_cites
  ayi/  author_id, year, institution_id, weight (sum over works of 1/n_inst of that authorship)
  af/   author_id, field_id, works
Stage B (per bucket): sums across files, then per author-year the main institution = largest
weight (ties: smaller institution_id), its share of the year's weight, the number of
institutions; per author the modal field; and along each author's active years
  move    main institution differs from the previous active year's
  blip    a one-year excursion: differs from both neighbours, which agree (A, B, A in
          consecutive years), usually a dual affiliation or a disambiguation error
Output build/openalex/panel/panel/b=XX.parquet, plus summary.json.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/openalex/panel.py \
        --stage build/openalex/stage --out build/openalex/panel --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path


def stage_a(stage: str, out: str, name: str, types: list[str], years: tuple[int, int], buckets: int) -> dict:
    import polars as pl

    c = pl.col
    t0 = time.time()
    w = (
        pl.scan_parquet(f"{stage}/works/{name}.parquet")
        .filter(
            c("type").is_in(types)
            & c("year").is_between(*years)
            & ~c("is_retracted").fill_null(False)
            & ~c("is_paratext").fill_null(False)
        )
        .select("work_id", "cited_by_count", "authors_count", "field_id")
    )
    a = (
        pl.scan_parquet(f"{stage}/authorships/{name}.parquet")
        .filter(c("author_id").is_not_null())
        .select("work_id", "year", "author_id", "n_inst", "institution_id")
        .join(w, on="work_id", how="inner")
        .collect()
    )
    aw = a.unique(["work_id", "author_id"])  # one row per authorship (rows repeat per institution)
    fr = 1.0 / c("authors_count").cast(pl.Float64).clip(lower_bound=1)
    tables = {
        "ay": aw.group_by("author_id", "year").agg(
            pl.len().cast(pl.Int32).alias("works"),
            fr.sum().alias("frac_works"),
            c("cited_by_count").sum().cast(pl.Int64).alias("cites"),
            (c("cited_by_count") * fr).sum().alias("frac_cites"),
        ),
        "ayi": a.filter(c("institution_id").is_not_null())
        .group_by("author_id", "year", "institution_id")
        .agg((1.0 / c("n_inst").cast(pl.Float64)).sum().alias("weight")),
        "af": aw.filter(c("field_id").is_not_null()).group_by("author_id", "field_id").agg(pl.len().cast(pl.Int32).alias("works")),
    }
    rows = {}
    for tn, t in tables.items():
        t = t.with_columns((c("author_id") % buckets).alias("_b"))
        rows[tn] = t.height
        for (b,), part in t.partition_by("_b", as_dict=True).items():
            d = Path(out) / "a" / tn / f"b={b:02d}"
            d.mkdir(parents=True, exist_ok=True)
            part.drop("_b").write_parquet(d / f"{name}.parquet")
    done = Path(out) / "a" / "_done"
    done.mkdir(parents=True, exist_ok=True)
    rec = {"name": name, "authorships": a.height, "rows": rows, "seconds": round(time.time() - t0, 1)}
    (done / f"{name}.json").write_text(json.dumps(rec))
    return rec


def stage_b(out: str, b: int) -> dict:
    import polars as pl

    c = pl.col
    t0 = time.time()
    src = Path(out) / "a"
    ay = (
        pl.scan_parquet(src / "ay" / f"b={b:02d}" / "*.parquet")
        .group_by("author_id", "year")
        .agg(c("works").sum(), c("frac_works").sum(), c("cites").sum(), c("frac_cites").sum())
    )
    ayi = (
        pl.scan_parquet(src / "ayi" / f"b={b:02d}" / "*.parquet")
        .group_by("author_id", "year", "institution_id")
        .agg(c("weight").sum())
        .collect()
    )
    main = (
        ayi.sort(["author_id", "year", "weight", "institution_id"], descending=[False, False, True, False])
        .group_by("author_id", "year", maintain_order=True)
        .agg(
            c("institution_id").first(),
            (c("weight").first() / c("weight").sum()).alias("main_share"),
            pl.len().cast(pl.Int16).alias("n_inst"),
        )
    )
    field = (
        pl.scan_parquet(src / "af" / f"b={b:02d}" / "*.parquet")
        .group_by("author_id", "field_id")
        .agg(c("works").sum())
        .sort(["author_id", "works", "field_id"], descending=[False, True, False])
        .group_by("author_id", maintain_order=True)
        .agg(c("field_id").first())
        .collect()
    )
    p = (
        ay.collect()
        .join(main, on=["author_id", "year"], how="left")
        .join(field, on="author_id", how="left")
        .sort("author_id", "year")
    )
    # moves along each author's years with a main institution
    q = p.filter(c("institution_id").is_not_null()).with_columns(
        c("institution_id").shift(1).over("author_id").alias("_prev"),
        c("institution_id").shift(-1).over("author_id").alias("_next"),
        c("year").shift(1).over("author_id").alias("_py"),
        c("year").shift(-1).over("author_id").alias("_ny"),
    ).with_columns(
        (c("_prev").is_not_null() & (c("institution_id") != c("_prev"))).alias("move"),
        (
            c("_prev").is_not_null()
            & (c("_prev") == c("_next"))
            & (c("institution_id") != c("_prev"))
            & (c("year") - c("_py") == 1)
            & (c("_ny") - c("year") == 1)
        ).alias("blip"),
    )
    p = p.join(q.select("author_id", "year", "move", "blip"), on=["author_id", "year"], how="left").with_columns(
        c("move").fill_null(False), c("blip").fill_null(False)
    )
    d = Path(out) / "panel"
    d.mkdir(parents=True, exist_ok=True)
    p.write_parquet(d / f"b={b:02d}.parquet")
    movers = p.group_by("author_id").agg(c("move").any()).filter(c("move")).height
    return {
        "bucket": b,
        "author_years": p.height,
        "authors": p["author_id"].n_unique(),
        "author_years_with_inst": int(p["institution_id"].is_not_null().sum()),
        "moves": int(p["move"].sum()),
        "blips": int(p["blip"].sum()),
        "movers": movers,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--buckets", type=int, default=64)
    ap.add_argument("--types", nargs="+", default=["article", "review"])
    ap.add_argument("--years", nargs=2, type=int, default=[1950, 2025])
    ap.add_argument("--limit", type=int, default=0, help="stage A on at most this many files (testing)")
    ap.add_argument("--only", choices=["a", "b"], help="run one stage")
    a = ap.parse_args()
    os.environ["POLARS_MAX_THREADS"] = str(a.threads)
    names = sorted(p.stem for p in (a.stage / "_done" / "works").glob("*.json"))
    if a.limit:
        names = names[: a.limit]
    cfg = {"types": a.types, "years": a.years, "buckets": a.buckets, "files": len(names)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(cfg, indent=1))
    ctx = get_context("spawn")
    if a.only in (None, "a"):
        todo = [n for n in names if not (a.out / "a" / "_done" / f"{n}.json").exists()]
        print(f"{time.strftime('%H:%M:%S')} stage A: {len(todo)} of {len(names)} files", flush=True)
        with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
            futs = {ex.submit(stage_a, str(a.stage), str(a.out), n, a.types, tuple(a.years), a.buckets): n for n in todo}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    r = f.result()
                    if i % 50 == 0 or i == len(todo):
                        print(f"{time.strftime('%H:%M:%S')} A {i}/{len(todo)} {r['name']} {r['authorships']:,} {r['seconds']}s", flush=True)
                except Exception as e:
                    print(f"{time.strftime('%H:%M:%S')} FAIL A {futs[f]}: {type(e).__name__}: {e}", flush=True)
    if a.only in (None, "b"):
        print(f"{time.strftime('%H:%M:%S')} stage B: {a.buckets} buckets", flush=True)
        res = []
        with ProcessPoolExecutor(max(1, a.workers // 2), mp_context=ctx) as ex:
            futs = {ex.submit(stage_b, str(a.out), b): b for b in range(a.buckets)}
            for f in as_completed(futs):
                try:
                    r = f.result()
                    res.append(r)
                    print(f"{time.strftime('%H:%M:%S')} B {r}", flush=True)
                except Exception as e:
                    print(f"{time.strftime('%H:%M:%S')} FAIL B {futs[f]}: {type(e).__name__}: {e}", flush=True)
        tot = {k: sum(r[k] for r in res) for k in ("author_years", "authors", "author_years_with_inst", "moves", "blips", "movers")}
        (a.out / "summary.json").write_text(json.dumps({"config": cfg, "totals": tot, "buckets": sorted(res, key=lambda r: r["bucket"])}, indent=1))
        print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
