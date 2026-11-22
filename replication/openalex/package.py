# SPDX-License-Identifier: GPL-3.0-or-later
"""D14: package the full-panel edu variant for OSCER, with every top-25 definition (Ted 2026-11-22).

Input build/openalex/full/edu/b=XX.parquet (`full.py`). One row per author x year, 1950-2025, every
author with 2+ active years. Added columns:
  field_id     modal OpenAlex field (26 fields, 11-36), kept as is
  top_kmz      main institution is one of KMZ's 25 economics schools
  top_fYYYY    main institution is one of the author's field's top 25 education institutions ranked
               by frac_cites of authors whose (unfilled) main institution it was in decade YYYY,
               fixed for all years; YYYY = 1970, 1980, 1990, 2000, 2010
  top_fdec     the same ranking taken from the row's own decade (1950s-2020s; 2020s partial)
  ever_<def>   author has 2+ unfilled years in 1970-2025 at a top institution under <def>
  core         career span (last_year - first_year) 5+ years and 3+ works in 1970-2025
  frac_works_n, frac_cites_n
               outcome / its mean over all rows of the same field and year (all authors)
  field_decade field_id * 10000 + decade (pooled field x decade FE)
  cy, cy2, cy3 career year (year - first_year), its square / 100, cube / 1000
Lookups: lookups/fields.csv, lookups/tops.csv (every ranking: definition, field, rank, institution).
Parts: data/part-XXXX.parquet, authors whole within a part (author_id % 64 buckets, grouped).

    cd /srv/dfeols && .venv/bin/python -I repo/replication/openalex/package.py --out outbox/openalex_kmz_panel_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from all_fields import FIELDS  # noqa: E402

c = pl.col
BASES = (1970, 1980, 1990, 2000, 2010)
DEFS = ["kmz", *[f"f{b}" for b in BASES], "fdec"]


def rankings(full: Path, oa: Path) -> pl.DataFrame:
    inst = pl.read_parquet(oa / "stage" / "institutions" / "*.parquet", columns=["institution_id", "display_name", "type"])
    edu = inst.filter(c("type") == "education").select(c("institution_id").alias("inst"), "display_name")
    by_dec = (pl.scan_parquet(full / "*.parquet").filter(~c("filled") & c("field_id").is_in(list(FIELDS)))
              .join(edu.lazy().select("inst"), on="inst", how="semi")
              .group_by("field_id", "decade", "inst").agg(c("frac_cites").sum())
              .collect(engine="streaming")
              .sort(["field_id", "decade", "frac_cites", "inst"], descending=[False, False, True, False])
              .group_by("field_id", "decade", maintain_order=True).head(25)
              .with_columns((pl.int_range(pl.len()).over("field_id", "decade") + 1).alias("rank")))
    return by_dec.join(edu, on="inst", how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openalex", type=Path, default=Path("build/openalex"))
    ap.add_argument("--full", type=Path, default=Path("build/openalex/full/edu"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    (a.out / "data").mkdir(parents=True, exist_ok=True)
    (a.out / "lookups").mkdir(parents=True, exist_ok=True)
    rk = rankings(a.full, a.openalex)
    rk.select("field_id", "decade", "rank", "inst", "display_name", "frac_cites").write_csv(a.out / "lookups" / "tops.csv")
    pl.DataFrame({"field_id": list(FIELDS), "field": list(FIELDS.values())}).write_csv(a.out / "lookups" / "fields.csv")
    means = (pl.scan_parquet(a.full / "*.parquet").group_by("field_id", "year")
             .agg(c("frac_works").mean().alias("_mw"), c("frac_cites").mean().alias("_mc")).collect(engine="streaming"))
    fixed = pl.concat([rk.filter(c("decade") == b).select("field_id", "inst", pl.lit(True).alias(f"top_f{b}")) for b in BASES],
                      how="diagonal")
    fixed = fixed.group_by("field_id", "inst").agg(*[c(f"top_f{b}").any() for b in BASES])
    dec = rk.select("field_id", "decade", "inst", pl.lit(True).alias("top_fdec"))
    print(f"{time.time() - t0:.0f}s rankings and means", flush=True)
    files = sorted(a.full.glob("b=*.parquet"))
    per = -(-len(files) // a.parts)
    summary = []
    for p in range(a.parts):
        s = pl.concat([pl.read_parquet(f) for f in files[p * per:(p + 1) * per]])
        s = (s.rename({"top25": "top_kmz"})
             .join(fixed, on=["field_id", "inst"], how="left").join(dec, on=["field_id", "decade", "inst"], how="left")
             .with_columns(*[c(f"top_{d}").fill_null(False) for d in DEFS if d != "kmz"]))
        w = s.filter(c("year").is_between(1970, 2025))
        ev = w.filter(~c("filled")).group_by("author_id").agg(*[(c(f"top_{d}").sum() >= 2).alias(f"ever_{d}") for d in DEFS],
                                                              c("works").sum().alias("_k"))
        ev = ev.join(s.group_by("author_id").agg((c("last_year") - c("first_year")).first().alias("_span")), on="author_id")
        ev = ev.with_columns(((c("_k") >= 3) & (c("_span") >= 5)).alias("core")).drop("_k", "_span")
        cy = (c("year") - c("first_year")).cast(pl.Float64)
        s = (s.join(ev, on="author_id", how="left")
             .with_columns(*[c(f"ever_{d}").fill_null(False) for d in DEFS], c("core").fill_null(False))
             .join(means, on=["field_id", "year"], how="left")
             .with_columns((c("frac_works") / c("_mw")).alias("frac_works_n"), (c("frac_cites") / c("_mc")).alias("frac_cites_n"),
                           (c("field_id").cast(pl.Int32) * 10000 + c("decade")).alias("field_decade"),
                           cy.alias("cy"), (cy**2 / 100).alias("cy2"), (cy**3 / 1000).alias("cy3"))
             .drop("_mw", "_mc").sort("author_id", "year"))
        f = a.out / "data" / f"part-{p:04d}.parquet"
        s.write_parquet(f, row_group_size=1_000_000)
        summary.append({"file": f"data/{f.name}", "rows": s.height, "authors": s["author_id"].n_unique(),
                        "buckets": [x.stem for x in files[p * per:(p + 1) * per]][::max(1, per - 1)],
                        "bytes": f.stat().st_size, **{f"ever_{d}_authors": int(ev[f"ever_{d}"].sum()) for d in DEFS},
                        "core_authors": int(ev["core"].sum())})
        print(f"{time.time() - t0:.0f}s {summary[-1]}", flush=True)
        del s
    (a.out / "build_summary.json").write_text(json.dumps({"parts": summary}, indent=1))
    print(f"{time.time() - t0:.0f}s done", flush=True)


if __name__ == "__main__":
    main()
