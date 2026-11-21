# SPDX-License-Identifier: GPL-3.0-or-later
"""D13: Weidner & Zylkin (2021, JIE) Table 4 data, from their replication package.

Reads `baci_example.dta` (from `raw/papers/Code repository for Weidner and Zylkin JIE
2021.zip`, folder "Empirical application (Table 4)", unzipped to tmp/wz_pkg/ as data; none of
the package's do-files are run) in chunks, and writes it to Parquet with integer codes:

    cd /srv/dfeols && .venv/bin/python -I repo/replication/wz2021/build.py \
        --dta "tmp/wz_pkg/Empirical application (Table 4)/baci_example.dta" --out build/wz2021

Columns: exp_id, imp_id, year, isic_id (1-36), isiccode, trade, FTA, plus pair = exp x imp,
expyear = exp x year and impyear = imp x year as integer codes (the do-file's FEs
exp_id#year, imp_id#year, exp_id#imp_id; cluster exp_id#imp_id). The package's
BACI_results.tex gives the published b and SE (Table 4, cols "PPML" and "SE").
Output: baci_example.parquet and build.json (rows, levels, per-industry counts).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import polars as pl

c = pl.col


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    parts, descr = [], {}
    with pd.read_stata(a.dta, iterator=True, chunksize=2_000_000, convert_categoricals=False) as r:
        for ch in r:
            for k, d in ch[["isic_id", "isicdescr"]].drop_duplicates().itertuples(index=False):
                descr[int(k)] = d
            parts.append(
                pl.from_pandas(ch[["exp_id", "imp_id", "year", "isic_id", "isiccode", "trade", "FTA"]]).select(
                    c("exp_id").cast(pl.Int32),
                    c("imp_id").cast(pl.Int32),
                    c("year").cast(pl.Int16),
                    c("isic_id").cast(pl.Int16),
                    c("isiccode").cast(pl.Int16),
                    c("trade").cast(pl.Float64),
                    c("FTA").cast(pl.Float64),
                )
            )
    d = pl.concat(parts)
    if d.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise SystemExit("nulls in baci_example")
    if d.select(pl.struct("exp_id", "imp_id", "year", "isic_id").is_duplicated().any()).item():
        raise SystemExit("duplicate exp x imp x year x isic rows")
    ne = int(d["exp_id"].max()) + 1
    d = d.with_columns(
        (c("exp_id") * ne + c("imp_id")).alias("pair"),
        (c("exp_id") * 10000 + c("year")).alias("expyear"),
        (c("imp_id") * 10000 + c("year")).alias("impyear"),
    ).sort("isic_id", "exp_id", "imp_id", "year")
    d.write_parquet(a.out / "baci_example.parquet")
    per = (
        d.group_by("isic_id", "isiccode")
        .agg(pl.len().alias("rows"), (c("trade") > 0).sum().alias("positive"), c("FTA").mean().alias("fta_mean"))
        .sort("isic_id")
    )
    log = {
        "rows": d.height,
        "years": sorted(d["year"].unique().to_list()),
        "exporters": d["exp_id"].n_unique(),
        "importers": d["imp_id"].n_unique(),
        "pairs": d["pair"].n_unique(),
        "industries": {int(r["isic_id"]): {**r, "descr": descr.get(int(r["isic_id"]))} for r in per.to_dicts()},
    }
    (a.out / "build.json").write_text(json.dumps(log, indent=1, default=str))
    print(json.dumps({k: v for k, v in log.items() if k != "industries"}, indent=1))
    print(per)


if __name__ == "__main__":
    main()
