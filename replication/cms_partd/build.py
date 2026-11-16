# SPDX-License-Identifier: GPL-3.0-or-later
"""D9: CMS Medicare Part D Prescribers by Provider and Drug, 2013-2024, to an analysis panel.

Reads the 12 yearly CSVs in `raw/cms_partd/partd_prov_drug/` (docs/sources/cms_partd.md) and
writes one row per prescriber (NPI) x drug (brand name, generic name) x year. Rules, columns
and the model: docs/cms_partd.md (Ted 2026-11-16, STATE.md Decisions).

    cd /srv/dfeols && .venv/bin/python -I repo/replication/cms_partd/build.py --out build/cms_partd

Output in --out:
  panel/dy<YYYY>.parquet        one file per year, sorted by npi
  drugs.parquet                 drug and generic codes with their names
  specialties.parquet           spec codes (Prscrbr_Type as printed, over all years)
  states.parquet                state codes
  R0/data/part-0000.parquet     reference tier: NPIs with r0_bucket(npi) == 0 (2%), all years
  build.json                    per-year rows read, dropped by each rule, kept; R0 rows

Codes are dense ints over all years, from the sorted distinct strings, so they don't depend
on which years are built. The CSVs are read as lossy UTF-8: one 2020 line has a broken byte
in Prscrbr_City, which isn't used.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

RAW = Path("raw/cms_partd/partd_prov_drug")
c = pl.col

# R0: 2% of NPIs. NPIs end in a Luhn check digit, so `npi % 50` isn't uniform; mix first.
R0_MOD = 50


def r0_bucket(npi: pl.Expr) -> pl.Expr:
    """Deterministic NPI hash bucket in [0, R0_MOD); stable across Polars versions."""
    return ((npi * 2654435761) % (1 << 32)) % R0_MOD


def files() -> dict[int, Path]:
    out = {2000 + int(f.stem.split("_")[-2][2:]): f for f in sorted(RAW.glob("*_NPIBN.csv"))}
    if sorted(out) != list(range(2013, 2025)):
        raise SystemExit(f"expected DY13..DY24 in {RAW}, found {sorted(out)}")
    return out


def scan(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, infer_schema=False, encoding="utf8-lossy").select(
        c("Prscrbr_NPI").cast(pl.Int64).alias("npi"),
        c("Prscrbr_State_Abrvtn").alias("state_name"),
        c("Prscrbr_Type").alias("spec_name"),
        c("Prscrbr_Type_Src").alias("spec_src_name"),
        c("Brnd_Name").alias("brand_name"),
        c("Gnrc_Name").alias("generic_name"),
        c("Tot_Clms").cast(pl.Float64).alias("claims"),
        c("Tot_30day_Fills").cast(pl.Float64).alias("fills_30day"),
        c("Tot_Day_Suply").cast(pl.Float64).alias("days_supply"),
        c("Tot_Drug_Cst").cast(pl.Float64).alias("cost"),
        c("Tot_Benes").cast(pl.Float64).alias("benes"),
    )


SPEC_SRC = {"Claim-Specialty": 0, "NPPES-Specialty": 1, "NPPES-Taxonomy": 2}


def lookups(fs: dict[int, Path], out: Path) -> dict[str, pl.DataFrame]:
    """Code tables over all years (one pass over the CSVs)."""
    allyears = pl.concat([scan(f) for f in fs.values()])
    drugs, specs, states = pl.collect_all(
        [
            allyears.select("brand_name", "generic_name").unique(),
            allyears.select("spec_name").drop_nulls().unique(),
            allyears.select("state_name").unique(),
        ],
        engine="streaming",
    )
    gen = drugs.select("generic_name").unique().sort("generic_name").with_row_index("generic")
    drugs = (
        drugs.sort("brand_name", "generic_name")
        .with_row_index("drug")
        .join(gen, on="generic_name")
        .select(c("drug").cast(pl.Int32), c("generic").cast(pl.Int32), "brand_name", "generic_name")
        .sort("drug")
    )
    specs = specs.sort("spec_name").with_row_index("spec").with_columns(c("spec").cast(pl.Int16))
    states = states.sort("state_name").with_row_index("state").with_columns(c("state").cast(pl.Int16))
    srcs = set(allyears.select("spec_src_name").unique().collect(engine="streaming")["spec_src_name"])
    if not srcs <= set(SPEC_SRC):
        raise SystemExit(f"unknown Prscrbr_Type_Src values: {srcs - set(SPEC_SRC)}")
    for name, df in (("drugs", drugs), ("specialties", specs), ("states", states)):
        df.write_parquet(out / f"{name}.parquet")
    return {"drugs": drugs, "specs": specs, "states": states}


def build_year(year: int, path: Path, lk: dict[str, pl.DataFrame], out: Path) -> dict:
    lf = scan(path)
    rules = (
        lf.select(
            pl.len().alias("rows_read"),
            (c("cost") <= 0).sum().alias("drop_cost_le0"),
            ((c("cost") > 0) & (c("days_supply") <= 0)).sum().alias("drop_days_le0"),
            ((c("cost") > 0) & (c("days_supply") > 0) & c("spec_name").is_null())
            .sum()
            .alias("drop_spec_null"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    kept = (
        lf.filter((c("cost") > 0) & (c("days_supply") > 0) & c("spec_name").is_not_null())
        .join(
            lk["drugs"].lazy().select("drug", "generic", "brand_name", "generic_name"),
            on=["brand_name", "generic_name"],
            how="left",
        )
        .join(lk["specs"].lazy(), on="spec_name", how="left")
        .join(lk["states"].lazy(), on="state_name", how="left")
        .select(
            "npi",
            pl.lit(year, pl.Int16).alias("year"),
            "drug",
            "generic",
            (c("brand_name") == c("generic_name")).cast(pl.Int8).alias("brand_eq_generic"),
            "spec",
            c("spec_src_name").replace_strict(SPEC_SRC, return_dtype=pl.Int8).alias("spec_src"),
            "state",
            (c("cost") / c("claims")).log().alias("log_cost_per_claim"),
            (c("days_supply") / c("claims")).log().alias("log_days_per_claim"),
            c("claims").log().alias("log_claims"),
            "claims",
            "fills_30day",
            "days_supply",
            "cost",
            "benes",
        )
        .sort("npi", "drug")
    )
    dst = out / "panel" / f"dy{year}.parquet"
    tmp = dst.with_suffix(".tmp")
    kept.sink_parquet(tmp, compression="zstd", statistics=True)
    chk = (
        pl.scan_parquet(tmp)
        .select(
            pl.len().alias("rows"),
            *[c(k).null_count().alias(f"null_{k}") for k in ("drug", "generic", "spec", "state", "spec_src")],
            pl.struct("npi", "drug").n_unique().alias("npi_drug"),
        )
        .collect()
        .row(0, named=True)
    )
    expect = rules["rows_read"] - rules["drop_cost_le0"] - rules["drop_days_le0"] - rules["drop_spec_null"]
    if (
        chk["rows"] != expect
        or chk["npi_drug"] != chk["rows"]
        or any(v for k, v in chk.items() if k.startswith("null_"))
    ):
        raise SystemExit(f"{year}: check failed: {chk}, expected {expect} rows")
    tmp.rename(dst)
    return {**rules, "rows": chk["rows"], "file": str(dst), "bytes": dst.stat().st_size}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("build/cms_partd"))
    ap.add_argument("--years", default="", help="comma list; default all (lookups always use all)")
    a = ap.parse_args(argv)
    (a.out / "panel").mkdir(parents=True, exist_ok=True)
    fs = files()
    t0 = time.time()
    lk = lookups(fs, a.out)
    print(
        f"lookups: {lk['drugs'].height} drugs, {lk['drugs']['generic'].n_unique()} generics, "
        f"{lk['specs'].height} specialties, {lk['states'].height} states ({time.time() - t0:.0f} s)",
        flush=True,
    )
    bj = a.out / "build.json"
    info = json.loads(bj.read_text()) if bj.exists() else {"years": {}}
    years = [int(y) for y in a.years.split(",")] if a.years else sorted(fs)
    for y in years:
        t1 = time.time()
        info["years"][str(y)] = r = build_year(y, fs[y], lk, a.out)
        print(f"{y}: {r['rows_read']:,} read, {r['rows']:,} kept ({time.time() - t1:.0f} s)", flush=True)
        bj.write_text(json.dumps(info, indent=1))
    if len(info["years"]) == 12:
        r0 = a.out / "R0" / "data"
        r0.mkdir(parents=True, exist_ok=True)
        (
            pl.scan_parquet(a.out / "panel" / "dy*.parquet")
            .filter(r0_bucket(c("npi")) == 0)
            .sort("npi", "year", "drug")
            .sink_parquet(r0 / "part-0000.parquet", compression="zstd", statistics=True)
        )
        info["R0"] = {
            "rule": f"((npi * 2654435761) mod 2^32) mod {R0_MOD} == 0",
            "rows": pl.scan_parquet(r0 / "part-0000.parquet").select(pl.len()).collect().item(),
        }
        print(f"R0: {info['R0']['rows']:,} rows", flush=True)
    info["totals"] = {
        k: sum(v[k] for v in info["years"].values())
        for k in ("rows_read", "drop_cost_le0", "drop_days_le0", "drop_spec_null", "rows")
    }
    info["seconds"] = round(time.time() - t0, 1)
    bj.write_text(json.dumps(info, indent=1))
    print(json.dumps(info["totals"]), flush=True)


if __name__ == "__main__":
    main()
