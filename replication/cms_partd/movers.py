# SPDX-License-Identifier: GPL-3.0-or-later
"""D9 extension: prescriber x year panel with HRRs and mover flags (Beheshti & Neller 2026).

Reads the Part D by-Provider files (`raw/cms_partd/partd_prov/`, one row per NPI x year, with
ZIP) and the Dartmouth ZIP-HSA-HRR crosswalks (`raw/dartmouth/`), and writes one row per NPI
x year with the prescriber's HRR and totals, plus mover flags. Sources:
docs/sources/cms_partd_provider.md; rules: docs/cms_partd.md (Ted 2026-11-21, STATE.md
Decisions).

    cd /srv/dfeols && .venv/bin/python -I repo/replication/cms_partd/movers.py --out build/cms_partd/movers

Rules:
  - HRR from Prscrbr_Zip5 and that year's crosswalk; years after 2019 use the 2019 crosswalk.
  - Prescriber-years whose ZIP is not in the crosswalk are dropped (counted in build.json).
  - All prescriber types are kept (B&N keep non-physicians such as nurse practitioners).
  - Censored beneficiary counts: both variants are flagged, not dropped here:
      benes_censored      this prescriber-year's Tot_Benes is suppressed (<11)
      npi_any_censored    some year of this NPI (within the window) is suppressed
  - Movers, per window (2013-2019 = B&N; 2013-2024 = extension), on the rows kept:
      contig_<w>    the NPI's years in the window are contiguous
      moves_<w>     number of year-to-year HRR changes in the window
      move_year_<w> year of the first HRR change (null if none)
    B&N movers: contig and moves == 1. Multi-movers keep move_year (first move) for the
    extension.

Output in --out: panel.parquet (sorted by npi, year), hrr.parquet (HRR codes and names),
build.json (rows read, dropped, kept, mover counts).
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import polars as pl

RAW = Path("raw/cms_partd/partd_prov")
DART = Path("raw/dartmouth")
LAST_XW = 2019
WINDOWS = {"bn": (2013, 2019), "all": (2013, 2024)}
c = pl.col

NUM = {
    "Tot_Clms": "claims",
    "Tot_30day_Fills": "fills_30day",
    "Tot_Drug_Cst": "cost",
    "Tot_Day_Suply": "days",
    "Tot_Benes": "benes",
    "Brnd_Tot_Clms": "brand_claims",
    "Brnd_Tot_Drug_Cst": "brand_cost",
    "Gnrc_Tot_Clms": "generic_claims",
    "Gnrc_Tot_Drug_Cst": "generic_cost",
    "Opioid_Tot_Clms": "opioid_claims",
    "Opioid_Tot_Drug_Cst": "opioid_cost",
    "Bene_Avg_Age": "bene_avg_age",
    "Bene_Avg_Risk_Scre": "bene_avg_risk",
    "Bene_Dual_Cnt": "bene_dual",
}


def crosswalk(year: int) -> pl.DataFrame:
    """ZIP -> HRR for a crosswalk year (1 row per ZIP)."""
    yy = f"{year % 100:02d}"
    if year >= 2018:
        z = zipfile.ZipFile(DART / f"ZipHsaHrr{yy}.csv.zip")
        d = pd.read_csv(io.BytesIO(z.read(f"ZipHsaHrr{yy}.csv")))
    else:
        d = pd.read_excel(DART / f"ZipHsaHrr{yy}.xls")
    zcol = [x for x in d.columns if x.lower().startswith("zipcode")][0]
    out = pl.from_pandas(d[[zcol, "hrrnum", "hrrcity", "hrrstate"]]).select(
        c(zcol).cast(pl.Int64).alias("zip"),
        c("hrrnum").cast(pl.Int32).alias("hrr"),
        c("hrrcity").alias("hrr_city"),
        c("hrrstate").alias("hrr_state"),
    )
    if out["zip"].n_unique() != out.height:
        raise SystemExit(f"crosswalk {year}: ZIPs not unique")
    return out


def files() -> dict[int, Path]:
    out = {2000 + int(f.stem.split("_")[-2][2:]): f for f in sorted(RAW.glob("*_npi.csv"))}
    if sorted(out) != list(range(2013, 2025)):
        raise SystemExit(f"expected DY13..DY24 in {RAW}, found {sorted(out)}")
    return out


def read_year(year: int, path: Path) -> pl.DataFrame:
    return (
        pl.scan_csv(path, infer_schema=False, encoding="utf8-lossy")
        .select(
            c("Prscrbr_NPI").cast(pl.Int64).alias("npi"),
            c("Prscrbr_State_Abrvtn").alias("state"),
            c("Prscrbr_Zip5").cast(pl.Int64, strict=False).alias("zip"),
            c("Prscrbr_RUCA").alias("ruca"),
            c("Prscrbr_Type").alias("spec"),
            c("Prscrbr_Ent_Cd").alias("entity"),
            *[c(k).cast(pl.Float64).alias(v) for k, v in NUM.items()],
        )
        .with_columns(pl.lit(year, pl.Int16).alias("year"))
        .collect(engine="streaming")
    )


def mover_flags(p: pl.DataFrame, w: str, lo: int, hi: int) -> pl.DataFrame:
    return (
        p.filter(c("year").is_between(lo, hi))
        .sort("npi", "year")
        .with_columns(
            (c("year") - c("year").shift(1).over("npi")).alias("_gap"),
            (c("hrr") != c("hrr").shift(1).over("npi")).fill_null(False).alias("_chg"),
        )
        .group_by("npi")
        .agg(
            (c("_gap").fill_null(1) == 1).all().alias(f"contig_{w}"),
            c("_chg").sum().cast(pl.Int16).alias(f"moves_{w}"),
            c("year").filter(c("_chg")).min().alias(f"move_year_{w}"),
            c("benes_censored").any().alias(f"npi_any_censored_{w}"),
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    xws = {y: crosswalk(y) for y in range(2013, LAST_XW + 1)}
    log: dict = {"years": {}, "crosswalk_last": LAST_XW}
    kept = []
    for year, path in files().items():
        d = read_year(year, path).join(xws[min(year, LAST_XW)].select("zip", "hrr"), on="zip", how="left")
        if d["npi"].n_unique() != d.height:
            raise SystemExit(f"{year}: duplicate NPIs")
        no_hrr = d["hrr"].is_null()
        log["years"][year] = {
            "rows": d.height,
            "dropped_zip_missing": int(d["zip"].is_null().sum()),
            "dropped_zip_not_in_crosswalk": int((d["zip"].is_not_null() & no_hrr).sum()),
            "kept": int((~no_hrr).sum()),
            "benes_censored": int(d.filter(~no_hrr)["benes"].is_null().sum()),
        }
        kept.append(d.filter(~no_hrr).with_columns(c("benes").is_null().alias("benes_censored")))
    p = pl.concat(kept)
    for w, (lo, hi) in WINDOWS.items():
        p = p.join(mover_flags(p, w, lo, hi), on="npi", how="left")
        f = p.filter(c("year").is_between(lo, hi)).unique("npi")
        bn = f.filter(c(f"contig_{w}") & (c(f"moves_{w}") == 1))
        log[f"window_{w}"] = {
            "years": [lo, hi],
            "npis": f.height,
            "contiguous": int(f[f"contig_{w}"].sum()),
            "movers_contig_one_move": bn.height,
            "movers_contig_one_move_never_censored": int((~bn[f"npi_any_censored_{w}"]).sum()),
            "multi_movers_contig": int((f[f"contig_{w}"] & (f[f"moves_{w}"] > 1)).sum()),
            "npis_never_censored": int((~f[f"npi_any_censored_{w}"]).sum()),
        }
    p = p.sort("npi", "year")
    p.write_parquet(args.out / "panel.parquet")
    pl.concat([xws[y].select("hrr", "hrr_city", "hrr_state") for y in xws]).unique("hrr").sort("hrr").write_parquet(
        args.out / "hrr.parquet"
    )
    log["rows_out"] = p.height
    log["seconds"] = round(time.time() - t0, 1)
    (args.out / "build.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != "years"}, indent=1))


if __name__ == "__main__":
    main()
