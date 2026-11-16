# SPDX-License-Identifier: GPL-3.0-or-later
"""Package the D9 panel (build.py output) as outbox/cms_partd_v1 (CLAUDE.md layout).

    cd /srv/dfeols && .venv/bin/python -I repo/replication/cms_partd/package.py \
        --build build/cms_partd --out outbox/cms_partd_v1 --commit <sha>

Writes data/part-NNNN.parquet (contiguous NPI ranges, sorted by npi), ref/data/
(the R0 tier) and ref/reference.json (from reference.py), lookups/ (code tables), README.md.
The manifest is written afterwards by the shell command in CLAUDE.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import outbox_tools as ot

FES = {
    "npi": ["npi"],
    "npi^year": ["npi", "year"],
    "drug": ["drug"],
    "generic": ["generic"],
    "drug^year": ["drug", "year"],
    "spec": ["spec"],
    "state": ["state"],
    "spec^state^year": ["spec", "state", "year"],
    "drug^state^year": ["drug", "state", "year"],
}
NOTES = {
    "npi": "Prscrbr_NPI; FE, cluster and partition key",
    "year": "data year 2013-2024 (file name DY??)",
    "drug": "code of (Brnd_Name, Gnrc_Name) over all years; lookups/drugs.parquet",
    "generic": "code of Gnrc_Name; lookups/drugs.parquet",
    "brand_eq_generic": "Brnd_Name == Gnrc_Name (the generic version)",
    "spec": "code of Prscrbr_Type as printed, over all years; lookups/specialties.parquet",
    "spec_src": "Prscrbr_Type_Src: 0 Claim-Specialty, 1 NPPES-Specialty, 2 NPPES-Taxonomy",
    "state": "code of Prscrbr_State_Abrvtn (incl. DC, territories, AA/AP, XX); lookups/states.parquet",
    "log_cost_per_claim": "ln(Tot_Drug_Cst / Tot_Clms): the outcome",
    "log_days_per_claim": "ln(Tot_Day_Suply / Tot_Clms): the regressor",
    "log_claims": "ln(Tot_Clms): alternative outcome",
    "claims": "Tot_Clms (>= 11: CMS suppresses smaller cells)",
    "fills_30day": "Tot_30day_Fills",
    "days_supply": "Tot_Day_Suply",
    "cost": "Tot_Drug_Cst, $",
    "benes": "Tot_Benes; null when suppressed (< 11)",
}
LADDER = {
    "r1": "npi + drug^year",
    "r2": "npi + drug^year + spec^state^year",
    "r3": "npi^year + drug^year",
    "r4": "npi^year + drug^state^year",
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=Path("build/cms_partd"))
    ap.add_argument("--out", type=Path, default=Path("outbox/cms_partd_v1"))
    ap.add_argument("--commit", required=True, help="dfeols repo commit of the build code")
    ap.add_argument("--parts", type=int, default=8)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; packages are never changed after creation (build _v2)")
    ref_json = a.build / "R0" / "reference.json"
    if not ref_json.exists():
        raise SystemExit(f"{ref_json} missing: run reference.py first")
    info = json.loads((a.build / "build.json").read_text())

    panel = sorted((a.build / "panel").glob("dy*.parquet"))
    lf = pl.scan_parquet(panel)
    tmp = a.out.with_name(a.out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    parts = ot.write_parts(lf, "npi", tmp, a.parts)
    files = [tmp / p["file"] for p in parts]
    levels = ot.level_counts_parts(files, FES, within={"npi", "npi^year"})

    (tmp / "ref" / "data").mkdir(parents=True)
    shutil.copy2(a.build / "R0" / "data" / "part-0000.parquet", tmp / "ref" / "data" / "part-0000.parquet")
    ref = json.loads(ref_json.read_text())
    ot.dump_json(
        tmp / "ref" / "reference.json",
        {"pyfixest": ref["pyfixest"], "dfeols_local": ref["dfeols"], "tier": info["R0"]},
    )
    ref_levels = ot.level_counts(pl.scan_parquet(tmp / "ref" / "data" / "*.parquet"), FES)
    (tmp / "lookups").mkdir()
    for n in ("drugs", "specialties", "states"):
        shutil.copy2(a.build / f"{n}.parquet", tmp / "lookups" / f"{n}.parquet")

    total = sum(p["rows"] for p in parts)
    if total != info["totals"]["rows"]:
        raise SystemExit(f"parts hold {total} rows, build {info['totals']['rows']}")
    t = info["totals"]
    yr = "\n".join(
        f"| {y} | {v['rows_read']:,} | {v['drop_cost_le0']:,} | {v['drop_days_le0']:,} | "
        f"{v['drop_spec_null']:,} | {v['rows']:,} |"
        for y, v in sorted(info["years"].items())
    )
    pt = "\n".join(
        f"| {p['file']} | {p['rows']:,} | {p['npi_min']}..{p['npi_max']} | {p['bytes'] / 1e9:.2f} GB |"
        for p in parts
    )
    lv = "\n".join(f"| {k} | {levels[k]:,} | {ref_levels[k]:,} |" for k in FES)
    rf = "\n".join(
        f"| {k} | `log_cost_per_claim ~ log_days_per_claim \\| {fe}` | {ref['pyfixest'][k]['N']:,} | "
        f"{ref['pyfixest'][k]['coefs']['CRV1']['log_days_per_claim'][0]:.10f} | "
        f"{ref['pyfixest'][k]['coefs']['CRV1']['log_days_per_claim'][1]:.10f} |"
        for k, fe in LADDER.items()
        if k in ref["pyfixest"]
    )
    readme = f"""# cms_partd_v1

CMS Medicare Part D Prescribers by Provider and Drug, 2013-2024: one row per prescriber (NPI)
x drug (brand name, generic name) x year. {total:,} rows in {len(parts)} parts. A dfeols test
dataset (a large three-way FE panel), not a research project.

- **Source:** data.cms.gov, "Medicare Part D Prescribers - by Provider and Drug", the 12 yearly
  CSVs `MUP_DPR_RY24_P04_V10_DY13..22_NPIBN.csv` (release RY24), `..._RY25_..._DY23_...` and
  `..._RY26_..._DY24_...` (43.8 GB; sha256 in `raw/cms_partd/SHA256SUMS` on valiant). US
  government works (https://www.usa.gov/government-works). Notes: `docs/cms_partd.md` and
  `docs/sources/cms_partd.md` on valiant
- **Build:** `repo/replication/cms_partd/build.py` and `package.py` (jemather/dfeols `{a.commit}`)
- **Selection by the source:** CMS suppresses prescriber x drug cells with fewer than 11
  claims, so `claims` >= 11 in every row and small cells are missing, not zero
- **Sample** (Decisions 2026-11-16): all 12 years; dropped cost <= 0 ({t["drop_cost_le0"]:,}),
  days' supply 0 ({t["drop_days_le0"]:,}), specialty missing ({t["drop_spec_null"]:,}) of
  {t["rows_read"]:,} rows read. (npi, drug) is unique within a year; specialty and state are
  constant within npi x year (they change across years for ~300k and ~200k NPIs)

| Year | Read | Cost <= 0 | Days 0 | Specialty null | Kept |
|---|---:|---:|---:|---:|---:|
{yr}

## Model (Decisions 2026-11-16)

`log_cost_per_claim ~ log_days_per_claim | FEs`, CRV1 clustered by `npi` (also hetero),
partition `npi`. FE ladder:

| Spec | FEs | What it tests |
|---|---|---|
| r1 | `npi + drug^year` | two-way, one big FE |
| r2 | `npi + drug^year + spec^state^year` | three-way; spec^state is a function of npi within a year |
| r3 | `npi^year + drug^year` | 11.6M-level first FE |
| r4 | `npi^year + drug^state^year` | two large FEs; drug^state^year crosses every shard |

## Parts

Contiguous NPI ranges, sorted by npi within a part.

| File | Rows | NPI range | Size |
|---|---:|---|---:|
{pt}

## Reference tier (`ref/`)

`ref/data/part-0000.parquet`: NPIs with `{info["R0"]["rule"]}` (2%), all years,
{info["R0"]["rows"]:,} rows, same columns. `ref/reference.json`: pyfixest (demeaning tolerance
1e-13, singletons dropped) for r1-r4, CRV1 by npi and hetero, plus the local dfeols fits
(partitioned by npi and by year, CG iter_tol 1e-10, level space off). Coefficient on
`log_days_per_claim`, CRV1:

| Spec | Formula | N | b | SE |
|---|---|---:|---:|---:|
{rf}

## Schema

{ot.schema_table(pl.scan_parquet(files[0]), NOTES)}

`lookups/`: `drugs.parquet` (drug, generic, brand_name, generic_name), `specialties.parquet`
(spec, spec_name), `states.parquet` (state, state_name).

## FE / cluster levels

| FE | Full | ref |
|---|---:|---:|
{lv}
"""
    (tmp / "README.md").write_text(readme)
    tmp.rename(a.out)
    print(readme)


if __name__ == "__main__":
    main()
