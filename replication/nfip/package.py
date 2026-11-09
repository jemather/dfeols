# SPDX-License-Identifier: GPL-3.0-or-later
"""D3: package the NFIP redacted policies (OpenFEMA V3) for OSCER as ``outbox/nfip_policies_v1/``.

    python repo/replication/nfip/package.py [--src raw/nfip/NfipPoliciesV3.parquet] [--out outbox]
        [--parts 6] [--states RI]      # --states: a test on those states only

Run from the project root. One row per policy transaction (no policy or building ID: FEMA redacts
them, so there is no policy panel). Specs agreed 2026-11-09 (STATE.md Decisions):

A. Risk Rating 2.0 event study: pre-FIRM against other policies in the same tract, by quarter
   relative to 2022Q2 (renewals moved to Risk Rating 2.0 from 2022-04-01; ref 2021Q3), 2018-2025,
   with two pre-FIRM definitions: subsidy eligible (``preFIRMConstructionIndicator``, which FEMA's
   dictionary says is misnamed) and built before the community's first FIRM (not
   ``postFIRMConstructionIndicator``).
B. CRS discount check: premium by CRS class (blank = 10), before Risk Rating 2.0 in the SFHA (the
   only place the legacy 5%-per-class discount applied) and under Risk Rating 2.0 in all zones,
   with county x month FEs (Ted, 2026-11-09): the file's CRS class never changes within a
   community (20,757 communities, 2009-2026; apparently the current class on every record), so a
   community FE would absorb it. Communities are compared within county, month and flood zone.

Each spec runs on all buildings and on single-family homes (occupancyType 1 legacy, 11 RR2.0).
Sub-samples are coded as nulls, which fits drop: ``event_quarter`` is null outside 2018-2025,
``crs_class_pre`` / ``crs_class_rr2`` outside their sample, ``log_premium_sf`` for other
building types. Sample: premium > 0, effective 2009-01-01..2026-08-31, known tract.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outbox_tools import dump_json, fit_reference, level_counts, schema_table, write_parts

NAME = "nfip_policies_v1"
FIRST, LAST = date(2009, 1, 1), date(2026, 8, 31)
RR2_RENEWALS_Q = 2022 * 4 + 1  # 2022Q2 as year*4 + (quarter-1)
EVENT_LO, EVENT_HI = 2018 * 4 + 0 - RR2_RENEWALS_Q, 2025 * 4 + 3 - RR2_RENEWALS_Q  # 2018Q1..2025Q4
REF_Q = -3  # 2021Q3, the last quarter before any policy was rated under Risk Rating 2.0
REF_STATES = ["RI"]
CONTROLS = "log_building_cov + log_contents_cov + elevated + primary_residence"
FE_A = "tract^{d} + tract^quarter + flood_zone^quarter"
FE_B = "county^month + flood_zone^month"  # community: CRS class is constant within one
CLUSTER = "community"


def specs():
    """(label, dfeols formula, pyfixest formula) for every spec and sample."""
    out = []
    for y, sample in [("log_premium", "all"), ("log_premium_sf", "single-family")]:
        for d in ["subsidy_eligible", "built_pre_firm"]:
            fe = FE_A.format(d=d)
            out.append((
                f"A {d}, {sample}",
                f"{y} ~ {d}:i(event_quarter, ref={REF_Q}) + {CONTROLS} | {fe}",
                f"{y} ~ i(event_quarter, {d}, ref={REF_Q}) + {CONTROLS} | {fe}",
            ))  # fmt: skip
        for c, when in [("crs_class_pre", "before RR2.0, SFHA"), ("crs_class_rr2", "RR2.0, all zones")]:
            f = f"{y} ~ i({c}, ref=10) + {CONTROLS} | {FE_B}"
            out.append((f"B {when}, {sample}", f, f))
    return out


def frame(lf: pl.LazyFrame) -> pl.LazyFrame:
    d = pl.col("policyEffectiveDate")
    q = d.dt.year().cast(pl.Int32) * 4 + (d.dt.month().cast(pl.Int32) - 1) // 3
    zone = pl.col("ratedFloodZone")
    sfha = (zone.str.starts_with("A") | zone.str.starts_with("V")) & ~zone.is_in(["AR", "A99"])
    rr2 = pl.col("rateMethod") == "RE"
    crs = pl.col("crsClassCode").fill_null(10)
    sf = pl.col("occupancyType").is_in([1, 11])
    lp = pl.col("totalInsurancePremiumOfThePolicy").log()
    ev = q - RR2_RENEWALS_Q
    return (
        lf.filter(
            (pl.col("totalInsurancePremiumOfThePolicy") > 0)
            & d.is_between(FIRST, LAST)
            & pl.col("censusGeoid").is_not_null()
        )
        .select(
            id=pl.col("id"),
            policy_effective_date=d,
            month=((d.dt.year().cast(pl.Int32) - 2009) * 12 + d.dt.month().cast(pl.Int32) - 1).cast(pl.Int16),
            quarter=(q - 2009 * 4).cast(pl.Int16),
            event_quarter=pl.when(ev.is_between(EVENT_LO, EVENT_HI)).then(ev).cast(pl.Int16),
            tract=pl.col("censusGeoid").str.slice(0, 11),
            block_group=pl.col("censusGeoid"),
            community=pl.col("nfipCommunityNumberCurrent"),
            county=pl.col("censusGeoid").str.slice(0, 5),
            state=pl.col("propertyState"),
            flood_zone=zone,
            sfha=sfha.cast(pl.Int8),
            rate_method=pl.col("rateMethod"),
            rr2=rr2.cast(pl.Int8),
            occupancy_type=pl.col("occupancyType"),
            single_family=sf.cast(pl.Int8),
            subsidy_eligible=pl.col("preFIRMConstructionIndicator").cast(pl.Int8),
            built_pre_firm=(~pl.col("postFIRMConstructionIndicator")).cast(pl.Int8),
            crs_class=crs.cast(pl.Int16),
            crs_class_pre=pl.when(~rr2 & (d < date(2021, 10, 1)) & sfha).then(crs).cast(pl.Int16),
            crs_class_rr2=pl.when(rr2).then(crs).cast(pl.Int16),
            premium=pl.col("totalInsurancePremiumOfThePolicy"),
            policy_cost=pl.col("policyCost"),
            full_risk_premium=pl.col("fullRiskPremium"),
            log_premium=lp,
            log_premium_sf=pl.when(sf).then(lp),
            building_cov=pl.col("totalBuildingInsuranceCoverage"),
            contents_cov=pl.col("totalContentsInsuranceCoverage"),
            log_building_cov=pl.col("totalBuildingInsuranceCoverage").cast(pl.Float64).log1p(),
            log_contents_cov=pl.col("totalContentsInsuranceCoverage").cast(pl.Float64).log1p(),
            elevated=pl.col("elevatedBuildingIndicator").cast(pl.Int8),
            primary_residence=pl.col("primaryResidenceIndicator").cast(pl.Int8),
            policy_count=pl.col("policyCount"),
        )
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="raw/nfip/NfipPoliciesV3.parquet")
    ap.add_argument("--out", type=Path, default=Path("outbox"))
    ap.add_argument("--parts", type=int, default=6)
    ap.add_argument("--states", default=None, help="comma list: a test on those states only")
    ap.add_argument("--name", default=NAME)
    a = ap.parse_args(argv)
    commit = (
        subprocess.run(["git", "-C", "repo", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        .stdout.strip() or "unknown"
    )  # fmt: skip
    raw = pl.scan_parquet(a.src)
    if a.states:
        raw = raw.filter(pl.col("propertyState").is_in(a.states.split(",")))
    pre = (
        raw.select(
            rows_in=pl.len(),
            premium_le0=(pl.col("totalInsurancePremiumOfThePolicy") <= 0).sum(),
            outside_dates=(~pl.col("policyEffectiveDate").is_between(FIRST, LAST)).sum(),
            tract_missing=pl.col("censusGeoid").is_null().sum(),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    print(pre, flush=True)

    d = a.out / a.name
    parts = write_parts(frame(raw), "tract", d, a.parts)
    data = pl.scan_parquet(str(d / "data" / "*.parquet"))
    rows = sum(p["rows"] for p in parts)
    levels = level_counts(
        data,
        {
            "tract": ["tract"],
            "block_group": ["block_group"],
            "community": ["community"],
            "county": ["county"],
            "state": ["state"],
            "flood_zone": ["flood_zone"],
            "month": ["month"],
            "quarter": ["quarter"],
            "tract^quarter": ["tract", "quarter"],
            "tract^subsidy_eligible": ["tract", "subsidy_eligible"],
            "tract^built_pre_firm": ["tract", "built_pre_firm"],
            "flood_zone^quarter": ["flood_zone", "quarter"],
            "flood_zone^month": ["flood_zone", "month"],
            "county^month": ["county", "month"],
        },
    )
    stats = (
        data.select(
            single_family=pl.col("single_family").mean(),
            subsidy_eligible=pl.col("subsidy_eligible").mean(),
            built_pre_firm=pl.col("built_pre_firm").mean(),
            rr2=pl.col("rr2").mean(),
            sfha=pl.col("sfha").mean(),
            in_event_window=pl.col("event_quarter").is_not_null().mean(),
            crs_pre_rows=pl.col("crs_class_pre").is_not_null().sum(),
            crs_rr2_rows=pl.col("crs_class_rr2").is_not_null().sum(),
            crs_blank_as_10=pl.col("crs_class").eq(10).mean(),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    ref = data.filter(pl.col("state").is_in(REF_STATES))
    if ref.select(pl.len()).collect().item() == 0:
        ref = data
    cols = ["log_premium", "log_premium_sf", "event_quarter", "subsidy_eligible", "built_pre_firm",
            "crs_class_pre", "crs_class_rr2", "log_building_cov", "log_contents_cov", "elevated",
            "primary_residence", "tract", "quarter", "flood_zone", "community", "county", "month", "state"]  # fmt: skip
    pdf_all = ref.select(cols).collect().to_pandas()
    fits = []
    for label, f_dfeols, f_pyfixest in specs():
        y = f_pyfixest.split(" ~ ")[0]
        need = [y, "event_quarter" if label.startswith("A") else f_pyfixest.split("i(")[1].split(",")[0]]
        pdf = pdf_all.dropna(subset=need)
        r = fit_reference(pdf, f_pyfixest, CLUSTER)
        r.update(label=label, dfeols_formula=f_dfeols, subsample=f"state in {REF_STATES}, rows with {need} non-null")
        fits.append(r)
        print(label, r["nobs"], flush=True)
    dump_json(
        d / "reference.json",
        {
            "dataset": a.name,
            "build": f"jemather/dfeols {commit}",
            "tool": "pyfixest (in memory, valiant)",
            "rows": rows,
            "sample_counts": pre,
            "fe_levels": levels,
            "stats": stats,
            "fits": fits,
            "parts": parts,
        },
    )
    notes = {
        "id": "OpenFEMA record id (a policy transaction, not a policy: no policy or building ID)",
        "month": "months since 2009-01 (policy effective date)",
        "quarter": "quarters since 2009Q1",
        "event_quarter": f"quarters relative to 2022Q2; null outside 2018Q1-2025Q4; reference {REF_Q} (2021Q3)",
        "tract": "first 11 digits of censusGeoid (state, county, tract); partition key",
        "block_group": "censusGeoid as published (12 digits: a block group)",
        "community": "nfipCommunityNumberCurrent; cluster. CRS class is constant within a community",
        "county": "first 5 digits of censusGeoid (state and county FIPS)",
        "sfha": "ratedFloodZone starts with A or V, except AR and A99 (FEMA's CRS rule)",
        "rr2": "rateMethod == 'RE' (Risk Rating 2.0; the dictionary's 'RatingEngine')",
        "single_family": "occupancyType 1 (legacy) or 11 (Risk Rating 2.0)",
        "subsidy_eligible": "preFIRMConstructionIndicator (misnamed per FEMA: pre-FIRM subsidy eligible)",
        "built_pre_firm": "not postFIRMConstructionIndicator (construction before the first FIRM or 1975)",
        "crs_class": "crsClassCode, blank = 10 (no discount)",
        "crs_class_pre": "crs_class for legacy-rated SFHA policies effective before 2021-10-01, else null",
        "crs_class_rr2": "crs_class for Risk Rating 2.0 policies, else null",
        "premium": "totalInsurancePremiumOfThePolicy, dollars (> 0 kept)",
        "log_premium_sf": "log_premium for single-family homes, else null",
        "log_building_cov": "log(1 + totalBuildingInsuranceCoverage)",
        "log_contents_cov": "log(1 + totalContentsInsuranceCoverage)",
        "policy_count": "insured units (> 1 for condo association policies)",
    }
    fit_rows = "\n".join(
        f"| {r['label']} | `{r['dfeols_formula'].replace('|', chr(92) + '|')}` | {r['nobs']:,} | "
        + ", ".join(f"{t} {v[0]:.5f} ({v[1]:.5f})" for t, v in list(r["coef"].items())[:4])
        + (" ..." if len(r["coef"]) > 4 else "")
        + " |"
        for r in fits
    )
    lv = "\n".join(f"| {k} | {v:,} |" for k, v in levels.items())
    pt = "\n".join(
        f"| {p['file']} | {p['rows']:,} | {p['tract_min']} to {p['tract_max']} | {p['bytes'] / 1e9:.2f} GB |"
        for p in parts
    )
    (d / "README.md").write_text(f"""# {a.name}

FEMA NFIP redacted policies (OpenFEMA V3), one row per policy transaction, effective
{FIRST} to {LAST}: {rows:,} rows. For a Risk Rating 2.0 event study and a CRS discount check
(D3). No policy or building ID (redacted), so there is no policy panel.

- **Source:** OpenFEMA "FIMA NFIP Redacted Policies – v3",
  https://www.fema.gov/openfema-data-page/nfip-redacted-policies-v3, file `NfipPoliciesV3.parquet`
  (as of 2026-09, sha256 b830cd23…; `docs/sources/nfip.md` on valiant). Public; FEMA asks for
  citation and no implied endorsement.
- **Build:** `repo/replication/nfip/package.py`, jemather/dfeols `{commit}`
- **Built:** {date.today()} on valiant
- **Rows:** {rows:,} in {len(parts)} parts by `tract`, of {pre["rows_in"]:,} in the file
  (premium ≤ 0 {pre["premium_le0"]:,}, outside the dates {pre["outside_dates"]:,}, no census
  geography {pre["tract_missing"]:,}; they overlap)
- **Shares:** single-family {stats["single_family"]:.3f}; subsidy eligible
  {stats["subsidy_eligible"]:.3f}; built before the FIRM {stats["built_pre_firm"]:.3f}; Risk
  Rating 2.0 {stats["rr2"]:.3f}; SFHA {stats["sfha"]:.3f}; in the event window
  {stats["in_event_window"]:.3f}; CRS class 10 (incl. blank) {stats["crs_blank_as_10"]:.3f}
- **CRS samples:** before RR2.0 in the SFHA {stats["crs_pre_rows"]:,} rows; RR2.0
  {stats["crs_rr2_rows"]:,} rows. Expected legacy SFHA discount: 5% per class below 10.

## Parts

| File | Rows | tract | Size |
|---|---|---|---|
{pt}

## Schema

{schema_table(data, notes)}

## FE / cluster levels

| Variable | Distinct |
|---|---:|
{lv}

## Suggested partition key and specs

Partition `tract` (spec A's FEs are all nested in it; spec B's county^month crosses it); cluster
`{CLUSTER}`. Sub-samples are nulls, which the fit drops (see the schema notes). Formulas in
dfeols syntax (`x:i(c, ref=r)`); the pyfixest reference uses `i(c, x, ref=r)`, the same model.

| Spec | Formula | N (pyfixest, {"/".join(REF_STATES)}) | pyfixest b (se), first terms |
|---|---|---:|---|
{fit_rows}

Reference subsample: policies in {", ".join(REF_STATES)}. Full-precision values (every event-time
and CRS term) in `reference.json`. Then the full file; no reference outside dfeols.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
