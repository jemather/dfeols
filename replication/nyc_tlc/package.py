# SPDX-License-Identifier: GPL-3.0-or-later
"""D5: package the NYC TLC rideshare (fhvhv) trips for OSCER as ``outbox/nyc_fhvhv_v1/``.

    python repo/replication/nyc_tlc/package.py [--stage build/nyc_tlc/stage] [--out outbox]
        [--parts 50] [--months 2024-03,2024-03]   # --months: a test on those stage files only

Run from the project root. Reads the stage-1 files (``stage.py``), applies the sample rules and
model columns agreed 2026-11-09 (STATE.md Decisions), writes data/ in contiguous ranges of
pickup date, then README.md and reference.json (pyfixest on one day). Manifest afterwards
(CLAUDE.md).

Sample: a trip is kept if both zones are known (not null, not 264 "Unknown" or 265 "Outside of
NYC") and base_passenger_fare, trip_miles and trip_time are all > 0. log_wait is null where
the request time is missing or not before pickup, and driver_share where driver_pay <= 0; such
rows stay in the data (fits drop them).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outbox_tools import dump_json, fit_reference, level_counts, schema_table, write_parts

NAME = "nyc_fhvhv_v1"
T0 = datetime(2019, 2, 1)  # date_hour = hours since the first file's month
REF_DAY = date(2024, 3, 6)  # a Wednesday; ~720k trips
MARGIN = 0  # months between a pickup and its file's month (checked; 0 for all of 2019-02..2026-09)
X = 'log_miles + log_time + shared_request + i(company, ref="HV0003")'
FE = "pu_location_id^date_hour + do_location_id^date_hour + pu_location_id^do_location_id"
SPECS = [
    ("price", f"log_fare ~ {X} | {FE}"),
    ("wait", f"log_wait ~ {X} | {FE}"),
    ("driver share", f"driver_share ~ {X} | {FE}"),
]
CLUSTER = "od_pair"
UNKNOWN_ZONES = [264, 265]
KEEP = [
    "hvfhs_license_num", "dispatching_base_num", "originating_base_num", "request_datetime",
    "on_scene_datetime", "pickup_datetime", "dropoff_datetime", "pu_location_id", "do_location_id",
    "trip_miles", "trip_time", "base_passenger_fare", "tolls", "bcf", "sales_tax",
    "congestion_surcharge", "airport_fee", "cbd_congestion_fee", "tips", "driver_pay",
    "shared_request_flag", "shared_match_flag", "access_a_ride_flag", "wav_request_flag",
    "wav_match_flag", "source_month",
]  # fmt: skip


def month_offset() -> pl.Expr:
    p, s = pl.col("pickup_datetime"), pl.col("source_month")
    return (p.dt.year() * 12 + p.dt.month()) - (s.dt.year() * 12 + s.dt.month())


def sample(lf: pl.LazyFrame) -> pl.LazyFrame:
    zones_ok = pl.all_horizontal(
        pl.col(c).is_not_null() & ~pl.col(c).is_in(UNKNOWN_ZONES) for c in ["pu_location_id", "do_location_id"]
    )
    return lf.filter(
        zones_ok & (pl.col("base_passenger_fare") > 0) & (pl.col("trip_miles") > 0) & (pl.col("trip_time") > 0)
    )


def frame(lf: pl.LazyFrame) -> pl.LazyFrame:
    wait = (pl.col("pickup_datetime") - pl.col("request_datetime")).dt.total_seconds() / 60
    return (
        sample(lf.select(KEEP))
        .with_columns(
            company=pl.col("hvfhs_license_num"),
            pickup_date=pl.col("pickup_datetime").dt.date(),
            date_hour=((pl.col("pickup_datetime") - T0).dt.total_hours()).cast(pl.Int32),
            od_pair=(pl.col("pu_location_id").cast(pl.Int32) * 1000 + pl.col("do_location_id")).cast(pl.Int32),
            wait_min=wait,
            log_fare=pl.col("base_passenger_fare").log(),
            log_miles=pl.col("trip_miles").log(),
            log_time=pl.col("trip_time").cast(pl.Float64).log(),
            # null where driver_pay <= 0 (Decisions 2026-11-09): 87% of those are shared requests, mostly
            # Via 2019-21 and Lyft 2019, so a recording convention rather than unpaid trips
            driver_share=pl.when(pl.col("driver_pay") > 0).then(pl.col("driver_pay") / pl.col("base_passenger_fare")),
            shared_request=(pl.col("shared_request_flag") == "Y").cast(pl.Int8),
            shared_match=(pl.col("shared_match_flag") == "Y").cast(pl.Int8),
        )
        .with_columns(log_wait=pl.when(pl.col("wait_min") > 0).then(pl.col("wait_min").log()))
    )


def stage_files(stage: Path, months: tuple[str, str] | None) -> dict[date, str]:
    out = {}
    for p in sorted((stage / "fhvhv").glob("fhvhv_*.parquet")):
        m = p.stem.split("_")[1]
        if months and not (months[0] <= m <= months[1]):
            continue
        out[date(int(m[:4]), int(m[5:7]), 1)] = str(p)
    return out


def add_months(d: date, k: int) -> date:
    n = d.year * 12 + d.month - 1 + k
    return date(n // 12, n % 12 + 1, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", type=Path, default=Path("build/nyc_tlc/stage"))
    ap.add_argument("--out", type=Path, default=Path("outbox"))
    ap.add_argument("--parts", type=int, default=50)
    ap.add_argument("--months", default=None, help="FROM,TO (YYYY-MM): test on these stage files")
    ap.add_argument("--name", default=NAME)
    a = ap.parse_args(argv)
    commit = (
        subprocess.run(["git", "-C", "repo", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        .stdout.strip() or "unknown"
    )  # fmt: skip
    files = stage_files(a.stage, tuple(a.months.split(",")) if a.months else None)
    raw = pl.scan_parquet(list(files.values()))

    # counts before and after the sample rules, and the month-offset check the part scans rely on
    pre = (
        raw.select(
            rows_in=pl.len(),
            zone_missing=(pl.col("pu_location_id").is_null() | pl.col("do_location_id").is_null()).sum(),
            zone_unknown=(pl.col("pu_location_id").is_in(UNKNOWN_ZONES) | pl.col("do_location_id").is_in(UNKNOWN_ZONES)).sum(),
            fare_le0=(pl.col("base_passenger_fare") <= 0).sum(),
            miles_le0=(pl.col("trip_miles") <= 0).sum(),
            time_le0=(pl.col("trip_time") <= 0).sum(),
            max_abs_month_offset=month_offset().abs().max(),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )  # fmt: skip
    if pre["max_abs_month_offset"] > MARGIN:
        raise SystemExit(f"a pickup lies {pre['max_abs_month_offset']} months from its file's month (> {MARGIN})")
    print(pre, flush=True)

    def scan_for(lo: date, hi: date) -> pl.LazyFrame:
        a_, b_ = add_months(lo.replace(day=1), -MARGIN), add_months(hi.replace(day=1), MARGIN)
        return frame(pl.scan_parquet([p for m, p in files.items() if a_ <= m <= b_]))

    d = a.out / a.name
    parts = write_parts(frame(raw), "pickup_date", d, a.parts, scan_for=scan_for)
    data = pl.scan_parquet(str(d / "data" / "*.parquet"))
    rows = sum(p["rows"] for p in parts)
    levels = level_counts(
        data,
        {
            "company": ["company"],
            "dispatching_base_num": ["dispatching_base_num"],
            "pu_location_id": ["pu_location_id"],
            "pickup_date": ["pickup_date"],
            "date_hour": ["date_hour"],
            "od_pair": ["od_pair"],
            "pu_location_id^date_hour": ["pu_location_id", "date_hour"],
            "do_location_id^date_hour": ["do_location_id", "date_hour"],
            "pu_location_id^do_location_id": ["pu_location_id", "do_location_id"],
        },
    )
    stats = (
        data.select(
            log_wait_null=pl.col("log_wait").is_null().mean(),
            driver_share_null=pl.col("driver_share").is_null().mean(),
            shared_request=pl.col("shared_request").mean(),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    # quantiles on every 100th row: exact ones would hold whole 1.7B-row columns in memory
    stats |= (
        data.select("driver_share", "wait_min")
        .gather_every(100)
        .select(
            driver_share_p01=pl.col("driver_share").quantile(0.01),
            driver_share_p50=pl.col("driver_share").median(),
            driver_share_p99=pl.col("driver_share").quantile(0.99),
            wait_min_p50=pl.col("wait_min").median(),
            wait_min_p99=pl.col("wait_min").quantile(0.99),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    by_company = (
        data.group_by("company").len().sort("len", descending=True).collect(engine="streaming").rows()
    )
    cols = ["log_fare", "log_wait", "driver_share", "log_miles", "log_time", "shared_request", "company",
            "pu_location_id", "do_location_id", "date_hour", CLUSTER]  # fmt: skip
    day = data.filter(pl.col("pickup_date") == REF_DAY)
    if day.select(pl.len()).collect().item() == 0:  # a --months test without REF_DAY
        first = data.select(pl.col("pickup_date").min()).collect().item()
        day = data.filter(pl.col("pickup_date") == first)
    ref_day = day.select(pl.col("pickup_date").first()).collect().item()
    fits = []
    for _, f in SPECS:
        y = f.split(" ~ ")[0]
        pdf = day.select(cols).drop_nulls([y, "log_miles", "log_time"]).collect().to_pandas()
        r = fit_reference(pdf, f, CLUSTER)
        r["subsample"] = f"pickup_date {ref_day}, rows with {y} null dropped"
        fits.append(r)
        print(r["nobs"], r["coef"], flush=True)
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
            "by_company": by_company,
            "fits": fits,
            "parts": parts,
        },
    )
    notes = {
        "company": "= hvfhs_license_num: HV0003 Uber, HV0005 Lyft, HV0004 Via, HV0002 Juno (TLC dictionary)",
        "pickup_date": "partition key; parts are contiguous date ranges",
        "date_hour": "hours since 2019-02-01 00:00 (local time as recorded)",
        "od_pair": "pu_location_id * 1000 + do_location_id; cluster",
        "wait_min": "pickup minus request, minutes",
        "log_wait": "log wait_min; null if request missing or wait_min <= 0",
        "log_fare": "log base_passenger_fare (before tolls, fees, tax and tips)",
        "log_time": "log trip_time (seconds)",
        "driver_share": "driver_pay / base_passenger_fare; null if driver_pay <= 0 (mostly shared rides, Via, "
        "Lyft 2019: a recording convention); not trimmed otherwise",
        "shared_request": "shared_request_flag == 'Y'",
        "source_month": "month of the TLC file the trip came from",
    }
    fit_rows = "\n".join(
        f"| {k} | `{f.replace('|', chr(92) + '|')}` | {r['nobs']:,} | "
        + ", ".join(f"{t} {v[0]:.6f} ({v[1]:.6f})" for t, v in r["coef"].items())
        + " |"
        for (k, f), r in zip(SPECS, fits)
    )
    lv = "\n".join(f"| {k} | {v:,} |" for k, v in levels.items())
    pt = "\n".join(
        f"| {p['file']} | {p['rows']:,} | {p['pickup_date_min']} to {p['pickup_date_max']} | {p['bytes'] / 1e9:.2f} GB |"
        for p in parts
    )
    comp = ", ".join(f"{c} {n:,}" for c, n in by_company)
    dropped = pre["rows_in"] - rows
    (d / "README.md").write_text(f"""# {a.name}

NYC TLC high-volume for-hire (rideshare) trip records, one row per trip, {min(files):%Y-%m} to
{max(files):%Y-%m}: {rows:,} trips. A scale test for dfeols (D5), not a replication: two FEs of
zone × hour (each in the millions of levels, nested in the date partition) plus the zone pair,
which crosses the partition.

- **Source:** NYC TLC trip records, `fhvhv_tripdata_YYYY-MM.parquet` from
  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page ({len(files)} monthly file{'s' if len(files) != 1 else ''},
  sha256-checked; `docs/sources/nyc_tlc.md` on valiant). No license is stated; the TLC disclaims
  accuracy. No driver or vehicle IDs.
- **Build:** `repo/replication/nyc_tlc/stage.py` (harmonised schema), `package.py`,
  jemather/dfeols `{commit}`
- **Built:** {date.today()} on valiant
- **Rows:** {rows:,} in {len(parts)} parts by `pickup_date`, of {pre["rows_in"]:,} in the files
  ({dropped:,} dropped)
- **Sample** (Decisions 2026-11-09): both zones known (not null, not 264 or 265), fare, miles and
  time > 0; driver_share null where driver_pay ≤ 0. Rows failing each rule (they overlap): zone missing {pre["zone_missing"]:,}, zone
  unknown {pre["zone_unknown"]:,}, fare ≤ 0 {pre["fare_le0"]:,}, miles ≤ 0 {pre["miles_le0"]:,},
  time ≤ 0 {pre["time_le0"]:,}
- **Companies:** {comp}
- **Distributions:** log_wait null {stats["log_wait_null"]:.4f}; on every 100th row, wait minutes median
  {stats["wait_min_p50"]:.2f}, p99 {stats["wait_min_p99"]:.1f}; driver_share null
  {stats["driver_share_null"]:.4f}, p1 / p50 / p99 {stats["driver_share_p01"]:.3f} /
  {stats["driver_share_p50"]:.3f} / {stats["driver_share_p99"]:.3f}; shared request
  {stats["shared_request"]:.4f}

## Parts

| File | Rows | pickup_date | Size |
|---|---|---|---|
{pt}

## Schema

{schema_table(data, notes)}

## FE / cluster levels

| Variable | Distinct |
|---|---:|
{lv}

## Suggested partition key and specs

Partition `pickup_date` (both zone × hour FEs are nested in it); cluster `{CLUSTER}` (the zone
pair, which crosses the partition). FEs `{FE}`.

| Spec | Formula | N (pyfixest, one day) | pyfixest b (se) |
|---|---|---:|---|
{fit_rows}

Reference subsample: pickups on {ref_day} (rows with the outcome null dropped; pyfixest drops
singletons). Full-precision values in `reference.json`. Then the full window; no reference
outside dfeols.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
