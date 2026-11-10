# SPDX-License-Identifier: GPL-3.0-or-later
"""D5 at full scale: package every NYC TLC service as ``outbox/nyc_tlc_trips_v1/``.

    python repo/replication/nyc_tlc/package_trips.py [--stage build/nyc_tlc/stage]
        [--zoned build/nyc_tlc/zoned] [--out outbox] [--parts 70] [--months 2019-03,2019-03]

Run from the project root, after ``stage.py`` and ``zones.py`` (yellow 2009-10 zones). Reads
yellow 2009-01..2026-09 (2009-10 from ``--zoned``), green 2014-01.., fhv 2015-01.. and fhvhv
2019-02.., maps each to one schema, applies the sample rules (Decisions 2026-11-09 and the rules
agreed for this build, see SAMPLE below), writes data/ in contiguous ranges of pickup date, then
README.md and reference.json (pyfixest on one day; spec 3 on two days across a year boundary).
Manifest afterwards (CLAUDE.md).

Columns common to all services: ``service`` (yellow, green, fhv, or the fhvhv company licence
HV0002..HV0005), ``base`` (dispatching base; null for yellow and green), times, zones, miles,
trip time, fare and their logs. ``trip_time`` is dropoff minus pickup in seconds, except fhvhv,
which keeps its recorded ``trip_time`` as in nyc_fhvhv_v1. ``fare`` is ``fare_amount`` (yellow,
green: the metered fare) or ``base_passenger_fare`` (fhvhv: before tolls, fees, tax and tips);
fhv has no miles or fare.
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

NAME = "nyc_tlc_trips_v1"
T0 = datetime(2009, 1, 1)  # date_hour = hours since the first yellow file's month
REF_DAY = date(2019, 3, 6)  # a Wednesday with all four services; ~1.2M trips
REF_YEAR_DAYS = (date(2018, 12, 31), date(2019, 1, 1))  # spec 3: the zone pair crosses the year
UNKNOWN_ZONES = [264, 265]
SERVICES = ["yellow", "green", "fhv", "fhvhv"]
LATLON_MONTHS = ("2009-01", "2010-12")  # yellow files zoned by zones.py

FE = "pu_location_id^date_hour + do_location_id^date_hour + pu_location_id^do_location_id"
FE3 = "pu_location_id^date_hour + do_location_id^date_hour + pu_location_id^do_location_id^year"
SVC = 'i(service, ref="yellow")'
SPECS = [  # (name, formula, services it runs on)
    ("duration", f"log_time ~ {SVC} | {FE}", None),
    ("fare", f"log_fare ~ log_miles + log_time + {SVC} | {FE}",
     ["yellow", "green", "HV0002", "HV0003", "HV0004", "HV0005"]),
    ("duration, pair x year", f"log_time ~ {SVC} | {FE3}", None),
]  # fmt: skip
CLUSTER = "od_pair"

# SAMPLE: a trip is kept if
#   - both zones are known (not null, not 264/265; 2009-10 points in no zone are null);
#   - the pickup lies in its file's month (DROP_OFF_MONTH);
#   - trip_time > 0 (every spec needs log_time), and where the service records them, fare > 0
#     and miles > 0 (V1_RULES: nyc_fhvhv_v1's rule, so the fhvhv rows equal that dataset's).
DROP_OFF_MONTH = True
V1_RULES = True


def month_offset() -> pl.Expr:
    p, s = pl.col("pickup_datetime"), pl.col("source_month")
    return (p.dt.year() * 12 + p.dt.month()) - (s.dt.year() * 12 + s.dt.month())


def seconds() -> pl.Expr:
    return (pl.col("dropoff_datetime") - pl.col("pickup_datetime")).dt.total_seconds()


NULL_F, NULL_S = pl.lit(None, pl.Float64), pl.lit(None, pl.String)


def common(lf: pl.LazyFrame, service: str) -> pl.LazyFrame:
    """One service's stage frame in the common schema, before the sample rules."""
    if service in ("yellow", "green"):
        sel = dict(service=pl.lit(service), base=NULL_S, trip_miles=pl.col("trip_distance"),
                   trip_time=seconds(), fare=pl.col("fare_amount"))  # fmt: skip
    elif service == "fhv":
        sel = dict(service=pl.lit("fhv"), base=pl.col("dispatching_base_num"), trip_miles=NULL_F,
                   trip_time=seconds(), fare=NULL_F)  # fmt: skip
    else:
        sel = dict(service=pl.col("hvfhs_license_num"), base=pl.col("dispatching_base_num"),
                   trip_miles=pl.col("trip_miles"), trip_time=pl.col("trip_time"),
                   fare=pl.col("base_passenger_fare"))  # fmt: skip
    return lf.select(
        sel["service"].cast(pl.String).alias("service"),
        sel["base"].cast(pl.String).alias("base"),
        "pickup_datetime", "dropoff_datetime",
        pl.col("pu_location_id").cast(pl.Int16), pl.col("do_location_id").cast(pl.Int16),
        sel["trip_miles"].cast(pl.Float64).alias("trip_miles"),
        sel["trip_time"].cast(pl.Int64).alias("trip_time"),
        sel["fare"].cast(pl.Float64).alias("fare"),
        "source_month",
    )  # fmt: skip


def zones_ok() -> pl.Expr:
    return pl.all_horizontal(
        pl.col(c).is_not_null() & ~pl.col(c).is_in(UNKNOWN_ZONES)
        for c in ["pu_location_id", "do_location_id"]
    )


def keep() -> pl.Expr:
    k = zones_ok() & (pl.col("trip_time") > 0)
    if DROP_OFF_MONTH:
        k &= month_offset() == 0
    if V1_RULES:  # null fare/miles (fhv) pass
        k &= (pl.col("fare") > 0).fill_null(True) & (pl.col("trip_miles") > 0).fill_null(True)
    return k


def frame(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Sample rules and model columns on the common schema."""
    pos_log = lambda c: pl.when(pl.col(c) > 0).then(pl.col(c).cast(pl.Float64).log())
    return lf.filter(keep()).with_columns(
        pickup_date=pl.col("pickup_datetime").dt.date(),
        year=pl.col("pickup_datetime").dt.year().cast(pl.Int16),
        date_hour=((pl.col("pickup_datetime") - T0).dt.total_hours()).cast(pl.Int32),
        od_pair=(pl.col("pu_location_id").cast(pl.Int32) * 1000 + pl.col("do_location_id")).cast(pl.Int32),
        log_time=pos_log("trip_time"),
        log_miles=pos_log("trip_miles"),
        log_fare=pos_log("fare"),
    )


def stage_files(stage: Path, zoned: Path, months: tuple[str, str] | None) -> dict[str, dict[date, str]]:
    out: dict[str, dict[date, str]] = {}
    for s in SERVICES:
        out[s] = {}
        for p in sorted((stage / s).glob(f"{s}_*.parquet")):
            m = p.stem.split("_")[1]
            if months and not (months[0] <= m <= months[1]):
                continue
            if s == "yellow" and LATLON_MONTHS[0] <= m <= LATLON_MONTHS[1]:
                p = zoned / p.name
                if not p.exists():
                    raise SystemExit(f"{p} missing: run zones.py first")
            out[s][date(int(m[:4]), int(m[5:7]), 1)] = str(p)
    return out


def scan(files: dict[str, dict[date, str]], lo: date | None = None, hi: date | None = None) -> pl.LazyFrame:
    """Every service's files (or only the months lo..hi), in the common schema."""
    parts = []
    for s, by_m in files.items():
        ps = [p for m, p in by_m.items() if lo is None or lo <= m <= hi]
        if ps:
            parts.append(common(pl.scan_parquet(ps), s))
    return pl.concat(parts, how="vertical")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", type=Path, default=Path("build/nyc_tlc/stage"))
    ap.add_argument("--zoned", type=Path, default=Path("build/nyc_tlc/zoned"))
    ap.add_argument("--out", type=Path, default=Path("outbox"))
    ap.add_argument("--parts", type=int, default=70)
    ap.add_argument("--months", default=None, help="FROM,TO (YYYY-MM): test on these stage files")
    ap.add_argument("--name", default=NAME)
    a = ap.parse_args(argv)
    commit = (
        subprocess.run(["git", "-C", "repo", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        .stdout.strip() or "unknown"
    )  # fmt: skip
    files = stage_files(a.stage, a.zoned, tuple(a.months.split(",")) if a.months else None)
    raw = scan(files)

    # rows failing each rule (they overlap), by service group
    grp = pl.when(pl.col("service").str.starts_with("HV")).then(pl.lit("fhvhv")).otherwise("service")
    pre = (
        raw.group_by(grp.alias("svc"))
        .agg(
            rows_in=pl.len(),
            zone_bad=(~zones_ok()).sum(),
            off_month=(month_offset() != 0).sum(),
            time_le0=(pl.col("trip_time") <= 0).sum(),
            time_null=pl.col("trip_time").is_null().sum(),
            fare_le0=(pl.col("fare") <= 0).sum(),
            miles_le0=(pl.col("trip_miles") <= 0).sum(),
            kept=keep().sum(),
        )
        .sort("svc")
        .collect(engine="streaming")
    )
    print(pre, flush=True)
    sample_counts = {r.pop("svc"): r for r in pre.rows(named=True)}

    def scan_for(lo: date, hi: date) -> pl.LazyFrame:
        # with DROP_OFF_MONTH every kept pickup is in its file's month, so the part's months suffice
        return frame(scan(files, lo.replace(day=1), hi.replace(day=1)))

    if not DROP_OFF_MONTH:
        raise SystemExit("scan_for assumes DROP_OFF_MONTH")
    d = a.out / a.name
    parts = write_parts(frame(raw), "pickup_date", d, a.parts, scan_for=scan_for)
    data = pl.scan_parquet(str(d / "data" / "*.parquet"))
    rows = sum(p["rows"] for p in parts)
    levels = level_counts(
        data,
        {
            "service": ["service"],
            "base": ["base"],
            "pu_location_id": ["pu_location_id"],
            "pickup_date": ["pickup_date"],
            "date_hour": ["date_hour"],
            "od_pair": ["od_pair"],
            "pu_location_id^date_hour": ["pu_location_id", "date_hour"],
            "do_location_id^date_hour": ["do_location_id", "date_hour"],
            "pu_location_id^do_location_id": ["pu_location_id", "do_location_id"],
            "pu_location_id^do_location_id^year": ["pu_location_id", "do_location_id", "year"],
        },
    )
    by_service = (
        data.group_by("service")
        .agg(rows=pl.len(), first=pl.col("pickup_date").min(), last=pl.col("pickup_date").max(),
             log_fare_null=pl.col("log_fare").is_null().sum())  # fmt: skip
        .sort("rows", descending=True)
        .collect(engine="streaming")
        .rows(named=True)
    )
    # quantiles on every 1000th row: exact ones would hold whole multi-billion-row columns in memory
    stats = (
        data.select("trip_time", "trip_miles", "fare")
        .gather_every(1000)
        .select(
            **{f"{c}_p{int(q * 100):02d}": pl.col(c).quantile(q) for c in ["trip_time", "trip_miles", "fare"]
               for q in (0.01, 0.5, 0.99)}  # fmt: skip
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    cols = ["log_time", "log_fare", "log_miles", "service", "pu_location_id", "do_location_id", "date_hour",
            "year", CLUSTER]  # fmt: skip

    def ref_frame(days):
        f = data.filter(pl.col("pickup_date").is_in(list(days)))
        if f.select(pl.len()).collect().item() == 0:  # a --months test without the day
            days = [data.select(pl.col("pickup_date").min()).collect().item()]
            f = data.filter(pl.col("pickup_date").is_in(days))
        return f, days

    fits = []
    for name, f, svcs in SPECS:
        day, days = ref_frame(REF_YEAR_DAYS if "year" in f else [REF_DAY])
        if svcs is not None:
            day = day.filter(pl.col("service").is_in(svcs))
        y = f.split(" ~ ")[0]
        need = [y, "log_time"] + (["log_miles"] if "log_miles" in f else [])
        pdf = day.select(cols).drop_nulls(need).collect().to_pandas()
        r = fit_reference(pdf, f, CLUSTER)
        r["subsample"] = (
            f"pickup_date in {[str(x) for x in days]}"
            + (f", service in {svcs}" if svcs else "")
            + ", rows with a model variable null dropped"
        )
        r["spec"] = name
        fits.append(r)
        print(name, r["nobs"], r["coef"], flush=True)
    dump_json(
        d / "reference.json",
        {
            "dataset": a.name,
            "build": f"jemather/dfeols {commit}",
            "tool": "pyfixest (in memory, valiant)",
            "rows": rows,
            "sample_counts": sample_counts,
            "fe_levels": levels,
            "stats": stats,
            "by_service": by_service,
            "fits": fits,
            "parts": parts,
        },
    )
    write_readme(d, a.name, commit, files, rows, parts, sample_counts, levels, by_service, stats, fits, data)
    return 0


def write_readme(d, name, commit, files, rows, parts, sample_counts, levels, by_service, stats, fits, data):
    notes = {
        "service": "yellow, green, fhv, or the fhvhv company licence: HV0003 Uber, HV0005 Lyft, HV0004 Via, "
        "HV0002 Juno (TLC dictionary)",
        "base": "dispatching base licence (fhv, fhvhv); null for yellow and green",
        "pu_location_id": "TLC taxi zone 1-263; yellow 2009-10 by point-in-polygon on the TLC shapefile",
        "trip_miles": "trip_distance (yellow, green) or trip_miles (fhvhv); null for fhv",
        "trip_time": "seconds: dropoff minus pickup, except fhvhv's recorded trip_time",
        "fare": "fare_amount (yellow, green: metered fare) or base_passenger_fare (fhvhv: before tolls, "
        "fees, tax and tips); null for fhv. Not the same concept across services",
        "source_month": "month of the TLC file the trip came from (always the pickup's month)",
        "pickup_date": "partition key; parts are contiguous date ranges",
        "year": "pickup year (for the pu^do^year FE)",
        "date_hour": "hours since 2009-01-01 00:00 (local time as recorded)",
        "od_pair": "pu_location_id * 1000 + do_location_id; cluster",
        "log_time": "log trip_time",
        "log_miles": "log trip_miles; null for fhv",
        "log_fare": "log fare; null for fhv",
    }
    esc = lambda s: s.replace("|", chr(92) + "|")
    fit_rows = "\n".join(
        f"| {r['spec']} | `{esc(r['formula'])}` | {r['subsample'].split(', rows')[0]} | {r['nobs']:,} | "
        + ", ".join(f"{t} {v[0]:.6f} ({v[1]:.6f})" for t, v in r["coef"].items())
        + " |"
        for r in fits
    )
    lv = "\n".join(f"| {k} | {v:,} |" for k, v in levels.items())
    pt = "\n".join(
        f"| {p['file']} | {p['rows']:,} | {p['pickup_date_min']} to {p['pickup_date_max']} | "
        f"{p['bytes'] / 1e9:.2f} GB |"
        for p in parts
    )
    sv = "\n".join(
        f"| {r['service']} | {r['rows']:,} | {r['first']} to {r['last']} |" for r in by_service
    )
    sc = "\n".join(
        f"| {s} | {c['rows_in']:,} | {c['zone_bad']:,} | {c['off_month']:,} | {c['time_le0']:,} | "
        f"{c['fare_le0']:,} | {c['miles_le0']:,} | {c['kept']:,} |"
        for s, c in sample_counts.items()
    )
    nfiles = sum(len(v) for v in files.values())
    rows_in = sum(c["rows_in"] for c in sample_counts.values())
    st = ", ".join(f"{k} {v:,.2f}" for k, v in stats.items())
    (d / "README.md").write_text(f"""# {name}

NYC TLC trip records from every service, one row per trip: yellow taxis, green (boro) taxis,
for-hire vehicles (fhv) and high-volume for-hire rideshare (fhvhv), {rows:,} trips. A scale
test for dfeols (D5 at full scale), not a replication: two FEs of zone × hour (tens of millions
of levels, nested in the date partition) plus the zone pair (or zone pair × year), which crosses
the partition. `nyc_fhvhv_v1` is the fhvhv rows alone, with more columns.

- **Source:** NYC TLC trip records, `{{yellow,green,fhv,fhvhv}}_tripdata_YYYY-MM.parquet` from
  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page ({nfiles} monthly files,
  sha256-checked; `docs/sources/nyc_tlc.md` on valiant), and the TLC taxi zone shapefile for
  yellow 2009-10. No license is stated; the TLC disclaims accuracy. No driver or vehicle IDs.
- **Build:** `repo/replication/nyc_tlc/stage.py` (harmonised schema), `zones.py` (2009-10
  zones), `package_trips.py`, jemather/dfeols `{commit}`
- **Built:** {date.today()} on valiant
- **Rows:** {rows:,} in {len(parts)} parts by `pickup_date`, of {rows_in:,} in the files
- **Sample** (STATE.md Decisions 2026-11-09 and 2026-11-10): both zones known (not null, not
  264 "Unknown" or 265 "Outside of NYC"; a 2009-10 point in no zone counts as unknown); pickup in
  its file's month; trip_time > 0; fare > 0 and miles > 0 where the service records them
  (yellow, green, fhvhv). No trimming otherwise (trips over a day are kept).
- **fare is not one concept:** taxi `fare_amount` is the metered fare; fhvhv
  `base_passenger_fare` is the base fare before tolls, fees, tax and tips. Spec 2's `i(service)`
  absorbs the level difference only.
- **Distributions** (every 1000th row): {st}

Rows failing each rule (they overlap):

| Service | Rows in files | Zone unknown | Off month | Time ≤ 0 | Fare ≤ 0 | Miles ≤ 0 | Kept |
|---|---:|---:|---:|---:|---:|---:|---:|
{sc}

## Services

| service | Rows | pickup_date |
|---|---:|---|
{sv}

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
pair, which crosses the partition).

| Spec | Formula | Reference subsample | N (pyfixest) | pyfixest b (se) |
|---|---|---|---:|---|
{fit_rows}

Spec 2 runs on yellow, green and fhvhv only (fhv has no fare). pyfixest drops singletons;
rows with a model variable null are dropped. Full-precision values in `reference.json`. Then
the full window; no reference outside dfeols.
""")


if __name__ == "__main__":
    sys.exit(main())
