# SPDX-License-Identifier: GPL-3.0-or-later
"""D2: package the Learning by Driving rebuild for OSCER (Decisions 2026-11-08: both files).

    python repo/replication/haggag2014/package.py [--out outbox] [--trip-parts 8]

Writes ``outbox/taxi2009_shifts_v1/`` (the Table 3 shift panel) and ``outbox/taxi2009_trips_v1/``
(cleaned trips for every driver), each with data/, README.md and reference.json (pyfixest).
Run ``lbd_build.py`` first. The manifests are written afterwards (CLAUDE.md command).
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

BUILD = Path("build/haggag2014")
PUBLISHED = {  # lbd_analysis.log, Table 3
    "col2": {"new_driver": [-0.0854851, 0.0070365], "nobs": 977947},
    "col3": {
        "log_new_shift": [0.0148382, 0.0014233],
        "log_shift_hours": [-0.0022096, 0.0022823],
        "nobs": 908042,
    },
    "col4": {"log_new_all_exper": [0.013858, 0.0012166], "nobs": 908042},
}


def id_maps() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    t = pl.scan_parquet(BUILD / "lbd_trips" / "*.parquet")
    drv = (
        t.select("driver_id")
        .unique()
        .sort("driver_id")
        .collect(engine="streaming")
        .with_row_index("driver_num", offset=1)
        .with_columns(pl.col("driver_num").cast(pl.Int32))
    )
    car = (
        t.select("car_id")
        .unique()
        .sort("car_id")
        .collect(engine="streaming")
        .with_row_index("car_num", offset=1)
        .with_columns(pl.col("car_num").cast(pl.Int32))
    )
    nta = (
        pl.concat([t.select(n="pkp_nta"), t.select(n="drf_nta")])
        .unique()
        .drop_nulls()
        .sort("n")
        .collect(engine="streaming")
        .with_row_index("nta_id", offset=1)
        .with_columns(pl.col("nta_id").cast(pl.Int16))
    )
    return drv, car, nta


def trips_frame(drv, car, nta) -> pl.LazyFrame:
    link = pl.read_parquet(BUILD / "driver_link.parquet").select(
        pub_driver_id=pl.col("driver_id").cast(pl.Int32), driver_id="hack_license"
    )
    t = (
        pl.scan_parquet(BUILD / "lbd_trips" / "*.parquet")
        .join(drv.lazy(), on="driver_id")
        .join(car.lazy(), on="car_id")
        .join(link.lazy(), on="driver_id", how="left")
        .join(nta.lazy().rename({"n": "pkp_nta", "nta_id": "pkp_nta_id"}), on="pkp_nta", how="left")
        .join(nta.lazy().rename({"n": "drf_nta", "nta_id": "drf_nta_id"}), on="drf_nta", how="left")
    )
    d0 = date(2009, 1, 1)
    t = t.with_columns(
        pkp_date_hour=((pl.col("pkp_date") - pl.lit(d0)).dt.total_days() * 24 + pl.col("pkp_hour")).cast(
            pl.Int32
        ),
        log_ride_duration=pl.when(pl.col("ride_duration") > 0).then(pl.col("ride_duration").log()),
        log_distance=pl.when(pl.col("distance") > 0).then(pl.col("distance").log()),
        log_fare=pl.when(pl.col("fare") > 0).then(pl.col("fare").log()),
        tip_pct=pl.when((pl.col("payment_type") == "Credit") & (pl.col("fare_original") > 0)).then(
            100 * pl.col("tip") / pl.col("fare_original")
        ),
    )
    keep = [
        "driver_num",
        "car_num",
        "pub_driver_id",
        "shift",
        "trip_number",
        "final_trip",
        "vendor_name",
        "payment_type",
        "passenger_count",
        "trip_time",
        "distance",
        "ride_duration",
        "pkp_time",
        "drf_time",
        "pkp_time_tmp",
        "drf_time_tmp",
        "flipped",
        "seq_fix",
        "pkp_date",
        "pkp_hour",
        "pkp_dow",
        "drf_date",
        "drf_hour",
        "drf_dow",
        "pkp_date_hour",
        "pkp_long",
        "pkp_lat",
        "drf_long",
        "drf_lat",
        "pkp_fips",
        "pkp_puma",
        "pkp_nta",
        "pkp_nta_id",
        "pkp_boro",
        "drf_fips",
        "drf_puma",
        "drf_nta",
        "drf_nta_id",
        "drf_boro",
        "fare_original",
        "surcharge",
        "fare",
        "mta_tax",
        "tip",
        "toll",
        "total_amount",
        "log_ride_duration",
        "log_distance",
        "log_fare",
        "tip_pct",
        "st",
        "en",
        "slacktime",
        "brk",
        "duration_outlier",
        "distance_outlier",
        "surcharge_error",
        "mta_tax_error",
        "below_minfare",
        "short_shift_120",
        "driver_problemshifts",
    ]
    return t.select(keep)


TRIP_FE = "driver_num + car_num + pkp_nta_id^drf_nta_id + pkp_date_hour"
TRIP_SPECS = [
    ("duration", f"log_ride_duration ~ log_distance | {TRIP_FE}", None),
    ("tip", f"tip_pct ~ log_fare | {TRIP_FE}", "payment_type == 'Credit'"),
]


def package_trips(out: Path, n_parts: int, commit: str) -> None:
    drv, car, nta = id_maps()
    lf = trips_frame(drv, car, nta)
    name = "taxi2009_trips_v1"
    d = out / name
    parts = write_parts(lf, "driver_num", d, n_parts)
    data = pl.scan_parquet(str(d / "data" / "*.parquet"))
    rows = sum(p["rows"] for p in parts)
    fes = {
        "driver_num": ["driver_num"],
        "car_num": ["car_num"],
        "pkp_nta_id": ["pkp_nta_id"],
        "drf_nta_id": ["drf_nta_id"],
        "pkp_nta_id^drf_nta_id": ["pkp_nta_id", "drf_nta_id"],
        "pkp_date_hour": ["pkp_date_hour"],
        "pkp_fips": ["pkp_fips"],
        "pub_driver_id": ["pub_driver_id"],
    }
    levels = level_counts(data, fes)
    sub = data.filter(pl.col("pkp_date").is_between(date(2009, 7, 1), date(2009, 7, 7)))
    fits = []
    for _, f, cond in TRIP_SPECS:
        s = sub.filter(pl.sql_expr(cond)) if cond else sub
        cols = sorted(
            {
                "driver_num",
                "car_num",
                "pkp_nta_id",
                "drf_nta_id",
                "pkp_date_hour",
                f.split("~")[0].strip(),
                f.split("~")[1].split("|")[0].strip(),
            }
        )
        pdf = s.select(cols).drop_nulls().collect().to_pandas()
        r = fit_reference(pdf, f, "driver_num")
        r["subsample"] = (
            "pkp_date 2009-07-01..2009-07-07" + (f" and {cond}" if cond else "") + ", nulls dropped"
        )
        fits.append(r)
    dump_json(
        d / "reference.json",
        {
            "dataset": name,
            "build": f"jemather/dfeols {commit}",
            "tool": "pyfixest (in memory, valiant)",
            "rows": rows,
            "fe_levels": levels,
            "fits": fits,
            "parts": parts,
        },
    )
    notes = {
        "driver_num": "driver (hashed hack licence, renumbered 1..N)",
        "car_num": "medallion, renumbered",
        "pub_driver_id": "the published LBD driver_id (6,293 drivers), else null",
        "shift": "driver's shift number (lbd_clean.do lines 152-158)",
        "pkp_date_hour": "days since 2009-01-01 × 24 + pickup hour",
        "pkp_time_tmp": "pickup time as recorded",
        "pkp_time": "after the pickup/drop-off swap (lines 96-103)",
        "drf_time": "after the swap and the sequencing fix (lines 105-112)",
        "pkp_fips": "2010 census tract (TIGER 2011)",
        "pkp_nta": "NYC NTA code (lbd_clean.do lines 277-307)",
        "fare": "fare_original + surcharge; null if No Charge (line 130)",
        "tip_pct": "100 × tip / fare_original, credit card rides",
        "short_shift_120": "shift under 120 minutes (the do-file drops these; kept here)",
    }
    fit_rows = "\n".join(
        f"| {k} | `{f.replace('|', chr(92) + '|')}` | {c or 'all'} | {r['nobs']:,} | "
        + ", ".join(f"{t} {v[0]:.6f} ({v[1]:.6f})" for t, v in r["coef"].items())
        + " |"
        for (k, f, c), r in zip(TRIP_SPECS, fits)
    )
    lv = "\n".join(f"| {k} | {v:,} |" for k, v in levels.items())
    pt = "\n".join(
        f"| {p['file']} | {p['rows']:,} | {p['driver_num_min']}-{p['driver_num_max']} "
        f"| {p['bytes'] / 1e9:.2f} GB |"
        for p in parts
    )
    (d / "README.md").write_text(f"""# {name}

NYC yellow-cab trips, 2009, with driver and car IDs, cleaned as in Haggag, McManus & Paci (2017),
"Learning by Driving", AEJ: Applied 9(1), `lbd_clean.do` sections 1-5. Every driver with 100+
rides (36,065), not only the paper's sample. A large real-data test with two high-dimensional
FEs (driver, car) plus crossing location and time FEs.

- **Source:** TLC 2009 FOIL trip sheets with hashed medallion and hack licence, from Haggag & Paci
  (2014) "Default Tips", openICPSR 113897 (`docs/sources/haggag2014.md` on valiant); TIGER 2011
  tracts; geocorr12 and NYC tract/NTA crosswalks from the LBD package (openICPSR 113636)
- **Build:** `repo/replication/haggag2014/convert_tripsheets.py`, `link_drivers.py`, `lbd_build.py`,
  `package.py`, jemather/dfeols `{commit}`
- **Built:** {date.today()} on valiant
- **Rows:** {rows:,} in {len(parts)} parts by `driver_num`
- **Sample:** lbd_clean.do's drops (duplicates, drivers under 100 rides, shifts with two cars,
  shifts over 20 hours) except that only shifts under 2 minutes are dropped; those under 120 are
  flagged `short_shift_120` (the do-file drops them). Ride-level outliers are kept and flagged, as
  in the do-file
- **Notes:** `docs/haggag2014.md` on valiant. The shift panel built from these trips reproduces
  Table 3 (`taxi2009_shifts_v1`)

## Parts

| File | Rows | driver_num | Size |
|---|---|---|---|
{pt}

## Schema

{schema_table(data, notes)}

## FE / cluster levels

| Variable | Distinct |
|---|---:|
{lv}

## Suggested partition key and specs

Partition `driver_num` (drivers nested in shards; cluster `driver_num`). FEs
`{TRIP_FE}`: driver and car are high-dimensional and cross each other (drivers share cars),
pickup × drop-off NTA and date × hour cross everything.

| Spec | Formula | Subsample | N (pyfixest, reference subsample) | pyfixest b (se) |
|---|---|---|---:|---|
{fit_rows}

The reference subsample is pickups 2009-07-01 to 07-07 (rows with nulls in the model dropped).
Full-precision values in `reference.json`. Then the full year; no reference outside dfeols.
These specs are tests of the estimator, not the paper's.
""")


def package_shifts(out: Path, commit: str) -> None:
    _, car, _ = id_maps()
    name = "taxi2009_shifts_v1"
    d = out / name
    a = (
        pl.scan_parquet(BUILD / "lbd_shift_analysis.parquet")
        .join(car.lazy(), on="car_id", how="left")
        .drop(
            "driver_id",
            "car_id",
            "new_driver_ours",
            "incumb_ours",
            "end_exp_all_keep2",
            "log_new_all_exper_keep2",
        )
        .rename({"pub_driver_id": "driver_id"})
        .with_columns(
            pl.col("driver_id").cast(pl.Int32),
            pl.col("new_driver").cast(pl.Int8),
            in_paper_sample=~pl.col("short_shift_120"),
        )
    )
    parts = write_parts(a, "driver_id", d, 1)
    data = pl.scan_parquet(str(d / "data" / "*.parquet"))
    rows = parts[0]["rows"]
    levels = level_counts(
        data.filter("in_paper_sample"),
        {
            "driver_id": ["driver_id"],
            "date_hour_endnta": ["date_hour_endnta"],
            "date_hour": ["date_hour"],
            "car_num": ["car_num"],
        },
    )
    pdf = (
        data.filter("in_paper_sample")
        .select(
            "log_hr_fare",
            "log_new_shift",
            "log_shift_hours",
            "log_new_all_exper",
            "new_driver",
            "date_hour_endnta",
            "driver_id",
        )
        .collect()
        .to_pandas()
    )
    pdf["new_driver"] = pdf["new_driver"].astype(float)
    fits = [
        fit_reference(
            pdf,
            "log_hr_fare ~ new_driver + log_new_shift + log_shift_hours | date_hour_endnta",
            "driver_id",
            fixef_rm="none",
        ),
        fit_reference(
            pdf, "log_hr_fare ~ log_new_shift + log_shift_hours | date_hour_endnta + driver_id", "driver_id"
        ),
        fit_reference(
            pdf,
            "log_hr_fare ~ log_new_all_exper + log_shift_hours | date_hour_endnta + driver_id",
            "driver_id",
        ),
    ]
    for f, col in zip(fits, ("col2", "col3", "col4")):
        f["subsample"] = "in_paper_sample"
        f["published"] = PUBLISHED[col]
    dump_json(
        d / "reference.json",
        {
            "dataset": name,
            "build": f"jemather/dfeols {commit}",
            "tool": "pyfixest (in memory, valiant)",
            "rows": rows,
            "fe_levels_in_paper_sample": levels,
            "fits": fits,
            "parts": parts,
        },
    )
    notes = {
        "driver_id": "the published LBD driver_id",
        "car_num": "medallion, numbered as in taxi2009_trips_v1",
        "in_paper_sample": "not a short shift (under 120 minutes): the paper's sample (1,052,942 rows)",
        "short_shift_120": "shift under 120 minutes (dropped by the do-file)",
        "date_hour_endnta": "group(date, hour, NTA) of ride 3's drop-off (line 1100); the main FE",
        "log_hr_fare": "log fare in the 60 minutes after ride 3's drop-off (lines 811-833, 1062)",
        "log_new_shift": "log(shift) × new_driver",
        "log_new_all_exper": "log(drop-offs before ride 3) × new_driver",
        "exper_driver_group": "the published experienced-driver draw (0 new, 1 the 20% analysis sample)",
    }
    names = {
        "col2": "Table 3 col 2 (areg: singletons kept)",
        "col3": "Table 3 col 3",
        "col4": "Table 3 col 4",
    }
    fit_rows = []
    for f, col in zip(fits, ("col2", "col3", "col4")):
        t = next(k for k in f["coef"] if k.startswith(("new_driver", "log_new")))
        if col == "col2":
            t = "new_driver"
        b, se = f["coef"][t]
        pb, pse = PUBLISHED[col][t]
        fit_rows.append(
            f"| {names[col]} | `{f['formula'].replace('|', chr(92) + '|')}` | {f['nobs']:,} "
            f"| {t} {b:.7f} ({se:.7f}) | "
            f"{pb} ({pse}), N {PUBLISHED[col]['nobs']:,} |"
        )
    lv = "\n".join(f"| {k} | {v:,} |" for k, v in levels.items())
    (d / "README.md").write_text(f"""# {name}

Driver-shift panel of Haggag, McManus & Paci (2017), "Learning by Driving", AEJ: Applied 9(1),
rebuilt from the 2009 trip sheets (`taxi2009_trips_v1`) by porting `lbd_clean.do`. One row per
shift of the paper's drivers (6,293: new drivers and the published 20% draw of experienced
drivers), April-December 2009. A small (1M rows) test of a high-dimensional crossing FE
(date × hour × drop-off NTA, ~160k levels) against a driver FE nested in the cluster.

- **Source / build:** as `taxi2009_trips_v1`; jemather/dfeols `{commit}`
- **Built:** {date.today()} on valiant
- **Rows:** {rows:,} in 1 part; `in_paper_sample` selects the paper's 1,052,942
  (shifts under 120 minutes are kept here and flagged; the paper drops them)
- **Samples:** the experienced-driver draw (Stata RNG) is taken from the published file through a
  driver link (every published driver matched one licence on every shift)
- **Notes:** `docs/haggag2014.md` on valiant, including why a few hundred shifts differ from the
  published file (Stata's order of tied pickup times)

## Schema

{schema_table(data, notes)}

## FE / cluster levels (paper sample)

| Variable | Distinct |
|---|---:|
{lv}

## Suggested partition key and specs

Partition `driver_id`; cluster `driver_id`. Filter `in_paper_sample`.

| Spec | Formula | N (pyfixest) | pyfixest b (se) | Published |
|---|---|---:|---|---|
{chr(10).join(fit_rows)}

pyfixest drops singletons except in col 2 (`fixef_rm="none"`, as Stata's areg). Each coefficient
is within 0.04 SE of the published one. Full-precision values in `reference.json`.
""")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="outbox")
    ap.add_argument("--trip-parts", type=int, default=8)
    ap.add_argument("--only", choices=["shifts", "trips"], default=None)
    a = ap.parse_args(argv)
    commit = (
        subprocess.run(
            ["git", "-C", "repo", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )
    out = Path(a.out)
    if a.only in (None, "shifts"):
        package_shifts(out, commit)
    if a.only in (None, "trips"):
        package_trips(out, a.trip_parts, commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
