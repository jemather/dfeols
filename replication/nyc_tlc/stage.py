# SPDX-License-Identifier: GPL-3.0-or-later
"""D5 stage 1: harmonise the NYC TLC monthly trip files to one schema per type.

    python repo/replication/nyc_tlc/stage.py [--types yellow,green,fhv,fhvhv]
        [--from 2009-01] [--to 2026-09] [--raw raw/nyc_tlc] [--out build/nyc_tlc/stage]
        [--workers 4] [--force]

Run from the project root. For each ``<type>_tripdata_YYYY-MM.parquet``:

  1. check the file against ``raw/nyc_tlc/SHA256SUMS``;
  2. rename columns to the canonical snake_case names (RENAME), dropping only pandas'
     ``__index_level_0__`` (in yellow 2010-02/03);
  3. cast to the type's SCHEMA. Values are kept as recorded: coded fields whose encoding
     changed over the years (``vendor``, ``payment_type``, ``store_and_fwd_flag``) become
     strings of the raw value, with no recoding. A float column cast to an integer must hold
     whole numbers (checked; the file fails otherwise). Columns a month lacks are typed nulls;
  4. add ``source_month`` (first of the month), write
     ``<out>/<type>/<type>_YYYY-MM.parquet`` atomically, then a ``.json`` sidecar (source hash,
     rows in and out, null counts, build commit). Rows out must equal rows in.

Not merged: yellow 2009-10 ``surcharge`` is kept apart from 2011+ ``extra``, and 2009's
``store_and_forward`` (a double) apart from ``store_and_fwd_flag``. Any recoding is a later,
agreed step. A file whose sidecar matches its source hash is skipped, so a rerun resumes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")
import polars as pl

log = logging.getLogger("nyc_tlc.stage")

TS = pl.Datetime("us")  # local New York time, as published (no time zone)
F, I16, I64, S = pl.Float64, pl.Int16, pl.Int64, pl.String

TAXI = {  # yellow and green share one schema; green-only columns are null for yellow
    "vendor": S, "pickup_datetime": TS, "dropoff_datetime": TS, "passenger_count": I16,
    "trip_distance": F, "rate_code": I16, "store_and_fwd_flag": S, "store_and_forward": F,
    "pickup_longitude": F, "pickup_latitude": F, "dropoff_longitude": F, "dropoff_latitude": F,
    "pu_location_id": I16, "do_location_id": I16, "payment_type": S, "trip_type": I16,
    "fare_amount": F, "surcharge": F, "extra": F, "mta_tax": F, "tip_amount": F,
    "tolls_amount": F, "ehail_fee": F, "improvement_surcharge": F, "congestion_surcharge": F,
    "airport_fee": F, "cbd_congestion_fee": F, "total_amount": F, "request_source": S,
}
SCHEMA = {
    "yellow": TAXI,
    "green": TAXI,
    "fhv": {
        "dispatching_base_num": S, "pickup_datetime": TS, "dropoff_datetime": TS,
        "pu_location_id": I16, "do_location_id": I16, "sr_flag": I16,
        "affiliated_base_number": S,
    },
    "fhvhv": {
        "hvfhs_license_num": S, "dispatching_base_num": S, "originating_base_num": S,
        "request_datetime": TS, "on_scene_datetime": TS, "pickup_datetime": TS,
        "dropoff_datetime": TS, "pu_location_id": I16, "do_location_id": I16, "trip_miles": F,
        "trip_time": I64, "base_passenger_fare": F, "tolls": F, "bcf": F, "sales_tax": F,
        "congestion_surcharge": F, "airport_fee": F, "cbd_congestion_fee": F, "tips": F,
        "driver_pay": F, "shared_request_flag": S, "shared_match_flag": S,
        "access_a_ride_flag": S, "wav_request_flag": S, "wav_match_flag": S,
    },
}
RENAME = {  # raw name -> canonical; names already canonical are absent
    # yellow 2009
    "vendor_name": "vendor", "Trip_Pickup_DateTime": "pickup_datetime",
    "Trip_Dropoff_DateTime": "dropoff_datetime", "Passenger_Count": "passenger_count",
    "Trip_Distance": "trip_distance", "Start_Lon": "pickup_longitude",
    "Start_Lat": "pickup_latitude", "Rate_Code": "rate_code", "End_Lon": "dropoff_longitude",
    "End_Lat": "dropoff_latitude", "Payment_Type": "payment_type", "Fare_Amt": "fare_amount",
    "Tip_Amt": "tip_amount", "Tolls_Amt": "tolls_amount", "Total_Amt": "total_amount",
    # yellow 2010
    "vendor_id": "vendor",
    # yellow/green 2011+, fhv, fhvhv
    "VendorID": "vendor", "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime", "lpep_pickup_datetime": "pickup_datetime",
    "lpep_dropoff_datetime": "dropoff_datetime", "RatecodeID": "rate_code",
    "PULocationID": "pu_location_id", "DOLocationID": "do_location_id",
    "PUlocationID": "pu_location_id", "DOlocationID": "do_location_id",
    "Airport_fee": "airport_fee", "dropOff_datetime": "dropoff_datetime", "SR_Flag": "sr_flag",
    "Affiliated_base_number": "affiliated_base_number",
}
DROP = {"__index_level_0__"}
NAME = re.compile(r"^(yellow|green|fhv|fhvhv)_tripdata_(\d{4})-(\d{2})\.parquet$")
INTS = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def to_type(col: str, src: pl.DataType, dst: pl.DataType) -> pl.Expr:
    c = pl.col(col)
    if src == pl.Null:
        return pl.lit(None, dtype=dst).alias(col)
    if dst == TS:
        if src == pl.String:
            return c.str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us", strict=True).alias(col)
        return c.cast(TS, strict=True)
    if dst == S:
        if src.is_float():  # whole-number codes stored as doubles -> "1", not "1.0"
            return c.cast(pl.Int64, strict=True).cast(S)
        return c.cast(S)
    if dst in INTS and src == pl.String:
        return c.str.strip_chars().replace("", None).cast(dst, strict=True)
    return c.cast(dst, strict=True)


def stage_one(src: str, typ: str, month: str, digest: str, dest: str, commit: str) -> dict:
    t0 = time.time()
    lf = pl.scan_parquet(src)
    raw_schema = lf.collect_schema()
    target = SCHEMA[typ]
    names = {}
    for n in raw_schema.names():
        if n in DROP:
            continue
        canon = RENAME.get(n, n)
        if canon not in target:
            raise ValueError(f"{Path(src).name}: column {n!r} has no place in the {typ} schema")
        if canon in names.values():
            raise ValueError(f"{Path(src).name}: two columns map to {canon!r}")
        names[n] = canon
    lf = lf.select([pl.col(n).alias(c) for n, c in names.items()])
    src_types = {c: raw_schema[n] for n, c in names.items()}

    # floats going to integer columns (or coded strings) must be whole numbers
    fl = [c for c, t in src_types.items() if t.is_float() and (target[c] in INTS or target[c] == S)]
    checks = [pl.len().alias("__rows")] + [
        ((pl.col(c) - pl.col(c).round()).abs() > 0).sum().alias(c) for c in fl]
    chk = lf.select(checks).collect(engine="streaming").row(0, named=True)
    rows_in = chk.pop("__rows")
    bad = {c: n for c, n in chk.items() if n}
    if bad:
        raise ValueError(f"{Path(src).name}: non-integral values in {bad}")

    exprs = []
    for c, t in target.items():
        if c in src_types:
            exprs.append(to_type(c, src_types[c], t))
        else:
            exprs.append(pl.lit(None, dtype=t).alias(c))
    y, m = map(int, month.split("-"))
    out = lf.select(exprs).with_columns(source_month=pl.lit(date(y, m, 1)))

    d = Path(dest)
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.with_suffix(".parquet.part")
    out.sink_parquet(tmp, compression="zstd", statistics=True)
    st = pl.scan_parquet(tmp).select(
        pl.len().alias("rows"), *[pl.col(c).null_count() for c in target]).collect().row(0, named=True)
    rows_out = st.pop("rows")
    if rows_out != rows_in:
        tmp.unlink()
        raise ValueError(f"{Path(src).name}: rows out {rows_out} != rows in {rows_in}")
    os.replace(tmp, d)
    side = {"source": f"{typ}/{Path(src).name}", "sha256": digest, "build_commit": commit,
            "rows": rows_out, "renamed": {n: c for n, c in names.items() if n != c},
            "absent": [c for c in target if c not in src_types],
            "nulls": {c: n for c, n in st.items() if n}, "seconds": round(time.time() - t0, 1)}
    d.with_suffix(".json").write_text(json.dumps(side, indent=1))
    return side


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--types", default="yellow,green,fhv,fhvhv")
    ap.add_argument("--from", dest="start", default="0000-00")
    ap.add_argument("--to", dest="end", default="9999-99")
    ap.add_argument("--raw", default="raw/nyc_tlc")
    ap.add_argument("--out", default="build/nyc_tlc/stage")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    raw, out = Path(a.raw), Path(a.out)
    types = a.types.split(",")
    sums = {}
    for line in (raw / "SHA256SUMS").read_text().splitlines():
        digest, _, name = line.partition("  ")
        sums[name] = digest
    commit = subprocess.run(["git", "-C", "repo", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    todo = []
    for name in sorted(sums):
        m = NAME.match(Path(name).name)
        if not m or m.group(1) not in types:
            continue
        typ, month = m.group(1), f"{m.group(2)}-{m.group(3)}"
        if not (a.start <= month <= a.end):
            continue
        dest = out / typ / f"{typ}_{month}.parquet"
        side = dest.with_suffix(".json")
        if not a.force and dest.exists() and side.exists():
            try:
                if json.loads(side.read_text()).get("sha256") == sums[name]:
                    continue
            except (OSError, ValueError):
                pass
        todo.append((str(raw / name), typ, month, sums[name], str(dest), commit))
    log.info("%d files to stage, %d workers, commit %s", len(todo), a.workers, commit)

    # hash check in the parent, before any work is handed out
    import hashlib
    for t in todo:
        h = hashlib.sha256()
        with open(t[0], "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        if h.hexdigest() != t[3]:
            log.error("sha256 mismatch for %s; nothing staged", t[0])
            return 1
    log.info("sha256 verified for all %d files", len(todo))

    failed = done = rows = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(stage_one, *t): t[0] for t in todo}
        for f in as_completed(futs):
            try:
                s = f.result()
                done += 1
                rows += s["rows"]
                log.info("ok %s %d rows %.0fs (%d/%d, %.0f s)", s["source"], s["rows"], s["seconds"],
                         done, len(todo), time.time() - t0)
            except Exception as e:
                failed += 1
                log.error("FAILED %s: %s", futs[f], e)
    log.info("finished: %d staged, %d failed, %.3fB rows", done, failed, rows / 1e9)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
