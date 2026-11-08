# SPDX-License-Identifier: GPL-3.0-or-later
"""D2 stage 1: convert the 2009 TLC FOIL trip sheets (Haggag & Paci 2014 package, openICPSR
113897) to Parquet, one file per month, values as recorded.

    python repo/replication/haggag2014/convert_tripsheets.py
        [--zip build/haggag2014/extract/Raw-Data--FOIL_TPEP_2009-.zip]
        [--out build/haggag2014/tripsheets] [--tmp tmp/haggag2014] [--workers 3] [--force]

Run from the project root. The inner zip was extracted unchanged from
``raw/haggag2014/113897-V1.zip`` (sha256 608d5a31...). For each ``Tripsheet_2009_M.csv``
(tab-separated, header line):

  1. stream it out of the zip to ``--tmp`` (zipfile checks the CRC), counting lines;
  2. read it as strings, cast strictly to SCHEMA (anything that doesn't parse fails the
     month), with names lower-cased; datetimes are local New York time as published;
  3. write ``<out>/tripsheet_2009_MM.parquet`` atomically and a ``.json`` sidecar (member CRC,
     rows, null counts, build commit); rows out must equal data lines in.

No recoding: ``payment_type`` keeps both ``CASH`` and ``Cash`` (lbd_clean.do lines 83-84 merge
them later), and the hashed ``medallion`` and ``hack_license`` stay as 40-character strings.
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
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")
import polars as pl

log = logging.getLogger("haggag2014.convert")
F, I16, I32, S = pl.Float64, pl.Int16, pl.Int32, pl.String
TS = pl.Datetime("us")
SCHEMA = {
    "medallion": S, "hack_license": S, "vendor_name": S, "rate_code": I16,
    "store_and_forward": I16, "trip_pickup_datetime": TS, "trip_dropoff_datetime": TS,
    "passenger_count": I16, "trip_time": I32, "trip_distance": F, "start_lon": F,
    "start_lat": F, "end_lon": F, "end_lat": F, "payment_type": S, "fare_amt": F,
    "surcharge": F, "mta_tax": F, "tip_amt": F, "tolls_amt": F, "total_amt": F,
}
NAME = re.compile(r"^Tripsheet_2009_(\d{1,2})\.csv$")


def cast(c: str, t: pl.DataType) -> pl.Expr:
    x = pl.col(c).str.strip_chars().replace("", None)
    if t == TS:
        return x.str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us", strict=True).alias(c)
    if t == S:
        return pl.col(c)
    return x.cast(t, strict=True).alias(c)


def convert_one(zpath: str, member: str, month: int, tmpdir: str, dest: str, commit: str) -> dict:
    t0 = time.time()
    tmp = Path(tmpdir) / member
    tmp.parent.mkdir(parents=True, exist_ok=True)
    lines = 0
    with zipfile.ZipFile(zpath) as z, z.open(member) as f, open(tmp, "wb") as o:
        crc = z.getinfo(member).CRC
        while block := f.read(1 << 24):
            lines += block.count(b"\n")
            o.write(block)
    try:
        lf = pl.scan_csv(tmp, separator="\t", infer_schema=False, quote_char=None, encoding="utf8-lossy")
        names = lf.collect_schema().names()
        if [n.lower() for n in names] != list(SCHEMA):
            raise ValueError(f"{member}: header differs: {names}")
        lf = lf.rename({n: n.lower() for n in names})
        out = lf.select([cast(c, t) for c, t in SCHEMA.items()]).with_columns(
            source_month=pl.lit(month, pl.Int8))
        d = Path(dest)
        d.parent.mkdir(parents=True, exist_ok=True)
        part = d.with_suffix(".parquet.part")
        out.sink_parquet(part, compression="zstd", statistics=True)
        st = (pl.scan_parquet(part).select(pl.len().alias("rows"), *[pl.col(c).null_count() for c in SCHEMA])
              .collect().row(0, named=True))
        rows = st.pop("rows")
        if rows != lines - 1:
            part.unlink()
            raise ValueError(f"{member}: {rows} rows written, {lines - 1} data lines")
        os.replace(part, d)
    finally:
        tmp.unlink(missing_ok=True)
    side = {"source": f"Raw-Data--FOIL_TPEP_2009-.zip/{member}", "crc32": f"{crc:08x}",
            "build_commit": commit, "rows": rows, "nulls": {c: n for c, n in st.items() if n},
            "seconds": round(time.time() - t0, 1)}
    d.with_suffix(".json").write_text(json.dumps(side, indent=1))
    return side


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zip", default="build/haggag2014/extract/Raw-Data--FOIL_TPEP_2009-.zip")
    ap.add_argument("--out", default="build/haggag2014/tripsheets")
    ap.add_argument("--tmp", default="tmp/haggag2014")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    commit = subprocess.run(["git", "-C", "repo", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    with zipfile.ZipFile(a.zip) as z:
        members = [i for i in z.infolist() if NAME.match(i.filename)]
    todo = []
    for i in members:
        m = int(NAME.match(i.filename).group(1))
        dest = Path(a.out) / f"tripsheet_2009_{m:02d}.parquet"
        side = dest.with_suffix(".json")
        if not a.force and dest.exists() and side.exists():
            try:
                if json.loads(side.read_text()).get("crc32") == f"{i.CRC:08x}":
                    continue
            except (OSError, ValueError):
                pass
        todo.append((a.zip, i.filename, m, a.tmp, str(dest), commit))
    log.info("%d months, %d to convert, %d workers, commit %s", len(members), len(todo), a.workers, commit)
    failed = rows = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(convert_one, *t): t[1] for t in todo}
        for f in as_completed(futs):
            try:
                s = f.result()
                rows += s["rows"]
                log.info("ok %s %d rows %.0fs nulls %s", s["source"], s["rows"], s["seconds"], s["nulls"])
            except Exception as e:
                failed += 1
                log.error("FAILED %s: %s", futs[f], e)
    log.info("finished: %d converted, %d failed, %.1fM rows", len(todo) - failed, failed, rows / 1e6)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
