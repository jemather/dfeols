# SPDX-License-Identifier: GPL-3.0-or-later
"""D6 stage 1: convert EVE Ref market-order snapshots (``market-orders-*.v3.csv.bz2``) to
Parquet, one file per snapshot, losslessly.

    python repo/replication/eve/convert_orders.py [--raw raw/eve] [--out tmp/eve/stage]
        [--from 2023-06-18] [--to 2023-11-11] [--workers 8] [--force]

Run from the project root. For each snapshot:

  1. read the .bz2 bytes and check their SHA-256 against the torrent's
     ``everef-eve-online-deep-archive-NN.sha256``;
  2. decompress in memory and require the v3 header exactly (HEADER below);
  3. cast to SCHEMA strictly (anything that doesn't parse fails the file), add ``snap_time``
     (UTC, from the file name);
  4. write ``<out>/market_orders/<YYYY>/<YYYY-MM-DD>/<stem>.parquet`` atomically, then a
     ``.json`` sidecar (source path and hash, rows, distinct orders, null counts, build commit).

A file whose sidecar matches its source hash is skipped, so a rerun resumes. Failures are
logged and the run carries on; the exit code is 1 if any file failed. No code from the
archive is run; files are read as data.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")
import polars as pl  # noqa: E402

log = logging.getLogger("eve.convert_orders")

HEADER = [
    "duration", "is_buy_order", "issued", "location_id", "min_volume", "order_id", "price",
    "range", "system_id", "type_id", "volume_remain", "volume_total", "http_last_modified",
    "region_id", "station_id", "constellation_id",
]
UTC = pl.Datetime("us", "UTC")
SCHEMA = {
    "duration": pl.Int32, "is_buy_order": pl.Boolean, "issued": UTC, "location_id": pl.Int64,
    "min_volume": pl.Int64, "order_id": pl.Int64, "price": pl.Float64, "range": pl.String,
    "system_id": pl.Int32, "type_id": pl.Int32, "volume_remain": pl.Int64,
    "volume_total": pl.Int64, "http_last_modified": UTC, "region_id": pl.Int32,
    "station_id": pl.Int64, "constellation_id": pl.Int32,
}
NAME = re.compile(r"market-orders-(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})\.v3\.csv\.bz2$")


def load_sums(raw: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    for f in sorted(raw.glob("everef-eve-online-deep-archive-*.sha256")):
        for line in f.read_text().splitlines():
            digest, _, name = line.partition("  ")
            if name.startswith("market-orders/"):
                sums[name] = digest
    return sums


def snap_time(name: str) -> datetime:
    m = NAME.search(name)
    if not m:
        raise ValueError(f"unexpected file name {name}")
    d, hh, mm, ss = m.groups()
    return datetime.fromisoformat(f"{d}T{hh}:{mm}:{ss}").replace(tzinfo=timezone.utc)


def parse_ts(col: str) -> pl.Expr:
    return pl.col(col).str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_unit="us", time_zone="UTC", strict=True)


def convert_one(src: str, rel: str, digest: str, out: str, commit: str) -> dict:
    t0 = time.time()
    data = Path(src).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != digest:
        raise ValueError(f"sha256 mismatch for {rel}: {got} != {digest}")
    text = bz2.decompress(data)
    del data
    header = text[: text.index(b"\n")].decode().strip().split(",")
    if header != HEADER:
        raise ValueError(f"header differs in {rel}: {header}")
    df = pl.read_csv(io.BytesIO(text), infer_schema=False)
    del text
    df = df.with_columns(
        parse_ts("issued"), parse_ts("http_last_modified"),
        pl.col("is_buy_order").replace_strict({"true": True, "false": False}, return_dtype=pl.Boolean),
        *[pl.col(c).cast(t, strict=True) for c, t in SCHEMA.items()
          if c not in ("issued", "http_last_modified", "is_buy_order", "range")],
    ).with_columns(snap_time=pl.lit(snap_time(rel)).cast(UTC))
    rows = df.height
    if rows == 0:
        raise ValueError(f"{rel} has no rows")
    stats = {
        "rows": rows,
        "orders": df["order_id"].n_unique(),
        "nulls": {c: n for c, n in df.null_count().row(0, named=True).items() if n},
    }
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.part")
    df.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, dest)
    side = {"source": rel, "sha256": digest, "build_commit": commit,
            "seconds": round(time.time() - t0, 2), **stats}
    dest.with_suffix(".json").write_text(json.dumps(side, indent=1))
    return side


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", default="raw/eve")
    ap.add_argument("--out", default="tmp/eve/stage")
    ap.add_argument("--from", dest="start", default=None, help="first date, YYYY-MM-DD")
    ap.add_argument("--to", dest="end", default=None, help="last date, YYYY-MM-DD")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    raw, out = Path(a.raw), Path(a.out) / "market_orders"
    sums = load_sums(raw)
    commit = subprocess.run(["git", "-C", "repo", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    todo = []
    for rel in sorted(sums):
        m = NAME.search(rel)
        if not m or (a.start and m.group(1) < a.start) or (a.end and m.group(1) > a.end):
            continue
        d = m.group(1)
        dest = out / d[:4] / d / Path(rel).name.replace(".v3.csv.bz2", ".parquet")
        side = dest.with_suffix(".json")
        if not a.force and dest.exists() and side.exists():
            try:
                if json.loads(side.read_text()).get("sha256") == sums[rel]:
                    continue
            except (OSError, ValueError):
                pass
        todo.append((str(raw / rel), rel, sums[rel], str(dest), commit))
    log.info("%d snapshots listed, %d to convert, %d workers, commit %s",
             sum(1 for r in sums if NAME.search(r)), len(todo), a.workers, commit)

    failed = done = rows = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(convert_one, *t): t[1] for t in todo}
        for f in as_completed(futs):
            rel = futs[f]
            try:
                s = f.result()
                done += 1
                rows += s["rows"]
                if done % 50 == 0 or done == len(todo):
                    el = time.time() - t0
                    log.info("%d/%d done, %.2fB rows, %.0f s, eta %.0f min", done, len(todo),
                             rows / 1e9, el, el / done * (len(todo) - done) / 60)
            except Exception as e:  # noqa: BLE001 - log and carry on
                failed += 1
                log.error("FAILED %s: %s", rel, e)
    log.info("finished: %d converted, %d failed, %.3fB rows", done, failed, rows / 1e9)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
