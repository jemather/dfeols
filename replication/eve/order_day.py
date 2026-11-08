# SPDX-License-Identifier: GPL-3.0-or-later
"""D6 stage 2: collapse the staged market-order snapshots to one row per order per day.

    python repo/replication/eve/order_day.py [--stage tmp/eve/stage] [--out tmp/eve/build]
        [--from 2023-06-18] [--to 2023-11-11] [--workers 4] [--force]

Run from the project root, after ``convert_orders.py``. Rules (Ted, Decisions 2026-11-08):

Pass 1, per UTC day (``<out>/order_day_raw/<day>.parquet``): every order seen in any of the
day's snapshots gets one row, holding the values of its **latest** snapshot that day, plus
summaries of what changed within the day:

  n_snap / day_snaps       snapshots the order was seen in / snapshots that day (all regions)
  first_seen, last_seen    first and last snapshot time it was seen that day
  price_first/min/max/mean, n_price, n_price_changes   (changes between consecutive sightings)
  volume_first             volume_remain at the first sighting (volume_remain is the latest)
  n_issued                 distinct ``issued`` values (an edit resets ``issued``)

Pass 2, over all days (``<out>/order_day/<day>.parquet``): per order, its first and last day
in the window, and for each order-day:

  day_index, first_day, last_day, n_days, day_gap   (days since its previous order-day; 1 = none missed)
  edited_since_prev        ``issued`` differs from the previous order-day's latest ``issued``
  price_change_prev        price minus the previous order-day's latest price
  left_censored            order seen in the window's first snapshot (open before the window)
  new_order                first order-day of an order not open at the window's start
  exit                     1 on the order's last day, unless right-censored; 0 otherwise
  exit_type                null if no exit; else "expired" (issued + duration within 35 minutes
                           after the last sighting, or already past), "partly_filled"
                           (volume_remain < volume_total at the last sighting), "vanished"
                           (neither: filled at once or cancelled; the snapshots can't tell)
  right_censored           last seen on the window's last day

An order missing from a snapshot (a region fetch failure) or from a whole day is not an exit;
only its final disappearance is. A day whose output exists with a matching sidecar is skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")
import polars as pl

log = logging.getLogger("eve.order_day")
KEEP = ["order_id", "type_id", "region_id", "constellation_id", "system_id", "location_id",
        "is_buy_order", "range", "duration", "min_volume", "volume_total", "volume_remain",
        "price", "issued", "http_last_modified"]
EXPIRY_SLACK = pl.duration(minutes=35)


def day_files(stage: Path, d: str) -> list[str]:
    return sorted(str(p) for p in (stage / "market_orders" / d[:4] / d).glob("*.parquet"))


def collapse_day(files: list[str], d: str, dest: str, commit: str) -> dict:
    t0 = time.time()
    lf = pl.scan_parquet(files).select(*KEEP, "snap_time")
    day_snaps = lf.select(pl.col("snap_time").n_unique()).collect().item()
    s = pl.col("snap_time")
    by_t = lambda c: pl.col(c).sort_by("snap_time")
    o = (lf.group_by("order_id")
         .agg(*[by_t(c).last() for c in KEEP if c != "order_id"],
              n_snap=pl.len(), first_seen=s.min(), last_seen=s.max(),
              price_first=by_t("price").first(), price_min=pl.col("price").min(),
              price_max=pl.col("price").max(), price_mean=pl.col("price").mean(),
              n_price=pl.col("price").n_unique(),
              n_price_changes=(by_t("price").diff().fill_null(0) != 0).sum().cast(pl.Int32),
              volume_first=by_t("volume_remain").first(),
              n_issued=pl.col("issued").n_unique().cast(pl.Int32))
         .with_columns(day=pl.lit(date.fromisoformat(d)), day_snaps=pl.lit(day_snaps, pl.Int32),
                       n_snap=pl.col("n_snap").cast(pl.Int32), n_price=pl.col("n_price").cast(pl.Int32))
         .sort("order_id"))
    rows_in = lf.select(pl.len()).collect().item()
    out = o.collect(engine="streaming")
    if out["n_snap"].sum() != rows_in:
        raise ValueError(f"{d}: n_snap sums to {out['n_snap'].sum()}, snapshots hold {rows_in} rows")
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".parquet.part")
    out.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, p)
    side = {"day": d, "files": len(files), "day_snaps": day_snaps, "rows_in": rows_in,
            "orders": out.height, "build_commit": commit, "seconds": round(time.time() - t0, 1)}
    p.with_suffix(".json").write_text(json.dumps(side, indent=1))
    return side


def finish(raw_dir: Path, out_dir: Path, days: list[str], first_snap, commit: str) -> dict:
    """Pass 2: per-order spans and exit coding, written back one day per file."""
    t0 = time.time()
    lf = pl.scan_parquet([str(raw_dir / f"{d}.parquet") for d in days])
    last_day = date.fromisoformat(days[-1])
    # one small row per order-day: what pass 2 needs from the neighbours
    k = (lf.select("order_id", "day", "issued", "price", "first_seen")
         .sort("order_id", "day")
         .with_columns(
             prev_day=pl.col("day").shift(1).over("order_id"),
             prev_issued=pl.col("issued").shift(1).over("order_id"),
             prev_price=pl.col("price").shift(1).over("order_id"),
             first_day=pl.col("day").min().over("order_id"),
             last_day=pl.col("day").max().over("order_id"),
             n_days=pl.len().over("order_id").cast(pl.Int32),
             left_censored=(pl.col("first_seen").min().over("order_id") <= first_snap))
         .drop("issued", "price", "first_seen")
         .sort("day", "order_id"))  # by day, so each day's read prunes row groups
    k_path = out_dir / "_keys.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    k.sink_parquet(k_path, compression="zstd")
    keys = pl.scan_parquet(k_path)
    total = exits = 0
    types: dict[str, int] = {}
    for d in days:
        day = date.fromisoformat(d)
        x = (pl.scan_parquet(str(raw_dir / f"{d}.parquet"))
             .join(keys.filter(pl.col("day") == day).drop("day"), on="order_id", how="left")
             .with_columns(
                 day_index=pl.lit((day - date.fromisoformat(days[0])).days, pl.Int32),
                 day_gap=(pl.col("day") - pl.col("prev_day")).dt.total_days().cast(pl.Int32),
                 edited_since_prev=(pl.col("issued") != pl.col("prev_issued")),
                 price_change_prev=pl.col("price") - pl.col("prev_price"),
                 new_order=(pl.col("day") == pl.col("first_day")) & ~pl.col("left_censored"),
                 right_censored=pl.col("last_day") == last_day)
             .with_columns(
                 exit=((pl.col("day") == pl.col("last_day")) & ~pl.col("right_censored")).cast(pl.Int8))
             .with_columns(exit_type=pl.when(pl.col("exit") == 0).then(None)
                           .when(pl.col("issued") + pl.duration(days=1) * pl.col("duration")
                                 <= pl.col("last_seen") + EXPIRY_SLACK).then(pl.lit("expired"))
                           .when(pl.col("volume_remain") < pl.col("volume_total"))
                           .then(pl.lit("partly_filled"))
                           .otherwise(pl.lit("vanished")))
             .drop("prev_day", "prev_issued", "prev_price")
             .collect(engine="streaming"))
        p = out_dir / f"{d}.parquet"
        tmp = p.with_suffix(".parquet.part")
        x.write_parquet(tmp, compression="zstd", statistics=True)
        os.replace(tmp, p)
        total += x.height
        exits += int(x["exit"].sum())
        for t, n in x["exit_type"].drop_nulls().value_counts().iter_rows():
            types[t] = types.get(t, 0) + n
    k_path.unlink()
    summary = {"days": len(days), "first_day": days[0], "last_day": days[-1], "order_days": total,
               "exits": exits, "exit_types": types, "build_commit": commit,
               "seconds": round(time.time() - t0, 1)}
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="tmp/eve/stage")
    ap.add_argument("--out", default="tmp/eve/build")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", dest="end", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    stage, out = Path(a.stage), Path(a.out)
    raw_dir, fin_dir = out / "order_day_raw", out / "order_day"
    commit = subprocess.run(["git", "-C", "repo", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    days = sorted(p.name for p in (stage / "market_orders").glob("*/*") if p.is_dir()
                  and (not a.start or p.name >= a.start) and (not a.end or p.name <= a.end))
    if not days:
        log.error("no staged days found")
        return 1
    # the window must be contiguous: a missing day would look like exits
    want = [str(date.fromisoformat(days[0]) + timedelta(i))
            for i in range((date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days + 1)]
    if days != want:
        log.error("staged days are not contiguous; missing %s", sorted(set(want) - set(days)))
        return 1
    todo = []
    for d in days:
        files = day_files(stage, d)
        dest = raw_dir / f"{d}.parquet"
        side = dest.with_suffix(".json")
        if not a.force and dest.exists() and side.exists():
            try:
                if json.loads(side.read_text()).get("files") == len(files):
                    continue
            except (OSError, ValueError):
                pass
        todo.append((files, d, str(dest), commit))
    log.info("%d days %s..%s, %d to collapse, %d workers, commit %s",
             len(days), days[0], days[-1], len(todo), a.workers, commit)
    failed = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(collapse_day, *t): t[1] for t in todo}
        for f in as_completed(futs):
            try:
                s = f.result()
                log.info("day %s: %d snapshots, %d rows -> %d orders, %.0f s (%.0f s total)",
                         s["day"], s["day_snaps"], s["rows_in"], s["orders"], s["seconds"],
                         time.time() - t0)
            except Exception as e:
                failed += 1
                log.error("FAILED %s: %s", futs[f], e)
    if failed:
        log.error("%d days failed; pass 2 not run", failed)
        return 1
    first_snap = (pl.scan_parquet(day_files(stage, days[0])).select(pl.col("snap_time").min())
                  .collect().item())
    s = finish(raw_dir, fin_dir, days, first_snap, commit)
    log.info("pass 2: %d order-days, %d exits %s, %.0f s", s["order_days"], s["exits"],
             s["exit_types"], s["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
