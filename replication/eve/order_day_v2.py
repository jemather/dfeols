# SPDX-License-Identifier: GPL-3.0-or-later
"""D6 v2: one row per order per day from the Fuzzwork ordersets, straight from raw.

    python repo/replication/eve/order_day_v2.py [--raw raw/eve] [--out tmp/eve/build/v2]
        [--from 2017-09-25] [--to 2026-02-14] [--workers 6] [--pass 1|2|both] [--force]
        [--pass1 <dir>] [--bucket-rows 60000000]

Run from the project root. The rules are v1's (Decisions 2026-11-08, ``order_day.py``); what
differs is the source (Fuzzwork, ``fuzzwork.py``; Decisions 2026-11-09) and the scale:

Pass 1, per UTC day (``<out>/pass1/<YYYY>/<day>.parquet`` + ``.json`` sidecar; resumable): the
day's ordersets in id order, each hash-checked and parsed strictly. Exact duplicate rows within an
orderset are dropped (counted); an orderset whose rows equal the previous one's is dropped as a
copy (744 same-time pairs in 00-13). Then, as v1, one row per order seen that day holding its
latest sighting's values plus the day's summaries (n_snap, day_snaps, first/last_seen,
first/last_seq, price_first/min/max/mean, n_price, n_price_changes, volume_first, n_issued). The
sidecar also records each region's sightings that day (``regions``: snapshots present, orders).
A day is skipped when its sidecar lists the same ordersets, so new torrents add days on resume.

Pass 2, over the whole window (``<out>/order_day/<YYYY>/<day>.parquet``, ``<out>/regions.parquet``,
``<out>/_summary.json``): v1's per-order spans and exit coding, computed in buckets of order_id
(each bucket reads only the days whose order_id range overlaps it), then joined back by day.
Adds ``region_covered_next``: the order's region appears in at least one snapshot of the next
day (null on the window's last day). Fuzzwork's coverage changes (about 30 null-sec regions end
in 2018-06; a few small ones come and go), and an exit whose region is not covered the next day
may be the coverage, not the order; how to treat those is Ted's decision, so it is a flag here.

Fuzzwork lacks EVE Ref's system_id, constellation_id and http_last_modified. Nothing from the
archive is run; files are read as data.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuzzwork import inventory, read_orderset

log = logging.getLogger("eve.order_day_v2")
KEEP = ["order_id", "type_id", "region_id", "location_id", "is_buy_order", "range", "duration",
        "min_volume", "volume_total", "volume_remain", "price", "issued"]
CONTENT = ["order_id", "price", "volume_remain", "issued"]  # what makes two ordersets copies
EXPIRY_SLACK = pl.duration(minutes=35)
BUCKET_ROWS = 60_000_000


def commit_id() -> str:
    return subprocess.run(["git", "-C", "repo", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "unknown"


# ---------------------------------------------------------------- pass 1

def collapse_day(d: str, sets: list[dict], dest: str, scratch: str, commit: str) -> dict:
    t0 = time.time()
    sdir = Path(scratch) / d
    shutil.rmtree(sdir, ignore_errors=True)
    sdir.mkdir(parents=True)
    files, used, copies = [], [], []
    dup_rows = dup_conflicts = 0
    prev = None  # (snap_time, frame) of the last orderset kept
    reg_snaps: dict[int, int] = {}
    try:
        for s in sets:  # id order
            df = read_orderset(s["path"], s["sha256"], s["orderset_id"])
            if df["order_id"].is_duplicated().any():
                n0 = df.height
                df = df.unique(maintain_order=True)
                dup_rows += n0 - df.height
                n1 = df.height
                df = df.unique("order_id", keep="first", maintain_order=True)
                dup_conflicts += n1 - df.height
            # copies share a time with the previous orderset (744 pairs in 00-13, to the
            # second); only then compare contents
            if prev is not None and abs((s["snap_time"] - prev[0]).total_seconds()) <= 60:
                content = lambda x: int(x.sort("order_id").select(CONTENT).hash_rows(seed=7).sum())
                if df.height == prev[1].height and content(df) == content(prev[1]):
                    copies.append(s["orderset_id"])
                    continue
            prev = (s["snap_time"], df)
            for r in df["region_id"].unique().to_list():
                reg_snaps[r] = reg_snaps.get(r, 0) + 1
            f = sdir / f"{s['orderset_id']}.parquet"
            df.select(KEEP).with_columns(snap_seq=pl.lit(s["orderset_id"], pl.Int32),
                                         snap_time=pl.lit(s["snap_time"]).cast(pl.Datetime("us", "UTC"))
                                         ).write_parquet(f, compression="lz4")
            files.append(str(f))
            used.append(s["orderset_id"])
        lf = pl.scan_parquet(files)
        by_q = lambda c: pl.col(c).sort_by("snap_seq")
        o = (lf.group_by("order_id")
             .agg(*[by_q(c).last() for c in KEEP if c != "order_id"],
                  n_snap=pl.len().cast(pl.Int32),
                  first_seen=by_q("snap_time").first(), last_seen=by_q("snap_time").last(),
                  first_seq=pl.col("snap_seq").min(), last_seq=pl.col("snap_seq").max(),
                  price_first=by_q("price").first(), price_min=pl.col("price").min(),
                  price_max=pl.col("price").max(), price_mean=pl.col("price").mean(),
                  n_price=pl.col("price").n_unique().cast(pl.Int32),
                  n_price_changes=(by_q("price").diff().fill_null(0) != 0).sum().cast(pl.Int32),
                  volume_first=by_q("volume_remain").first(),
                  n_issued=pl.col("issued").n_unique().cast(pl.Int32))
             .with_columns(day=pl.lit(date.fromisoformat(d)), day_snaps=pl.lit(len(used), pl.Int32))
             .sort("order_id"))
        rows_in = lf.select(pl.len()).collect().item()
        out = o.collect(engine="streaming")
    finally:
        shutil.rmtree(sdir, ignore_errors=True)
    if out["n_snap"].sum() != rows_in:
        raise ValueError(f"{d}: n_snap sums to {out['n_snap'].sum()}, snapshots hold {rows_in} rows")
    reg = out.group_by("region_id").agg(orders=pl.len()).sort("region_id")
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".parquet.part")
    out.write_parquet(tmp, compression="zstd", statistics=True, row_group_size=256_000)
    os.replace(tmp, p)
    q = out["order_id"].quantile
    side = {"day": d, "ordersets": [s["orderset_id"] for s in sets], "used": used, "copies": copies,
            "day_snaps": len(used), "dup_rows": dup_rows, "dup_conflicts": dup_conflicts,
            "rows_in": rows_in, "orders": out.height,
            "order_id_min": int(out["order_id"].min()), "order_id_max": int(out["order_id"].max()),
            "order_id_q": [int(q(i / 20, "nearest")) for i in range(21)],
            "regions": [{"region_id": r, "snaps": reg_snaps[r], "orders": n}
                        for r, n in reg.iter_rows()],
            "build_commit": commit, "seconds": round(time.time() - t0, 1)}
    p.with_suffix(".json").write_text(json.dumps(side, indent=1))
    return side


def pass1(inv: pl.DataFrame, days: list[str], p1: Path, scratch: Path, workers: int, force: bool,
          commit: str) -> int:
    todo = []
    by_day = {d.isoformat(): g.sort("orderset_id").to_dicts()
              for (d,), g in inv.group_by("day")}
    for d in days:
        sets = by_day[d]
        dest = p1 / d[:4] / f"{d}.parquet"
        side = dest.with_suffix(".json")
        if not force and dest.exists() and side.exists():
            try:
                if json.loads(side.read_text())["ordersets"] == [s["orderset_id"] for s in sets]:
                    continue
            except (OSError, ValueError, KeyError):
                pass
        todo.append((d, sets, str(dest), str(scratch), commit))
    log.info("pass 1: %d days, %d to collapse, %d workers, commit %s", len(days), len(todo),
             workers, commit)
    failed = done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(collapse_day, *t): t[0] for t in todo}
        for f in as_completed(futs):
            try:
                s = f.result()
                done += 1
                el = time.time() - t0
                log.info("day %s: %d snapshots (%d copies), %d rows -> %d orders, %d dup rows, "
                         "%.0f s; %d/%d, eta %.0f min", s["day"], s["day_snaps"], len(s["copies"]),
                         s["rows_in"], s["orders"], s["dup_rows"], s["seconds"], done, len(todo),
                         el / done * (len(todo) - done) / 60)
            except Exception as e:
                failed += 1
                log.error("FAILED %s: %s", futs[f], e)
    return failed


# ---------------------------------------------------------------- pass 2

def buckets(sides: list[dict], rows: int = BUCKET_ROWS) -> list[tuple[int, int]]:
    """order_id ranges holding about BUCKET_ROWS order-days each, from the days' quantiles."""
    pts = []
    for s in sides:
        q, w = s["order_id_q"], s["orders"] / (len(s["order_id_q"]) - 1)
        pts += [(q[i], w) for i in range(1, len(q))]
    pts.sort()
    total = sum(w for _, w in pts)
    n = max(1, math.ceil(total / rows))
    cuts, acc, k = [], 0.0, 1
    for v, w in pts:
        acc += w
        if k < n and acc >= total * k / n:
            cuts.append(v + 1)
            k += 1
    lo = min(s["order_id_min"] for s in sides)
    hi = max(s["order_id_max"] for s in sides) + 1
    edges = [lo, *sorted(set(c for c in cuts if lo < c < hi)), hi]
    return list(zip(edges[:-1], edges[1:]))


def finish_day(d: str, src: str, kdir: str, regions: str, d0: date, dn: date, fin: str) -> dict:
    """Pass 2 for one day: join the day's keys and the next day's region coverage."""
    day = date.fromisoformat(d)
    keys = pl.scan_parquet(f"{kdir}/*.parquet").filter(pl.col("day") == day).drop("day")
    covered = (pl.scan_parquet(regions).filter(pl.col("day") == day + timedelta(1))
               .select("region_id", region_covered_next=pl.lit(True)))
    x = (pl.scan_parquet(src)
         .join(keys, on="order_id", how="left")
         .join(covered, on="region_id", how="left")
         .with_columns(
             day_index=pl.lit((day - d0).days, pl.Int32),
             day_gap=(pl.col("day") - pl.col("prev_day")).dt.total_days().cast(pl.Int32),
             edited_since_prev=(pl.col("issued") != pl.col("prev_issued")),
             price_change_prev=pl.col("price") - pl.col("prev_price"),
             new_order=(pl.col("day") == pl.col("first_day")) & ~pl.col("left_censored"),
             right_censored=pl.col("last_day") == dn,
             region_covered_next=(pl.col("region_covered_next").fill_null(False)
                                  if day < dn else pl.lit(None, pl.Boolean)))
         .with_columns(
             exit=((pl.col("day") == pl.col("last_day")) & ~pl.col("right_censored")).cast(pl.Int8))
         .with_columns(exit_type=pl.when(pl.col("exit") == 0).then(None)
                       .when(pl.col("issued") + pl.duration(days=1) * pl.col("duration")
                             <= pl.col("last_seen") + EXPIRY_SLACK).then(pl.lit("expired"))
                       .when(pl.col("volume_remain") < pl.col("volume_total"))
                       .then(pl.lit("partly_filled"))
                       .otherwise(pl.lit("vanished")))
         .drop("prev_day", "prev_issued", "prev_price")
         .collect(engine="streaming")
         .sort("order_id"))  # the streaming join doesn't keep row order
    if x["first_day"].null_count():
        raise ValueError(f"{d}: {x['first_day'].null_count()} order-days without keys")
    p = Path(fin) / d[:4] / f"{d}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".parquet.part")
    x.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, p)
    return {"order_days": x.height, "exits": int(x["exit"].sum()),
            "exit_types": dict(x["exit_type"].drop_nulls().value_counts().iter_rows())}


def pass2(p1: Path, out: Path, days: list[str], workers: int, commit: str,
          bucket_rows: int = BUCKET_ROWS) -> dict:
    t0 = time.time()
    sides = [json.loads((p1 / d[:4] / f"{d}.json").read_text()) for d in days]
    first_seq = min(sides[0]["used"])
    d0, dn = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
    path = lambda d: str(p1 / d[:4] / f"{d}.parquet")
    regions = pl.DataFrame([{"day": date.fromisoformat(s["day"]), **r, "day_snaps": s["day_snaps"]}
                            for s in sides for r in s["regions"]],
                           schema={"day": pl.Date, "region_id": pl.Int32, "snaps": pl.Int32,
                                   "orders": pl.Int64, "day_snaps": pl.Int32})
    out.mkdir(parents=True, exist_ok=True)
    regions.write_parquet(out / "regions.parquet")
    kdir = out / "_keys"
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir()
    bks = buckets(sides, bucket_rows)
    log.info("pass 2: %d days, %d order-days, %d buckets", len(days),
             sum(s["orders"] for s in sides), len(bks))
    for i, (lo, hi) in enumerate(bks):
        src = [path(s["day"]) for s in sides if s["order_id_max"] >= lo and s["order_id_min"] < hi]
        (pl.scan_parquet(src)
         .filter(pl.col("order_id").is_between(lo, hi, closed="left"))
         .select("order_id", "day", "issued", "price", "first_seq")
         .sort("order_id", "day")
         .with_columns(
             prev_day=pl.col("day").shift(1).over("order_id"),
             prev_issued=pl.col("issued").shift(1).over("order_id"),
             prev_price=pl.col("price").shift(1).over("order_id"),
             first_day=pl.col("day").min().over("order_id"),
             last_day=pl.col("day").max().over("order_id"),
             n_days=pl.len().over("order_id").cast(pl.Int32),
             left_censored=(pl.col("first_seq").min().over("order_id") <= first_seq))
         .drop("issued", "price", "first_seq")
         .sort("day", "order_id")
         .sink_parquet(kdir / f"b{i:04d}.parquet", compression="zstd", row_group_size=200_000))
        log.info("bucket %d/%d [%d, %d): %d days read, %.0f s", i + 1, len(bks), lo, hi, len(src),
                 time.time() - t0)
    total = exits = 0
    types: dict[str, int] = {}
    args = [(d, path(d), str(kdir), str(out / "regions.parquet"), d0, dn, str(out / "order_day"))
            for d in days]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for i, r in enumerate(ex.map(finish_day, *zip(*args), chunksize=4)):
            total += r["order_days"]
            exits += r["exits"]
            for t, n in r["exit_types"].items():
                types[t] = types.get(t, 0) + n
            if (i + 1) % 100 == 0 or i + 1 == len(days):
                log.info("pass 2 days: %d/%d, %.0f s", i + 1, len(days), time.time() - t0)
    shutil.rmtree(kdir)
    summary = {"days": len(days), "first_day": days[0], "last_day": days[-1],
               "ordersets": sum(len(s["ordersets"]) for s in sides),
               "snapshots_used": sum(s["day_snaps"] for s in sides),
               "copies_dropped": sum(len(s["copies"]) for s in sides),
               "dup_rows": sum(s["dup_rows"] for s in sides),
               "dup_conflicts": sum(s["dup_conflicts"] for s in sides),
               "order_days": total, "exits": exits, "exit_types": types, "buckets": len(bks),
               "build_commit": commit, "seconds": round(time.time() - t0, 1)}
    (out / "_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", default="raw/eve")
    ap.add_argument("--out", default="tmp/eve/build/v2")
    ap.add_argument("--pass1", default=None, help="pass 1 folder (default <out>/pass1)")
    ap.add_argument("--scratch", default="tmp/eve/spill/v2_pass1")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", dest="end", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pass", dest="which", choices=["1", "2", "both"], default="both")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bucket-rows", type=int, default=BUCKET_ROWS, help="order-days per pass-2 bucket")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out, commit = Path(a.out), commit_id()
    p1 = Path(a.pass1) if a.pass1 else out / "pass1"
    inv = inventory(Path(a.raw))
    days = sorted(d.isoformat() for d in inv["day"].unique()
                  if (not a.start or d.isoformat() >= a.start) and (not a.end or d.isoformat() <= a.end))
    if not days:
        log.error("no days in range")
        return 1
    want = [str(date.fromisoformat(days[0]) + timedelta(i))
            for i in range((date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days + 1)]
    if days != want:  # a missing day would look like exits
        log.error("days are not contiguous; missing %s", sorted(set(want) - set(days)))
        return 1
    log.info("inventory: %d ordersets in %d torrents; window %s..%s", inv.height,
             inv["torrent"].n_unique(), days[0], days[-1])
    if a.which in ("1", "both"):
        if pass1(inv, days, p1, Path(a.scratch), a.workers, a.force, commit):
            log.error("pass 1 had failures; pass 2 not run")
            return 1
    if a.which in ("2", "both"):
        s = pass2(p1, out, days, a.workers, commit, a.bucket_rows)
        log.info("pass 2: %d order-days, %d exits %s, %.0f s", s["order_days"], s["exits"],
                 s["exit_types"], s["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
