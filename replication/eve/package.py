# SPDX-License-Identifier: GPL-3.0-or-later
"""D6: package the EVE order-day panel (torrent 09) for OSCER as ``outbox/eve_order_day_v1/``.

    python repo/replication/eve/package.py [--src tmp/eve/build/order_day] [--out outbox] [--parts 6]

Adds model columns to the order-day build (``order_day.py``) and writes data/ (by ``type_id``),
README.md and reference.json (pyfixest on one week). The specs are a scale test, not a model
anyone has settled on (Ted, 2026-11-08: "just try something"). Manifest afterwards (CLAUDE.md).
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

NAME = "eve_order_day_v1"
X = "log_rel_price + log_volume_remain + log_age_days + n_price_changes + is_buy"
FE = "type_id^region_id^day_index + location_id"
SPECS = [
    ("any exit", f"exit ~ {X} | {FE}"),
    ("vanished (fill or cancel)", f"exit_vanished ~ {X} | {FE}"),
    ("expired", f"exit_expired ~ {X} | {FE}"),
]


def frame(lf: pl.LazyFrame) -> pl.LazyFrame:
    lp = pl.col("price").log()
    return lf.with_columns(
        is_buy=pl.col("is_buy_order").cast(pl.Int8),
        log_rel_price=lp - lp.mean().over("type_id", "region_id", "day", "is_buy_order"),
        log_volume_remain=pl.when(pl.col("volume_remain") > 0).then(pl.col("volume_remain").log()),
        age_days=(pl.col("day") - pl.col("issued").dt.date()).dt.total_days().cast(pl.Int32),
        exit_expired=(pl.col("exit_type") == "expired").fill_null(False).cast(pl.Int8),
        exit_partly_filled=(pl.col("exit_type") == "partly_filled").fill_null(False).cast(pl.Int8),
        exit_vanished=(pl.col("exit_type") == "vanished").fill_null(False).cast(pl.Int8),
        edited_since_prev=pl.col("edited_since_prev").fill_null(False),
    ).with_columns(log_age_days=(pl.col("age_days").clip(lower_bound=0) + 1).log())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="tmp/eve/build/order_day")
    ap.add_argument("--out", default="outbox")
    ap.add_argument("--parts", type=int, default=6)
    a = ap.parse_args(argv)
    commit = (
        subprocess.run(
            ["git", "-C", "repo", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )
    d = Path(a.out) / NAME
    # the relative price is a mean within type_id groups, so each part can compute its own
    parts = write_parts(pl.scan_parquet(f"{a.src}/*.parquet"), "type_id", d, a.parts, transform=frame)
    data = pl.scan_parquet(str(d / "data" / "*.parquet"))
    rows = sum(p["rows"] for p in parts)
    levels = level_counts(
        data,
        {
            "order_id": ["order_id"],
            "type_id": ["type_id"],
            "region_id": ["region_id"],
            "location_id": ["location_id"],
            "day_index": ["day_index"],
            "type_id^region_id": ["type_id", "region_id"],
            "type_id^region_id^day_index": ["type_id", "region_id", "day_index"],
        },
    )
    stats = (
        data.select(
            pl.col("exit").mean(),
            pl.col("exit_vanished").mean(),
            pl.col("exit_partly_filled").mean(),
            pl.col("exit_expired").mean(),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    sub = data.filter(pl.col("day").is_between(date(2023, 7, 10), date(2023, 7, 16)))
    cols = [
        "exit",
        "exit_vanished",
        "exit_expired",
        "log_rel_price",
        "log_volume_remain",
        "log_age_days",
        "n_price_changes",
        "is_buy",
        "type_id",
        "region_id",
        "day_index",
        "location_id",
    ]
    pdf = sub.select(cols).drop_nulls().collect().to_pandas()
    fits = []
    for _, f in SPECS:
        r = fit_reference(pdf, f, "type_id")
        r["subsample"] = "day 2023-07-10..2023-07-16, nulls dropped"
        fits.append(r)
    dump_json(
        d / "reference.json",
        {
            "dataset": NAME,
            "build": f"jemather/dfeols {commit}",
            "tool": "pyfixest (in memory, valiant)",
            "rows": rows,
            "fe_levels": levels,
            "fits": fits,
            "parts": parts,
        },
    )
    notes = {
        "order_id": "EVE market order; one row per order per UTC day it was seen",
        "day": "UTC day",
        "day_index": "days since 2023-06-18",
        "n_snap": "snapshots the order was seen in that day",
        "day_snaps": "snapshots that day (all regions)",
        "exit": "1 on the order's last day seen, unless right-censored",
        "exit_type": "expired (issued + duration within 35 min of last sighting), partly_filled "
        "(volume_remain < volume_total), vanished (filled at once or cancelled)",
        "left_censored": "open at the window's first snapshot",
        "right_censored": "seen on the window's last day",
        "new_order": "first order-day of an order placed in the window",
        "log_rel_price": "log price minus its type × region × day × side mean",
        "age_days": "day minus the date of `issued` (an edit resets issued)",
        "price": "latest snapshot of the day; price_first/min/max/mean summarise the day",
    }
    fit_rows = "\n".join(
        f"| {k} | `{f.replace('|', chr(92) + '|')}` | {r['nobs']:,} | "
        + ", ".join(f"{t} {v[0]:.6f} ({v[1]:.6f})" for t, v in r["coef"].items())
        + " |"
        for (k, f), r in zip(SPECS, fits)
    )
    lv = "\n".join(f"| {k} | {v:,} |" for k, v in levels.items())
    pt = "\n".join(
        f"| {p['file']} | {p['rows']:,} | {p['type_id_min']}-{p['type_id_max']} | {p['bytes'] / 1e9:.2f} GB |"
        for p in parts
    )
    (d / "README.md").write_text(f"""# {NAME}

EVE Online market orders, one row per order per day, 2023-06-18 to 2023-11-11 (147 days), built
from EVE Ref's 30-minute market snapshots (Deep Archive torrent 09), for a discrete-time hazard
LPM of order exit. A scale test only, not for the paper (Decisions 2026-11-07): {rows:,} rows,
a three-way crossing FE with ~25M levels, plus location.

- **Source:** EVE Ref Deep Archive torrent 09, `market-orders/history/*.v3.csv.bz2` (7,005
  snapshots, 9.44B order-snapshot rows), sha256-checked; `docs/sources/eve.md` on valiant.
  EVE Online data is CCP hf.'s (Third-Party Developer License, EULA); EVE Ref is MIT-0
- **Build:** `repo/replication/eve/convert_orders.py`, `order_day.py`, `package.py`,
  jemather/dfeols `{commit}`
- **Built:** {date.today()} on valiant
- **Rows:** {rows:,} in {len(parts)} parts by `type_id`; {levels["order_id"]:,} orders
- **Rules** (Decisions 2026-11-08): each order-day keeps the day's latest snapshot plus summaries of
  what changed; an order exits on the last day it's seen if it never reappears (a missed snapshot
  is a region fetch failure, not an exit); exit types coded; censoring flagged
- **Means:** exit {stats["exit"]:.4f}; vanished {stats["exit_vanished"]:.4f}, partly filled
  {stats["exit_partly_filled"]:.4f}, expired {stats["exit_expired"]:.4f}

## Parts

| File | Rows | type_id | Size |
|---|---|---|---|
{pt}

## Schema

{schema_table(data, notes)}

## FE / cluster levels

| Variable | Distinct |
|---|---:|
{lv}

## Suggested partition key and specs

Partition `type_id` (keeps type × region × day within a shard); cluster `type_id`. The risk set
is every order-day; right-censored orders never exit. These are a first try at a model (Ted:
"just try something"), chosen to be big, not to answer a question.

| Spec | Formula | N (pyfixest, one week) | pyfixest b (se) |
|---|---|---:|---|
{fit_rows}

Reference subsample: days 2023-07-10 to 07-16 (rows with nulls in the model dropped).
Full-precision values in `reference.json`. Then the full window; no reference outside dfeols.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
