# SPDX-License-Identifier: GPL-3.0-or-later
"""D6 v2: package the Fuzzwork order-day panel (2017-09 to 2026-02) as ``outbox/eve_order_day_v2/``.

    python repo/replication/eve/package_v2.py [--src tmp/eve/build/v2] [--out outbox] [--parts 160]
        [--from 2017-09-25 --to 2026-02-14] [--name eve_order_day_v2] [--no-ref]

Run from the project root after ``order_day_v2.py``. Keeps the regions seen in a snapshot on
every day of the window (region rule A, Decisions 2026-11-10; the list is read from the build's
``regions.parquet`` and written to the README), adds v1's model columns (``package.py``), and
writes data/ in contiguous ranges of ``day`` (each source file is one day, so a part reads only
its own days), then README.md and reference.json (pyfixest on v1's reference week, 2023-07-10..16,
specs r1, r2 and r4 of the v1 OSCER runs; Decisions 2026-11-09). A part already written with the
expected row count is kept, so a killed run resumes. Manifest afterwards (CLAUDE.md).

At 4.4B rows Polars counts in UInt32 (``polars[rt64]`` is not installed), so every count is per
part or per day file, cast to Int64 before summing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outbox_tools import dump_json, level_counts_parts, schema_table, split_counts

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package import frame  # v1's model columns, unchanged

NAME = "eve_order_day_v2"
REF = (date(2023, 7, 10), date(2023, 7, 16))  # v1's reference week
X = "log_rel_price + log_volume_remain + log_age_days + n_price_changes + is_buy"
FE_BASE = "type_id^region_id^day_index + location_id"
FE_R4 = "type_id^region_id^day_index + location_id^day_index"
OUTCOMES = [("exit", "any exit"), ("exit_vanished", "vanished (fill or cancel)"),
            ("exit_expired", "expired")]
SPECS = {  # the v1 OSCER runs (scripts/oscer_eve_fit.py), r3 left out (its added FE is nested)
    "r1": dict(fe=FE_BASE, vcov=[{"CRV1": "type_id"}, "hetero"]),
    "r2": dict(fe=FE_BASE, vcov=[{"CRV1": "type_id+location_id"}]),
    "r4": dict(fe=FE_R4, vcov=[{"CRV1": "type_id"}]),
}
DROP = ["first_seq", "last_seq"]  # build-internal snapshot sequence numbers
FES = {
    "order_id": ["order_id"], "type_id": ["type_id"], "region_id": ["region_id"],
    "location_id": ["location_id"], "day_index": ["day_index"],
    "type_id^region_id": ["type_id", "region_id"],
    "type_id^region_id^day_index": ["type_id", "region_id", "day_index"],
    "location_id^day_index": ["location_id", "day_index"],
    "type_id^location_id": ["type_id", "location_id"],
}
WITHIN_DAY = {"day_index", "type_id^region_id^day_index", "location_id^day_index"}


def day_files(src: Path, lo: date, hi: date) -> list[Path]:
    fs = sorted((src / "order_day").glob("*/*.parquet"))
    return [f for f in fs if lo <= date.fromisoformat(f.stem) <= hi]


def kept_regions(src: Path, lo: date, hi: date) -> tuple[list[int], pl.DataFrame]:
    """Regions present on every day of the build's window (region rule A)."""
    r = pl.read_parquet(src / "regions.parquet")
    days = r["day"].n_unique()
    g = (r.group_by("region_id").agg(days=pl.len(), orders_median=pl.col("orders").median())
         .sort("region_id"))
    keep = g.filter(pl.col("days") == days)["region_id"].to_list()
    return keep, g.with_columns(kept=pl.col("region_id").is_in(keep), window_days=pl.lit(days))


def write_part(files: list[Path], keep: list[int], lo: date, hi: date, p: Path, rows: int) -> dict:
    if p.exists() and pl.scan_parquet(p).select(pl.len()).collect().item() == rows:
        print(f"{p.name}: kept ({rows:,} rows)", flush=True)
    else:
        tmp = p.with_suffix(".parquet.part")
        lf = (pl.scan_parquet(files).filter(pl.col("region_id").is_in(keep)).drop(DROP))
        (frame(lf).sort("day", "order_id")
         .sink_parquet(tmp, compression="zstd", statistics=True, row_group_size=1_000_000))
        got = pl.scan_parquet(tmp).select(pl.len()).collect().item()
        if got != rows:
            raise ValueError(f"{p}: {got} rows written, {rows} expected")
        os.replace(tmp, p)
        print(f"{p.name}: {lo}..{hi}, {rows:,} rows, {p.stat().st_size / 1e9:.2f} GB", flush=True)
    return {"file": f"data/{p.name}", "rows": rows, "day_min": lo, "day_max": hi,
            "bytes": p.stat().st_size}


def fits(data_files: list[Path]) -> list[dict]:
    import pyfixest as pf

    cols = ["exit", "exit_vanished", "exit_expired", "log_rel_price", "log_volume_remain",
            "log_age_days", "n_price_changes", "is_buy", "type_id", "region_id", "day_index",
            "location_id"]
    pdf = (pl.scan_parquet(data_files).filter(pl.col("day").is_between(*REF)).select(cols)
           .drop_nulls().collect().to_pandas())
    out = []
    for spec, s in SPECS.items():
        for y, label in OUTCOMES:
            for v in s["vcov"]:
                m = pf.feols(f"{y} ~ {X} | {s['fe']}", data=pdf, vcov=v)
                coef, se = m.coef(), m.se()
                out.append({"spec": spec, "outcome": y, "label": label,
                            "formula": f"{y} ~ {X} | {s['fe']}", "vcov": v, "nobs": int(m._N),
                            "subsample": f"day {REF[0]}..{REF[1]}, nulls dropped",
                            "coef": {k: [float(coef[k]), float(se[k])] for k in coef.index}})
                print(f"ref {spec} {y} {v}: N {int(m._N):,}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="tmp/eve/build/v2")
    ap.add_argument("--out", default="outbox")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--parts", type=int, default=160)
    ap.add_argument("--from", dest="lo", type=date.fromisoformat, default=date(2017, 9, 25))
    ap.add_argument("--to", dest="hi", type=date.fromisoformat, default=date(2026, 2, 14))
    ap.add_argument("--no-ref", action="store_true", help="skip the pyfixest reference fits")
    a = ap.parse_args(argv)
    src, d = Path(a.src), Path(a.out) / a.name
    commit = subprocess.run(["git", "-C", "repo", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    summary = json.loads((src / "_summary.json").read_text())
    files = day_files(src, a.lo, a.hi)
    if not files:
        raise SystemExit(f"no day files in {src}/order_day for {a.lo}..{a.hi}")
    keep, regions = kept_regions(src, a.lo, a.hi)
    print(f"{len(files)} days, {len(keep)} regions kept of {regions.height}", flush=True)

    # rows per day after the region filter (one file per day, each well under 2^32 rows)
    per_day = pl.concat([
        pl.scan_parquet(f).select(day=pl.col("day").first(),
                                  n=pl.col("region_id").is_in(keep).sum().cast(pl.Int64),
                                  all=pl.len().cast(pl.Int64)).collect()
        for f in files])
    total, dropped = int(per_day["n"].sum()), int((per_day["all"] - per_day["n"]).sum())
    src_bytes = sum(f.stat().st_size for f in files)
    need = int(src_bytes * 1.25) + 20 * 2**30  # v1's package was ~1.1x its source
    free = shutil.disk_usage(Path(a.out)).free
    if not d.exists() and free < need:
        raise SystemExit(f"{a.out} has {free / 1e9:.0f} GB free; this needs about {need / 1e9:.0f} GB")
    print(f"{total:,} order-days kept, {dropped:,} dropped; source {src_bytes / 1e9:.1f} GB, "
          f"{free / 1e9:.0f} GB free", flush=True)

    (d / "data").mkdir(parents=True, exist_ok=True)
    by_day = dict(zip(per_day["day"].to_list(), files))
    parts = []
    for i, (lo, hi, rows) in enumerate(split_counts(per_day["day"], per_day["n"], a.parts)):
        fs = [f for dd, f in by_day.items() if lo <= dd <= hi]
        parts.append(write_part(fs, keep, lo, hi, d / "data" / f"part-{i:04d}.parquet", rows))
    rows = sum(p["rows"] for p in parts)
    if rows != total:
        raise ValueError(f"parts hold {rows:,} rows, expected {total:,}")
    stale = sorted(set((d / "data").glob("*.parquet")) - {d / p["file"] for p in parts})
    if stale:
        raise SystemExit(f"{len(stale)} part files from another split in {d / 'data'}: {stale[0]}")
    data_files = [d / p["file"] for p in parts]

    print("level counts", flush=True)
    fes = {k: v for k, v in FES.items() if k != "order_id"}
    levels = {"order_id": 0, **level_counts_parts(data_files, fes, WITHIN_DAY)}
    sums = {k: 0 for k in ("exit", "exit_vanished", "exit_partly_filled", "exit_expired",
                           "right_censored", "left_censored", "new_order")}
    for f in data_files:  # an order's first order-day in the window counts it once
        r = (pl.scan_parquet(f).select(
            first=(pl.col("day") == pl.col("first_day")).sum().cast(pl.Int64),
            **{k: pl.col(k).cast(pl.Int64).sum() for k in sums}).collect().row(0, named=True))
        levels["order_id"] += r.pop("first")
        for k in sums:
            sums[k] += r[k]
    stats = {k: v / rows for k, v in sums.items()}
    data = pl.scan_parquet(data_files)
    ref_fits = [] if a.no_ref or not (a.lo <= REF[0] and REF[1] <= a.hi) else fits(data_files)

    dump_json(d / "reference.json", {
        "dataset": a.name, "build": f"jemather/dfeols {commit}",
        "order_day_build": summary.get("build_commit"), "tool": "pyfixest (in memory, valiant)",
        "rows": rows, "rows_dropped_by_region_rule": dropped, "regions_kept": keep,
        "fe_levels": levels, "means": stats, "fits": ref_fits, "parts": parts,
    })
    write_readme(d, a, commit, summary, files, rows, dropped, keep, regions, parts, levels,
                 stats, ref_fits, data)
    return 0


NOTES = {
    "order_id": "EVE market order; one row per order per UTC day it was seen",
    "region_id": "only the regions in every day's snapshots (see Sample)",
    "day": "UTC day",
    "day_index": "days since 2017-09-25 (the window's first day)",
    "n_snap": "snapshots the order was seen in that day",
    "day_snaps": "snapshots that day (all regions)",
    "price": "latest snapshot of the day; price_first/min/max/mean summarise the day",
    "first_day": "order's first day seen in the window",
    "left_censored": "open at the window's first snapshot (2017-09-25 20:07 UTC)",
    "right_censored": "seen on the window's last day",
    "region_covered_next": "the region is in at least one snapshot the next day (null on the last day)",
    "new_order": "first order-day of an order placed in the window",
    "exit": "1 on the order's last day seen, unless right-censored",
    "exit_type": "expired (issued + duration within 35 min of last sighting), partly_filled "
    "(volume_remain < volume_total), vanished (filled at once or cancelled)",
    "log_rel_price": "log price minus its type × region × day × side mean",
    "age_days": "day minus the date of `issued` (an edit resets issued)",
}


def write_readme(d, a, commit, summary, files, rows, dropped, keep, regions, parts, levels, stats,
                 ref_fits, data) -> None:
    def esc(s: str) -> str:
        return s.replace("|", "\\|")

    pt = "\n".join(f"| {p['file']} | {p['rows']:,} | {p['day_min']}..{p['day_max']} | "
                   f"{p['bytes'] / 1e9:.2f} GB |" for p in parts)
    lv = "\n".join(f"| {k} | {v:,} |" for k, v in levels.items())
    sp = "\n".join(f"| {k} | `{esc(s['fe'])}` | "
                   + ", ".join(f"CRV1 {v['CRV1']}" if isinstance(v, dict) else v for v in s["vcov"])
                   + " |" for k, s in SPECS.items())
    fr = "\n".join(
        f"| {r['spec']} | {r['outcome']} | {r['vcov']['CRV1'] if isinstance(r['vcov'], dict) else r['vcov']} "
        f"| {r['nobs']:,} | " + ", ".join(f"{t} {v[0]:.6f} ({v[1]:.6f})" for t, v in r["coef"].items())
        + " |" for r in ref_fits) or "| (not run) | | | | |"
    gone = regions.filter(~pl.col("kept"))
    gb = sum(p["bytes"] for p in parts) / 1e9
    (d / "README.md").write_text(f"""# {a.name}

EVE Online market orders, one row per order per day, {files[0].stem} to {files[-1].stem}
({len(files):,} days), built from the Fuzzwork market snapshots in EVE Ref's Deep Archive (all 16
torrents, about 44 snapshots a day), for a discrete-time hazard LPM of order exit. A scale test
only, not for the paper (Decisions 2026-11-07). {rows:,} rows in {len(parts)} parts ({gb:.0f} GB).
v1 (`eve_order_day_v1`) is the same panel on 147 days of EVE Ref's own snapshots.

- **Source:** `fuzzwork/ordersets/` in EVE Ref Deep Archive torrents 00-15 ({summary['ordersets']:,}
  snapshots, ids 21000 on; sha256-checked against each torrent's list), `docs/sources/eve.md` on
  valiant. EVE Online data is CCP hf.'s (Third-Party Developer License, EULA); EVE Ref is MIT-0
- **Build:** `repo/replication/eve/fuzzwork.py`, `order_day_v2.py` (jemather/dfeols
  `{summary.get('build_commit')}`), `package_v2.py` (`{commit}`)
- **Built:** {date.today()} on valiant
- **Rules:** v1's (Decisions 2026-11-08): each order-day keeps the day's latest snapshot plus
  summaries of what changed; an order exits on the last day it's seen if it never reappears (a
  missed snapshot is a fetch failure, not an exit); exit types coded; censoring flagged.
  Fuzzwork-specific (Decisions 2026-11-09): snapshots in id order; {summary['copies_dropped']}
  snapshots identical to the one before dropped as copies; {summary['dup_rows']:,} exact
  duplicate rows within a snapshot dropped, and for {summary['dup_conflicts']:,} order ids listed
  twice with different values the first row kept
- **Sample** (region rule A, Decisions 2026-11-10): the {len(keep)} regions present on every day.
  Fuzzwork stops covering about 30 null-sec regions in 2018-06 and others come and go; those
  {gone.height} regions ({dropped:,} order-days, mostly before 2018-07) are dropped, so the risk
  set is the same throughout. Kept: {", ".join(map(str, keep))}
- **Checked against v1** on torrent 09's 147 days (`docs/sources/eve.md`): 98.7% of v1's
  order-days in these regions found; on those, exit agrees 99.95%, censoring 99.99%, exit type
  96.8% where both exit; type and location identical
- **Means:** exit {stats['exit']:.4f}; vanished {stats['exit_vanished']:.4f}, partly filled
  {stats['exit_partly_filled']:.4f}, expired {stats['exit_expired']:.4f}; new order
  {stats['new_order']:.4f}, right-censored {stats['right_censored']:.4f}
- **Not in v2** (Fuzzwork lacks them): system_id, constellation_id, http_last_modified

## Parts

Contiguous ranges of `day`, sorted by day and order_id within a part.

| File | Rows | Days | Size |
|---|---|---|---|
{pt}

## Schema

{schema_table(data, NOTES)}

## FE / cluster levels

`order_id` counts orders (each order's first order-day); the rest are distinct values.

| Variable | Distinct |
|---|---:|
{lv}

## Suggested partition key and specs

Partition `day` (the parts' key; `type_id^region_id^day_index` and `location_id^day_index` nest
in it); cluster `type_id`. The risk set is every order-day; right-censored orders never exit.
The specs are v1's OSCER runs r1, r2 and r4 (Decisions 2026-11-09), each on the three outcomes
`exit`, `exit_vanished`, `exit_expired`, with `{X}`:

| Spec | FE | vcov |
|---|---|---|
{sp}

Reference (pyfixest, days {REF[0]} to {REF[1]}, rows with nulls in the model dropped; full
precision in `reference.json`):

| Spec | Outcome | vcov | N | b (se) |
|---|---|---|---:|---|
{fr}

Then the full window; no reference outside dfeols.
""")


if __name__ == "__main__":
    sys.exit(main())
