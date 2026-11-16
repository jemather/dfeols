# SPDX-License-Identifier: GPL-3.0-or-later
"""EVE contracts: package single-item item-exchange contracts as ``outbox/eve_contracts_v1/``.

    python repo/replication/eve/package_contracts.py market  [--from 2021-06-17 --to 2026-02-14] [--workers 8]
    python repo/replication/eve/package_contracts.py package [--from ... --to ...] [--parts 4] [--no-ref]
        [--name eve_contracts_v1] [--out outbox]
    python repo/replication/eve/package_contracts.py all ...   (market, then package)

Run from the project root after ``contracts.py build`` (tables in ``tmp/eve/build/contracts/``)
and ``order_day_v2.py`` (``tmp/eve/build/v2/order_day/``). ``--from/--to`` select contracts by
issue day (the test uses a few months; the full window is the default). Rules (Decisions
2026-11-16, ``docs/eve_contracts.md``):

* **Unit:** one row per item_exchange contract with exactly one item record over all its
  snapshots. Contracts already open in the archive's first snapshot (left-censored) are dropped.
  Terms are the contract's last terms version (``date_issued`` moves on a few contracts in 2021;
  ``station_id`` is filled for structures from 2023-05-23); the item is its first run.
* **Exit:** after the contract's last snapshot, the archive's next snapshot. ``right_censored``:
  the last snapshot is the archive's last. Otherwise ``expired`` if the next snapshot is at or
  after the final ``date_expired``, else ``vanished`` (accepted or deleted by the issuer; the
  data can't tell). Missing snapshots in between are not exits (``n_missing_snapshots``).
  ``exit_in_gap``: the next snapshot is more than 90 min after the last. ``hours_open``: from
  ``date_issued`` to the midpoint of the last and next snapshot (null if right-censored).
* **Prices:** ``unit_price`` = price / quantity for included items (null for wanted items).
  ``log_rel_price_market``: log unit price over the median price of the sell orders for the
  type in the contract's region on its issue day (order-day v2 build, all regions), plain items
  only (not blueprint copies, not mutated). ``log_rel_price_contract``: log unit price minus
  the leave-one-out median log unit price of the other packaged contracts with the same type,
  ``is_blueprint_copy``, runs, ME, TE and issue month, where there are at least 5 others; null
  for mutated items. Blueprint copies, mutated items and wanted items are kept with flags.
* **Specs:** LPMs of ``vanished`` (r1-r3) and OLS of ``log_hours_open`` among vanished (r4) on
  the relative price and ``log_quantity``, each on the two price samples (SPECS, SAMPLES);
  reference fits with pyfixest on the full samples.

``market`` writes ``tmp/eve/build/contracts_pkg/market/<day>.parquet`` (only the type × region
× day keys the contracts need; resumes by day). ``package`` writes the contract frame to
``tmp/eve/build/contracts_pkg/frame.parquet``, then data/ parts by issue month, README.md and
reference.json. The manifest is written afterwards (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outbox_tools import dump_json, level_counts, schema_table, split_counts

NAME = "eve_contracts_v1"
SRC = Path("tmp/eve/build/contracts")
ORDERS = Path("tmp/eve/build/v2/order_day")
WORK = Path("tmp/eve/build/contracts_pkg")
LO, HI = date(2021, 6, 17), date(2026, 2, 14)
DAY0 = date(2021, 6, 17)  # issue_day_index origin (the archive's first day)
GAP_MIN = 90
MIN_OTHERS = 5
FIXEF_MAXITER = 100_000  # r3 needs more than the default 10,000 on a test quarter
GROUP = ["type_id", "is_blueprint_copy", "runs", "material_efficiency", "time_efficiency",
         "issue_month"]

FE1 = "item_group + region_id^issue_day_index + duration_days"
FE2 = FE1 + " + issuer_id"
FE3 = FE2 + " + start_location_id"
SPECS = {  # Decisions 2026-11-16
    "r1": dict(y="vanished", fe=FE1, vcov={"CRV1": "type_id"}),
    "r2": dict(y="vanished", fe=FE2, vcov={"CRV1": "type_id"}),
    "r3": dict(y="vanished", fe=FE3, vcov={"CRV1": "type_id+issuer_id"}),
    "r4": dict(y="log_hours_open", fe=FE2, vcov={"CRV1": "type_id"}, vanished_only=True),
}
SAMPLES = {"market": "log_rel_price_market", "contract": "log_rel_price_contract"}
FES = {k: [k] for k in ["contract_id", "issuer_id", "issuer_corporation_id", "type_id",
                        "item_group", "region_id", "system_id", "start_location_id",
                        "issue_day_index", "duration_days"]}
FES["region_id^issue_day_index"] = ["region_id", "issue_day_index"]


# --------------------------------------------------------------------------- market prices
def _market_day(day: date, keys: str, out: str) -> tuple[str, int, int]:
    os.environ["POLARS_MAX_THREADS"] = "2"
    f = ORDERS / f"{day.year}" / f"{day}.parquet"
    k = pl.scan_parquet(keys).filter(pl.col("issue_day") == day).select("type_id", "region_id")
    if not f.exists():
        return str(day), 0, -1
    m = (pl.scan_parquet(f).filter(~pl.col("is_buy_order"))
         .join(k, on=["type_id", "region_id"], how="semi")
         .group_by("type_id", "region_id")
         .agg(market_median_price=pl.col("price").median(), market_n_orders=pl.len().cast(pl.Int32))
         .with_columns(issue_day=pl.lit(day))
         .collect())
    tmp = Path(out + ".part")
    m.write_parquet(tmp)
    os.replace(tmp, out)
    return str(day), m.height, 1


def market(lo: date, hi: date, workers: int) -> None:
    """Median sell-order price per type × region × issue day, for the keys plain items need."""
    t0 = time.time()
    (WORK / "market").mkdir(parents=True, exist_ok=True)
    keys = WORK / "market_keys.parquet"
    f = base(lo, hi).filter(pl.col("plain")).select("type_id", "region_id", "issue_day").unique()
    f.collect().write_parquet(keys)
    days = [lo + timedelta(i) for i in range((hi - lo).days + 1)]
    todo = [d for d in days if not (WORK / "market" / f"{d}.parquet").exists()]
    print(f"market: {len(days)} days, {len(todo)} to do, {workers} workers", flush=True)
    missing = 0
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_market_day, d, str(keys), str(WORK / "market" / f"{d}.parquet"))
                for d in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            d, n, ok = fu.result()
            missing += ok < 0
            if i % 100 == 0 or i == len(futs):
                print(f"market: {i}/{len(futs)} days ({time.time() - t0:.0f} s)", flush=True)
    print(f"market: done, {missing} days without an order-day file, {time.time() - t0:.0f} s",
          flush=True)


# --------------------------------------------------------------------------- contract frame
def base(lo: date, hi: date) -> pl.LazyFrame:
    """One row per single-item item_exchange contract issued in [lo, hi], not left-censored,
    with terms, item and exit coding (no prices relative to others yet)."""
    S = pl.scan_parquet(SRC / "snapshots.parquet").select("snapshot_id", "snap_time").sort("snapshot_id")
    n_snap = S.select(pl.len()).collect().item()
    S = S.with_columns(next_time=pl.col("snap_time").shift(-1))
    c = pl.scan_parquet(SRC / "contracts" / "*.parquet").filter(pl.col("type") == "item_exchange")
    terms = (c.group_by("contract_id")
             .agg(pl.all().exclude("contract_id", "terms_version", "first_snapshot",
                                   "last_snapshot", "n_snapshots", "n_terms", "n_changes",
                                   "type").sort_by("terms_version").last(),
                  first_snapshot=pl.col("first_snapshot").min(),
                  last_snapshot=pl.col("last_snapshot").max(),
                  n_snapshots=pl.col("n_snapshots").sum(),
                  n_terms=pl.col("n_terms").first()))
    exp = (pl.scan_parquet(SRC / "contract_changes" / "*.parquet")
           .group_by("contract_id")
           .agg(date_expired_first=pl.col("date_expired").sort_by("from_snapshot").first(),
                date_expired=pl.col("date_expired").sort_by("from_snapshot").last(),
                n_expiry=pl.col("date_expired").n_unique().cast(pl.Int16)))
    icols = ["type_id", "quantity", "is_included", "is_blueprint_copy", "material_efficiency",
             "time_efficiency", "runs", "dynamic"]
    items = (pl.scan_parquet(SRC / "items" / "*.parquet")
             .group_by("contract_id")
             .agg(n_records=pl.col("record_id").n_unique(),
                  n_item_values=pl.struct(icols).n_unique(),
                  **{k: pl.col(k).sort_by("from_snapshot").first() for k in icols})
             .filter(pl.col("n_records") == 1))
    d = (terms.filter(pl.col("first_snapshot") > 0)
         .with_columns(issue_day=pl.col("date_issued").dt.date())
         .filter(pl.col("issue_day").is_between(lo, hi))
         .join(items, on="contract_id")
         .join(exp, on="contract_id", how="left")
         .join(S.select(first_snapshot="snapshot_id", first_seen="snap_time"), on="first_snapshot")
         .join(S.select(last_snapshot="snapshot_id", last_seen="snap_time",
                        next_snapshot_time="next_time"), on="last_snapshot"))
    bpc = pl.col("is_blueprint_copy").fill_null(False)
    mutated = pl.col("dynamic").fill_null(False)
    rc = pl.col("last_snapshot") == n_snap - 1
    mid = pl.col("last_seen") + (pl.col("next_snapshot_time") - pl.col("last_seen")) / 2
    hours = (mid - pl.col("date_issued")).dt.total_seconds() / 3600
    unit = pl.when(pl.col("is_included")).then(pl.col("price") / pl.col("quantity"))
    return d.with_columns(
        is_blueprint_copy=bpc, mutated=mutated,
        is_blueprint_original=~bpc & pl.col("material_efficiency").is_not_null(),
        plain=pl.col("is_included") & ~bpc & ~mutated,
        item_group=(pl.col("type_id").cast(pl.Int64) * 2 + bpc.cast(pl.Int64)),
        issue_month=pl.col("issue_day").dt.truncate("1mo"),
        issue_day_index=(pl.col("issue_day") - pl.lit(DAY0)).dt.total_days().cast(pl.Int32),
        duration_days=((pl.col("date_expired_first") - pl.col("date_issued")).dt.total_seconds()
                       / 86400).round().cast(pl.Int32),
        n_missing_snapshots=(pl.col("last_snapshot") - pl.col("first_snapshot") + 1
                             - pl.col("n_snapshots")).cast(pl.Int32),
        right_censored=rc,
        expired=pl.when(rc).then(None).otherwise(
            (pl.col("next_snapshot_time") >= pl.col("date_expired")).cast(pl.Int8)),
        vanished=pl.when(rc).then(None).otherwise(
            (pl.col("next_snapshot_time") < pl.col("date_expired")).cast(pl.Int8)),
        exit_in_gap=pl.when(rc).then(None).otherwise(
            (pl.col("next_snapshot_time") - pl.col("last_seen")).dt.total_minutes() > GAP_MIN),
        hours_open=pl.when(rc).then(None).otherwise(hours),
        unit_price=unit,
        log_unit_price=unit.log(),
        log_quantity=pl.col("quantity").cast(pl.Float64).log(),
    ).with_columns(log_hours_open=pl.when(pl.col("hours_open") > 0).then(pl.col("hours_open").log()))


def loo_median(df: pl.DataFrame, x: str, group: list[str], min_others: int) -> pl.Series:
    """Median of x over the other rows of the row's group (null below min_others others).
    Groups are sorted; for a row at rank r of n, the others' k-th value is the group's k-th if
    k < r, else its (k+1)-th."""
    d = (df.with_row_index("_row")
         .with_columns(_ok=pl.col(x).is_not_null())
         .sort([pl.col("_ok")] + group + [x], nulls_last=True, maintain_order=True)
         .with_row_index("_pos"))
    gk = group + ["_ok"]
    d = d.with_columns(_n=pl.len().over(gk), _start=pl.col("_pos").min().over(gk))
    d = d.with_columns(_r=pl.col("_pos") - pl.col("_start"), _m=pl.col("_n") - 1)
    k1 = (pl.col("_m") - 1) // 2
    k2 = pl.col("_m") // 2
    def at(k):  # global position of the others' k-th value
        return pl.col("_start") + k + (k >= pl.col("_r")).cast(pl.UInt32)
    ok = pl.col("_ok") & (pl.col("_m") >= min_others)
    d = d.with_columns(_i1=pl.when(ok).then(at(k1)), _i2=pl.when(ok).then(at(k2)))
    vals = d[x]
    v1 = vals.gather(d["_i1"].fill_null(0)).to_frame("v1")
    v2 = vals.gather(d["_i2"].fill_null(0)).to_frame("v2")
    d = pl.concat([d, v1, v2], how="horizontal").with_columns(
        _med=pl.when(ok).then((pl.col("v1") + pl.col("v2")) / 2),
        _others=pl.when(pl.col("_ok")).then(pl.col("_m")))
    d = d.sort("_row")
    return d["_med"], d["_others"]


def frame(lo: date, hi: date) -> pl.DataFrame:
    d = base(lo, hi).collect(engine="streaming")
    mk = pl.scan_parquet(sorted((WORK / "market").glob("*.parquet")))
    d = (d.join(mk.collect(), on=["type_id", "region_id", "issue_day"], how="left")
         .with_columns(log_rel_price_market=pl.when(pl.col("plain"))
                       .then(pl.col("log_unit_price") - pl.col("market_median_price").log())))
    x = d.with_columns(_lp=pl.when(~pl.col("mutated")).then(pl.col("log_unit_price")))
    med, others = loo_median(x, "_lp", GROUP, MIN_OTHERS)
    d = d.with_columns(contract_ref_median_log_price=med, contract_ref_others=others.cast(pl.Int32))
    d = d.with_columns(log_rel_price_contract=pl.col("log_unit_price")
                       - pl.col("contract_ref_median_log_price"))
    cols = ["contract_id", "issuer_id", "issuer_corporation_id", "for_corporation", "type_id",
            "item_group", "quantity", "is_included", "is_blueprint_copy", "is_blueprint_original", "runs",
            "material_efficiency", "time_efficiency", "mutated", "plain", "n_item_values",
            "region_id", "constellation_id", "system_id", "start_location_id", "station_id",
            "date_issued", "issue_day", "issue_month", "issue_day_index", "date_expired_first",
            "date_expired", "n_expiry", "duration_days", "n_terms", "price", "unit_price",
            "log_unit_price", "log_quantity", "market_median_price", "market_n_orders",
            "log_rel_price_market", "contract_ref_median_log_price", "contract_ref_others",
            "log_rel_price_contract", "first_seen", "last_seen", "next_snapshot_time",
            "n_snapshots", "n_missing_snapshots", "right_censored", "vanished", "expired",
            "exit_in_gap", "hours_open", "log_hours_open", "title"]
    return d.select(cols).sort("issue_day", "contract_id")


# --------------------------------------------------------------------------- fits
def sample_frame(d: pl.DataFrame, sample: str, spec: dict) -> pl.DataFrame:
    col = SAMPLES[sample]
    f = d.filter(pl.col(col).is_not_null() & ~pl.col("right_censored"))
    if spec.get("vanished_only"):
        f = f.filter(pl.col("vanished") == 1)
    return f.rename({col: "log_rel_price"})


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.5f}"


def fits(d: pl.DataFrame) -> list[dict]:
    """pyfixest on each sample and spec. Two-way clustering (r3) can give a negative variance
    (V_type + V_issuer - V_both); then se is null and `var` holds the variance."""
    import numpy as np
    import pyfixest as pf

    cols = ["vanished", "log_hours_open", "log_rel_price", "log_quantity", "type_id",
            "item_group", "region_id", "issue_day_index", "duration_days", "issuer_id",
            "start_location_id"]
    out = []
    for sample in SAMPLES:
        for name, s in SPECS.items():
            t0 = time.time()
            pdf = sample_frame(d, sample, s).select(cols).drop_nulls().to_pandas()
            fml = f"{s['y']} ~ log_rel_price + log_quantity | {s['fe']}"
            m = pf.feols(fml, data=pdf, vcov=s["vcov"], fixef_maxiter=FIXEF_MAXITER)
            coef, var = m.coef(), dict(zip(m.coef().index, np.diagonal(m._vcov)))
            se = {k: float(np.sqrt(v)) if v >= 0 else None for k, v in var.items()}
            out.append({"sample": sample, "spec": name, "outcome": s["y"], "formula": fml,
                        "price_column": SAMPLES[sample], "vcov": s["vcov"], "nobs": int(m._N),
                        "rows_in": len(pdf), "seconds": round(time.time() - t0, 1),
                        "fixef_maxiter": FIXEF_MAXITER,
                        "coef": {k: [float(coef[k]), se[k]] for k in coef.index},
                        "var": {k: float(v) for k, v in var.items()}})
            print(f"ref {sample} {name}: N {int(m._N):,} of {len(pdf):,}, "
                  + ", ".join(f"{k} {coef[k]:.5f} ({_f(se[k])})" for k in coef.index)
                  + f", {time.time() - t0:.0f} s", flush=True)
    return out


# --------------------------------------------------------------------------- package
def package(lo: date, hi: date, out: Path, name: str, n_parts: int, ref: bool) -> None:
    t0 = time.time()
    commit = subprocess.run(["git", "-C", "repo", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    WORK.mkdir(parents=True, exist_ok=True)
    d = frame(lo, hi)
    d.write_parquet(WORK / "frame.parquet")
    print(f"frame: {d.height:,} contracts, {time.time() - t0:.0f} s", flush=True)
    dest = out / name
    (dest / "data").mkdir(parents=True, exist_ok=True)
    per = d.group_by("issue_month").len().sort("issue_month")
    parts = []
    for i, (a, b, rows) in enumerate(split_counts(per["issue_month"], per["len"], n_parts)):
        p = dest / "data" / f"part-{i:04d}.parquet"
        part = d.filter(pl.col("issue_month").is_between(a, b))
        assert part.height == rows
        tmp = p.with_suffix(".parquet.part")
        part.write_parquet(tmp, compression="zstd", statistics=True, row_group_size=500_000)
        os.replace(tmp, p)
        parts.append({"file": f"data/{p.name}", "rows": rows, "issue_month_min": a,
                      "issue_month_max": b, "bytes": p.stat().st_size})
        print(f"{p.name}: {a}..{b}, {rows:,} rows, {p.stat().st_size / 1e6:.0f} MB", flush=True)
    stale = sorted(set((dest / "data").glob("*.parquet")) - {dest / p["file"] for p in parts})
    if stale:
        raise SystemExit(f"{len(stale)} part files from another split in {dest / 'data'}: {stale[0]}")
    levels = level_counts(d.lazy(), FES)
    st = stats(d)
    ref_fits = fits(d) if ref else []
    dump_json(dest / "reference.json", {
        "dataset": name, "build": f"jemather/dfeols {commit}", "tool": "pyfixest (in memory, valiant)",
        "issue_days": [lo, hi], "rows": d.height, "fe_levels": levels, "stats": st,
        "fits": ref_fits, "parts": parts})
    write_readme(dest, name, commit, lo, hi, d, parts, levels, st, ref_fits)
    print(f"package: {dest}, {time.time() - t0:.0f} s", flush=True)


def stats(d: pl.DataFrame) -> dict:
    ex = d.filter(~pl.col("right_censored"))
    return {
        "rows": d.height,
        "blueprint_copy": int(d["is_blueprint_copy"].sum()), "mutated": int(d["mutated"].sum()),
        "wanted": int((~d["is_included"]).sum()), "plain": int(d["plain"].sum()),
        "blueprint_original": int(d["is_blueprint_original"].sum()),
        "market_price_bpo": int((d["log_rel_price_market"].is_not_null() & d["is_blueprint_original"]).sum()),
        "log_rel_price_market_median_bpo": float(d.filter(pl.col("is_blueprint_original"))["log_rel_price_market"].median() or float("nan")),
        "log_rel_price_market_median_not_bpo": float(d.filter(~pl.col("is_blueprint_original"))["log_rel_price_market"].median() or float("nan")),
        "right_censored": int(d["right_censored"].sum()),
        "vanished": int(ex["vanished"].sum()), "expired": int(ex["expired"].sum()),
        "exit_in_gap": int(ex["exit_in_gap"].sum()),
        "market_price": int(d["log_rel_price_market"].is_not_null().sum()),
        "contract_price": int(d["log_rel_price_contract"].is_not_null().sum()),
        "item_values_changed": int((d["n_item_values"] > 1).sum()),
        "expiry_changed": int((d["n_expiry"] > 1).sum()),
        "with_missing_snapshots": int((d["n_missing_snapshots"] > 0).sum()),
        "mean_vanished": float(ex["vanished"].mean()),
        "median_hours_open_vanished": float(ex.filter(pl.col("vanished") == 1)["hours_open"].median()),
        "log_rel_price_market_quartiles": [float(d["log_rel_price_market"].quantile(q)) for q in (.25, .5, .75)],
        "log_rel_price_contract_quartiles": [float(d["log_rel_price_contract"].quantile(q)) for q in (.25, .5, .75)],
    }


NOTES = {
    "contract_id": "one row per single-item item_exchange contract",
    "item_group": "type_id * 2 + is_blueprint_copy (FE)",
    "is_included": "false: a wanted item (the acceptor supplies it); prices null",
    "mutated": "mutated (abyssal) item; relative prices null",
    "plain": "included, not a blueprint copy, not mutated (the market price sample)",
    "is_blueprint_original": "blueprint original (ME/TE set, not a copy); in the market sample, "
    "but market sell orders for BPOs are mostly unresearched NPC seed orders (Decisions 2026-11-16)",
    "n_item_values": "distinct item values over the contract's snapshots (>1: changed on a refetch; first kept)",
    "date_issued": "terms are the contract's last terms version",
    "issue_day_index": "days since 2021-06-17",
    "date_expired_first": "expiry when first seen; duration_days from it",
    "date_expired": "expiry when last seen (exit coding)",
    "n_expiry": "distinct expiry times seen",
    "unit_price": "price / quantity; null for wanted items",
    "market_median_price": "median sell-order price, type × region × issue day (order-day v2 build)",
    "log_rel_price_market": "log(unit_price / market_median_price), plain items",
    "contract_ref_median_log_price": "leave-one-out median log unit price of the other contracts "
    "with the same type, is_blueprint_copy, runs, ME, TE and issue month (>= 5 others)",
    "contract_ref_others": "number of those others",
    "log_rel_price_contract": "log_unit_price - contract_ref_median_log_price; null for mutated items",
    "first_seen": "snapshot time; contracts in the archive's first snapshot are dropped",
    "next_snapshot_time": "the archive's snapshot after last_seen (null if right-censored)",
    "n_missing_snapshots": "snapshots between first and last seen without the contract (not exits)",
    "right_censored": "in the archive's last snapshot (2026-02-14)",
    "vanished": "gone before expiry: accepted or deleted (indistinguishable); null if right-censored",
    "expired": "next snapshot at or after date_expired; null if right-censored",
    "exit_in_gap": "next snapshot more than 90 min after last_seen (exit time loose)",
    "hours_open": "date_issued to the midpoint of last_seen and next_snapshot_time",
}


def write_readme(dest, name, commit, lo, hi, d, parts, levels, st, ref_fits) -> None:
    pt = "\n".join(f"| {p['file']} | {p['rows']:,} | {p['issue_month_min']:%Y-%m}..{p['issue_month_max']:%Y-%m} "
                   f"| {p['bytes'] / 1e6:.0f} MB |" for p in parts)
    lv = "\n".join(f"| {k} | {v:,} |" for k, v in levels.items())
    sp = "\n".join(f"| {k} | `{s['y']}`{' (vanished only)' if s.get('vanished_only') else ''} | "
                   f"`{s['fe']}` | CRV1 {s['vcov']['CRV1']} |" for k, s in SPECS.items())
    fr = "\n".join(
        f"| {r['sample']} | {r['spec']} | {r['outcome']} | {r['nobs']:,} | "
        + ", ".join(f"{t} {v[0]:.6f} ({'n/a' if v[1] is None else f'{v[1]:.6f}'})"
                    for t, v in r["coef"].items()) + " |"
        for r in ref_fits) or "| (not run) | | | | |"
    q = lambda k: ", ".join(f"{v:.2f}" for v in st[k])
    from importlib.metadata import version
    pf_version = version("pyfixest")
    n = st["rows"]
    (dest / "README.md").write_text(f"""# {name}

EVE Online public contracts: one row per single-item item-exchange contract, issued {lo} to
{hi}, built from EVE Ref's public-contracts snapshots (every 30 min, 2021-06-17..2026-02-14),
for LPMs of whether a listing goes before it expires, with issuer, item and region × day fixed
effects. A test dataset for dfeols, not for the paper (Decisions 2026-11-07). {n:,} rows in
{len(parts)} parts.

- **Source:** `public-contracts/history/` in EVE Ref Deep Archive torrents 04-15 (78,200 v2
  snapshots, sha256-checked), consolidated losslessly by `repo/replication/eve/contracts.py`
  (rebuild check: every 47th snapshot and 16 full days equal raw, content and stamps);
  market prices from the Fuzzwork order-day build (`order_day_v2.py`). EVE Online data is CCP
  hf.'s (Third-Party Developer License, EULA); EVE Ref is MIT-0. Notes: `docs/eve_contracts.md`
  on valiant
- **Build:** `repo/replication/eve/package_contracts.py` (jemather/dfeols `{commit}`), {date.today()}
- **Sample** (Decisions 2026-11-16): item_exchange contracts with one item record over all
  snapshots; contracts open in the archive's first snapshot dropped. Blueprint copies
  ({st['blueprint_copy']:,}), mutated items ({st['mutated']:,}) and wanted items
  ({st['wanted']:,}) kept with flags; plain items {st['plain']:,}, of them blueprint originals
  {st['blueprint_original']:,} (`is_blueprint_original`)
- **Exit:** the snapshot after the contract's last: `expired` if at or after its final expiry,
  else `vanished` (accepted or deleted; the data can't tell); right-censored if still listed in
  the last snapshot ({st['right_censored']:,}). Vanished {st['vanished']:,}, expired
  {st['expired']:,} (mean vanished {st['mean_vanished']:.3f}); {st['exit_in_gap']:,} exits
  fall in a gap of more than 90 min (in a long gap an exit after the expiry time is coded
  expired). Median hours open among vanished {st['median_hours_open_vanished']:.1f}.
  Missing snapshots between sightings are not exits ({st['with_missing_snapshots']:,} contracts
  have some)
- **Prices:** `log_rel_price_market` on {st['market_price']:,} rows (quartiles {q('log_rel_price_market_quartiles')});
  `log_rel_price_contract` on {st['contract_price']:,} (quartiles {q('log_rel_price_contract_quartiles')}).
  Market prices exist only where Fuzzwork has sell orders for the type in the region that day
  (about 35 regions; contracts are in more). Blueprint originals are in the market sample
  ({st['market_price_bpo']:,} rows) but sit far above it (median {st['log_rel_price_market_median_bpo']:.2f}
  against {st['log_rel_price_market_median_not_bpo']:.2f} for other items): the market's BPO sell
  orders are mostly unresearched NPC seed orders. Flagged, kept (Decisions 2026-11-16)
- **Refetches:** {st['item_values_changed']:,} contracts' item changed on an archive refetch
  (first value kept); {st['expiry_changed']:,} had their expiry move

## Parts

Contiguous ranges of `issue_month`, sorted by issue_day and contract_id within a part.

| File | Rows | Issue months | Size |
|---|---|---|---|
{pt}

## Schema

{schema_table(d.lazy(), NOTES)}

## FE / cluster levels

| Variable | Distinct |
|---|---:|
{lv}

## Suggested partition key and specs

Partition `type_id`; cluster `type_id` (r3: two-way `type_id` and `issuer_id`). Regressors:
`log_rel_price + log_quantity`, where `log_rel_price` is `log_rel_price_market` (sample
`market`: plain items with a market price) or `log_rel_price_contract` (sample `contract`);
rows right-censored or with nulls in the model dropped; r4 on vanished contracts only.

| Spec | Outcome | FE | vcov |
|---|---|---|---|
{sp}

Reference (pyfixest {pf_version} on the full samples, singleton FE levels dropped, so N is below
the rows in; demeaning up to {FIXEF_MAXITER:,} iterations; full precision in `reference.json`).
Two-way clustering (r3) can give a negative variance (V_type + V_issuer - V_both): then the se is
n/a and `reference.json` keeps the variance (`var`):

| Sample | Spec | Outcome | N | b (se) |
|---|---|---|---:|---|
{fr}
""")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["market", "package", "all"])
    ap.add_argument("--from", dest="lo", type=date.fromisoformat, default=LO)
    ap.add_argument("--to", dest="hi", type=date.fromisoformat, default=HI)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outbox")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--no-ref", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd in ("market", "all"):
        market(a.lo, a.hi, a.workers)
    if a.cmd in ("package", "all"):
        package(a.lo, a.hi, Path(a.out), a.name, a.parts, not a.no_ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
