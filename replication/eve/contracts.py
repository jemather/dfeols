# SPDX-License-Identifier: GPL-3.0-or-later
"""EVE Ref public-contracts snapshots, consolidated into six tables, losslessly.

    python repo/replication/eve/contracts.py build [--raw raw/eve] [--out tmp/eve/build/contracts]
        [--from 2021-06-17] [--to 2026-02-14] [--days d1,d2,...] [--workers 8] [--pass 1|2|both]
        [--force] [--buckets 32]
    python repo/replication/eve/contracts.py check [--out ...] [--days d1,...] [--every 47]
        [--workers 8] [--json <file>]

Run from the project root. Source: ``public-contracts/history/<YYYY>/<day>/
public-contracts-<day>_<HH-MM-SS>.v2.tar.bz2`` in torrents 04-15 (2021-06-17..2026-02-14, every
30 min; ``docs/sources/eve.md``). Each tar holds ``meta.json`` and seven CSVs (``contracts``,
``contract_items``, ``contract_bids``, ``contract_dynamic_items``,
``contract_dynamic_items_dogma_attributes`` / ``_effects``, ``contract_non_dynamic_items``), whose
column order varies between files, so they are read by name and cast strictly (SCHEMAS below;
anything that does not parse fails the snapshot). The April 2019 ``.json.gz`` snapshots and
``backfills/`` (other formats, third-party collections) are not read.

What the probes found (2021-06-17..20, 2023-07-11..13, 2025-03-01, 2026-02-14), and so how the
tables are cut:

* Contract terms almost never change: ``date_issued`` moves on one or two contracts a day in
  2021 (a midnight placeholder replaced by the real time, or back). ``date_expired`` moves
  forward every snapshot on auctions. ``universe_id`` (eve, wormhole or null) is in the files
  from 2021-06-19 ~16:00 to 2023-05-23 only. ``http_last_modified`` is one value per snapshot
  and region (the region's page fetch), with a few exceptions in June 2021.
* A contract's items are all present or all absent in a snapshot; they can appear one or two
  snapshots after the contract does, never earlier, and stay until it leaves. The same holds
  for its mutated (dynamic) items and their dogma rows, and bids stay once placed. So every
  child row carries the first and last snapshot it is in.
* Bids' ``http_last_modified`` is one value per contract and snapshot (the bids page fetch).
  Items' and mutated items' stamps are fixed per row, except 2021-06-17..07-05, when the
  archive refetched every contract's items about hourly and each fetch has a new stamp.
* No snapshots on 19 days (2021-06-24..07-01, 07-03, 2022-01-12..13, 2023-01-27..31,
  2024-11-22..23, 2025-05-19), and 98 gaps over 90 minutes between snapshots: a contract seen
  on both sides of a gap is in adjacent snapshot ids, so its spell runs through the gap.

Tables (``<out>/<table>/part-NN.parquet``, buckets of ``contract_id``; ``snapshots.parquet``):

``snapshots``        one row per snapshot file: ``snapshot_id`` (0.. in time order, over the whole
                     inventory, so a subset build uses the same ids), ``snap_time`` (from the file
                     name), ``scrape_start``/``scrape_end``/``datasource`` (datasource, or server
                     before 2023-05-24; times are ISO strings, or epoch-second numbers from
                     2022-10-29 to 2023-05-23) and ``meta`` (meta.json as parsed), torrent, rel, sha256,
                     bytes, per-table row counts, ``headers`` (JSON: each CSV's columns as
                     written), ``absent`` (CSVs the tar lacks, e.g. contract_non_dynamic_items
                     before 2024) and ``regions`` (list of region_id, http_last_modified,
                     contracts: the contracts' page stamp per region, the most common value).
``contracts``        one row per contract and terms version: every contracts column but
                     ``date_expired``, ``universe_id`` and ``http_last_modified``, plus
                     ``terms_version`` (0 = first seen; >0 only where a term changed, so
                     ``terms_version == 0`` gives one row per contract), ``first_snapshot``,
                     ``last_snapshot``, ``n_snapshots`` (of this version), ``n_terms``,
                     ``n_changes`` (the contract's rows in contract_changes).
``contract_changes`` spells: ``contract_id``, ``from_snapshot``, ``to_snapshot`` (inclusive),
                     ``terms_version``, ``date_expired``, ``universe_id``,
                     ``http_last_modified`` (only where it differs from the region's stamp in
                     ``snapshots.regions``, else null), ``bids_http_last_modified`` (null when no
                     bids rows). A new spell starts when any of these changes or the contract
                     is missing from a snapshot (region fetch failures look like that), so the
                     contract is in snapshot s exactly when a spell covers s.
``items``            contract_items rows (key contract_id, record_id) with ``first_snapshot``,
                     ``last_snapshot``; for mutated items also ``created_by``,
                     ``mutator_type_id``, ``source_type_id``, ``dynamic_http_last_modified``,
                     ``dynamic_first_snapshot``, ``dynamic_last_snapshot`` (from
                     contract_dynamic_items, matched on contract_id and item_id); for items
                     listed in contract_non_dynamic_items (2024 on), ``non_dynamic_first_snapshot``
                     and ``non_dynamic_last_snapshot``. ``http_last_modified`` is the first seen.
``item_attributes``  the mutated items' dogma rows, long: ``contract_id``, ``item_id``, ``kind``
                     (``attribute`` or ``effect``), ``dogma_id`` (attribute_id or effect_id),
                     ``value`` (attributes), ``is_default`` (effects), ``first_snapshot``,
                     ``last_snapshot``. Their http_last_modified equals the item's
                     ``dynamic_http_last_modified`` (counted in pass 1; the check confirms).
``bids``             ``bid_id``, ``contract_id``, ``amount``, ``date_bid``, ``first_snapshot``,
                     ``last_snapshot``.

Rebuild rule (``check``): a contract is in snapshot s when a spell covers s, with that spell's
terms version; a child row is in s when s is in its [first, last] range and its contract is
in s. http_last_modified comes from the spell or ``snapshots.regions`` (contracts),
``contract_changes`` (bids) or ``items`` (items, dynamic rows and dogma rows). ``check``
rebuilds every table of the chosen snapshots and compares them with the raw tars, value for
value after the same strict parse, and splits differing rows into content (some column other
than http_last_modified differs) and stamp-only. Stamp-only rows are expected in the 2021
refetch weeks (items keep their first stamp). Typed values are compared, so the CSV's spelling
of a number (``45000000`` vs ``4.5e7``) is not kept, nor, outside text columns, quoted-empty
(``""``) vs empty: both are null (some files quote empty numbers and booleans). In text
columns (``title``, ``type``, ``universe_id``) ``""`` and null stay distinct.

Pass 1, per day (``<out>/pass1/<YYYY>/<day>/*.parquet`` + ``_day.json``; resumable): the day's
snapshots, hash-checked, parsed, exact duplicate rows dropped (counted); rows collapsed to the
tables above within the day; anything that should be constant but varies is counted
(``conflicts``; the first sighting is kept). Pass 2 merges the days in buckets of contract_id
(ranges by min/max, spells joined across day boundaries when the ids are adjacent and the values
equal; terms versions numbered by first sighting) and writes the tables and ``_summary.json``.
Nothing in the archive is run.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import polars as pl

log = logging.getLogger("eve.contracts")
UTC = pl.Datetime("us", "UTC")
TORRENT = re.compile(r"everef-eve-online-deep-archive-(\d{2})$")
SNAP = re.compile(r"^public-contracts/history/\d{4}/(\d{4}-\d{2}-\d{2})/"
                  r"public-contracts-(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})\.v2\.tar\.bz2$")
TS = "%Y-%m-%dT%H:%M:%SZ"
I32, I64, F64, S, B, T = pl.Int32, pl.Int64, pl.Float64, pl.String, pl.Boolean, UTC

SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "contracts": {
        "contract_id": I64, "type": S, "issuer_id": I64, "issuer_corporation_id": I64,
        "for_corporation": B, "date_issued": T, "date_expired": T, "days_to_complete": I32,
        "start_location_id": I64, "end_location_id": I64, "region_id": I32, "station_id": I64,
        "system_id": I32, "constellation_id": I32, "price": F64, "reward": F64,
        "collateral": F64, "buyout": F64, "volume": F64, "title": S, "universe_id": S,
        "http_last_modified": T},
    "contract_items": {
        "contract_id": I64, "record_id": I64, "item_id": I64, "type_id": I32, "quantity": I64,
        "is_included": B, "is_blueprint_copy": B, "material_efficiency": I32,
        "time_efficiency": I32, "runs": I32, "http_last_modified": T},
    "contract_bids": {"contract_id": I64, "bid_id": I64, "amount": F64, "date_bid": T,
                      "http_last_modified": T},
    "contract_dynamic_items": {"contract_id": I64, "item_id": I64, "created_by": I64,
                               "mutator_type_id": I32, "source_type_id": I32,
                               "http_last_modified": T},
    "contract_dynamic_items_dogma_attributes": {"contract_id": I64, "item_id": I64,
                                                "attribute_id": I32, "value": F64,
                                                "http_last_modified": T},
    "contract_dynamic_items_dogma_effects": {"contract_id": I64, "item_id": I64,
                                             "effect_id": I32, "is_default": B,
                                             "http_last_modified": T},
    "contract_non_dynamic_items": {"contract_id": I64, "item_id": I64, "type_id": I32},
}
RAW_TABLES = list(SCHEMAS)
KEYS = {"contracts": ["contract_id"], "contract_items": ["contract_id", "record_id"],
        "contract_bids": ["bid_id"], "contract_dynamic_items": ["contract_id", "item_id"],
        "contract_dynamic_items_dogma_attributes": ["contract_id", "item_id", "attribute_id"],
        "contract_dynamic_items_dogma_effects": ["contract_id", "item_id", "effect_id"],
        "contract_non_dynamic_items": ["contract_id", "item_id"]}
CONTRACT_TERMS = [c for c in SCHEMAS["contracts"]
                  if c not in ("date_expired", "universe_id", "http_last_modified")]
SPELL_VALS = ["terms_hash", "date_expired", "universe_id", "http_last_modified",
              "bids_http_last_modified"]
ITEM_COLS = [c for c in SCHEMAS["contract_items"] if c not in KEYS["contract_items"]]
DYN_COLS = ["created_by", "mutator_type_id", "source_type_id", "http_last_modified"]
STAMP = "http_last_modified"
TABLES = ["contracts", "contract_changes", "items", "item_attributes", "bids"]


def commit_id() -> str:
    return subprocess.run(["git", "-C", "repo", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "unknown"


# ---------------------------------------------------------------- inventory and reader

def inventory(raw: Path) -> pl.DataFrame:
    """One row per v2 snapshot in every torrent: snapshot_id, day, snap_time, torrent, rel, path,
    sha256. Ids are 0.. in (snap_time, rel) order."""
    rows = []
    for tdir in sorted(p for p in raw.iterdir() if p.is_dir() and TORRENT.match(p.name)):
        sums = tdir / f"{tdir.name}.sha256"
        if not sums.exists():
            continue
        for ln in sums.read_text().splitlines():
            digest, rel = ln.split("  ", 1)
            if m := SNAP.match(rel):
                if m.group(1) != m.group(2):
                    raise ValueError(f"folder and name dates differ: {rel}")
                t = datetime.fromisoformat(f"{m.group(2)}T{m.group(3)}:{m.group(4)}:{m.group(5)}")
                rows.append({"day": m.group(2), "snap_time": t, "torrent": tdir.name[-2:],
                             "rel": rel, "path": str(tdir / rel), "sha256": digest})
    inv = pl.DataFrame(rows).with_columns(pl.col("snap_time").dt.replace_time_zone("UTC"))
    inv = inv.sort("snap_time", "rel").with_row_index("snapshot_id").with_columns(
        pl.col("snapshot_id").cast(I32))
    if inv["rel"].n_unique() != inv.height:
        raise ValueError("a snapshot path is in two torrents")
    return inv


def _cast(df: pl.DataFrame, schema: dict) -> pl.DataFrame:
    """Strict cast of an all-String frame; columns the file lacks become nulls (recorded in
    snapshots.headers). Unknown columns fail."""
    extra = set(df.columns) - set(schema)
    if extra:
        raise ValueError(f"unknown columns {sorted(extra)}")
    exprs = []
    for c, t in schema.items():
        if c not in df.columns:
            exprs.append(pl.lit(None, dtype=t).alias(c))
            continue
        if t == B:
            bad = df.filter(~(pl.col(c).is_null() | pl.col(c).is_in(["", "true", "false"])))
            if bad.height:
                raise ValueError(f"{c}: not a boolean: {bad[c][0]!r}")
            exprs.append(pl.col(c).replace_strict({"true": True, "false": False, "": None},
                                                  default=None, return_dtype=B).alias(c))
        elif t == UTC:
            exprs.append(pl.col(c).replace("", None).str.strptime(pl.Datetime("us"), TS, strict=True)
                         .dt.replace_time_zone("UTC").alias(c))
        elif t == S:
            exprs.append(pl.col(c))
        else:  # some files quote empty numbers (""): null, as an unquoted empty field
            exprs.append(pl.col(c).replace("", None).cast(t, strict=True))
    return df.select(exprs)


def read_snapshot(path: str, digest: str | None) -> tuple[dict, dict[str, pl.DataFrame]]:
    """meta + header info and one typed frame per raw table (empty frames for empty CSVs)."""
    data = Path(path).read_bytes()
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"sha256 mismatch: {path}")
    meta: dict = {"bytes": len(data), "headers": {}}
    frames: dict[str, pl.DataFrame] = {}
    with tarfile.open(fileobj=io.BytesIO(bz2.decompress(data)), mode="r:") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                raise ValueError(f"not a regular file in tar: {m.name}")
            b = tf.extractfile(m).read()
            if m.name == "meta.json":
                meta["meta"] = json.loads(b)
                continue
            name = m.name.removesuffix(".csv")
            if name not in SCHEMAS or not m.name.endswith(".csv"):
                raise ValueError(f"unexpected member {m.name}")
            if not b.strip():
                meta["headers"][name] = []
                frames[name] = _cast(pl.DataFrame(schema={c: S for c in SCHEMAS[name]}), SCHEMAS[name])
                continue
            df = pl.read_csv(io.BytesIO(b), infer_schema=False)
            meta["headers"][name] = df.columns
            frames[name] = _cast(df, SCHEMAS[name])
    if "meta" not in meta:
        raise ValueError("no meta.json")
    for name in set(SCHEMAS) - set(frames):  # e.g. contract_non_dynamic_items before 2024:
        frames[name] = _cast(pl.DataFrame(schema={c: S for c in SCHEMAS[name]}), SCHEMAS[name])
    meta["absent"] = sorted(set(SCHEMAS) - set(meta["headers"]))  # not in the tar at all
    return meta, frames


# ---------------------------------------------------------------- pass 1

def _ranges(df: pl.DataFrame, key: list[str], cols: list[str]) -> tuple[pl.DataFrame, dict]:
    """Collapse a stack of snapshot rows (column ``s``) to one row per key: first sighting's cols,
    first/last snapshot, n (sightings, so pass 2 can find gaps across days). Counts keys whose
    cols vary (http_last_modified apart, as ``stamp``) and keys with a gap."""
    vals = [c for c in cols if c != STAMP]
    g = df.sort("s").group_by(key, maintain_order=True).agg(
        *[pl.col(c).first() for c in cols],
        pl.col("s").min().alias("first_snapshot"), pl.col("s").max().alias("last_snapshot"),
        pl.col("s").n_unique().cast(I32).alias("n"),
        (pl.struct(vals).n_unique() if vals else pl.lit(1, pl.UInt32)).alias("_v"),
        (pl.col(STAMP).n_unique() if STAMP in cols else pl.lit(1, pl.UInt32)).alias("_st"))
    flags = {"vary": g.filter(pl.col("_v") > 1).height,
             "stamp": g.filter(pl.col("_st") > 1).height,
             "gaps": g.filter(pl.col("last_snapshot") - pl.col("first_snapshot") + 1
                              != pl.col("n")).height}
    return g.drop("_v", "_st"), flags


def _spell_ids(o: pl.DataFrame, key: str, start: str, end: str, vals: list[str]) -> pl.DataFrame:
    """Sorted rows -> a spell number: a new spell when start != previous end + 1 within the key
    or a value differs (nulls equal)."""
    same = (pl.col(key) == pl.col(key).shift()) & (pl.col(start) == pl.col(end).shift() + 1)
    for v in vals:
        same = same & pl.col(v).eq_missing(pl.col(v).shift())
    return o.with_columns(_spell=(~same.fill_null(False)).cum_sum())


def spells(obs: pl.DataFrame, vals: list[str]) -> pl.DataFrame:
    """Rows (contract_id, s, *vals) -> spells (contract_id, from_snapshot, to_snapshot, *vals)."""
    o = _spell_ids(obs.sort("contract_id", "s"), "contract_id", "s", "s", vals)
    return (o.group_by("_spell", maintain_order=True)
            .agg(pl.col("contract_id").first(), pl.col("s").min().alias("from_snapshot"),
                 pl.col("s").max().alias("to_snapshot"), *[pl.col(v).first() for v in vals])
            .drop("_spell"))


def region_stamps(c: pl.DataFrame) -> pl.DataFrame:
    """Per region: the most common contracts stamp (earliest on ties), distinct stamps, rows."""
    k = c.group_by("region_id", STAMP).len()
    top = k.sort("region_id", "len", STAMP, descending=[False, True, False]).group_by(
        "region_id", maintain_order=True).first().select("region_id", STAMP)
    n = k.group_by("region_id").agg(pl.len().alias("k"), pl.col("len").sum().cast(I32).alias("contracts"))
    return top.join(n, on="region_id").sort("region_id")


UTC_US = pl.Datetime("us", "UTC")


def meta_time(v) -> datetime | None:
    """A meta.json scrape time: an ISO string (``TS``), or, from 2022-10-29 ~23:00 to 2023-05-23
    ~23:00, Unix epoch seconds written as a JSON number (``1667086203.000000000``). Strict:
    anything else raises."""
    if v is None:
        return None
    if isinstance(v, str):
        return datetime.strptime(v, TS).replace(tzinfo=timezone.utc)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return datetime.fromtimestamp(round(v * 1_000_000) / 1_000_000, timezone.utc)
    raise ValueError(f"meta.json time {v!r}")


def collapse_day(day: str, snaps: list[dict], dest: str, commit: str) -> dict:
    t0 = time.time()
    stacks: dict[str, list[pl.DataFrame]] = {t: [] for t in RAW_TABLES}
    srows, dups, conflicts = [], {t: 0 for t in RAW_TABLES}, {}

    def add(k, v):
        conflicts[k] = conflicts.get(k, 0) + int(v)

    for sn in snaps:
        meta, fr = read_snapshot(sn["path"], sn["sha256"])
        sid = sn["snapshot_id"]
        counts = {}
        for t, df in fr.items():
            counts[t] = df.height
            df = df.unique(maintain_order=True)
            dups[t] += counts[t] - df.height
            kd = int(df.select(KEYS[t]).is_duplicated().sum())
            if kd:
                add(f"{t}.key_dup", kd)
                df = df.unique(subset=KEYS[t], keep="first", maintain_order=True)
            fr[t] = df.with_columns(s=pl.lit(sid, I32))
        reg = region_stamps(fr["contracts"])
        add("contracts.region_stamp", reg.filter(pl.col("k") > 1).height)
        fr["contracts"] = fr["contracts"].join(
            reg.select("region_id", pl.col(STAMP).alias("_r")), on="region_id", how="left"
        ).with_columns(pl.when(pl.col(STAMP) != pl.col("_r")).then(pl.col(STAMP))
                       .alias(STAMP)).drop("_r")
        for t in RAW_TABLES:
            stacks[t].append(fr[t])
        m = meta["meta"]
        srows.append({"snapshot_id": sid, "snap_time": sn["snap_time"],
                      "scrape_start": meta_time(m.get("scrape_start")),
                      "scrape_end": meta_time(m.get("scrape_end")),
                      "datasource": m.get("datasource", m.get("server")),
                      "meta": json.dumps(m, sort_keys=True),
                      "torrent": sn["torrent"], "rel": sn["rel"], "sha256": sn["sha256"],
                      "bytes": meta["bytes"], "headers": json.dumps(meta["headers"]),
                      "absent": ",".join(meta["absent"]),
                      **{f"n_{t}": counts[t] for t in RAW_TABLES},
                      "regions": reg.select("region_id", STAMP, "contracts").to_dicts()})
    st = {t: pl.concat(v) for t, v in stacks.items()}
    out = Path(dest)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    info: dict = {"day": day, "snapshots": [s["snapshot_id"] for s in snaps], "commit": commit,
                  "polars": pl.__version__, "dup_rows": dups,
                  "rows_in": {t: st[t].height for t in RAW_TABLES}}
    flags: dict = {}

    # contracts: terms versions (by hash) and spells
    c = st["contracts"].with_columns(terms_hash=pl.struct(CONTRACT_TERMS).hash(seed=0))
    con, f = _ranges(c, ["contract_id", "terms_hash"], [x for x in CONTRACT_TERMS if x != "contract_id"])
    add("contracts.terms_changed", con["contract_id"].is_duplicated().sum())
    con.write_parquet(tmp / "contracts.parquet")
    bstamp = (st["contract_bids"].group_by("contract_id", "s")
              .agg(pl.col(STAMP).n_unique().alias("k"), pl.col(STAMP).min().alias("bids_http_last_modified")))
    add("bids.stamp", bstamp.filter(pl.col("k") > 1).height)
    present = c.select("contract_id", "s")
    orph = {"contract_bids": bstamp.join(present, on=["contract_id", "s"], how="anti").height}
    obs = c.select("contract_id", "s", "terms_hash", "date_expired", "universe_id", STAMP).join(
        bstamp.drop("k"), on=["contract_id", "s"], how="left")
    spells(obs, SPELL_VALS).write_parquet(tmp / "spells.parquet")

    # children: ranges per key; orphans (rows whose contract is not in that snapshot) counted
    for t, cols in [("contract_items", ITEM_COLS), ("contract_dynamic_items", DYN_COLS),
                    ("contract_non_dynamic_items", ["type_id"]),
                    ("contract_bids", ["contract_id", "amount", "date_bid"])]:
        df = st[t]
        if t != "contract_bids":
            orph[t] = df.select("contract_id", "s").unique().join(
                present, on=["contract_id", "s"], how="anti").height
        r, flags[t] = _ranges(df, KEYS[t], cols)
        r.write_parquet(tmp / f"{t}.parquet")
    # dogma rows: long; stamp checked against the dynamic item's stamp in the same snapshot
    dyn = st["contract_dynamic_items"].select("contract_id", "item_id", "s", pl.col(STAMP).alias("_d"))
    longs = []
    for t, kind, idc in [("contract_dynamic_items_dogma_attributes", "attribute", "attribute_id"),
                         ("contract_dynamic_items_dogma_effects", "effect", "effect_id")]:
        df = st[t].join(dyn, on=["contract_id", "item_id", "s"], how="left")
        orph[t] = df.filter(pl.col("_d").is_null()).height
        add(f"{t}.stamp_vs_item", df.filter(pl.col("_d").is_not_null() & (pl.col("_d") != pl.col(STAMP))).height)
        df = df.select("contract_id", "item_id", pl.lit(kind).alias("kind"),
                       pl.col(idc).alias("dogma_id"),
                       (pl.col("value") if kind == "attribute" else pl.lit(None, F64)).alias("value"),
                       (pl.col("is_default") if kind == "effect" else pl.lit(None, B)).alias("is_default"),
                       "s")
        r, flags[t] = _ranges(df, ["contract_id", "item_id", "kind", "dogma_id"], ["value", "is_default"])
        longs.append(r)
    pl.concat(longs).write_parquet(tmp / "item_attributes.parquet")
    pl.DataFrame(srows, schema_overrides={"scrape_start": UTC_US, "scrape_end": UTC_US,
                                          "datasource": S, "meta": S, "absent": S}
                 ).write_parquet(tmp / "snapshots.parquet")
    info.update({"conflicts": conflicts, "flags": flags, "orphans": orph,
                 "contracts": con["contract_id"].n_unique(), "seconds": round(time.time() - t0, 1)})
    (tmp / "_day.json").write_text(json.dumps(info, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return info


def _day_flags(s: dict) -> dict:
    out = {k: v for k, v in s["conflicts"].items() if v}
    out.update({f"{t}.{k}": v for t, f in s["flags"].items() for k, v in f.items() if v})
    out.update({f"orphan.{t}": v for t, v in s["orphans"].items() if v})
    out.update({f"dup.{t}": v for t, v in s["dup_rows"].items() if v})
    return out


def pass1(inv: pl.DataFrame, days: list[str], p1: Path, workers: int, force: bool, commit: str) -> int:
    by_day = {d: g.sort("snapshot_id").to_dicts() for (d,), g in inv.group_by("day")}
    todo = []
    for d in days:
        dest = p1 / d[:4] / d
        ids = [s["snapshot_id"] for s in by_day[d]]
        if not force and (dest / "_day.json").exists():
            try:
                if json.loads((dest / "_day.json").read_text())["snapshots"] == ids:
                    continue
            except (OSError, ValueError, KeyError):
                pass
        todo.append((d, by_day[d], str(dest), commit))
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
                bad = _day_flags(s)
                el = time.time() - t0
                log.info("day %s: %d snapshots, %d contracts, %d item rows in, %.0f s%s; %d/%d, "
                         "eta %.0f min", s["day"], len(s["snapshots"]), s["contracts"],
                         s["rows_in"]["contract_items"], s["seconds"],
                         f"; flags {bad}" if bad else "", done, len(todo),
                         el / done * (len(todo) - done) / 60)
            except Exception as e:
                failed += 1
                log.error("FAILED %s: %r", futs[f], e)
    return failed


# ---------------------------------------------------------------- pass 2

def _merge_ranges(lf: pl.LazyFrame, key: list[str], cols: list[str]) -> pl.DataFrame:
    """Day rows -> one row per key over the window; the first day's values kept. Flags: keys
    whose values (stamp apart) or stamp differ between days, and keys with a gap."""
    vals = [c for c in cols if c != STAMP]
    g = (lf.sort("first_snapshot").group_by(key).agg(
        *[pl.col(c).first() for c in cols],
        pl.col("first_snapshot").min(), pl.col("last_snapshot").max(), pl.col("n").sum(),
        (pl.struct(vals).n_unique() if vals else pl.lit(1, pl.UInt32)).alias("_v"),
        (pl.col(STAMP).n_unique() if STAMP in cols else pl.lit(1, pl.UInt32)).alias("_st"))
        .collect())
    flags = {"vary": g.filter(pl.col("_v") > 1).height, "stamp": g.filter(pl.col("_st") > 1).height,
             "gaps": g.filter(pl.col("last_snapshot") - pl.col("first_snapshot") + 1 != pl.col("n")).height}
    return g.drop("_v", "_st"), flags


def _merge_spells(lf: pl.LazyFrame) -> pl.DataFrame:
    o = _spell_ids(lf.sort("contract_id", "from_snapshot").collect(), "contract_id",
                   "from_snapshot", "to_snapshot", SPELL_VALS)
    return (o.group_by("_spell", maintain_order=True)
            .agg(pl.col("contract_id").first(), pl.col("from_snapshot").min(),
                 pl.col("to_snapshot").max(), *[pl.col(v).first() for v in SPELL_VALS])
            .drop("_spell"))


def _add(acc: dict, name: str, f: dict) -> None:
    for k, v in f.items():
        acc[f"{name}.{k}"] = acc.get(f"{name}.{k}", 0) + v


def pass2(p1: Path, out: Path, days: list[str], nb: int, commit: str) -> dict:
    t0 = time.time()
    dirs = [p1 / d[:4] / d for d in days]
    infos = [json.loads((d / "_day.json").read_text()) for d in dirs]
    pv = {i["polars"] for i in infos}
    if len(pv) > 1:  # terms_hash must come from one Polars version
        raise RuntimeError(f"pass 1 days made with different Polars versions {pv}; rerun with --force")

    def scan(name: str) -> pl.LazyFrame:
        return pl.scan_parquet([str(d / f"{name}.parquet") for d in dirs])

    snaps = scan("snapshots").sort("snapshot_id").collect()
    for t in TABLES:
        shutil.rmtree(out / t, ignore_errors=True)
        (out / t).mkdir(parents=True)
    snaps.write_parquet(out / "snapshots.parquet")
    stats: dict = {t: 0 for t in TABLES}
    fl: dict = {}
    terms = [x for x in CONTRACT_TERMS if x != "contract_id"]
    for b in range(nb):
        inb = (pl.col("contract_id") % nb) == b
        part = f"part-{b:02d}.parquet"
        # contracts: versions numbered by first sighting
        con, f = _merge_ranges(scan("contracts").filter(inb), ["contract_id", "terms_hash"], terms)
        _add(fl, "contracts", {"gaps": f["gaps"]})
        con = con.sort("contract_id", "first_snapshot").with_columns(
            terms_version=pl.int_range(pl.len()).over("contract_id").cast(pl.Int16),
            n_terms=pl.len().over("contract_id").cast(pl.Int16))
        sp = _merge_spells(scan("spells").filter(inb)).join(
            con.select("contract_id", "terms_hash", "terms_version"), on=["contract_id", "terms_hash"],
            how="left")
        _add(fl, "contract_changes", {"no_terms": sp.filter(pl.col("terms_version").is_null()).height})
        sp = sp.select("contract_id", "from_snapshot", "to_snapshot", "terms_version", "date_expired",
                       "universe_id", STAMP, "bids_http_last_modified").sort("contract_id", "from_snapshot")
        nch = sp.group_by("contract_id").agg(pl.len().cast(I32).alias("n_changes"))
        con = (con.join(nch, on="contract_id", how="left").rename({"n": "n_snapshots"})
               .select(["contract_id", "terms_version"] + terms +
                       ["first_snapshot", "last_snapshot", "n_snapshots", "n_terms", "n_changes"])
               .sort("contract_id", "terms_version"))
        con.write_parquet(out / "contracts" / part)
        sp.write_parquet(out / "contract_changes" / part)
        # items, with the dynamic and non-dynamic rows matched on (contract_id, item_id)
        ik = KEYS["contract_items"]
        items, f = _merge_ranges(scan("contract_items").filter(inb), ik, ITEM_COLS)
        _add(fl, "items", f)
        dyn, f = _merge_ranges(scan("contract_dynamic_items").filter(inb), ["contract_id", "item_id"], DYN_COLS)
        _add(fl, "dynamic", f)
        nd, f = _merge_ranges(scan("contract_non_dynamic_items").filter(inb), ["contract_id", "item_id"], ["type_id"])
        _add(fl, "non_dynamic", f)
        withid = items.filter(pl.col("item_id").is_not_null())
        _add(fl, "items", {"item_id_dup": int(withid.select("contract_id", "item_id").is_duplicated().sum())})
        dyn = dyn.rename({STAMP: "dynamic_http_last_modified", "first_snapshot": "dynamic_first_snapshot",
                          "last_snapshot": "dynamic_last_snapshot"}).drop("n")
        nd = nd.rename({"type_id": "_nd_type", "first_snapshot": "non_dynamic_first_snapshot",
                        "last_snapshot": "non_dynamic_last_snapshot"}).drop("n")
        for name, x in (("dynamic", dyn), ("non_dynamic", nd)):
            _add(fl, name, {"unmatched": x.join(withid, on=["contract_id", "item_id"], how="anti").height})
        items = (items.join(dyn, on=["contract_id", "item_id"], how="left")
                 .join(nd, on=["contract_id", "item_id"], how="left"))
        _add(fl, "non_dynamic", {"type_differs": items.filter(
            pl.col("_nd_type").is_not_null() & (pl.col("_nd_type") != pl.col("type_id"))).height})
        items = items.drop("_nd_type", "n").sort("contract_id", "record_id")
        items.write_parquet(out / "items" / part)
        att, f = _merge_ranges(scan("item_attributes").filter(inb),
                               ["contract_id", "item_id", "kind", "dogma_id"], ["value", "is_default"])
        _add(fl, "item_attributes", f)
        att = att.drop("n").sort("contract_id", "item_id", "kind", "dogma_id")
        att.write_parquet(out / "item_attributes" / part)
        bids, f = _merge_ranges(scan("contract_bids").filter(inb), ["bid_id"], ["contract_id", "amount", "date_bid"])
        _add(fl, "bids", f)
        bids = bids.drop("n").sort("contract_id", "bid_id").select(
            "bid_id", "contract_id", "amount", "date_bid", "first_snapshot", "last_snapshot")
        bids.write_parquet(out / "bids" / part)
        for t, df in (("contracts", con), ("contract_changes", sp), ("items", items),
                      ("item_attributes", att), ("bids", bids)):
            stats[t] += df.height
        log.info("bucket %d/%d: %d contract versions, %d spells, %d items, %d dogma rows, %d bids",
                 b + 1, nb, con.height, sp.height, items.height, att.height, bids.height)
    day_flags: dict = {}
    for i in infos:
        for k, v in _day_flags(i).items():
            day_flags[k] = day_flags.get(k, 0) + v
    summary = {"days": len(days), "from": days[0], "to": days[-1], "snapshots": snaps.height,
               "rows": stats, "pass2_flags": {k: v for k, v in fl.items() if v},
               "pass1_flags": day_flags,
               "raw_rows": {t: int(snaps[f"n_{t}"].sum()) for t in RAW_TABLES},
               "build_commit": commit, "polars": pv.pop(), "seconds": round(time.time() - t0, 1)}
    (out / "_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# ---------------------------------------------------------------- rebuild check

def rebuild(out: Path, sids: list[int]) -> dict[int, dict[str, pl.DataFrame]]:
    """Rebuild the raw tables of the given snapshots from the consolidated tables."""
    lo, hi = min(sids), max(sids)

    def scan(t):
        return pl.scan_parquet(str(out / t / "*.parquet"))

    snaps = pl.read_parquet(out / "snapshots.parquet").filter(pl.col("snapshot_id").is_in(sids))
    ch = scan("contract_changes").filter((pl.col("from_snapshot") <= hi) & (pl.col("to_snapshot") >= lo)).collect()
    ids = ch["contract_id"].unique().implode()
    con = scan("contracts").filter(pl.col("contract_id").is_in(ids)).collect()

    def ov(f, l):
        return (pl.col(f) <= hi) & (pl.col(l) >= lo) & pl.col("contract_id").is_in(ids)

    items = scan("items").filter(ov("first_snapshot", "last_snapshot")
                                 | ov("dynamic_first_snapshot", "dynamic_last_snapshot")
                                 | ov("non_dynamic_first_snapshot", "non_dynamic_last_snapshot")).collect()
    att = scan("item_attributes").filter(ov("first_snapshot", "last_snapshot")).collect()
    bids = scan("bids").filter(ov("first_snapshot", "last_snapshot")).collect()
    res = {}
    for row in snaps.iter_rows(named=True):
        s = row["snapshot_id"]

        def inr(f, l):
            return (pl.col(f) <= s) & (pl.col(l) >= s)

        here = ch.filter(inr("from_snapshot", "to_snapshot"))
        reg = pl.DataFrame(row["regions"], schema={"region_id": I32, STAMP: UTC, "contracts": I32})
        c = (here.select("contract_id", "terms_version", "date_expired", "universe_id",
                         pl.col(STAMP).alias("_o"))
             .join(con, on=["contract_id", "terms_version"], how="left")
             .join(reg.select("region_id", STAMP), on="region_id", how="left")
             .with_columns(pl.coalesce("_o", STAMP).alias(STAMP)))
        cin = here.select("contract_id")
        it = items.filter(inr("first_snapshot", "last_snapshot")).join(cin, on="contract_id")
        dy = (items.filter(inr("dynamic_first_snapshot", "dynamic_last_snapshot")).join(cin, on="contract_id")
              .drop(STAMP).rename({"dynamic_http_last_modified": STAMP}))
        nd = items.filter(inr("non_dynamic_first_snapshot", "non_dynamic_last_snapshot")).join(cin, on="contract_id")
        a = att.filter(inr("first_snapshot", "last_snapshot")).join(
            dy.select("contract_id", "item_id", STAMP), on=["contract_id", "item_id"])
        bd = bids.filter(inr("first_snapshot", "last_snapshot")).join(
            here.select("contract_id", pl.col("bids_http_last_modified").alias(STAMP)), on="contract_id")
        frames = {
            "contracts": c, "contract_items": it, "contract_bids": bd, "contract_dynamic_items": dy,
            "contract_dynamic_items_dogma_attributes": a.filter(pl.col("kind") == "attribute")
                .rename({"dogma_id": "attribute_id"}),
            "contract_dynamic_items_dogma_effects": a.filter(pl.col("kind") == "effect")
                .rename({"dogma_id": "effect_id"}),
            "contract_non_dynamic_items": nd,
        }
        res[s] = {t: df.select([pl.col(k).cast(v) for k, v in SCHEMAS[t].items()])
                  for t, df in frames.items()}
    return res


def _compare(raw: pl.DataFrame, reb: pl.DataFrame) -> dict:
    raw = raw.unique()
    cols = raw.columns
    oraw = raw.join(reb, on=cols, how="anti", nulls_equal=True)
    oreb = reb.join(raw, on=cols, how="anti", nulls_equal=True)
    nost = [c for c in cols if c != STAMP]
    craw = oraw.join(reb, on=nost, how="anti", nulls_equal=True).height
    creb = oreb.join(raw, on=nost, how="anti", nulls_equal=True).height
    return {"raw": raw.height, "rebuilt": reb.height, "only_raw": oraw.height,
            "only_rebuilt": oreb.height, "content_raw": craw, "content_rebuilt": creb,
            "stamp_only": oraw.height - craw}


def check_day(out: str, snaps: list[dict]) -> dict:
    t0 = time.time()
    reb = rebuild(Path(out), [s["snapshot_id"] for s in snaps])
    per, content, stamp = [], 0, 0
    for sn in snaps:
        _, fr = read_snapshot(sn["path"], sn["sha256"])
        r = {"snapshot_id": sn["snapshot_id"], "rel": sn["rel"]}
        for t in RAW_TABLES:
            cmp = _compare(fr[t], reb[sn["snapshot_id"]][t])
            r[t] = cmp
            content += cmp["content_raw"] + cmp["content_rebuilt"] + abs(
                cmp["raw"] - cmp["rebuilt"])
            stamp += cmp["stamp_only"]
        per.append(r)
    return {"day": snaps[0]["day"], "snapshots": per, "content_mismatches": content,
            "stamp_only": stamp, "seconds": round(time.time() - t0, 1)}


CMP_KEYS = ("raw", "rebuilt", "only_raw", "only_rebuilt", "content_raw", "content_rebuilt", "stamp_only")


def check(inv: pl.DataFrame, out: Path, days: list[str], every: int, workers: int) -> dict:
    built = pl.read_parquet(out / "snapshots.parquet", columns=["snapshot_id"])["snapshot_id"].implode()
    sel = inv.filter(pl.col("snapshot_id").is_in(built))
    if days:
        sel = sel.filter(pl.col("day").is_in(days))
    if every > 1:
        sel = sel.filter(pl.col("snapshot_id") % every == 0)
    groups = [g.sort("snapshot_id").to_dicts() for _, g in sel.group_by("day")]
    log.info("check: %d snapshots on %d days", sel.height, len(groups))
    results, failed = [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(check_day, str(out), g): g[0]["day"] for g in groups}
        for f in as_completed(futs):
            try:
                r = f.result()
                results.append(r)
                log.info("check %s: %d snapshots, %d content mismatches, %d stamp-only, %.0f s",
                         r["day"], len(r["snapshots"]), r["content_mismatches"], r["stamp_only"],
                         r["seconds"])
            except Exception as e:
                failed.append(futs[f])
                log.error("check FAILED %s: %r", futs[f], e)
    tot = {t: {k: sum(s[t][k] for r in results for s in r["snapshots"]) for k in CMP_KEYS}
           for t in RAW_TABLES}
    return {"snapshots": sum(len(r["snapshots"]) for r in results), "days": len(results),
            "failed_days": sorted(failed),
            "content_mismatches": sum(r["content_mismatches"] for r in results),
            "stamp_only": sum(r["stamp_only"] for r in results),
            "stamp_only_days": sorted(r["day"] for r in results if r["stamp_only"]),
            "tables": tot, "per_day": sorted(results, key=lambda r: r["day"])}


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--raw", default="raw/eve")
    ap.add_argument("--out", default="tmp/eve/build/contracts")
    ap.add_argument("--pass1", default=None, help="pass 1 folder (default <out>/pass1)")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", dest="end", default=None)
    ap.add_argument("--days", default=None, help="comma-separated days (instead of a range)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pass", dest="which", choices=["1", "2", "both"], default="both")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--buckets", type=int, default=32, help="contract_id buckets in pass 2")
    ap.add_argument("--every", type=int, default=1, help="check: every k-th snapshot id")
    ap.add_argument("--json", default=None, help="check: write the result here")
    ap.add_argument("--stamp-ok-until", default="2021-07-05",
                    help="check: stamp-only differences on or before this day pass (the refetch weeks)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out, commit = Path(a.out), commit_id()
    p1 = Path(a.pass1) if a.pass1 else out / "pass1"
    inv = inventory(Path(a.raw))
    all_days = sorted(inv["day"].unique())
    if a.days:
        days = sorted(a.days.split(","))
        if set(days) - set(all_days):
            log.error("days not in the inventory: %s", sorted(set(days) - set(all_days)))
            return 1
    else:
        days = [d for d in all_days if (not a.start or d >= a.start) and (not a.end or d <= a.end)]
    log.info("inventory: %d snapshots on %d days in %d torrents (%s..%s); selected %d days",
             inv.height, len(all_days), inv["torrent"].n_unique(), all_days[0], all_days[-1], len(days))
    if a.cmd == "check":
        r = check(inv, out, days if (a.days or a.start or a.end) else [], a.every, a.workers)
        r["build_commit"] = json.loads((out / "_summary.json").read_text()).get("build_commit")
        r["check_commit"] = commit
        late = [d for d in r["stamp_only_days"] if d > a.stamp_ok_until]
        r["stamp_only_late_days"] = late
        if a.json:
            Path(a.json).write_text(json.dumps(r, indent=1, default=str))
        log.info("check: %d snapshots, %d days (%d failed); content mismatches %d; stamp-only %d "
                 "(on %d days, %d after %s)", r["snapshots"], r["days"], len(r["failed_days"]),
                 r["content_mismatches"], r["stamp_only"], len(r["stamp_only_days"]), len(late),
                 a.stamp_ok_until)
        for t, v in r["tables"].items():
            log.info("  %s: %s", t, v)
        ok = r["content_mismatches"] == 0 and not r["failed_days"] and not late
        log.info("check %s", "PASSED" if ok else "FAILED")
        return 0 if ok else 1
    if a.which in ("1", "both"):
        if pass1(inv, days, p1, a.workers, a.force, commit):
            log.error("pass 1 had failures; pass 2 not run")
            return 1
    if a.which in ("2", "both"):
        s = pass2(p1, out, days, a.buckets, commit)
        log.info("pass 2: %s", json.dumps({k: s[k] for k in ("snapshots", "rows", "pass2_flags",
                                                              "pass1_flags", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
