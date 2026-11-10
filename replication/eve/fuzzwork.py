# SPDX-License-Identifier: GPL-3.0-or-later
"""D6 v2: the Fuzzwork market ordersets in EVE Ref's Deep Archive, read straight from raw.

Inventory and reader for ``fuzzwork/ordersets/`` in every torrent under ``raw/eve/``
(``everef-eve-online-deep-archive-NN/``; the torrents unpack into one tree and no path is in
two of them). Each orderset is one market snapshot, a headerless tab-separated ``.csv.gz``:

  order_id, type_id, issued, is_buy, volume_remain, volume_total, min_volume, price,
  location_id, range, duration, region_id, orderset_id

(``volume_remain`` before ``volume_total``: checked against EVE Ref's snapshot 10 minutes
later, every matched order agrees on volume_total and 99.8% on volume_remain.)

Snapshot time: the file name's time for dated files (``fuzzwork-orderset-<id>-<date>_<HH-MM-SS>``);
for ``backfills/`` (named by id only) the ``last_modified`` time in the torrent's index
snapshot (``-index.tar.xz``), which equals the name's time on every dated file. The archive is
read as data; nothing in it is run.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

COLUMNS = ["order_id", "type_id", "issued", "is_buy_order", "volume_remain", "volume_total",
           "min_volume", "price", "location_id", "range", "duration", "region_id", "orderset_id"]
UTC = pl.Datetime("us", "UTC")
INTS = {"order_id": pl.Int64, "type_id": pl.Int32, "volume_remain": pl.Int64,
        "volume_total": pl.Int64, "min_volume": pl.Int64, "location_id": pl.Int64,
        "duration": pl.Int32, "region_id": pl.Int32, "orderset_id": pl.Int32}
SCHEMA = {c: INTS.get(c, pl.Float64 if c == "price" else pl.String) for c in COLUMNS}
TORRENT = re.compile(r"everef-eve-online-deep-archive-(\d{2})$")
DATED = re.compile(r"fuzzwork-orderset-(\d+)-(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})\.csv\.gz$")
BACKFILL = re.compile(r"orderset-(\d+)\.csv\.gz$")
PREFIX = "fuzzwork/ordersets/"


def torrents(raw: Path) -> list[Path]:
    return sorted(p for p in raw.iterdir() if p.is_dir() and TORRENT.match(p.name))


def _index_times(tdir: Path) -> dict[str, str]:
    """rel path -> last_modified, from the torrent's index snapshot (Fuzzwork folders only)."""
    out: dict[str, str] = {}
    idx = next(tdir.glob("*-index.tar.xz"), None)
    if idx is None:
        return out
    with tarfile.open(idx, "r:xz") as tf:
        for m in tf:
            if not (m.isfile() and m.name.startswith(PREFIX) and m.name.endswith("index.json")):
                continue
            folder = m.name[: -len("index.json")]
            for e in json.load(tf.extractfile(m))["files"]:
                out[folder + e["name"]] = e["last_modified"]
    return out


def inventory(raw: Path) -> pl.DataFrame:
    """One row per orderset: orderset_id, snap_time, day, torrent, rel, path, sha256, source."""
    rows = []
    for tdir in torrents(raw):
        sums = [ln.split("  ", 1) for ln in (tdir / f"{tdir.name}.sha256").read_text().splitlines()]
        sums = [(d, r) for d, r in sums if r.startswith(PREFIX)]
        times = _index_times(tdir) if any("/backfills/" in r for _, r in sums) else {}
        for digest, rel in sums:
            if m := DATED.search(rel):
                oid = int(m.group(1))
                t = datetime.fromisoformat(f"{m.group(2)}T{m.group(3)}:{m.group(4)}:{m.group(5)}")
                src = "dated"
            elif m := BACKFILL.search(rel):
                oid = int(m.group(1))
                if rel not in times:
                    raise ValueError(f"{tdir.name}: no index time for {rel}")
                t = datetime.fromisoformat(times[rel].rstrip("Z"))
                src = "backfill"
            else:
                raise ValueError(f"{tdir.name}: unexpected Fuzzwork file {rel}")
            rows.append((oid, t.replace(tzinfo=timezone.utc), TORRENT.match(tdir.name).group(1),
                         rel, str(tdir / rel), digest, src))
    df = pl.DataFrame(rows, schema={"orderset_id": pl.Int32, "snap_time": UTC, "torrent": pl.String,
                                    "rel": pl.String, "path": pl.String, "sha256": pl.String,
                                    "source": pl.String}, orient="row")
    if df["orderset_id"].is_duplicated().any():
        raise ValueError("an orderset id appears twice in the archive")
    df = df.sort("orderset_id").with_columns(day=pl.col("snap_time").dt.date())
    # Snapshot order is the id (EVE Ref's sequence number). A time is when the file was written:
    # near downtime (09:00-11:00 UTC) a snapshot sometimes finishes late, so time order and id
    # order disagree by up to ~15 minutes (100 of 117,884), and 744 pairs share a time (different
    # files, identical rows). Anything worse means the inventory is wrong.
    back = df.select(pl.col("snap_time").diff().dt.total_seconds().min()).item()
    if back is not None and back < -1800:
        raise ValueError(f"snapshot time goes back {-back} s between consecutive ids")
    return df


def read_orderset(path: str, digest: str | None, orderset_id: int) -> pl.DataFrame:
    """Read one orderset strictly: hash, 13 columns, every value parsed, one orderset id."""
    data = Path(path).read_bytes()
    if digest is not None and (got := hashlib.sha256(data).hexdigest()) != digest:
        raise ValueError(f"sha256 mismatch for {path}: {got} != {digest}")
    # Polars decompresses the gzip itself (3x faster than the gzip module; same frame on every
    # file tested). Typed read: a value that doesn't parse, or a short or long row, raises.
    df = pl.read_csv(data, separator="\t", has_header=False, schema=SCHEMA, quote_char=None)
    del data
    df = df.with_columns(
        pl.col("issued").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_unit="us", time_zone="UTC",
                                         strict=True),
        pl.col("is_buy_order").replace_strict({"True": True, "False": False}, return_dtype=pl.Boolean),
    )
    if df.height == 0:
        raise ValueError(f"{path} has no rows")
    if (n := df.null_count().sum_horizontal().item()):
        raise ValueError(f"{path}: {n} null values")
    ids = df["orderset_id"].unique()
    if ids.len() != 1 or ids[0] != orderset_id:
        raise ValueError(f"{path}: orderset_id column {ids.to_list()[:5]}, expected {orderset_id}")
    return df
