# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Out-of-core preparation: encode fixed effects and clusters, check nesting,
and write the data as Parquet shards partitioned by a key.

Everything runs on Polars LazyFrames with the streaming engine, so the full
data never has to fit in memory. Shards go to `shard_dir`, which should be on a
filesystem every worker can read (e.g. a shared scratch/temp filesystem).
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .formula import fe_column, fe_components
from .rowhash import ALGORITHM as ROWSET
from .rowhash import column_bits, hexdigest, rowset

log = logging.getLogger("dfeols")

META = "dfeols_meta.json"
WORK_DIRS = ("shards", "_staging", "_buckets", "_codes")


def remove_dir(path: Path) -> None:
    """Delete a folder by renaming it out of the way first, so a new one can be
    created at once and a slow delete on a parallel filesystem cannot race
    with new writes ("Directory not empty" when a straggler still writes)."""
    if not path.exists():
        return
    trash = path.parent / f".{path.name}.deleting-{uuid.uuid4().hex[:8]}"
    os.replace(path, trash)
    shutil.rmtree(trash, ignore_errors=True)
    if trash.exists():
        log.warning("prep: could not fully delete %s; remove it by hand", trash)


def clear_incomplete(root: Path) -> None:
    """Remove what an interrupted prep left in `root` (shards, staging, code
    tables). Without the metadata file the directory is not a dataset, and
    stale piece files would otherwise be picked up as part of the new one."""
    if (root / META).exists():
        return
    stale = [root / d for d in WORK_DIRS if (root / d).exists()]
    if stale:
        log.warning("prep: %s has no %s (an earlier prep did not finish); clearing it", root, META)
        for d in stale:
            remove_dir(d)


@dataclass
class Dataset:
    """A prepared, sharded dataset. Open an existing one with Dataset.open()."""

    root: Path
    partition: list[str]
    shards: list[dict]  # {"id", "files", "rows"}
    fes: dict[str, dict]  # name -> {"column", "n_levels", "nested_in_partition"}
    clusters: dict[str, dict]  # name -> same, used for vcov
    columns: list[str]
    n_rows: int
    meta: dict = field(default_factory=dict)

    @classmethod
    def open(cls, root: str | os.PathLike) -> Dataset:
        root = Path(root)
        m = json.loads((root / META).read_text())
        return cls(
            root=root,
            partition=m["partition"],
            shards=m["shards"],
            fes=m["fes"],
            clusters=m["clusters"],
            columns=m["columns"],
            n_rows=m["n_rows"],
            meta=m,
        )

    def shard_paths(self, shard: dict) -> list[str]:
        return [str(self.root / f) for f in shard["files"]]

    def scan(self, columns: list[str] | None = None) -> pl.LazyFrame:
        files = [str(self.root / f) for s in self.shards for f in s["files"]]
        lf = pl.scan_parquet(files)
        return lf.select(columns) if columns else lf

    def __repr__(self) -> str:
        return (
            f"Dataset({self.root}, rows={self.n_rows:,}, shards={len(self.shards)}, "
            f"partition={self.partition}, fes={list(self.fes)}, clusters={list(self.clusters)})"
        )


def _scan(source) -> pl.LazyFrame:
    if isinstance(source, pl.LazyFrame):
        return source
    if isinstance(source, pl.DataFrame):
        return source.lazy()
    try:
        import pandas as pd

        if isinstance(source, pd.DataFrame):
            return pl.from_pandas(source).lazy()
    except ImportError:
        pass
    s = str(source)
    if s.endswith(".csv") or s.endswith(".csv.gz"):
        return pl.scan_csv(s)
    return pl.scan_parquet(s)


def _lpt_bins(sizes: np.ndarray, n_bins: int) -> np.ndarray:
    """Longest-processing-time bin packing. Returns the bin of each item."""
    order = np.argsort(-sizes, kind="stable")
    heap = [(0, b) for b in range(n_bins)]
    heapq.heapify(heap)
    out = np.empty(len(sizes), dtype=np.int64)
    for i in order:
        load, b = heapq.heappop(heap)
        out[i] = b
        heapq.heappush(heap, (load + int(sizes[i]), b))
    return out


def _nested(lf: pl.LazyFrame, inner: str, outer: list[str]) -> bool:
    """True if every level of column `inner` maps to one combination of `outer`."""
    outer_key = pl.struct(outer) if len(outer) > 1 else pl.col(outer[0])
    k = (
        lf.group_by(inner)
        .agg(outer_key.n_unique().alias("_k"))
        .select(pl.col("_k").max())
        .collect(engine="streaming")
        .item()
    )
    return k is None or k <= 1


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def source_info(source, checksum: bool = True) -> dict:
    """What prep read, for dfeols_meta.json and the records' provenance.data: the files
    (name, path, bytes, sha256, Parquet row count) and one checksum: the file's sha256,
    for several files the sha256 of their `sha256sum` lines (sorted by name), or, for an
    in-memory frame, a dfeols-rowset-v1 content hash. A LazyFrame gets no checksum."""
    if isinstance(source, pl.LazyFrame):
        return {"kind": "lazyframe"}
    if isinstance(source, pl.DataFrame) or hasattr(source, "to_numpy"):
        df = source if isinstance(source, pl.DataFrame) else pl.from_pandas(source)
        info: dict = {"kind": "frame", "rows": df.height, "columns": len(df.columns)}
        if checksum:
            cols = []
            for c in df.columns:
                try:
                    column_bits(df[c].head(1))
                    cols.append(c)
                except TypeError:
                    pass
            info["checksum"] = {
                "algorithm": ROWSET,
                "value": hexdigest(rowset(df, cols)),
                "of": "in-memory",
                "method": ROWSET,
            }
            if len(cols) < len(df.columns):
                info["checksum"]["columns"] = cols
        return info
    from .prep_dist import _files

    try:
        files = _files(source)
    except ValueError:  # not Parquet: a single CSV file
        files = [str(source)]
    t0 = time.time()
    entries = []
    for f in files:
        p = Path(f).resolve()
        e: dict = {"name": p.name, "path": str(p), "bytes": p.stat().st_size}
        if checksum:
            e["sha256"] = _sha256_file(str(p))
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq

            e["rows"] = pq.read_metadata(str(p)).num_rows
        entries.append(e)
    info = {"kind": "files", "files": entries}
    if all("rows" in e for e in entries):
        info["rows"] = sum(e["rows"] for e in entries)
    if checksum:
        if len(entries) == 1:
            info["checksum"] = {"algorithm": "sha256", "value": entries[0]["sha256"], "of": "file"}
        else:
            lines = "".join(f"{e['sha256']}  {e['name']}\n" for e in sorted(entries, key=lambda e: e["name"]))
            info["checksum"] = {
                "algorithm": "sha256",
                "value": hashlib.sha256(lines.encode()).hexdigest(),
                "of": "manifest",
                "method": "sha256 of the files' sha256sum lines (hash, two spaces, base name), "
                "sorted by name",
            }
        log.info("prep: sha256 of %d source file(s) in %.0fs", len(entries), time.time() - t0)
    return info


def prep(
    source,
    *,
    shard_dir: str | os.PathLike,
    partition: str | list[str],
    fixed_effects: list[str] = (),
    clusters: list[str] = (),
    keep: list[str] | None = None,
    target_rows: int = 1_000_000,
    overwrite: bool = False,
    keep_staging: bool = False,
    executor=None,
    checksum: bool = True,
    **dist_options,
) -> Dataset:
    """Encode FEs and clusters and write sharded Parquet.

    source         Parquet path or glob, CSV path, or a Polars/pandas frame.
    shard_dir      Output directory. Put it on the shared scratch/temp filesystem.
    partition      Key whose groups never split across shards (e.g. "id_a").
                   FEs and clusters nested in this key are handled locally.
    fixed_effects  FE specs to encode, e.g. ["spell", "week", "id_a^week"].
                   Any FE used later in fit() must be listed here.
    clusters       Cluster variables to encode, e.g. ["id_a"].
    keep           Data columns to keep (default: all).
    target_rows    Approximate rows per shard. Peak worker memory is about
                   rows x (outcomes + regressors + dummies) x 8 bytes x 3.
    checksum       record the source's sha256 (files) or content hash (frames) in
                   dfeols_meta.json; estimation records carry it
    executor       run prep as map/reduce rounds on this executor's workers
                   (Parquet input). Use it for data too large to stream on the
                   coordinator; output is identical in structure. Extra keyword
                   options (rows_per_task, max_pieces) tune the distributed path.
    """
    from .lock import DirLock

    with DirLock(Path(shard_dir), what="shard folder"):
        return _prep(
            source,
            shard_dir=shard_dir,
            partition=partition,
            fixed_effects=fixed_effects,
            clusters=clusters,
            keep=keep,
            target_rows=target_rows,
            overwrite=overwrite,
            keep_staging=keep_staging,
            executor=executor,
            checksum=checksum,
            **dist_options,
        )


def _prep(
    source,
    *,
    shard_dir,
    partition,
    fixed_effects,
    clusters,
    keep,
    target_rows,
    overwrite,
    keep_staging,
    executor,
    checksum=True,
    **dist_options,
) -> Dataset:
    if (
        executor is not None
        and not isinstance(source, (pl.DataFrame, pl.LazyFrame))
        and not hasattr(source, "to_numpy")
    ):
        from .prep_dist import prep_distributed

        return prep_distributed(
            source,
            shard_dir=shard_dir,
            partition=partition,
            fixed_effects=list(fixed_effects),
            clusters=list(clusters),
            keep=keep,
            target_rows=target_rows,
            overwrite=overwrite,
            executor=executor,
            source_meta=source_info(source, checksum),
            **dist_options,
        )
    t_start = time.time()
    source_meta = source_info(source, checksum)
    root = Path(shard_dir)
    if (root / META).exists() and not overwrite:
        raise FileExistsError(f"{root} already holds a prepared dataset (overwrite=True to replace)")
    if overwrite:
        remove_dir(root)
    clear_incomplete(root)
    root.mkdir(parents=True, exist_ok=True)

    partition = [partition] if isinstance(partition, str) else list(partition)
    lf = _scan(source)
    schema = lf.collect_schema()
    if keep is not None:
        need = set(keep) | set(partition)
        for spec in list(fixed_effects) + list(clusters):
            need.update(fe_components(spec))
        lf = lf.select([c for c in schema.names() if c in need])
        schema = lf.collect_schema()

    key_cols = set(partition)
    specs: dict[str, list[str]] = {}
    for spec in list(fixed_effects) + list(clusters):
        comps = fe_components(spec)
        key_cols.update(comps)
        specs["^".join(comps)] = comps
    missing = key_cols - set(schema.names())
    if missing:
        raise KeyError(f"Columns not in data: {sorted(missing)}")

    lf = lf.drop_nulls(subset=sorted(key_cols))

    # Global integer codes for each FE / cluster spec (exact; no hashing).
    for name, comps in specs.items():
        col = fe_column(name)
        codes = lf.select(comps).unique().with_row_index(col).with_columns(pl.col(col).cast(pl.UInt32))
        lf = lf.join(codes, on=comps, how="inner")

    staging = root / "_staging"
    staging.mkdir(exist_ok=True)
    stage_file = staging / "encoded.parquet"
    log.info("prep: encoding %d FE/cluster specs and staging data", len(specs))
    lf.sink_parquet(str(stage_file), engine="streaming")
    st = pl.scan_parquet(str(stage_file))
    n_rows = st.select(pl.len()).collect().item()
    log.info("prep: staged %s rows in %.0fs", f"{n_rows:,}", time.time() - t_start)

    # Nesting checks and level counts.
    fe_meta: dict[str, dict] = {}
    for name in specs:
        col = fe_column(name)
        n_levels = int(st.select(pl.col(col).max()).collect().item()) + 1
        fe_meta[name] = {
            "column": col,
            "components": specs[name],
            "n_levels": n_levels,
            "nested_in_partition": _nested(st, col, partition),
        }
    cl_names = ["^".join(fe_components(c)) for c in clusters]
    for cname in cl_names:
        for fname in ["^".join(fe_components(f)) for f in fixed_effects]:
            fe_meta[fname].setdefault("nested_in_cluster", {})[cname] = _nested(
                st, fe_meta[fname]["column"], [fe_meta[cname]["column"]]
            )
    log.info("prep: nesting checks done (%.0fs)", time.time() - t_start)

    # Assign partition groups to shards (LPT bin packing on row counts).
    counts = st.group_by(partition).agg(pl.len().cast(pl.Int64).alias("_n")).collect(engine="streaming")
    n_bins = max(1, int(np.ceil(n_rows / target_rows)))
    bins = _lpt_bins(counts["_n"].to_numpy(), n_bins)
    shard_map = counts.select(partition).with_columns(pl.Series("__shard", bins, dtype=pl.UInt32))
    big = int((counts["_n"] > target_rows).sum())
    if big:
        log.warning("prep: %d partition groups exceed target_rows on their own", big)

    out = root / "shards"
    (
        st.join(shard_map.lazy(), on=partition, how="inner").sink_parquet(
            pl.PartitionBy(str(out), key="__shard", include_key=False), mkdir=True, engine="streaming"
        )
    )

    # Collect shard files and row counts.
    shard_rows = (
        st.join(shard_map.lazy(), on=partition)
        .group_by("__shard")
        .agg(pl.len().alias("rows"))
        .collect(engine="streaming")
    )
    rows_by = dict(zip(shard_rows["__shard"].to_list(), shard_rows["rows"].to_list()))
    shards = []
    for d in sorted(out.iterdir()):
        if not d.is_dir():
            continue
        sid = int(d.name.split("=")[-1])
        files = sorted(str(p.relative_to(root)) for p in d.glob("*.parquet"))
        shards.append({"id": sid, "files": files, "rows": int(rows_by.get(sid, 0))})
    shards.sort(key=lambda s: s["id"])

    if not keep_staging:
        shutil.rmtree(staging, ignore_errors=True)

    columns = [c for c in st.collect_schema().names()]
    from . import __version__

    meta = {
        "version": 1,
        "prep_version": __version__,
        "partition": partition,
        "shards": shards,
        "fes": {k: v for k, v in fe_meta.items() if k in {"^".join(fe_components(f)) for f in fixed_effects}},
        "clusters": {k: fe_meta[k] for k in cl_names},
        "columns": columns,
        "n_rows": int(n_rows),
        "target_rows": target_rows,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prep_seconds": round(time.time() - t_start, 1),
        "source": source_meta,
    }
    (root / META).write_text(json.dumps(meta, indent=1))
    log.info("prep: wrote %d shards in %.0fs", len(shards), time.time() - t_start)
    return Dataset.open(root)
