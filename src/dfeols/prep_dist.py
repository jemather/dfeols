# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Distributed prep: the same output as prep(), built by map/reduce rounds on
the workers, so the coordinator never holds or streams the full data.

  scan    (per input file)    row counts, distinct levels of every FE/cluster
                              spec, partition-group sizes, nesting pairs
  reduce  (coordinator)       global integer codes (sorted levels), LPT shard
                              assignment, nesting flags -> small Parquet tables
  encode  (per file batch)    join the code tables, bucket rows by shard
  merge   (per shard)         concatenate a shard's pieces into one file

The coordinator holds only the code tables (about 8 bytes per distinct level:
10M units is ~80 MB) and per-group row counts.
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from .formula import fe_column, fe_components

log = logging.getLogger("dfeols")

SEED_FE = 11  # hash seed for FE/cluster ids of very large nested specs
SEED_PART = 7  # hash seed for range-based shard assignment


def _files(source) -> list[str]:
    if isinstance(source, (list, tuple)):
        out = [str(s) for s in source]
    else:
        s = str(source)
        out = (
            sorted(_glob.glob(s))
            if any(ch in s for ch in "*?[")
            else (sorted(str(p) for p in Path(s).glob("*.parquet")) if Path(s).is_dir() else [s])
        )
    if not out:
        raise FileNotFoundError(f"no input files match {source!r}")
    bad = [f for f in out if not f.endswith(".parquet")]
    if bad:
        raise ValueError(f"distributed prep reads Parquet; convert first: {bad[:3]}")
    return out


def _nrows(f: str) -> int:
    import pyarrow.parquet as pq

    return pq.read_metadata(f).num_rows


class _Compactor:
    """Accumulates per-file scan frames and reduces them as they arrive.

    Each file returns, e.g., every group x time level it contains; at 1e10
    rows that is 5,000 overlapping copies of a 500k-row table (tens of GB
    held until the scan ended, which got the coordinator killed). Merging
    whenever the pending rows reach the size of what is already merged (and
    at least `min_rows`) keeps memory near the number of distinct keys, at
    O(n log n) total work. Frames whose keys don't overlap across files
    (unit-level ones) don't shrink when merged, so merging stops for them
    after a merge of several files' frames that saves less than 10%."""

    def __init__(self, reduce, min_rows: int = 1_000_000):
        self.reduce, self.min_rows = reduce, min_rows
        self.merged: pl.DataFrame | None = None
        self.pending: list[pl.DataFrame] = []
        self.pending_rows = 0
        self.worthwhile = True

    def add(self, df: pl.DataFrame) -> None:
        self.pending.append(df)
        self.pending_rows += df.height
        merged = self.merged.height if self.merged is not None else 0
        if self.worthwhile and self.pending_rows >= max(self.min_rows, merged):
            self._compact()

    def _compact(self) -> None:
        parts = ([self.merged] if self.merged is not None else []) + self.pending
        before = sum(p.height for p in parts)
        self.merged = self.reduce(pl.concat(parts)) if parts else None
        self.pending, self.pending_rows = [], 0
        # judge only merges across several files (one file's rows are unique anyway)
        if len(parts) >= 8 and self.merged is not None and self.merged.height > 0.9 * before:
            self.worthwhile = False

    def result(self) -> pl.DataFrame:
        if self.pending or self.merged is None:
            self._compact()
        return self.merged


# ------------------------------------------------------------------ tasks
def _scan_task(files, cols, key_cols, specs, partition, checks):
    t = time.perf_counter()
    lf = pl.scan_parquet(files).select(cols).drop_nulls(subset=key_cols)
    q = {"__n": lf.select(pl.len())}
    for name, comps in specs.items():
        q[f"L:{name}"] = lf.select(comps).unique()
    # Int64, not polars' UInt32 len(): row arithmetic on these counts (cumulative
    # sums, x n_bins) exceeds 2**32 at 1e9 rows and would wrap silently.
    q["P"] = lf.group_by(partition).agg(pl.len().cast(pl.Int64).alias("_n"))
    # nesting: per inner level, the min and max of a hash of the outer key.
    # Inner is nested in outer iff min == max for every level across all
    # tasks. Size scales with inner levels per file, never with rows.
    for i, (inner, outer) in enumerate(checks):
        h = pl.struct(outer).hash(seed=0)
        q[f"C:{i}"] = lf.group_by(inner).agg(h.min().alias("lo"), h.max().alias("hi"))
    keys = list(q)
    frames = pl.collect_all([q[k] for k in keys], engine="streaming")
    out = dict(zip(keys, frames))
    return {"n": int(out.pop("__n").item()), "frames": out, "timing": {"total": time.perf_counter() - t}}


def _range_bins(group_rows: np.ndarray, n_rows: int, n_bins: int) -> np.ndarray:
    """Bin of each partition group (in key order) for contiguous range shards:
    a group goes to the bin where its first row falls when rows are cut into
    n_bins equal parts. Computed in int64: at 1e9 rows, rows x n_bins
    overflows 32 bits."""
    sizes = np.asarray(group_rows, dtype=np.int64)
    before = np.cumsum(sizes) - sizes
    if int(sizes.sum()) * n_bins < 2**62:
        out = (before * n_bins) // max(int(n_rows), 1)
    else:  # beyond int64 range: exact Python integers
        out = np.array([b * n_bins // int(n_rows) for b in before.tolist()], dtype=np.int64)
    out = np.minimum(out, n_bins - 1)
    if np.any(np.diff(out) < 0):
        raise RuntimeError("prep: internal error, range shard bins are not monotone")
    return out


def _check_shard_sizes(shard_rows: np.ndarray, group_rows: np.ndarray, target_rows: float, n_bins: int):
    """Guard against a broken shard assignment before writing anything.

    Both assignments (LPT and contiguous ranges) keep every shard within
    target_rows plus one partition group of the target, so a shard far larger
    than that is a bug, not data skew."""
    biggest_group = int(group_rows.max()) if len(group_rows) else 0
    limit = 2 * target_rows + biggest_group
    if len(shard_rows) and shard_rows.max() > limit:
        raise RuntimeError(
            f"prep: shard assignment produced a shard of {int(shard_rows.max()):,} rows "
            f"({len(shard_rows)} shards for {n_bins} bins; target {int(target_rows):,}, largest "
            f"partition group {biggest_group:,}). This is a dfeols bug; please report it."
        )
    if biggest_group > 4 * target_rows:
        log.warning(
            "prep: the largest partition group has %s rows (%.0fx the shard target); it becomes one "
            "oversized shard, and its worker needs memory for all of it",
            f"{biggest_group:,}",
            biggest_group / target_rows,
        )


def _range_key(partition: list[str], schema) -> pl.Expr:
    """Ordering for range-based shard assignment: the partition column itself
    when it is a single numeric or string column (data sorted by that key then
    stays local: each input file touches few shards), else a hash."""
    if len(partition) == 1:
        dt = schema[partition[0]]
        if dt.is_numeric() or dt == pl.String:
            return pl.col(partition[0])
    return pl.struct(partition).hash(seed=SEED_PART)


def _piece_name(sid: int, task_id: int, token: str, i: int) -> str:
    return f"s{sid:06d}_t{task_id:05d}_{token}_{i}.parquet"


def _attempt_token() -> str:
    """Unique per task *execution*. A task can run twice at once (a worker the
    scheduler gave up on keeps writing while the retry runs elsewhere), so
    file names must never collide between attempts. The coordinator keeps only
    the files listed by the attempt whose result it accepted."""
    return uuid.uuid4().hex[:10]


def _sweep(directory: Path, keep: set[str], what: str) -> int:
    """Delete files in `directory` that no accepted task attempt produced."""
    if not directory.exists():
        return 0
    stray = [directory / n for n in os.listdir(directory) if n not in keep]
    if stray:
        with ThreadPoolExecutor(32) as pool:
            list(pool.map(lambda q: q.unlink(missing_ok=True), stray))
        log.warning("prep: removed %d %s files from duplicate or failed task attempts", len(stray), what)
    return len(stray)


def _remove(paths) -> None:
    with ThreadPoolExecutor(32) as pool:
        list(pool.map(lambda q: Path(q).unlink(missing_ok=True), paths))


def _encode_task(
    files,
    task_id,
    root,
    cols,
    key_cols,
    specs,
    partition,
    hashed=(),
    range_mode=False,
    n_buckets=0,
    schema_key=None,
):
    """Encode a batch of files and write one flat piece file per shard it
    touches: shards/s<shard>_t<task>_<attempt>_<i>.parquet (no per-task
    directories, so nothing slow to delete on a parallel filesystem).
    Returns rows and file names per shard, for this attempt only."""
    t = time.perf_counter()
    token = _attempt_token()
    root = Path(root)
    out = root / "shards"
    out.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_parquet(files).select(cols).drop_nulls(subset=key_cols)
    for name, comps in specs.items():
        if name in hashed:  # millions of levels, nested in the partition: hash id, no join
            lf = lf.with_columns(pl.struct(comps).hash(seed=SEED_FE).alias(fe_column(name)))
        else:
            lf = lf.join(
                pl.scan_parquet(str(root / "_codes" / f"{fe_column(name)}.parquet")), on=comps, how="inner"
            )
    if range_mode:  # shards are contiguous ranges of the partition key (or its hash)
        bounds = pl.read_parquet(root / "_codes" / "__shard_bounds.parquet")["b"]
        key = (
            _range_key(partition, schema_key)
            if schema_key is not None
            else pl.struct(partition).hash(seed=SEED_PART)
        )
        lf = lf.with_columns(pl.lit(bounds).search_sorted(key, side="left").cast(pl.UInt32).alias("__shard"))
    else:
        lf = lf.join(pl.scan_parquet(str(root / "_codes" / "__shard_map.parquet")), on=partition, how="inner")

    if n_buckets:
        # two-level shuffle: one file per bucket (a set of shards) per task;
        # the split step later turns each bucket into its shards
        bdir = root / "_buckets"
        bdir.mkdir(exist_ok=True)
        lf = lf.with_columns((pl.col("__shard") % n_buckets).cast(pl.UInt32).alias("__bucket"))

        def bprov(a):
            b = int(a.partition_keys["__bucket"][0])
            return str(bdir / f"b{b:05d}_t{task_id:05d}_{token}_{a.index_in_partition}.parquet")

        lf.sink_parquet(
            pl.PartitionBy(str(bdir), key="__bucket", include_key=False, file_path_provider=bprov),
            mkdir=True,
            engine="streaming",
        )
        rows, names = {}, {}
        for f in sorted(bdir.glob(f"b*_t{task_id:05d}_{token}_*.parquet")):
            k = -1 - int(f.name[1:6])
            rows[k] = rows.get(k, 0) + _nrows(str(f))
            names.setdefault(k, []).append(f.name)
        return {"rows": rows, "files": names, "timing": {"total": time.perf_counter() - t}}

    def provider(a):
        return str(
            out / _piece_name(int(a.partition_keys["__shard"][0]), task_id, token, a.index_in_partition)
        )

    lf.sink_parquet(
        pl.PartitionBy(str(out), key="__shard", include_key=False, file_path_provider=provider),
        mkdir=True,
        engine="streaming",
    )
    rows, names = {}, {}
    for f in sorted(out.glob(f"s*_t{task_id:05d}_{token}_*.parquet")):
        sid = int(f.name[1:7])
        rows[sid] = rows.get(sid, 0) + _nrows(str(f))
        names.setdefault(sid, []).append(f.name)
    return {"rows": rows, "files": names, "timing": {"total": time.perf_counter() - t}}


def _run_checked(fn, expect: int, what: str, root: str, args: tuple, tries: int = 3) -> dict:
    """Run an encode or split task and check that it wrote exactly the rows
    it read (`expect`, from the scan). On a mismatch its own files are
    deleted and it runs again; if it never matches, it fails with the
    details and leaves the last attempt's files for inspection."""
    got = None
    for attempt in range(tries):
        r = fn(*args)
        got = sum(r["rows"].values())
        if got == expect:
            r["retries"] = attempt
            return r
        names = [(k, n) for k, v in r["files"].items() for n in v]
        log.warning(
            "prep: %s wrote %s rows, expected %s (attempt %d)", what, f"{got:,}", f"{expect:,}", attempt + 1
        )
        if attempt + 1 < tries:
            for k, n in names:
                (Path(root) / ("_buckets" if k < 0 else "shards") / n).unlink(missing_ok=True)
    raise RuntimeError(
        f"prep: {what} wrote {got:,} rows but its input has {expect:,}, in {tries} attempts; the last "
        f"attempt's files are kept for inspection: {sorted(n for v in r['files'].values() for n in v)[:12]}"
    )


def _encode_checked(expect, *args):
    return _run_checked(_encode_task, expect, f"encode task {args[1]} ({len(args[0])} files)", args[2], args)


def _split_checked(expect, *args):
    return _run_checked(_split_task, expect, f"split of bucket {args[0]}", args[1], args)


def _split_task(bucket, root, inputs):
    """Second level of the shuffle: read one bucket's files (the accepted
    encode attempts' files, listed by the coordinator) and write each of its
    shards as a single piece (s<shard>_b<bucket>_<attempt>_<i>)."""
    t = time.perf_counter()
    token = _attempt_token()
    root = Path(root)
    out = root / "shards"
    out.mkdir(parents=True, exist_ok=True)
    files = [str(root / "_buckets" / n) for n in inputs]
    rows, names = {}, {}
    if files:

        def prov(a):
            sid = int(a.partition_keys["__shard"][0])
            return str(out / f"s{sid:06d}_b{bucket:05d}_{token}_{a.index_in_partition}.parquet")

        pl.scan_parquet(files).sink_parquet(
            pl.PartitionBy(str(out), key="__shard", include_key=False, file_path_provider=prov),
            mkdir=True,
            engine="streaming",
        )
        for f in sorted(out.glob(f"s*_b{bucket:05d}_{token}_*.parquet")):
            sid = int(f.name[1:7])
            rows[sid] = rows.get(sid, 0) + _nrows(str(f))
            names.setdefault(sid, []).append(f.name)
    return {"rows": rows, "files": names, "timing": {"total": time.perf_counter() - t}}


def _merge_task(shard_id, root, pieces):
    """Concatenate a shard's pieces into one file. The pieces are left in
    place: the coordinator deletes them once it accepts this result, so a
    duplicate attempt of the same merge still finds its inputs."""
    t = time.perf_counter()
    token = _attempt_token()
    root = Path(root)
    out = root / "shards"
    paths = [str(out / p) for p in pieces]
    target = out / f"s{shard_id:06d}_merged_{token}.parquet"
    tmp = out / f"s{shard_id:06d}_merged_{token}.parquet.tmp"
    pl.scan_parquet(paths).sink_parquet(str(tmp), engine="streaming")
    n = _nrows(str(tmp))
    os.replace(tmp, target)
    return {"rows": n, "file": target.name, "timing": {"total": time.perf_counter() - t}}


# ----------------------------------------------------------------- driver
def prep_distributed(
    source,
    *,
    shard_dir,
    partition,
    fixed_effects,
    clusters,
    keep,
    target_rows,
    overwrite,
    executor,
    rows_per_task: float | None = None,
    max_pieces: int = 8,
    hash_levels: float = 2e6,
    range_groups: float = 5e6,
    max_files: float = 5e4,
):
    """rows_per_task  rows per encode task (default: ~4 tasks per worker, 1M-10M
                   rows each, so the last wave of tasks is short)
    max_pieces     shards made of more piece files than this are merged into one
                   file; shards with fewer keep their pieces (no copy needed)
    hash_levels    FE/cluster specs nested in the partition with more levels than
                   this get 64-bit hash ids instead of dense codes, so encode
                   tasks don't join against huge code tables (collisions are
                   checked exactly; codes are used if any occur)
    range_groups   with more partition groups than this, shards are contiguous
                   ranges of the partition-key hash (balanced by row counts)
                   instead of an LPT assignment table joined on every row
    max_files      if writing one piece per (encode task, shard) would create
                   more files than this (partition groups scattered across the
                   input files), use a two-level shuffle through buckets"""
    from . import __version__
    from .executors import SerialExecutor
    from .prep import META, Dataset, _lpt_bins, clear_incomplete, remove_dir
    from .progress import Progress

    t_start = time.time()
    root = Path(shard_dir)
    if (root / META).exists() and not overwrite:
        raise FileExistsError(f"{root} already holds a prepared dataset (overwrite=True to replace)")
    if overwrite:
        remove_dir(root)
    clear_incomplete(root)
    root.mkdir(parents=True, exist_ok=True)
    executor = executor or SerialExecutor()

    files = _files(source)
    partition = [partition] if isinstance(partition, str) else list(partition)
    schema = pl.scan_parquet(files[0]).collect_schema()
    specs: dict[str, list[str]] = {}
    for spec in list(fixed_effects) + list(clusters):
        comps = fe_components(spec)
        specs["^".join(comps)] = comps
    key_cols = sorted(set(partition) | {c for comps in specs.values() for c in comps})
    missing = set(key_cols) - set(schema.names())
    if missing:
        raise KeyError(f"Columns not in data: {sorted(missing)}")
    cols = [c for c in schema.names() if keep is None or c in set(keep) | set(key_cols)]

    fe_names = ["^".join(fe_components(f)) for f in fixed_effects]
    cl_names = ["^".join(fe_components(c)) for c in clusters]
    # nesting checks: (inner FE comps, outer comps); trivially true when the
    # outer columns are part of the inner key
    checks, check_keys = [], []
    for name in specs:
        if not set(partition) <= set(specs[name]):
            checks.append((specs[name], partition))
            check_keys.append((name, "__partition__"))
    for f in fe_names:
        for c in cl_names:
            if not set(specs[c]) <= set(specs[f]):
                checks.append((specs[f], specs[c]))
                check_keys.append((f, c))

    rows_in = {i: _nrows(f) for i, f in enumerate(files)}
    prog = Progress(root / "_prep_status", label="prep", every=15)
    prog.add_stage("scan", {i: max(n, 1) for i, n in rows_in.items()})

    # ---------------- scan ----------------
    def _minmax(inner):
        return lambda df: df.group_by(inner).agg(pl.col("lo").min(), pl.col("hi").max())

    acc = {
        "n": 0,
        "levels": {k: _Compactor(lambda df: df.unique()) for k in specs},
        "P": [],
        "C": {i: _Compactor(_minmax(checks[i][0])) for i in range(len(checks))},
    }

    scanned: set = set()
    scan_n: dict[int, int] = {}  # rows per input file after dropping null keys

    def on_scan(tid, r):
        if tid in scanned:  # a second result for the same task: keep the first
            return
        scanned.add(tid)
        scan_n[tid] = r["n"]
        acc["n"] += r["n"]
        for k in specs:
            acc["levels"][k].add(r["frames"][f"L:{k}"])
        acc["P"].append(r["frames"]["P"].with_columns(pl.lit(tid, pl.UInt32).alias("__file")))
        for i in range(len(checks)):
            acc["C"][i].add(r["frames"][f"C:{i}"])

    log.info("prep: scanning %d files (%s rows) on workers", len(files), f"{sum(rows_in.values()):,}")
    prog.start("scan")
    scan_tasks = [
        (i, rows_in[i], ([f], cols, key_cols, specs, partition, checks)) for i, f in enumerate(files)
    ]
    # the scan is read-only, so the coordinator helps while no workers are up yet
    executor.run(_scan_task, scan_tasks, on_scan, prog, "scan", local_help=True)
    prog.finish("scan")
    n_rows = acc["n"]

    # ---------------- reduce ----------------
    codes_dir = root / "_codes"
    codes_dir.mkdir(exist_ok=True)
    fe_meta: dict[str, dict] = {}
    levels = {}
    for name, comps in specs.items():
        levels[name] = acc["levels"].pop(name).result()
        fe_meta[name] = {
            "column": fe_column(name),
            "components": comps,
            "n_levels": levels[name].height,
            "nested_in_partition": True,
        }
    for (name, outer), i in zip(check_keys, range(len(checks))):
        mm = acc["C"].pop(i).result()
        ok = bool((mm["lo"] == mm["hi"]).all()) if mm.height else True
        if outer == "__partition__":
            fe_meta[name]["nested_in_partition"] = ok
        else:
            fe_meta[name].setdefault("nested_in_cluster", {})[outer] = ok
    for f in fe_names:  # trivially nested pairs
        for c in cl_names:
            fe_meta[f].setdefault("nested_in_cluster", {}).setdefault(c, True)

    hashed = []
    for name, comps in specs.items():
        col = fe_column(name)
        lv = levels.pop(name)
        if fe_meta[name]["nested_in_partition"] and lv.height > hash_levels:
            n_hash = lv.select(pl.struct(comps).hash(seed=SEED_FE).n_unique()).item()
            if n_hash == lv.height:
                hashed.append(name)
                fe_meta[name]["id"] = "hash"
                continue
            log.warning(
                "prep: hash collision among %s levels of %s; using dense codes", f"{lv.height:,}", name
            )
        (
            lv.sort(comps)
            .with_row_index(col)
            .with_columns(pl.col(col).cast(pl.UInt32))
            .write_parquet(codes_dir / f"{col}.parquet")
        )
        fe_meta[name]["id"] = "code"
    if hashed:
        log.info(
            "prep: hash ids for %s (more than %s levels, nested in the partition)",
            ", ".join(hashed),
            f"{int(hash_levels):,}",
        )

    acc["P_files"] = acc.pop("P")
    counts = pl.concat(acc["P_files"]).group_by(partition).agg(pl.col("_n").cast(pl.Int64).sum())
    n_bins = max(1, int(np.ceil(n_rows / target_rows)))
    range_mode = counts.height > range_groups
    if range_mode:
        # contiguous ranges of the partition-key hash, cut by cumulative rows;
        # encode tasks only need the n_bins boundaries
        counts = counts.with_columns(_range_key(partition, schema).alias("__h")).sort("__h")
        raw_bin = _range_bins(counts["_n"].to_numpy(), n_rows, n_bins)
        counts = counts.with_columns(pl.Series("__b", raw_bin))
        bounds = counts.group_by("__b").agg(pl.col("__h").max().alias("b")).sort("__b")
        if bounds.height > 1 and not bool((bounds["b"].diff().drop_nulls() > 0).all()):
            raise RuntimeError("prep: internal error, range shard boundaries are not increasing")
        bounds.select("b").write_parquet(codes_dir / "__shard_bounds.parquet")
        # shard id = position of the group's hash among the boundaries
        counts = counts.with_columns(
            pl.lit(bounds["b"]).search_sorted(pl.col("__h"), side="left").cast(pl.UInt32).alias("__shard")
        )
        log.info("prep: %s partition groups -> %d range shards", f"{counts.height:,}", bounds.height)
    else:
        bins = _lpt_bins(counts["_n"].to_numpy(), n_bins)
        counts = counts.with_columns(pl.Series("__shard", bins, dtype=pl.UInt32))
        counts.select(partition + ["__shard"]).write_parquet(codes_dir / "__shard_map.parquet")
    shard_rows = counts.group_by("__shard").agg(pl.col("_n").sum())
    _check_shard_sizes(shard_rows["_n"].to_numpy(), counts["_n"].to_numpy(), target_rows, n_bins)
    # which shards each input file touches (for the shuffle plan)
    p_all = pl.concat(acc.pop("P_files"))
    if range_mode:
        file_shards = (
            p_all.with_columns(
                pl.lit(bounds["b"]).search_sorted(_range_key(partition, schema), side="left").alias("__shard")
            )
            .select("__file", "__shard")
            .unique()
        )
    else:
        file_shards = (
            p_all.join(counts.select(partition + ["__shard"]), on=partition)
            .select("__file", "__shard")
            .unique()
        )
    del p_all
    rows_by = dict(zip(shard_rows["__shard"].to_list(), shard_rows["_n"].to_list()))
    big = int((counts["_n"] > target_rows).sum())
    if big:
        log.warning("prep: %d partition groups exceed target_rows on their own", big)
    del counts
    log.info(
        "prep: %s rows, codes for %d specs, %d shards (%.0fs)",
        f"{n_rows:,}",
        len(specs),
        len(rows_by),
        time.time() - t_start,
    )

    # ---------------- encode ----------------
    if rows_per_task is None:
        n_workers = max(int(getattr(executor, "max_workers", None) or 1), 1)
        rows_per_task = min(max(n_rows / (4 * n_workers), 1e6), 1e7)
    batches, cur, cur_n = [], [], 0
    for i, f in enumerate(files):
        cur.append(f)
        cur_n += rows_in[i]
        if cur_n >= rows_per_task:
            batches.append((cur, cur_n))
            cur, cur_n = [], 0
    if cur:
        batches.append((cur, cur_n))
    batch_of = {i: b for b, (fs, _) in enumerate(batches) for i in [files.index(f) for f in fs]}
    pieces_est = (
        file_shards.with_columns(pl.col("__file").replace_strict(batch_of).alias("__b"))
        .select("__b", "__shard")
        .unique()
        .height
    )
    n_shards = len(rows_by)
    n_buckets = 0
    if pieces_est > max_files:
        n_buckets = int(min(max(max_files // max(len(batches), 1), 2), n_shards))
        log.info(
            "prep: one piece per task and shard would be %s files; using a two-level shuffle "
            "through %d buckets (%s files)",
            f"{pieces_est:,}",
            n_buckets,
            f"{len(batches) * n_buckets + n_shards:,}",
        )
    # Rows and files come only from the result the executor accepted for each
    # task; files written by any other attempt of the task are swept away.
    written: dict[int, int] = {}
    accepted: dict[int, list[str]] = {}  # shard id (or -1 - bucket) -> file names
    seen: set = set()

    def on_encode(tid, r):
        if tid in seen:  # a second result for the same task: keep the first
            return
        seen.add(tid)
        if r.get("retries"):
            log.warning("prep: task %s needed %d retries to write the right row count", tid, r["retries"])
        for sid, n in r["rows"].items():
            written[sid] = written.get(sid, 0) + n
        for sid, names in r["files"].items():
            accepted.setdefault(sid, []).extend(names)

    prog.add_stage("encode", {b: max(n, 1) for b, (_, n) in enumerate(batches)})
    prog.start("encode")
    log.info("prep: encoding in %d tasks of ~%s rows", len(batches), f"{int(rows_per_task):,}")
    file_idx = {f: i for i, f in enumerate(files)}
    executor.run(
        _encode_checked,
        [
            (
                b,
                n,
                (
                    sum(scan_n[file_idx[f]] for f in fs),
                    fs,
                    b,
                    str(root),
                    cols,
                    key_cols,
                    specs,
                    partition,
                    tuple(hashed),
                    range_mode,
                    n_buckets,
                    schema,
                ),
            )
            for b, (fs, n) in enumerate(batches)
        ],
        on_encode,
        prog,
        "encode",
    )
    prog.finish("encode")
    if sum(written.values()) != n_rows:
        raise RuntimeError(f"prep: encoded {sum(written.values()):,} rows, expected {n_rows:,}")
    if n_buckets:
        brows = {-1 - k: v for k, v in written.items()}
        binputs = {-1 - k: v for k, v in accepted.items()}
        _sweep(root / "_buckets", {n for v in binputs.values() for n in v}, "bucket")
        written, accepted, seen = {}, {}, set()
        prog.add_stage("split", {b: max(brows.get(b, 1), 1) for b in range(n_buckets)})
        prog.start("split")
        executor.run(
            _split_checked,
            [
                (b, brows.get(b, 1), (brows.get(b, 0), b, str(root), sorted(binputs.get(b, []))))
                for b in range(n_buckets)
            ],
            on_encode,
            prog,
            "split",
        )
        prog.finish("split")
        remove_dir(root / "_buckets")
        if sum(written.values()) != n_rows:
            raise RuntimeError(f"prep: split {sum(written.values()):,} rows, expected {n_rows:,}")
    _sweep(root / "shards", {n for v in accepted.values() for n in v}, "piece")

    # ---------------- merge (only shards split into many pieces) ----------------
    pieces: dict[int, list[str]] = {sid: sorted(v) for sid, v in accepted.items() if v}
    to_merge = [s_ for s_, v in pieces.items() if len(v) > max_pieces]
    if to_merge:
        prog.add_stage("merge", {s_: max(written.get(s_, 1), 1) for s_ in to_merge})
        prog.start("merge")
        log.info(
            "prep: merging %d of %d shards (more than %d pieces each)", len(to_merge), len(pieces), max_pieces
        )

        merged_from: dict[int, list[str]] = {}

        def on_merge(tid, r):
            if tid in merged_from:  # a second result for the same task: keep the first
                return
            if r["rows"] != written.get(tid):
                raise RuntimeError(
                    f"prep: merged shard {tid} has {r['rows']:,} rows, expected {written.get(tid):,}"
                )
            merged_from[tid] = pieces[tid]
            pieces[tid] = [r["file"]]

        executor.run(
            _merge_task,
            [(s_, written.get(s_, 1), (s_, str(root), pieces[s_])) for s_ in to_merge],
            on_merge,
            prog,
            "merge",
        )
        prog.finish("merge")
        _remove([root / "shards" / n for v in merged_from.values() for n in v])
        _sweep(root / "shards", {n for v in pieces.values() for n in v}, "merged")
    else:
        log.info("prep: no merge needed (at most %d pieces per shard)", max_pieces)

    shutil.rmtree(codes_dir, ignore_errors=True)
    shards = [
        {"id": int(s_), "files": [f"shards/{p}" for p in pieces[s_]], "rows": int(written[s_])}
        for s_ in sorted(pieces)
        if written.get(s_)
    ]
    columns = pl.scan_parquet(str(root / shards[0]["files"][0])).collect_schema().names()
    meta = {
        "version": 1,
        "partition": partition,
        "shards": shards,
        "fes": {k: v for k, v in fe_meta.items() if k in set(fe_names)},
        "clusters": {k: fe_meta[k] for k in cl_names},
        "columns": columns,
        "n_rows": int(n_rows),
        "target_rows": target_rows,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prep_seconds": round(time.time() - t_start, 1),
        "prep_mode": "distributed",
        "prep_version": __version__,
        "shard_assignment": "range" if range_mode else "lpt",
        "shuffle": f"two-level ({n_buckets} buckets)" if n_buckets else "direct",
    }
    (root / META).write_text(json.dumps(meta, indent=1))
    log.info("prep: wrote %d shards in %.0fs (distributed)", len(shards), time.time() - t_start)
    return Dataset.open(root)
