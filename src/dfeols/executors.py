# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Executors run a list of shard tasks and stream results back.

Every executor implements:

    run(fn, tasks, on_result, progress, stage)

where tasks is a list of (task_id, rows, args). Results are handed to
`on_result(task_id, result)` as they arrive, so the coordinator reduces them
incrementally and never holds all partials at once.

SerialExecutor   in-process; laptops and tests.
DaskExecutor     any dask.distributed Client, including an adaptive
                 dask-jobqueue cluster where workers join as the scheduler
                 starts them (see dfeols.cluster).

Crossing fixed effects later only need more rounds through the same run()
interface (map over shards, reduce on the coordinator, broadcast back).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger("dfeols")


@dataclass
class Reduce:
    """A part of each task's result summed in a tree instead of handed to on_result: `get`
    takes the part from a result, `strip` the rest (what on_result sees), `combine` sums
    parts (fan_in at a time), and `done` receives the sum (None with no tasks). On Dask the
    parts and sums stay on the workers; the coordinator fetches only the final sum."""

    get: Callable
    strip: Callable
    combine: Callable
    done: Callable
    fan_in: int = 4


class _Tree:
    """Balanced combine tree: fan_in items at one level make one at the next."""

    def __init__(self, reduce: Reduce, submit: Callable):
        self.r, self.submit, self.levels, self.combines = reduce, submit, [[]], 0

    def add(self, item, level: int = 0) -> None:
        if level == len(self.levels):
            self.levels.append([])
        self.levels[level].append(item)
        if len(self.levels[level]) == self.r.fan_in:
            items, self.levels[level] = self.levels[level], []
            self.combines += 1
            self.add(self.submit(self.r.combine, *items), level + 1)

    def finish(self):
        carry = None
        for items in self.levels:
            items = items + ([carry] if carry is not None else [])
            if len(items) > 1:
                self.combines += 1
                carry = self.submit(self.r.combine, *items)
            elif items:
                carry = items[0]
        self.levels = [[]]
        return carry


class _Pool:
    """Combine in completion order (DaskExecutor): parts and sums wait in `ready` once their
    futures finish; fan_in of them make a combine as soon as they exist, and once every part
    is in, any two ready ones are combined at once. The chain left after the last shard is
    then the combines already running plus one or two more, not the tree's depth (DB1B,
    342 shards, fan_in 4: five combines of A-sized sums one after another, 57 s). The order
    of the sums follows the workers, so A can differ between runs in the last bits."""

    def __init__(self, reduce: Reduce, submit: Callable):
        self.r, self.submit = reduce, submit
        self.ready: list = []
        self.running: set = set()  # part and combine futures not finished yet
        self.parts_left = 0
        self.combines = 0

    def add(self, part) -> None:
        self.running.add(part)
        self.parts_left += 1

    def arrived(self, f, is_part: bool) -> None:
        self.running.discard(f)
        self.ready.append(f)
        self.parts_left -= is_part
        k = self.r.fan_in
        while len(self.ready) >= k or (self.parts_left == 0 and len(self.ready) >= 2):
            items, self.ready = self.ready[:k], self.ready[k:]
            self.combines += 1
            self.running.add(self.submit(self.r.combine, *items))

    def finished(self) -> bool:
        return not self.running and self.parts_left == 0 and len(self.ready) <= 1

    def root(self):
        return self.ready[0] if self.ready else None


def _nbytes(a) -> int:
    """Bytes of an array, or of the arrays in a dict (nested dicts and lists included)."""
    import numpy as np

    if isinstance(a, np.ndarray):
        return a.nbytes
    if isinstance(a, dict):
        return sum(_nbytes(v) for v in a.values())
    if isinstance(a, (list, tuple)):
        return sum(_nbytes(v) for v in a)
    return 0


def _order(tasks):
    """Largest-first (LPT) for load balance, but one small task first so the
    ETA calibrates quickly."""
    if len(tasks) < 3:
        return list(tasks)
    s = sorted(tasks, key=lambda t: -t[1])
    small = s.pop(-1)
    return [small] + s


class SerialExecutor:
    def slots(self):
        return 1, 1

    def run(
        self,
        fn: Callable,
        tasks,
        on_result,
        progress,
        stage: str,
        local_help: bool = False,
        reduce: Reduce | None = None,
    ):
        progress.set_slots(1, 1)
        tree = _Tree(reduce, lambda f, *a: f(*a)) if reduce else None
        for tid, _rows, args in _order(tasks):
            progress.running[stage].add(tid)
            t = time.perf_counter()
            res = fn(*args)
            secs = time.perf_counter() - t
            if tree:
                tree.add(reduce.get(res))
                res = reduce.strip(res)
            on_result(tid, res)
            progress.task_done(stage, tid, res.get("timing", {}).get("total", secs), res.get("timing"))
        if tree:
            reduce.done(tree.finish())


class DaskExecutor:
    """Run tasks on a dask.distributed Client.

    With `worker_memory` (bytes; dfeols.cluster's workers declare it as the Dask resource
    MEMORY, with a thread per core), each task asks for `progress.task_gb` of it, so a
    worker runs as many shards at once as fit its memory: the tasks are mostly
    single-threaded numpy and scipy (DB1B 2005Q2-Q3: 4 threads made a task 0-15% faster).
    Without an estimate, or with a plain Client (no worker_memory), one task per worker
    (nthreads=1 per job), which keeps memory predictable.
    """

    MARGIN = 1.25  # on the worker-memory estimate

    def __init__(
        self,
        client,
        max_workers: int | None = None,
        tick: float = 5.0,
        retries: int = 3,
        worker_memory: int | None = None,
    ):
        self.client = client
        self.max_workers = max_workers
        self.worker_memory = worker_memory
        self.tick_s = tick
        self.retries = retries
        self.scatter_min_bytes = 1 << 20  # arrays this large shared by every task are scattered

    def slots(self):
        try:
            info = self.client.scheduler_info(n_workers=-1)
        except TypeError:
            info = self.client.scheduler_info()
        workers = info.get("workers", {})
        return len(workers), self.max_workers  # workers (jobs), as max_workers counts them

    def _running_keys(self):
        try:
            proc = self.client.processing()
            return {k for ks in proc.values() for k in ks}
        except Exception:
            return set()

    def _scatter_shared(self, tasks) -> dict:
        """Large arguments passed to every task go to the workers once, not once per task:
        arrays (plan C's CG block) and dicts of them (round 2's linear maps, with plan C's FE
        solution: D6, 758k levels, a 46 MiB graph). {id(argument): future}."""
        if not tasks:
            return {}
        ids = None
        for _tid, _rows, args in tasks:
            mine = {id(a) for a in args if _nbytes(a) >= self.scatter_min_bytes}
            ids = mine if ids is None else ids & mine
        if not ids:
            return {}
        shared = {id(a): a for a in tasks[0][2] if id(a) in ids}
        # in a list, so Dask sends a dict as one object rather than one future per key
        return {i: self.client.scatter([a], broadcast=True, hash=False)[0] for i, a in shared.items()}

    def run(
        self,
        fn: Callable,
        tasks,
        on_result,
        progress,
        stage: str,
        local_help: bool = False,
        reduce: Reduce | None = None,
    ):
        """local_help: while no workers are connected, the coordinator runs
        queued tasks itself (lowest priority first). Only for tasks without
        side effects, since a task a worker had just picked up may then run
        twice (its worker result is discarded). reduce: see Reduce; a lost part
        is recomputed by Dask from its task."""
        from distributed import wait

        if reduce and local_help:
            raise ValueError("reduce and local_help together are not supported")
        ordered = _order(tasks)
        futs, fargs = {}, {}
        n = len(ordered)
        run_id = uuid.uuid4().hex[:8]  # task keys must never repeat across runs (e.g. two preps)
        shared = self._scatter_shared(ordered)
        pool = self._pool(reduce, stage, run_id, n + 2) if reduce else None
        part_futs: set = set()
        kw = {}
        if self.worker_memory:
            gb = getattr(progress, "task_gb", None)
            need = self.worker_memory if gb is None else min(gb * 1e9 * self.MARGIN, self.worker_memory)
            kw["resources"] = {"MEMORY": int(need)}
        for i, (tid, _rows, args) in enumerate(ordered):
            key = f"dfeols-{stage}-{progress.label}-{run_id}-{tid}"
            sargs = [shared.get(id(a), a) for a in args]
            f = self.client.submit(
                fn, *sargs, key=key, priority=n - i, retries=self.retries, pure=False, **kw
            )
            if reduce:  # on_result sees the result without its part; the part goes in the tree
                hi = {"priority": n + 1, "pure": False}  # ahead of the shards still queued
                part = self.client.submit(reduce.get, f, key=f"{key}-part", **hi)
                pool.add(part)
                part_futs.add(part)
                f = self.client.submit(reduce.strip, f, key=f"{key}-rest", **hi)
            futs[f] = tid
            fargs[f] = args
        order = list(futs)  # submission order = priority order
        pending = set(futs)
        key_to_tid = {f.key: futs[f] for f in futs}
        last_task = None
        while pending or (pool and not pool.finished()):
            if local_help and self.slots()[0] == 0:
                running = self._running_keys()
                cand = [
                    f
                    for f in reversed(order)
                    if f in pending and f.key not in running and f.status == "pending"
                ]
                if cand:
                    f = cand[0]
                    f.cancel()
                    pending.discard(f)
                    tid = futs[f]
                    progress.running[stage].add(tid)
                    res = fn(*fargs[f])
                    on_result(tid, res)
                    progress.task_done(stage, tid, res.get("timing", {}).get("total", 0.0), res.get("timing"))
                    progress.tick()
                    continue
            watch = pending | pool.running if pool else pending
            try:
                done, _ = wait(watch, timeout=self.tick_s, return_when="FIRST_COMPLETED")
            except Exception:  # timeout
                done = set()
            for f in done:
                if f not in futs:  # a part or a combine: into the pool
                    if f.status == "error":
                        exc = f.exception()
                        raise RuntimeError(f"combining the parts failed in {stage}: {exc!r}") from exc
                    pool.arrived(f, f in part_futs)
                    part_futs.discard(f)
                    continue
                tid = futs[f]
                if f.status == "error":
                    exc = f.exception()
                    raise RuntimeError(f"shard {tid} failed in {stage}: {exc!r}") from exc
                res = f.result()
                on_result(tid, res)
                progress.task_done(stage, tid, res.get("timing", {}).get("total", 0.0), res.get("timing"))
                pending.discard(f)
                f.release()
                if not pending:
                    last_task = time.perf_counter()
            s, mx = self.slots()
            progress.set_slots(s, mx)
            progress.running[stage] = {key_to_tid[k] for k in self._running_keys() if k in key_to_tid}
            progress.tick()
        if pool:
            root = pool.root()
            reduce.done(root.result() if root is not None else None)
            log.info(
                "%s %s: parts summed on the workers in %d combines (%.0f s after the last task)",
                stage,
                progress.label,
                pool.combines,
                time.perf_counter() - (last_task or time.perf_counter()),
            )

    def _pool(self, reduce: Reduce, stage: str, run_id: str, priority: int) -> _Pool:
        """Combine tasks, each summing fan_in parts as soon as they exist, ahead of the shards
        still queued (parts do not pile up on the workers). A combine holds fan_in parts at
        once: it asks for a whole worker."""
        kw = {"resources": {"MEMORY": int(self.worker_memory)}} if self.worker_memory else {}
        seq = iter(range(1 << 30))

        def submit(fn, *items):
            key = f"dfeols-{stage}-combine-{run_id}-{next(seq)}"
            return self.client.submit(
                fn, *items, key=key, pure=False, retries=self.retries, priority=priority, **kw
            )

        return _Pool(reduce, submit)
