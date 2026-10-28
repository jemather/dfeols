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

log = logging.getLogger("dfeols")


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

    def run(self, fn: Callable, tasks, on_result, progress, stage: str, local_help: bool = False):
        progress.set_slots(1, 1)
        for tid, _rows, args in _order(tasks):
            progress.running[stage].add(tid)
            t = time.perf_counter()
            res = fn(*args)
            secs = time.perf_counter() - t
            on_result(tid, res)
            progress.task_done(stage, tid, res.get("timing", {}).get("total", secs), res.get("timing"))


class DaskExecutor:
    """Run tasks on a dask.distributed Client.

    Each worker should run one task at a time (nthreads=1, processes=1 per
    Slurm job); numpy/BLAS and the Rust demeaner use the job's cores inside
    the task. That keeps memory per worker predictable: one shard at a time.
    """

    def __init__(self, client, max_workers: int | None = None, tick: float = 5.0, retries: int = 3):
        self.client = client
        self.max_workers = max_workers
        self.tick_s = tick
        self.retries = retries
        self.scatter_min_bytes = 1 << 20  # arrays this large shared by every task are scattered

    def slots(self):
        try:
            info = self.client.scheduler_info(n_workers=-1)
        except TypeError:
            info = self.client.scheduler_info()
        workers = info.get("workers", {})
        return sum(int(w.get("nthreads", 1)) for w in workers.values()), self.max_workers

    def _running_keys(self):
        try:
            proc = self.client.processing()
            return {k for ks in proc.values() for k in ks}
        except Exception:
            return set()

    def _scatter_shared(self, tasks) -> dict:
        """Large arrays passed to every task (plan C's CG block) go to the workers once,
        not once per task: {id(array): future}."""
        import numpy as np

        if not tasks:
            return {}
        ids = None
        for _tid, _rows, args in tasks:
            mine = {id(a) for a in args if isinstance(a, np.ndarray) and a.nbytes >= self.scatter_min_bytes}
            ids = mine if ids is None else ids & mine
        if not ids:
            return {}
        arrays = {id(a): a for a in tasks[0][2] if id(a) in ids}
        return {i: self.client.scatter(a, broadcast=True, hash=False) for i, a in arrays.items()}

    def run(self, fn: Callable, tasks, on_result, progress, stage: str, local_help: bool = False):
        """local_help: while no workers are connected, the coordinator runs
        queued tasks itself (lowest priority first). Only for tasks without
        side effects, since a task a worker had just picked up may then run
        twice (its worker result is discarded)."""
        from distributed import wait

        ordered = _order(tasks)
        futs, fargs = {}, {}
        n = len(ordered)
        run_id = uuid.uuid4().hex[:8]  # task keys must never repeat across runs (e.g. two preps)
        shared = self._scatter_shared(ordered)
        for i, (tid, _rows, args) in enumerate(ordered):
            key = f"dfeols-{stage}-{progress.label}-{run_id}-{tid}"
            sargs = [shared.get(id(a), a) for a in args]
            f = self.client.submit(fn, *sargs, key=key, priority=n - i, retries=self.retries, pure=False)
            futs[f] = tid
            fargs[f] = args
        order = list(futs)  # submission order = priority order
        pending = set(futs)
        key_to_tid = {f.key: futs[f] for f in futs}
        while pending:
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
            try:
                done, _ = wait(pending, timeout=self.tick_s, return_when="FIRST_COMPLETED")
            except Exception:  # timeout
                done = set()
            for f in done:
                tid = futs[f]
                if f.status == "error":
                    exc = f.exception()
                    raise RuntimeError(f"shard {tid} failed in {stage}: {exc!r}") from exc
                res = f.result()
                on_result(tid, res)
                progress.task_done(stage, tid, res.get("timing", {}).get("total", 0.0), res.get("timing"))
                pending.discard(f)
                f.release()
            s, mx = self.slots()
            progress.set_slots(s, mx)
            progress.running[stage] = {key_to_tid[k] for k in self._running_keys() if k in key_to_tid}
            progress.tick()
