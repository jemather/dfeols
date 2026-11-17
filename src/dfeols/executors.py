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
import os
import time
import uuid
from collections import deque
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
    # memory a combine asks of its worker (GB; DaskExecutor's MEMORY resource), None: a whole
    # worker. A combine is the top task in its worker's queue, so while it waits for its
    # memory the worker starts nothing else: with whole-worker combines D6 r4's CG passes ran
    # one task per worker (36011018: ~970 combines a pass, 4-thread workers)
    combine_gb: float | None = None


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
        self._counted = True  # add() counts the parts, unless expect() gave the total

    def expect(self, n: int) -> None:
        """All n parts counted before they are submitted: with tasks submitted as workers
        free up (_Feeder), the parts so far could all arrive mid-run and look like the last."""
        self.parts_left, self._counted = n, False

    def add(self, part) -> None:
        self.running.add(part)
        self.parts_left += self._counted

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


def _unbox(box):
    """A shared argument's task: returns the one object it was submitted with (see
    DaskExecutor._scatter_shared)."""
    return box[0]


def _same(x):
    """A copy of a shared argument on one worker (DaskExecutor._spread): the object itself,
    which the worker fetched from the copy above it in the tree."""
    return x


def _on_worker(fn, *args):
    """Run fn and note the worker that ran it (DaskExecutor.run with affinity)."""
    res = fn(*args)
    if isinstance(res, dict):
        try:
            from distributed import get_worker

            res["_worker"] = get_worker().address
        except Exception:  # not on a worker (local_help runs fn itself)
            pass
    return res


def _order(tasks):
    """Largest-first (LPT) for load balance, but one small task first so the
    ETA calibrates quickly."""
    if len(tasks) < 3:
        return list(tasks)
    s = sorted(tasks, key=lambda t: -t[1])
    small = s.pop(-1)
    return [small] + s


# _Feeder: a task whose shard is not cached where it goes reads it again (D6 r4: 4-5 s
# against ~0.7 s from the cache), so it counts this many times its rows
MISS_COST = 4.0


class _Feeder:
    """Hands out a run's tasks (with affinity) to workers as they free up, instead of placing
    them all at the start. Dask keeps a loose restriction (workers=[w],
    allow_other_workers=True) on w while w exists, however busy, and work stealing will not
    move a task whose input is that worker's copy of the CG block, so a placement made up
    front stays: with any fixed one, D11 L3 (job 36113341, cg836) had six workers with 18-29
    tasks queued at 100 s of a ~190 s step while the other 17 ran nothing.

    Each worker has its own queue (the tasks whose shard it holds, `where`, in run order) and
    holds at most `depth` x its threads tasks submitted. A worker with room takes from its
    own queue, else from the shared one (no live owner), else the last task of the worker
    whose queue would take longest, when that worker would finish it later than this one,
    reading the shard again (MISS_COST), would. Work is counted in rows, a hit 1 and a miss
    MISS_COST a row, per thread, at each worker's own pace (`rates`: work a second a thread,
    learnt from the tasks' times, observe(); kept by the executor across runs, since a slow
    node stays slow: the tail of cg836 was on five hosts). tasks: (tid, rows, args) in run
    order."""

    def __init__(self, tasks, where: dict, threads: dict, depth: float = 2.0, rates: dict | None = None):
        self.rows = {tid: max(int(r or 0), 1) for tid, r, _a in tasks}
        self.pos = {tid: i for i, (tid, _r, _a) in enumerate(tasks)}
        self.where = where
        self.depth = depth
        self.threads: dict = {}
        self.inflight: dict = {}
        self.work: dict = {}  # worker -> work in flight (rows x hit or miss cost)
        self.cost: dict = {}  # task in flight -> its work there
        self.ran: dict = {}  # task finished, its time not seen yet -> (worker, work)
        self.rates = {} if rates is None else rates
        self.own: dict = {}
        self.qrows: dict = {}
        self.rest: deque = deque()
        for tid, _r, _a in tasks:
            self.rest.append(tid)
        self.workers(threads)

    def __len__(self):
        return len(self.rest) + sum(len(q) for q in self.own.values())

    def workers(self, threads: dict) -> None:
        """Follow the connected workers: a new one starts empty; a gone one's queue goes
        back to the shared queue (in run order), and its tasks in flight are Dask's."""
        back, changed = [], False
        for w in [w for w in self.own if w not in threads]:
            back += self.own.pop(w)
            del self.qrows[w], self.threads[w], self.inflight[w], self.work[w]
            changed = True
        for w, t in threads.items():
            if w not in self.own:
                self.own[w], self.qrows[w], self.inflight[w], self.work[w] = deque(), 0, 0, 0.0
                changed = True
            self.threads[w] = max(int(t or 1), 1)
        if changed:
            keep = []
            for t in list(self.rest) + back:
                w = self.where.get(t)
                if w in self.own:
                    self.own[w].append(t)
                    self.qrows[w] += self.rows[t]
                else:
                    keep.append(t)
            self.rest = deque(sorted(keep, key=self.pos.__getitem__))
            for w, q in self.own.items():
                self.own[w] = deque(sorted(q, key=self.pos.__getitem__))

    def _take(self, w):
        if self.own[w]:
            t = self.own[w].popleft()
            self.qrows[w] -= self.rows[t]
            return t
        if self.rest:
            return self.rest.popleft()
        busy = [v for v in self.own if v != w and self.own[v]]
        if not busy:
            return None
        def left(v):  # seconds until v is through its work, queue and in flight
            return (self.work[v] + self.qrows[v]) / self.threads[v] / self.rate(v)

        v = max(busy, key=left)
        t = self.own[v][-1]
        if left(v) <= (self.work[w] / self.threads[w] + MISS_COST * self.rows[t]) / self.rate(w):
            return None  # its owner finishes it sooner than w would, reading the shard
        self.own[v].pop()
        self.qrows[v] -= self.rows[t]
        return t

    def fill(self) -> list:
        """[(tid, worker)] to submit now: every worker topped up to depth x threads, a task
        at a time in turn, so no worker takes the whole shared queue."""
        out = []
        room = {w: int(self.depth * self.threads[w]) - self.inflight[w] for w in self.own}
        while True:
            moved = False
            for w in sorted(room, key=lambda w: -room[w]):
                if room[w] <= 0:
                    continue
                t = self._take(w)
                if t is None:
                    room[w] = 0
                    continue
                out.append((t, w))
                room[w] -= 1
                self.inflight[w] += 1
                self.cost[t] = self.rows[t] * (1.0 if self.where.get(t) == w else MISS_COST)
                self.work[w] += self.cost[t]
                moved = True
            if not moved:
                return out

    def rate(self, w) -> float:
        """w's work a second a thread; a worker not timed yet counts as the median one."""
        if w in self.rates:
            return self.rates[w]
        known = sorted(self.rates.values())
        return known[len(known) // 2] if known else 1.0

    def done(self, w, tid) -> None:
        c = self.cost.pop(tid, 0.0)
        self.ran[tid] = (w, c)
        if w in self.inflight:
            self.inflight[w] -= 1
            self.work[w] = max(self.work[w] - c, 0.0)

    def observe(self, tid, seconds) -> None:
        """A finished task's own time (its result's timing): w's pace, a moving average."""
        w, c = self.ran.pop(tid, (None, 0.0))
        if w is None or not c or not seconds or seconds <= 0:
            return
        r = c / seconds
        self.rates[w] = r if w not in self.rates else 0.7 * self.rates[w] + 0.3 * r

    def stragglers(self, out: dict, per_row: float, busy: dict, min_s: float, factor: float) -> list:
        """[(tid, worker)]: tasks to copy to another worker, once nothing is left to hand out.
        out: {tid: (its worker, seconds since it was handed out)} of the tasks not finished
        and not copied yet; per_row: seconds a row of this run's finished tasks; busy:
        {worker: copies running there}. A task is late past max(min_s, factor x its expected
        time); the latest go first, each to the fastest other worker with a thread free.
        Tasks already running are Dask's, and the feeder's stealing moves only queued ones:
        in D11 L3 (36130493) 65 of 310 CG steps each waited on one task of 60-430 s
        (median 3-4 s) on a node shared with other jobs, 3.2 of 8.1 h."""
        free = {w: self.threads[w] - self.inflight[w] - busy.get(w, 0) for w in self.own}
        late = []
        for t, (w, age) in out.items():
            due = max(min_s, factor * self.rows.get(t, 1) * per_row)
            if age > due:
                late.append((age - due, t, w))
        res = []
        for _over, t, w in sorted(late, key=lambda x: -x[0]):
            cands = [v for v in free if v != w and free[v] > 0]
            if not cands:
                break
            v = max(cands, key=self.rate)
            free[v] -= 1
            res.append((t, v))
        return res


class SerialExecutor:
    """Run every shard task in this process, one after another.

    The default executor for `fit` and `prep`: no cluster needed, so it suits small data,
    tests and checking a specification before a full run.
    """

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
        affinity: str | None = None,  # one process: every task's state is here
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
    """Run shard tasks on a `dask.distributed` Client.

    Usually made by `dfeols.executor(profile)`, which starts the workers through the
    cluster's scheduler. Build one directly to use a Client you already have.

    With `worker_memory` (dfeols.cluster's workers declare it as the Dask resource MEMORY,
    with a thread per core), each task asks for its estimated memory, so a worker runs as
    many shards at once as fit: the tasks are mostly single-threaded numpy and scipy (DB1B
    2005Q2-Q3: 4 threads made a task 0-15% faster). Without it, or with a plain Client, one
    task runs per worker, which keeps memory predictable.

    Parameters
    ----------
    client : distributed.Client
        A connected Dask client.
    max_workers : int, optional
        Most workers the fit expects, shown in progress reports ("12 of 40 workers").
    tick : float, default 5.0
        Seconds between scheduler queries while tasks run.
    retries : int, default 3
        Times Dask reruns a task that fails, for example because its worker died.
    worker_memory : int, optional
        Bytes of memory each worker declares as the Dask resource MEMORY.
    """

    MARGIN = 1.25  # on the worker-memory estimate
    # seconds to fetch one finished result before giving up: a worker that cannot read a
    # result it spilled (job 35937527: ENOSPC on 3 workers' local disk) leaves Dask's gather
    # retrying forever ("Couldn't gather keys"), and the coordinator sat 53 min until cancelled
    GATHER_TIMEOUT_S = float(os.environ.get("DFEOLS_GATHER_TIMEOUT_S", "900"))

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
        self.spread_min_bytes = 32 << 20  # ... and this large copied to every worker in a tree
        self.spread_fan_out = 4
        self._affinity: dict[str, dict] = {}  # affinity key -> {task id: worker address}
        self._rates: dict = {}  # worker address -> its pace (_Feeder), across runs
        # finished results fetched at once, each its own round trip to the worker that holds
        # it: one at a time, a fetch from a busy worker held up the rest (D6 r4 job 36097628,
        # cg5: one fetch blocked 148 s while 1,078 results waited; fetching was 300-700 s
        # of each 500-900 s pass)
        self.fetch_threads = int(os.environ.get("DFEOLS_FETCH_THREADS", "16"))
        # tasks with affinity held submitted per worker thread (_Feeder): enough that a worker
        # never waits on the coordinator for its next task, few enough that the rest stay
        # free to go to whichever worker frees up first
        self.queue_depth = float(os.environ.get("DFEOLS_QUEUE_DEPTH", "2"))
        # a task handed out (_Feeder) and not back after this many seconds, and factor x its
        # expected time, gets a copy on an idle worker once no task is left to hand out; the
        # first of the two to finish counts (_Feeder.stragglers). 0 turns copies off
        self.copy_after_s = float(os.environ.get("DFEOLS_COPY_AFTER_S", "30"))
        self.copy_factor = float(os.environ.get("DFEOLS_COPY_FACTOR", "4"))

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

    def _fetch(self, f, what: str):
        """f.result(), or a clear error when the result cannot be fetched in GATHER_TIMEOUT_S."""
        try:
            return f.result(timeout=self.GATHER_TIMEOUT_S)
        except TimeoutError:
            try:
                where = sorted(self.client.who_has([f]).get(f.key, ()))
            except Exception:
                where = []
            raise RuntimeError(
                f"{what}: finished, but its result could not be fetched from "
                f"{', '.join(where) or 'its worker'} in {self.GATHER_TIMEOUT_S:.0f} s (the "
                "worker may have run out of local disk for spilled data, or died; see its log "
                "and the scheduler log); DFEOLS_GATHER_TIMEOUT_S sets the limit"
            ) from None

    def _workers(self) -> list[str]:
        return sorted(self._threads())

    def _threads(self) -> dict:
        """{worker address: threads} of the connected workers."""
        try:
            info = self.client.scheduler_info(n_workers=-1)
        except TypeError:
            info = self.client.scheduler_info()
        return {a: int(w.get("nthreads") or 1) for a, w in info.get("workers", {}).items()}

    def _spread(self, root, key: str, priority: int, workers: list | None = None) -> dict:
        """Copies of a shared argument (the future `root`) on every worker, in a tree: each
        copy is fetched from the one above it, fan_out at a time, so no worker sends more than
        fan_out copies. Tasks whose worker is known (affinity) take that worker's copy. D6 r4
        (36011018, 12.1M levels x 7: 0.68 GB) waited ~2.5 min at each CG step while 36 workers
        fetched the block from the one that held it. {worker address: future}."""
        workers = self._workers() if workers is None else list(workers)
        if len(workers) < 2:
            return {}
        k = self.spread_fan_out
        out: dict = {}
        for i, w in enumerate(workers):
            parent = root if i < k else out[workers[i // k - 1]]
            out[w] = self.client.submit(
                _same,
                parent,
                key=f"{key}-on-{i}",
                workers=[w],
                allow_other_workers=True,
                pure=False,
                retries=self.retries,
                priority=priority,
            )
        return out

    def _scatter_shared(
        self, tasks, run_id: str = "", spread: bool = False, workers: list | None = None
    ) -> dict:
        """Large arguments passed to every task go to the workers once, not once per task:
        arrays (plan C's CG block) and dicts of them (round 2's linear maps, with plan C's FE
        solution: D6, 758k levels, a 46 MiB graph).

        Each is a task returning it (_unbox), not client.scatter: scatter(broadcast=True)
        calls Scheduler.replicate, which loops until every worker holds a copy while the
        Active Memory Manager (on by default) drops the extra copies, so it can spin for
        hours (D5 pairyear 35923333 at its first CG step, D6 35924356 at singletons2:
        2 h each, until the workers reached their walltime). And scattered data has no
        recipe: when its last holder left, Dask logged that it "can't be recovered" and the
        tasks waiting on it never started. A task's argument stays in the scheduler's graph,
        so Dask recomputes it on a lost worker, and workers fetch it from each other as they
        need it.

        With `spread` (the tasks' workers are known), an argument of spread_min_bytes or more
        is also copied to every worker (_spread). {id(argument): (future, {worker: copy})}."""
        if not tasks:
            return {}
        ids = None
        for _tid, _rows, args in tasks:
            mine = {id(a) for a in args if _nbytes(a) >= self.scatter_min_bytes}
            ids = mine if ids is None else ids & mine
        if not ids:
            return {}
        shared = {id(a): a for a in tasks[0][2] if id(a) in ids}
        import warnings

        out = {}
        with warnings.catch_warnings():  # "Sending large graph": by design, once per run
            warnings.filterwarnings("ignore", message=".*[Ll]arge graph.*")
            for j, (i, a) in enumerate(shared.items()):
                # in a list, so Dask passes a dict as one object rather than walking its keys
                key = f"dfeols-shared-{run_id or uuid.uuid4().hex[:8]}-{j}"
                root = self.client.submit(_unbox, [a], key=key, pure=False, retries=self.retries)
                big = spread and _nbytes(a) >= self.spread_min_bytes
                out[i] = (root, self._spread(root, key, len(tasks) + 3, workers) if big else {})
        return out

    def run(
        self,
        fn: Callable,
        tasks,
        on_result,
        progress,
        stage: str,
        local_help: bool = False,
        reduce: Reduce | None = None,
        affinity: str | None = None,
        affinity_prune: Callable | None = None,
    ):
        """local_help: while no workers are connected, the coordinator runs
        queued tasks itself (lowest priority first). Only for tasks without
        side effects, since a task a worker had just picked up may then run
        twice (its worker result is discarded). reduce: see Reduce; a lost part
        is recomputed by Dask from its task. affinity: a name for state the
        tasks leave in their worker's memory (plan C's shard cache): each task
        goes to the worker that last ran the same task id under that name, if it
        is still there (else anywhere), so CG steps find their shard in the
        cache (D6 r4 job 36011018: tasks went to any worker, 2.5% hits, and
        reading the shard again was 4-5 s of each ~5 s task). A result's "cached" (False: the
        worker did not keep the state, its cache was full) keeps the map from following a
        task to a worker without it. affinity_prune(affinity, {task id: worker}) runs on
        every worker after a run in which a task moved to another worker (_Feeder, or Dask
        stealing), so the old worker drops its stale copy (worker.iter_cache_keep). With
        affinity and connected workers, tasks are submitted as workers free up (_Feeder),
        at most queue_depth x a worker's threads at a time."""
        import queue
        from concurrent.futures import ThreadPoolExecutor

        if reduce and local_help:
            raise ValueError("reduce and local_help together are not supported")
        ordered = _order(tasks)
        futs, fargs = {}, {}
        n = len(ordered)
        run_id = uuid.uuid4().hex[:8]  # task keys must never repeat across runs (e.g. two preps)
        where = self._affinity.setdefault(affinity, {}) if affinity else {}
        # every task given to a worker by the coordinator (and its worker's copy of the shared
        # block): a task left to Dask took the single root copy, so each miss fetched the
        # block from one worker again (36097628: ~2.5 min at the start of each pass)
        threads = self._threads() if affinity else {}
        feeder = _Feeder(ordered, where, threads, self.queue_depth, self._rates) if threads else None
        shared = self._scatter_shared(ordered, run_id, spread=feeder is not None, workers=sorted(threads))
        # finished futures arrive here from their done callbacks (run off the event loop), once
        # each: distributed.wait(FIRST_COMPLETED) over every pending future, called again after
        # each completion, leaves a waiter task per future on the loop it shares with the
        # scheduler, so the loop's work grows as calls x pending (the stalls of 36011018 and
        # 36016180, 5 s a median mid-step, measured by the D11 chat)
        doneq: queue.Queue = queue.Queue()
        pool = self._pool(reduce, stage, run_id, n + 2, doneq.put) if reduce else None
        if pool is not None:
            pool.expect(n)
        part_futs: set = set()
        kw = {}
        gb_of = getattr(progress, "task_gb_of", None) if self.worker_memory else None
        if self.worker_memory:
            gb = getattr(progress, "task_gb", None)
            need = self.worker_memory if gb is None else min(gb * 1e9 * self.MARGIN, self.worker_memory)
            kw["resources"] = {"MEMORY": int(need)}
        trows = {tid: r for tid, r, _a in ordered}

        def task_kw(tid):  # a task's own memory, when the estimate depends on its rows
            if gb_of is None:
                return kw
            gb = gb_of(max(int(trows[tid] or 0), 1))
            return {**kw, "resources": {"MEMORY": int(min(gb * 1e9 * self.MARGIN, self.worker_memory))}}
        run_fn = (lambda *a: (_on_worker, fn, *a)) if affinity else (lambda *a: (fn, *a))

        def arg(a, w):
            if id(a) not in shared:
                return a
            root, copies = shared[id(a)]
            return copies.get(w, root)

        targs = {tid: args for tid, _rows, args in ordered}
        pos = {tid: i for i, (tid, _r, _a) in enumerate(ordered)}
        fworker: dict = {}  # future -> the worker the feeder gave it to
        key_to_tid: dict = {}
        pending: set = set()
        # copies of late tasks (_Feeder.stragglers): each attempt is (task, part) by its result
        # future; the first attempt of a task to bring back its part or result wins, and the
        # other is cancelled and forgotten, so every part is summed once
        attempt: dict = {}  # result future -> (task future, part future or None)
        part_of: dict = {}  # part future -> its attempt's result future
        first: dict = {}  # tid -> its first attempt's result future
        copied: dict = {}  # tid -> its copy's result future
        copy_w: dict = {}  # a copy's result future -> its worker
        busy: dict = {}  # worker -> copies running there
        handed: dict = {}  # tid -> when it was handed out
        won: set = set()
        per_row: list = []  # seconds a row of the finished tasks
        n_won = 0

        def submit(tid, w=None, copy=False):
            args = targs[tid]
            key = f"dfeols-{stage}-{progress.label}-{run_id}-{tid}" + ("-copy" if copy else "")
            place = {"workers": [w], "allow_other_workers": True} if w else {}
            sargs = [arg(a, w) for a in args]
            m = f = self.client.submit(
                *run_fn(*sargs), key=key, priority=n - pos[tid], retries=self.retries, pure=False,
                **task_kw(tid), **place,
            )
            part = None
            if reduce:  # on_result sees the result without its part; the part goes in the tree
                hi = {"priority": n + 1, "pure": False}  # ahead of the shards still queued
                part = self.client.submit(reduce.get, m, key=f"{key}-part", **hi)
                pool.add(part)
                part_futs.add(part)
                part.add_done_callback(doneq.put)
                f = self.client.submit(reduce.strip, m, key=f"{key}-rest", **hi)
                part_of[part] = f
            f.add_done_callback(doneq.put)
            attempt[f] = (m, part)
            futs[f] = tid
            fargs[f] = args
            key_to_tid[f.key] = tid
            pending.add(f)
            if copy:
                copied[tid], copy_w[f] = f, w
                busy[w] = busy.get(w, 0) + 1
            else:
                first[tid] = f
                handed[tid] = time.monotonic()
                if w:
                    fworker[f] = w

        def drop(f):  # an attempt that lost, or a copy that failed: cancelled and forgotten
            tid = futs[f]
            m, part = attempt[f]
            pending.discard(f)
            if part is not None:
                pool.running.discard(part)
                part_futs.discard(part)
            if f in fworker:  # its worker's slot is free; its pace: as slow as it was
                feeder.done(fworker.pop(f), tid)
                feeder.observe(tid, time.monotonic() - handed[tid])
            if f in copy_w:
                busy[copy_w.pop(f)] -= 1
            for x in (f, part, m):
                if x is not None:
                    x.cancel()

        def settle(f):  # f's attempt brought back something first: the other one goes
            nonlocal n_won
            tid = futs[f]
            if tid not in copied or tid in won:
                return
            won.add(tid)
            n_won += f in copy_w
            for g in (first[tid], copied[tid]):
                if g is not f and g in pending:
                    drop(g)

        if feeder is None:
            for tid, _rows, _args in ordered:
                submit(tid)
        else:
            for tid, w in feeder.fill():
                submit(tid, w)
        order = list(futs)  # submission order = priority order (local_help: no feeder)
        from .diag import RunLog

        diag = RunLog(self.client, progress, stage, n)

        def backlog():  # results finished on the workers that the coordinator has not taken
            return sum(1 for f in list(pending) if f.status == "finished")

        fetcher = ThreadPoolExecutor(max(1, self.fetch_threads), thread_name_prefix="dfeols-fetch")

        def fetch_one(f, tid):  # in a fetch thread; the result comes back through doneq
            t = time.perf_counter()
            try:
                doneq.put((f, tid, self._fetch(f, f"shard {tid} in {stage}"), time.perf_counter() - t, None))
            except BaseException as e:  # raised on the coordinator's thread
                doneq.put((f, tid, None, 0.0, e))

        moved = False  # a task's state now on another worker than before: prune after
        last_task = None
        t_status = float("-inf")
        try:
            while pending or (feeder is not None and len(feeder)) or (pool and not pool.finished()):
                if local_help and feeder is None and self.slots()[0] == 0:
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
                        timing = res.get("timing", {})
                        progress.task_done(stage, tid, timing.get("total", 0.0), res.get("timing"))
                        progress.tick()
                        continue
                got = []
                try:
                    got.append(doneq.get(timeout=self.tick_s))
                    while True:
                        got.append(doneq.get_nowait())
                except queue.Empty:
                    pass
                fetched = [g for g in got if isinstance(g, tuple)]
                # a future cancelled by local_help, or already handled, is not waited on any more
                done = [
                    f
                    for f in got
                    if not isinstance(f, tuple) and (f in pending or (pool is not None and f in pool.running))
                ]
                for f in done:
                    if not (f in pending or (pool is not None and f in pool.running)):
                        continue  # its attempt lost to a copy earlier in this batch
                    if f not in futs:  # a part or a combine: into the pool
                        own = part_of.get(f)
                        if f.status == "error":
                            if own in copy_w and futs[own] not in won:  # a failed copy
                                drop(own)
                                continue
                            exc = f.exception()
                            raise RuntimeError(f"combining the parts failed in {stage}: {exc!r}") from exc
                        if own is not None:
                            settle(own)
                        pool.arrived(f, f in part_futs)
                        part_futs.discard(f)
                        continue
                    tid = futs[f]
                    if f.status == "error":
                        if f in copy_w and tid not in won:  # the first attempt may still finish
                            log.debug("%s: copy of shard %s failed: %r", stage, tid, f.exception())
                            drop(f)
                            continue
                        exc = f.exception()
                        raise RuntimeError(f"shard {tid} failed in {stage}: {exc!r}") from exc
                    settle(f)
                    if f in fworker:  # its worker has room again
                        feeder.done(fworker.pop(f), tid)
                    if f in copy_w:
                        busy[copy_w.pop(f)] -= 1
                    fetcher.submit(fetch_one, f, tid)
                if feeder is not None and done:
                    for tid, w in feeder.fill():
                        submit(tid, w)
                for f, tid, res, fetch_s, err in fetched:
                    if err is not None:
                        raise err
                    if f not in pending:
                        continue
                    if affinity and isinstance(res, dict) and "_worker" in res:
                        res = dict(res)
                        w = res.pop("_worker")
                        if res.pop("cached", True):
                            moved = moved or where.get(tid) not in (None, w)
                            where[tid] = w
                        elif where.get(tid) == w:  # its worker no longer has it
                            where.pop(tid, None)
                    if feeder is not None:
                        secs = (res.get("timing") or {}).get("total")
                        feeder.observe(tid, secs)
                        if secs:
                            per_row.append(secs / feeder.rows.get(tid, 1))
                    t_on = time.perf_counter()
                    on_result(tid, res)
                    t_end = time.perf_counter()
                    timing = res.get("timing", {})
                    progress.task_done(stage, tid, timing.get("total", 0.0), res.get("timing"))
                    diag.task(tid, res.get("timing"), fetch_s, t_end - t_on)
                    pending.discard(f)
                    f.release()
                    if not pending:
                        last_task = time.perf_counter()
                if time.monotonic() - t_status >= self.tick_s:  # scheduler queries: once a tick
                    t_status = time.monotonic()
                    if feeder is not None:  # workers that joined or left
                        live = self._threads()
                        feeder.workers(live)
                        for tid, w in feeder.fill():
                            submit(tid, w)
                        if self.copy_after_s > 0 and not len(feeder) and len(per_row) >= 3:
                            now = time.monotonic()
                            out = {
                                futs[f]: (w, now - handed[futs[f]])
                                for f, w in fworker.items()
                                if futs[f] not in copied
                            }
                            rate = sorted(per_row)[len(per_row) // 2]
                            late = feeder.stragglers(out, rate, busy, self.copy_after_s, self.copy_factor)
                            for tid, w in late:
                                submit(tid, w, copy=True)
                        s, mx = len(live), self.max_workers
                    else:
                        s, mx = self.slots()
                    progress.set_slots(s, mx)
                    progress.running[stage] = {key_to_tid[k] for k in self._running_keys() if k in key_to_tid}
                progress.tick()
                diag.tick(backlog)
        finally:
            fetcher.shutdown(wait=False, cancel_futures=True)
        if affinity_prune is not None and moved:
            try:
                self.client.run(affinity_prune, affinity, dict(where))
            except Exception as e:  # the cache's housekeeping only
                log.debug("%s: affinity prune failed: %r", stage, e)
        summ = diag.close()
        log.debug("%s %s: %s", stage, progress.label, summ)
        if copied:
            log.info(
                "%s %s: %d late tasks copied to another worker; the copy finished first for %d",
                stage, progress.label, len(copied), n_won,
            )
        if pool:
            root = pool.root()
            reduce.done(self._fetch(root, f"the sum of {stage}'s parts") if root is not None else None)
            log.info(
                "%s %s: parts summed on the workers in %d combines (%.0f s after the last task)",
                stage,
                progress.label,
                pool.combines,
                time.perf_counter() - (last_task or time.perf_counter()),
            )

    def _pool(self, reduce: Reduce, stage: str, run_id: str, priority: int, on_done=None) -> _Pool:
        """Combine tasks, each summing fan_in parts as soon as they exist, ahead of the shards
        still queued (parts do not pile up on the workers). A combine holds fan_in parts at
        once: it asks for reduce.combine_gb, else a whole worker."""
        kw = {}
        if self.worker_memory:
            gb = reduce.combine_gb
            need = self.worker_memory if gb is None else min(gb * 1e9, self.worker_memory)
            kw["resources"] = {"MEMORY": int(need)}
        seq = iter(range(1 << 30))

        def submit(fn, *items):
            key = f"dfeols-{stage}-combine-{run_id}-{next(seq)}"
            f = self.client.submit(fn, *items, key=key, pure=False, retries=self.retries, priority=priority, **kw)
            if on_done is not None:
                f.add_done_callback(on_done)
            return f

        return _Pool(reduce, submit)
