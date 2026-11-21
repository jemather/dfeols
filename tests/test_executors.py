"""Executors: the combine tree (Reduce) on Dask and in-process."""

import threading

import pytest

from dfeols.executors import MISS_COST, DaskExecutor, Reduce, SerialExecutor, _Feeder
from dfeols.progress import Progress


def _task(i):
    return {"part": [i], "x": i, "timing": {"total": 0.0}}


def _combine(*parts):
    return sorted(sum(parts, []))


@pytest.mark.parametrize("n", [1, 4, 5, 17])
@pytest.mark.parametrize("dask", [False, True])
def test_reduce_sums_every_part_once(n, dask):
    got, seen = {}, []
    red = Reduce(
        get=lambda r: r["part"],
        strip=lambda r: {k: v for k, v in r.items() if k != "part"},
        combine=_combine,
        done=lambda p: got.setdefault("sum", p),
    )
    prog = Progress(None, label="t", every=1e9)
    prog.add_stage("s", {i: 1 for i in range(n)})
    prog.start("s")
    tasks = [(i, 1, (i,)) for i in range(n)]
    if dask:
        distributed = pytest.importorskip("distributed")
        with (
            distributed.LocalCluster(n_workers=2, threads_per_worker=2, dashboard_address=None) as lc,
            distributed.Client(lc) as c,
        ):
            DaskExecutor(c, max_workers=2, tick=0.1).run(
                _task, tasks, lambda t, r: seen.append(r), prog, "s", reduce=red
            )
    else:
        SerialExecutor().run(_task, tasks, lambda t, r: seen.append(r), prog, "s", reduce=red)
    assert sorted(got["sum"]) == list(range(n))  # every part exactly once
    assert len(seen) == n and all("part" not in r for r in seen)


@pytest.mark.parametrize("seed", range(20))
def test_pool_combines_in_completion_order(seed):
    """_Pool with parts and combines finishing in random order: every part summed once. When
    combines finish before the next part does, one combine is left after the last part."""
    import random

    from dfeols.executors import _Pool

    rng = random.Random(seed)
    n = rng.randint(1, 60)
    fan_in = rng.choice([2, 3, 4])
    for instant in (False, True):
        vals, submitted = {}, []

        def submit(fn, *items, vals=vals, submitted=submitted):
            fut = object()
            vals[fut] = fn(*(vals[i] for i in items))
            submitted.append(fut)
            return fut

        pool = _Pool(Reduce(get=None, strip=None, combine=_combine, done=None, fan_in=fan_in), submit)
        parts = []
        for i in range(n):
            p = object()
            vals[p] = [i]
            pool.add(p)
            parts.append(p)
        rng.shuffle(parts)
        after_last = None
        while not pool.finished():
            combines = [f for f in pool.running if f not in parts]
            if parts and not (instant and combines):
                pool.arrived(parts.pop(), True)
                if not parts:
                    after_last = len(submitted)
            else:
                f = combines[0] if instant else rng.choice(sorted(pool.running, key=id))
                if f in parts:
                    parts.remove(f)
                pool.arrived(f, f not in submitted)
        assert vals[pool.root()] == list(range(n))
        if instant and n > 1:
            assert len(submitted) - after_last <= 1


class _NoReduce:
    """An executor whose run() has no reduce: each shard's part goes to the coordinator."""

    def __init__(self):
        self.e = SerialExecutor()

    def slots(self):
        return self.e.slots()

    def run(self, fn, tasks, on_result, progress, stage, local_help=False):
        return self.e.run(fn, tasks, on_result, progress, stage)


@pytest.mark.parametrize("level_space_gb", [None, 2e-4])  # 2e-4: A over budget, blocks only
def test_parts_summed_on_workers_equal_per_shard(tmp_path, level_space_gb):
    import sys
    from pathlib import Path

    import numpy as np
    import polars as pl

    import dfeols

    sys.path.insert(0, str(Path(__file__).parent))
    from simdata import sim_panel

    d = sim_panel().with_columns(cw=(pl.col("id_f") % 100) * 100 + pl.col("week"))
    ds = dfeols.prep(
        d, shard_dir=tmp_path / "s", partition="id_a", fixed_effects=["spell", "cw", "week"], target_rows=3000
    )
    f = "eng ~ feat + posts | spell + cw + week"
    kw = dict(crossing_fe="iterate", log_every=1e9, level_space_gb=level_space_gb)
    a = dfeols.fit(ds, f, executor=_NoReduce(), workdir=str(tmp_path / "a"), **kw)
    distributed = pytest.importorskip("distributed")
    with (
        distributed.LocalCluster(n_workers=2, threads_per_worker=1, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        b = dfeols.fit(
            ds, f, executor=DaskExecutor(c, max_workers=2, tick=0.1), workdir=str(tmp_path / "b"), **kw
        )
    ca, cb = a.options["crossing_cg"], b.options["crossing_cg"]
    assert len(ds.shards) > 4 and ca["mode"] == cb["mode"] == ("level" if level_space_gb is None else "pass")
    assert (
        ca["A_nnz"] == cb["A_nnz"]
        and ca["block_nnz"] == cb["block_nnz"]
        and ca["parts_nnz"] == cb["parts_nnz"]
    )
    assert ca["shipped_nnz"] == ca["parts_nnz"] and cb["shipped_nnz"] <= ca["shipped_nnz"]
    if level_space_gb is None:
        assert cb["shipped_nnz"] == cb["A_nnz"] < ca["shipped_nnz"]  # the coordinator got A once
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-10)


def _sum_shared(i, big):
    return {"x": float(big.sum()) + i, "timing": {"total": 0.0}}


def test_shared_argument_survives_losing_its_worker():
    """A shared argument (DaskExecutor._scatter_shared) is a task: when the worker holding it
    dies before the tasks run, Dask recomputes it rather than logging "can't be recovered" and
    leaving them waiting (D5 35923333, D6 35924356). No broadcast either: no Scheduler.replicate."""
    import numpy as np

    distributed = pytest.importorskip("distributed")
    big = np.ones(300_000)  # over scatter_min_bytes
    with (
        distributed.LocalCluster(n_workers=2, threads_per_worker=1, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        ex = DaskExecutor(c, max_workers=2, tick=0.1)
        tasks = [(i, 1, (i, big)) for i in range(4)]
        shared = ex._scatter_shared(tasks, "t")
        ((fut, copies),) = shared.values()
        assert copies == {}  # the tasks' workers are not known: no copies on every worker
        distributed.wait(fut)
        holder = next(iter(c.who_has(fut)[fut.key]))
        import os

        try:  # the worker dies mid-call (its nanny starts a fresh one, without the data)
            c.run(os._exit, 1, workers=[holder])
        except Exception:
            pass
        assert holder not in c.who_has(fut).get(fut.key, ())
        got = {}
        prog = Progress(None, label="t", every=1e9)
        prog.add_stage("s", {i: 1 for i in range(4)})
        prog.start("s")
        ex.run(_sum_shared, tasks, lambda t, r: got.setdefault(t, r["x"]), prog, "s")
    assert got == {i: 300_000.0 + i for i in range(4)}


def test_diagnostics_written(tmp_path, monkeypatch):
    """A Dask fit writes tasks.jsonl (a line per task, a summary per stage) and cluster.jsonl
    (scheduler snapshots) in its workdir; the scheduler log goes next to the worker logs.
    CG stages after the second are summarised only. DFEOLS_DIAG=0 writes neither."""
    import json
    import logging

    import polars as pl

    import dfeols
    from dfeols import diag
    from simdata import sim_panel

    distributed = pytest.importorskip("distributed")
    monkeypatch.setenv("DFEOLS_DIAG_SECONDS", "0")
    d = sim_panel().with_columns(cw=(pl.col("id_f") % 100) * 100 + pl.col("week"))
    ds = dfeols.prep(
        d, shard_dir=tmp_path / "s", partition="id_a", fixed_effects=["spell", "cw"], target_rows=3000
    )
    f = "eng ~ feat + posts | spell + cw"
    kw = dict(crossing_fe="iterate", log_every=1e9, level_space_gb=0, iter_tol=1e-12)
    with (
        distributed.LocalCluster(n_workers=2, threads_per_worker=2, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        sched = diag.attach_scheduler_log(tmp_path / "logs")
        res = dfeols.fit(
            ds, f, executor=DaskExecutor(c, max_workers=2, tick=0.1), workdir=str(tmp_path / "b"), **kw
        )
        monkeypatch.setenv("DFEOLS_DIAG", "0")
        dfeols.fit(
            ds, f, executor=DaskExecutor(c, max_workers=2, tick=0.1), workdir=str(tmp_path / "c"), **kw
        )
    for name in ("distributed", "dask_jobqueue"):  # leave other tests' logging alone
        lg = logging.getLogger(name)
        for h in [h for h in lg.handlers if getattr(h, "baseFilename", "") == str(sched.resolve())]:
            lg.removeHandler(h)
            h.close()
    assert sched.exists()
    lines = [json.loads(x) for x in (tmp_path / "b" / "tasks.jsonl").read_text().splitlines()]
    summ = {r["stage"]: r["summary"] for r in lines if "summary" in r}
    tasks = [r for r in lines if "summary" not in r]
    steps = res.options["crossing_cg"]["steps"]
    assert steps > diag.DETAIL_CG_STEPS
    assert {"round1", "round2", "cg1", f"cg{steps}"} <= set(summ)
    assert summ["round1"]["tasks"] == len(ds.shards) == sum(r["stage"] == "round1" for r in tasks)
    assert not any(r["stage"] == f"cg{diag.DETAIL_CG_STEPS + 1}" for r in tasks)
    assert all(r["fetch_s"] >= 0 and r["on_result_s"] >= 0 and "total" in r for r in tasks)
    assert summ["round1"]["mean_running"] > 0
    snaps = [json.loads(x) for x in (tmp_path / "b" / "cluster.jsonl").read_text().splitlines()]
    assert snaps and all("states" in s and len(s["workers"]) == 2 for s in snaps)
    assert {w["threads"] for s in snaps for w in s["workers"]} == {2}
    assert not (tmp_path / "c" / "tasks.jsonl").exists() and not (tmp_path / "c" / "cluster.jsonl").exists()


def _where(i, big=None):
    from distributed import get_worker

    return {"w": get_worker().address, "s": None if big is None else float(big.sum()), "timing": {"total": 0.0}}


def test_affinity_sends_tasks_back_to_their_worker():
    """Tasks run under an affinity name go to the worker that last ran the same task id (plan C:
    round 1 leaves each shard's state there, and the CG steps find it; job 36011018 hit the
    cache on 2.5% of tasks). A large shared argument then reaches every worker as a copy in a
    tree (_spread) and the tasks take their worker's copy."""
    import numpy as np

    distributed = pytest.importorskip("distributed")
    n = 24
    big = np.ones(5_000_000)  # 40 MB: over spread_min_bytes
    with (
        distributed.LocalCluster(n_workers=4, threads_per_worker=2, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        ex = DaskExecutor(c, max_workers=4, tick=0.05)
        runs = []
        for k in range(3):
            got = {}
            prog = Progress(None, label="t", every=1e9)
            prog.add_stage(f"s{k}", {i: 1 for i in range(n)})
            prog.start(f"s{k}")
            args = (lambda i: (i,)) if k == 0 else (lambda i: (i, big))
            ex.run(_where, [(i, 1, args(i)) for i in range(n)], lambda t, r, g=got: g.setdefault(t, r),
                   prog, f"s{k}", affinity="fit1")
            runs.append(got)
        assert all("_worker" not in r for r in runs[1].values())  # the note is the executor's
        # each task goes back to its worker unless another would finish it sooner (_Feeder):
        # with equal workers, nearly all
        for prev, cur in ((runs[0], runs[1]), (runs[1], runs[2])):
            back = sum(cur[t]["w"] == prev[t]["w"] for t in cur)
            assert back >= 0.75 * n, (back, n)
        assert all(r["s"] == 5_000_000.0 for r in runs[2].values())
        assert ex._affinity["fit1"] == {t: r["w"] for t, r in runs[2].items()}
        root = c.submit(lambda: big, pure=False)
        copies = ex._spread(root, "k", 10)
        assert sorted(copies) == sorted(c.scheduler_info()["workers"])
        assert all(f.result().sum() == 5_000_000.0 for f in copies.values())


def _simulate(feeder, speed=None, hit=1.0, miss=MISS_COST, joins=None):
    """Run a _Feeder against simulated workers: each thread runs a task in rows x (hit if it
    ran on the task's owner, else miss) / speed seconds. {tid: worker}, makespan."""
    import heapq

    speed = speed or {}
    joins = dict(joins or {})  # time -> {worker: threads}
    free = {w: t for w, t in feeder.threads.items()}
    queued = {w: [] for w in feeder.threads}
    events, ran, now, seq = [], {}, 0.0, 0

    def start(w):
        nonlocal seq
        while free[w] and queued[w]:
            t = queued[w].pop(0)
            cost = feeder.rows[t] * (hit if feeder.where.get(t) == w else miss) / speed.get(w, 1.0)
            free[w] -= 1
            seq += 1
            heapq.heappush(events, (now + cost, seq, t, w))

    def submit():
        for t, w in feeder.fill():
            assert t not in ran
            ran[t] = w
            queued[w].append(t)
        for w in queued:
            start(w)

    submit()
    while events or len(feeder):
        if joins and (not events or min(joins) <= events[0][0]):
            now = min(joins)
            new = joins.pop(now)
            feeder.workers({**feeder.threads, **new})
            for w, k in new.items():
                free[w], queued[w] = k, []
            submit()
            continue
        now, _s, t, w = heapq.heappop(events)
        free[w] += 1
        feeder.done(w, t)
        feeder.observe(t, feeder.rows[t] * (hit if feeder.where.get(t) == w else miss) / speed.get(w, 1.0))
        submit()
    return ran, now


def test_feeder_keeps_owners_until_another_worker_would_be_sooner():
    """_Feeder: a task runs on its owner unless that owner's queue is longer than reading the
    shard again elsewhere (MISS_COST); unowned tasks go to whoever is free. D11 L3 job
    36113341 (cg836): with a fixed placement six workers held 18-29 tasks at 100 s while 17
    ran nothing."""
    threads = {"a": 4, "b": 4, "c": 4, "d": 4}
    tasks = [(i, 10, ()) for i in range(40)]  # a hit 10 s, a miss 40 s
    even = {i: "abcd"[i % 4] for i in range(40)}
    ran, span = _simulate(_Feeder(tasks, even, threads))
    assert ran == even and span == 30.0  # balanced already: nothing moves
    # everything cached on a: a keeps 28 (70 s), the others take one 40 s miss a thread
    ran, span = _simulate(_Feeder(tasks, {i: "a" for i in range(40)}, threads))
    assert sorted(ran) == list(range(40)) and sum(w == "a" for w in ran.values()) == 28
    assert span == 70.0  # a alone: 100 s
    # no owner: spread over everyone, 10 misses each, 3 rounds
    ran, span = _simulate(_Feeder(tasks, {}, threads))
    assert span == 120.0
    # a slow owner sheds its tail once its pace is known; the pace carries to the next run
    tasks = [(i, 10, ()) for i in range(160)]
    even = {i: "abcd"[i % 4] for i in range(160)}
    rates: dict = {}
    ran, span = _simulate(_Feeder(tasks, even, threads, rates=rates), speed={"b": 0.25})
    assert span <= 200.0 and sum(w == "b" for w in ran.values()) < 25  # placed up front: 400 s
    assert rates["b"] == pytest.approx(0.25) and rates["a"] == pytest.approx(1.0)
    ran, span = _simulate(_Feeder(tasks, even, threads, rates=rates), speed={"b": 0.25})
    assert span <= 200.0


def test_feeder_follows_workers_that_leave_and_join():
    """A gone worker's queue goes back to the shared one, a new worker takes tasks at once."""
    tasks = [(i, 10, ()) for i in range(24)]
    f = _Feeder(tasks, {i: "abc"[i % 3] for i in range(24)}, {"a": 1, "b": 1, "c": 1}, depth=1)
    first = dict(f.fill())
    assert first == {0: "a", 1: "b", 2: "c"}
    f.workers({"a": 1, "b": 1})  # c left: its 7 queued tasks are anyone's
    assert len(f.rest) == 7 and "c" not in f.own
    f.workers({"a": 1, "b": 1, "z": 2})
    assert sorted(w for _t, w in f.fill()) == ["z", "z"]
    ran, _span = _simulate(_Feeder(tasks, {}, {"a": 1}), joins={25.0: {"b": 1}})
    assert sorted(ran) == list(range(24)) and 8 <= sum(w == "b" for w in ran.values()) <= 12


def _kept(i, keep):
    from distributed import get_worker

    return {"w": get_worker().address, "cached": keep(i), "timing": {"total": 0.0}}


def _keep_even(i):
    return i % 2 == 0


def test_affinity_follows_only_kept_state():
    """A task whose worker did not keep its state ("cached": False, the cache was full) is not
    pinned there: D6 r4 job 36097628's hits fell from 2241 of 2902 to 485 as the map followed
    tasks to workers without their shard."""
    distributed = pytest.importorskip("distributed")
    n = 12
    with (
        distributed.LocalCluster(n_workers=3, threads_per_worker=1, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        ex = DaskExecutor(c, max_workers=3, tick=0.05)
        prog = Progress(None, label="t", every=1e9)
        prog.add_stage("s", {i: 1 for i in range(n)})
        prog.start("s")
        got = {}
        ex.run(_kept, [(i, 1, (i, _keep_even)) for i in range(n)], lambda t, r: got.setdefault(t, r),
               prog, "s", affinity="fit")
        assert all("cached" not in r for r in got.values())
        assert ex._affinity["fit"] == {t: r["w"] for t, r in got.items() if t % 2 == 0}


def test_iter_cache_keep_drops_other_owners_shards():
    """worker.iter_cache_keep: on each worker, the fit's entries whose shard is owned by another
    worker go (a task that moved left its state behind); others' fits and unowned shards stay."""
    import dfeols.worker as wk

    distributed = pytest.importorskip("distributed")
    with (
        distributed.LocalCluster(n_workers=2, threads_per_worker=1, processes=False,
                                 dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        a, b = sorted(c.scheduler_info()["workers"])
        with wk._ITER_LOCK:
            wk._ITER_CACHE.clear()
            for k in [("f", 0), ("f", 1), ("f", 2), ("g", 0)]:
                wk._ITER_CACHE[k] = (None, 1)
        c.run(wk.iter_cache_keep, "f", {0: a, 1: b}, workers=[a])
        assert sorted(wk._ITER_CACHE) == [("f", 0), ("f", 2), ("g", 0)]
        wk._ITER_CACHE.clear()


def _rows_task(i):
    return {"timing": {"total": 0.0}}


def test_task_memory_follows_each_tasks_rows():
    """progress.task_gb_of: each task asks the MEMORY its own rows need (plan C's CG steps:
    D6 r4 job 36113346 asked the largest shard's 14.7 GB of a 24 GB worker for every task,
    so workers ran one step task at a time)."""
    distributed = pytest.importorskip("distributed")
    mem = 10e9
    with (
        distributed.LocalCluster(n_workers=1, threads_per_worker=2, resources={"MEMORY": mem},
                                 dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        ex = DaskExecutor(c, max_workers=1, tick=0.05, worker_memory=mem)
        seen = {}

        def record(dask_scheduler=None):
            return {k: dict(ts.resource_restrictions or {}) for k, ts in dask_scheduler.tasks.items()}

        prog = Progress(None, label="t", every=1e9)
        prog.add_stage("s", {i: 1 for i in range(3)})
        prog.start("s")
        prog.task_gb, prog.task_gb_of = 8.0, lambda r: r / 1e6  # 1 GB a million rows

        def on(tid, res):
            seen.update(c.run_on_scheduler(record))

        ex.run(_rows_task, [(0, 1_000_000, (0,)), (1, 2_000_000, (1,)), (2, 4_000_000, (2,))], on, prog, "s")
        asked = {int(k.rsplit("-", 1)[1]): r["MEMORY"] for k, r in seen.items() if k.startswith("dfeols-s-")}
        assert asked == {0: 1.25e9, 1: 2.5e9, 2: 5e9}


def test_feeder_stragglers_go_to_the_fastest_free_worker():
    """_Feeder.stragglers: a task far past its expected time is copied to the fastest other
    worker with a thread free; the latest first, one copy a free thread."""
    f = _Feeder([(0, 10, ()), (1, 10, ()), (2, 1000, ())], {}, {"a": 1, "b": 1, "c": 2})
    f.rates.update({"a": 1.0, "b": 3.0, "c": 2.0})
    f.inflight.update({"a": 1, "b": 0, "c": 1})
    out = {0: ("a", 100.0), 1: ("a", 40.0), 2: ("c", 100.0)}
    # 2 is expected to take 4 x 1000 x 0.05 = 200 s: not late; 0 is later than 1
    assert f.stragglers(out, 0.05, {}, 30.0, 4.0) == [(0, "b"), (1, "c")]
    assert f.stragglers(out, 0.05, {"b": 1}, 30.0, 4.0) == [(0, "c")]  # b runs a copy already
    assert f.stragglers({0: ("a", 20.0)}, 0.05, {}, 30.0, 4.0) == []  # under min_s


def _slow_once(i, d):
    """Task 0's first attempt hangs (a slow node) until the test ends; any other runs at once."""
    import os
    import time

    if i == 0:
        try:
            os.close(os.open(os.path.join(d, "started"), os.O_CREAT | os.O_EXCL))
            while not os.path.exists(os.path.join(d, "stop")):
                time.sleep(0.05)
        except FileExistsError:
            pass
    time.sleep(0.05)
    return {"part": [i], "x": i, "timing": {"total": 0.05}}


def test_late_task_copied_and_counted_once(tmp_path, monkeypatch):
    """A task stuck on its worker gets a copy on an idle one (DFEOLS_COPY_AFTER_S) once nothing
    is left to hand out; the copy's part is summed, the stuck one is cancelled, and every part
    and result comes back once."""
    import time

    distributed = pytest.importorskip("distributed")
    n = 12
    got, seen = {}, []
    red = Reduce(
        get=lambda r: r["part"],
        strip=lambda r: {k: v for k, v in r.items() if k != "part"},
        combine=_combine,
        done=lambda p: got.setdefault("sum", p),
    )
    prog = Progress(None, label="t", every=1e9)
    prog.add_stage("s", {i: 1 for i in range(n)})
    prog.start("s")
    try:
        with (
            distributed.LocalCluster(n_workers=3, threads_per_worker=1, dashboard_address=None) as lc,
            distributed.Client(lc) as c,
        ):
            ex = DaskExecutor(c, max_workers=3, tick=0.05)
            ex.copy_after_s = 0.5
            t = time.perf_counter()
            ex.run(_slow_once, [(i, 1, (i, str(tmp_path))) for i in range(n)],
                   lambda tid, r: seen.append(tid), prog, "s", reduce=red, affinity="fit")
            took = time.perf_counter() - t
    finally:
        (tmp_path / "stop").touch()
    assert (tmp_path / "started").exists()
    assert sorted(got["sum"]) == list(range(n))  # every part once
    assert sorted(seen) == list(range(n))
    assert took < 15, took


@pytest.mark.parametrize("n", [1, 5, 17])
@pytest.mark.parametrize("dask", [False, True])
def test_reduce_trees_sum_each_slice_once(n, dask):
    # Reduce.trees: each result's part taken in slices get(result, j), one tree a slice; done
    # gets the list of the trees' sums
    # done holds an unpicklable lock: the tasks must not carry the Reduce (b0df5d7 submitted the
    # bound Reduce.part, which put round 1's done and its sums in every task: H L2 36318522)
    got = {}
    k = 3
    lock = threading.Lock()

    def done(sums):
        with lock:
            got.setdefault("sums", sums)

    red = Reduce(
        get=lambda r, j: [x for x in r["part"] if x % k == j],
        strip=lambda r: {a: b for a, b in r.items() if a != "part"},
        combine=_combine,
        done=done,
        trees=k,
    )
    prog = Progress(None, label="t", every=1e9)
    prog.add_stage("s", {i: 1 for i in range(n)})
    prog.start("s")
    tasks = [(i, 1, (i,)) for i in range(n)]
    if dask:
        distributed = pytest.importorskip("distributed")
        with (
            distributed.LocalCluster(n_workers=2, threads_per_worker=2, dashboard_address=None) as lc,
            distributed.Client(lc) as c,
        ):
            DaskExecutor(c, max_workers=2, tick=0.1).run(_task, tasks, lambda t, r: None, prog, "s", reduce=red)
    else:
        SerialExecutor().run(_task, tasks, lambda t, r: None, prog, "s", reduce=red)
    sums = got["sums"]
    assert len(sums) == k
    for j, s in enumerate(sums):
        assert sorted(s or []) == [i for i in range(n) if i % k == j]
