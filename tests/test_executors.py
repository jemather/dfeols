"""Executors: the combine tree (Reduce) on Dask and in-process."""

import pytest

from dfeols.executors import DaskExecutor, Reduce, SerialExecutor
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
        (fut,) = shared.values()
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
