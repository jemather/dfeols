"""Worker parking for the IO saturation test (DFEOLS_PARK_SCHEDULE): in a CG step only the first
k workers by join order get tasks, and each such step leaves a probe.jsonl record."""

import json
import time

import pytest

from dfeols.executors import DaskExecutor, park_active, park_schedule
from dfeols.progress import Progress


def test_schedule_parsing_and_lookup():
    assert park_schedule(None) is None and park_schedule("") is None and park_schedule("  ") is None
    assert park_schedule("48, 48,36,") == [48, 48, 36]
    with pytest.raises(ValueError):
        park_schedule("48,0")
    s = [48, 36, 24]
    assert park_active("cg1", s) == 48 and park_active("cg3", s) == 24
    assert park_active("cg9", s) == 24  # the last entry repeats
    assert park_active("round1", s) is None and park_active("cgx", s) is None
    assert park_active("cg2", None) is None


def _probe_task(i):
    from distributed import get_worker

    time.sleep(0.02)
    return {"w": get_worker().address, "timing": {"total": 0.02, "read": 0.01, "cache_hit": i % 2 == 0}}


def test_cg_steps_use_only_the_active_workers(monkeypatch, tmp_path):
    distributed = pytest.importorskip("distributed")
    import dask

    monkeypatch.setenv("DFEOLS_PARK_SCHEDULE", "2,3")
    n = 40
    with (
        dask.config.set({"distributed.scheduler.work-stealing": False}),
        distributed.LocalCluster(n_workers=4, threads_per_worker=1, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        c.wait_for_workers(4)
        ex = DaskExecutor(c, max_workers=4, tick=0.05)
        assert ex.park == [2, 3]
        used = {}
        for stage in ("round1", "cg1", "cg2", "cg5"):
            got = {}
            prog = Progress(tmp_path / "fit", label="t", every=1e9)
            prog.verbosity = 0
            prog.add_stage(stage, {i: 10 for i in range(n)})
            prog.start(stage)
            ex.run(_probe_task, [(i, 10, (i,)) for i in range(n)], lambda t, r, g=got: g.setdefault(t, r),
                   prog, stage, affinity="fit1")
            assert len(got) == n
            used[stage] = {r["w"] for r in got.values()}
        order = sorted(ex._joined, key=ex._joined.__getitem__)
        assert used["cg1"] <= set(order[:2])
        assert used["cg2"] <= set(order[:3]) and used["cg5"] <= set(order[:3])
        recs = [json.loads(x) for x in (tmp_path / "fit" / "probe.jsonl").read_text().splitlines()]
        assert [r["stage"] for r in recs] == ["cg1", "cg2", "cg5"]  # round1 isn't parked
        r1 = recs[0]
        assert r1["active"] == 2 and r1["pool"] == 4 and r1["asked"] == 2
        assert r1["tasks"] == n and r1["rows"] == 10 * n and r1["hits"] == n // 2
        assert r1["on_parked"] == 0 and r1["workers_used"] <= 2
        assert r1["step_s"] > 0 and r1["rows_per_s"] > 0 and abs(r1["read_s"] - 0.01 * n) < 1e-6
        assert recs[1]["active"] == 3


def test_without_a_schedule_nothing_changes(monkeypatch):
    distributed = pytest.importorskip("distributed")
    monkeypatch.delenv("DFEOLS_PARK_SCHEDULE", raising=False)
    with (
        distributed.LocalCluster(n_workers=2, threads_per_worker=1, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        ex = DaskExecutor(c, max_workers=2, tick=0.05)
        assert ex.park is None
        prog = Progress(None, label="t", every=1e9)
        prog.add_stage("cg1", {i: 1 for i in range(8)})
        prog.start("cg1")
        got = {}
        ex.run(_probe_task, [(i, 1, (i,)) for i in range(8)], lambda t, r: got.setdefault(t, r), prog, "cg1",
               affinity="f")
        assert len(got) == 8


def test_warns_when_stealing_is_on(monkeypatch, caplog):
    distributed = pytest.importorskip("distributed")
    import dask

    monkeypatch.setenv("DFEOLS_PARK_SCHEDULE", "1")
    with (
        dask.config.set({"distributed.scheduler.work-stealing": True}),
        distributed.LocalCluster(n_workers=2, threads_per_worker=1, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
        caplog.at_level("WARNING", logger="dfeols"),
    ):
        ex = DaskExecutor(c, max_workers=2, tick=0.05)
        prog = Progress(None, label="t", every=1e9)
        prog.add_stage("cg1", {i: 1 for i in range(4)})
        prog.start("cg1")
        ex.run(_probe_task, [(i, 1, (i,)) for i in range(4)], lambda t, r: None, prog, "cg1", affinity="f")
    assert any("work stealing on" in r.message for r in caplog.records)


def _part_task(i):
    time.sleep(0.01)
    return {"part": [i], "x": i, "timing": {"total": 0.01, "read": 0.0, "cache_hit": False}}


def _combine(*parts):
    return sorted(sum(parts, []))


def test_parking_with_the_combine_tree(monkeypatch, tmp_path):
    """CG steps sum their parts on the workers (Reduce): every part once, and the step's record."""
    distributed = pytest.importorskip("distributed")
    import dask

    from dfeols.executors import Reduce

    monkeypatch.setenv("DFEOLS_PARK_SCHEDULE", "2")
    got = {}
    red = Reduce(get=lambda r: r["part"], strip=lambda r: {k: v for k, v in r.items() if k != "part"},
                 combine=_combine, done=lambda p: got.setdefault("sum", p))
    n = 30
    with (
        dask.config.set({"distributed.scheduler.work-stealing": False}),
        distributed.LocalCluster(n_workers=3, threads_per_worker=1, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        c.wait_for_workers(3)
        ex = DaskExecutor(c, max_workers=3, tick=0.05)
        prog = Progress(tmp_path, label="t", every=1e9)
        prog.verbosity = 0
        prog.add_stage("cg1", {i: 1 for i in range(n)})
        prog.start("cg1")
        ex.run(_part_task, [(i, 1, (i,)) for i in range(n)], lambda t, r: None, prog, "cg1", reduce=red,
               affinity="f")
    assert got["sum"] == list(range(n))
    rec = json.loads((tmp_path / "probe.jsonl").read_text())
    assert rec["active"] == 2 and rec["pool"] == 3 and rec["tasks"] == n
