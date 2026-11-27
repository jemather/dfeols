"""The IO ceiling: once a bigger pool stopped adding rows/s, workers retiring above the smaller
one aren't replaced (D11 H on OSCER, /scratch-bound: 23-24 workers 28-30M rows/s, 40 26-27M,
74-79 ~22M)."""

import asyncio
from types import SimpleNamespace

from dfeols import cluster as cmod
from dfeols import progress
from dfeols.cluster import PoolCeiling
from dfeols.progress import stage_kind

M = 1e6


def S(t, workers, rate, kind="cg"):
    return {"t": float(t), "kind": kind, "workers": float(workers), "rate": rate * M}


def test_clip_never_cancels_and_replaces_only_below_the_ceiling():
    c = PoolCeiling()
    assert c.clip(40, 4, 40, 4) == (44, 4)  # no ceiling: replace-ahead as before
    c.ceiling = 30
    assert c.clip(40, 0, 40, 0) == (40, 0)  # 40 running: none cancelled
    assert c.clip(40, 4, 40, 4) == (40, 0)  # 4 retiring leave 36: no replacements
    assert c.clip(40, 0, 39, 0) == (39, 0)  # one retired: not replaced
    assert c.clip(40, 3, 30, 3) == (33, 3)  # at the ceiling: its retiring workers are replaced
    assert c.clip(40, 0, 33, 3) == (33, 0)  # those replacements, queued, are kept
    assert c.clip(40, 0, 25, 0) == (30, 0)  # below it: adapt fills up to it
    assert c.clip(10, 2, 40, 0) == (12, 2)  # adapt scaling down while idle is untouched
    c.min_workers = 35
    assert c.clip(40, 0, 32, 0) == (35, 0)  # never below min_workers


def test_set_lower_and_lift():
    c = PoolCeiling(min_workers=1)
    c.add(S(0, 40, 26.5))
    assert c.ceiling is None
    c.add(S(600, 44, 27.5))  # 10% more workers: not enough to judge
    assert c.ceiling is None
    c.add(S(1200, 48, 26.8))  # 20% more for 1% more rows/s
    assert c.ceiling == 40 and c.saturated == ("cg", 26.8 * M)
    c.add(S(1800, 30, 26.0))  # 40 gained 2% over 30: lowered to 30
    assert c.ceiling == 30
    c.add(S(2400, 30, 28.0))  # within the noise: stays
    assert c.ceiling == 30
    c.add(S(3000, 30, 34.0))  # 27% over the saturated 26.8M: the filesystem got faster
    assert c.ceiling is None and len(c.samples) == 1


def test_a_bigger_pool_that_gains_sets_nothing():
    c = PoolCeiling()
    c.add(S(0, 20, 15.0))
    c.add(S(600, 30, 22.0))
    c.add(S(1200, 40, 26.5))
    assert c.ceiling is None


def test_pass_types_and_age_kept_apart():
    c = PoolCeiling(pair_s=7200)
    c.add(S(0, 40, 26.5, kind="round1"))
    c.add(S(600, 50, 26.5, kind="cg"))
    assert c.ceiling is None  # different pass types
    c.add(S(9000, 60, 26.5, kind="cg"))  # 2.3 h after the 50-worker cg sample: too far apart
    assert c.ceiling is None
    c.add(S(9600, 50, 26.5, kind="cg"))
    assert c.ceiling == 50


def test_lift_needs_the_same_pass_type_and_a_pool_at_or_below_it():
    c = PoolCeiling()
    c.add(S(0, 30, 27.0))
    c.add(S(600, 40, 27.0))
    assert c.ceiling == 30
    c.add(S(1200, 30, 40.0, kind="round2"))
    assert c.ceiling == 30
    c.add(S(1800, 34, 40.0))  # above the ceiling (queued ones arrived): not evidence
    assert c.ceiling == 30


class Passes:
    def __init__(self):
        self.d = None

    def __call__(self):
        return self.d

    def set(self, t, rows, tasks, run=1, kind="cg", queued=100):
        self.d = {"t": t, "run": run, "kinds": {kind: (rows, tasks)}, "stage": kind, "queued": queued}


def _feed(c, ps, t0, t1, workers, rate, run=1, queued=100, step=10.0, rows0=0, tasks0=0, kind="cg"):
    t = t0
    while t <= t1:
        w = workers(t) if callable(workers) else workers
        ps.set(t, rows0 + rate * (t - t0), tasks0 + int((t - t0) / 5), run=run, queued=queued, kind=kind)
        c.observe(t, w)
        t += step
    return rows0 + rate * (t1 - t0), tasks0 + int((t1 - t0) / 5)


def test_windows_become_samples_only_when_steady_and_busy():
    ps = Passes()
    c = PoolCeiling(window_s=600, passes=ps)
    _feed(c, ps, 0, 610, 40, 26.5 * M)
    assert len(c.samples) == 1
    s = c.samples[0]
    assert s["workers"] == 40 and abs(s["rate"] - 26.5 * M) < 1e3 and s["kind"] == "cg"
    # the pool swings 30-40 within the window (retirement churn): no sample
    c = PoolCeiling(window_s=600, passes=ps)
    _feed(c, ps, 0, 610, lambda t: 30 if t % 100 < 50 else 40, 26.5 * M)
    assert c.samples == []
    # a stage's tail, nothing waiting to run: no sample
    c = PoolCeiling(window_s=600, passes=ps)
    _feed(c, ps, 0, 610, 40, 26.5 * M, queued=0)
    assert c.samples == []


def test_observe_sets_the_ceiling_from_the_pool_drifting():
    ps = Passes()
    c = PoolCeiling(window_s=600, passes=ps)
    r, k = _feed(c, ps, 0, 610, 40, 26.5 * M)
    # the window the pool grew in is unsteady and dropped; the next one at 50 counts
    _feed(c, ps, 620, 1210, 50, 26.7 * M, rows0=r, tasks0=k)
    assert c.ceiling is None
    _feed(c, ps, 1220, 1900, 50, 26.7 * M, rows0=r + 26.7 * M * 590, tasks0=k + 118)
    assert c.ceiling == 40
    # a new fit: the ceiling comes off
    _feed(c, ps, 1910, 1950, 50, 26.7 * M, run=2)
    assert c.ceiling is None and c.samples == []


def test_stale_counters_reset_the_window():
    ps = Passes()
    c = PoolCeiling(window_s=600, passes=ps)
    ps.set(0, 0, 0)
    c.observe(0, 40)
    assert c.window is not None
    c.observe(PoolCeiling.STALE_S + 1, 40)  # no status tick for 3 min
    assert c.window is None


class FakeAdaptive:
    def __init__(self, target):
        self.plan, self.observed, self.target = set(), set(), target

    async def safe_target(self):
        return self.target


def test_with_replace_ahead_retiring_workers_above_it_go_unreplaced(monkeypatch):
    LIFE, STAG = 3 * 3600.0, 300.0
    monkeypatch.setattr(progress, "WORK_LEFT", None)
    now = [0.0]
    ad = FakeAdaptive(target=40)
    cl = SimpleNamespace(_adaptive=ad, dfeols_lifetime=(LIFE, STAG))
    cmod._replace_ahead(cl, max_workers=40, prior_wait=1800, clock=lambda: now[0])
    ceil = cmod._pool_ceiling(cl, {"min_workers": 1}, passes=lambda: None)
    assert cl.dfeols_pool_ceiling is ceil
    tgt = lambda: asyncio.run(ad.safe_target())  # noqa: E731
    old = {f"w{i}" for i in range(40)}
    ad.plan = set(old)
    tgt()
    now[0] = 480.0
    ad.observed = set(old)
    tgt()
    ceil.ceiling = 30
    now[0] = 480.0 + LIFE - STAG - 60  # all 40 come due
    assert tgt() == 40  # none cancelled, none replaced ahead
    assert cl.dfeols_replace_ahead["extra"] == 0
    left = set(sorted(old)[:12])  # 12 retire: 28 left, all still due
    ad.plan = ad.observed = old - left
    assert tgt() == 34  # back up to the ceiling, plus 4 replacements ahead (the spares cap)
    assert cl.dfeols_replace_ahead["extra"] == 4


def test_without_replace_ahead_it_wraps_the_target():
    ad = FakeAdaptive(target=40)
    cl = SimpleNamespace(_adaptive=ad)
    ceil = cmod._pool_ceiling(cl, {"min_workers": 2, "io_ceiling_window_minutes": 5}, passes=lambda: None)
    assert ceil.window_s == 300 and ceil.min_workers == 2
    ad.plan = ad.observed = {f"w{i}" for i in range(36)}
    assert asyncio.run(ad.safe_target()) == 40
    ceil.ceiling = 30
    assert asyncio.run(ad.safe_target()) == 36
    ad.plan = ad.observed = {f"w{i}" for i in range(29)}
    assert asyncio.run(ad.safe_target()) == 30


def test_executor_installs_it_by_default_and_a_profile_turns_it_off(monkeypatch, tmp_path):
    from dataclasses import replace

    import dfeols

    monkeypatch.setenv("DFEOLS_CONFIG", str(tmp_path / "none.toml"))
    p = replace(dfeols.profile("local", scratch=str(tmp_path)), scheduler="slurm", interface=None)
    assert p.worker["io_ceiling"] is True
    for on in (True, False):
        q = p.with_overrides(io_ceiling=on)
        cl = cmod.make_cluster(q)
        try:
            cl.adapt(minimum_jobs=0, maximum_jobs=1)
            cmod._replace_ahead(cl, 1, 1800)
            if q.worker["io_ceiling"]:
                cmod._pool_ceiling(cl, q.worker)
            assert hasattr(cl, "dfeols_pool_ceiling") is on
        finally:
            cl.close()


def test_progress_publishes_passes(monkeypatch):
    assert stage_kind("cg12") == "cg" and stage_kind("singletons3") == "singletons"
    assert stage_kind("round1") == "round1" and stage_kind("cgx") == "cgx"
    monkeypatch.setattr(progress, "PASSES", None)
    pr = progress.Progress(None)
    pr.verbosity = 0
    pr.add_stage("cg1", {i: 100 for i in range(10)})
    pr.start("cg1")
    pr.running["cg1"] = {0, 1, 2}
    pr.task_done("cg1", 0, 1.0)
    pr.tick(force=True)
    ps = progress.PASSES
    assert ps["run"] == pr.run_id and ps["stage"] == "cg" and ps["kinds"]["cg"] == (100, 1)
    assert ps["queued"] == 10 - 1 - 2
    pr.finish("cg1", log=False)
    pr.add_stage("cg2", {i: 100 for i in range(10)})
    pr.start("cg2")
    pr.task_done("cg2", 0, 1.0)
    pr.tick(force=True)
    assert progress.PASSES["kinds"]["cg"] == (200, 2)  # steps add up in one pass type
    assert progress.Progress(None).run_id != pr.run_id
