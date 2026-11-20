"""Workers retiring at their lifetime get replacements asked for ahead, by the queue waits seen
(OSCER, 2026-11-20: all 40 workers retired within 20 min and the pool took ~50 min to refill)."""

import asyncio
from types import SimpleNamespace

from dfeols import cluster as cmod
from dfeols import progress
from dfeols.cluster import queue_lead, retiring

LIFE, STAG = 3 * 3600.0, 300.0


def test_queue_lead_uses_seen_waits_with_prior_and_cap():
    assert queue_lead([], prior=1800, life=LIFE) == 1.25 * 1800 + 300
    assert queue_lead([600.0] * 4 + [3000.0], prior=1800, life=LIFE) == 1.25 * 3000 + 300
    assert queue_lead([20000.0], prior=1800, life=LIFE) == LIFE / 2


def test_retiring_window_and_work_left():
    t0 = 0.0
    early = t0 + LIFE - STAG
    lead = 1800.0
    w = {"a": t0}
    assert retiring(w, early - lead - 1, LIFE, STAG, lead, None) == set()
    assert retiring(w, early - lead + 1, LIFE, STAG, lead, None) == {"a"}
    # still counted while it may yet retire, then not (late: adapt sees any extra as surplus)
    assert retiring(w, t0 + LIFE + STAG + 599, LIFE, STAG, lead, None) == {"a"}
    assert retiring(w, t0 + LIFE + STAG + 601, LIFE, STAG, lead, None) == set()
    # the work ends before it retires: no replacement
    assert retiring(w, early - 600, LIFE, STAG, lead, work_left=300) == set()
    assert retiring(w, early - 600, LIFE, STAG, lead, work_left=4 * 3600) == {"a"}


class FakeAdaptive:
    def __init__(self, target):
        self.plan, self.observed, self.target = set(), set(), target

    async def safe_target(self):
        return self.target


def test_target_rises_ahead_and_falls_when_the_old_worker_leaves(monkeypatch):
    monkeypatch.setattr(progress, "WORK_LEFT", None)
    now = [0.0]
    ad = FakeAdaptive(target=2)
    cl = SimpleNamespace(_adaptive=ad, dfeols_lifetime=(LIFE, STAG))
    cmod._replace_ahead(cl, max_workers=2, prior_wait=1800, clock=lambda: now[0])
    tgt = lambda: asyncio.run(ad.safe_target())  # noqa: E731

    ad.plan = {"w0", "w1"}
    assert tgt() == 2
    now[0] = 600.0  # both connect after a 10 min wait
    ad.observed = {"w0", "w1"}
    assert tgt() == 2
    assert cl.dfeols_replace_ahead["waits"] == [600.0, 600.0]
    lead = queue_lead([600.0, 600.0], 1800, LIFE)
    now[0] = 600.0 + LIFE - STAG - lead + 1
    assert tgt() == 4  # both retire soon: two replacements asked for now
    ad.plan = {"w0", "w1", "w2", "w3"}
    assert tgt() == 4
    now[0] += lead  # w0 retires and leaves; w2 has arrived
    ad.plan, ad.observed = {"w1", "w2", "w3"}, {"w1", "w2"}
    assert tgt() == 3  # only w1 still to cover: no third replacement
    ad.plan, ad.observed = {"w2", "w3"}, {"w2", "w3"}
    assert tgt() == 2


def test_no_replacements_when_the_run_ends_first(monkeypatch):
    now = [0.0]
    ad = FakeAdaptive(target=1)
    cl = SimpleNamespace(_adaptive=ad, dfeols_lifetime=(LIFE, STAG))
    cmod._replace_ahead(cl, max_workers=1, prior_wait=1800, clock=lambda: now[0])
    ad.plan = ad.observed = {"w0"}
    asyncio.run(ad.safe_target())
    now[0] = LIFE - STAG - 200  # inside the lead (no wait seen: 5 min)
    monkeypatch.setattr(progress, "WORK_LEFT", (now[0], 120.0))
    assert asyncio.run(ad.safe_target()) == 1
    monkeypatch.setattr(progress, "WORK_LEFT", (now[0], 5 * 3600.0))
    assert asyncio.run(ad.safe_target()) == 2
    # a stale report (no status line for 10 min) counts as unknown: work to come
    monkeypatch.setattr(progress, "WORK_LEFT", (now[0] - 700, 120.0))
    assert asyncio.run(ad.safe_target()) == 2


def test_executor_installs_it_on_a_jobqueue_cluster(monkeypatch, tmp_path):
    from dataclasses import replace

    import dfeols

    monkeypatch.setenv("DFEOLS_CONFIG", str(tmp_path / "none.toml"))
    p = replace(dfeols.profile("local", scratch=str(tmp_path)), scheduler="slurm", interface=None)
    cl = cmod.make_cluster(p)
    try:
        cl.adapt(minimum_jobs=0, maximum_jobs=1)
        cmod._replace_ahead(cl, 1, 1800)
        assert "dfeols_replace_ahead" in vars(cl)
        assert cl.dfeols_lifetime[0] > 0
    finally:
        cl.close()


def test_retire_spread_and_allowed_failures(monkeypatch, tmp_path):
    """Workers retire over a window below the walltime margin (OSCER H L2, 2026-11-20: 40 workers
    retired within 20 min); Dask allows 20 failures a task, not 3 (KilledWorker after 4)."""
    import re
    from dataclasses import replace

    import dfeols

    assert cmod.retire_spread_seconds(11 * 3600) == int(0.1 * 11 * 3600)
    assert cmod.retire_spread_seconds(1800) == 300
    assert cmod.allowed_failures(40) == 20  # fails on the 21st: half the pool plus one
    assert cmod.allowed_failures(2) == 3 and cmod.allowed_failures(40, 7) == 7
    assert cmod.retire_spread_seconds(3600, minutes=60) == 900  # at most a quarter
    monkeypatch.setenv("DFEOLS_CONFIG", str(tmp_path / "none.toml"))
    p = replace(dfeols.profile("local", scratch=str(tmp_path)), scheduler="slurm", interface=None)
    p = p.with_overrides(walltime="12:00:00", retire_minutes=60)
    cl = cmod.make_cluster(p)
    try:
        js = cl.job_script()
        center = float(re.search(r"--lifetime (\d+)s", js).group(1))
        stag = float(re.search(r"--lifetime-stagger (\d+)s", js).group(1))
        assert center + stag == 11 * 3600  # never past walltime minus retire_minutes
        assert stag == int(0.1 * 11 * 3600)
        assert cl.dfeols_lifetime == (center, stag)
        assert cl.scheduler.allowed_failures == cmod.allowed_failures(p.worker["max_workers"])
    finally:
        cl.close()
    cl = cmod.make_cluster(p.with_overrides(allowed_failures=7, retire_spread_minutes=10))
    try:
        assert "--lifetime-stagger 600s" in cl.job_script()
        assert cl.scheduler.allowed_failures == 7
    finally:
        cl.close()
