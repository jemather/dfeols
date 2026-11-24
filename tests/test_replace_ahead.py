"""Workers retiring at their lifetime get replacements asked for ahead, by the queue waits seen
(OSCER, 2026-11-20: all 40 workers retired within 20 min and the pool took ~50 min to refill)."""

import asyncio
from types import SimpleNamespace

from dfeols import cluster as cmod
from dfeols import progress
from dfeols.cluster import braked, queue_lead, retiring, spares_cap

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
    cmod._replace_ahead(cl, max_workers=2, prior_wait=1800, clock=lambda: now[0], max_spares=2)
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


def test_spares_cap_and_brake_rules():
    assert spares_cap(40) == 4 and spares_cap(5) == 1 and spares_cap(40, 0) == 0 and spares_cap(40, 7) == 7
    assert not braked([60.0], 180)  # one quick wait is not enough
    assert braked([600.0, 60.0, 90.0], 180)  # the last two were quick
    assert not braked([60.0, 90.0, 600.0], 180)  # a long one again: ask ahead


def _pool(n, t0):
    return {f"w{i}" for i in range(n)}


def test_a_pool_due_together_gets_at_most_the_cap(monkeypatch):
    """40 workers that started together all come due at once (H L2 36358879 asked for 40 at
    21:44 and ran 79 workers): only spares_cap of them get a replacement ahead; each one that
    leaves frees a place for the next."""
    monkeypatch.setattr(progress, "WORK_LEFT", None)
    now = [0.0]
    ad = FakeAdaptive(target=40)
    cl = SimpleNamespace(_adaptive=ad, dfeols_lifetime=(LIFE, STAG))
    cmod._replace_ahead(cl, max_workers=40, prior_wait=1800, clock=lambda: now[0])
    tgt = lambda: asyncio.run(ad.safe_target())  # noqa: E731
    old = _pool(40, 0)
    ad.plan = set(old)
    tgt()
    now[0] = 480.0  # an 8 min wait each: no brake
    ad.observed = set(old)
    assert tgt() == 40
    now[0] = 480.0 + LIFE - STAG - 60  # all 40 inside the lead
    assert tgt() == 44  # at most 4 ahead
    new = {f"r{i}" for i in range(4)}
    ad.plan = old | new
    tgt()
    now[0] += 480.0  # the 4 arrive after 8 min; one old worker leaves
    ad.observed = (old - {"w0"}) | new
    ad.plan = ad.observed
    assert tgt() == 44  # still 39 to cover: the place w0 left goes to a fifth replacement


def test_quick_waits_brake_asking_ahead(monkeypatch):
    """Two workers that connected within quick_wait of being asked for: none asked ahead, and a
    retiring worker is replaced after it leaves; a long wait again lifts the brake."""
    monkeypatch.setattr(progress, "WORK_LEFT", None)
    now = [0.0]
    ad = FakeAdaptive(target=2)
    cl = SimpleNamespace(_adaptive=ad, dfeols_lifetime=(LIFE, STAG))
    cmod._replace_ahead(cl, max_workers=2, prior_wait=1800, clock=lambda: now[0], max_spares=2, quick_wait=180)
    tgt = lambda: asyncio.run(ad.safe_target())  # noqa: E731
    ad.plan = {"w0", "w1"}
    tgt()
    now[0] = 60.0  # both connect within a minute
    ad.observed = {"w0", "w1"}
    tgt()
    now[0] = 60.0 + LIFE - STAG - 60  # both inside the lead
    assert tgt() == 2  # braked: nothing ahead
    assert cl.dfeols_replace_ahead["braked"]
    ad.plan = {"w0", "w1", "w2"}  # a replacement asked for after w0 left ...
    ad.observed = {"w1"}
    tgt()
    now[0] += 1200.0  # ... waits 20 min
    ad.observed = {"w1", "w2"}
    assert tgt() == 3  # the brake is off: w1 gets its replacement ahead
    assert not cl.dfeols_replace_ahead["braked"]


def test_a_wait_in_progress_counts():
    """H L1 (36391325): replacements waited 16 min to 3 h while the lead still planned for the
    14 min seen; a worker still in the queue now counts, for the lead, the cap and the brake."""
    assert queue_lead([600.0], prior=1800, life=LIFE, pending=3600.0) == 1.25 * 3600 + 300
    assert queue_lead([600.0], prior=1800, life=LIFE, pending=60.0) == 1.25 * 600 + 300
    assert spares_cap(40, wait=600.0) == 4  # up to 15 min: 10%
    assert spares_cap(40, wait=1800.0) == 8  # 30 min: twice
    assert spares_cap(40, wait=4 * 3600.0) == 20  # at most half
    assert spares_cap(40, 3, wait=4 * 3600.0) == 3  # a fixed cap stays fixed
    assert spares_cap(5, wait=4 * 3600.0) == 2
    assert not braked([60.0, 90.0], 180, pending=1200.0)  # quick so far, but one is waiting


def test_a_slow_queue_asks_earlier_and_for_more(monkeypatch):
    monkeypatch.setattr(progress, "WORK_LEFT", None)
    now = [0.0]
    ad = FakeAdaptive(target=40)
    life = 11 * 3600.0
    cl = SimpleNamespace(_adaptive=ad, dfeols_lifetime=(life, 3960.0))
    cmod._replace_ahead(cl, max_workers=40, prior_wait=1800, clock=lambda: now[0])
    tgt = lambda: asyncio.run(ad.safe_target())  # noqa: E731
    old = _pool(40, 0)
    ad.plan = set(old)
    tgt()
    now[0] = 480.0
    ad.observed = set(old)
    tgt()
    # the pool comes due (8 min waits seen): 4 asked for ahead
    now[0] = 480.0 + life - 3960.0 - 600
    assert tgt() == 44
    new = {f"r{i}" for i in range(4)}
    ad.plan = old | new
    tgt()
    # an hour on, none of the 4 has started: the cap is now 16, so 12 more are asked for
    now[0] += 3600.0
    assert tgt() == 40 + 16
    assert cl.dfeols_replace_ahead["cap"] == 16
