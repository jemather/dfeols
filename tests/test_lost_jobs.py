"""Worker jobs that end without reaching the scheduler are replaced (OSCER, 2026-11-19: 16 of 40
workers missed the 60 s start timeout, and dask-jobqueue counted them as pending for good)."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import dfeols
from dfeols import cluster as cmod
from dfeols.cluster import lost_jobs, make_cluster


def test_lost_jobs_are_unconnected_and_gone():
    jobs = {"w0": "1", "w1": "2", "w2": "3", "w3": None}
    # w0 connected; w1 still queued; w2 gone without connecting; w3 not submitted yet
    assert lost_jobs(jobs, connected={"w0"}, live={"2"}) == {"w2"}


def test_worker_start_timeout_is_set(monkeypatch, tmp_path):
    monkeypatch.setenv("DFEOLS_CONFIG", str(tmp_path / "none.toml"))
    p = replace(dfeols.profile("local", scratch=str(tmp_path)), scheduler="slurm", interface=None)
    cl = make_cluster(p)
    try:
        assert "--death-timeout 600" in cl.job_script(), cl.job_script()
    finally:
        cl.close()
    p = p.with_overrides(start_minutes=3)
    cl = make_cluster(p)
    try:
        assert "--death-timeout 180" in cl.job_script()
    finally:
        cl.close()


class FakeCluster:
    def __init__(self, jobs, seen):
        self.workers = {n: SimpleNamespace(job_id=j) for n, j in jobs.items()}
        self.scheduler_info = {"workers": {f"tcp://x:{i}": {"name": n} for i, n in enumerate(seen)}}
        self.status = SimpleNamespace(name="running")
        self.corrected = 0
        self.loop = SimpleNamespace(add_callback=lambda f: None)

    async def _correct_state(self):
        self.corrected += 1


def test_replaced_after_two_misses():
    cl = FakeCluster({"w0": "1", "w1": "2", "w2": "3"}, seen={"w0"})
    cmod._replace_lost_jobs(cl, max_workers=3, live_jobs=lambda: {"1", "2"})
    check = cl.dfeols_lost_jobs_check
    asyncio.run(check())
    assert "w2" in cl.workers and cl.corrected == 0  # one miss: maybe squeue lagging
    asyncio.run(check())
    assert set(cl.workers) == {"w0", "w1"} and cl.corrected == 1
    assert cl.dfeols_lost_jobs["lost"] == 1


def test_stops_when_no_worker_ever_connects():
    cl = FakeCluster({"w0": "1", "w1": "2", "w2": "3"}, seen=set())
    cmod._replace_lost_jobs(cl, max_workers=1, live_jobs=lambda: set())
    check = cl.dfeols_lost_jobs_check
    asyncio.run(check())
    asyncio.run(check())
    st = cl.dfeols_lost_jobs
    assert st["lost"] == 3 and st["stopped"] and cl.corrected == 0


def test_squeue_failure_drops_nothing():
    cl = FakeCluster({"w0": "1"}, seen=set())
    cmod._replace_lost_jobs(cl, max_workers=1, live_jobs=lambda: None)
    for _ in range(3):
        asyncio.run(cl.dfeols_lost_jobs_check())
    assert set(cl.workers) == {"w0"}
