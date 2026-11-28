"""A worker job whose Dask process keeps getting killed is cancelled, and its node avoided
(OSCER H L3 36464754, 2026-11-27: node c880's workers were SIGKILLed 316 times; OSCER doesn't
enforce memory requests, so another job overrunning its request ran the node out of memory)."""

import asyncio
import os
import time
from dataclasses import replace
from types import SimpleNamespace

import dfeols
from dfeols import cluster as cmod
from dfeols.cluster import _restart_guard, exclude_directives, exclude_nodes, make_cluster, restarts_due


def test_exclude_directives_widen_or_add():
    assert exclude_directives(["--exclusive"], ["c880"]) == ["--exclusive", "--exclude=c880"]
    assert exclude_directives(["--exclude=c922", "--qos=x"], ["c880"]) == ["--qos=x", "--exclude=c922,c880"]
    assert exclude_directives(["-x c922"], ["c922", "c880"]) == ["--exclude=c922,c880"]
    assert exclude_directives(["--exclude c1,c2"], []) == ["--exclude=c1,c2"]
    assert exclude_directives([], []) == []


def test_restarts_due_counts_restarts_in_the_window():
    # the first registration is the job's start, not a restart
    assert not restarts_due([0], 0, 1, 60)
    assert restarts_due([0, 10], 10, 1, 60)
    assert not restarts_due([0, 10, 20], 20, 3, 60)
    assert restarts_due([0, 10, 20, 30], 30, 3, 60)
    assert not restarts_due([0, 10, 20, 100], 100, 3, 60)  # two of them too long ago
    assert not restarts_due([0, 1, 2, 3, 4], 4, 0, 60)  # 0: off


def test_exclusion_reaches_the_next_job_script(monkeypatch, tmp_path):
    monkeypatch.setenv("DFEOLS_CONFIG", str(tmp_path / "none.toml"))
    p = replace(dfeols.profile("local", scratch=str(tmp_path)), scheduler="slurm", interface=None,
                extra_directives=["--exclude=c922"])
    cl = make_cluster(p)
    try:
        assert "#SBATCH --exclude=c922\n" in cl.job_script()
        exclude_nodes(cl, ["c880"])
        assert "#SBATCH --exclude=c922,c880\n" in cl.job_script()
        assert cl.job_script().count("--exclude") == 1
    finally:
        cl.close()


class FakeJob:
    def __init__(self, job_id):
        self.job_id = job_id
        self.closed = False

    async def close(self):
        self.closed = True


class FakeCluster:
    def __init__(self, names):
        self.options = {"job_extra_directives": ["--exclude=c922"]}
        self.new_spec = {"cls": object, "options": self.options}
        self.worker_spec = {n: self.new_spec for n in names}
        self.workers = {n: FakeJob(str(100 + i)) for i, n in enumerate(names)}
        self.plugins = []
        self.scheduler = SimpleNamespace(add_plugin=self.plugins.append)
        self.corrected = 0

    async def _correct_state(self):
        self.corrected += 1
        for n in self.worker_spec:  # what SpecCluster does: a job for each spec without one
            self.workers.setdefault(n, FakeJob(f"new-{n}"))


def _register(plugin, name, addr):
    sched = SimpleNamespace(workers={addr: SimpleNamespace(name=name)})
    plugin.add_worker(scheduler=sched, worker=addr)


def test_a_crash_looping_job_is_cancelled_and_its_node_excluded():
    cl = FakeCluster(["w0", "w1"])
    nodes = {"100": "c880", "101": "c900"}
    clock = [0.0]

    async def go():
        _restart_guard(cl, limit=3, window=1800, node_of=nodes.get, clock=lambda: clock[0])
        g = cl.plugins[0]
        old = cl.workers["w0"]
        _register(g, "w1", "tcp://b:1")  # a healthy worker: one start
        for i in range(3):  # start + 2 restarts: not yet
            clock[0] += 30
            _register(g, "w0", f"tcp://a:{i}")
        await asyncio.sleep(0.05)
        assert not old.closed and cl.dfeols_restart_guard["cancelled"] == 0
        clock[0] += 30
        _register(g, "w0", "tcp://a:9")  # third restart
        for _ in range(100):
            await asyncio.sleep(0.01)
            if cl.dfeols_restart_guard["cancelled"]:
                break
        assert old.closed and cl.corrected == 1
        assert cl.workers["w0"].job_id == "new-w0" and not cl.workers["w1"].closed
        assert cl.options["job_extra_directives"] == ["--exclude=c922,c880"]
        assert cl.dfeols_restart_guard["excluded"] == ["c880"]
        # the replacement job starts afresh: its own first registration isn't a restart
        _register(g, "w0", "tcp://c:1")
        await asyncio.sleep(0.05)
        assert cl.dfeols_restart_guard["cancelled"] == 1

    asyncio.run(go())


def test_exclusions_are_capped_and_restarts_spread_out_are_left_alone():
    cl = FakeCluster(["w0", "w1", "w2"])
    nodes = {"100": "n0", "101": "n1", "102": "n2"}
    clock = [0.0]

    async def go():
        _restart_guard(cl, limit=2, window=600, node_of=nodes.get, max_excluded=2, clock=lambda: clock[0])
        g = cl.plugins[0]
        # w2 restarts every 20 min: never 2 within 10 min
        for i in range(5):
            clock[0] += 1200
            _register(g, "w2", f"tcp://s:{i}")
        for n in ("w0", "w1"):
            for i in range(3):
                clock[0] += 10
                _register(g, n, f"tcp://{n}:{i}")
        clock[0] += 10
        for i in range(3):  # w2 now loops fast too, but two nodes are already excluded
            clock[0] += 10
            _register(g, "w2", f"tcp://f:{i}")
        for _ in range(100):
            await asyncio.sleep(0.01)
            if cl.dfeols_restart_guard["cancelled"] == 3:
                break
        st = cl.dfeols_restart_guard
        assert st["cancelled"] == 3
        assert st["excluded"] == ["n0", "n1"]
        assert cl.options["job_extra_directives"] == ["--exclude=c922,n0,n1"]

    asyncio.run(go())


async def _die():
    os._exit(1)


def test_local_nanny_restarts_get_the_worker_replaced():
    from distributed import Client, LocalCluster

    with LocalCluster(n_workers=0, threads_per_worker=1, processes=True, dashboard_address=None) as cl, \
            Client(cl) as c:
        _restart_guard(cl, limit=2, window=600, node_of=None)  # installed before workers start, as in executor()
        cl.scale(1)
        c.wait_for_workers(1)
        t0 = time.time()
        while not cl.workers and time.time() - t0 < 30:
            time.sleep(0.05)
        old = next(iter(cl.workers.values()))
        for _ in range(2):
            addr = next(iter(cl.scheduler.workers))
            c.run(_die, workers=[addr], wait=False)
            t0 = time.time()
            while time.time() - t0 < 30:  # the nanny brings it back under a new address
                ws = dict(cl.scheduler.workers)
                if ws and addr not in ws:
                    break
                time.sleep(0.1)
        t0 = time.time()
        while time.time() - t0 < 30 and not cl.dfeols_restart_guard["cancelled"]:
            time.sleep(0.1)
        assert cl.dfeols_restart_guard["cancelled"] == 1
        c.wait_for_workers(1, timeout=30)
        t0 = time.time()
        while time.time() - t0 < 30 and not [w for w in cl.workers.values() if w is not old]:
            time.sleep(0.05)
        assert old not in cl.workers.values() and len(cl.workers) == 1
