"""fit() refuses a shard that cannot fit in the profile's worker memory."""

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("dask")
from dfeols.estimate import _check_worker_memory


def _ex(mem):
    return SimpleNamespace(profile=SimpleNamespace(name="site", worker={"memory": mem}))


def test_refuses_far_over_worker_memory():
    with pytest.raises(MemoryError, match="993,842,500 rows"):
        _check_worker_memory(_ex("24GB"), "twfe", 993_842_500, 379.2)


def test_warns_slightly_over(caplog):
    with caplog.at_level(logging.WARNING, logger="dfeols"):
        _check_worker_memory(_ex("24GB"), "twfe", 50_000_000, 30.0)
    assert "may run out of memory" in caplog.text


def test_no_profile_no_check():
    _check_worker_memory(object(), "twfe", 10**9, 1e6)  # serial / plain Dask client: nothing known


def _stamp(i):
    import time

    t = time.time()
    time.sleep(0.4)
    return {"t": (t, time.time())}


@pytest.mark.parametrize("task_gb, together", [(1.0, True), (None, False)])
def test_tasks_share_a_worker_by_memory(task_gb, together):
    # a worker with 4 threads and 4 GB of MEMORY runs 1 GB tasks together (1.25 GB with the
    # margin: 3 at once), and tasks with no estimate one at a time (a whole worker each)
    distributed = pytest.importorskip("distributed")
    from dfeols.executors import DaskExecutor
    from dfeols.progress import Progress

    with (
        distributed.LocalCluster(
            n_workers=1, threads_per_worker=4, resources={"MEMORY": 4e9}, dashboard_address=None
        ) as lc,
        distributed.Client(lc) as c,
    ):
        ex = DaskExecutor(c, max_workers=1, tick=0.1, worker_memory=int(4e9))
        prog = Progress(None, label="t", every=1e9)
        prog.task_gb = task_gb
        prog.add_stage("s", {i: 1 for i in range(6)})
        prog.start("s")
        spans = []
        ex.run(_stamp, [(i, 1, (i,)) for i in range(6)], lambda tid, r: spans.append(r["t"]), prog, "s")
        assert ex.slots()[0] == 1  # workers, not threads
    at_once = max(sum(a <= s < b for a, b in spans) for s, _ in spans)  # running at each start
    assert at_once == (3 if together else 1)
