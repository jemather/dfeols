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
            DaskExecutor(c, max_workers=2, tick=0.1).run(_task, tasks, lambda t, r: seen.append(r), prog, "s", reduce=red)
    else:
        SerialExecutor().run(_task, tasks, lambda t, r: seen.append(r), prog, "s", reduce=red)
    assert sorted(got["sum"]) == list(range(n))  # every part exactly once
    assert len(seen) == n and all("part" not in r for r in seen)


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
    ds = dfeols.prep(d, shard_dir=tmp_path / "s", partition="id_a", fixed_effects=["spell", "cw", "week"], target_rows=3000)
    f = "eng ~ feat + posts | spell + cw + week"
    kw = dict(crossing_fe="iterate", log_every=1e9, level_space_gb=level_space_gb)
    a = dfeols.fit(ds, f, executor=_NoReduce(), workdir=str(tmp_path / "a"), **kw)
    distributed = pytest.importorskip("distributed")
    with (
        distributed.LocalCluster(n_workers=2, threads_per_worker=1, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        b = dfeols.fit(ds, f, executor=DaskExecutor(c, max_workers=2, tick=0.1), workdir=str(tmp_path / "b"), **kw)
    ca, cb = a.options["crossing_cg"], b.options["crossing_cg"]
    assert len(ds.shards) > 4 and ca["mode"] == cb["mode"] == ("level" if level_space_gb is None else "pass")
    assert ca["A_nnz"] == cb["A_nnz"] and ca["block_nnz"] == cb["block_nnz"] and ca["parts_nnz"] == cb["parts_nnz"]
    assert ca["shipped_nnz"] == ca["parts_nnz"] and cb["shipped_nnz"] <= ca["shipped_nnz"]
    if level_space_gb is None:
        assert cb["shipped_nnz"] == cb["A_nnz"] < ca["shipped_nnz"]  # the coordinator got A once
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-10)
