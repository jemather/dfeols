"""Plan C's parts of A within a worker's memory (taxi2009, OSCER 2026-11-09: the full-year fit
died in round 1, four 24 GB workers killed summing shards' parts whose limit came from the
coordinator's memory, and its rerun died in round 2, which rebuilt each shard's part of A when
it recomputed round 1). Shards cap their part by the worker's memory, combines add their
inputs one at a time and fall back to the diagonal blocks, then to nothing, before passing
it, and round 2 builds no part of A. Plus: round 2's linear maps go to the workers once."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

import dfeols
from dfeols import crossing as cx
from dfeols import worker as wk
from dfeols.executors import DaskExecutor

sys.path.insert(0, str(Path(__file__).parent))
from simdata import sim_panel

OFFS = np.array([0, 30, 50])


def _parts(n=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r, c = rng.integers(0, 50, 80), rng.integers(0, 50, 80)
        S = sp.csr_matrix((rng.random(80), (r, c)), shape=(50, 50))
        S = (S + S.T).tocoo()
        out.append(
            {
                "full": True,
                "blocks": True,
                "nnz": S.nnz,
                "r": S.row.astype(np.int32),
                "c": S.col.astype(np.int32),
                "v": S.data,
            }
        )
    return out


def _peak(parts, blocks: bool) -> int:
    """Largest (sum so far + next part) nonzeros while combine_parts adds the parts."""
    fe_of = np.repeat(np.arange(2), np.diff(OFFS))
    S, peak = None, 0
    for p in parts:
        M = cx.part_matrix(p, 50)
        if blocks:
            M = cx._blocks_only(M, fe_of)
        peak = max(peak, (S.nnz if S is not None else 0) + M.nnz)
        S = M if S is None else S + M
    return peak


@pytest.mark.parametrize("room", ["all", "blocks", "none"])
def test_combine_within_worker_memory(room):
    parts = _parts()
    inputs = sum(cx._part_bytes(p) for p in parts)
    full, blocks = _peak(parts, False), _peak(parts, True)
    assert blocks < full
    n = {"all": full, "blocks": (full + blocks) // 2, "none": blocks - 1}[room]
    wb = (inputs + cx.MERGE_BYTES * n) / cx.WORKER_SHARE
    out = cx.combine_parts(*parts, offs=OFFS, max_nnz=10**9, worker_bytes=wb)
    ref = cx.combine_parts(*parts, offs=OFFS, max_nnz=10**9)  # no worker limit: all of A
    if room == "all":
        assert out["full"] and out["nnz"] == ref["nnz"]
        np.testing.assert_array_equal(out["data"], ref["data"])
    elif room == "blocks":
        assert not out["full"] and out["blocks"] and "worker" in out["reason"]
        fe_of = np.repeat(np.arange(2), np.diff(OFFS))
        want = cx._blocks_only(cx.part_matrix(ref, 50), fe_of)
        got = cx.part_matrix(out, 50)
        assert abs(got - want).max() < 1e-12 and got.nnz == want.nnz
    else:
        assert not out["blocks"] and "worker" in out["reason"]
        # the coordinator then has no blocks either (LevelGram's rule for such a sum)
        g = cx.LevelGram(OFFS, ["a", "b"], 10**9)
        g.add(out, None)
        A, B = g.finish()
        assert A is None and B is None


def test_part_budget_from_worker_memory():
    assert cx.part_budget(24 * 10**9) == int(0.6 * 24e9 / (cx.NNZ_BYTES * 4))
    assert cx.worker_bytes(DaskExecutor(None, worker_memory=7 * 10**9)) == 7 * 10**9


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    d = sim_panel().with_columns(cw=(pl.col("id_f") % 100) * 100 + pl.col("week"))
    return dfeols.prep(
        d,
        shard_dir=tmp_path_factory.mktemp("s"),
        partition="id_a",
        fixed_effects=["spell", "cw", "week"],
        target_rows=3000,
    )


F = "eng ~ feat + posts | spell + cw + week"


def _fit(ds, path, **kw):
    return dfeols.fit(ds, F, workdir=str(path), crossing_fe="iterate", log_every=1e9, iter_tol=1e-12, **kw)


def test_shard_parts_capped_by_worker_memory(ds, tmp_path, monkeypatch):
    """Parts over the worker's cap leave level space (the pass-per-step fallback), same answer."""
    a = _fit(ds, tmp_path / "a")
    assert a.options["crossing_cg"]["mode"] == "level"
    monkeypatch.setattr(cx, "WORKER_SHARE", 1e-12)  # every part over the cap, blocks too
    with pytest.warns(cx.SlowConvergenceWarning, match="worker"):
        b = _fit(ds, tmp_path / "b")
    assert b.options["crossing_cg"]["mode"] == "pass"
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-9)
    np.testing.assert_allclose(a.vcov(), b.vcov(), rtol=1e-8)


def test_round2_builds_no_part_of_A(ds, tmp_path, monkeypatch):
    calls = []
    orig = wk._dummy_gram

    def spy(plan, *a, **kw):
        calls.append(plan.level_nnz)
        return orig(plan, *a, **kw)

    monkeypatch.setattr(wk, "_dummy_gram", spy)
    r = _fit(ds, tmp_path, cache=False)
    assert r.options["crossing_cg"]["mode"] == "level"
    assert len(calls) == len(ds.shards) and all(calls)  # round 1 only


class _Client:
    def __init__(self):
        self.sent = []

    def scatter(self, data, broadcast=False, hash=True):
        self.sent.append(data)
        return [f"future-{len(self.sent)}"]


def test_shared_dict_of_arrays_scattered_once():
    lin = {"A": np.zeros((10, 2)), "G": np.zeros((200_000, 3)), "fs": {"G": np.zeros(10)}}
    small = {"A": np.zeros(3)}
    ex = DaskExecutor(_Client())
    tasks = [(i, 1, ("plan", i, [f"f{i}"], lin, small)) for i in range(5)]
    shared = ex._scatter_shared(tasks)
    assert list(shared) == [id(lin)] and ex.client.sent == [[lin]]
