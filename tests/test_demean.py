"""Within-shard demeaning: compiled backend availability and agreement of all paths."""

import numpy as np
import pytest

from dfeols import demean as dm


def _problem(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 200, n)
    b = rng.integers(0, 30, n)
    c = rng.integers(0, 7, n)
    x = rng.normal(size=(n, 3)) + a[:, None] * 0.01 + b[:, None] * 0.1
    w = rng.uniform(0.5, 2.0, n)
    return x, [a, b, c], w


def test_compiled_demeaner_available():
    """pyfixest's demeaner is a private API; fail loudly (e.g. in the nightly
    job against the latest pyfixest) if dfeols can no longer find it."""
    pytest.importorskip("pyfixest")
    assert dm._pf_demean is not None, "pyfixest.core.demean moved; dfeols fell back to NumPy"


@pytest.mark.parametrize("k", [2, 3])
def test_compiled_matches_numpy(k):
    pytest.importorskip("pyfixest")
    x, fes, w = _problem()
    flist = np.column_stack([dm.densify(c)[0] for c in fes[:k]])
    ours, ok1 = dm.demean(x, fes[:k], w, tol=1e-12, method="iterate")
    ref, ok2 = dm._map_numpy(x, flist.astype(np.int64), w, 1e-12, 100_000)
    assert ok1 and ok2
    np.testing.assert_allclose(ours, ref, atol=1e-8)


def test_exact_two_way_matches_iteration():
    x, fes, w = _problem(seed=1)
    exact, _ = dm.demean(x, fes[:2], w, method="exact")
    it, ok = dm.demean(x, fes[:2], w, tol=1e-13, method="iterate")
    assert ok
    np.testing.assert_allclose(exact, it, atol=1e-8)
