"""collinear_keep is blocked (right-looking, panels of columns) for thousands of
crossing-FE dummies; it must keep exactly the columns the column-by-column
incremental Cholesky keeps (fixest's order)."""

import numpy as np
import pytest
from scipy.linalg import solve_triangular

from dfeols.linalg import collinear_keep


def _reference(ZtZ, tol=1e-9):
    k = ZtZ.shape[0]
    keep = np.zeros(k, dtype=bool)
    L = np.zeros((k, k))
    m, kept = 0, []
    for j in range(k):
        djj = ZtZ[j, j]
        if not djj > 0:
            continue
        l = solve_triangular(L[:m, :m], ZtZ[kept, j], lower=True) if m else np.zeros(0)
        r = djj - l @ l
        if r <= tol * djj:
            continue
        L[m, :m], L[m, m] = l, np.sqrt(r)
        m += 1
        kept.append(j)
        keep[j] = True
    return keep


@pytest.mark.parametrize("seed", range(40))
def test_blocked_matches_column_by_column(seed):
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(5, 80)), int(rng.integers(1, 40))
    X = rng.standard_normal((n, k))
    for _ in range(rng.integers(0, 6)):  # zero, duplicate and linear-combination columns
        j, t = int(rng.integers(k)), int(rng.integers(3))
        if t == 0:
            X[:, j] = 0
        elif t == 1:
            X[:, j] = X[:, rng.integers(k)]
        else:
            X[:, j] = X[:, : max(j, 1)] @ rng.standard_normal(max(j, 1))
    G = X.T @ X
    ref = _reference(G)
    for block in (1, 3, 7, 256):
        assert (collinear_keep(G, block=block) == ref).all()


def test_dummies_with_intercept_drop_one_level():
    rng = np.random.default_rng(1)
    n, L = 5000, 600
    D = np.zeros((n, L))
    D[np.arange(n), rng.integers(0, L, n)] = 1
    X = np.c_[np.ones(n), rng.standard_normal((n, 3)), D, D[:, :5] + D[:, 5:10]]
    G = X.T @ X
    keep = collinear_keep(G, block=64)
    assert (keep == _reference(G)).all()
    assert keep[:4].all() and (~keep).sum() == 1 + 5  # one level, and the 5 sums of two levels
