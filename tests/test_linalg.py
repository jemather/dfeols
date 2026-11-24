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


def test_collinear_keep_drops_noise_columns_against_reference():
    # a column an FE absorbs is rounding noise net of it: against its own diagonal it looks
    # independent; against its sum of squares before the FEs it is dropped
    import numpy as np

    from dfeols.linalg import absorbed, collinear_keep, reference_ss

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    X[:, 1] = rng.normal(scale=1e-15, size=200)  # what an absorbed column looks like net of the FE
    G = X.T @ X
    assert collinear_keep(G).tolist() == [True, True, True]
    raw = np.c_[X[:, 0], 5 + rng.normal(size=200), X[:, 2]]  # before the FE it varied
    w = np.ones(200)
    ref = reference_ss(w @ raw, w @ (raw * raw), w.sum())
    assert collinear_keep(G, ref_diag=ref).tolist() == [True, False, True]
    assert absorbed(np.diag(G), ref).tolist() == [False, True, False]
    assert reference_ss(np.array([200.0]), np.array([200.0]), 200.0).tolist() == [0.0]  # a constant


def test_collinear_keep_tightens_until_the_kept_gram_factors(monkeypatch, caplog):
    """If the kept columns pass the pivot test but don't factor (rounding in an ill-conditioned
    Gram, e.g. W&Z Table 4 isic 40's IRLS step with near-zero weights), the test is redone with
    tol x10 until they do; the later of a near-dependent pair is dropped, with a warning."""
    import logging

    import dfeols.linalg as L

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    X[:, 3] = X[:, 2] + 1e-4 * rng.normal(size=200)  # nearly dependent on column 2
    G = X.T @ X
    real = L.cho_factor

    def fussy(A, lower=True):  # stand-in for the rounding failure: fails while col 3 is in
        if A.shape[0] == 4:
            raise L.LinAlgError("not positive definite")
        return real(A, lower=lower)

    monkeypatch.setattr(L, "cho_factor", fussy)
    with caplog.at_level(logging.WARNING, logger="dfeols.linalg"):
        keep = L.collinear_keep(G, 1e-9)
    assert keep.tolist() == [True, True, True, False]
    assert "tightened" in caplog.text

    monkeypatch.setattr(L, "cho_factor", lambda A, lower=True: (_ for _ in ()).throw(L.LinAlgError("x")))
    with pytest.raises(L.LinAlgError, match="numerically singular"):
        L.collinear_keep(G, 1e-9)


def test_collinear_keep_spread_dependency_under_extreme_weights():
    """Exporter and importer dummies (sum of one set = sum of the other) under weights spanning
    ~1e10: the pivot test keeps every column, and the kept Gram still factors by Cholesky, but
    it is singular; the eigenvalue check must catch it and drop one dummy (W&Z isic 7)."""
    from scipy.linalg import eigvalsh

    rng = np.random.default_rng(84)
    n, a, b = 3000, 60, 60
    i, j = rng.integers(0, a, n), rng.integers(0, b, n)
    w = np.exp(rng.normal(0, 6, n))
    D = np.zeros((n, a + b))
    D[np.arange(n), i] = 1
    D[np.arange(n), a + j] = 1
    Z = np.column_stack([D, rng.normal(size=n)])
    G = (Z * w[:, None]).T @ Z
    from dfeols.linalg import _collinear_keep

    assert _collinear_keep(G, 1e-9).sum() == a + b + 1  # the pivot test alone misses it
    keep = collinear_keep(G, 1e-9)
    assert keep.sum() == a + b and keep[-1]  # one dummy dropped, the regressor kept
    K = G[np.ix_(keep, keep)]
    s = 1 / np.sqrt(np.diag(K))
    ev = eigvalsh(K * s[:, None] * s[None, :])
    assert ev[0] / ev[-1] > 1e-12
