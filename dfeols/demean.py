"""Within-shard demeaning for fixed effects nested in the partition key."""

from __future__ import annotations

import numpy as np

try:  # Rust alternating-projections demeaner shipped with pyfixest
    from pyfixest.core.demean import demean as _pf_demean
except Exception:  # pragma: no cover
    _pf_demean = None


def densify(codes: np.ndarray) -> tuple[np.ndarray, int]:
    uniq, inv = np.unique(codes, return_inverse=True)
    return inv.astype(np.uint64), len(uniq)


def drop_singletons(fe_codes: list[np.ndarray]) -> np.ndarray:
    """Iteratively drop observations that are alone in any FE group."""
    n = len(fe_codes[0]) if fe_codes else 0
    keep = np.ones(n, dtype=bool)
    changed = True
    while changed and fe_codes:
        changed = False
        for c in fe_codes:
            cnt = np.bincount(c[keep])
            bad = keep & (cnt[np.minimum(c, len(cnt) - 1)] <= 1)
            if bad.any():
                keep &= ~bad
                changed = True
    return keep


def _one_way(x: np.ndarray, g: np.ndarray, ng: int, w: np.ndarray) -> np.ndarray:
    """Exact one-way demeaning, vectorized across columns (sort once, then
    segment sums with reduceat)."""
    if len(g) == 0:
        return x
    order = np.argsort(g, kind="stable")
    gs = g[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ws = w[order]
    sw = np.add.reduceat(ws, starts)
    means = np.add.reduceat(x[order] * ws[:, None], starts, axis=0) / sw[:, None]
    full = np.zeros((ng, x.shape[1]))
    full[gs[starts]] = means
    return x - full[g]


def exact_two_way(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray, max_block: int = 20_000
) -> np.ndarray:
    """Exact residuals of x on two FEs (codes a, b; weights w), no iteration.

    Eliminates FE `a` (diagonal) and solves the Schur complement for FE `b`
    within each connected component of the a-b bipartite graph, pinning one
    b level per component (the FE normalization):
        M_c gamma_c = D_b' W (I - P_a) x,   M_c = D_b'W D_b - D_b'W D_a (D_a'W D_a)^-1 D_a'W D_b
        x~ = (I - P_a) x - (I - P_a) D_b gamma
    Each M_c is a small dense SPD matrix factored with LAPACK (Cholesky). The
    FE with fewer levels per component goes on the Schur side (chosen here).
    Raises ValueError if a component exceeds max_block levels on that side
    (the caller falls back to iteration).
    """
    import scipy.sparse as sp
    from scipy.linalg import cho_factor, cho_solve
    from scipy.sparse.csgraph import connected_components

    n, k = x.shape
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    nA, nB = int(a.max()) + 1, int(b.max()) + 1
    G = sp.coo_matrix((np.ones(n), (a, nA + b)), shape=(nA + nB, nA + nB)).tocsr()
    ncomp, lab = connected_components(G, directed=False)
    compA, compB = lab[:nA], lab[nA:]
    sizesA = np.bincount(compA, minlength=ncomp)
    sizesB = np.bincount(compB, minlength=ncomp)
    # the dense solves cost ~ sum of (levels per component)^3: put the FE with
    # fewer levels per component on the Schur side
    if (sizesA.astype(float) ** 3).sum() < (sizesB.astype(float) ** 3).sum():
        a, b, nA, nB = b, a, nB, nA
        compA, compB, sizesA, sizesB = compB, compA, sizesB, sizesA
    if sizesB.max() > max_block:
        raise ValueError(f"component with {int(sizesB.max())} levels exceeds max_block={max_block}")

    xA = _one_way(x, a, nA, w)
    rhs = sp.csr_matrix((w, (b, np.arange(n))), shape=(nB, n)) @ xA
    WA = np.bincount(a, weights=w, minlength=nA)
    WB = np.bincount(b, weights=w, minlength=nB)
    cell = sp.coo_matrix((w, (a, b)), shape=(nA, nB)).tocsr()
    cell.sum_duplicates()
    cellT = cell.T.tocsr()

    orderB = np.argsort(compB, kind="stable")
    bstart = np.searchsorted(compB[orderB], np.arange(ncomp + 1))
    orderA = np.argsort(compA, kind="stable")
    astart = np.searchsorted(compA[orderA], np.arange(ncomp + 1))
    gamma = np.zeros((nB, k))
    for c in np.flatnonzero(sizesB > 1):
        bl = orderB[bstart[c] : bstart[c + 1]]
        al = orderA[astart[c] : astart[c + 1]]
        S = cellT[bl][:, al].toarray().T  # (nA_c x nB_c) weight sums
        M = np.diag(WB[bl]) - S.T @ (S / WA[al, None])
        f = cho_factor(M[1:, 1:], lower=True, check_finite=False)
        gamma[bl[1:]] = cho_solve(f, rhs[bl[1:]], check_finite=False)
    return xA - _one_way(gamma[b], a, nA, w)


def _map_numpy(x, flist, w, tol, maxiter):
    """Plain alternating projections (fallback when pyfixest is unavailable)."""
    x = x.copy()
    ngs = [int(f.max()) + 1 for f in flist.T]
    sws = [np.bincount(f, weights=w, minlength=ng) for f, ng in zip(flist.T, ngs)]
    for _ in range(maxiter):
        prev = x.copy()
        for f, ng, sw in zip(flist.T, ngs, sws):
            for j in range(x.shape[1]):
                m = np.bincount(f, weights=w * x[:, j], minlength=ng) / sw
                x[:, j] -= m[f]
        if np.max(np.abs(x - prev)) < tol:
            return x, True
    return x, False


def demean(
    x: np.ndarray,
    fe_codes: list[np.ndarray],
    w: np.ndarray,
    tol: float = 1e-10,
    maxiter: int = 100_000,
    method: str = "auto",
) -> tuple[np.ndarray, bool]:
    """Demean the columns of x by the given FEs.

    One FE: exact one pass. Two FEs: exact Schur-complement solve (method
    "auto"/"exact"), falling back to alternating projections if a connected
    component is too large or numerically singular. Three or more, or
    method="iterate": alternating projections (pyfixest's compiled core).
    """
    if not fe_codes:
        return x, True
    dense = [densify(c) for c in fe_codes]
    if len(dense) == 1:
        g, ng = dense[0]
        return _one_way(x, g.astype(np.int64), ng, w), True
    if len(dense) == 2 and method in ("auto", "exact"):
        try:
            return exact_two_way(x, dense[0][0], dense[1][0], w), True
        except (ValueError, np.linalg.LinAlgError) as e:
            if method == "exact":
                raise
            import logging

            logging.getLogger("dfeols").debug("exact two-way demeaning fell back to iteration: %s", e)
    flist = np.column_stack([d[0] for d in dense]).astype(np.uint64)
    if _pf_demean is not None:
        res, ok = _pf_demean(
            np.ascontiguousarray(x, dtype=np.float64),
            flist,
            np.ascontiguousarray(w, dtype=np.float64),
            tol,
            maxiter,
        )
        return np.asarray(res), bool(ok)
    return _map_numpy(x, flist.astype(np.int64), w, tol, maxiter)
