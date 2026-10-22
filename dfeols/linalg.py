from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


def collinear_keep(ZtZ: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Detect collinear columns from the Gram matrix with an incremental
    Cholesky factorization (columns are tested in order, like fixest).
    Returns a boolean mask of columns to keep."""
    k = ZtZ.shape[0]
    keep = np.zeros(k, dtype=bool)
    L = np.zeros((k, k))  # lower-triangular factor of kept columns
    m = 0
    kept: list[int] = []
    for j in range(k):
        djj = ZtZ[j, j]
        if not djj > 0:
            continue
        if m:
            l = solve_triangular(L[:m, :m], ZtZ[kept, j], lower=True)
            r = djj - l @ l
        else:
            l, r = np.zeros(0), djj
        if r <= tol * djj:
            continue
        L[m, :m] = l
        L[m, m] = np.sqrt(r)
        m += 1
        kept.append(j)
        keep[j] = True
    return keep


def solve_spd(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (A^-1 B, A^-1) for symmetric positive definite A."""
    c = cho_factor(A, lower=True)
    inv = cho_solve(c, np.eye(A.shape[0]))
    return cho_solve(c, B), (inv + inv.T) / 2
