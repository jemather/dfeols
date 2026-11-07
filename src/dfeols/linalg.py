# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


def collinear_keep(
    ZtZ: np.ndarray, tol: float = 1e-9, block: int = 256, ref_diag: np.ndarray | None = None
) -> np.ndarray:
    """Detect collinear columns from the Gram matrix with an incremental
    Cholesky factorization (columns are tested in order, like fixest).
    Returns a boolean mask of columns to keep.

    Column j is dropped when its squared Cholesky pivot, given the columns kept
    before it, is <= tol * ZtZ[j, j]. Right-looking and blocked: columns are
    tested one at a time within a panel of `block` columns, against the Schur
    complement S of the columns kept so far; then the rest of S is updated with
    one matrix product. One triangular solve per column against all kept
    columns took minutes with thousands of crossing-FE dummies.

    `ref_diag` replaces ZtZ's diagonal as the scale of each test. ZtZ is the Gram of columns
    net of FEs, so a column an FE absorbs has a diagonal of rounding noise (~1e-29) and, tested
    against itself, would look independent and be kept; the callers pass each column's
    centered sum of squares before the FEs (`reference_ss`), against which it is dropped."""
    k = ZtZ.shape[0]
    keep = np.zeros(k, dtype=bool)
    d = np.diag(ZtZ).copy() if ref_diag is None else np.asarray(ref_diag, np.float64).copy()
    S = np.array(ZtZ, dtype=np.float64)  # trailing part: Schur complement given the kept columns
    for p in range(0, k, block):
        q = min(p + block, k)
        P = S[p:q, p:q]
        L = np.zeros((q - p, q - p))  # factor of this panel's kept columns
        m = 0
        kept: list[int] = []
        for j in range(q - p):
            djj = d[p + j]
            if not djj > 0:
                continue
            if m:
                l = solve_triangular(L[:m, :m], P[kept, j], lower=True)
                r = P[j, j] - l @ l
            else:
                l, r = np.zeros(0), P[j, j]
            if r <= tol * djj:
                continue
            L[m, :m] = l
            L[m, m] = np.sqrt(r)
            m += 1
            kept.append(j)
            keep[p + j] = True
        if m and q < k:
            W = solve_triangular(L[:m, :m], S[p + np.array(kept), q:], lower=True)
            S[q:, q:] -= W.T @ W
    return keep


def reference_ss(s1: np.ndarray, s2: np.ndarray, sumw: float) -> np.ndarray:
    """Each column's weighted sum of squares about its weighted mean, from its weighted sum
    s1 and sum of squares s2 before any FE: the scale against which collinear_keep tests a
    column net of the FEs. Every FE absorbs the constant, so centering is the least any FE
    removes; a constant column gets 0 and is dropped."""
    s1, s2 = np.asarray(s1, np.float64), np.asarray(s2, np.float64)
    if not sumw > 0:
        return np.zeros_like(s2)
    return np.maximum(s2 - s1 * s1 / sumw, 0.0)


def absorbed(net_ss: np.ndarray, ref: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Columns whose sum of squares net of the FEs (net_ss) is at most tol of their centered
    sum of squares before them (ref): absorbed by the FEs, as collinear_keep tests them."""
    return ~(np.asarray(net_ss) > tol * np.asarray(ref))


def solve_spd(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (A^-1 B, A^-1) for symmetric positive definite A."""
    c = cho_factor(A, lower=True)
    inv = cho_solve(c, np.eye(A.shape[0]))
    return cho_solve(c, B), (inv + inv.T) / 2
