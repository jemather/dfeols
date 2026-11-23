# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Weighted demeaning by several FEs in coefficient space: a port of fixest's algorithm.

Ported from fixest 0.14.2 (Laurent Bergé; GPL-3), src/demeaning.cpp: `demean_acc_gnl`,
`compute_fe_gnl`, `dm_update_X_IronsTuck`, `demean_single_gnl`, and the stopping rules in
src/fixest_main.h (`continue_crit`, `stopping_crit`); defaults from R/miscfuns.R
`demeaning_algo` (iter_warmup 15, iter_projAfterAcc 40, iter_grandAcc 4, extraProj 0). See
Bergé (2018), "Efficient estimation of maximum likelihood models with multiple fixed-effects:
the R package FENmlm".

The algorithm iterates on the FE coefficients, not on the N-row data:

* G (compute_fe_gnl): update each FE's coefficients in turn from the last (fewest levels) to
  the first, each from the weighted sums of (input - output - the other FEs) in its groups;
* Irons-Tuck acceleration of X <- G(X) on the coefficients of all FEs but the last (G does not
  depend on the last FE's input values), a "grand" acceleration every 4 iterations over
  every-4th iterates, and an extra projection after each acceleration from iteration 40;
* stopping when no coefficient moves by more than tol both absolutely and relative to
  0.1 + its size, or when the sum of squared residuals (checked every 40 iterations) stops
  changing by tol;
* with three or more FEs: 15 accelerated iterations; if not converged, the two largest FEs to
  convergence on what is left (the others held), then all FEs again on what is left.

FEs are sorted by number of levels, most first, as fixest does. Not ported: varying slopes.

Two FEs (and the two-FE phase above) use fixest's two-FE map (compute_fe_coef_2): the state
is the first FE's coefficients only, and G sets the second FE from the first, then the first
from the second, without an N-length vector of the other FEs. fixest still loops over rows
in that map; here it loops over the distinct (FE 1, FE 2) cells with their summed weights,
which is the same arithmetic (sum over a cell of w * a[g_a] = W_cell * a), so a panel with
many rows per level pair costs per iteration what its cells do, not its rows.

On Poisson working weights spread over 14 orders of magnitude (D13 BYZ R0 shard, 2026-11-23)
pyfixest's row-space alternating projections hit 100,000 sweeps without converging, their
output off by up to 0.8.

    xt, ok = demean_fx(x, [codes_1, ..., codes_Q], w, tol=1e-6, maxiter=10_000)
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Kernels release the GIL (nogil): a long demean must not starve the Dask worker's
# heartbeat thread, or the scheduler kills the worker (seen on BACI R0, 2026-11-23).
ITER_WARMUP, ITER_PROJ_AFTER_ACC, ITER_GRAND_ACC = 15, 40, 4


@njit(cache=True, nogil=True)
def _continue(a, b, tol):
    d = abs(a - b)
    return d > tol and d / (0.1 + abs(a)) > tol


@njit(cache=True, nogil=True)
def _stop(a, b, tol):
    d = abs(a - b)
    return d < tol or d / (0.1 + abs(a)) < tol


@njit(cache=True, nogil=True)
def _compute_fe(origin, dest, G, w, offs, isw, sum_in_out, Qa, other):
    """compute_fe_gnl: dest <- G(origin) over FEs 0..Qa-1, last first."""
    n = G.shape[1]
    for q in range(Qa - 1, -1, -1):
        for i in range(n):
            other[i] = 0.0
        for h in range(Qa):
            if h == q:
                continue
            src = origin if h < q else dest
            o = offs[h]
            g = G[h]
            for i in range(n):
                other[i] += src[o + g[i]] * w[i]
        lo, hi = offs[q], offs[q + 1]
        for m in range(lo, hi):
            dest[m] = sum_in_out[m]
        g = G[q]
        for i in range(n):
            dest[lo + g[i]] -= other[i]
        for m in range(lo, hi):
            dest[m] *= isw[m]


@njit(cache=True, nogil=True)
def _irons_tuck(n_no_q, X, GX, GGX):
    """dm_update_X_IronsTuck: X <- GGX - (dGX . d2X)/(d2X . d2X) dGX on the first n_no_q
    coefficients. True when d2X is zero (numerically converged)."""
    vprod, ssq = 0.0, 0.0
    for i in range(n_no_q):
        dg = GGX[i] - GX[i]
        d2 = dg - GX[i] + X[i]
        vprod += dg * d2
        ssq += d2 * d2
    if ssq == 0.0:
        return True
    c = vprod / ssq
    for i in range(n_no_q):
        X[i] = GGX[i] - c * (GGX[i] - GX[i])
    return False


@njit(cache=True, nogil=True)
def _acc(x, out, G, w, offs, isw, Qa, tol, iter_max):
    """demean_acc_gnl on FEs 0..Qa-1 from zero coefficients, on what is left (x - out).
    Adds the fitted FEs to `out`. Returns (iterations, converged)."""
    n = len(x)
    nT, n_no_q = offs[Qa], offs[Qa - 1]
    sum_in_out = np.zeros(nT)
    for q in range(Qa):
        o = offs[q]
        g = G[q]
        for i in range(n):
            sum_in_out[o + g[i]] += (x[i] - out[i]) * w[i]
    other = np.empty(n)
    X = np.zeros(nT)
    GX = np.empty(nT)
    GGX = np.empty(nT)
    Y = np.zeros(nT)
    GY = np.zeros(nT)
    GGY = np.zeros(nT)
    _compute_fe(X, GX, G, w, offs, isw, sum_in_out, Qa, other)
    keep = False
    for i in range(nT):
        if _continue(X[i], GX[i], tol):
            keep = True
            break
    ssr, it, grand = 0.0, 0, 0
    while keep and it < iter_max:
        it += 1
        _compute_fe(GX, GGX, G, w, offs, isw, sum_in_out, Qa, other)
        if _irons_tuck(n_no_q, X, GX, GGX):
            break
        if it >= ITER_PROJ_AFTER_ACC:
            Y[:] = X
            _compute_fe(Y, X, G, w, offs, isw, sum_in_out, Qa, other)
        _compute_fe(X, GX, G, w, offs, isw, sum_in_out, Qa, other)
        keep = False
        for i in range(n_no_q):
            if _continue(X[i], GX[i], tol):
                keep = True
                break
        if it % ITER_GRAND_ACC == 0:
            grand += 1
            if grand == 1:
                Y[:] = GX
            elif grand == 2:
                GY[:] = GX
            else:
                GGY[:] = GX
                if _irons_tuck(n_no_q, Y, GY, GGY):
                    break
                _compute_fe(Y, GX, G, w, offs, isw, sum_in_out, Qa, other)
                grand = 0
        if it % 40 == 0:  # the SSR of the input less the current FEs (fixest's second rule)
            old = ssr
            ssr = 0.0
            for i in range(n):
                mu = 0.0
                for q in range(Qa):
                    mu += GX[offs[q] + G[q, i]]
                r = x[i] - mu
                ssr += r * r
            if _stop(old, ssr, tol):
                break
    for q in range(Qa):
        o = offs[q]
        g = G[q]
        for i in range(n):
            out[i] += GX[o + g[i]]
    return it, it != iter_max


@njit(cache=True, nogil=True)
def _map2(a, b, ca, cb, cw, offs, isw, sum_in_out, a_out):
    """compute_fe_coef_2: b <- (in_out_b - sum W a) / SW_b, then a_out <- (in_out_a - sum W b) / SW_a,
    over the cells (ca, cb, cw): FE 0's and FE 1's codes of each distinct pair, its weight."""
    na = offs[1]
    nb = offs[2] - na
    for m in range(nb):
        b[m] = sum_in_out[na + m]
    for c in range(len(cw)):
        b[cb[c]] -= cw[c] * a[ca[c]]
    for m in range(nb):
        b[m] *= isw[na + m]
    for m in range(na):
        a_out[m] = sum_in_out[m]
    for c in range(len(cw)):
        a_out[ca[c]] -= cw[c] * b[cb[c]]
    for m in range(na):
        a_out[m] *= isw[m]


@njit(cache=True, nogil=True)
def _acc2(x, out, G, w, offs, isw, ca, cb, cw, tol, iter_max):
    """demean_acc_gnl with two_fe (Q == 2, or the first two FEs of Q >= 3, the others held):
    the state is FE 0's coefficients. Adds the fitted FEs to `out`; returns (iterations,
    converged)."""
    n = len(x)
    na = offs[1]
    nb = offs[2] - na
    sum_in_out = np.zeros(offs[2])
    for q in range(2):
        o = offs[q]
        g = G[q]
        for i in range(n):
            sum_in_out[o + g[i]] += (x[i] - out[i]) * w[i]
    b = np.zeros(nb)
    X = np.zeros(na)
    GX = np.empty(na)
    GGX = np.empty(na)
    Y = np.zeros(na)
    GY = np.zeros(na)
    GGY = np.zeros(na)
    _map2(X, b, ca, cb, cw, offs, isw, sum_in_out, GX)
    keep = False
    for i in range(na):
        if _continue(X[i], GX[i], tol):
            keep = True
            break
    ssr, it, grand = 0.0, 0, 0
    while keep and it < iter_max:
        it += 1
        _map2(GX, b, ca, cb, cw, offs, isw, sum_in_out, GGX)
        if _irons_tuck(na, X, GX, GGX):
            break
        if it >= ITER_PROJ_AFTER_ACC:
            Y[:] = X
            _map2(Y, b, ca, cb, cw, offs, isw, sum_in_out, X)
        _map2(X, b, ca, cb, cw, offs, isw, sum_in_out, GX)
        keep = False
        for i in range(na):
            if _continue(X[i], GX[i], tol):
                keep = True
                break
        if it % ITER_GRAND_ACC == 0:
            grand += 1
            if grand == 1:
                Y[:] = GX
            elif grand == 2:
                GY[:] = GX
            else:
                GGY[:] = GX
                if _irons_tuck(na, Y, GY, GGY):
                    break
                _map2(Y, b, ca, cb, cw, offs, isw, sum_in_out, GX)
                grand = 0
        if it % 40 == 0:  # SSR of the input less the current two FEs (b from GX)
            _map2(GX, b, ca, cb, cw, offs, isw, sum_in_out, GGX)  # b <- b(GX); GGX scratch
            old = ssr
            ssr = 0.0
            for i in range(n):
                r = x[i] - GX[G[0, i]] - b[G[1, i]]
                ssr += r * r
            if _stop(old, ssr, tol):
                break
    # a last pass (fixest: compute_fe_coef_2(alpha, alpha, beta)): b from GX, then a from b
    _map2(GX, b, ca, cb, cw, offs, isw, sum_in_out, X)
    for i in range(n):
        out[i] += X[G[0, i]] + b[G[1, i]]
    return it, it != iter_max


@njit(cache=True, nogil=True)
def _column(x, G, w, offs, isw, Q, tol, iter_max, ca, cb, cw):
    """demean_single_gnl (and demean_single_1 for one FE)."""
    n = len(x)
    out = np.zeros(n)
    total, ok = 0, True
    if Q == 1:
        g = G[0]
        s = np.zeros(offs[1])
        for i in range(n):
            s[g[i]] += w[i] * x[i]
        for i in range(n):
            out[i] = s[g[i]] * isw[g[i]]
    elif Q == 2:
        total, ok = _acc2(x, out, G, w, offs, isw, ca, cb, cw, tol, iter_max)
    else:
        warm = ITER_WARMUP
        it, ok = _acc(x, out, G, w, offs, isw, Q, tol, warm)
        total += it
        if not ok and iter_max > warm:
            it2 = iter_max // 2 - warm
            if it2 > 0:
                it, _ = _acc2(x, out, G, w, offs, isw, ca, cb, cw, tol, it2)
                total += it
            it, ok = _acc(x, out, G, w, offs, isw, Q, tol, iter_max - total)
            total += it
    r = np.empty(n)
    for i in range(n):
        r[i] = x[i] - out[i]
    return r, total, ok


def demean_fx(x, fe_codes, w, tol: float = 1e-6, maxiter: int = 10_000, info: dict | None = None):
    """x (n,) or (n, k) net of the FEs (integer codes, one array per FE) with weights w, by
    fixest's algorithm. Returns (x~, converged). `info`, if given, receives the iterations
    per column."""
    x = np.asarray(x, dtype=np.float64)
    one = x.ndim == 1
    X = x[:, None] if one else x
    n = X.shape[0]
    w = np.ascontiguousarray(w, dtype=np.float64)
    codes = []
    for c in fe_codes:  # dense 0..L-1 codes
        c = np.asarray(c)
        u, inv = np.unique(c, return_inverse=True)
        codes.append(inv.astype(np.int64) if len(u) != (int(c.max()) + 1 if n else 0) else c.astype(np.int64))
    sizes = [int(c.max()) + 1 if n else 0 for c in codes]
    order = sorted(range(len(codes)), key=lambda q: -sizes[q])  # most levels first (fixest)
    Q = len(codes)
    if Q == 0 or n == 0:
        return x.copy(), True
    G = np.ascontiguousarray(np.vstack([codes[q] for q in order]))
    offs = np.zeros(Q + 1, dtype=np.int64)
    offs[1:] = np.cumsum([sizes[q] for q in order])
    isw = np.zeros(offs[-1])
    for j, q in enumerate(order):
        sw = np.bincount(codes[q], weights=w, minlength=sizes[q])
        isw[offs[j] : offs[j + 1]] = np.divide(1.0, sw, out=np.zeros_like(sw), where=sw > 0)
    # the distinct (FE 0, FE 1) cells and their weights, for the two-FE map
    if Q >= 2:
        key = G[0] * np.int64(sizes[order[1]]) + G[1]
        uk, inv = np.unique(key, return_inverse=True)
        ca = (uk // sizes[order[1]]).astype(np.int64)
        cb = (uk % sizes[order[1]]).astype(np.int64)
        cw = np.bincount(inv.ravel(), weights=w, minlength=len(uk))
    else:
        ca = cb = np.zeros(0, np.int64)
        cw = np.zeros(0)
    out = np.empty_like(X)
    ok_all, its = True, []
    for j in range(X.shape[1]):
        r, it, ok = _column(np.ascontiguousarray(X[:, j]), G, w, offs, isw, Q, tol, maxiter, ca, cb, cw)
        out[:, j] = r
        its.append(int(it))
        ok_all &= bool(ok)
    if info is not None:
        info["iterations"] = its
    return (out[:, 0] if one else out), ok_all
