# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Fitting from cells (planner.md, "Fitting from cells"; decided 2026-10-30): `fit(cells=True)`.

A cell is a distinct combination of every FE key and the cluster, so each FE and the cluster are
constant within it. The OLS fit with CRV1 (and iid) needs only, per cell, the rows n, the weight W,
the weighted means m of the outcomes and regressors and their cross-products about those means c
(worker.shard_cells, one pass over the shards). With z~ the cell means net of all FEs (a weighted
FE regression on the cells) and C the summed within-cell cross-products,

    X^'W X^ = C_xx + sum_c W_c x~_c x~_c',   X^'W y^ = C_xy + sum_c W_c x~_c y~_c,

the residuals are e_i = (y_i - ybar_c) - (x_i - xbar_c)'b + e~_c, so the RSS is
C_yy - 2 b'C_xy + b'C_xx b + sum_c W_c e~_c^2, and a cell's contribution to its cluster's score
is c_xy - c_xx b + W_c x~_c e~_c. Exact, not an approximation.

On the cells the FE with the most levels is absorbed exactly (group means) and the others are
solved as crossing FEs by conjugate gradient (dfeols.crossing): in level space when A fits the
budget, else with a product over the cells each step. The cells give round 1's Gram matrix of
every column net of the FEs exactly (`absorb`), so OLS and IV (2SLS, LIML, Fuller) are solved by
the row path's own solvers, and round 2's RSS and CRV1 meats (one- or two-way, IV first stages
too) come from the solve's linear maps applied to each cell's sums (`meats`). vcov CRV1 and iid;
HC1 needs the rows' own residuals.
"""

from __future__ import annotations

import logging
import time
import warnings

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import crossing as cx
from .linalg import absorbed

log = logging.getLogger("dfeols")

VCOV = ("CRV1", "iid")


def _upper(p: int) -> dict:
    """(i, j) -> column of the upper-triangle cross-products, either order."""
    out = {}
    pairs = [(i, j) for i in range(p) for j in range(i, p)]
    for col, (a, b) in enumerate(pairs):
        out[(a, b)] = out[(b, a)] = col
    return out


def combine(
    parts: list[dict], n_local: int, cluster_local: bool, cluster2_local: bool = False
) -> pl.DataFrame:
    """The shards' cells as one frame with global keys. Local FE codes and a local cluster are
    dense within a shard, so the shard id joins their key. Without a local FE a cell can span
    shards; those are merged (weighted means, cross-products by the parallel-variance formula)."""
    frames = []
    for r in parts:
        if not len(r["n"]):
            continue
        p = r["m"].shape[1]
        cols = {k: r["keys"][:, i] for i, k in enumerate(r["key_names"])}
        cols["shard"] = np.full(len(r["n"]), r["shard"], np.int64)
        cols.update(n=r["n"], W=r["W"])
        cols.update({f"m{j}": r["m"][:, j] for j in range(p)})
        cols.update({f"c{q}": r["c"][:, q] for q in range(r["c"].shape[1])})
        frames.append(pl.DataFrame(cols))
    df = pl.concat(frames)
    keys = [c for c in df.columns if c[0] in "lg" or c in ("cl", "cl2")]
    if n_local or cluster_local or cluster2_local:
        return df
    # merge cells that span shards
    ms = [c for c in df.columns if c.startswith("m")]
    cs = [c for c in df.columns if c.startswith("c") and c not in ("cl", "cl2")]
    p = len(ms)
    up = _upper(p)
    tot = df.group_by(keys).agg(
        pl.col("n").sum(),
        pl.col("W").sum().alias("Wt"),
        *[((pl.col(m) * pl.col("W")).sum() / pl.col("W").sum()).alias("M" + m[1:]) for m in ms],
    )
    j = df.join(tot.select(keys + ["M" + m[1:] for m in ms]), on=keys)
    pairs = [(i, k) for i in range(p) for k in range(i, p)]
    j = j.with_columns(
        [
            (
                pl.col(f"c{up[(i, k)]}")
                + pl.col("W") * (pl.col(f"m{i}") - pl.col(f"M{i}")) * (pl.col(f"m{k}") - pl.col(f"M{k}"))
            ).alias(f"c{up[(i, k)]}")
            for i, k in pairs
        ]
    )
    csum = j.group_by(keys).agg([pl.col(c).sum() for c in cs])
    out = tot.join(csum, on=keys).rename({"Wt": "W", **{"M" + m[1:]: m for m in ms}})
    return out.with_columns(shard=pl.lit(0, pl.Int64))


def _dense(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    if len(cols) == 1:
        return df.select(pl.col(cols[0]).rank("dense").cast(pl.Int64) - 1).to_series().to_numpy()
    return df.select(pl.struct(cols).rank("dense").cast(pl.Int64) - 1).to_series().to_numpy()


def _build_A(W, g, Ng, fs, offs, max_nnz: int, chunks: int = 40):
    """A = D'W D - S' diag(1/N_g) S over the crossing levels, S = G'W D, built in row chunks and
    abandoned (None) once its nonzeros pass max_nnz."""
    P, K = int(offs[-1]), len(fs)
    cols = np.concatenate([f + o for f, o in zip(fs, offs[:-1])])
    S = sp.csr_matrix((np.tile(W, K), (np.tile(g, K), cols)), shape=(len(Ng), P))
    ST = S.T.tocsr()
    Sd = (sp.diags(1.0 / Ng) @ S).tocsr()
    parts, nnz = [], 0
    step = max(-(-P // chunks), 1)
    for lo in range(0, P, step):
        blk = (ST[lo : lo + step] @ Sd).tocsr()
        nnz += blk.nnz
        if nnz > max_nnz:
            return None, nnz
        parts.append(blk)
    A2 = sp.vstack(parts).tocsr()
    rows = np.concatenate([fs[j] + offs[j] for j in range(K) for _ in range(K)])
    cl = np.concatenate([fs[k] + offs[k] for _ in range(K) for k in range(K)])
    A1 = sp.csr_matrix((np.tile(W, K * K), (rows, cl)), shape=(P, P))
    A = ((A1 - A2) + (A1 - A2).T) * 0.5
    A = A.tocsr()
    A.eliminate_zeros()
    return A, A.nnz


def _block(W, g, Ng, f, P):
    """A_kk = D_k'W D_k - S_k' diag(1/N_g) S_k."""
    Sk = sp.csr_matrix((W, (g, f)), shape=(len(Ng), P))
    B = sp.diags(np.bincount(f, weights=W, minlength=P)) - Sk.T @ sp.diags(1.0 / Ng) @ Sk
    B = ((B + B.T) * 0.5).tocsr()
    B.eliminate_zeros()
    return B


def absorb(
    df: pl.DataFrame,
    *,
    fe_names: list[str],
    fe_cols: list[list[str]],
    p: int,
    drop_singletons: bool,
    collin_tol: float,
    tol: float,
    maxiter: int,
    max_nnz: int,
    precond: str,
    label: str,
) -> dict:
    """Every FE out of the combined cells: singletons dropped, the FE with the most levels
    absorbed exactly and the others by conjugate gradient. Returns the cells' facts for the
    solve and the meats: the cells kept (`df`), N, the cell weights W, the cell means net of
    every FE `Zh` (C x p), the within-cell cross-products (`cmat`, `up`, summed in `Ctot`),
    the Gram matrix of all p columns net of every FE `ZZ` (round 1's Z'WZ, exact), each
    column's centered sum of squares before the FEs `ref` (the collinearity scale), the FE
    codes and levels, and the solve's `info`."""
    t0 = time.time()
    up = _upper(p)
    n = df["n"].to_numpy().astype(np.int64)
    codes = [_dense(df, c) for c in fe_cols]
    rows_in = int(n.sum())
    keep = np.ones(len(n), bool)
    passes = 0
    while drop_singletons:  # singletons over all FEs, iterated (a level with one row)
        passes += 1
        bad = np.zeros(len(n), bool)
        for f in codes:
            cnt = np.bincount(f[keep], weights=n[keep], minlength=int(f.max()) + 1 if len(f) else 0)
            bad |= cnt[f] == 1
        bad &= keep
        if not bad.any():
            break
        keep &= ~bad
    if not keep.all():
        df = df.filter(pl.Series(keep))
        n = n[keep]
        codes = [np.unique(f[keep], return_inverse=True)[1] for f in codes]
    N = int(n.sum())
    C = len(n)
    W = df["W"].to_numpy()
    Mz = df.select([f"m{j}" for j in range(p)]).to_numpy()
    cmat = df.select([f"c{q}" for q in range(p * (p + 1) // 2)]).to_numpy()
    Ctot = np.empty((p, p))
    for i in range(p):
        for j in range(p):
            Ctot[i, j] = cmat[:, up[(i, j)]].sum()
    levels = [int(f.max()) + 1 if C else 0 for f in codes]

    # absorb the FE with most levels exactly; the others cross
    a = int(np.argmax(levels))
    g = codes[a]
    Ng = np.bincount(g, weights=W)
    gm = np.column_stack([np.bincount(g, weights=W * Mz[:, j]) for j in range(p)]) / Ng[:, None]
    Zq = Mz - gm[g]
    # the collinearity scale: each column's centered sum of squares before the FEs (within
    # cells plus between them about the weighted mean), and net of the absorbed FE
    zbar = (W @ Mz) / W.sum() if C else np.zeros(p)
    ref = np.diag(Ctot) + (W[:, None] * (Mz - zbar) ** 2).sum(0)
    net0 = np.diag(Ctot) + (W[:, None] * Zq**2).sum(0)
    gone = absorbed(net0, ref, collin_tol)  # absorbed by the absorbed FE: rounding noise
    cross = [i for i in range(len(codes)) if i != a]
    fs = [codes[i] for i in cross]
    Ps = [levels[i] for i in cross]
    offs = np.r_[0, np.cumsum(Ps)].astype(np.int64)
    P = int(offs[-1])
    info: dict = {"cells": C, "absorbed": fe_names[a], "crossing": [fe_names[i] for i in cross]}
    if P:
        R = np.vstack(
            [
                np.column_stack([np.bincount(f, weights=W * Zq[:, j], minlength=L) for j in range(p)])
                for f, L in zip(fs, Ps)
            ]
        )
        A, nnz = _build_A(W, g, Ng, fs, offs, max_nnz)
        level = A is not None
        blocks = None
        if level:
            blocks = [A[o0:o1, o0:o1].tocsr() for o0, o1 in zip(offs[:-1], offs[1:])]
            diag = A.diagonal()
        else:
            # each block's size bound: sum over absorbed groups of (levels it touches)^2
            bound = 0
            for f in fs:
                lg = np.bincount(np.unique(np.c_[g, f], axis=0)[:, 0], minlength=len(Ng))
                bound += float((lg.astype(np.float64) ** 2).sum())
            if bound <= max_nnz:
                blocks = [_block(W, g, Ng, f, L) for f, L in zip(fs, Ps)]
                diag = np.concatenate([b.diagonal() for b in blocks])
            else:
                diag = np.concatenate(
                    [
                        np.bincount(f, weights=W, minlength=L)
                        - np.asarray(
                            sp.csr_matrix((W, (g, f)), shape=(len(Ng), L)).power(2).T @ (1.0 / Ng)
                        ).ravel()
                        for f, L in zip(fs, Ps)
                    ]
                )
            msg = (
                f"fit {label}: the crossing-FE system on the cells has more than {max_nnz:,} nonzeros "
                f"({P:,} levels: " + ", ".join(f"{fe_names[i]} {levels[i]:,}" for i in cross) + "), so "
                "each conjugate-gradient step is a product over the cells and convergence may be slow; "
                "level_space_gb raises the budget"
            )
            log.warning(msg)
            warnings.warn(msg, cx.SlowConvergenceWarning, stacklevel=4)
        solvers = None
        if blocks is not None and precond != "jacobi":
            solvers = [cx.BlockSolve(b, True, fe_names[i]) for b, i in zip(blocks, cross)]
        method = cx.choose_method(solvers, level, precond)
        Mp = cx.make_precond(method, diag, offs, solvers, A)
        act = diag > 1e-12 * max(float(diag.max(initial=0.0)), 1.0)
        D = sp.csr_matrix(
            (
                np.ones(C * len(fs)),
                (np.tile(np.arange(C), len(fs)), np.concatenate([f + o for f, o in zip(fs, offs[:-1])])),
            ),
            shape=(C, P),
        )
        if level:

            def matvec(v):
                return A @ v
        else:
            Gt = sp.csr_matrix((W, (g, np.arange(C))), shape=(len(Ng), C))

            def matvec(v):
                U = D @ v
                U -= ((Gt @ U) / Ng[:, None])[g]
                return D.T @ (U * W[:, None])

        R[:, gone] = 0.0  # its CG column could not reach a relative tolerance; Gamma 0
        Gam, steps, rel = cx.pcg(matvec, R, Mp, act, tol, maxiter)
        if rel > tol:
            raise RuntimeError(
                f"fit {label}: crossing-FE CG on the cells did not converge in {maxiter} steps "
                f"(largest relative residual {rel:.1e} > {tol:g}); raise iter_maxiter"
            )
        U = D @ Gam  # the crossing FEs' part, demeaned by the absorbed FE as Zq is
        U -= (np.column_stack([np.bincount(g, weights=W * U[:, j]) for j in range(p)]) / Ng[:, None])[g]
        Zh = Zq - U  # cell means net of every FE
        RG = R.T @ Gam
        info.update(
            mode="level" if level else "cells",
            preconditioner=method,
            steps=steps,
            max_relative_residual=rel,
            A_nnz=int(nnz) if level else None,
        )
    else:
        Zh = Zq
        RG = np.zeros((p, p))
        info.update(mode="none", steps=0)
    Z0 = Ctot + Zq.T @ (W[:, None] * Zq)  # net of the absorbed FE only
    ZZ = Z0 - RG
    ZZ = (ZZ + ZZ.T) / 2
    info.update(seconds=time.time() - t0, singleton_passes=passes)
    return {
        "df": df,
        "N": N,
        "rows": rows_in,
        "W": W,
        "Zh": Zh,
        "cmat": cmat,
        "up": up,
        "Ctot": Ctot,
        "ZZ": ZZ,
        "ref": ref,
        "codes": codes,
        "levels": levels,
        "info": info,
    }


def cluster_codes(df: pl.DataFrame, cols: list[str], codes: list[np.ndarray], levels: list[int]):
    """A cluster's dense codes over the cells, its count G, and whether each FE is nested in it
    (one cluster per FE level)."""
    cl = _dense(df, cols)
    G = int(cl.max()) + 1 if len(cl) else 0
    nested = []
    for f, L in zip(codes, levels):
        lo = np.full(L, np.iinfo(np.int64).max)
        hi = np.full(L, np.iinfo(np.int64).min)
        np.minimum.at(lo, f, cl)
        np.maximum.at(hi, f, cl)
        nested.append(bool((lo == hi).all()))
    return cl, G, nested


def meats(group: dict, cols, m: int, a: dict, clusters: list[np.ndarray]) -> dict:
    """Round 2 on the cells (worker.shard_round2 on rows), exact: for the linear maps of T =
    [Y (m columns), Z[:, cols]] in `group` (residuals E = T A, score regressors X = T H, from
    the solve), each equation's RSS and, per cluster code array in `clusters`, its meat.

    Within a cell the FEs are constant, so row i's T_i is the cell's within deviation plus the
    cell mean net of every FE, t~_c. Then sum_i w_i T_i T_i' over the cell is the within
    cross-products c_c plus W_c t~_c t~_c', and a cell's score sum_i w_i X_i E_ij is
    H'(c_c a_j) + W_c (H't~_c)(t~_c'a_j); the RSS is a_j'(C + sum_c W_c t~_c t~_c')a_j."""
    idx = np.r_[np.arange(m), m + np.asarray(cols, np.int64)]
    nT = len(idx)
    A = np.asarray(group["A"], np.float64)
    H = group["H"]
    if np.asarray(H).ndim == 1:
        Hm = np.zeros((nT, len(H)))
        Hm[np.asarray(H, np.int64), np.arange(len(H))] = 1.0
    else:
        Hm = np.asarray(H, np.float64)
    W, Zt, cmat, up = a["W"], a["Zh"][:, idx], a["cmat"], a["up"]
    CT = a["Ctot"][np.ix_(idx, idx)] + Zt.T @ (W[:, None] * Zt)
    ne, k = A.shape[1], Hm.shape[1]
    rss = np.einsum("ij,ik,kj->j", A, CT, A)
    crv = [np.zeros((ne, k, k)) for _ in clusters]
    if clusters:
        ZH = Zt @ Hm
        # each cell's within cross-products of T, row r: columns cmat[:, ucol[r]]
        ucol = [np.array([up[(idx[r], idx[c])] for c in range(nT)]) for r in range(nT)]
        for j in range(ne):
            aj = A[:, j]
            nz = np.flatnonzero(aj)
            v = np.column_stack([cmat[:, ucol[r][nz]] @ aj[nz] for r in range(nT)])  # c_c a_j
            sc = v @ Hm + (W * (Zt @ aj))[:, None] * ZH
            for q, cl in enumerate(clusters):
                G = int(cl.max()) + 1 if len(cl) else 0
                Sg = np.column_stack([np.bincount(cl, weights=sc[:, t], minlength=G) for t in range(k)])
                crv[q][j] = Sg.T @ Sg
    return {"rss": rss, "crv": crv}
