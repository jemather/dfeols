# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Shard-level work. These functions run on workers and must stay picklable.

Singletons (`shard_singletons`): with FEs that cross the partition, singletons
are found over all FEs at once in passes over the shards (see
estimate._drop_singletons); each shard's keep mask goes to plan.keep_dir.

Round 1 (`shard_round1`): build the design for one shard, drop singletons,
demean by FEs nested in the partition key, and return cross-products.
The demeaned arrays are cached to disk for round 2.

Round 2 (`shard_round2`): residuals, RSS and the meat of the sandwich, from linear
maps of the round-1 columns that the coordinator passes (OLS and IV alike).

Plan C (`shard_matvec`, crossing_fe="iterate"): one conjugate-gradient step for the
crossing-FE coefficients, M p = D'W Q D p with Q the within-local-FEs residual maker,
for a block p the coordinator broadcasts. A shard keeps its codes and weights in
process memory between steps (_ITER_CACHE), so a step reads nothing from disk.

IV (docs/design/iv.md): the endogenous regressors are built into the outcome block,
after the outcomes, so they are demeaned (and partialled against crossing-FE dummies)
like an outcome; the excluded instruments go into the regressor block after the
reported regressors, before the dummies.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict

try:
    import resource
except ImportError:  # pragma: no cover (Windows)
    resource = None
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .demean import demean, densify, drop_singletons
from .formula import Formula, Term
from .rowhash import rowset

# Sparse path: above this many crossing-FE levels a shard returns its Gram matrix sparse
SPARSE_GRAM_LEVELS = 2000

# Plan C: shard state kept between conjugate-gradient steps, least recently used first out
_ITER_CACHE: OrderedDict = OrderedDict()
ITER_CACHE_GB = float(os.environ.get("DFEOLS_ITER_CACHE_GB", "8"))


@dataclass
class Plan:
    """Everything a worker needs, resolved once on the coordinator."""

    formula: Formula
    spec_id: str
    local_fes: list[tuple[str, str]]  # (name, column) nested in partition
    global_fes: list[tuple[str, str, int]]  # (name, column, n_levels) -> dummies
    cluster: tuple[str, str, int] | None  # (name, column, n_levels)
    cluster_local: bool  # cluster nested in partition key
    weights: str | None
    cat_levels: dict[str, list]  # categorical var -> global sorted levels
    colnames: list[str]  # reported regressors (incl. intercept)
    n_dummy: int  # global-FE dummy columns appended after
    vcov_types: tuple[str, ...]
    cache_dir: str | None
    drop_singletons: bool = True
    fe_tol: float = 1e-10
    fe_maxiter: int = 100_000
    fe_method: str = "auto"  # auto | exact | iterate
    sparse_global: bool = False  # crossing FEs via per-level sums, no dense dummies
    crossing_iterate: bool = False  # plan C: crossing-FE coefficients by distributed CG
    iter_token: str | None = None  # plan C: names this fit's shard state in _ITER_CACHE
    level_nnz: int = 0  # plan C: most nonzeros of A a shard returns (0: no level space)
    keep_dir: str | None = None  # per-shard singleton keep masks (when FEs cross the partition)
    row_key_cols: tuple[str, ...] = ()  # raw columns hashed into the record's row_key
    endog_names: list[str] = field(default_factory=list)  # IV: endogenous columns (outcome block)
    inst_names: list[str] = field(default_factory=list)  # IV: excluded instruments (regressor block)

    @property
    def k_x(self) -> int:
        """Regressor-block columns before the crossing-FE dummies."""
        return len(self.colnames) + len(self.inst_names)


def plan_columns(f: Formula, cat_levels: dict[str, list], has_fe: bool):
    """Reported regressors. With any FE the constant is absorbed (with only
    dummy FEs it is kept as a hidden column after the dummies, standing in
    for their dropped first level, and is not reported)."""
    return (["Intercept"] if f.intercept and not has_fe else []) + term_names(f.terms, cat_levels)


def term_names(terms, cat_levels: dict[str, list]) -> list[str]:
    """Column names of `terms`, one per column `_term_columns` builds."""
    names: list[str] = []
    for t in terms:
        if t.cat is None:
            names.append(":".join(t.conts))
            continue
        levels = list(cat_levels[t.cat])
        if t.ref is not None:
            levels = [l for l in levels if l != t.ref]
        elif t.drop_first:
            levels = levels[1:]
        for l in levels:
            lab = f"{t.cat}::{l}"
            names.append(lab + (":" + ":".join(t.conts) if t.conts else ""))
    return names


def _term_columns(t: Term, df: pl.DataFrame, cat_levels: dict[str, list]) -> list[np.ndarray]:
    base = None
    for v in t.conts:
        a = df[v].to_numpy().astype(np.float64)
        base = a if base is None else base * a
    if t.cat is None:
        return [base]
    levels = list(cat_levels[t.cat])
    if t.ref is not None:
        levels = [l for l in levels if l != t.ref]
    elif t.drop_first:
        levels = levels[1:]
    c = df[t.cat].to_numpy()
    out = []
    for l in levels:
        d = (c == l).astype(np.float64)
        out.append(d if base is None else d * base)
    return out


def _load(plan: Plan, files: list[str], dropped: dict | None = None):
    """The shard's rows for this spec. `dropped`, if given, gets the rows removed by
    reason (missing, non_finite, zero_weights), for the record's sample counts."""
    f = plan.formula
    cols = set(f.variables)
    cols.update(c for _, c in plan.local_fes)
    cols.update(c for _, c, _ in plan.global_fes)
    if plan.cluster:
        cols.add(plan.cluster[1])
    if plan.weights:
        cols.add(plan.weights)
    cols.update(plan.row_key_cols)
    df = pl.read_parquet(files, columns=sorted(cols))
    # listwise deletion on the variables of this spec
    num = [c for c in f.variables if df.schema[c].is_float()]
    n = [df.height]
    df = df.drop_nulls()
    n.append(df.height)
    if num:
        df = df.filter(pl.all_horizontal([pl.col(c).is_finite() for c in num]))
    n.append(df.height)
    if plan.weights:
        df = df.filter(pl.col(plan.weights) > 0)
    n.append(df.height)
    if dropped is not None:
        dropped.update(
            {"rows": n[0], "missing": n[0] - n[1], "non_finite": n[1] - n[2], "zero_weights": n[2] - n[3]}
        )
    return df


def _outcome_block(plan: Plan, df, n: int) -> np.ndarray:
    """Outcomes, then (IV) the endogenous regressors' columns."""
    f = plan.formula
    cols = [df[y].to_numpy().astype(np.float64) for y in f.depvars] if n else []
    for t in f.endog:
        cols.extend(_term_columns(t, df, plan.cat_levels) if n else [])
    width = len(f.depvars) + len(plan.endog_names)
    return np.column_stack(cols) if n else np.zeros((0, width))


def _maxrss_gb() -> float:
    """Peak resident memory of this worker process so far (Linux reports KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6 if resource else float("nan")


def _cache_path(plan: Plan, shard_id: int) -> Path:
    return Path(plan.cache_dir) / plan.spec_id / f"shard_{shard_id:06d}.npz"


def _keep_path(plan: Plan, shard_id: int) -> Path:
    return Path(plan.keep_dir) / f"shard_{shard_id:06d}.npy"


def shard_singletons(plan: Plan, shard_id: int, files: list[str], gcount, first: bool) -> dict:
    """One pass of singleton removal over every FE, as fixest and reghdfe do.

    gcount: per crossing FE, the count of kept rows in each level over all shards
    after the previous pass (None on the first pass, which drops local-FE
    singletons only). Other shards only drop rows, so gcount minus this shard's
    own rows plus its current rows is an upper bound on a level's true count: a
    row is dropped only if it is truly alone, and anything missed is caught on a
    later pass. Writes the keep mask and returns this shard's level counts."""
    t0 = time.perf_counter()
    df = _load(plan, files)  # the same rows round 1 sees (listwise deletion included)
    n0 = df.height
    keep = np.ones(n0, bool) if first else np.load(_keep_path(plan, shard_id))
    local = [densify(df[c].to_numpy())[0] for _, c in plan.local_fes]
    glob = [(df[c].to_numpy().astype(np.int64), L) for _, c, L in plan.global_fes]
    before = int(keep.sum())
    others = None
    if gcount is not None:
        others = [gc - np.bincount(g[keep], minlength=L) for gc, (g, L) in zip(gcount, glob)]
    changed = n0 > 0
    while changed:
        changed = False
        for c in local:
            cnt = np.bincount(c[keep], minlength=int(c.max()) + 1)
            bad = keep & (cnt[c] <= 1)
            if bad.any():
                keep &= ~bad
                changed = True
        for i, (g, L) in enumerate(glob if others is not None else ()):
            cnt = others[i] + np.bincount(g[keep], minlength=L)
            bad = keep & (cnt[g] <= 1)
            if bad.any():
                keep &= ~bad
                changed = True
    p = _keep_path(plan, shard_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npy")
    np.save(tmp, keep)
    os.replace(tmp, p)
    return {
        "shard": shard_id,
        "dropped": before - int(keep.sum()),
        "gcount": [np.bincount(g[keep], minlength=L) for g, L in glob],
        "timing": {"total": time.perf_counter() - t0, "maxrss_gb": _maxrss_gb()},
    }


def shard_round1(plan: Plan, shard_id: int, files: list[str]) -> dict:
    out, _ = _round1(plan, shard_id, files, write_cache=True)
    return out


def _global_layout(plan: Plan):
    """Combined level index over all global FEs, and each level's column in Z
    (-1 for the dropped first level of each FE)."""
    offs, zpos, o, zo = [], [], 0, plan.k_x
    for _, _, L in plan.global_fes:
        offs.append(o)
        zpos.append(np.r_[-1, zo + np.arange(L - 1)])
        o += L
        zo += L - 1
    return np.array(offs, np.int64), np.concatenate(zpos) if zpos else np.zeros(0, np.int64), o


def _global_matrix(df, plan: Plan, n: int):
    """Sparse n x P indicator matrix of all global FE levels."""
    import scipy.sparse as sp

    offs, _, P = _global_layout(plan)
    cols = (
        np.concatenate(
            [df[c].to_numpy().astype(np.int64) + off for (_, c, _), off in zip(plan.global_fes, offs)]
        )
        if n
        else np.zeros(0, np.int64)
    )
    rows = np.tile(np.arange(n), len(plan.global_fes))
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, P)), cols.reshape(
        len(plan.global_fes), n
    )


def _projected_dummy_gram(local_codes, w, Dm, max_block: int = 20_000):
    """D'W P_L D for the n x P dummy block D (sparse Dm) and P_L the W-orthogonal
    projection on the local FEs' dummies, as a sparse P x P matrix.

    One local FE A:  C_A' diag(1/W_A) C_A, with C_A = A'W D the weight in each
    (local group, level) cell. Several: A is the one with most levels and B the others'
    dummies stacked; P_L = P_A + projection on Q_A B, so add F' S^+ F with
        S = B'W B - E' diag(1/W_A) E,   F = B'W D - E' diag(1/W_A) C_A,   E = A'W B.
    S is block-diagonal over the connected components of the local FEs' graph and
    singular where the FEs overlap (a FE nested in another, one level per component),
    so each block is solved densely with a pseudo-inverse: F's range lies in S's.
    Raises ValueError if a component has more than max_block levels of B."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    n = len(w)
    codes = sorted((c.astype(np.int64) for c in local_codes), key=lambda c: -(int(c.max()) + 1))
    W = sp.diags(w)

    def ind(c):
        return sp.csr_matrix((np.ones(n), (np.arange(n), c)), shape=(n, int(c.max()) + 1))

    A = ind(codes[0])
    WA = np.bincount(codes[0], weights=w)
    CA = (A.T @ W @ Dm).tocsr()
    out = CA.T @ sp.diags(1.0 / WA) @ CA
    if len(codes) == 1:
        return out
    B = sp.hstack([ind(c) for c in codes[1:]], format="csr")
    E = (A.T @ W @ B).tocsr()
    EA = sp.diags(1.0 / WA) @ E
    S = (B.T @ W @ B - E.T @ EA).tocsr()
    F = (B.T @ W @ Dm - EA.T @ CA).tocsr()
    ncomp, lab = connected_components(S, directed=False)
    sizes = np.bincount(lab, minlength=ncomp)
    if sizes.max() > max_block:
        raise ValueError(
            f"a connected component of the local FEs has {int(sizes.max()):,} levels in its "
            f"Schur block (> {max_block:,}); partition by a finer key"
        )
    order = np.argsort(lab, kind="stable")
    starts = np.searchsorted(lab[order], np.arange(ncomp + 1))
    rows, cols, vals = [], [], []
    for c in range(ncomp):
        bl = order[starts[c] : starts[c + 1]]
        Fc = F[bl]
        lv = np.unique(Fc.indices)
        if not len(lv):
            continue
        Fd = Fc[:, lv].toarray()
        Sc = S[bl][:, bl].toarray()
        G = Fd.T @ np.linalg.pinv(Sc, rcond=1e-10, hermitian=True) @ Fd
        r, q = np.meshgrid(lv, lv, indexing="ij")
        rows.append(r.ravel())
        cols.append(q.ravel())
        vals.append(G.ravel())
    if rows:
        P = Dm.shape[1]
        out = out + sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(P, P)
        )
    return out


def _dummy_gram(plan: Plan, local_codes, w, Dm, gcodes, P: int, sparse_out: bool):
    """D~'W D~ = D'W D - D'W P_L D over all P crossing levels, sparse (csr) or dense; with no
    local FE, D'W D. Raises ValueError where _projected_dummy_gram does."""
    import scipy.sparse as sp

    n = len(w)
    DWD = Dm.T @ sp.diags(w) @ Dm
    if not sparse_out:
        DWD = DWD.toarray()
    if n and len(local_codes) > 1:
        PL = _projected_dummy_gram(local_codes, w, Dm)
        return (DWD - PL).tocsr() if sparse_out else DWD - PL.toarray()
    if not n or not local_codes:
        return DWD.tocsr() if sparse_out else DWD
    a = local_codes[0].astype(np.int64)
    nA = int(a.max()) + 1
    C = sp.csr_matrix(
        (
            np.tile(w, len(plan.global_fes)),
            (np.tile(a, len(plan.global_fes)), gcodes.ravel()),  # gcodes carry the offsets
        ),
        shape=(nA, P),
    )
    C.sum_duplicates()
    WA = np.bincount(a, weights=w, minlength=nA)
    if sparse_out:  # many levels: a group spans few of them, so C'C stays sparse
        return (DWD - C.T @ sp.diags(1.0 / WA) @ C).tocsr()
    if nA * P <= 50_000_000:
        # dense BLAS product: cost nA*P^2 but vectorised; the sparse product
        # costs sum over groups of (levels per group)^2 and is slow when
        # each group spans many levels (e.g. units observed in 1000 periods)
        Cd = C.toarray()
        return DWD - (Cd / WA[:, None]).T @ Cd
    return DWD - (C.T @ sp.diags(1.0 / WA) @ C).toarray()


def _round1_sparse(plan: Plan, shard_id: int, df, n0: int, local_codes, w, T, t0):
    """Local FEs plus crossing FEs, without dense dummies.

    With Q the within-local-FEs residual maker (W-orthogonal), for the dummy
    block D:  D~'W X~ = D'W X~ (per-level sums),  D~'W D~ = D'W D - D'W P_L D
    (_projected_dummy_gram; with one local FE, C'diag(1/W_a) C where C holds the
    weight in each (local group, level) cell).
    """
    import scipy.sparse as sp

    f = plan.formula
    t = time.perf_counter()
    n = df.height
    Y = _outcome_block(plan, df, n)
    cols = []
    for term in (*f.terms, *f.instruments):
        cols.extend(_term_columns(term, df, plan.cat_levels))
    X = np.column_stack(cols) if cols else np.zeros((n, 0))
    Dm, gcodes = _global_matrix(df, plan, n)
    T["build"] = time.perf_counter() - t

    t = time.perf_counter()
    m = Y.shape[1]
    converged = True
    if n:
        YX, converged = demean(
            np.hstack([Y, X]), local_codes, w, plan.fe_tol, plan.fe_maxiter, plan.fe_method
        )
    else:
        YX = np.zeros((0, m + X.shape[1]))
    Yt, Xt = YX[:, :m], YX[:, m:]
    T["demean"] = time.perf_counter() - t

    t = time.perf_counter()
    k = Xt.shape[1]
    offs, zpos, P = _global_layout(plan)
    k_tot = k + plan.n_dummy
    Xw, Yw = Xt * w[:, None], Yt * w[:, None]
    DX = Dm.T @ Xw  # P x k
    DY = Dm.T @ Yw  # P x m
    if plan.crossing_iterate:
        # no dummy block: the coordinator solves D~'W D~ G = D'W [X~ Y~] by CG. With
        # plan.level_nnz the shard also returns its part of A = D~'W D~ (crossing.LevelGram):
        # all of it up to level_nnz nonzeros, else the crossing FEs' own blocks, else none
        sw = Dm.T @ w
        idx = np.flatnonzero(sw > 0)
        aux = {"idx": idx, "DR": np.hstack([DX, DY])[idx]}
        part = None
        if plan.level_nnz:
            try:
                DD = _dummy_gram(plan, local_codes, w, Dm, gcodes, P, sparse_out=True).tocoo()
            except ValueError:  # a local FE component too large for the Schur solve
                DD = None
            if DD is not None:
                DD.sum_duplicates()
                d = DD.diagonal()
                r, c, v = DD.row, DD.col, DD.data
                part = {"full": len(v) <= plan.level_nnz, "blocks": True, "nnz": int(len(v))}
                if not part["full"]:
                    fe_of = np.repeat(np.arange(len(plan.global_fes)), [L for *_, L in plan.global_fes])
                    keep = fe_of[r] == fe_of[c]
                    r, c, v = r[keep], c[keep], v[keep]
                    part["blocks"] = len(v) <= plan.level_nnz
                    part["block_nnz"] = int(len(v))
                if part["blocks"]:
                    it = np.int32 if P < 2**31 else np.int64
                    part.update(r=r.astype(it), c=c.astype(it), v=v)
            aux["part"] = part
        if part is None:
            # diag(D'W Q_A D), A the largest local FE (exact with one): Jacobi
            d = sw.copy()
            if n and local_codes:
                a = max(local_codes, key=lambda c: int(c.max())).astype(np.int64)
                WA = np.bincount(a, weights=w)
                CA = sp.csr_matrix((w, (a, np.arange(n))), shape=(len(WA), n)) @ Dm
                d -= np.asarray(CA.multiply(CA).T @ (1.0 / WA)).ravel()
        aux["ddiag"] = d[idx]
        T["crossprod"] = time.perf_counter() - t
        return Y, Yt, Xt, Xw.T @ Xt, Xw.T @ Yt, (local_codes, Dm), converged, aux
    sparse_out = P > SPARSE_GRAM_LEVELS
    DD = _dummy_gram(plan, local_codes, w, Dm, gcodes, P, sparse_out)
    sel = np.flatnonzero(zpos >= 0)
    zp = zpos[sel]
    if sparse_out:
        # a dense k_tot^2 Gram per shard is too much to ship (450 MB at 7,500 levels);
        # the dummy block is sparse. Dummies follow the k regressors in Z (zp = k, k+1, ...)
        DXs = sp.csr_matrix(DX[sel])
        ZtZ = sp.bmat(
            [[sp.csr_matrix(Xw.T @ Xt), DXs.T], [DXs, sp.csr_matrix(DD)[sel][:, sel]]], format="coo"
        )
        ZtZ.sum_duplicates()  # unique (row, col) pairs: the coordinator adds them in place
        ZtY = np.zeros((k_tot, m))
        ZtY[:k] = Xw.T @ Yt
        ZtY[zp] = DY[sel]
        T["crossprod"] = time.perf_counter() - t
        return Y, Yt, Xt, ZtZ, ZtY, (local_codes, Dm), converged, None
    ZtZ = np.zeros((k_tot, k_tot))
    ZtY = np.zeros((k_tot, m))
    ZtZ[:k, :k] = Xw.T @ Xt
    ZtZ[np.ix_(zp, zp)] = DD[np.ix_(sel, sel)]
    ZtZ[np.ix_(zp, np.arange(k))] = DX[sel]
    ZtZ[np.ix_(np.arange(k), zp)] = DX[sel].T
    ZtY[:k] = Xw.T @ Yt
    ZtY[zp] = DY[sel]
    T["crossprod"] = time.perf_counter() - t
    return Y, Yt, Xt, ZtZ, ZtY, (local_codes, Dm), converged, None


def _prepare(plan: Plan, shard_id: int, files: list[str], loaded: dict | None = None):
    """The shard's estimation sample: rows, local FE codes (dense) and weights."""
    df = _load(plan, files, loaded)
    n0 = df.height
    if plan.keep_dir:  # singletons over all FEs, found in the passes before round 1
        keep = np.load(_keep_path(plan, shard_id))
        if not keep.all():
            df = df.filter(pl.Series(keep))
    local_codes = [densify(df[c].to_numpy())[0] for _, c in plan.local_fes]
    if plan.drop_singletons and local_codes and not plan.keep_dir:
        keep = drop_singletons(local_codes)
        if not keep.all():
            df = df.filter(pl.Series(keep))
            local_codes = [densify(c[keep])[0] for c in local_codes]
    w = df[plan.weights].to_numpy().astype(np.float64) if plan.weights else np.ones(df.height)
    return df, n0, local_codes, w


def _round1(plan: Plan, shard_id: int, files: list[str], write_cache: bool):
    t0 = time.perf_counter()
    T = {}
    loaded: dict = {}
    df, n0, local_codes, w = _prepare(plan, shard_id, files, loaded)
    T["read"] = time.perf_counter() - t0
    n = df.height
    gpresent = [
        np.bincount(df[c].to_numpy().astype(np.int64), minlength=L) > 0 for _, c, L in plan.global_fes
    ]

    if plan.sparse_global:
        Y, Yt, Zt, ZtZ, ZtY, extra, converged, aux = _round1_sparse(
            plan, shard_id, df, n0, local_codes, w, T, t0
        )
        t = time.perf_counter() - T.pop("crossprod")  # keep timing the cluster bookkeeping
    else:
        Y, Yt, Zt, ZtZ, ZtY, converged, t = _round1_dense(plan, df, n, local_codes, w, T)
        extra = aux = None
    out = {
        "shard": shard_id,
        "rows_in": n0,
        "load": loaded,
        "row_key": rowset(df, list(plan.row_key_cols)) if plan.row_key_cols else None,
        "n": n,
        "sumw": float(w.sum()),
        "ZtZ": ZtZ,
        "ZtY": ZtY,
        "YtY": (Yt * w[:, None]).T @ Yt,  # full: IV needs the endogenous columns' cross-products
        # raw sums for R2 when there are no fixed effects
        "sum_wy": (Y * w[:, None]).sum(0),
        "sum_wy2": (Y * Y * w[:, None]).sum(0),
        "fe_levels": [int(c.max()) + 1 if len(c) else 0 for c in local_codes],
        "global_present": gpresent,
        "converged": converged,
    }
    if aux is not None:
        out["crossing"] = aux
        _iter_store(plan, shard_id, extra[0], extra[1], w)
    return _round1_finish(plan, shard_id, df, local_codes, w, Yt, Zt, out, T, t, t0, write_cache, extra)


def _round1_dense(plan: Plan, df, n, local_codes, w, T):
    f = plan.formula
    t = time.perf_counter()
    Y = _outcome_block(plan, df, n)
    cols: list[np.ndarray] = []
    if "Intercept" in plan.colnames:
        cols.append(np.ones(n))
    for term in (*f.terms, *f.instruments):
        cols.extend(_term_columns(term, df, plan.cat_levels))
    # Global (low-cardinality) FEs enter as dummies, first level dropped.
    for _, c, L in plan.global_fes:
        codes = df[c].to_numpy().astype(np.int64)
        D = np.zeros((n, L - 1))
        m = codes > 0
        D[np.nonzero(m)[0], codes[m] - 1] = 1.0
        cols.append(D)
    if plan.global_fes and not plan.local_fes:
        cols.append(np.ones(n))  # hidden constant: the dummies' dropped level
    Z = np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols]) if cols else np.zeros((n, 0))
    T["build"] = time.perf_counter() - t

    t = time.perf_counter()
    YZ = np.hstack([Y, Z])
    converged = True
    if local_codes and n:
        YZ, converged = demean(YZ, local_codes, w, plan.fe_tol, plan.fe_maxiter, plan.fe_method)
    m = Y.shape[1]
    Yt, Zt = YZ[:, :m], YZ[:, m:]
    T["demean"] = time.perf_counter() - t

    t = time.perf_counter()
    Zw = Zt * w[:, None]
    return Y, Yt, Zt, Zw.T @ Zt, Zw.T @ Yt, converged, t


def _round1_finish(plan, shard_id, df, local_codes, w, Yt, Zt, out, T, t, t0, write_cache, extra):
    if plan.cluster:
        cname, ccol, G = plan.cluster
        cl = df[ccol].to_numpy().astype(np.int64)
        out["cluster_present"] = np.unique(cl)
        # is each local FE nested in the cluster? (exact: FE groups are local)
        nested = []
        for codes in local_codes:
            if not len(codes):
                nested.append(True)
                continue
            lo = np.full(codes.max() + 1, np.iinfo(np.int64).max)
            hi = np.full(codes.max() + 1, np.iinfo(np.int64).min)
            np.minimum.at(lo, codes.astype(np.int64), cl)
            np.maximum.at(hi, codes.astype(np.int64), cl)
            nested.append(bool((lo == hi).all()))
        out["local_fe_nested_in_cluster"] = nested
        # global FEs: per-level min/max cluster, reduced on the coordinator
        gmm = []
        for _, c, L in plan.global_fes:
            codes = df[c].to_numpy().astype(np.int64)
            lo = np.full(L, np.iinfo(np.int64).max)
            hi = np.full(L, np.iinfo(np.int64).min)
            np.minimum.at(lo, codes, cl)
            np.maximum.at(hi, codes, cl)
            gmm.append((lo, hi))
        out["global_cluster_minmax"] = gmm
    else:
        cl = None
    T["crossprod"] = time.perf_counter() - t

    t = time.perf_counter()
    if plan.cache_dir and write_cache and not plan.sparse_global:
        p = _cache_path(plan, shard_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, Y=Yt, Z=Zt, w=w, cl=cl if cl is not None else np.zeros(0, np.int64))
        os.replace(tmp, p)
    T["write"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t0
    T["maxrss_gb"] = _maxrss_gb()
    out["timing"] = T
    arrays = (Yt, Zt, w, cl if cl is not None else np.zeros(0, np.int64), extra)
    return out, arrays


def _iter_store(plan: Plan, shard_id: int, local_codes, Dm, w) -> tuple:
    idx = np.unique(Dm.indices)  # levels present in the shard
    nbytes = sum(c.nbytes for c in local_codes) + Dm.data.nbytes + Dm.indices.nbytes + w.nbytes
    entry = (local_codes, Dm, w, idx, nbytes)
    if plan.iter_token:
        _ITER_CACHE[(plan.iter_token, shard_id)] = entry
        used = sum(v[-1] for v in _ITER_CACHE.values())
        while _ITER_CACHE and used > ITER_CACHE_GB * 1e9:
            used -= _ITER_CACHE.popitem(last=False)[1][-1]
    return entry


def shard_matvec(plan: Plan, shard_id: int, files: list[str], p: np.ndarray) -> dict:
    """One CG step on this shard: D'W Q D p (P x c) for the broadcast block p,
    returned on the levels the shard has (idx)."""
    t0 = time.perf_counter()
    key = (plan.iter_token, shard_id)
    hit = key in _ITER_CACHE
    if hit:
        _ITER_CACHE.move_to_end(key)
        entry = _ITER_CACHE[key]
    else:  # a worker that has not seen this shard (new, or the cache was full)
        df, _, local_codes, w = _prepare(plan, shard_id, files)
        entry = _iter_store(plan, shard_id, local_codes, _global_matrix(df, plan, df.height)[0], w)
    local_codes, Dm, w, idx, _ = entry
    t_read = time.perf_counter() - t0
    U = Dm @ p
    if local_codes and len(w):
        U = demean(U, local_codes, w, plan.fe_tol, plan.fe_maxiter, plan.fe_method)[0]
    S = Dm.T @ (U * w[:, None])
    return {
        "shard": shard_id,
        "idx": idx,
        "S": S[idx],
        "timing": {
            "read": t_read,
            "total": time.perf_counter() - t0,
            "cache_hit": hit,
            "maxrss_gb": _maxrss_gb(),
        },
    }


def _apply(T: np.ndarray, H) -> np.ndarray:
    """T @ H, with H given as a matrix or as column indices of T."""
    return T[:, H] if np.asarray(H).ndim == 1 else T @ H


def _meat(plan: Plan, X: np.ndarray, E: np.ndarray, w: np.ndarray, cl, out: dict, prefix: str = "") -> None:
    """RSS and the meats of the equations in E's columns, with score regressors X (n x k)."""
    m = E.shape[1]
    out[prefix + "rss"] = np.einsum("ij,ij,i->j", E, E, w)
    S = [X * (w * E[:, j])[:, None] for j in range(m)]
    k = X.shape[1]
    if "hetero" in plan.vcov_types:
        out[prefix + "meat_hetero"] = np.array([s.T @ s for s in S]).reshape(m, k, k)
    if "CRV1" in plan.vcov_types and plan.cluster:
        order = np.argsort(cl, kind="stable")
        cls = cl[order]
        starts = np.flatnonzero(np.r_[True, cls[1:] != cls[:-1]])
        codes = cls[starts]
        Sg = np.empty((m, len(codes), k))
        for j in range(m):
            Sg[j] = np.add.reduceat(S[j][order], starts, axis=0) if len(cls) else np.zeros((0, k))
        if plan.cluster_local:
            out[prefix + "meat_crv"] = np.einsum("mgk,mgl->mkl", Sg, Sg)
        else:
            out[prefix + "cluster_codes"] = codes
            out[prefix + "cluster_scores"] = Sg


def shard_round2(plan: Plan, shard_id: int, files: list[str], lin: dict) -> dict:
    """Residuals, RSS and meats from linear maps of the shard's columns
    T = [outcome block, Z[:, lin["cols"]]] (outcome block: outcomes, then IV's endogenous
    regressors; built by estimate._ols_solve or iv._round2_maps).

    For each equation j of an equation group (lin itself: the outcomes; lin["fs"]: IV's
    first stages, one per endogenous regressor):
        residual            T @ A[:, j] - D~ G[:, j]
        score regressors    T @ H - D~ GH          (shared by the group's equations)
    D~ (the sparse crossing-FE path only) is the dummy block with the local FE partialled
    out, and G, GH are the dummies' coefficients, so the scores are the regressors with the
    dummies partialled out (Frisch-Waugh-Lovell: the reported block of the full sandwich).
    Elsewhere the dummies are columns of Z and G, GH are None. H may be column indices."""
    t0 = time.perf_counter()
    T = {}
    if plan.cache_dir and _cache_path(plan, shard_id).exists():
        with np.load(_cache_path(plan, shard_id)) as d:
            Yt, Zt, w, cl = d["Y"], d["Z"], d["w"], d["cl"]
        extra = None
    else:  # no cache: recompute the round-1 transformation
        _, (Yt, Zt, w, cl, extra) = _round1(plan, shard_id, files, write_cache=False)
    T["read"] = time.perf_counter() - t0

    t = time.perf_counter()
    Tm = np.hstack([Yt, Zt[:, lin["cols"]]])
    groups = [("", lin)] + ([("fs_", lin["fs"])] if lin.get("fs") else [])
    out = {"shard": shard_id, "n": len(w)}
    dterms = {}
    if lin.get("G") is not None:  # sparse path: one demeaning of every dummy term
        local_codes, Dm = extra
        blocks = []
        for _, g in groups:
            blocks += [g["G"], g["GH"]]
        B = np.hstack(blocks)
        if len(w):
            V = demean(Dm @ B, local_codes, w, plan.fe_tol, plan.fe_maxiter, plan.fe_method)[0]
        else:
            V = np.zeros((0, B.shape[1]))
        o = 0
        for prefix, g in groups:
            ne, kk = g["G"].shape[1], g["GH"].shape[1]
            dterms[prefix] = (V[:, o : o + ne], V[:, o + ne : o + ne + kk])
            o += ne + kk
    for prefix, g in groups:
        E = Tm @ g["A"]
        X = _apply(Tm, g["H"])
        if prefix in dterms:
            E = E - dterms[prefix][0]
            X = X - dterms[prefix][1]
        _meat(plan, X, E, w, cl, out, prefix)
    T["meat"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t0
    T["maxrss_gb"] = _maxrss_gb()
    out["timing"] = T
    return out
