# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Shard-level work. These functions run on workers and must stay picklable.

Singletons (`shard_singletons`): with FEs that cross the partition, singletons
are found over all FEs at once in passes over the shards (see
estimate._drop_singletons); each shard's keep mask goes to plan.keep_dir.

Round 1 (`shard_round1`): build the design for one shard, drop singletons,
demean by FEs nested in the partition key, and return cross-products.
The demeaned arrays are cached to disk for round 2.

Round 2 (`shard_round2`): residuals, RSS and the meat of the sandwich.
"""

from __future__ import annotations

import os
import time

try:
    import resource
except ImportError:  # pragma: no cover (Windows)
    resource = None
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .demean import demean, densify, drop_singletons
from .formula import Formula, Term
from .rowhash import rowset


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
    keep_dir: str | None = None  # per-shard singleton keep masks (when FEs cross the partition)
    row_key_cols: tuple[str, ...] = ()  # raw columns hashed into the record's row_key


def plan_columns(f: Formula, cat_levels: dict[str, list], has_fe: bool):
    """Reported regressors. With any FE the constant is absorbed (with only
    dummy FEs it is kept as a hidden column after the dummies, standing in
    for their dropped first level, and is not reported)."""
    names: list[str] = []
    if f.intercept and not has_fe:
        names.append("Intercept")
    for t in f.terms:
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
    k_rep = len(plan.colnames)
    offs, zpos, o, zo = [], [], 0, k_rep
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


def _round1_sparse(plan: Plan, shard_id: int, df, n0: int, local_codes, w, T, t0):
    """One local FE plus crossing FEs, without dense dummies.

    With Q the within-local-FE residual maker (W-orthogonal), for the dummy
    block D:  D~'W X~ = D'W X~ (per-level sums),  D~'W D~ = D'W D - C'diag(1/W_a) C
    where C holds the weight in each (local group, level) cell.
    """
    import scipy.sparse as sp

    f = plan.formula
    t = time.perf_counter()
    n = df.height
    a = local_codes[0].astype(np.int64)
    nA = int(a.max()) + 1 if n else 0
    Y = (
        np.column_stack([df[y].to_numpy().astype(np.float64) for y in f.depvars])
        if n
        else np.zeros((0, len(f.depvars)))
    )
    cols = []
    for term in f.terms:
        cols.extend(_term_columns(term, df, plan.cat_levels))
    X = np.column_stack(cols) if cols else np.zeros((n, 0))
    Dm, gcodes = _global_matrix(df, plan, n)
    T["build"] = time.perf_counter() - t

    t = time.perf_counter()
    m = Y.shape[1]
    YX = demean(np.hstack([Y, X]), [a], w)[0] if n else np.zeros((0, m + X.shape[1]))
    Yt, Xt = YX[:, :m], YX[:, m:]
    T["demean"] = time.perf_counter() - t

    t = time.perf_counter()
    k = Xt.shape[1]
    offs, zpos, P = _global_layout(plan)
    k_tot = k + plan.n_dummy
    Xw, Yw = Xt * w[:, None], Yt * w[:, None]
    DX = Dm.T @ Xw  # P x k
    DY = Dm.T @ Yw  # P x m
    DWD = (Dm.T @ sp.diags(w) @ Dm).toarray()
    if n:
        C = sp.csr_matrix(
            (
                np.tile(w, len(plan.global_fes)),
                (np.tile(a, len(plan.global_fes)), gcodes.ravel()),  # gcodes carry the offsets
            ),
            shape=(nA, P),
        )
        C.sum_duplicates()
        WA = np.bincount(a, weights=w, minlength=nA)
        if nA * P <= 50_000_000:
            # dense BLAS product: cost nA*P^2 but vectorised; the sparse product
            # costs sum over groups of (levels per group)^2 and is slow when
            # each group spans many levels (e.g. units observed in 1000 periods)
            Cd = C.toarray()
            DD = DWD - (Cd / WA[:, None]).T @ Cd
        else:
            DD = DWD - (C.T @ sp.diags(1.0 / WA) @ C).toarray()
    else:
        DD = DWD
    sel = np.flatnonzero(zpos >= 0)
    zp = zpos[sel]
    ZtZ = np.zeros((k_tot, k_tot))
    ZtY = np.zeros((k_tot, m))
    ZtZ[:k, :k] = Xw.T @ Xt
    ZtZ[np.ix_(zp, zp)] = DD[np.ix_(sel, sel)]
    ZtZ[np.ix_(zp, np.arange(k))] = DX[sel]
    ZtZ[np.ix_(np.arange(k), zp)] = DX[sel].T
    ZtY[:k] = Xw.T @ Yt
    ZtY[zp] = DY[sel]
    T["crossprod"] = time.perf_counter() - t
    return Y, Yt, Xt, ZtZ, ZtY, (a, Dm)


def _round1(plan: Plan, shard_id: int, files: list[str], write_cache: bool):
    t0 = time.perf_counter()
    T = {}
    loaded: dict = {}
    df = _load(plan, files, loaded)
    T["read"] = time.perf_counter() - t0

    t = time.perf_counter()
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
    n = df.height
    w = df[plan.weights].to_numpy().astype(np.float64) if plan.weights else np.ones(n)
    gpresent = [
        np.bincount(df[c].to_numpy().astype(np.int64), minlength=L) > 0 for _, c, L in plan.global_fes
    ]

    if plan.sparse_global:
        Y, Yt, Zt, ZtZ, ZtY, extra = _round1_sparse(plan, shard_id, df, n0, local_codes, w, T, t0)
        converged = True
        t = time.perf_counter() - T.pop("crossprod")  # keep timing the cluster bookkeeping
    else:
        Y, Yt, Zt, ZtZ, ZtY, converged, t = _round1_dense(plan, df, n, local_codes, w, T)
        extra = None
    out = {
        "shard": shard_id,
        "rows_in": n0,
        "load": loaded,
        "row_key": rowset(df, list(plan.row_key_cols)) if plan.row_key_cols else None,
        "n": n,
        "sumw": float(w.sum()),
        "ZtZ": ZtZ,
        "ZtY": ZtY,
        "YtY": np.einsum("ij,ij,i->j", Yt, Yt, w),
        # raw sums for R2 when there are no fixed effects
        "sum_wy": (Y * w[:, None]).sum(0),
        "sum_wy2": (Y * Y * w[:, None]).sum(0),
        "fe_levels": [int(c.max()) + 1 if len(c) else 0 for c in local_codes],
        "global_present": gpresent,
        "converged": converged,
    }
    return _round1_finish(plan, shard_id, df, local_codes, w, Yt, Zt, out, T, t, t0, write_cache, extra)


def _round1_dense(plan: Plan, df, n, local_codes, w, T):
    f = plan.formula
    t = time.perf_counter()
    Y = (
        np.column_stack([df[y].to_numpy().astype(np.float64) for y in f.depvars])
        if n
        else np.zeros((0, len(f.depvars)))
    )
    cols: list[np.ndarray] = []
    if "Intercept" in plan.colnames:
        cols.append(np.ones(n))
    for term in f.terms:
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


def shard_round2(
    plan: Plan, shard_id: int, files: list[str], beta: np.ndarray, keep: np.ndarray, fwl: dict | None = None
) -> dict:
    """beta: (k_kept x m) on the kept columns of Z.

    fwl (sparse crossing-FE path): {"rep": kept reported columns, "beta_x",
    "gamma": (P x m) dummy coefficients by level, "pi": (P x |rep|)}. The meat
    is then formed for the reported coefficients only, from X~ with the dummies
    partialled out (Frisch-Waugh-Lovell), which equals the corresponding block
    of the full sandwich."""
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
    if fwl is not None:
        a, Dm = extra
        Xr = Zt[:, fwl["rep"]]
        if len(a):
            V = demean(np.hstack([Dm @ fwl["gamma"], Dm @ fwl["pi"]]), [a], w)[0]
        else:
            V = np.zeros((0, fwl["gamma"].shape[1] + fwl["pi"].shape[1]))
        m_ = fwl["gamma"].shape[1]
        E = Yt - Xr @ fwl["beta_x"] - V[:, :m_]
        Zk = Xr - V[:, m_:]  # X with dummies partialled out
    else:
        Zk = Zt[:, keep]
        E = Yt - Zk @ beta  # n x m
    m = E.shape[1]
    out = {"shard": shard_id, "n": len(w), "rss": np.einsum("ij,ij,i->j", E, E, w)}
    k = Zk.shape[1]
    if "hetero" in plan.vcov_types:
        H = np.empty((m, k, k))
        for j in range(m):
            S = Zk * (w * E[:, j])[:, None]
            H[j] = S.T @ S
        out["meat_hetero"] = H
    if "CRV1" in plan.vcov_types and plan.cluster:
        order = np.argsort(cl, kind="stable")
        cls = cl[order]
        starts = np.flatnonzero(np.r_[True, cls[1:] != cls[:-1]])
        codes = cls[starts]
        Sg = np.empty((m, len(codes), k))
        for j in range(m):
            S = (Zk * (w * E[:, j])[:, None])[order]
            Sg[j] = np.add.reduceat(S, starts, axis=0) if len(S) else np.zeros((0, k))
        if plan.cluster_local:
            out["meat_crv"] = np.einsum("mgk,mgl->mkl", Sg, Sg)
        else:
            out["cluster_codes"] = codes
            out["cluster_scores"] = Sg
    T["meat"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t0
    T["maxrss_gb"] = _maxrss_gb()
    out["timing"] = T
    return out
