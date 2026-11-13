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
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from .crossing import FAN_IN
from .demean import demean, densify, drop_singletons
from .formula import Formula, Term
from .rowhash import rowset

# Sparse path: above this many crossing-FE levels a shard returns its Gram matrix sparse
SPARSE_GRAM_LEVELS = 2000

# Plan C: shard state kept between conjugate-gradient steps, least recently used first out.
# Its budget is the plan's (crossing.iter_cache_budget: a share of a worker's memory), or
# DFEOLS_ITER_CACHE_GB if set, else ITER_CACHE_DEFAULT_GB. A fit's first entry on a worker
# drops every earlier fit's: the module outlives fits, and D5 (job 35888031) ran pair x year
# after fare on workers still holding fare's 8 GB of state, 15-17 GB idle of 22 GiB, until
# round-1 tasks pushed them past the nanny's 95% limit.
_ITER_CACHE: OrderedDict = OrderedDict()
ITER_CACHE_DEFAULT_GB = 8.0


def _iter_cache_bytes(plan) -> float:
    env = os.environ.get("DFEOLS_ITER_CACHE_GB")
    if env:
        return float(env) * 1e9
    return float(plan.iter_cache_bytes or ITER_CACHE_DEFAULT_GB * 1e9)


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
    iter_cache_bytes: int = 0  # plan C: a worker's budget for that state (0: the default)
    cg_checkpoint: str | None = None  # plan C with a pass per step: where its CG state is kept
    level_nnz: int = 0  # plan C: most nonzeros of A a shard returns (0: no level space)
    part_nnz: int = 0  # and at most this many, for the workers' memory (0: level_nnz alone)
    keep_dir: str | None = None  # per-shard singleton keep masks (when FEs cross the partition)
    row_key_cols: tuple[str, ...] = ()  # raw columns hashed into the record's row_key
    endog_names: list[str] = field(default_factory=list)  # IV: endogenous columns (outcome block)
    inst_names: list[str] = field(default_factory=list)  # IV: excluded instruments (regressor block)
    # sessions' created variables: the sample is the declared variables' (listwise deletion on
    # `filter_vars`, not on the outcomes), and each (name, expression) in `created` is computed
    # from raw columns after it; rows where one is missing are counted, not dropped
    filter_vars: tuple[str, ...] | None = None
    created: tuple[tuple[str, str], ...] = ()
    # two-way clustering: the second cluster (name, column, n_levels) and whether it is nested
    # in the partition key; round 2 adds its meat and the intersection's (prefixes c1_, c2_)
    cluster2: tuple[str, str, int] | None = None
    cluster2_local: bool = False
    # varying slopes (fe[x], fe[[x]]): FE key -> (slope variables, whether the FE keeps its
    # intercept); FEs not in it are plain
    slopes: dict[str, tuple[tuple[str, ...], bool]] = field(default_factory=dict)
    # fit(predict=...): round 2 writes each row's xb, d (the FEs' sum) and residual here,
    # with these raw columns carried along (read only for that; they don't touch the sample)
    predict_dir: str | None = None
    predict_keep: tuple[str, ...] = ()

    @property
    def k_x(self) -> int:
        """Regressor-block columns before the crossing-FE dummies."""
        return len(self.colnames) + len(self.inst_names)

    def fe_slopes(self, key: str) -> tuple[str, ...]:
        return self.slopes.get(key, ((), True))[0]

    def fe_intercept(self, key: str) -> bool:
        return self.slopes.get(key, ((), True))[1]

    @property
    def fe_keys(self) -> list[str]:
        """Every FE, local then crossing (the order of fe_levels and nesting flags)."""
        return [n for n, _ in self.local_fes] + [n for n, *_ in self.global_fes]

    @property
    def fe_labels(self) -> list[str]:
        """Each FE as the formula names it, with its slopes: unit, unit[t], unit[[t]]."""
        from .formula import FETerm

        return [FETerm(k, self.fe_slopes(k), self.fe_intercept(k)).label for k in self.fe_keys]

    @property
    def local_intercept(self) -> list[bool]:
        return [self.fe_intercept(n) for n, _ in self.local_fes]

    @property
    def const_hidden(self) -> bool:
        """A constant column after the dummies, not reported: it stands in for the crossing
        dummies' dropped first levels when no local FE has an intercept. (With only slope-only
        FEs, fe[[x]], the constant is a reported regressor instead, as in fixest.)"""
        return any(self.fe_intercept(n) for n, *_ in self.global_fes) and not any(self.local_intercept)

    def local_slope_columns(self, df) -> list:
        """Per local FE, its slope variables as an (n, s) array, or None for a plain FE."""
        out = []
        for name, _ in self.local_fes:
            v = self.fe_slopes(name)
            out.append(np.column_stack([df[c].to_numpy().astype(np.float64) for c in v]) if v else None)
        return out


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


def _load(plan: Plan, files: list[str], dropped: dict | None = None, extra: tuple[str, ...] = ()):
    """The shard's rows for this spec. `dropped`, if given, gets the rows removed by
    reason (missing, non_finite, zero_weights), for the record's sample counts. `extra`
    columns are read too but play no part in which rows are kept."""
    f = plan.formula
    created = dict(plan.created)
    sample = set(plan.filter_vars) if plan.filter_vars is not None else set(f.variables)
    cols = sample | (set(f.variables) - set(created))
    cols.update(c for _, c in plan.local_fes)
    cols.update(c for _, c, _ in plan.global_fes)
    if plan.cluster:
        cols.add(plan.cluster[1])
    if plan.cluster2:
        cols.add(plan.cluster2[1])
    if plan.weights:
        cols.add(plan.weights)
    cols.update(plan.row_key_cols)
    raw = set()
    if created:
        from .expr import columns as _expr_columns

        for e in created.values():
            raw.update(_expr_columns(e))
    df = pl.read_parquet(files, columns=sorted(cols | raw | set(extra)))
    # listwise deletion on the variables of this spec (a session's: its declared variables)
    num = [c for c in sorted(sample) if df.schema[c].is_float()]
    n = [df.height]
    df = df.drop_nulls(subset=sorted(cols)) if (raw | set(extra)) - cols else df.drop_nulls()
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
    if created:
        from .expr import compile as _expr

        # missing: null, or not finite (log of 0); counted (the session refuses the variable)
        # and set to 0 so the shard's work goes on
        df = df.with_columns([_expr(e).alias(name) for name, e in created.items()]).with_columns(
            [pl.when(pl.col(n).is_finite()).then(pl.col(n)).otherwise(None).alias(n) for n in created]
        )
        if dropped is not None:
            dropped["created_missing"] = {name: int(df[name].is_null().sum()) for name in created}
        df = df.with_columns([pl.col(name).fill_null(0.0) for name in created])
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


def _read_keep(plan: Plan, shard_id: int, n: int) -> np.ndarray:
    """A shard's keep mask: the passes' .npy, or a session's packed copy (.bits.npy, one bit
    a row, kept with the demeaned data so created variables get the same rows)."""
    p = _keep_path(plan, shard_id)
    if p.exists():
        return np.load(p)
    return np.unpackbits(np.load(p.with_suffix(".bits.npy")), count=n).astype(bool)


def _sparse_count(lv: np.ndarray, inv: np.ndarray, keep: np.ndarray):
    """Kept rows per level as (levels, counts), levels with none left out."""
    cnt = np.bincount(inv[keep], minlength=len(lv))
    nz = cnt > 0
    return lv[nz], cnt[nz]


def shard_singletons(plan: Plan, shard_id: int, files: list[str], gcount, first: bool) -> dict:
    """One pass of singleton removal over every FE, as fixest and reghdfe do.

    gcount: per crossing FE, the count of kept rows in each level over all shards
    after the previous pass (None on the first pass, which drops local-FE
    singletons only). Other shards only drop rows, so gcount minus this shard's
    own rows plus its current rows is an upper bound on a level's true count: a
    row is dropped only if it is truly alone, and anything missed is caught on a
    later pass. Writes the keep mask and returns this shard's level counts.

    Works in the shard's own levels (np.unique) and returns them sparse, (levels, counts):
    a dense count per crossing level made every task allocate and ship L int64s (D6 r4,
    15.5M location x day levels: 124 MB per shard, 540 GB to the coordinator over 4,355
    shards; singletons1 ran at a third of r1's speed)."""
    t0 = time.perf_counter()
    df = _load(plan, files)  # the same rows round 1 sees (listwise deletion included)
    n0 = df.height
    keep = np.ones(n0, bool) if first else np.load(_keep_path(plan, shard_id))
    # only FEs with an intercept have singletons (fixest: a slope-only FE drops none)
    local = [densify(df[c].to_numpy())[0] for n, c in plan.local_fes if plan.fe_intercept(n)]
    glob = []  # (this shard's levels, each row's index into them)
    for n, c, _L in plan.global_fes:
        if plan.fe_intercept(n):
            lv, inv = np.unique(df[c].to_numpy().astype(np.int64), return_inverse=True)
            glob.append((lv, inv.ravel()))
    before = int(keep.sum())
    others = None
    if gcount is not None:
        others = [
            np.asarray(gc)[lv] - np.bincount(inv[keep], minlength=len(lv)) for gc, (lv, inv) in zip(gcount, glob)
        ]
    changed = n0 > 0
    while changed:
        changed = False
        for c in local:
            cnt = np.bincount(c[keep], minlength=int(c.max()) + 1)
            bad = keep & (cnt[c] <= 1)
            if bad.any():
                keep &= ~bad
                changed = True
        for i, (lv, inv) in enumerate(glob if others is not None else ()):
            cnt = others[i] + np.bincount(inv[keep], minlength=len(lv))
            bad = keep & (cnt[inv] <= 1)
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
        "gcount": [_sparse_count(lv, inv, keep) for lv, inv in glob],
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


def _fe_matrix(codes: np.ndarray, P: int):
    """n x P indicator of one crossing FE's levels (codes carry its offset)."""
    import scipy.sparse as sp

    n = len(codes)
    return sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, P))


def _local_groups(local_codes) -> np.ndarray:
    """Each row's group for the pattern of D'W P_L D: the local FE's own groups with one, else
    the connected components of the local FEs (each Schur block of _projected_dummy_gram is
    dense over the crossing levels its component touches)."""
    if len(local_codes) == 1:
        return local_codes[0].astype(np.int64)
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    offs = np.cumsum([0] + [int(c.max()) + 1 for c in local_codes])
    a = local_codes[0].astype(np.int64)
    r = np.concatenate([a] * (len(local_codes) - 1))
    c = np.concatenate([b.astype(np.int64) + o for b, o in zip(local_codes[1:], offs[1:-1])])
    G = sp.csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(offs[-1], offs[-1]))
    _, lab = connected_components(G, directed=False)
    return lab[a].astype(np.int64)


CHUNK_FLOPS = 50_000_000  # products per chunk when a shard counts its part of A


def _pattern_nnz(g: np.ndarray, lev: np.ndarray, P: int, lim: int, exact: bool = False) -> int:
    """Nonzeros of the pattern of B'B, B the (group, level) incidence of the pairs (g, lev),
    counted up to `lim` (a number above it once past): first the bounds, sum over groups of
    (levels touched)^2 above and, per level, its largest group below, then by chunks of rows
    (as the planner's _a_size). Within `lim` the result may be the upper bound, unless `exact`."""
    import scipy.sparse as sp

    if not len(g):
        return 0
    B = sp.csr_matrix((np.ones(len(g), np.float32), (g, lev)), shape=(int(g.max()) + 1, P))
    B.sum_duplicates()
    B.data[:] = 1
    deg = np.diff(B.indptr).astype(np.int64)
    Bt = B.T.tocsr()
    present = np.flatnonzero(np.diff(Bt.indptr))
    upper = min(int((deg * deg).sum()), len(present) ** 2)
    if upper <= lim and not exact:
        return upper
    lower = int(np.maximum.reduceat(deg[Bt.indices], Bt.indptr[present]).sum())
    if lower > lim:
        return lower
    work = np.asarray(Bt @ deg.astype(np.float64)).ravel()
    cw = np.cumsum(work)
    cuts = np.unique(
        np.r_[0, np.searchsorted(cw, np.arange(1, int(cw[-1] // CHUNK_FLOPS) + 1) * CHUNK_FLOPS), P]
    )
    nnz = 0
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        nnz += int((Bt[c0:c1] @ B).nnz)
        if nnz > lim:
            return nnz
    return nnz


def _part_size(local_codes, gcodes: np.ndarray, P: int, lim: int) -> dict:
    """Whether a shard's part of A = D~'W D~ fits in `lim` nonzeros, and else its diagonal
    blocks, before building either (its pattern: crossing levels sharing a local group, or
    component, plus D'WD's, which lies inside it)."""
    g = _local_groups(local_codes)
    k = len(gcodes)
    nnz = _pattern_nnz(np.tile(g, k), gcodes.ravel(), P, lim)
    if nnz <= lim:
        return {"full": True, "blocks": True, "nnz": nnz}
    if k == 1:  # one crossing FE: its block is all of A
        return {"full": False, "blocks": False, "nnz": nnz, "block_nnz": nnz}
    bn = sum(_pattern_nnz(g, gcodes[i], P, lim) for i in range(k))
    if bn > lim:  # the blocks' bounds add up past the cap: count them
        bn = 0
        for i in range(k):
            bn += _pattern_nnz(g, gcodes[i], P, lim - bn, exact=True)
            if bn > lim:
                break
    return {"full": False, "blocks": bn <= lim, "nnz": nnz, "block_nnz": bn}


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
            np.tile(w, len(gcodes)),
            (np.tile(a, len(gcodes)), gcodes.ravel()),  # gcodes carry the offsets
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


def _round1_sparse(plan: Plan, shard_id: int, df, n0: int, local_codes, w, T, t0, raw: dict | None = None):
    """Local FEs plus crossing FEs, without dense dummies.

    With Q the within-local-FEs residual maker (W-orthogonal), for the dummy
    block D:  D~'W X~ = D'W X~ (per-level sums),  D~'W D~ = D'W D - D'W P_L D
    (_projected_dummy_gram; with one local FE, C'diag(1/W_a) C where C holds the
    weight in each (local group, level) cell).

    `raw`, if given, receives each column's weighted sum and sum of squares before the FEs
    ("s1", "s2", in the order of the returned Gram's columns: the regressors, then on the
    sparse path the dummies), the scale of the collinearity test (linalg.reference_ss).
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
    if raw is not None:
        raw["s1"], raw["s2"] = w @ X, w @ (X * X)

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
        part, d = None, None
        if plan.level_nnz:
            lim = min(plan.level_nnz, plan.part_nnz) if plan.part_nnz else plan.level_nnz
            # count the part before building it: building it whole and checking after took
            # 110 GB on two days of D5 (A nearly dense over 60,720 zone pairs)
            size = _part_size(local_codes, gcodes, P, lim) if n and local_codes else None
            if size is not None and not size["blocks"]:
                part = {"full": False, "blocks": False, "nnz": size["nnz"], "block_nnz": size["block_nnz"]}
                part["reason"] = (
                    f"shard {shard_id}'s part of A has more than {lim:,} nonzeros, its diagonal "
                    f"blocks too"
                    + (f" (a worker sums {FAN_IN} parts within its memory)" if lim < plan.level_nnz else "")
                )
            else:
                blocks_only = size is not None and not size["full"]
                try:
                    if blocks_only:  # one crossing FE at a time: never all of the part at once
                        DD = sum(
                            _dummy_gram(
                                plan, local_codes, w, _fe_matrix(gcodes[i], P), gcodes[i : i + 1], P, True
                            )
                            for i in range(len(plan.global_fes))
                        ).tocoo()
                    else:
                        DD = _dummy_gram(plan, local_codes, w, Dm, gcodes, P, sparse_out=True).tocoo()
                except ValueError:  # a local FE component too large for the Schur solve
                    DD = None
                if DD is not None:
                    DD.sum_duplicates()
                    d = DD.diagonal()
                    r, c, v = DD.row, DD.col, DD.data
                    full = not blocks_only and len(v) <= lim
                    part = {"full": full, "blocks": True, "nnz": size["nnz"] if blocks_only else int(len(v))}
                    if not full:
                        fe_of = np.repeat(np.arange(len(plan.global_fes)), [L for *_, L in plan.global_fes])
                        keep = fe_of[r] == fe_of[c]
                        r, c, v = r[keep], c[keep], v[keep]
                        part["blocks"] = len(v) <= lim
                        part["block_nnz"] = int(len(v))
                    if lim < plan.level_nnz and not full:
                        part["reason"] = (
                            f"shard {shard_id}'s part has more than {lim:,} nonzeros "
                            f"(a worker sums {FAN_IN} parts within its memory)"
                        )
                    if part["blocks"]:
                        it = np.int32 if P < 2**31 else np.int64
                        part.update(r=r.astype(it), c=c.astype(it), v=v)
            aux["part"] = part
        if d is None:
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
    if raw is not None:  # a 0/1 dummy's sum and sum of squares are both its weight
        sw = np.asarray(Dm.T @ w).ravel()
        for key in ("s1", "s2"):
            v = np.zeros(k_tot)
            v[:k] = raw[key]
            v[zp] = sw[sel]
            raw[key] = v
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


def _prepare(
    plan: Plan, shard_id: int, files: list[str], loaded: dict | None = None, extra: tuple[str, ...] = ()
):
    """The shard's estimation sample: rows, local FE codes (dense) and weights."""
    df = _load(plan, files, loaded, extra)
    n0 = df.height
    if plan.keep_dir:  # singletons over all FEs, found in the passes before round 1
        keep = _read_keep(plan, shard_id, n0)
        if not keep.all():
            df = df.filter(pl.Series(keep))
    local_codes = [densify(df[c].to_numpy())[0] for _, c in plan.local_fes]
    with_int = [c for c, ok in zip(local_codes, plan.local_intercept) if ok]
    if plan.drop_singletons and with_int and not plan.keep_dir:
        keep = drop_singletons(with_int)
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
    gpresent = [np.unique(df[c].to_numpy().astype(np.int64)) for _, c, _ in plan.global_fes]

    raw: dict = {}
    if plan.sparse_global:
        Y, Yt, Zt, ZtZ, ZtY, extra, converged, aux = _round1_sparse(
            plan, shard_id, df, n0, local_codes, w, T, t0, raw
        )
        t = time.perf_counter() - T.pop("crossprod")  # keep timing the cluster bookkeeping
    else:
        Y, Yt, Zt, ZtZ, ZtY, converged, t = _round1_dense(plan, df, n, local_codes, w, T, raw)
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
        # each Z column's sums before the FEs: the collinearity test's scale
        "raw_s1": raw["s1"],
        "raw_s2": raw["s2"],
        "fe_levels": [int(c.max()) + 1 if len(c) else 0 for c in local_codes],
        "global_present": gpresent,
        "converged": converged,
    }
    if aux is not None:
        out["crossing"] = aux
        _iter_store(plan, shard_id, extra[0], extra[1], w)
    return _round1_finish(plan, shard_id, df, local_codes, w, Yt, Zt, out, T, t, t0, write_cache, extra)


def _round1_dense(plan: Plan, df, n, local_codes, w, T, raw: dict | None = None):
    f = plan.formula
    t = time.perf_counter()
    Y = _outcome_block(plan, df, n)
    cols: list[np.ndarray] = []
    if "Intercept" in plan.colnames:
        cols.append(np.ones(n))
    for term in (*f.terms, *f.instruments):
        cols.extend(_term_columns(term, df, plan.cat_levels))
    # Global (low-cardinality) FEs enter as dummies, first level dropped; then each slope
    # variable times every level's dummy (fe[x]; all levels, as the intercept is not theirs)
    for name, c, L in plan.global_fes:
        codes = df[c].to_numpy().astype(np.int64)
        if plan.fe_intercept(name):
            D = np.zeros((n, L - 1))
            m = codes > 0
            D[np.nonzero(m)[0], codes[m] - 1] = 1.0
            cols.append(D)
        for v in plan.fe_slopes(name):
            D = np.zeros((n, L))
            D[np.arange(n), codes] = df[v].to_numpy().astype(np.float64)
            cols.append(D)
    if plan.const_hidden:
        cols.append(np.ones(n))  # hidden constant: the dummies' dropped level, or slopes' intercept
    Z = np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols]) if cols else np.zeros((n, 0))
    T["build"] = time.perf_counter() - t
    if raw is not None:  # before the FEs: the collinearity test's scale (_round1_sparse)
        raw["s1"], raw["s2"] = w @ Z, w @ (Z * Z)

    t = time.perf_counter()
    YZ = np.hstack([Y, Z])
    converged = True
    if local_codes and n:
        YZ, converged = demean(
            YZ,
            local_codes,
            w,
            plan.fe_tol,
            plan.fe_maxiter,
            plan.fe_method,
            slopes=plan.local_slope_columns(df) if plan.slopes else None,
            intercept=plan.local_intercept,
        )
    m = Y.shape[1]
    Yt, Zt = YZ[:, :m], YZ[:, m:]
    T["demean"] = time.perf_counter() - t

    t = time.perf_counter()
    Zw = Zt * w[:, None]
    return Y, Yt, Zt, Zw.T @ Zt, Zw.T @ Yt, converged, t


def _round1_finish(plan, shard_id, df, local_codes, w, Yt, Zt, out, T, t, t0, write_cache, extra):
    if plan.cluster:
        cl = df[plan.cluster[1]].to_numpy().astype(np.int64)
        present, nested, gmm = _cluster_stats(plan, df, local_codes, cl)
        out["cluster_present"] = present
        out["local_fe_nested_in_cluster"] = nested
        out["global_cluster_minmax"] = gmm
        if plan.cluster2:  # two-way: the same for the second, and both codes kept for round 2
            cl2 = df[plan.cluster2[1]].to_numpy().astype(np.int64)
            present, nested, gmm = _cluster_stats(plan, df, local_codes, cl2)
            out["cluster2"] = {"present": present, "local_nested": nested, "global_minmax": gmm}
            cl = np.column_stack([cl, cl2])
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


def _cluster_stats(plan: Plan, df, local_codes, cl: np.ndarray):
    """A shard's cluster codes present; whether each local FE is nested in the cluster (exact:
    FE groups are local); and per global-FE level the min/max cluster, reduced on the
    coordinator (nested iff equal at every level present)."""
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
    gmm = [level_minmax(df[c].to_numpy().astype(np.int64), cl) for _, c, _ in plan.global_fes]
    return np.unique(cl), nested, gmm


# Global-FE levels present in a shard, and each one's smallest and largest cluster, travel
# sparse: only the shard's own levels. Dense over all L levels they were ~265 MB a shard at
# 15.5M location-days (D6 r4, job 35953385: 1.15 TB through the coordinator, round 1 ~4 h).


def level_minmax(codes: np.ndarray, cl: np.ndarray) -> tuple:
    """(levels present, sorted; each one's smallest cluster code; its largest)."""
    lev, inv = np.unique(codes, return_inverse=True)
    lo = np.full(len(lev), np.iinfo(np.int64).max)
    hi = np.full(len(lev), np.iinfo(np.int64).min)
    np.minimum.at(lo, inv, cl)
    np.maximum.at(hi, inv, cl)
    return lev, lo, hi


def merge_present(acc: np.ndarray, p: np.ndarray) -> None:
    """Mark a shard's present levels in the coordinator's dense mask (in place). `p` is the
    sorted level codes; a dense boolean mask (older shard results) is OR-ed in."""
    if p.dtype == bool:
        acc |= p
    else:
        acc[p] = True


def merge_minmax(lo_acc: np.ndarray, hi_acc: np.ndarray, item) -> None:
    """Fold a shard's per-level cluster min/max into the coordinator's (in place). `item` is
    level_minmax's (levels, lo, hi), or a dense (lo, hi) pair over all levels."""
    if len(item) == 3:
        lev, lo, hi = item
        lo_acc[lev] = np.minimum(lo_acc[lev], lo)
        hi_acc[lev] = np.maximum(hi_acc[lev], hi)
    else:
        lo, hi = item
        np.minimum(lo_acc, lo, out=lo_acc)
        np.maximum(hi_acc, hi, out=hi_acc)


def _iter_store(plan: Plan, shard_id: int, local_codes, Dm, w) -> tuple:
    idx = np.unique(Dm.indices)  # levels present in the shard
    nbytes = (
        sum(c.nbytes for c in local_codes)
        + Dm.data.nbytes
        + Dm.indices.nbytes
        + Dm.indptr.nbytes
        + w.nbytes
        + idx.nbytes
    )
    entry = (local_codes, Dm, w, idx, nbytes)
    if plan.iter_token:
        for k in [k for k in _ITER_CACHE if k[0] != plan.iter_token]:  # earlier fits' state
            del _ITER_CACHE[k]
        _ITER_CACHE[(plan.iter_token, shard_id)] = entry
        used = sum(v[-1] for v in _ITER_CACHE.values())
        budget = _iter_cache_bytes(plan)
        while _ITER_CACHE and used > budget:
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


def shard_cells(plan: Plan, shard_id: int, files: list[str]) -> dict:
    """The cell path (estimate.fit(cells=True)): the shard's rows grouped by every FE key and
    the cluster, so each FE and the cluster are constant within a cell. Per cell: rows `n`,
    weight `W`, weighted means `m` of the outcomes and regressors, and their weighted
    cross-products about the cell means `c` (upper triangle, row-major), computed in two
    passes so nothing cancels. Keys: local FE codes (dense within the shard), global FE codes,
    the cluster code. No singleton drop here; the coordinator drops them on the cells."""
    t0 = time.perf_counter()
    loaded: dict = {}
    df = _load(plan, files, loaded)
    n = df.height
    f = plan.formula
    Y = _outcome_block(plan, df, n)
    cols = []
    for term in (*f.terms, *f.instruments):  # the row path's Z block (IV: instruments after)
        cols.extend(_term_columns(term, df, plan.cat_levels) if n else [])
    X = np.column_stack(cols) if cols else np.zeros((n, 0))
    Z = np.hstack([Y, X]) if n else np.zeros((0, Y.shape[1] + plan.k_x))
    p = Z.shape[1]
    w = df[plan.weights].to_numpy().astype(np.float64) if plan.weights else np.ones(n)
    keys = {f"l{i}": densify(df[c].to_numpy())[0].astype(np.int64) for i, (_, c) in enumerate(plan.local_fes)}
    keys.update({f"g{i}": df[c].to_numpy().astype(np.int64) for i, (_, c, _) in enumerate(plan.global_fes)})
    if plan.cluster:
        keys["cl"] = df[plan.cluster[1]].to_numpy().astype(np.int64)
    if plan.cluster2:  # two-way: the second cluster is constant within a cell too
        keys["cl2"] = df[plan.cluster2[1]].to_numpy().astype(np.int64)
    kn = list(keys)
    frame = pl.DataFrame({**keys, "w": w, **{f"z{j}": Z[:, j] for j in range(p)}})
    W = pl.col("w").sum().over(kn)
    dev = [
        (pl.col(f"z{j}") - (pl.col(f"z{j}") * pl.col("w")).sum().over(kn) / W).alias(f"d{j}")
        for j in range(p)
    ]
    pairs = [(i, j) for i in range(p) for j in range(i, p)]
    cells = (
        frame.lazy()
        .with_columns(dev)
        .group_by(kn)
        .agg(
            pl.len().alias("n"),
            pl.col("w").sum().alias("W"),
            *[((pl.col(f"z{j}") * pl.col("w")).sum() / pl.col("w").sum()).alias(f"m{j}") for j in range(p)],
            *[(pl.col("w") * pl.col(f"d{i}") * pl.col(f"d{j}")).sum().alias(f"c{i}_{j}") for i, j in pairs],
        )
        .collect()
    )
    return {
        "shard": shard_id,
        "rows_in": n,
        "load": loaded,
        "row_key": rowset(df, list(plan.row_key_cols)) if plan.row_key_cols else None,
        "keys": cells.select(kn).to_numpy().astype(np.int64),
        "key_names": kn,
        "n": cells["n"].to_numpy().astype(np.int64),
        "W": cells["W"].to_numpy(),
        "m": cells.select([f"m{j}" for j in range(p)]).to_numpy(),
        "c": cells.select([f"c{i}_{j}" for i, j in pairs]).to_numpy(),
        "timing": {"read": 0.0, "total": time.perf_counter() - t0, "maxrss_gb": _maxrss_gb()},
    }


def _apply(T: np.ndarray, H) -> np.ndarray:
    """T @ H, with H given as a matrix or as column indices of T."""
    return T[:, H] if np.asarray(H).ndim == 1 else T @ H


def _meat(plan: Plan, X: np.ndarray, E: np.ndarray, w: np.ndarray, cl, out: dict, prefix: str = "") -> None:
    """RSS and the meats of the equations in E's columns, with score regressors X (n x k)."""
    m = E.shape[1]
    out[prefix + "rss"] = np.einsum("ij,ij,i->j", E, E, w)
    k = X.shape[1]
    hetero = "hetero" in plan.vcov_types
    crv = "CRV1" in plan.vcov_types and plan.cluster
    if not (hetero or crv):
        return
    # One equation's scores at a time: with many equations (IV first stages, one per
    # endogenous regressor) all of them at once is m x n x k (tens of GB with ~100 dummies).
    if hetero:
        meat_h = np.empty((m, k, k))
    if crv:
        order = np.argsort(cl, kind="stable")
        cls = cl[order]
        starts = np.flatnonzero(np.r_[True, cls[1:] != cls[:-1]])
        codes = cls[starts]
        if plan.cluster_local:
            meat_c = np.empty((m, k, k))
        else:
            Sg = np.empty((m, len(codes), k))
    for j in range(m):
        s = X * (w * E[:, j])[:, None]
        if hetero:
            meat_h[j] = s.T @ s
        if crv:
            sg = np.add.reduceat(s[order], starts, axis=0) if len(cls) else np.zeros((0, k))
            if plan.cluster_local:
                meat_c[j] = sg.T @ sg
            else:
                Sg[j] = sg
        del s
    if hetero:
        out[prefix + "meat_hetero"] = meat_h
    if crv:
        if plan.cluster_local:
            out[prefix + "meat_crv"] = meat_c
        else:
            out[prefix + "cluster_codes"] = codes
            out[prefix + "cluster_scores"] = Sg


def _reported_columns(plan: Plan, df, n: int) -> np.ndarray:
    """The reported regressors (plan.colnames) on raw rows, before any FE: round 1's first
    columns of Z."""
    cols: list[np.ndarray] = []
    if "Intercept" in plan.colnames:
        cols.append(np.ones(n))
    for term in plan.formula.terms:
        cols.extend(_term_columns(term, df, plan.cat_levels))
    X = np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols]) if cols else np.zeros((n, 0))
    if X.shape[1] != len(plan.colnames):
        raise RuntimeError(f"predict: rebuilt {X.shape[1]} regressor columns, expected {len(plan.colnames)}")
    return X


def _write_prediction(plan: Plan, shard_id: int, files: list[str], lin: dict, E: np.ndarray) -> int:
    """In-sample prediction for the shard's estimation rows: xb (reported regressors times
    their coefficients), resid (round 2's residual, net of every FE, local and crossing) and
    d = y - xb - resid, the sum of the row's FE terms (fixest's predict() is xb + d)."""
    df, _, _, _ = _prepare(plan, shard_id, files, extra=plan.predict_keep)
    n = df.height
    if n != E.shape[0]:
        raise RuntimeError(f"predict: shard {shard_id} has {n} rows, round 2 had {E.shape[0]}")
    ys = list(plan.formula.depvars)
    Y = _outcome_block(plan, df, n)[:, : len(ys)]
    X = _reported_columns(plan, df, n)
    xb = X[:, lin["xb_idx"]] @ lin["xb_beta"] if len(lin["xb_idx"]) else np.zeros((n, len(ys)))
    resid = E[:, : len(ys)]
    d = Y - xb - resid
    cols = {c: df[c] for c in plan.predict_keep}
    for j, y in enumerate(ys):
        sfx = "" if len(ys) == 1 else f"_{y}"
        cols.update({f"xb{sfx}": xb[:, j], f"d{sfx}": d[:, j], f"resid{sfx}": resid[:, j]})
    path = Path(plan.predict_dir) / f"part-{shard_id:05d}.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    pl.DataFrame(cols).write_parquet(tmp)
    os.replace(tmp, path)
    return n


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
    else:  # no cache: recompute the round-1 transformation, without plan C's part of A
        # (round 1 has summed it; taxi2009: rebuilding 100M+ nonzeros per shard here, outside
        # the task memory estimate, killed 24 GB workers)
        rplan = replace(plan, level_nnz=0, part_nnz=0) if plan.level_nnz else plan
        _, (Yt, Zt, w, cl, extra) = _round1(rplan, shard_id, files, write_cache=False)
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
        if prefix == "" and plan.predict_dir:
            t_p = time.perf_counter()
            out["predicted"] = _write_prediction(plan, shard_id, files, g, E)
            T["predict"] = time.perf_counter() - t_p
        if plan.cluster2 is None:
            _meat(plan, X, E, w, cl, out, prefix)
            continue
        # two-way (Cameron, Gelbach and Miller): the meats of a, b and the intersection a&b
        # (codes a * levels_b + b, nested in the partition if either is), combined on the
        # coordinator as V_a + V_b - V_ab
        a, b = cl[:, 0], cl[:, 1]
        _meat(plan, X, E, w, a, out, prefix)
        crv_only = ("CRV1",)
        nb = SimpleNamespace(vcov_types=crv_only, cluster=plan.cluster2, cluster_local=plan.cluster2_local)
        _meat(nb, X, E, w, b, out, prefix + "c1_")
        local_ab = plan.cluster_local or plan.cluster2_local
        nab = SimpleNamespace(vcov_types=crv_only, cluster=("ab", None, None), cluster_local=local_ab)
        _meat(nab, X, E, w, a * int(plan.cluster2[2]) + b, out, prefix + "c2_")
    T["meat"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t0
    T["maxrss_gb"] = _maxrss_gb()
    out["timing"] = T
    return out
