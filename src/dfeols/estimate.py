# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Coordinator: plan the fit, run two map/reduce rounds, compute inference."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from . import iv as _iv
from . import rowhash
from .executors import SerialExecutor
from .formula import fe_components, parse
from .linalg import collinear_keep, solve_spd
from .prep import Dataset
from .progress import Progress, fmt_td
from .results import Result
from .vcov import df_k as _df_k
from .vcov import vcov_crv1, vcov_hetero, vcov_iid
from .worker import (
    Plan,
    plan_columns,
    shard_cells,
    shard_matvec,
    shard_round1,
    shard_round2,
    shard_singletons,
    term_names,
)

log = logging.getLogger("dfeols")


def _norm(name: str) -> str:
    return "^".join(fe_components(name))


def _levels_task(paths: list[str], cols: list[str]) -> dict:
    t = time.perf_counter()
    lf = pl.scan_parquet(paths)
    frames = pl.collect_all([lf.select(c).drop_nulls().unique() for c in cols], engine="streaming")
    return {
        "levels": {c: fr[c].to_list() for c, fr in zip(cols, frames)},
        "timing": {"total": time.perf_counter() - t},
    }


def category_levels(ds: Dataset, cols: list[str], executor=None) -> dict[str, list]:
    """Sorted levels of each i() variable, read on the executor's workers and
    cached in <shard_dir>/_levels/ (the dataset never changes after prep).

    Reading one column of every shard on the coordinator took 55 minutes at
    1e10 rows on Lustre, before any worker started; as tasks over batches of
    shards it is a short map/reduce, and later fits on the same data reuse it."""
    out, todo = {}, []
    cache = ds.root / "_levels"
    for c in cols:
        f = cache / f"{c}.json"
        try:
            d = json.loads(f.read_text())
            if d.get("n_rows") == ds.n_rows:
                out[c] = d["levels"]
                continue
        except (FileNotFoundError, ValueError, KeyError):
            pass
        todo.append(c)
    if not todo:
        return out
    executor = executor or SerialExecutor()
    n_tasks = max(1, min(len(ds.shards), 4 * int(getattr(executor, "max_workers", None) or 1)))
    groups: list[list[dict]] = [[] for _ in range(n_tasks)]
    for i, s in enumerate(ds.shards):
        groups[i % n_tasks].append(s)
    tasks = [
        (i, max(sum(s["rows"] for s in g), 1), ([p for s in g for p in ds.shard_paths(s)], todo))
        for i, g in enumerate(groups)
        if g
    ]
    found: dict[str, set] = {c: set() for c in todo}
    seen: set = set()

    def on_result(tid, r):
        if tid in seen:
            return
        seen.add(tid)
        for c in todo:
            found[c].update(r["levels"][c])

    t0 = time.time()
    prog = Progress(None, label="levels", every=60)
    prog.add_stage("levels", {tid: rows for tid, rows, _ in tasks})
    prog.start("levels")
    executor.run(_levels_task, tasks, on_result, prog, "levels")
    prog.finish("levels")
    cache.mkdir(exist_ok=True)
    for c in todo:
        out[c] = sorted(found[c])
        try:
            tmp = cache / f".{c}.json.tmp"
            tmp.write_text(json.dumps({"levels": out[c], "n_rows": ds.n_rows}))
            os.replace(tmp, cache / f"{c}.json")
        except (TypeError, ValueError, OSError):  # unusual level types: just don't cache
            pass
    log.info("fit: levels of %s from %d tasks (%.0fs)", ", ".join(todo), len(tasks), time.time() - t0)
    return out


def build_plan(
    ds: Dataset,
    formula: str,
    *,
    cluster=None,
    vcov=None,
    weights=None,
    cache_dir=None,
    max_dummy_levels: int = 2000,
    drop_singletons=True,
    fe_tol: float = 1e-10,
    fe_method: str = "auto",
    crossing_fe: str = "auto",
    executor=None,
) -> Plan:
    f = parse(formula)
    local, glob = [], []
    for name in f.fes:
        key = _norm(name)
        if key not in ds.fes:
            raise KeyError(f"FE {name!r} was not encoded by prep(); available: {list(ds.fes)}")
        m = ds.fes[key]
        if m["nested_in_partition"]:
            local.append((key, m["column"]))
        else:
            glob.append((key, m["column"], m["n_levels"]))
    # Dense dummies cost rows x levels x 8 bytes per shard; the sparse path keeps
    # per-level sums instead, so only it takes crossing FEs above max_dummy_levels
    big = [(k, L) for k, _, L in glob if L > max_dummy_levels]
    n_dum = sum(L - 1 for *_, L in glob)
    iterate = _use_iterate(crossing_fe, local, glob, bool(big), n_dum)
    if iterate and parse(formula).iv:
        raise NotImplementedError(
            "IV with crossing_fe='iterate' (plan C) is not built yet; crossing FEs this large "
            "need it" + ("" if crossing_fe == "iterate" else f" ({n_dum:,} crossing-FE dummies)")
        )
    if big and not iterate and not (local and crossing_fe != "dense"):
        big_fe, big_L = big[0]
        raise NotImplementedError(
            f"FE {big_fe!r} crosses partitions and has {big_L:,} levels (> max_dummy_levels="
            f"{max_dummy_levels}). Crossing FEs that large need the sparse path, which needs "
            f"at least one FE nested in the partition key (this spec has {len(local)}"
            + (", and crossing_fe='dense'" if crossing_fe == "dense" else "")
            + "). Dense dummies would need about rows x levels x 8 bytes per shard."
        )
    cl = None
    cl_local = False
    if cluster:
        key = _norm(cluster)
        if key not in ds.clusters:
            raise KeyError(f"cluster {cluster!r} was not encoded by prep(); available: {list(ds.clusters)}")
        c = ds.clusters[key]
        cl = (key, c["column"], c["n_levels"])
        cl_local = bool(c["nested_in_partition"])
    if vcov is None:
        vcov = ["CRV1"] if cluster else ["iid"]
    vcov = tuple([vcov] if isinstance(vcov, str) else vcov)
    for v in vcov:
        if v not in ("iid", "hetero", "CRV1"):
            raise ValueError(f"vcov {v!r} not supported (iid, hetero, CRV1)")
        if v == "CRV1" and not cluster:
            raise ValueError("CRV1 needs cluster=")

    cats = sorted({t.cat for t in (*f.terms, *f.endog, *f.instruments) if t.cat})
    missing_cats = set(cats) - set(ds.columns)
    if missing_cats:
        raise KeyError(f"Variables not in the prepared data: {sorted(missing_cats)}")
    cat_levels = category_levels(ds, cats, executor)
    missing = (f.variables | ({weights} if weights else set())) - set(ds.columns)
    if missing:
        raise KeyError(f"Variables not in the prepared data: {sorted(missing)}")

    colnames = plan_columns(f, cat_levels, has_fe=bool(local or glob))
    # the spec's raw columns (FE and cluster components, not their codes) identify its rows
    key = set(f.variables) | ({weights} if weights else set())
    for name in [*f.fes, *([cluster] if cluster else [])]:
        key.update(fe_components(name))
    row_key_cols = tuple(sorted(c for c in key if c in ds.columns))
    spec_id = f.fingerprint(cluster=cluster, weights=weights, singletons=drop_singletons)
    return Plan(
        formula=f,
        spec_id=spec_id,
        local_fes=local,
        global_fes=glob,
        cluster=cl,
        cluster_local=cl_local,
        weights=weights,
        cat_levels=cat_levels,
        colnames=colnames,
        n_dummy=sum(L - 1 for *_, L in glob) + (1 if glob and not local else 0),
        vcov_types=vcov,
        cache_dir=str(cache_dir) if cache_dir else None,
        drop_singletons=drop_singletons,
        fe_tol=fe_tol,
        fe_method=fe_method,
        sparse_global=iterate or _use_sparse(crossing_fe, local, glob, bool(big)),
        crossing_iterate=iterate,
        row_key_cols=row_key_cols,
        endog_names=term_names(f.endog, cat_levels),
        inst_names=term_names(f.instruments, cat_levels),
    )


def estimate_worker_gb(plan: Plan, rows: int, n_cols: int) -> float:
    """Rough peak worker memory for one shard of `rows` rows.

    Columns held densely: outcomes and reported regressors always; crossing-FE
    dummies only on the dense path (the sparse path keeps per-level sums, a
    groups x levels block, instead). Calibrated on OSCER: 2M rows, 7 dense
    columns -> 2.8-3.4 GB observed (Polars read buffers, numpy temporaries,
    Dask worker overhead ~1 GB)."""
    m = len(plan.formula.depvars) + len(plan.endog_names)
    if plan.crossing_iterate:
        return worker_gb(rows, n_cols, m, plan.k_x, 0) + rows * (m + plan.k_x) * 8 * 2 / 1e9
    levels = sum(L for *_, L in plan.global_fes) if plan.sparse_global else 0
    return worker_gb(rows, n_cols, m, plan.k_x, plan.n_dummy, levels)


def worker_gb(rows: int, n_cols: int, m: int, k_x: int, n_dummy: int, sparse_levels: int = 0) -> float:
    """estimate_worker_gb from counts alone (the planner has no Plan): `m` outcomes and
    endogenous regressors, `k_x` regressor-block columns, `n_dummy` crossing-FE dummies,
    and `sparse_levels` the crossing FEs' levels on the sparse path (0 = dense path)."""
    dense = m + k_x + (0 if sparse_levels else n_dummy)
    gb = 1.5  # worker process, Dask, libraries
    gb += rows * n_cols * 8 * 1.5 / 1e9  # Parquet read and filtered copy
    gb += rows * dense * 8 * 4 / 1e9  # design, demeaned copy, products
    if sparse_levels:
        from .worker import SPARSE_GRAM_LEVELS

        levels = sparse_levels
        if 200 < levels <= SPARSE_GRAM_LEVELS:
            gb += rows * 0.05 * levels * 8 / 1e9  # group x level block
        gb += (min(levels, SPARSE_GRAM_LEVELS) ** 2 + k_x * levels) * 8 * 3 / 1e9  # Gram blocks
    return gb


def estimate_coordinator_gb(plan: Plan) -> float:
    """Rough coordinator memory for the dense Gram matrix of regressors and crossing-FE
    dummies: the accumulator, its kept-column copy, the Cholesky factor, the inverse and
    the dummy block solved for round 2's maps (about 6 copies). Plan C keeps only
    level x column blocks (CG's iterate, residual, direction and product)."""
    if plan.crossing_iterate:
        P = sum(L for *_, L in plan.global_fes)
        return coordinator_gb(plan.k_x) + P * (plan.k_x + len(plan.formula.depvars)) * 8 * 8 / 1e9
    return coordinator_gb(plan.k_x + plan.n_dummy)


def coordinator_gb(k_tot: int) -> float:
    """estimate_coordinator_gb for a Gram matrix of `k_tot` columns."""
    return k_tot**2 * 8 * 6 / 1e9


def _check_worker_memory(executor, label: str, rows: int, need_gb: float) -> None:
    """Fail before submitting work that cannot fit on a worker.

    A shard that exceeds worker memory kills every worker that tries it (the
    scheduler retries it on a fresh worker each time), which wastes the queue
    wait of each one. The estimate is rough, so warn above the profile's
    worker memory and refuse only well above it."""
    prof = getattr(executor, "profile", None)
    mem = getattr(prof, "worker", {}).get("memory") if prof is not None else None
    if not mem:
        return
    from .cluster import _memory_bytes

    have_gb = _memory_bytes(mem) / 1e9
    msg = (
        f"fit {label}: the largest shard ({rows:,} rows) needs roughly {need_gb:.0f} GB, "
        f"and profile {prof.name!r} gives workers {mem}"
    )
    if need_gb > 1.5 * have_gb:
        raise MemoryError(
            msg + ". Re-prep with a smaller target_rows or a partition key without one huge "
            "group (`dfeols info <shard_dir>` shows shard sizes), or raise [worker] memory."
        )
    if need_gb > have_gb:
        log.warning("%s; workers may run out of memory", msg)


# Above this many crossing-FE dummies the coordinator's dense solve (about 43 GB at 30k)
# gives way to conjugate gradient across shards (plan C)
DENSE_SOLVE_LEVELS = 30_000


def _use_iterate(crossing_fe: str, local, glob, big: bool, n_dummy: int) -> bool:
    """Plan C: the crossing-FE coefficients by distributed conjugate gradient. "auto" takes
    it when no direct path fits: more than DENSE_SOLVE_LEVELS dummies, or a crossing FE
    above max_dummy_levels with no FE nested in the partition."""
    if crossing_fe == "iterate":
        if not glob:
            raise ValueError("crossing_fe='iterate' needs a crossing FE")
        return True
    return crossing_fe == "auto" and bool(glob) and (n_dummy > DENSE_SOLVE_LEVELS or (big and not local))


def _use_sparse(crossing_fe: str, local, glob, big: bool = False) -> bool:
    """Crossing (dummy) FEs via per-level sums. Needs a local FE; "auto" takes it with
    exactly one (closed-form projection) or when a crossing FE exceeds max_dummy_levels
    (several local FEs: per-component Schur solves, worker._projected_dummy_gram)."""
    if crossing_fe not in ("auto", "sparse", "dense", "iterate"):
        raise ValueError("crossing_fe must be auto, sparse, dense or iterate")
    ok = len(local) >= 1 and len(glob) > 0
    use = ok and (crossing_fe == "sparse" or (crossing_fe == "auto" and (len(local) == 1 or big)))
    n_dummy = sum(L - 1 for *_, L in glob)
    if not use and n_dummy > 300 and crossing_fe != "dense":
        log.warning(
            "%d crossing-FE dummy columns will be built densely (crossing_fe='sparse' takes "
            "them as per-level sums when an FE is nested in the partition); watch worker memory",
            n_dummy,
        )
    if crossing_fe == "sparse" and not ok:
        raise ValueError(
            "crossing_fe='sparse' needs an FE nested in the partition and at least one crossing FE"
        )
    return use


def _drop_singletons(plan: Plan, ds: Dataset, executor, prog: Progress, rows: dict, label: str) -> dict:
    """Singletons over all FEs, as fixest and reghdfe do, when some FE crosses the
    partition (a shard can't see a crossing FE's other rows). Pass 1 drops singletons
    of the local FEs and counts the crossing FEs' levels; each later pass drops rows
    whose level count (from the previous pass) is 1, within shards iterating to a
    fixed point, until a pass drops nothing. Usually two or three passes."""
    t0 = time.time()
    gcount, p, total = None, 0, 0
    while True:
        p += 1
        name = f"singletons{p}"
        acc = {"g": [np.zeros(L, np.int64) for *_, L in plan.global_fes], "dropped": 0}

        def on(tid, r, acc=acc):
            acc["dropped"] += r["dropped"]
            for i, c in enumerate(r["gcount"]):
                acc["g"][i] += c

        tasks = [
            (s["id"], rows[s["id"]], (plan, s["id"], ds.shard_paths(s), gcount, p == 1)) for s in ds.shards
        ]
        prog.add_stage(name, rows)
        prog.start(name)
        executor.run(shard_singletons, tasks, on, prog, name)
        prog.finish(name)
        total += acc["dropped"]
        if p > 1 and acc["dropped"] == 0:
            break
        gcount = acc["g"]
    log.info(
        "fit %s: dropped %s singleton rows over all FEs in %d passes (%s)",
        label,
        f"{total:,}",
        p,
        fmt_td(time.time() - t0),
    )
    return {"passes": p, "seconds": time.time() - t0}


def _ols_solve(plan: Plan, acc: dict, collin_tol: float) -> _iv.Solution:
    """OLS from the round-1 cross-products, and round 2's maps (worker.shard_round2):
    residuals Y - Z beta, score regressors the kept columns (on the sparse path, the
    reported ones with the dummies partialled out)."""
    m = len(plan.formula.depvars)
    k_rep = len(plan.colnames)
    if plan.crossing_iterate:
        return _ols_solve_iterate(plan, acc, collin_tol)
    # dummies first, so a regressor the crossing FEs absorb is the column dropped (as fixest
    # does); in Z order it was kept and a dummy dropped, which reports an unidentified b
    k_x, k_tot = plan.k_x, acc["ZtZ"].shape[0]
    order = np.r_[np.arange(k_x, k_tot), np.arange(k_x)]
    keep = np.zeros(k_tot, bool)
    keep[order[collinear_keep(acc["ZtZ"][np.ix_(order, order)], collin_tol)]] = True
    beta_k, bread = solve_spd(acc["ZtZ"][np.ix_(keep, keep)], acc["ZtY"][keep])
    dropped = [plan.colnames[i] for i in range(k_rep) if not keep[i]]
    kept = np.flatnonzero(keep)
    rep_idx = np.flatnonzero(keep[:k_rep])  # reported & kept, in Z order
    pos_in_kept = np.cumsum(keep) - 1
    rep_kept_pos = pos_in_kept[rep_idx]
    if plan.sparse_global:
        from .worker import _global_layout

        _, zpos, P = _global_layout(plan)
        dum_kept_pos = pos_in_kept[kept[kept >= k_rep]]
        dum_z = kept[kept >= k_rep]
        level_of_z = {int(z): i for i, z in enumerate(zpos) if z >= 0}
        lv = np.array([level_of_z[int(z)] for z in dum_z], np.int64)
        A = acc["ZtZ"][np.ix_(kept, kept)]
        gamma = np.zeros((P, m))
        gamma[lv] = beta_k[dum_kept_pos]
        pi = np.zeros((P, len(rep_idx)))
        if len(dum_kept_pos):
            pi[lv] = np.linalg.solve(
                A[np.ix_(dum_kept_pos, dum_kept_pos)], A[np.ix_(dum_kept_pos, rep_kept_pos)]
            )
        lin = {
            "cols": rep_idx,
            "A": np.vstack([np.eye(m), -beta_k[rep_kept_pos]]),
            "H": m + np.arange(len(rep_idx)),
            "G": gamma,
            "GH": pi,
        }
        bread_use = bread[np.ix_(rep_kept_pos, rep_kept_pos)]
        meat_idx = np.arange(len(rep_idx))  # meat is already reported-only
    else:
        lin = {"cols": kept, "A": np.vstack([np.eye(m), -beta_k]), "H": m + np.arange(len(kept))}
        lin["G"] = lin["GH"] = None
        bread_use = bread
        meat_idx = rep_kept_pos
    names = [plan.colnames[i] for i in rep_idx]
    return _iv.Solution(names, beta_k[rep_kept_pos], [bread_use] * m, meat_idx, lin, dropped)


def _ols_solve_iterate(plan: Plan, acc: dict, collin_tol: float) -> _iv.Solution:
    """Plan C: OLS by Frisch-Waugh-Lovell from the CG solution Gamma of
    (D~'W D~) Gamma = D~'W [X~ Y~]: the regressors and outcomes net of the dummies
    are X~ - D~ Gamma_X and Y~ - D~ Gamma_Y, so
        X^'W X^ = X~'W X~ - (D~'W X~)' Gamma_X,   X^'W Y^ = X~'W Y~ - (D~'W X~)' Gamma_Y
    and round 2 subtracts D~ (Gamma_Y - Gamma_X beta) from the residuals and D~ Gamma_X
    from the scores (lin["G"], lin["GH"]), as on the sparse path."""
    m = len(plan.formula.depvars)
    k = plan.k_x
    Gam, R = acc["Gamma"], acc["DR"]
    XX = acc["ZtZ"] - R[:, :k].T @ Gam[:, :k]
    XX = (XX + XX.T) / 2
    XY = acc["ZtY"] - R[:, :k].T @ Gam[:, k:]
    keep = collinear_keep(XX, collin_tol, ref_diag=np.diag(acc["ZtZ"]))
    kept = np.flatnonzero(keep)
    beta, bread = solve_spd(XX[np.ix_(kept, kept)], XY[kept])
    lin = {
        "cols": kept,
        "A": np.vstack([np.eye(m), -beta]),
        "H": m + np.arange(len(kept)),
        "G": Gam[:, k:] - Gam[:, kept] @ beta,
        "GH": Gam[:, kept],
    }
    dropped = [plan.colnames[i] for i in range(k) if not keep[i]]
    names = [plan.colnames[i] for i in kept]
    return _iv.Solution(names, beta, [bread] * m, np.arange(len(kept)), lin, dropped)


def _cg_steps_left(hist: list[float], tol: float, window: int = 30) -> int | None:
    """Steps until the residual reaches tol at its average rate of fall over the last
    `window` steps (CG residuals stall and drop, so this is rough); None before 3 steps
    or while the residual is not falling."""
    if len(hist) < 4:
        return None
    h = np.log(hist[-window - 1 :])
    fall = (h[0] - h[-1]) / (len(h) - 1)
    return max(int(np.ceil((h[-1] - np.log(tol)) / fall - 1e-9)), 0) if fall > 0 else None


def _crossing_cg(
    plan: Plan,
    ds: Dataset,
    executor,
    prog: Progress,
    rows: dict,
    acc: dict,
    label: str,
    tol: float,
    maxiter: int,
    precond: str = "auto",
) -> np.ndarray:
    """Plan C: solve A Gamma = D~'W [X~ Y~], A = D~'W D~, by preconditioned conjugate gradient,
    one column per regressor and outcome (crossing.pcg). In level space (A summed from the
    shards' parts in round 1, acc["gram"]) every step is a product on the coordinator; otherwise
    each step is one map/reduce round (worker.shard_matvec), and dfeols warns that convergence
    may be slow. The preconditioner follows A's block structure (crossing.choose_method)."""
    from . import crossing as cx

    t0 = time.time()
    R = acc["DR"]
    P, c = R.shape
    offs = np.r_[0, np.cumsum([L for *_, L in plan.global_fes])].astype(np.int64)
    names = [n for n, *_ in plan.global_fes]
    gram = acc.get("gram")
    A, blocks = gram.finish() if gram is not None else (None, None)
    level = A is not None
    if not level:
        why = gram.reason if gram is not None else "level space is off (level_space_gb=0)"
        if gram is not None and gram.blocks_reason:
            why += "; without its diagonal blocks the preconditioner is Jacobi"
        msg = (
            f"fit {label}: the crossing-FE system is too large to solve on the coordinator ({why}), "
            "so each conjugate-gradient step is a pass over the shards and convergence may be slow. "
            "A partition key that nests the largest crossing FE makes the system smaller "
            "(`dfeols plan` compares keys); level_space_gb raises the budget"
        )
        log.warning(msg)
        warnings.warn(msg, cx.SlowConvergenceWarning, stacklevel=3)
    if level:
        diag = A.diagonal()
    elif blocks is not None:
        diag = np.concatenate([b.diagonal() for b in blocks])
    else:
        diag = acc["ddiag"]
    solvers = None
    if blocks is not None and precond != "jacobi":
        singular = bool(plan.local_fes)
        solvers = [cx.BlockSolve(b, singular, n) for b, n in zip(blocks, names)]
    method = cx.choose_method(solvers, level, precond)
    M = cx.make_precond(method, diag, offs, solvers, A)
    act = diag > 1e-12 * max(float(diag.max(initial=0.0)), 1.0)
    t_setup = time.time() - t0
    log.info(
        "fit %s: crossing-FE CG %s, %s preconditioner%s (setup %s)",
        label,
        "in level space" if level else "with a pass per step",
        {"jacobi": "Jacobi", "bjacobi": "block-Jacobi", "sgs": "AP-sweep"}[method],
        f"; A {A.nnz:,} nonzeros over {P:,} levels" if level else "",
        fmt_td(t_setup),
    )

    step = {"it": 0, "t": time.time()}

    def matvec_pass(p):
        Ap = np.zeros((P, c))

        def on(tid, res, Ap=Ap):
            Ap[res["idx"]] += res["S"]

        name = f"cg{step['it'] + 1}"
        tasks = [(s["id"], rows[s["id"]], (plan, s["id"], ds.shard_paths(s), p)) for s in ds.shards]
        prog.add_stage(name, rows, before="round2")
        prog.start(name)
        executor.run(shard_matvec, tasks, on, prog, name)
        prog.finish(name, log=False)
        return Ap

    hist = [1.0]  # relative residuals, starting from the right-hand side

    def on_step(it, rel):
        step["it"] = it
        hist.append(float(rel.max()))
        left = _cg_steps_left(hist, tol)
        dt = time.time() - step["t"]
        step["t"] = time.time()
        prog.extra_eta = None if left is None else left * dt
        prog.note = f"CG step {it}, residual {rel.max():.1e} (tol {tol:g})" + (
            f", about {left} more steps" if left is not None else ""
        )
        if it == 1 or it % (100 if level else 10) == 0:
            log.info("fit %s: crossing-FE CG step %d, largest relative residual %.1e", label, it, rel.max())

    X, it, rel = cx.pcg((lambda p: A @ p) if level else matvec_pass, R, M, act, tol, maxiter, on_step)
    prog.note, prog.extra_eta = "", None
    if rel > tol:
        raise RuntimeError(
            f"fit {label}: crossing-FE CG did not converge in {maxiter} steps (largest relative "
            f"residual {rel:.1e} > {tol:g}); raise iter_maxiter, or partition so more FEs "
            "are nested in the key"
        )
    log.info("fit %s: crossing-FE CG converged in %d steps (%s)", label, it, fmt_td(time.time() - t0))
    acc["cg"] = {
        "steps": it,
        "seconds": time.time() - t0,
        "setup_seconds": t_setup,
        "max_relative_residual": rel,
        "mode": "level" if level else "pass",
        "preconditioner": method,
        "passes": 0 if level else it,
        "A_nnz": int(A.nnz) if level else None,
        "block_nnz": [int(b.nnz) for b in blocks] if blocks is not None else None,
        "fallback_reason": None if level else (gram.reason if gram is not None else "level_space_gb=0"),
    }
    return X


def _fit_cells(
    plan: Plan,
    ds: Dataset,
    executor,
    prog: Progress,
    rows: dict,
    label: str,
    *,
    cluster_local: bool,
    collin_tol: float,
    tol: float,
    maxiter: int,
    level_space_gb: float | None,
    precond: str,
) -> dict:
    """fit(cells=True): one pass of worker.shard_cells, then dfeols.cells on the coordinator."""
    from . import cells as _cells
    from .crossing import level_space_budget

    parts: list = []
    acc = {
        "rows_in": 0,
        "load": {"rows": 0, "missing": 0, "non_finite": 0, "zero_weights": 0},
        "row_key": [],
        "converged": True,
    }
    peak = {"rss": 0.0}

    def on(tid, r):
        peak["rss"] = max(peak["rss"], r["timing"].get("maxrss_gb", 0.0))
        parts.append(r)
        acc["rows_in"] += r["rows_in"]
        for key, v in r.get("load", {}).items():
            acc["load"][key] += v
        if r.get("row_key"):
            acc["row_key"].append(r["row_key"])

    prog.add_stage("cells", rows)
    tasks = [(s["id"], rows[s["id"]], (plan, s["id"], ds.shard_paths(s))) for s in ds.shards]
    prog.start("cells")
    executor.run(shard_cells, tasks, on, prog, "cells")
    prog.finish("cells")
    t1 = time.time()
    df = _cells.combine(parts, len(plan.local_fes), bool(plan.cluster and cluster_local))
    parts.clear()  # the cells live in df now
    nl = len(plan.local_fes)
    fe_cols = [["shard", f"l{i}"] for i in range(nl)] + [[f"g{i}"] for i in range(len(plan.global_fes))]
    cl_cols = (["shard", "cl"] if cluster_local else ["cl"]) if plan.cluster else None
    f = plan.formula
    r = _cells.solve(
        df,
        fe_names=[n for n, _ in plan.local_fes] + [n for n, *_ in plan.global_fes],
        fe_cols=fe_cols,
        cluster_cols=cl_cols,
        m=len(f.depvars),
        k=len(plan.colnames),
        colnames=plan.colnames,
        depvars=list(f.depvars),
        vcov_types=plan.vcov_types,
        drop_singletons=plan.drop_singletons,
        collin_tol=collin_tol,
        tol=tol,
        maxiter=maxiter,
        max_nnz=level_space_budget(executor, level_space_gb),
        precond=precond,
        label=label,
    )
    acc["sumw"] = float(df["W"].sum())
    acc["rows_in"] = r["rows"]
    info = r["info"]
    log.info(
        "fit %s: %s rows in %s cells; %s absorbed, crossing %s (%s, %s steps); solve %s",
        label,
        f"{r['N']:,}",
        f"{info['cells']:,}",
        info["absorbed"],
        ", ".join(info["crossing"]) or "none",
        info["mode"],
        info["steps"],
        fmt_td(time.time() - t1),
    )
    st = prog.stages["cells"]
    return {
        **r,
        "acc": acc,
        "peak_rss": peak["rss"],
        "timing": {
            "cells_seconds": st.finished - st.started,
            "solve_seconds": time.time() - t1,
            "singleton_passes": info["singleton_passes"],
            "singleton_seconds": 0.0,
        },
    }


def fit(
    ds: Dataset | str,
    formula: str,
    *,
    cluster: str | None = None,
    vcov=None,
    weights: str | None = None,
    executor=None,
    workdir: str | None = None,
    cache: bool | str = "auto",
    keep_cache: bool = False,
    max_dummy_levels: int = 2000,
    drop_singletons: bool = True,
    fe_tol: float = 1e-10,
    fe_method: str = "auto",
    crossing_fe: str = "auto",
    collin_tol: float = 1e-9,
    iter_tol: float = 1e-8,
    iter_maxiter: int = 5_000,
    level_space_gb: float | None = None,
    crossing_precond: str = "auto",
    cells: bool = False,
    label: str | None = None,
    log_every: float = 30.0,
    record: bool | str | os.PathLike = True,
    publication: dict | None = None,
    redact: bool = False,
    estimator: str = "2sls",
    fuller: float | None = None,
) -> Result:
    """Estimate an HDFE OLS or IV model on a prepared, sharded dataset.

    cluster    cluster variable (encoded in prep); default vcov is then CRV1
    vcov       "iid", "hetero", "CRV1" or a list of them (all computed in one pass)
    executor   SerialExecutor() (default) or DaskExecutor(client)
    workdir    where status.json and the demeaned cache go; defaults to
               <shard_dir>/runs/<spec id>. Put it on shared scratch.
    record     write `estimation` records, one per response: True puts them at
               <workdir>/result.estimation.json (result.<y>.estimation.json with
               several responses), a path puts them there, False skips them
    publication  `publication` block for the record (citation, locator, links)
    redact     leave identifying details (user, host, paths, job ids) out of the record
    estimator  IV formulas (`y ~ x | fe | d ~ z`): "2sls" (default) or "liml"
    fuller     Fuller's modified LIML with constant a (estimator="liml"); a=1 is common
    crossing_fe  how FEs that cross the partition are absorbed: "dense" dummy columns,
               "sparse" per-level sums with a dense solve on the coordinator, "iterate"
               conjugate gradient across shards (plan C; iter_tol, iter_maxiter), or
               "auto" (the default) to pick from the FE structure
    iter_tol   plan C: CG stops when every column's residual is below this, relative to its
               right-hand side (default 1e-8)
    level_space_gb  plan C: coordinator memory for A = D~'W D~, built in round 1 so CG runs
               with no further passes (default: a quarter of the profile's coordinator memory,
               or of this machine's; 0 turns it off). Above it each CG step is a pass over the
               shards, and dfeols warns (crossing.SlowConvergenceWarning)
    crossing_precond  plan C: "auto" (from A's block structure), "jacobi", "bjacobi" or "sgs"
               (AP-sweep, level space only)
    cells      fit from per-cell sums (dfeols.cells): one pass groups each shard's rows by every
               FE key and the cluster, and the coordinator solves on the cells, exactly. Pays
               when the data compress (DB1B's market table: 32x). OLS; vcov CRV1 and iid
    """
    if not isinstance(ds, Dataset):
        ds = Dataset.open(ds)
    executor = executor or SerialExecutor()
    t_start = time.time()
    cpu_start = time.process_time()

    probe = parse(formula)
    if cache == "auto":
        # Caching demeaned arrays between rounds only pays when demeaning is
        # expensive: iterative (3+ local FEs, or 2 with fe_method="iterate"). With one local FE the
        # exact one-pass demean is cheaper than writing and re-reading
        # n x k doubles on shared scratch.
        n_local = sum(
            1 for f in probe.fes if ds.fes.get("^".join(fe_components(f)), {}).get("nested_in_partition")
        )
        cache = n_local >= 3 or (n_local == 2 and fe_method == "iterate")
    cache = bool(cache)
    iv_opts = {"estimator": estimator, "fuller": fuller} if probe.iv else {}
    if not probe.iv and (estimator != "2sls" or fuller is not None):
        raise ValueError("estimator= and fuller= apply to IV formulas (y ~ x | fe | d ~ z)")
    spec_id = probe.fingerprint(cluster=cluster, weights=weights, singletons=drop_singletons, **iv_opts)
    wd = Path(workdir) if workdir else ds.root / "runs" / spec_id
    wd.mkdir(parents=True, exist_ok=True)
    plan = build_plan(
        ds,
        formula,
        cluster=cluster,
        vcov=vcov,
        weights=weights,
        cache_dir=(wd / "cache") if cache else None,
        max_dummy_levels=max_dummy_levels,
        drop_singletons=drop_singletons,
        fe_tol=fe_tol,
        fe_method=fe_method,
        crossing_fe=crossing_fe,
        executor=executor,
    )
    if cells:
        from . import cells as _cells

        if f_iv := plan.formula.iv:
            raise NotImplementedError(f"cells=True is OLS only ({f_iv})")
        bad = [v for v in plan.vcov_types if v not in _cells.VCOV]
        if bad:
            raise ValueError(
                f"cells=True computes vcov {', '.join(_cells.VCOV)}; not {', '.join(bad)} "
                "(HC1 needs each row's residual)"
            )
        if not (plan.local_fes or plan.global_fes):
            raise ValueError("cells=True needs at least one fixed effect")
        plan.crossing_iterate = plan.sparse_global = False
        plan.cache_dir = None
        cache = False
    if plan.sparse_global and plan.cache_dir:
        plan.cache_dir = None  # recomputing is cheap on this path
        cache = False
    if plan.crossing_iterate:
        from .crossing import level_space_budget

        plan.iter_token = f"{wd}:{uuid.uuid4().hex}"
        plan.level_nnz = level_space_budget(executor, level_space_gb)
    f = plan.formula
    label = label or spec_id
    m = len(f.depvars)
    mY = m + len(plan.endog_names)  # outcome block: outcomes, then IV's endogenous regressors
    k_rep = len(plan.colnames)
    k_tot = plan.k_x + (0 if plan.crossing_iterate else plan.n_dummy)
    log.info(
        "fit %s: %s | %d regressors + %d FE dummies, %d local FEs, %d shards, %s rows",
        label,
        f.text,
        k_rep,
        plan.n_dummy,
        len(plan.local_fes),
        len(ds.shards),
        f"{ds.n_rows:,}",
    )

    biggest = max(s["rows"] for s in ds.shards)
    need_gb = estimate_worker_gb(plan, biggest, len(ds.columns))
    log.info(
        "fit %s: largest shard %s rows -> peak worker memory roughly %.1f GB; cache on disk roughly %.1f GB",
        label,
        f"{biggest:,}",
        need_gb,
        ds.n_rows * (m + k_tot + 2) * 8 / 1e9 if cache else 0,
    )
    _check_worker_memory(executor, label, biggest, need_gb)
    coord_gb = estimate_coordinator_gb(plan)
    if coord_gb > 4:
        log.warning(
            "fit %s: %d crossing-FE dummies: the coordinator needs roughly %.0f GB for the Gram matrix",
            label,
            plan.n_dummy,
            coord_gb,
        )

    prog = Progress(wd, label=label, every=log_every, round2_recomputes=not cache)
    rows = {s["id"]: max(s["rows"], 1) for s in ds.shards}
    sing = {"passes": 0, "seconds": 0.0}
    if cells:
        cres = _fit_cells(
            plan,
            ds,
            executor,
            prog,
            rows,
            label,
            cluster_local=plan.cluster_local,
            collin_tol=collin_tol,
            tol=iter_tol,
            maxiter=iter_maxiter,
            level_space_gb=level_space_gb,
            precond=crossing_precond,
        )
        acc, out, N, G = cres["acc"], cres["out"], cres["N"], cres["G"]
        fe_levels, nested, dropped, coefnames = (
            cres["fe_levels"],
            cres["nested"],
            cres["dropped"],
            cres["coefnames"],
        )
        iv_info = {}
        peak = {"rss": cres["peak_rss"]}
        timing = cres["timing"]
    else:
        if plan.drop_singletons and plan.global_fes:
            plan.keep_dir = str(wd / "singletons")
            sing = _drop_singletons(plan, ds, executor, prog, rows, label)
        prog.add_stage("round1", rows)
        prog.add_stage("round2", rows)

        # ---------------- round 1 ----------------
        acc = {
            "ZtZ": np.zeros((k_tot, k_tot)),
            "ZtY": np.zeros((k_tot, mY)),
            "YtY": np.zeros((mY, mY)),
            "sum_wy": np.zeros(mY),
            "sum_wy2": np.zeros(mY),
            "n": 0,
            "rows_in": 0,
            "sumw": 0.0,
            "converged": True,
            "fe_levels": np.zeros(len(plan.local_fes), np.int64),
            "local_nested": np.ones(len(plan.local_fes), bool),
            "gpresent": [np.zeros(L, bool) for *_, L in plan.global_fes],
            "glo": [np.full(L, np.iinfo(np.int64).max) for *_, L in plan.global_fes],
            "ghi": [np.full(L, np.iinfo(np.int64).min, np.int64) for *_, L in plan.global_fes],
            "cpresent": (
                np.zeros(plan.cluster[2], bool) if plan.cluster and not plan.cluster_local else None
            ),
            "G_local": 0,
            "row_n": {},
            "load": {"rows": 0, "missing": 0, "non_finite": 0, "zero_weights": 0},
            "row_key": [],
        }
        if plan.crossing_iterate:
            P_lev = sum(L for *_, L in plan.global_fes)
            acc["DR"] = np.zeros((P_lev, k_tot + mY))
            acc["ddiag"] = np.zeros(P_lev)
            if plan.level_nnz:
                from .crossing import LevelGram

                offs = np.r_[0, np.cumsum([L for *_, L in plan.global_fes])].astype(np.int64)
                acc["gram"] = LevelGram(offs, [n for n, *_ in plan.global_fes], plan.level_nnz)

        peak = {"rss": 0.0}

        def on1(tid, r):
            peak["rss"] = max(peak["rss"], r["timing"].get("maxrss_gb", 0.0))
            g = r["ZtZ"]
            if hasattr(g, "row"):  # sparse path with many levels: unique (row, col) pairs
                acc["ZtZ"][g.row, g.col] += g.data
            else:
                acc["ZtZ"] += g
            for key in ("ZtY", "YtY", "sum_wy", "sum_wy2"):
                acc[key] += r[key]
            acc["n"] += r["n"]
            acc["rows_in"] += r["rows_in"]
            for key, v in r.get("load", {}).items():
                acc["load"][key] += v
            if r.get("row_key"):
                acc["row_key"].append(r["row_key"])
            acc["sumw"] += r["sumw"]
            if "crossing" in r:
                cr = r["crossing"]
                acc["DR"][cr["idx"]] += cr["DR"]
                acc["ddiag"][cr["idx"]] += cr["ddiag"]
                if "gram" in acc:
                    acc["gram"].add(cr.get("part"), tid)
            acc["row_n"][tid] = r["n"]
            acc["converged"] &= r["converged"]
            acc["fe_levels"] += np.asarray(r["fe_levels"], np.int64)
            for i, p in enumerate(r["global_present"]):
                acc["gpresent"][i] |= p
            if plan.cluster:
                if plan.cluster_local:
                    # nested in the partition key: each cluster lives in one shard,
                    # so counts add up (ids may be hashes, not dense codes)
                    acc["G_local"] += len(r["cluster_present"])
                else:
                    acc["cpresent"][r["cluster_present"]] = True
                if r["n"]:
                    acc["local_nested"] &= np.asarray(r["local_fe_nested_in_cluster"], bool)
                for i, (lo, hi) in enumerate(r["global_cluster_minmax"]):
                    acc["glo"][i] = np.minimum(acc["glo"][i], lo)
                    acc["ghi"][i] = np.maximum(acc["ghi"][i], hi)

        tasks = [(s["id"], rows[s["id"]], (plan, s["id"], ds.shard_paths(s))) for s in ds.shards]
        prog.start("round1")
        executor.run(shard_round1, tasks, on1, prog, "round1")
        prog.finish("round1")
        if not acc["converged"]:
            log.warning("fit %s: FE demeaning did not converge in some shards", label)

        N = acc["n"]
        if plan.crossing_iterate:
            acc["Gamma"] = _crossing_cg(
                plan, ds, executor, prog, rows, acc, label, iter_tol, iter_maxiter, crossing_precond
            )
        if f.iv:
            sol = _iv.solve(plan, acc, N, estimator=estimator, fuller=fuller, collin_tol=collin_tol)
        else:
            sol = _ols_solve(plan, acc, collin_tol)
        dropped = sol.dropped
        if dropped:
            log.info("fit %s: dropped collinear regressors: %s", label, dropped)

        # ---------------- round 2 ----------------
        acc2: dict = {}
        kk = sol.bread[0].shape[0]
        p_ = len(plan.endog_names)
        shapes = {"": (m, kk)}
        if f.iv:
            shapes["fs_"] = (p_, sol.fs["q"])
        for pre, (ne, k_) in shapes.items():
            acc2[pre + "rss"] = np.zeros(ne)
            acc2[pre + "hetero"] = np.zeros((ne, k_, k_)) if "hetero" in plan.vcov_types else None
            acc2[pre + "crv"] = np.zeros((ne, k_, k_)) if "CRV1" in plan.vcov_types else None
        scores: dict = {}
        if "CRV1" in plan.vcov_types and not plan.cluster_local:
            G_tot = plan.cluster[2]
            need = G_tot * sum(ne * k_ for ne, k_ in shapes.values()) * 8
            if need > 8e9:
                log.warning(
                    "fit %s: cluster-score accumulator needs %.1f GB on the coordinator", label, need / 1e9
                )
            scores = {pre: np.zeros((ne, G_tot, k_)) for pre, (ne, k_) in shapes.items()}

        def on2(tid, r):
            peak["rss"] = max(peak["rss"], r["timing"].get("maxrss_gb", 0.0))
            for pre in shapes:
                acc2[pre + "rss"] += r[pre + "rss"]
                if acc2[pre + "hetero"] is not None:
                    acc2[pre + "hetero"] += r[pre + "meat_hetero"]
                if pre + "meat_crv" in r:
                    acc2[pre + "crv"] += r[pre + "meat_crv"]
                if pre + "cluster_scores" in r:
                    for j in range(len(scores[pre])):
                        np.add.at(scores[pre][j], r[pre + "cluster_codes"], r[pre + "cluster_scores"][j])

        tasks2 = [(s["id"], rows[s["id"]], (plan, s["id"], ds.shard_paths(s), sol.lin)) for s in ds.shards]
        prog.start("round2")
        executor.run(shard_round2, tasks2, on2, prog, "round2")
        for pre, sc in scores.items():
            for j in range(len(sc)):
                acc2[pre + "crv"][j] = sc[j].T @ sc[j]
        prog.finish("round2")

        # ---------------- inference ----------------
        k = len(sol.coefnames)
        fe_levels = list(map(int, acc["fe_levels"]))
        nested = list(map(bool, acc["local_nested"])) if plan.cluster else []
        for i in range(len(plan.global_fes)):
            pres = acc["gpresent"][i]
            fe_levels.append(int(pres.sum()))
            if plan.cluster:
                nested.append(bool((acc["glo"][i][pres] == acc["ghi"][i][pres]).all()))
        G = (acc["G_local"] if plan.cluster_local else int(acc["cpresent"].sum())) if plan.cluster else None

        out = {}
        meat_idx = sol.meat_idx
        overid = _iv.overid(sol, acc2["rss"], N) if f.iv else None
        for j, y in enumerate(f.depvars):
            b = sol.coef[:, j]
            bread_use = sol.bread[j]
            V = {}
            dfk_plain = _df_k(k, fe_levels, None)
            if "iid" in plan.vcov_types:
                V["iid"] = (vcov_iid(bread_use, acc2["rss"][j], N, dfk_plain), N - dfk_plain)
            if "hetero" in plan.vcov_types:
                V["hetero"] = (vcov_hetero(bread_use, acc2["hetero"][j], N, dfk_plain), N - dfk_plain)
            if "CRV1" in plan.vcov_types:
                dfk_cl = _df_k(k, fe_levels, nested)
                V["CRV1"] = (vcov_crv1(bread_use, acc2["crv"][j], N, dfk_cl, G), G - 1)
            V = {t: (v[np.ix_(meat_idx, meat_idx)], df) for t, (v, df) in V.items()}
            tss = (
                acc["YtY"][j, j]
                if plan.local_fes
                else acc["sum_wy2"][j] - acc["sum_wy"][j] ** 2 / acc["sumw"]
            )
            out[y] = {
                "coef": b,
                "vcov": V,
                "rss": float(acc2["rss"][j]),
                "tss": float(tss),
                "r2_within" if plan.local_fes else "r2": float(1 - acc2["rss"][j] / tss),
            }
            if f.iv:
                out[y]["kappa"] = float(sol.kappa[j])
                out[y]["overid"] = overid[j]
        iv_info = {}
        if f.iv:
            iv_info = {
                "estimator": "fuller" if fuller is not None else estimator,
                "fuller": fuller,
                "endogenous": list(plan.endog_names),
                "instruments": list(plan.inst_names),
                "first_stage": _iv.first_stage(plan, sol, acc, acc2, N, fe_levels, nested, G),
            }
        coefnames = sol.coefnames
        timing = {
            "round1_seconds": prog.stages["round1"].finished - prog.stages["round1"].started,
            "round2_seconds": prog.stages["round2"].finished - prog.stages["round2"].started,
            "solve_seconds": prog.stages["round2"].started - prog.stages["round1"].finished,
            "singleton_passes": sing["passes"],
            "singleton_seconds": sing["seconds"],
        }

    if cache and not keep_cache:
        shutil.rmtree(wd / "cache", ignore_errors=True)
    if plan.keep_dir:
        shutil.rmtree(plan.keep_dir, ignore_errors=True)

    from . import record as _record

    t_end = time.time()
    options = {
        "cluster": cluster,
        "vcov": list(plan.vcov_types),
        "weights": weights,
        "drop_singletons": bool(drop_singletons),
        "fe_tol": fe_tol,
        "fe_maxiter": plan.fe_maxiter,
        "fe_method": fe_method,
        "crossing_fe": "cells"
        if cells
        else "iterate"
        if plan.crossing_iterate
        else ("sparse" if plan.sparse_global else ("dense" if plan.global_fes else None)),
        "max_dummy_levels": max_dummy_levels,
        "collin_tol": collin_tol,
        "cache": cache,
        **iv_opts,
    }
    if cells:
        options.update(
            cells=True,
            iter_tol=iter_tol,
            iter_maxiter=iter_maxiter,
            level_space_gb=level_space_gb,
            crossing_precond=crossing_precond,
            cells_solve=cres["info"],
        )
    if plan.crossing_iterate:
        options.update(
            iter_tol=iter_tol,
            iter_maxiter=iter_maxiter,
            level_space_gb=level_space_gb,
            crossing_precond=crossing_precond,
            crossing_cg=acc["cg"],
        )
    run = _record.run_info(
        ds,
        executor,
        started=t_start,
        finished=t_end,
        cpu_seconds=(time.process_time() - cpu_start) if isinstance(executor, SerialExecutor) else None,
    )
    run["sample"] = {
        "rows_input": acc["load"]["rows"],
        "dropped": {k: acc["load"][k] for k in ("missing", "non_finite", "zero_weights")},
    }
    src_rows = (ds.meta.get("source") or {}).get("rows")
    if src_rows is not None and src_rows >= ds.n_rows:
        # prep drops rows with a missing partition, FE or cluster key
        run["sample"]["rows_input"] = int(src_rows)
        run["sample"]["dropped"] = {"missing_keys": int(src_rows - ds.n_rows), **run["sample"]["dropped"]}
    run["fit_id"] = str(uuid.uuid4())
    if plan.row_key_cols:
        rk = rowhash.combine(acc["row_key"])
        run["row_key"] = {
            "algorithm": rowhash.ALGORITHM,
            "value": rowhash.hexdigest(rk),
            "rows": rk["rows"],
            "columns": list(plan.row_key_cols),
        }
    run["converged"] = bool(acc["converged"])
    run["fe_nested_in_cluster"] = nested
    run["n_local_fes"] = len(plan.local_fes)
    run["absorption"] = {
        **{
            n: {"method": "cells" if cells else "local", "nested_in_partition": True}
            for n, _ in plan.local_fes
        },
        **{
            n: {
                "method": "cells"
                if cells
                else "crossing_iterate"
                if plan.crossing_iterate
                else ("crossing_sparse" if plan.sparse_global else "crossing_dense"),
                "nested_in_partition": False,
            }
            for n, *_ in plan.global_fes
        },
    }
    run["factor_terms"] = _record.factor_terms(f, plan.cat_levels)
    if weights:
        run["sum_weights"] = float(acc["sumw"])
    res = Result(
        formula=f.text,
        label=label,
        coefnames=coefnames,
        outcomes=out,
        N=N,
        N_dropped=acc["rows_in"] - N,
        G=G,
        cluster=plan.cluster[0] if plan.cluster else None,
        fe=[n for n, _ in plan.local_fes] + [n for n, *_ in plan.global_fes],
        fe_levels=fe_levels,
        collinear=dropped,
        vcov_default=plan.vcov_types[0],
        seconds=time.time() - t_start,
        workdir=str(wd),
        timing={
            **timing,
            "max_worker_rss_gb": peak["rss"],
            "n_shards": len(ds.shards),
            "worker_seconds": sum(sum(st.done.values()) for st in prog.stages.values()),
        },
        options=options,
        terms=_record.term_rows(f, plan.cat_levels, has_fe=bool(plan.local_fes or plan.global_fes)),
        run=run,
        iv=iv_info,
    )
    res.save(wd / "result.json")
    if record:
        rpath = wd / "result.estimation.json" if record is True else Path(record)
        try:
            res.write_estimation(rpath, publication=publication, redact=redact)
            log.info("fit %s: record %s", label, rpath)
        except Exception:  # the fit is done and saved; a record can be rewritten from result.json
            log.exception("fit %s: writing the estimation record failed", label)
    log.info("fit %s: done in %s", label, fmt_td(res.seconds))
    return res
