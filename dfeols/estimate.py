"""Coordinator: plan the fit, run two map/reduce rounds, compute inference."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from .executors import SerialExecutor
from .formula import parse, fe_components
from .linalg import collinear_keep, solve_spd
from .prep import Dataset
from .progress import Progress, fmt_td
from .results import Result
from .vcov import df_k as _df_k, vcov_crv1, vcov_hetero, vcov_iid
from .worker import Plan, plan_columns, shard_round1, shard_round2

log = logging.getLogger("dfeols")


def _norm(name: str) -> str:
    return "^".join(fe_components(name))


def build_plan(ds: Dataset, formula: str, *, cluster=None, vcov=None, weights=None,
               cache_dir=None, max_dummy_levels: int = 2000, drop_singletons=True,
               fe_tol: float = 1e-10, fe_method: str = "auto", crossing_fe: str = "auto") -> Plan:
    f = parse(formula)
    local, glob = [], []
    for name in f.fes:
        key = _norm(name)
        if key not in ds.fes:
            raise KeyError(f"FE {name!r} was not encoded by prep(); available: {list(ds.fes)}")
        m = ds.fes[key]
        if m["nested_in_partition"]:
            local.append((key, m["column"]))
        elif m["n_levels"] <= max_dummy_levels:
            glob.append((key, m["column"], m["n_levels"]))
        else:
            raise NotImplementedError(
                f"FE {name!r} crosses partitions and has {m['n_levels']:,} levels. "
                "v0 handles FEs nested in the partition key or with few levels "
                f"(<= max_dummy_levels={max_dummy_levels}).")
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

    cats = sorted({t.cat for t in f.terms if t.cat})
    cat_levels = {}
    for c in cats:
        lv = ds.scan([c]).drop_nulls().unique().collect(engine="streaming")[c].to_list()
        cat_levels[c] = sorted(lv)
    missing = (f.variables | ({weights} if weights else set())) - set(ds.columns)
    if missing:
        raise KeyError(f"Variables not in the prepared data: {sorted(missing)}")

    colnames = plan_columns(f, cat_levels, has_fe=bool(local or glob))
    spec_id = f.fingerprint(cluster=cluster, weights=weights, singletons=drop_singletons)
    return Plan(formula=f, spec_id=spec_id, local_fes=local, global_fes=glob, cluster=cl,
                cluster_local=cl_local, weights=weights, cat_levels=cat_levels,
                colnames=colnames, n_dummy=sum(L - 1 for *_, L in glob) + (1 if glob and not local else 0), vcov_types=vcov,
                cache_dir=str(cache_dir) if cache_dir else None,
                drop_singletons=drop_singletons, fe_tol=fe_tol, fe_method=fe_method,
                sparse_global=_use_sparse(crossing_fe, local, glob))


def estimate_worker_gb(plan: Plan, rows: int, n_cols: int) -> float:
    """Rough peak worker memory for one shard of `rows` rows.

    Columns held densely: outcomes and reported regressors always; crossing-FE
    dummies only on the dense path (the sparse path keeps per-level sums, a
    groups x levels block, instead). Calibrated on OSCER: 2M rows, 7 dense
    columns -> 2.8-3.4 GB observed (Polars read buffers, numpy temporaries,
    Dask worker overhead ~1 GB)."""
    m = len(plan.formula.depvars)
    k_rep = len(plan.colnames)
    dense = m + k_rep + (0 if plan.sparse_global else plan.n_dummy)
    gb = 1.5                                           # worker process, Dask, libraries
    gb += rows * n_cols * 8 * 1.5 / 1e9                # Parquet read and filtered copy
    gb += rows * dense * 8 * 4 / 1e9                   # design, demeaned copy, products
    if plan.sparse_global:
        levels = sum(L for *_, L in plan.global_fes)
        gb += rows * 0.05 * levels * 8 / 1e9 if levels > 200 else 0   # group x level block
    return gb


def _use_sparse(crossing_fe: str, local, glob) -> bool:
    """Crossing (dummy) FEs via per-level sums: needs exactly one local FE,
    whose within-group projection is closed form."""
    if crossing_fe not in ("auto", "sparse", "dense"):
        raise ValueError("crossing_fe must be auto, sparse or dense")
    ok = len(local) == 1 and len(glob) > 0
    n_dummy = sum(L - 1 for *_, L in glob)
    if not ok and n_dummy > 300 and crossing_fe != "dense":
        log.warning("%d crossing-FE dummy columns will be built densely (the sparse path needs "
                    "exactly one FE nested in the partition); watch worker memory", n_dummy)
    if crossing_fe == "sparse" and not ok:
        raise ValueError("crossing_fe='sparse' needs exactly one FE nested in the partition "
                         "and at least one crossing FE")
    return ok and crossing_fe != "dense"


def fit(ds: Dataset | str, formula: str, *, cluster: str | None = None,
        vcov=None, weights: str | None = None, executor=None,
        workdir: str | None = None, cache: bool | str = "auto", keep_cache: bool = False,
        max_dummy_levels: int = 2000, drop_singletons: bool = True,
        fe_tol: float = 1e-10, fe_method: str = "auto", crossing_fe: str = "auto",
        collin_tol: float = 1e-9,
        label: str | None = None,
        log_every: float = 30.0) -> Result:
    """Estimate an HDFE OLS model on a prepared, sharded dataset.

    cluster    cluster variable (encoded in prep); default vcov is then CRV1
    vcov       "iid", "hetero", "CRV1" or a list of them (all computed in one pass)
    executor   SerialExecutor() (default) or DaskExecutor(client)
    workdir    where status.json and the demeaned cache go; defaults to
               <shard_dir>/runs/<spec id>. Put it on shared scratch.
    """
    if not isinstance(ds, Dataset):
        ds = Dataset.open(ds)
    executor = executor or SerialExecutor()
    t_start = time.time()

    probe = parse(formula)
    if cache == "auto":
        # Caching demeaned arrays between rounds only pays when demeaning is
        # expensive: iterative (3+ local FEs, or 2 with fe_method="iterate"). With one local FE the
        # exact one-pass demean is cheaper than writing and re-reading
        # n x k doubles on shared scratch.
        n_local = sum(1 for f in probe.fes
                      if ds.fes.get("^".join(fe_components(f)), {}).get("nested_in_partition"))
        cache = n_local >= 3 or (n_local == 2 and fe_method == "iterate")
    cache = bool(cache)
    spec_id = probe.fingerprint(cluster=cluster, weights=weights, singletons=drop_singletons)
    wd = Path(workdir) if workdir else ds.root / "runs" / spec_id
    wd.mkdir(parents=True, exist_ok=True)
    plan = build_plan(ds, formula, cluster=cluster, vcov=vcov, weights=weights,
                      cache_dir=(wd / "cache") if cache else None,
                      max_dummy_levels=max_dummy_levels, drop_singletons=drop_singletons,
                      fe_tol=fe_tol, fe_method=fe_method, crossing_fe=crossing_fe)
    if plan.sparse_global and plan.cache_dir:
        plan.cache_dir = None                       # recomputing is cheap on this path
        cache = False
    f = plan.formula
    label = label or spec_id
    m = len(f.depvars)
    k_rep = len(plan.colnames)
    k_tot = k_rep + plan.n_dummy
    log.info("fit %s: %s | %d regressors + %d FE dummies, %d local FEs, %d shards, %s rows",
             label, f.text, k_rep, plan.n_dummy, len(plan.local_fes), len(ds.shards), f"{ds.n_rows:,}")

    biggest = max(s["rows"] for s in ds.shards)
    log.info("fit %s: largest shard %s rows -> peak worker memory roughly %.1f GB; "
             "cache on disk roughly %.1f GB", label, f"{biggest:,}",
             estimate_worker_gb(plan, biggest, len(ds.columns)),
             ds.n_rows * (m + k_tot + 2) * 8 / 1e9 if cache else 0)

    prog = Progress(wd, label=label, every=log_every, round2_recomputes=not cache)
    rows = {s["id"]: max(s["rows"], 1) for s in ds.shards}
    prog.add_stage("round1", rows)
    prog.add_stage("round2", rows)

    # ---------------- round 1 ----------------
    acc = {"ZtZ": np.zeros((k_tot, k_tot)), "ZtY": np.zeros((k_tot, m)),
           "YtY": np.zeros(m), "sum_wy": np.zeros(m), "sum_wy2": np.zeros(m),
           "n": 0, "rows_in": 0, "sumw": 0.0, "converged": True,
           "fe_levels": np.zeros(len(plan.local_fes), np.int64),
           "local_nested": np.ones(len(plan.local_fes), bool),
           "gpresent": [np.zeros(L, bool) for *_, L in plan.global_fes],
           "glo": [np.full(L, np.iinfo(np.int64).max) for *_, L in plan.global_fes],
           "ghi": [np.full(L, np.iinfo(np.int64).min, np.int64) for *_, L in plan.global_fes],
           "cpresent": (np.zeros(plan.cluster[2], bool) if plan.cluster and not plan.cluster_local else None),
           "G_local": 0,
           "row_n": {}}

    peak = {"rss": 0.0}

    def on1(tid, r):
        peak["rss"] = max(peak["rss"], r["timing"].get("maxrss_gb", 0.0))
        for key in ("ZtZ", "ZtY", "YtY", "sum_wy", "sum_wy2"):
            acc[key] += r[key]
        acc["n"] += r["n"]
        acc["rows_in"] += r["rows_in"]
        acc["sumw"] += r["sumw"]
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
    keep = collinear_keep(acc["ZtZ"], collin_tol)
    beta_k, bread = solve_spd(acc["ZtZ"][np.ix_(keep, keep)], acc["ZtY"][keep])
    dropped = [plan.colnames[i] for i in range(k_rep) if not keep[i]]
    if dropped:
        log.info("fit %s: dropped collinear regressors: %s", label, dropped)

    # ---------------- round 2 ----------------
    kept = np.flatnonzero(keep)
    rep_idx = np.flatnonzero(keep[:k_rep])               # reported & kept, in Z order
    pos_in_kept = np.cumsum(keep) - 1
    rep_kept_pos = pos_in_kept[rep_idx]
    fwl = None
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
            pi[lv] = np.linalg.solve(A[np.ix_(dum_kept_pos, dum_kept_pos)], A[np.ix_(dum_kept_pos, rep_kept_pos)])
        fwl = {"rep": rep_idx, "beta_x": beta_k[rep_kept_pos], "gamma": gamma, "pi": pi}
        bread_use = bread[np.ix_(rep_kept_pos, rep_kept_pos)]
        meat_idx = np.arange(len(rep_idx))                 # meat is already reported-only
    else:
        bread_use = bread
        meat_idx = rep_kept_pos
    kk = len(rep_idx) if plan.sparse_global else int(keep.sum())
    acc2 = {"rss": np.zeros(m),
            "hetero": np.zeros((m, kk, kk)) if "hetero" in plan.vcov_types else None,
            "crv": np.zeros((m, kk, kk)) if ("CRV1" in plan.vcov_types) else None}
    scores = None
    if "CRV1" in plan.vcov_types and not plan.cluster_local:
        G_tot = plan.cluster[2]
        need = G_tot * kk * m * 8
        if need > 8e9:
            log.warning("fit %s: cluster-score accumulator needs %.1f GB on the coordinator", label, need / 1e9)
        scores = np.zeros((m, G_tot, kk))

    def on2(tid, r):
        peak["rss"] = max(peak["rss"], r["timing"].get("maxrss_gb", 0.0))
        acc2["rss"] += r["rss"]
        if acc2["hetero"] is not None:
            acc2["hetero"] += r["meat_hetero"]
        if "meat_crv" in r:
            acc2["crv"] += r["meat_crv"]
        if "cluster_scores" in r:
            for j in range(m):
                np.add.at(scores[j], r["cluster_codes"], r["cluster_scores"][j])

    tasks2 = [(s["id"], rows[s["id"]], (plan, s["id"], ds.shard_paths(s), beta_k, keep, fwl)) for s in ds.shards]
    prog.start("round2")
    executor.run(shard_round2, tasks2, on2, prog, "round2")
    if scores is not None:
        for j in range(m):
            acc2["crv"][j] = scores[j].T @ scores[j]
    prog.finish("round2")

    # ---------------- inference ----------------
    k = len(rep_idx)
    fe_levels = list(map(int, acc["fe_levels"]))
    nested = list(map(bool, acc["local_nested"])) if plan.cluster else []
    for i in range(len(plan.global_fes)):
        pres = acc["gpresent"][i]
        fe_levels.append(int(pres.sum()))
        if plan.cluster:
            nested.append(bool((acc["glo"][i][pres] == acc["ghi"][i][pres]).all()))
    G = (acc["G_local"] if plan.cluster_local else int(acc["cpresent"].sum())) if plan.cluster else None

    out = {}
    for j, y in enumerate(f.depvars):
        b = beta_k[rep_kept_pos, j]
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
        tss = acc["YtY"][j] if plan.local_fes else acc["sum_wy2"][j] - acc["sum_wy"][j] ** 2 / acc["sumw"]
        out[y] = {"coef": b, "vcov": V, "rss": float(acc2["rss"][j]),
                  "r2_within" if plan.local_fes else "r2": float(1 - acc2["rss"][j] / tss)}

    if cache and not keep_cache:
        shutil.rmtree(wd / "cache", ignore_errors=True)

    res = Result(
        formula=f.text, label=label, coefnames=[plan.colnames[i] for i in rep_idx],
        outcomes=out, N=N, N_dropped=acc["rows_in"] - N, G=G, cluster=plan.cluster[0] if plan.cluster else None,
        fe=[n for n, _ in plan.local_fes] + [n for n, *_ in plan.global_fes],
        fe_levels=fe_levels, collinear=dropped, vcov_default=plan.vcov_types[0],
        seconds=time.time() - t_start, workdir=str(wd),
        timing={"round1_seconds": prog.stages["round1"].finished - prog.stages["round1"].started,
                "round2_seconds": prog.stages["round2"].finished - prog.stages["round2"].started,
                "solve_seconds": prog.stages["round2"].started - prog.stages["round1"].finished,
                "max_worker_rss_gb": peak["rss"], "n_shards": len(ds.shards),
                "worker_seconds": sum(sum(st.done.values()) for st in prog.stages.values())})
    res.save(wd / "result.json")
    log.info("fit %s: done in %s", label, fmt_td(res.seconds))
    return res
