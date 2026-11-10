# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""`dfeols plan`: probe a specification's FE structure before anything is queued.

Step 1 of docs/design/planner.md: rows and missingness, levels of every FE, cluster and
candidate partition key, nesting of each FE and cluster in each key (and of each FE in
the cluster), partition skew, singletons, and for each key the solver path, worker and
coordinator memory. On plan C keys, the size of A = D~'W D~ against the level-space budget
(level space, or the per-pass fallback, which may converge slowly); and the cell count, for
fit(cells=True). Everything is a Polars streaming aggregate on the source; nothing is sharded
or written. Wall time and node classes are later steps.

The recommended key is the one that makes the crossing system smallest (fewest crossing
levels; on plan C, level space before the fallback) among balanced keys (Ted, 2026-10-29): the
largest group within `target_rows` or a 1/(SHARDS_PER_WORKER x workers) share of the sample, so
no shard is more than half a worker's fair share of the work (Ted, 2026-11-02; was 5%).

A is nested in K when n_unique(A) == n_unique(A, K), so each FE x key pair costs one
distinct count; all of them run as one batch of group-bys over the source.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import timings as _timings
from .crossing import NNZ_BYTES, profile_budget, worker_a_budget, worker_bytes
from .estimate import DENSE_SOLVE_LEVELS, _n_dummy, coordinator_gb, worker_gb
from .formula import Formula, fe_components, parse

log = logging.getLogger("dfeols")

# The solver table (planner.md), in the order a path is tried
PATHS = {
    "local": ("local demeaning in each shard", "exists"),
    "dense": ("dense dummy columns for the crossing FEs", "exists"),
    "sparse": ("sparse path, dense solve on the coordinator", "exists"),
    "B": ("sparse path, several nested FEs projected in the shard (plan B)", "exists"),
    "C": ("conjugate gradient across shards (plan C; 2 passes + one round per step)", "exists"),
    "none": ("varying slopes (fe[x]) with a crossing FE above max_dummy_levels", "not built"),
}
FIT_PATH = {
    "local": "local",
    "dense": "dense",
    "sparse": "sparse",
    "B": "sparse",
    "C": "iterate",
    "none": None,
}  # as fit logs it
LEVEL_SPACE = "plan C in level space: A built in round 1, CG on the coordinator (2 passes)"
FALLBACK = "plan C, a pass over the shards per CG step (A over the budget; may converge slowly)"

SHARDS_PER_WORKER = 2  # balanced: largest group within target_rows or n / (this x workers)
DEFAULT_WORKERS = 20  # the default profile's max_workers, when plan is given no profile
CELLS_RATIO = 2  # recommend fit(cells=True) from this many rows per cell (Ted, 2026-11-02)
EXACT_PAIRS = 200_000_000  # most (group, level) pairs collected to count A's nonzeros exactly
CHUNK_FLOPS = 50_000_000  # products per chunk of that count


def _node_for(profile, need_gb: float) -> dict | None:
    """The smallest node class in service whose memory holds need_gb, or None."""
    from .cluster import _memory_bytes

    best = None
    for name, nc in (getattr(profile, "nodes", None) or {}).items():
        try:
            gb = _memory_bytes(str(nc.get("memory"))) / 1e9
        except Exception:
            continue
        if (
            nc.get("in_service", nc.get("count", 1))
            and gb >= need_gb
            and (best is None or gb < best["memory_gb"])
        ):
            best = {"name": name, "memory_gb": gb, "partition": nc.get("partition")}
    return best


def _key(name: str) -> tuple[str, ...]:
    return tuple(fe_components(name))


def _formula(formula: str | None, fe, vars) -> Formula:
    if formula:
        return parse(formula)
    if not vars:
        raise ValueError("give a formula, or fixed effects and variables (outcome first)")
    vs = [vars] if isinstance(vars, str) else list(vars)
    fes = [fe] if isinstance(fe, str) else list(fe or [])
    rhs = " + ".join(vs[1:]) or "1"
    return parse(f"{vs[0]} ~ {rhs}" + (f" | {' + '.join(fes)}" if fes else ""))


def _candidates(f: Formula, cluster: str | None, partition) -> list[str]:
    """Keys the user lists, else every FE, every interaction component and the cluster."""
    if partition:
        keys = [partition] if isinstance(partition, str) else list(partition)
    else:
        keys = []
        for name in [*(t.base for t in f.fe_terms), *([cluster] if cluster else [])]:
            keys.append(name)
            comps = fe_components(name)
            if len(comps) > 1:
                keys.extend(comps)
    out: list[str] = []
    for k in keys:
        k = "^".join(_key(k))
        if k not in out:
            out.append(k)
    return out


def _group_stats(lf: pl.LazyFrame, cols: tuple[str, ...], rows: str | None = None) -> pl.LazyFrame:
    """Levels, largest group, singletons and the 5 largest groups of `cols`; with `rows`, lf
    holds cells and that column their row counts."""
    return (
        lf.group_by(list(cols))
        .agg((pl.len() if rows is None else pl.col(rows).sum()).alias("_n"))
        .select(
            pl.len().alias("levels"),
            pl.col("_n").max().alias("largest"),
            (pl.col("_n") == 1).sum().alias("singletons"),
            pl.col("_n").top_k(5).sort(descending=True).alias("top"),
        )
    )


def _a_size(sample: pl.LazyFrame, gcols, cross: dict, L: dict, budget: int, count: bool = True) -> dict:
    """Nonzeros of A = D~'W D~ for one key's crossing FEs `cross` (name -> columns, L levels).

    With no local FE (gcols empty) A is D'WD: diagonal blocks, and off them the distinct
    (a, b) level pairs in the data, so the count is exact. Otherwise every entry of A lies
    within one local group g (the single local FE's groups, else the key's), so A_ab has at
    most min(sum_g t_ga t_gb, L_a L_b) nonzeros, t_ga the levels of a that g touches. The
    bound overstates where many groups touch the same pairs (110x on DB1B), so above the
    budget A is counted exactly: the pattern of B'B for the group x level incidence B, in
    chunks of levels, stopping once past the budget (as fit's LevelGram), if `count`."""
    names = list(cross)
    out = {"levels": int(sum(L.values())), "exact": False, "nnz": None, "over": False}
    if not gcols:
        nnz = out["levels"]
        blocks = {}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                cols = sorted(set(cross[a]) | set(cross[b]))
                n_ab = int(sample.select(cols).unique().select(pl.len()).collect(engine="streaming").item())
                blocks[(a, b)] = 2 * n_ab
                nnz += 2 * n_ab
        out.update(bound=nnz, nnz=nnz, exact=True, over=nnz > budget)
        out["blocks"] = blocks
        return out
    g = list(gcols)
    # each crossing FE's distinct (group, level) pairs, once: t below, and B for the count
    pairs, ts, kept = [], [], 0
    for i, a in enumerate(names):
        pr = sample.select(list(dict.fromkeys(g + list(cross[a])))).unique().collect(engine="streaming")
        ts.append(pr.group_by(g).agg(pl.len().cast(pl.Int64).alias(f"t{i}")))
        kept += pr.height
        pairs.append(pr if count and kept <= EXACT_PAIRS else None)
    t = ts[0]
    for x in ts[1:]:
        t = t.join(x, on=g, how="full", coalesce=True)
    T = t.select([f"t{i}" for i in range(len(names))]).fill_null(0).to_numpy().astype(np.float64)
    blocks, bound = {}, 0
    for i, a in enumerate(names):
        for j in range(i, len(names)):
            b = names[j]
            nb = min(float(T[:, i] @ T[:, j]), float(L[a]) * L[b]) * (1 if i == j else 2)
            blocks[(a,) if i == j else (a, b)] = int(nb)
            bound += int(nb)
    out.update(bound=bound, blocks=blocks, pairs=int(T.sum()))
    if bound <= budget or out["pairs"] > EXACT_PAIRS or not count:
        return out
    # exact count: codes for the groups and levels, then B'B by chunks of levels
    gidx = t.select(g).with_row_index("_g")
    rows, cols_, off = [], [], 0
    for a, pr in zip(names, pairs):
        ac = list(cross[a])
        lv = pr.select(ac).unique().with_row_index("_l")
        pr = pr.join(gidx, on=g).join(lv, on=ac)
        rows.append(pr["_g"].to_numpy().astype(np.int64))
        cols_.append(pr["_l"].to_numpy().astype(np.int64) + off)
        off += lv.height
    del pairs
    r, c = np.concatenate(rows), np.concatenate(cols_)
    B = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(gidx), off))
    # A is symmetric: count its upper triangle by chunks of rows of B'B, busiest levels first
    # so a key over the budget reaches it soonest; nnz = 2 x (strictly upper) + diagonal
    # (every level is in A). Row slices of the CSR B' are cheap (column slices copy the rest)
    deg = np.diff(B.indptr)
    work = B.T @ deg.astype(np.float64)  # products per level row
    Bt = B.T.tocsr()
    # lower bound: row l of A holds at least the levels of the largest group touching l
    lower = int(np.maximum.reduceat(deg[Bt.indices], Bt.indptr[:-1]).sum()) if off else 0
    out["lower"] = lower
    if lower > budget:
        out.update(over=True, nnz=lower)
        return out
    order = np.argsort(-work, kind="stable")
    B = B[:, order].tocsr()
    Bt = B.T.tocsr()
    cw = np.cumsum(work[order])
    steps = np.arange(1, int(cw[-1] // CHUNK_FLOPS) + 1) * CHUNK_FLOPS
    cuts = np.unique(np.r_[0, np.searchsorted(cw, steps), off])
    nnz = off
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        blk = (Bt[c0:c1] @ B).tocoo()  # rows c0:c1 of A, all columns
        nnz += 2 * int(np.count_nonzero(blk.col > blk.row + c0))
        if nnz > budget:
            out.update(over=True, nnz=nnz)
            return out
    out.update(nnz=int(nnz), exact=True)
    return out


def _blocks_text(blocks: dict) -> str:
    top = sorted(blocks.items(), key=lambda x: -x[1])[:3]
    return ", ".join(f"{' x '.join(k)} {n:,}" for k, n in top if n)


def plan(
    source,
    formula: str | None = None,
    *,
    fe=None,
    vars=None,
    cluster: str | None = None,
    weights: str | None = None,
    partition=None,
    target_rows: int = 1_000_000,
    max_dummy_levels: int = 2000,
    level_space_gb: float | None = None,
    profile=None,
    workers: int | None = None,
    specs: int = 1,
) -> dict:
    """Probe the FE structure of a specification and recommend a partition key and path.

    source      Parquet path or glob, CSV path, or a Polars/pandas frame (raw or prepped)
    formula     dfeols formula; or `fe` (FE specs) and `vars` (outcome first, then regressors)
    cluster     cluster variable (or interaction a^b)
    partition   candidate partition keys (a^b for a compound key); default: every FE,
                interaction component and the cluster
    target_rows prep's shard target
    level_space_gb  fit's budget for A on plan C (default: a quarter of the profile's
                coordinator memory, else of this machine's)
    profile     site profile (name or Profile) to check worker and coordinator memory against
    workers     workers the fit will run on, for shard balance and the wall-time estimate
                (default: the profile's max_workers, else DEFAULT_WORKERS)
    specs       how many specifications will share these FEs, sample and weights (more
                than one: demean-first, a session)

    Returns a JSON-serialisable dict; `format_report` prints it.
    """
    from .prep import _scan

    t0 = time.time()
    f = _formula(formula, fe, vars)
    fes = [t.base for t in f.fe_terms]
    cl = "^".join(_key(cluster)) if cluster else None
    keys = _candidates(f, cl, partition)
    if profile is not None and not hasattr(profile, "worker"):
        from .site import profile as load_profile

        profile = load_profile(profile)
    runs = _timings.history(profile)  # earlier fits on this site, for the wall-time estimate
    n_specs = int(specs)

    specs = {name: _key(name) for name in [*fes, *([cl] if cl else []), *keys]}
    cats = sorted({t.cat for t in (*f.terms, *f.endog, *f.instruments) if t.cat})
    needed = set(f.variables) | ({weights} if weights else set())
    for comps in specs.values():
        needed.update(comps)

    lf = _scan(source)
    have = set(lf.collect_schema().names())
    missing = needed - have
    if missing:
        raise KeyError(f"Columns not in the source: {sorted(missing)}")
    cols = sorted(needed)
    raw = lf.select(cols)
    # listwise deletion as the workers do it: nulls, non-finite floats, weights <= 0
    schema = lf.collect_schema()
    num = [c for c in f.variables if schema[c].is_float()]
    sample = raw.drop_nulls()
    if num:
        sample = sample.filter(pl.all_horizontal([pl.col(c).is_finite() for c in num]))
    if weights:
        sample = sample.filter(pl.col(weights) > 0)

    # every distinct column set: specs, categorical regressors, and the unions that nesting needs
    sets: dict[frozenset, tuple[str, ...]] = {}

    def want(c: tuple[str, ...]):
        sets.setdefault(frozenset(c), tuple(sorted(set(c))))  # a key may share an FE's columns

    for comps in specs.values():
        want(comps)
    for c in cats:
        want((c,))
    inner = [*fes, *([cl] if cl else [])]
    for a in inner:
        for k in keys:
            want(specs[a] + specs[k])
    if cl:
        for a in fes:
            want(specs[a] + specs[cl])

    # cells: every FE and the cluster constant within one
    cell_cols = tuple(sorted({c for name in inner for c in specs[name]}))
    if cell_cols:
        want(cell_cols)

    order = list(sets)
    # every column set inside the cell key is answered from the cells (rows per cell): one
    # scan, not one per set (DB1B, 683M rows: 16 scans of ~5 s; 21M cells in 7.5 s), if the
    # cells are few enough to hold (HyperLogLog estimate first)
    cells_df = None
    inside = [s for s in order if s <= set(cell_cols)]
    if len(inside) > 1:
        est = (
            sample.select(pl.struct(list(cell_cols)).hash().approx_n_unique())
            .collect(engine="streaming")
            .item()
        )
        if est <= EXACT_PAIRS:
            cells_df = sample.group_by(list(cell_cols)).agg(pl.len().alias("_w")).collect(engine="streaming")
    from_cells = set(inside) if cells_df is not None else set()
    frames = [
        raw.select(
            pl.len().alias("rows"),
            *[
                (pl.col(c).is_null() | ~pl.col(c).is_finite()).sum().alias(c)
                if c in num
                else pl.col(c).null_count().alias(c)
                for c in cols
            ],
        ),
        sample.select(pl.len().alias("rows"))
        if cells_df is None
        else cells_df.lazy().select(pl.col("_w").sum().alias("rows")),
        *[
            _group_stats(cells_df.lazy(), sets[s], "_w") if s in from_cells else _group_stats(sample, sets[s])
            for s in order
        ],
    ]
    # one query at a time: run together, every group-by's hash table is live at once
    # (DB1B market table, 683M rows: 31 GB peak RSS and 19 s together, 7.5 GB and 32 s apart)
    res = [lf.collect(engine="streaming") for lf in frames]
    nulls = res[0].row(0, named=True)
    n_raw = nulls.pop("rows")
    n = int(res[1].item())
    stats = {s: res[i + 2].row(0, named=True) for i, s in enumerate(order)}

    def st(comps) -> dict:
        return stats[frozenset(comps)]

    def nested(a: str, k: str) -> bool:
        return st(specs[a])["levels"] == st(specs[a] + specs[k])["levels"]

    levels = {name: st(specs[name]) for name in specs}
    cat_levels = {c: st((c,))["levels"] for c in cats}

    # regressor-block columns, as plan_columns and term_names count them
    def term_cols(terms) -> int:
        k = 0
        for t in terms:
            if t.cat is None:
                k += 1
            else:
                k += cat_levels[t.cat] - (1 if (t.ref is not None or t.drop_first) else 0)
        return k

    const = f.intercept and not any(t.intercept for t in f.fe_terms)  # reported (slope-only FEs too)
    k_x = term_cols(f.terms) + term_cols(f.instruments) + int(const)
    m = len(f.depvars) + term_cols(f.endog)
    n_cols = len(set(f.variables) | ({weights} if weights else set())) + len(specs)

    worker_mem = coord_mem = None
    if profile is not None:
        from .cluster import _memory_bytes

        worker_mem = _memory_bytes(profile.worker["memory"]) / 1e9
        coord_mem = _memory_bytes(profile.coordinator["memory"]) / 1e9

    # A is summed on the workers before the coordinator gets it: within both memories
    budget = profile_budget(profile, level_space_gb)
    if budget:
        budget = min(budget, worker_a_budget(worker_bytes(prof=profile)))
    if workers is None:
        workers = (profile.worker.get("max_workers") if profile is not None else None) or DEFAULT_WORKERS
    max_group = max(target_rows, n / (SHARDS_PER_WORKER * workers))
    # A's sizes need each key's distinct (group, level) pairs: take them from the distinct
    # cells (every FE and cluster column), collected once, not from a scan of all rows per
    # key and crossing FE (DB1B, 683M rows: 21M cells; 12 scans were 3 column sets)
    cell_frame, a_sizes = {}, {}

    def pair_source(cols) -> pl.LazyFrame:
        if not (set(cols) <= set(cell_cols) and st(cell_cols)["levels"] <= EXACT_PAIRS):
            return sample
        if cells_df is not None:
            return cells_df.lazy()
        if "f" not in cell_frame:
            cell_frame["f"] = sample.select(list(cell_cols)).unique().collect(engine="streaming")
        return cell_frame["f"].lazy()

    slopes = {t.base: (t.slopes, t.intercept) for t in f.fe_terms if t.slopes}
    options = []
    for k in keys:
        local = [a for a in fes if nested(a, k)]
        glob = [a for a in fes if a not in local]
        g_levels = {a: levels[a]["levels"] for a in glob}
        n_dummy = _n_dummy([(a, None) for a in local], [(a, None, L) for a, L in g_levels.items()], slopes)
        big = [a for a, L in g_levels.items() if L > max_dummy_levels]
        # as estimate._use_iterate and _use_sparse with crossing_fe="auto"; with varying slopes
        # crossing FEs are dense dummies only (estimate.build_plan)
        if not glob:
            path = "local"
        elif slopes:
            path = "none" if big else "dense"
        elif n_dummy > DENSE_SOLVE_LEVELS or (big and not local):
            path = "C"
        elif len(local) == 1:
            path = "sparse"
        elif not big:
            path = "dense"
        else:
            path = "B"
        kst = levels[k]
        n_bins = max(1, math.ceil(n / target_rows))
        shard_rows = max(kst["largest"] or 0, min(target_rows, n))
        balanced = not kst["largest"] or kst["largest"] <= max_group
        a_size, solve = None, None
        if path == "C":
            # as estimate_worker_gb / estimate_coordinator_gb: no dummy columns or dense
            # solve; shards hold a CG block, the coordinator level x column blocks
            w_gb = worker_gb(shard_rows, n_cols, m, k_x, 0) + shard_rows * (m + k_x) * 8 * 2 / 1e9
            c_gb = coordinator_gb(k_x) + sum(g_levels.values()) * (m + k_x) * 8 * 8 / 1e9
            gcols = specs[local[0]] if len(local) == 1 else (specs[k] if local else ())
            # counting A takes a while near the budget: not for keys the rule passes over
            memo = (tuple(gcols), tuple(sorted(glob)), balanced)  # keys with the same groups
            if memo not in a_sizes:  # and crossing FEs (DB1B: route, carrier^route) share A
                src = pair_source({*gcols, *(c for a in glob for c in specs[a])})
                a_sizes[memo] = _a_size(
                    src, gcols, {a: specs[a] for a in glob}, g_levels, budget, count=balanced
                )
            a_size = a_sizes[memo]
            if a_size["bound"] <= budget or (a_size["exact"] and not a_size["over"]):
                solve = "level"
                c_gb += min(a_size["nnz"] or a_size["bound"], budget) * NNZ_BYTES / 1e9
            else:
                solve = "fallback" if a_size["over"] else "maybe"
        else:
            sparse_levels = sum(g_levels.values()) if path in ("sparse", "B") else 0
            w_gb = worker_gb(shard_rows, n_cols, m, k_x, n_dummy, sparse_levels)
            c_gb = coordinator_gb(k_x + n_dummy)
        warnings = []
        if kst["largest"] and kst["largest"] > target_rows:
            warnings.append(
                f"largest group {kst['largest']:,} rows exceeds target_rows {target_rows:,} "
                f"({kst['largest'] / max(n, 1):.0%} of the sample)"
            )
        if not balanced:
            warnings.append(
                f"unbalanced: largest group over {max_group / max(n, 1):.1%} of the sample "
                f"(1/{SHARDS_PER_WORKER} of a worker's share with {workers} workers)"
            )
        if path == "dense" and n_dummy > 300:
            warnings.append(f"{n_dummy:,} dummy columns built densely in every shard")
        if solve in ("fallback", "maybe"):
            lv = ", ".join(f"{a} {L:,}" for a, L in g_levels.items())
            size = (
                f"A has more than {a_size['nnz']:,} nonzeros"
                if solve == "fallback"
                else f"A may have up to {a_size['bound']:,} nonzeros (the bound; not counted)"
            )
            warnings.append(
                f"{size} over {a_size['levels']:,} crossing levels ({lv}), over the level-space "
                f"budget of {budget:,} (~{budget * NNZ_BYTES / 1e9:.0f} GB); largest blocks by the bound: "
                f"{_blocks_text(a_size['blocks'])}. "
                + ("fit falls back to" if solve == "fallback" else "If so, fit falls back to")
                + " a pass over the shards per CG step, which may converge slowly"
            )
        # wall time from earlier runs on this site (timings.py), and the job's shape
        fit_path = FIT_PATH[path]
        est = _timings.estimate_fit(
            n,
            m + k_x + (n_dummy if path == "dense" else 0),
            fit_path,
            solve=solve if path == "C" else None,
            crossing=bool(g_levels),
            workers=workers,
            profile=profile,
            runs=runs,
        )
        how = _timings.shape(
            {"path": fit_path, "solve": solve},
            est,
            specs=n_specs,
            max_walltime=_timings.walltime_seconds(getattr(profile, "max_walltime", None)),
        )
        if how["shape"] == "demean-first":
            how["demean"] = _timings.estimate_demean(n, m + k_x, profile)
        fits = True
        if worker_mem is not None and w_gb > worker_mem:
            fits = False
            warnings.append(f"workers need ~{w_gb:.0f} GB; profile gives {worker_mem:.0f} GB")
        if coord_mem is not None and c_gb > coord_mem:
            fits = False
            warnings.append(f"coordinator needs ~{c_gb:.0f} GB; profile gives {coord_mem:.0f} GB")
        options.append(
            {
                "key": k,
                "levels": kst["levels"],
                "largest_group": kst["largest"],
                "top_groups": kst["top"],
                "shards": min(kst["levels"], n_bins),
                "largest_shard_rows": shard_rows,
                "local_fes": local,
                "crossing_fes": g_levels,
                "n_dummy": n_dummy,
                "cluster_nested": nested(cl, k) if cl else None,
                "path": path,
                "path_text": (
                    LEVEL_SPACE if solve == "level" else FALLBACK if solve == "fallback" else PATHS[path][0]
                ),
                "crossing_levels": sum(g_levels.values()),
                "solve": solve,
                "A": (
                    {
                        "bound": a_size["bound"],
                        "nnz": a_size["nnz"],
                        "exact": a_size["exact"],
                        "lower": a_size.get("lower"),
                        "budget": budget,
                    }
                    if a_size
                    else None
                ),
                "balanced": balanced,
                "status": PATHS[path][1],
                "worker_gb": round(w_gb, 1),
                "coordinator_gb": round(c_gb, 1),
                "fits": fits,
                "warnings": warnings,
                "estimate": est,
                "shape": how,
            }
        )

    # the smallest crossing system among balanced keys that fit (Ted, 2026-10-29)
    ranked = sorted(
        options,
        key=lambda o: (
            o["status"] != "exists",
            not o["fits"],
            not o["balanced"],
            {None: 0, "level": 0, "maybe": 1, "fallback": 2}[o["solve"]],
            o["crossing_levels"],
            (o["A"]["nnz"] or o["A"]["bound"]) if o["A"] else 0,
            o["worker_gb"],
            o["coordinator_gb"],
        ),
    )
    rec = ranked[0] if ranked and ranked[0]["status"] == "exists" else None

    cells = None
    if cell_cols:
        n_cells = st(cell_cols)["levels"]
        p = k_x + m
        floats = 2 + p + p * (p + 1) // 2 + len(cell_cols)
        ok = bool(fes) and not slopes  # OLS and IV; not varying slopes
        # the shards' cells, their concatenation, the merged frame and the solve's copies:
        # 6 copies (OSCER job 35625881: 21.2M cells, 20 floats each -> 19.4 GB peak)
        c_raw = n_cells * floats * 8 * 6 / 1e9
        c_need = round(c_raw, 1)
        fits = None if coord_mem is None else c_raw <= coord_mem
        cells = {
            "key": list(cell_cols),
            "cells": n_cells,
            "rows_per_cell": round(n / max(n_cells, 1), 1),
            "coordinator_gb": c_need,
            "coordinator_memory_gb": coord_mem,
            "fits": fits,
            # a node class whose memory holds the cells, when the coordinator's does not
            "node_class": _node_for(profile, c_raw) if fits is False else None,
            "eligible": ok,
            "compresses": n >= CELLS_RATIO * n_cells,
            "recommended": ok and n >= CELLS_RATIO * n_cells and fits is not False,
        }
    report = {
        "formula": f.text,
        "rows": n_raw,
        "sample_rows": n,
        "nulls": {c: v for c, v in nulls.items() if v},
        "fes": {
            a: {
                "levels": levels[a]["levels"],
                "singletons": levels[a]["singletons"],
                "nested_in": {k: nested(a, k) for k in keys},
                **({"nested_in_cluster": nested(a, cl)} if cl else {}),
            }
            for a in fes
        },
        "cluster": {"name": cl, "levels": levels[cl]["levels"]} if cl else None,
        "k_x": k_x,
        "target_rows": target_rows,
        "workers": workers,
        "max_balanced_group": int(max_group),
        "max_dummy_levels": max_dummy_levels,
        "profile": getattr(profile, "name", None),
        "options": ranked,
        "recommended": rec["key"] if rec else None,
        "cells": cells,
        "specs": n_specs,
        "history_runs": len(runs),
        "seconds": round(time.time() - t0, 1),
    }
    log.info("plan: probed %s rows, %d keys in %.0fs", f"{n:,}", len(keys), report["seconds"])
    return report


def _td(seconds: float) -> str:
    from .progress import fmt_td

    return fmt_td(seconds) if seconds >= 1 else f"{seconds:.1f} s"


def format_report(r: dict) -> str:
    """The text `dfeols plan` prints."""
    keys = [o["key"] for o in r["options"]]
    out = [f"  {r['formula']}", f"  {r['rows']:,} rows; {r['sample_rows']:,} with no missing values"]
    if r["nulls"]:
        out.append("  missing: " + ", ".join(f"{c} {v:,}" for c, v in sorted(r["nulls"].items())))
    w = max([len(a) for a in [*r["fes"], r["cluster"]["name"] if r["cluster"] else ""]] + [6])
    head = f"  {'FE':<{w}} {'levels':>12} {'singletons':>11}   nested in: " + "  ".join(keys)
    if r["cluster"]:
        head += "  cluster"
    out.append(head)
    for a, d in r["fes"].items():
        line = f"  {a:<{w}} {d['levels']:>12,} {d['singletons']:>11,}   {'':11}"
        line += "  ".join(f"{'yes' if d['nested_in'][k] else 'no':<{len(k)}}" for k in keys)
        if "nested_in_cluster" in d:
            line += "  " + ("yes" if d["nested_in_cluster"] else "no")
        out.append(line)
    if r["cluster"]:
        c = r["cluster"]
        out.append(f"  {c['name']:<{w}} {c['levels']:>12,} {'(cluster)':>11}")
    out.append("")
    if "workers" in r:
        out.append(
            f"  balanced: largest group within {r['max_balanced_group']:,} rows "
            f"(target_rows, or half a worker's share with {r['workers']} workers)"
        )
    out.append(f"  {'partition':<{w}} {'largest group':>14} {'shards':>7}  path")
    for o in r["options"]:
        cross = ", ".join(f"{a} {L:,}" for a, L in o["crossing_fes"].items()) or "none"
        out.append(
            f"  {o['key']:<{w}} {o['largest_group']:>14,} {o['shards']:>7,}  {o['path_text']}"
            + ("" if o["status"] == "exists" else " [not built]")
        )
        out.append(
            f"  {'':<{w}} crossing: {cross}; workers ~{o['worker_gb']:.1f} GB, "
            f"coordinator ~{o['coordinator_gb']:.1f} GB" + ("" if o["fits"] else " (does not fit)")
        )
        if o["A"]:
            a = o["A"]
            size = f"{a['nnz']:,} nonzeros" if a["exact"] else f"at most {a['bound']:,} nonzeros (bound)"
            if a["nnz"] and not a["exact"]:
                how = "lower bound" if a.get("lower") == a["nnz"] else "counted to the budget"
                size = f"at least {a['nnz']:,} nonzeros ({how})"
            out.append(f"  {'':<{w}} A: {size}; budget {a['budget']:,}")
        e = o.get("estimate")
        if e:
            out.append(
                f"  {'':<{w}} wall time ~{_td(e['low'])} to {_td(e['high'])} ({e['basis']})"
                + (f"; {e['unknown']}" if e.get("unknown") else "")
            )
        for msg in o["warnings"]:
            out.append(f"  {'':<{w}} ! {msg}")
    out.append("")
    if r["recommended"]:
        o = r["options"][0]
        out.append(
            f"  -> partition by {o['key']}: {o['path_text']}; ~{o['shards']:,} shards of up to "
            f"{o['largest_shard_rows']:,} rows"
        )
        sh = o.get("shape") or {}
        if sh.get("shape") == "demean-first":
            d = sh.get("demean")
            out.append(
                "  -> demean-first: "
                + "; ".join(sh["why"])
                + (
                    f". Demeaning estimated {_td(d[0])} to {_td(d[1])} from earlier session builds"
                    if d
                    else ""
                )
                + ". Run it as a session (dfeols session start <name> --data <shards> --fe ... "
                "--vars <every variable the specifications use>), then fit each specification there"
            )
        elif sh:
            out.append("  -> one job: dfeols submit running fit (fixed passes, within the walltime)")
    else:
        out.append("  -> no candidate key has a path dfeols runs today; see the [not built] paths above")
    c = r.get("cells")
    if c:
        line = f"  cells ({', '.join(c['key'])}): {c['cells']:,}, {c['rows_per_cell']:,} rows per cell"
        if c["recommended"]:
            line += (
                f" -> fit(cells=True) is exact here (iid, CRV1) from one pass; "
                f"coordinator ~{c['coordinator_gb']:.1f} GB"
                + (f" of {c['coordinator_memory_gb']:.0f}" if c.get("coordinator_memory_gb") else "")
            )
        elif not c["eligible"]:
            line += " (the cell path needs FEs, and takes no varying slopes)"
        elif c.get("compresses") and c.get("fits") is False:
            have = c["coordinator_memory_gb"]
            line += (
                f" -> fit(cells=True) would need ~{c['coordinator_gb']:,.0f} GB on the coordinator; "
                f"the profile gives it {have:,.0f} GB"
            )
            nc = c.get("node_class")
            line += (
                f" (node class {nc['name']} has {nc['memory_gb']:,.0f} GB: set the coordinator's "
                "memory to run it there)"
                if nc
                else " and no node class has that much: use the row path"
            )
        out.append(line)
    runs = r.get("history_runs", 0)
    wall = (
        f"wall time from {runs} earlier fit{'s' if runs != 1 else ''} on this site"
        if runs
        else "wall time not estimated: no earlier fits recorded on this site (fit records each run)"
    )
    out.append(f"  (probe took {r['seconds']:.0f} s; memory is rough; {wall})")
    return "\n".join(out)
