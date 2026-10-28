# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""`dfeols plan`: probe a specification's FE structure before anything is queued.

Step 1 of docs/design/planner.md: rows and missingness, levels of every FE, cluster and
candidate partition key, nesting of each FE and cluster in each key (and of each FE in
the cluster), partition skew, singletons, and for each key the solver path, worker and
coordinator memory. Everything is a Polars streaming aggregate on the source; nothing is
sharded or written. Wall time, crossing structure (fill, components) and node classes
are later steps.

A is nested in K when n_unique(A) == n_unique(A, K), so each FE x key pair costs one
distinct count; all of them run as one batch of group-bys over the source.
"""

from __future__ import annotations

import logging
import math
import time

import polars as pl

from .estimate import coordinator_gb, worker_gb
from .formula import Formula, fe_components, parse

log = logging.getLogger("dfeols")

# Plan A's coordinator dense solve: about 43 GB at 30k crossing levels (coordinator_gb)
DENSE_SOLVE_LEVELS = 30_000

# The solver table (planner.md), in the order a path is tried
PATHS = {
    "local": ("local demeaning in each shard", "exists"),
    "dense": ("dense dummy columns for the crossing FEs", "exists"),
    "sparse": ("sparse path, dense solve on the coordinator", "exists"),
    "B": ("sparse path, several nested FEs projected in the shard (plan B)", "to build"),
    "C": ("sparse direct solve or alternating projections across shards (plan C)", "to build"),
}


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
        for name in [*f.fes, *([cluster] if cluster else [])]:
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


def _group_stats(lf: pl.LazyFrame, cols: tuple[str, ...]) -> pl.LazyFrame:
    return (
        lf.group_by(list(cols))
        .agg(pl.len().alias("_n"))
        .select(
            pl.len().alias("levels"),
            pl.col("_n").max().alias("largest"),
            (pl.col("_n") == 1).sum().alias("singletons"),
            pl.col("_n").top_k(5).alias("top"),
        )
    )


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
    profile=None,
) -> dict:
    """Probe the FE structure of a specification and recommend a partition key and path.

    source      Parquet path or glob, CSV path, or a Polars/pandas frame (raw or prepped)
    formula     dfeols formula; or `fe` (FE specs) and `vars` (outcome first, then regressors)
    cluster     cluster variable (or interaction a^b)
    partition   candidate partition keys (a^b for a compound key); default: every FE,
                interaction component and the cluster
    target_rows prep's shard target
    profile     site profile (name or Profile) to check worker and coordinator memory against

    Returns a JSON-serialisable dict; `format_report` prints it.
    """
    from .prep import _scan

    t0 = time.time()
    f = _formula(formula, fe, vars)
    fes = ["^".join(_key(x)) for x in f.fes]
    cl = "^".join(_key(cluster)) if cluster else None
    keys = _candidates(f, cl, partition)
    if profile is not None and not hasattr(profile, "worker"):
        from .site import profile as load_profile

        profile = load_profile(profile)

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
    sample = raw.drop_nulls()

    # every distinct column set: specs, categorical regressors, and the unions that nesting needs
    sets: dict[frozenset, tuple[str, ...]] = {}

    def want(c: tuple[str, ...]):
        sets.setdefault(frozenset(c), tuple(sorted(c)))

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

    order = list(sets)
    frames = [
        raw.select(pl.len().alias("rows"), *[pl.col(c).null_count().alias(c) for c in cols]),
        sample.select(pl.len().alias("rows")),
        *[_group_stats(sample, sets[s]) for s in order],
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

    k_x = term_cols(f.terms) + term_cols(f.instruments) + (1 if f.intercept and not fes else 0)
    m = len(f.depvars) + term_cols(f.endog)
    n_cols = len(set(f.variables) | ({weights} if weights else set())) + len(specs)

    worker_mem = coord_mem = None
    if profile is not None:
        from .cluster import _memory_bytes

        worker_mem = _memory_bytes(profile.worker["memory"]) / 1e9
        coord_mem = _memory_bytes(profile.coordinator["memory"]) / 1e9

    options = []
    for k in keys:
        local = [a for a in fes if nested(a, k)]
        glob = [a for a in fes if a not in local]
        g_levels = {a: levels[a]["levels"] for a in glob}
        n_dummy = sum(L - 1 for L in g_levels.values()) + (1 if glob and not local else 0)
        big = [a for a, L in g_levels.items() if L > max_dummy_levels]
        if not glob:
            path = "local"
        elif len(local) == 1:
            path = "sparse" if n_dummy <= DENSE_SOLVE_LEVELS else "C"
        elif not big:
            path = "dense"
        elif len(local) > 1 and n_dummy <= DENSE_SOLVE_LEVELS:
            path = "B"
        else:
            path = "C"
        kst = levels[k]
        n_bins = max(1, math.ceil(n / target_rows))
        shard_rows = max(kst["largest"] or 0, min(target_rows, n))
        if path == "C":
            # no dummy columns or dense solve: shards hold the data, the coordinator per-level
            # sums of every column (a reduction's worth, twice)
            w_gb = worker_gb(shard_rows, n_cols, m, k_x, 0)
            c_gb = coordinator_gb(k_x) + sum(g_levels.values()) * (m + k_x) * 8 * 2 / 1e9
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
        if path == "dense" and n_dummy > 300:
            warnings.append(f"{n_dummy:,} dummy columns built densely in every shard")
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
                "path_text": PATHS[path][0],
                "status": PATHS[path][1],
                "worker_gb": round(w_gb, 1),
                "coordinator_gb": round(c_gb, 1),
                "fits": fits,
                "warnings": warnings,
            }
        )

    ranked = sorted(
        options,
        key=lambda o: (o["status"] != "exists", not o["fits"], o["worker_gb"], o["coordinator_gb"]),
    )
    rec = ranked[0] if ranked and ranked[0]["status"] == "exists" else None
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
        "max_dummy_levels": max_dummy_levels,
        "profile": getattr(profile, "name", None),
        "options": ranked,
        "recommended": rec["key"] if rec else None,
        "seconds": round(time.time() - t0, 1),
    }
    log.info("plan: probed %s rows, %d keys in %.0fs", f"{n:,}", len(keys), report["seconds"])
    return report


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
        for msg in o["warnings"]:
            out.append(f"  {'':<{w}} ! {msg}")
    out.append("")
    if r["recommended"]:
        o = r["options"][0]
        out.append(
            f"  -> partition by {o['key']}: {o['path_text']}; ~{o['shards']:,} shards of up to "
            f"{o['largest_shard_rows']:,} rows"
        )
    else:
        out.append("  -> no candidate key has a path dfeols runs today; see the [not built] paths above")
    out.append(f"  (probe took {r['seconds']:.0f} s; memory is rough, wall time is not estimated yet)")
    return "\n".join(out)
