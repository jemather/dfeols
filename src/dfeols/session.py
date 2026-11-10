# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Interactive sessions: absorb a set of fixed effects once, then fit many models against
them (docs/design/sessions.md, build order step 2: the session core).

A manager process (`dfeols session serve`, or a SLURM job from `dfeols session start`)
demeans every declared variable on all the session's FEs once and keeps the residualised
columns per shard, plus their full weighted Gram matrix. A model `y ~ x1 + x2` is then OLS on
residualised columns (Frisch-Waugh-Lovell): coefficients and iid standard errors come from
the Gram matrix with no pass over the data; hetero and CRV1 take one pass for the meat.
Clients talk to the manager through JSON files in the session folder, so no network path to
the compute node is needed and an R client needs only jsonlite.

    <root>/<name>.lock                       DirLock held by the manager
    <root>/<name>/
      session.json                          state: id, status, fixed spec, ends, heartbeat
      requests/<id>.json                    written by clients (atomic rename)
      responses/<id>.json                   written by the manager
      models/<nnn>-<label>/result.json      each model: Result, and its estimation record
      results/index.jsonl                   one line per model
      events.jsonl                          the timeline (the collection's events)
      session.estrec-collection.json    on export and when the session ends

The demeaned data go to <demeaned>/<dataset>/<key>/ (key: a hash of what fixes them) and
are reused by a later session with the same inputs.

Created variables (`add_variable`, an expression on raw columns, dfeols.expr) are computed
on the session's rows and demeaned like the declared ones, then kept in
<key>/created/<digest>/ for later sessions; `refit` reruns a model with another vcov.

Not built yet (build order 5-6): exports, the R client, IV.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl

from .executors import SerialExecutor
from .formula import Formula, fe_components, parse
from .linalg import absorbed, collinear_keep, reference_ss, solve_spd
from .lock import DirLock, LockedError
from .prep import Dataset
from .progress import Progress, fmt_td
from .results import Result
from .vcov import df_k as _df_k
from .vcov import vcov_crv1, vcov_hetero, vcov_iid
from .worker import Plan, _maxrss_gb, _meat, _prepare, _round1_dense

log = logging.getLogger("dfeols")

SESSION_FORMAT = "dfeols-session/1"
DEMEANED_FORMAT = "dfeols-demeaned/3"  # 2: raw_ss, the collinearity scale; 3: sample/ masks
CREATED_FORMAT = "dfeols-created/2"  # 2: a batch of columns per folder (i() levels together)
CROSSING_A = Path("crossing") / "A.npz"  # plan C's level-space A, kept for created variables
CREATED_COL = "__created_"  # + i: a created column in its batch's files (each session names it)
COLLECTION_NAME = "session.estrec-collection.json"
WEIGHT_COL = "__w"
TASK_OVERHEAD = 0.005  # seconds a shard task costs beyond its reading (scheduling, results)
VCOVS = ("iid", "hetero", "CRV1")
# export suffix -> estrec table format (.doc: an HTML document styled for Word, which opens it)
EXPORT_FORMATS = {
    ".tex": "latex",
    ".md": "markdown",
    ".html": "html",
    ".htm": "html",
    ".csv": "csv",
    ".txt": "text",
    ".doc": "word",
}
_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write_json(path: Path, obj: Any) -> None:
    """Write atomically: readers on another node see the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable) + "\n")
    os.replace(tmp, path)


def workers_text(w: dict) -> str:
    """'25 of 40 running (15 queued or starting)', from a session's worker counts."""
    r, m = w["running"], w["requested"]
    return f"{r} of {m} running" + (f" ({m - r} queued or starting)" if r < m else "")


def workers_note(w: dict | None) -> str:
    """What an estimate leaves out about the workers: nothing when all are up."""
    if not w or w["running"] >= w["requested"]:
        return ""
    if not w["running"]:
        return (
            f"; 0 of {w['requested']} workers running yet (their jobs are queued or starting),"
            " so it waits for them first"
        )
    return f"; on {workers_text(w)}"


def _appeared(path: Path) -> bool:
    """Whether another node's file is there yet. Listing the folder, not stat on the name: a
    network filesystem can cache "no such file" for the name for its attribute-cache time
    (about 30 s on OSCER's /scratch, 2026-11-08), while opening a folder revalidates it."""
    try:
        return path.name in os.listdir(path.parent)
    except FileNotFoundError:
        return False


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer | np.bool_):
        return o.item()
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _append_jsonl(path: Path, obj: Any) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_jsonable) + "\n")


def _secs(s: float) -> str:
    return fmt_td(s) if s >= 1 else f"{s:.1f} s"


def _cl_col(name: str) -> str:
    return f"__cl_{name}"


# ---------------------------------------------------------------- the fixed spec


@dataclasses.dataclass
class Spec:
    """What a session fixes: data, FEs, weights, declared variables, sample rule, tolerance."""

    data: str
    fe: list[str]
    variables: list[str]
    clusters: list[str] = dataclasses.field(default_factory=list)
    weights: str | None = None
    drop_singletons: bool = True
    fe_tol: float = 1e-10
    fe_method: str = "auto"
    max_dummy_levels: int = 2000
    # crossing FEs: "dense" dummies, "iterate" (plan C: one CG solve for every declared
    # variable), or "auto": iterate when any crossing FE has more than max_dummy_levels
    crossing_fe: str = "auto"
    iter_tol: float = 1e-8
    iter_maxiter: int = 5_000
    level_space_gb: float | None = None
    crossing_precond: str = "auto"

    @classmethod
    def make(cls, data, fe, variables, clusters=None, **kw) -> Spec:
        split = lambda v, sep: (
            [s.strip() for s in v.split(sep) if s.strip()] if isinstance(v, str) else list(v)
        )
        fes = split(fe, "+")
        _no_slopes(fes)
        return cls(
            data=str(Path(data).resolve()),
            fe=fes,
            variables=split(variables, ","),
            clusters=split(clusters or [], ","),
            **kw,
        )

    def key(self, ds: Dataset) -> str:
        """Names the demeaned data: the dataset (its meta and source manifest), FEs, weights,
        declared variables, singleton rule, tolerance. Same inputs, same key."""
        src = ds.meta.get("source") or {}
        h = hashlib.sha256(
            json.dumps(
                {
                    "format": DEMEANED_FORMAT,
                    "dataset": hashlib.sha256((ds.root / "dfeols_meta.json").read_bytes()).hexdigest(),
                    "source": {
                        k: src.get(k) for k in ("sha256", "manifest_sha256", "content_hash") if k in src
                    },
                    "fe": sorted(self.fe),
                    "variables": sorted(self.variables),
                    "clusters": sorted(self.clusters),
                    "weights": self.weights,
                    "drop_singletons": self.drop_singletons,
                    "fe_tol": self.fe_tol,
                    "fe_method": self.fe_method,
                    "crossing": self.crossing_method(ds),
                    **({"iter_tol": self.iter_tol} if self.crossing_method(ds) == "iterate" else {}),
                },
                sort_keys=True,
            ).encode()
        )
        return h.hexdigest()[:16]

    def crossing_method(self, ds: Dataset) -> str | None:
        """How the crossing FEs are absorbed: "dense", "iterate", or None (none cross)."""
        if self.crossing_fe not in ("auto", "dense", "iterate"):
            raise ValueError(f"crossing_fe must be auto, dense or iterate, not {self.crossing_fe!r}")
        levels = [
            ds.fes[k]["n_levels"]
            for k in ("^".join(fe_components(n)) for n in self.fe)
            if k in ds.fes and not ds.fes[k]["nested_in_partition"]
        ]
        if not levels:
            return None
        if self.crossing_fe == "auto":
            return "iterate" if max(levels) > self.max_dummy_levels else "dense"
        return self.crossing_fe


def _session_plan(spec: Spec, ds: Dataset, keep_dir: Path | None) -> tuple[Plan, list[dict]]:
    """A worker Plan whose outcome block is the declared variables and whose regressor
    block is the crossing-FE dummies, plus the session's cluster definitions."""
    if not spec.fe:
        raise ValueError("a session needs at least one fixed effect")
    if not spec.variables:
        raise ValueError("a session needs declared variables (--vars)")
    bad = [v for v in spec.variables if not re.match(r"^[A-Za-z_]\w*$", v)]
    if bad:
        raise ValueError(f"declared variables must be column names: {bad}")
    dup = sorted({v for v in spec.variables if spec.variables.count(v) > 1})
    if dup:
        raise ValueError(f"variables declared twice: {dup}")
    missing = (set(spec.variables) | ({spec.weights} if spec.weights else set())) - set(ds.columns)
    if missing:
        raise KeyError(f"not in the prepared data: {sorted(missing)}")
    local, glob = [], []
    _no_slopes(spec.fe)
    for name in spec.fe:
        key = "^".join(fe_components(name))
        if key not in ds.fes:
            raise KeyError(f"FE {name!r} was not encoded by prep(); available: {list(ds.fes)}")
        m = ds.fes[key]
        if m["nested_in_partition"]:
            local.append((key, m["column"]))
        else:
            glob.append((key, m["column"], m["n_levels"]))
    method = spec.crossing_method(ds)
    if method == "dense":
        big = [(k, L) for k, _, L in glob if L > spec.max_dummy_levels]
        if big:
            raise ValueError(
                f"FE {big[0][0]!r} crosses the partition with {big[0][1]:,} levels (> max_dummy_levels="
                f"{spec.max_dummy_levels}); dense dummies would need rows x levels x 8 bytes per shard. "
                "Use crossing_fe='iterate' (or 'auto')"
            )
    clusters = []
    for c in spec.clusters:
        key = "^".join(fe_components(c))
        if key not in ds.clusters:
            raise KeyError(f"cluster {c!r} was not encoded by prep(); available: {list(ds.clusters)}")
        m = ds.clusters[key]
        clusters.append(
            {
                "name": key,
                "column": m["column"],
                "levels": m["n_levels"],
                "local": bool(m["nested_in_partition"]),
            }
        )
    raw = set(spec.variables) | ({spec.weights} if spec.weights else set())
    for name in [*spec.fe, *spec.clusters]:
        raw.update(fe_components(name))
    raw_key = sorted(c for c in raw if c in ds.columns)
    f = Formula(
        text=f"{' + '.join(spec.variables)} ~ 1 | {' + '.join(spec.fe)}",
        depvars=tuple(spec.variables),
        terms=(),
        fes=tuple(spec.fe),
    )
    plan = Plan(
        formula=f,
        spec_id="session",
        local_fes=local,
        global_fes=glob,
        cluster=None,
        cluster_local=False,
        weights=spec.weights,
        cat_levels={},
        colnames=[],
        n_dummy=sum(L - 1 for *_, L in glob) + (1 if glob and not local else 0),
        vcov_types=(),
        cache_dir=None,
        drop_singletons=spec.drop_singletons,
        fe_tol=spec.fe_tol,
        fe_method=spec.fe_method,
        keep_dir=str(keep_dir) if keep_dir and glob and spec.drop_singletons else None,
        # every cluster's code column is read with the rows (prep drops rows missing a key,
        # so this deletes no extra rows); the raw ones hash into the records' row_key
        row_key_cols=tuple(sorted({*raw_key, *(c["column"] for c in clusters)})),
        sparse_global=method == "iterate",
        crossing_iterate=method == "iterate",
    )
    plan.raw_key = raw_key  # type: ignore[attr-defined]
    return plan, clusters


# ---------------------------------------------------------------- shard tasks


def _stage_path(out: Path, sid: int) -> Path:
    return out / "stage" / f"shard_{sid:06d}.npz"


def _part_path(out: Path, sid: int) -> Path:
    return out / "shards" / f"part-{sid:06d}.parquet"


def _write_part(
    out: Path, sid: int, names: list[str], V: np.ndarray, w: np.ndarray | None, cls: dict
) -> None:
    """One shard's final columns; with the weights and cluster codes, except for a created
    variable (w None), whose files hold its column only."""
    cols = {n: V[:, j] for j, n in enumerate(names)}
    if w is not None:
        cols[WEIGHT_COL] = w
    for name, codes in cls.items():
        cols[_cl_col(name)] = codes
    p = _part_path(out, sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    pl.DataFrame(cols).write_parquet(tmp)
    os.replace(tmp, p)


def _cross(V: np.ndarray, w: np.ndarray, sid: int, others) -> np.ndarray:
    """V'W O for the session's existing columns O of this shard (created variables)."""
    O = _read_columns(others, sid, V.shape[0])
    return (V * w[:, None]).T @ O


def _read_columns(groups, sid: int, n: int | None = None) -> np.ndarray:
    """The columns of `groups` [(root, [(file column, name)])] for one shard, in order."""
    blocks = []
    for root, cols in groups:
        if not cols:
            continue
        df = pl.read_parquet(_part_path(Path(root), sid), columns=[c for c, _ in cols])
        if n is not None and df.height != n:
            raise RuntimeError(f"shard {sid}: {root} has {df.height} rows, not {n}")
        blocks.append(df.to_numpy().astype(np.float64))
    return np.hstack(blocks) if blocks else np.zeros((n or 0, 0))


def demean_round1(
    plan: Plan, clusters: list[dict], sid: int, files: list[str], out: str, others=None
) -> dict:
    """Per shard: the session sample, every declared variable demeaned by the local FEs and
    the crossing-FE dummies too, their cross-products, and the cluster facts. With no
    crossing FE the columns are final and written; else they wait in a stage file. For a
    created variable (plan.created), `others` are the session's columns to cross it with."""
    from .rowhash import rowset

    t0 = time.perf_counter()
    out_dir = Path(out)
    T: dict = {}
    loaded: dict = {}
    df, n0, local_codes, w = _prepare(plan, sid, files, loaded)
    n = df.height
    T["read"] = time.perf_counter() - t0
    aux = None
    raw: dict = {}
    if plan.crossing_iterate:  # plan C: per-level sums D'W V~ and the shard's part of A
        from .worker import _iter_store, _round1_sparse

        Y, Yt, _, _, _, extra, converged, aux = _round1_sparse(plan, sid, df, n0, local_codes, w, T, t0)
        _iter_store(plan, sid, extra[0], extra[1], w)  # for CG steps with a pass each
        Zt = ZtZ = ZtY = None
    else:
        Y, Yt, Zt, ZtZ, ZtY, converged, _ = _round1_dense(plan, df, n, local_codes, w, T, raw)
    r: dict[str, Any] = {
        "shard": sid,
        "rows_in": n0,
        "load": loaded,
        "n": n,
        "sumw": float(w.sum()),
        "converged": converged,
        "fe_levels": [int(c.max()) + 1 if len(c) else 0 for c in local_codes],
        "global_present": [
            np.bincount(df[c].to_numpy().astype(np.int64), minlength=L) > 0 for _, c, L in plan.global_fes
        ],
        "row_key": rowset(df, plan.raw_key) if plan.raw_key else None,  # type: ignore[attr-defined]
        "clusters": {},
        # the collinearity scale: each variable's sums before the FEs, and net of the local FEs
        "y_s1": w @ Y,
        "y_s2": w @ (Y * Y),
        "y_net": w @ (Yt * Yt),
    }
    if raw:  # the dense dummy block's, likewise
        r["z_s1"], r["z_s2"] = raw["s1"], raw["s2"]
    cls = {}
    for c in clusters:
        codes = df[c["column"]].to_numpy().astype(np.int64)
        cls[c["name"]] = codes
        nested = []
        for lc in local_codes:
            if not len(lc):
                nested.append(True)
                continue
            lo = np.full(lc.max() + 1, np.iinfo(np.int64).max)
            hi = np.full(lc.max() + 1, np.iinfo(np.int64).min)
            np.minimum.at(lo, lc.astype(np.int64), codes)
            np.maximum.at(hi, lc.astype(np.int64), codes)
            nested.append(bool((lo == hi).all()))
        gmm = []
        for _, col, L in plan.global_fes:
            g = df[col].to_numpy().astype(np.int64)
            lo = np.full(L, np.iinfo(np.int64).max)
            hi = np.full(L, np.iinfo(np.int64).min)
            np.minimum.at(lo, g, codes)
            np.maximum.at(hi, g, codes)
            gmm.append((lo, hi))
        r["clusters"][c["name"]] = {"present": np.unique(codes), "local_nested": nested, "global_minmax": gmm}
    names = list(plan.formula.depvars)
    if plan.global_fes:
        stage: dict[str, np.ndarray] = {"Y": Yt, "w": w, **{_cl_col(k): v for k, v in cls.items()}}
        if aux is not None:
            r["crossing"] = aux
            offs = np.r_[0, np.cumsum([L for *_, L in plan.global_fes])[:-1]]
            stage["G"] = (
                np.stack(
                    [df[c].to_numpy().astype(np.int64) + o for (_, c, _), o in zip(plan.global_fes, offs)]
                )
                if n
                else np.zeros((len(plan.global_fes), 0), np.int64)
            )
            for i, lc in enumerate(local_codes):
                stage[f"L{i}"] = lc
        else:
            r["ZtZ"], r["ZtY"] = ZtZ, ZtY
            stage["Z"] = Zt
        p = _stage_path(out_dir, sid)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **stage)
        os.replace(tmp, p)
    else:
        r["gram"] = (Yt * w[:, None]).T @ Yt
        if others:
            r["cross"] = _cross(Yt, w, sid, others)
        _write_part(out_dir, sid, names, Yt, None if plan.created else w, cls)
    T["total"] = time.perf_counter() - t0
    T["maxrss_gb"] = _maxrss_gb()
    r["timing"] = T
    return r


def demean_round2(
    sid: int,
    out: str,
    names: list[str],
    keep: np.ndarray | None,
    gamma: np.ndarray,
    plan: Plan | None = None,
    others=None,
    bare: bool = False,
) -> dict:
    """Per shard: partial the crossing-FE dummies out (V~ - D~ Gamma), write the final
    columns, and return their Gram matrix. Dense: D~ is the stage's demeaned dummy block.
    Plan C (`plan` given): D~ Gamma is D Gamma (each row's levels summed) demeaned by the
    local FEs."""
    from .demean import demean

    t0 = time.perf_counter()
    out_dir = Path(out)
    p = _stage_path(out_dir, sid)
    with np.load(p) as d:
        Yt, w = d["Y"], d["w"]
        cls = {k.removeprefix("__cl_"): d[k] for k in d.files if k.startswith("__cl_")}
        if plan is None:
            Zt = d["Z"]
        else:
            G = d["G"]
            local = [
                d[k]
                for k in sorted((k for k in d.files if re.fullmatch(r"L\d+", k)), key=lambda s: int(s[1:]))
            ]
    if not len(w):
        V = Yt
    elif plan is None:
        V = Yt - Zt[:, keep] @ gamma
    else:
        U = sum(gamma[g] for g in G)
        if local:
            U = demean(U, local, w, plan.fe_tol, plan.fe_maxiter, plan.fe_method)[0]
        V = Yt - U
    if plan is not None and plan.iter_token:
        from .worker import _ITER_CACHE

        _ITER_CACHE.pop((plan.iter_token, sid), None)
    _write_part(out_dir, sid, names, V, None if bare else w, cls)
    p.unlink()
    r = {
        "shard": sid,
        "gram": (V * w[:, None]).T @ V,
        "timing": {"total": time.perf_counter() - t0, "maxrss_gb": _maxrss_gb()},
    }
    if others:
        r["cross"] = _cross(V, w, sid, others)
    return r


def model_meat(parts, T: list[str], maps, vcov_types: tuple, cluster) -> dict:
    """Per shard, for one model: residuals, RSS and the meats (hetero, CRV1) of each group of
    `maps` [(prefix, A, H)], whose residuals are T @ A and score regressors T @ H, T the
    columns named `T`. `parts`: the declared variables' file and, for created variables,
    theirs, [(path, [(file column, name)])], the first holding the weights and cluster codes.
    `cluster`: one cluster, or for two-way clustering a list of three, the third the two's
    intersection ({"pair": [a, b], "levels_b": ...}; its codes a * levels_b + b); a group's
    results for cluster i > 0 carry the prefix f"{group}c{i}_"."""
    t0 = time.perf_counter()
    path, own = parts[0]
    cls = cluster if isinstance(cluster, list) else ([cluster] if cluster else [])
    extra = [WEIGHT_COL] + [_cl_col(c["name"]) for c in cls if "pair" not in c]
    df = pl.read_parquet(path, columns=list(dict.fromkeys([*(c for c, _ in own), *extra])))
    df = df.rename({c: nm for c, nm in own if c != nm})
    for p_, cols in parts[1:]:
        o = pl.read_parquet(p_, columns=[c for c, _ in cols])
        if o.height != df.height:
            raise RuntimeError(f"{p_} has {o.height} rows, {path} {df.height}")
        df = df.hstack(o.rename({c: nm for c, nm in cols}))
    Tm = df.select(T).to_numpy().astype(np.float64) if T else np.zeros((df.height, 0))
    w = df[WEIGHT_COL].to_numpy()
    codes = []
    for c in cls:
        if "pair" in c:
            a, b = c["pair"]
            codes.append(
                df[_cl_col(a)].to_numpy().astype(np.int64) * int(c["levels_b"])
                + df[_cl_col(b)].to_numpy().astype(np.int64)
            )
        else:
            codes.append(df[_cl_col(c["name"])].to_numpy())
    out: dict = {"n": df.height}
    for gp, A, H in maps:
        E, X = Tm @ A, Tm @ H
        if not cls:
            ns = SimpleNamespace(vcov_types=vcov_types, cluster=None, cluster_local=False)
            _meat(ns, X, E, w, None, out, prefix=gp)
        for i, c in enumerate(cls):
            ns = SimpleNamespace(
                vcov_types=vcov_types if i == 0 else ("CRV1",),
                cluster=(c["name"], None, c.get("levels")),
                cluster_local=bool(c["local"]),
            )
            _meat(ns, X, E, w, codes[i], out, prefix=gp if i == 0 else f"{gp}c{i}_")
    out["timing"] = {"total": time.perf_counter() - t0, "maxrss_gb": _maxrss_gb()}
    return out


class _MeatAcc:
    """One group's meats over the shards: RSS, hetero, and CRV1 per cluster (two-way: a, b
    and a&b; a cluster crossing the partition by its scores per level, the intersection's
    sparsely), combined at the end as V_a + V_b - V_ab is linear in the meat."""

    def __init__(self, ne: int, kk: int, vcov: tuple, cls: list[dict]):
        self.ne, self.kk = ne, kk
        self.rss = np.zeros(ne)
        self.hetero = np.zeros((ne, kk, kk)) if "hetero" in vcov else None
        self.crvs = []
        if "CRV1" in vcov:
            for c in cls:
                e: dict[str, Any] = {"meat": np.zeros((ne, kk, kk)), "scores": None, "parts": None}
                if not c["local"]:
                    if c.get("levels") is not None:
                        e["scores"] = np.zeros((ne, c["levels"], kk))
                    else:
                        e["parts"] = []
                self.crvs.append(e)

    def add(self, r: dict, gp: str) -> None:
        self.rss += r[gp + "rss"]
        if self.hetero is not None:
            self.hetero += r[gp + "meat_hetero"]
        for i, e in enumerate(self.crvs):
            px = gp if i == 0 else f"{gp}c{i}_"
            if px + "meat_crv" in r:
                e["meat"] += r[px + "meat_crv"]
            elif px + "cluster_scores" in r and e["scores"] is not None:
                for j in range(self.ne):
                    np.add.at(e["scores"][j], r[px + "cluster_codes"], r[px + "cluster_scores"][j])
            elif px + "cluster_scores" in r:
                e["parts"].append((r[px + "cluster_codes"], r[px + "cluster_scores"]))

    def finish(self) -> dict:
        for e in self.crvs:
            if e["scores"] is not None:
                for j in range(self.ne):
                    e["meat"][j] = e["scores"][j].T @ e["scores"][j]
            elif e["parts"]:
                codes = np.concatenate([c_ for c_, _ in e["parts"]])
                uniq, inv = np.unique(codes, return_inverse=True)
                for j in range(self.ne):
                    sc = np.zeros((len(uniq), self.kk))
                    np.add.at(sc, inv, np.concatenate([s_[j] for _, s_ in e["parts"]]))
                    e["meat"][j] = sc.T @ sc
        crv = None
        if self.crvs:
            signs = [1.0] if len(self.crvs) == 1 else [1.0, 1.0, -1.0]
            crv = sum(sg * e["meat"] for sg, e in zip(signs, self.crvs))
        return {"rss": self.rss, "hetero": self.hetero, "crv": crv}


# ---------------------------------------------------------------- demean once


def _level_reduce(gram, plan: Plan, executor=None):
    """Sum the shards' parts of A on the workers as they arrive (as fit's round 1)."""
    import functools

    from .crossing import combine_parts, part_of, without_part, worker_bytes
    from .executors import Reduce

    wb = worker_bytes(executor)
    return Reduce(
        get=part_of,
        strip=without_part,
        combine=functools.partial(combine_parts, offs=gram.offs, max_nnz=plan.level_nnz, worker_bytes=wb),
        done=lambda part: gram.add(part, None),
    )


class Demeaned:
    """The persisted demeaned data: residualised columns per shard and manifest.json, plus
    the variables this session created (each in created/<digest>/, with variable.json).

    `names`, `gram` and `raw_ss` cover the declared variables and then the created ones;
    `ids` name the columns across sessions (a declared variable's name, a created one's
    "created/<digest>": another session may call it something else)."""

    def __init__(self, root: Path, manifest: dict):
        self.root = Path(root)
        self.manifest = manifest
        self.declared = list(manifest["variables"])
        self.names = list(self.declared)
        self.ids = list(self.declared)
        self.gram = np.asarray(manifest["gram"], float)
        rs = manifest.get("raw_ss")
        self.raw_ss = None if rs is None else np.asarray(rs, float)
        self.shards = manifest["shards"]
        self.created: list[dict] = []  # {name, id, dir (relative), expression, info}

    @classmethod
    def open(cls, root) -> Demeaned:
        return cls(Path(root), _read_json(Path(root) / "manifest.json"))

    def part(self, sid: int) -> str:
        return str(_part_path(self.root, sid))

    def groups(self, names=None) -> list[tuple[str, list[tuple[str, str]]]]:
        """[(folder, [(file column, name)])] holding `names` (default: every column), the
        declared variables' folder first (it has the weights and cluster codes)."""
        want = set(self.names if names is None else names)
        out = [(str(self.root), [(n, n) for n in self.declared if n in want])]
        by_dir: dict[str, list] = {}
        for c in self.created:
            if c["name"] in want:
                by_dir.setdefault(str(self.root / c["dir"]), []).append((c["info"]["column"], c["name"]))
        return out + list(by_dir.items())

    def files(self, sid: int, names) -> list[tuple[str, list[tuple[str, str]]]]:
        return [(str(_part_path(Path(r), sid)), cols) for r, cols in self.groups(names)]

    def expression(self, name: str) -> str | None:
        return next((c["expression"] for c in self.created if c["name"] == name), None)

    def add_created(
        self, name: str, info: dict, cross: dict[str, float], expression: str | None, dir: str
    ) -> None:
        """Add a created column: its entry in its batch's variable.json (`info`), its
        cross-product with each current column (by id; `known_cross` and a pass for the
        rest), the expression as this session wrote it (default: as built), and its batch
        folder relative to the root."""
        cid = f"created/{info['digest']}"
        row = np.array([cross[i] for i in self.ids], float)
        k = len(self.ids)
        g = np.zeros((k + 1, k + 1))
        g[:k, :k] = self.gram
        g[k, :k] = g[:k, k] = row
        g[k, k] = float(info["gram_self"])
        self.gram = g
        if self.raw_ss is not None:
            self.raw_ss = np.r_[self.raw_ss, float(info["raw_ss"])]
        self.names.append(name)
        self.ids.append(cid)
        self.created.append(
            {
                "name": name,
                "id": cid,
                "dir": dir,
                "expression": expression or info["expression"],
                "info": info,
            }
        )

    def created_index(self) -> dict[str, tuple[str, dict]]:
        """Every created column kept with these data, by digest: (batch folder relative to
        the root, its entry in the batch's variable.json)."""
        out: dict[str, tuple[str, dict]] = {}
        for vj in sorted((self.root / "created").glob("*/variable.json")):
            try:
                doc = _read_json(vj)
            except (OSError, ValueError):
                continue
            if doc.get("format") != CREATED_FORMAT:
                continue
            for v in doc.get("variables", []):
                out.setdefault(v["digest"], (f"created/{vj.parent.name}", v))
        return out

    def known_cross(self, info: dict) -> dict[str, float]:
        """A created variable's cross-products with the current columns that are on file:
        its own variable.json's, and the other created variables' with it."""
        cid = f"created/{info['digest']}"
        out = {i: float(v) for i, v in info.get("cross", {}).items() if i in self.ids}
        for c in self.created:
            v = c["info"].get("cross", {}).get(cid)
            if v is not None and c["id"] not in out:
                out[c["id"]] = float(v)
        return out

    def bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("part-*.parquet"))


def demean_once(
    spec: Spec,
    ds: Dataset,
    out: Path,
    executor,
    label: str = "session",
    log_every: float = 30.0,
    progress_dir: Path | None = None,
    create: str | None = None,
    base: Demeaned | None = None,
):
    """Demean every declared variable on all the session's FEs; write out/shards and
    out/manifest.json (last, so a manifest means the data are complete).

    With `create` (expressions on raw columns, dfeols.expr) and `base` (the session's
    demeaned data): those columns instead, demeaned together, on the same rows (the declared
    variables' sample and base's singleton masks), crossed with every column of `base` as it
    goes; returns what the batch's variable.json needs (the caller writes it). A row of the
    sample where an expression is missing or not finite is an error, with the count."""
    from . import rowhash
    from .estimate import _drop_singletons
    from .record import dfeols_source

    t_start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    plan, clusters = _session_plan(spec, ds, out / "singletons")
    names = list(spec.variables)
    others = None
    if isinstance(create, str):
        create = [create]
    if create is not None:
        names = [f"{CREATED_COL}{i}" for i in range(len(create))]
        sample = base.root / "sample"
        plan = dataclasses.replace(
            plan,
            formula=Formula(
                text=f"{' + '.join(names)} ~ 1 | {' + '.join(spec.fe)}",
                depvars=tuple(names),
                terms=(),
                fes=tuple(spec.fe),
            ),
            filter_vars=tuple(spec.variables),
            created=tuple(zip(names, create)),
            keep_dir=str(sample) if plan.keep_dir else None,
            row_key_cols=(),
        )
        plan.raw_key = []  # type: ignore[attr-defined]
        if plan.keep_dir and not sample.is_dir():
            raise RuntimeError(
                f"{base.root} has no singleton masks (sample/): rebuild it to create variables"
            )
        clusters = []
        others = base.groups()
    prog = Progress(progress_dir or out, label=label, every=log_every)
    rows = {s["id"]: max(s["rows"], 1) for s in ds.shards}
    sing = {"passes": 0, "seconds": 0.0}
    if plan.keep_dir and create is None:
        sing = _drop_singletons(plan, ds, executor, prog, rows, label)
    acc: dict[str, Any] = {
        "n": 0,
        "rows_in": 0,
        "sumw": 0.0,
        "converged": True,
        "load": {"rows": 0, "missing": 0, "non_finite": 0, "zero_weights": 0},
        "fe_levels": np.zeros(len(plan.local_fes), np.int64),
        "gpresent": [np.zeros(L, bool) for *_, L in plan.global_fes],
        "row_key": [],
        "gram": np.zeros((len(names), len(names))),
        "y_s1": np.zeros(len(names)),
        "y_s2": np.zeros(len(names)),
        "y_net": np.zeros(len(names)),
        "rss_gb": 0.0,
        "missing": {},
        "cross": np.zeros((len(names), len(base.names) if create is not None else 0)),
        "cl": {
            c["name"]: {
                "G_local": 0,
                "present": None if c["local"] else np.zeros(c["levels"], bool),
                "local_nested": np.ones(len(plan.local_fes), bool),
                "glo": [np.full(L, np.iinfo(np.int64).max) for *_, L in plan.global_fes],
                "ghi": [np.full(L, np.iinfo(np.int64).min) for *_, L in plan.global_fes],
            }
            for c in clusters
        },
    }
    k_tot = plan.n_dummy
    reduce = None
    if plan.crossing_iterate:
        from .crossing import LevelGram, level_space_budget, part_budget, worker_bytes

        plan.iter_token = f"{out}:{uuid.uuid4().hex}"
        # beside the folder, which an interrupted build's rerun deletes
        plan.cg_checkpoint = str(out.parent / f"{out.name}.cg_checkpoint.npz")
        plan.level_nnz = level_space_budget(executor, spec.level_space_gb)
        plan.part_nnz = part_budget(worker_bytes(executor)) if plan.level_nnz else 0
        offs = np.r_[0, np.cumsum([L for *_, L in plan.global_fes])].astype(np.int64)
        acc["DR"] = np.zeros((int(offs[-1]), len(names)))
        acc["ddiag"] = np.zeros(int(offs[-1]))
        saved_A = base.root / CROSSING_A if create is not None else None
        if saved_A is not None and saved_A.exists():
            # A depends only on the FEs, sample and weights: the build's, so round 1 skips it
            import scipy.sparse as sp

            acc["gram_A"] = LevelGram(offs, [n for n, *_ in plan.global_fes], max(plan.level_nnz, 1))
            acc["gram_A"].A = sp.load_npz(saved_A).tocsr()
            plan.level_nnz = 0
        elif plan.level_nnz:
            acc["gram_A"] = LevelGram(offs, [n for n, *_ in plan.global_fes], plan.level_nnz)
            reduce = _level_reduce(acc["gram_A"], plan, executor)
    elif plan.global_fes:
        acc["ZtZ"] = np.zeros((k_tot, k_tot))
        acc["ZtY"] = np.zeros((k_tot, len(names)))
        acc["z_s1"], acc["z_s2"] = np.zeros(k_tot), np.zeros(k_tot)
    local_of = {c["name"]: c["local"] for c in clusters}

    def on1(tid, r):
        acc["rss_gb"] = max(acc["rss_gb"], r["timing"].get("maxrss_gb", 0.0))
        for k in ("n", "rows_in", "sumw", "y_s1", "y_s2", "y_net"):
            acc[k] += r[k]
        for k in ("z_s1", "z_s2"):  # the dense dummy block, when there is one
            if k in acc:
                acc[k] += r[k]
        for k, v in r["load"].items():
            if k == "created_missing":
                for nm, c_ in v.items():
                    acc["missing"][nm] = acc["missing"].get(nm, 0) + c_
            else:
                acc["load"][k] += v
        if "cross" in r:
            acc["cross"] += r["cross"]
        acc["converged"] &= r["converged"]
        acc["fe_levels"] += np.asarray(r["fe_levels"], np.int64)
        for i, p in enumerate(r["global_present"]):
            acc["gpresent"][i] |= p
        if r["row_key"]:
            acc["row_key"].append(r["row_key"])
        if "gram" in r:
            acc["gram"] += r["gram"]
        elif "crossing" in r:
            cr = r["crossing"]
            acc["DR"][cr["idx"]] += cr["DR"]
            acc["ddiag"][cr["idx"]] += cr["ddiag"]
            if "gram_A" in acc and "part" in cr:  # else summed on the workers (reduce)
                acc["gram_A"].add(cr["part"], tid)
        else:
            acc["ZtZ"] += r["ZtZ"]
            acc["ZtY"] += r["ZtY"]
        for name, c in r["clusters"].items():
            a = acc["cl"][name]
            if local_of[name]:
                a["G_local"] += len(c["present"])
            else:
                a["present"][c["present"]] = True
            if r["n"]:
                a["local_nested"] &= np.asarray(c["local_nested"], bool)
            for i, (lo, hi) in enumerate(c["global_minmax"]):
                a["glo"][i] = np.minimum(a["glo"][i], lo)
                a["ghi"][i] = np.maximum(a["ghi"][i], hi)

    tasks = [
        (s["id"], rows[s["id"]], (plan, clusters, s["id"], ds.shard_paths(s), str(out), others))
        for s in ds.shards
    ]
    prog.add_stage("demean", rows)
    prog.start("demean")
    if reduce is not None and "reduce" in inspect.signature(executor.run).parameters:
        prog.task_gb = None  # a shard's part of A is not in any estimate: a whole worker
        executor.run(demean_round1, tasks, on1, prog, "demean", reduce=reduce)
    else:
        executor.run(demean_round1, tasks, on1, prog, "demean")
    prog.finish("demean")
    from .estimate import check_sample

    check_sample(int(acc["n"]), int(acc["rows_in"]), acc["load"])
    bad = [(e, acc["missing"].get(nm, 0)) for nm, e in zip(names, create or [])]
    bad = [(e, c_) for e, c_ in bad if c_]
    if bad:
        raise ValueError(
            "; ".join(f"{e!r} is missing or not finite on {c_:,}" for e, c_ in bad)
            + f" of the session's {int(acc['n']):,} rows; a created variable must be defined on "
            "every row of the sample, which is fixed (a new session with it declared changes "
            "the sample)"
        )
    crossing: dict[str, Any] = {}
    extra: tuple = ()
    # each variable's centered sum of squares before the FEs (linalg.reference_ss): the scale
    # fit_model tests a column of the Gram against, so one the FEs absorb is dropped
    has_fe = bool(plan.local_fes or plan.global_fes)
    ref_y = reference_ss(acc["y_s1"], acc["y_s2"], acc["sumw"]) if has_fe else None
    if plan.crossing_iterate:
        if plan.local_fes:  # absorbed by the local FEs: rounding noise, zero CG column
            gone = absorbed(acc["y_net"], ref_y)
            acc["DR"][:, gone] = 0.0
        from .estimate import _crossing_cg

        # _crossing_cg reads A (a LevelGram) as "gram"
        cg_acc = {"DR": acc["DR"], "ddiag": acc["ddiag"], "gram": acc.pop("gram_A", None)}
        gamma = _crossing_cg(
            plan,
            ds,
            executor,
            prog,
            rows,
            cg_acc,
            label,
            spec.iter_tol,
            spec.iter_maxiter,
            spec.crossing_precond,
        )
        keep = None
        extra = (plan,)
        crossing = {"method": "iterate", "levels": int(len(acc["DR"])), "cg": cg_acc["cg"]}
        g_A = cg_acc.get("gram")
        if create is None and g_A is not None and g_A.full and g_A.blocks_ok:
            # kept for created variables: their CG then needs no A from the shards
            import scipy.sparse as sp

            (out / CROSSING_A).parent.mkdir(parents=True, exist_ok=True)
            sp.save_npz(out / CROSSING_A, g_A.finish()[0])
            crossing["A_saved"] = str(CROSSING_A)
        elif create is not None and saved_A is not None and saved_A.exists():
            crossing["A_from"] = str(CROSSING_A)
        from .worker import _ITER_CACHE

        for k in [k for k in _ITER_CACHE if k[0] == plan.iter_token]:  # this process's (serial)
            _ITER_CACHE.pop(k, None)
    elif plan.global_fes:
        zref = reference_ss(acc["z_s1"], acc["z_s2"], acc["sumw"]) if plan.local_fes else None
        keep = collinear_keep(acc["ZtZ"], 1e-9, ref_diag=zref)
        gamma, _ = solve_spd(acc["ZtZ"][np.ix_(keep, keep)], acc["ZtY"][keep])
        crossing = {"method": "dense", "dummies": int(k_tot), "kept": int(keep.sum())}
    if plan.global_fes:

        def on2(tid, r):
            acc["rss_gb"] = max(acc["rss_gb"], r["timing"].get("maxrss_gb", 0.0))
            acc["gram"] += r["gram"]
            if "cross" in r:
                acc["cross"] += r["cross"]

        xplan = extra[0] if extra else None
        tasks2 = [
            (
                s["id"],
                rows[s["id"]],
                (s["id"], str(out), names, keep, gamma, xplan, others, create is not None),
            )
            for s in ds.shards
        ]
        prog.add_stage("partial_out", rows)
        prog.start("partial_out")
        executor.run(demean_round2, tasks2, on2, prog, "partial_out")
        prog.finish("partial_out")
        shutil.rmtree(out / "stage", ignore_errors=True)
    if plan.keep_dir and create is None:
        _pack_masks(Path(plan.keep_dir), out / "sample")
    if not acc["converged"]:
        log.warning("%s: FE demeaning did not converge in some shards", label)
    if create is not None:
        from . import expr as _expr

        gram = (acc["gram"] + acc["gram"].T) / 2
        digests = [_expr.digest(e) for e in create]
        cols = []
        for i, e in enumerate(create):
            cross = dict(zip(base.ids, map(float, acc["cross"][i])))
            cross.update({f"created/{digests[j]}": float(gram[i, j]) for j in range(len(create)) if j != i})
            cols.append(
                {
                    "expression": e,
                    "canonical": _expr.canonical(e),
                    "digest": digests[i],
                    "columns": _expr.columns(e),
                    "column": names[i],
                    "rows": int(acc["n"]),
                    "gram_self": float(gram[i, i]),
                    "raw_ss": float(ref_y[i]),
                    "cross": cross,
                }
            )
        return {
            "format": CREATED_FORMAT,
            "key": base.manifest["key"],
            "rows": int(acc["n"]),
            "variables": cols,
            "crossing": crossing,
            "converged": bool(acc["converged"]),
            "created": _now(),
            "seconds": time.time() - t_start,
            "max_worker_rss_gb": acc["rss_gb"],
        }

    fe_names = [n for n, _ in plan.local_fes] + [n for n, *_ in plan.global_fes]
    fe_levels = list(map(int, acc["fe_levels"])) + [int(p.sum()) for p in acc["gpresent"]]
    cl_info = {}
    for c in clusters:
        a = acc["cl"][c["name"]]
        nested = list(map(bool, a["local_nested"]))
        for i in range(len(plan.global_fes)):
            pres = acc["gpresent"][i]
            nested.append(bool((a["glo"][i][pres] == a["ghi"][i][pres]).all()))
        G = a["G_local"] if c["local"] else int(a["present"].sum())
        cl_info[c["name"]] = {"levels": c["levels"], "local": c["local"], "G": int(G), "fe_nested": nested}
    gram = (acc["gram"] + acc["gram"].T) / 2
    src = dfeols_source()
    from . import __version__

    rk = rowhash.combine(acc["row_key"]) if acc["row_key"] else None
    manifest = {
        "format": DEMEANED_FORMAT,
        "key": spec.key(ds),
        "spec": dataclasses.asdict(spec),
        "dataset": {
            "root": str(ds.root),
            "name": ds.root.name,
            "rows": int(ds.n_rows),
            "partition": ds.partition,
            "meta_sha256": hashlib.sha256((ds.root / "dfeols_meta.json").read_bytes()).hexdigest(),
            "source": ds.meta.get("source"),
        },
        "variables": names,
        "rows": int(acc["n"]),
        "rows_in": int(acc["rows_in"]),
        "sum_weights": float(acc["sumw"]),
        "load": acc["load"],
        "singletons": int(acc["rows_in"] - acc["n"]),
        "fixed_effects": [
            {
                "name": n,
                "levels": L,
                "absorbed": "local" if i < len(plan.local_fes) else f"crossing_{crossing['method']}",
            }
            for i, (n, L) in enumerate(zip(fe_names, fe_levels))
        ],
        "crossing": crossing,
        "clusters": cl_info,
        "converged": bool(acc["converged"]),
        "row_key": (
            {
                "algorithm": rowhash.ALGORITHM,
                "value": rowhash.hexdigest(rk),
                "rows": rk["rows"],
                "columns": plan.raw_key,  # type: ignore[attr-defined]
            }
            if rk
            else None
        ),
        "gram": gram,
        "raw_ss": ref_y,
        "shards": [s["id"] for s in ds.shards],
        "n_shards": len(ds.shards),
        "fe_maxiter": plan.fe_maxiter,
        "dfeols": {"version": __version__, **({"commit": src["commit"]} if src.get("commit") else {})},
        "created": _now(),
        "seconds": time.time() - t_start,
        "singleton_passes": sing["passes"],
        "max_worker_rss_gb": acc["rss_gb"],
    }
    _write_json(out / "manifest.json", manifest)
    d = Demeaned(out, json.loads(json.dumps(manifest, default=_jsonable)))
    log.info(
        "%s: demeaned %d variables on %s rows in %s (%.2f GB)",
        label,
        len(names),
        f"{acc['n']:,}",
        fmt_td(manifest["seconds"]),
        d.bytes() / 1e9,
    )
    return d


def _pack_masks(src: Path, dest: Path) -> None:
    """Keep the singleton masks with the demeaned data, one bit a row, so a created variable
    is computed on the same rows; then remove the passes' folder."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("shard_*.npy")):
        np.save(dest / f.name.replace(".npy", ".bits.npy"), np.packbits(np.load(f)))
    shutil.rmtree(src, ignore_errors=True)


# ---------------------------------------------------------------- one model


def _model_terms(formula: str, spec_fe: list[str], factors: dict | None = None):
    """The formula's parts: f, responses, exogenous regressors, and for IV (`y ~ x | d ~ z`)
    the endogenous regressors and the excluded instruments, each a single column; an i()
    term is its indicator columns, `factors` {term label: column names} (Manager.fit makes
    them)."""
    factors = factors or {}
    f = parse(formula)
    _no_slopes(f.fes)
    if f.fes and {"^".join(fe_components(x)) for x in f.fes} != {"^".join(fe_components(x)) for x in spec_fe}:
        raise ValueError(
            f"the session's FEs are fixed ({' + '.join(spec_fe)}); leave the FE part out of the formula"
        )

    def single(terms):
        out = []
        for t in terms:
            if t.cat is not None and t.label in factors:
                out.extend(factors[t.label])
                continue
            if t.cat is not None:
                raise NotImplementedError(f"{t.label}: i() needs the session manager (Manager.fit)")
            if len(t.conts) != 1:
                raise ValueError(
                    f"{t.label}: an interaction of demeaned columns is not the demeaned interaction; "
                    f'create the product from raw columns: add_variable("{"_x_".join(t.conts)}", '
                    f'"{" * ".join(t.conts)}")'
                )
            out.append(t.conts[0])
        return out

    return f, list(f.depvars), single(f.terms), single(f.endog), single(f.instruments)


def _no_slopes(fes) -> None:
    """Sessions do not take varying slopes yet (fit does: docs/design/slopes.md)."""
    bad = [x for x in fes if "[" in x]
    if bad:
        raise NotImplementedError(
            f"sessions do not take varying slopes (fe[x]) yet: {', '.join(bad)}; use dfeols.fit"
        )


def _cluster_names(cluster) -> list[str]:
    """One cluster, or two for two-way clustering ("a+b" or ["a", "b"]), each normalised."""
    if not cluster:
        return []
    parts = [p.strip() for p in cluster.split("+")] if isinstance(cluster, str) else list(cluster)
    out = ["^".join(fe_components(p)) for p in parts if p]
    if len(out) > 2:
        raise NotImplementedError(f"clustering on {len(out)} variables: one or two (two-way) only")
    if len(set(out)) < len(out):
        raise ValueError(f"cluster {cluster!r} names a variable twice")
    return out


def fit_model(
    d: Demeaned,
    formula: str,
    *,
    vcov=None,
    cluster=None,
    label: str | None = None,
    executor=None,
    workdir: Path,
    session: dict | None = None,
    collin_tol: float = 1e-9,
    log_every: float = 30.0,
    estimator: str = "2sls",
    fuller: float | None = None,
    factors: dict | None = None,
    cat_levels: dict | None = None,
) -> Result:
    """OLS of `formula` (y ~ x1 + x2, declared or created variables) on the demeaned data,
    or IV (`y ~ x | d ~ z1 + z2`: 2SLS, or LIML with `estimator="liml"`, Fuller with
    `fuller=`). Exact by Frisch-Waugh-Lovell: every column is residualised on all the FEs,
    so the cross-products come from the Gram matrix, as do b, iid SEs and (IV) the first
    stage; hetero and CRV1 (one or two clusters) take one pass for the meats."""
    from . import iv as _iv
    from . import record as _record

    executor = executor or SerialExecutor()
    t_start = time.time()
    cpu_start = time.process_time()
    m = d.manifest
    spec = m["spec"]
    f, ys, xs, endog, insts = _model_terms(formula, spec["fe"], factors)
    is_iv = bool(endog)
    if not is_iv and (estimator != "2sls" or fuller is not None):
        raise ValueError("estimator= and fuller= apply to IV formulas (y ~ x | d ~ z)")
    unknown = [v for v in [*ys, *xs, *endog, *insts] if v not in d.names]
    if unknown:
        raise KeyError(
            f"not declared or created in this session: {unknown} (declared: {', '.join(d.declared)}"
            + (f"; created: {', '.join(c['name'] for c in d.created)}" if d.created else "")
            + "); add_variable(name, expression) creates one from raw columns"
        )
    names_cl = _cluster_names(cluster)
    for c in names_cl:
        if c not in m["clusters"]:
            raise KeyError(
                f"cluster {c!r} was not declared for this session ({', '.join(m['clusters']) or 'none'})"
            )
    cluster = "+".join(names_cl) if names_cl else None
    if vcov is None:
        vcov = ["CRV1"] if cluster else ["iid"]
    vcov = tuple([vcov] if isinstance(vcov, str) else vcov)
    for v in vcov:
        if v not in VCOVS:
            raise ValueError(f"vcov {v!r} not supported ({', '.join(VCOVS)})")
        if v == "CRV1" and not cluster:
            raise ValueError("CRV1 needs cluster=")
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    pos = {n: i for i, n in enumerate(d.names)}
    Gm = d.gram
    N = int(m["rows"])
    yy = np.diag(Gm)[[pos[y] for y in ys]]
    raw = lambda names: None if d.raw_ss is None else d.raw_ss[[pos[n] for n in names]]

    # the model as linear maps of T, a block of the session's columns: each group (the
    # outcomes; IV: also the first stages) has residuals T @ A and score regressors T @ H
    sol = None
    if is_iv:
        Zn, Yn = [*xs, *insts], [*ys, *endog]
        iz, iyy = [pos[n] for n in Zn], [pos[n] for n in Yn]
        plan_iv = SimpleNamespace(
            formula=SimpleNamespace(depvars=tuple(ys)),
            endog_names=list(endog),
            colnames=list(xs),
            inst_names=list(insts),
            k_x=len(Zn),
            sparse_global=False,
            local_fes=[("fe", "fe")],  # every column is net of the FEs (first_stage's TSS)
            vcov_types=vcov,
        )
        acc_iv = {
            "ZtZ": Gm[np.ix_(iz, iz)],
            "ZtY": Gm[np.ix_(iz, iyy)],
            "YtY": Gm[np.ix_(iyy, iyy)],
            "ref": raw(Zn),
            "ref_y": raw(Yn),
        }
        sol = _iv.solve(plan_iv, acc_iv, N, estimator=estimator, fuller=fuller, collin_tol=collin_tol)
        T = [*Yn, *(Zn[i] for i in sol.lin["cols"])]
        groups = [("", sol.lin["A"], sol.lin["H"]), ("fs_", sol.lin["fs"]["A"], sol.lin["fs"]["H"])]
        coefnames, dropped = list(sol.coefnames), list(sol.dropped)
        breads, meat_idx = sol.bread, sol.meat_idx
        coef = sol.coef
    else:
        ix = [pos[x] for x in xs]
        keep = collinear_keep(Gm[np.ix_(ix, ix)], collin_tol, ref_diag=raw(xs)) if xs else np.zeros(0, bool)
        kept = [xs[i] for i in np.flatnonzero(keep)]
        dropped = [xs[i] for i in np.flatnonzero(~keep)]
        ik, iy = [pos[x] for x in kept], [pos[y] for y in ys]
        if kept:
            coef, bread = solve_spd(Gm[np.ix_(ik, ik)], Gm[np.ix_(ik, iy)])
        else:
            coef, bread = np.zeros((0, len(ys))), np.zeros((0, 0))
        T = [*ys, *kept]
        A = np.vstack([np.eye(len(ys)), -coef])
        H = np.vstack([np.zeros((len(ys), len(kept))), np.eye(len(kept))])
        groups = [("", A, H)]
        coefnames, breads, meat_idx = kept, [bread] * len(ys), np.arange(len(kept))
    it = [pos[n] for n in T]
    GT = Gm[np.ix_(it, it)]
    rss_gram = {gp: np.einsum("tj,tu,uj->j", A_, GT, A_) for gp, A_, _ in groups}
    k = len(coefnames)
    fe_levels = [int(fe["levels"]) for fe in m["fixed_effects"]]
    cls = [dict(m["clusters"][c], name=c) for c in names_cl]
    nested = [any(c["fe_nested"][i] for c in cls) for i in range(len(fe_levels))] if cls else []
    G = min(int(c["G"]) for c in cls) if cls else None  # two-way: the smaller (pyfixest, fixest)
    if len(cls) == 2:  # two-way: V_a + V_b - V_ab, ab the intersection
        a, b = cls
        cls.append(
            {
                "name": f"{a['name']}&{b['name']}",
                "pair": [a["name"], b["name"]],
                "levels_b": b["levels"],
                "levels": None,
                "local": a["local"] or b["local"],
            }
        )
    cl = cls if len(cls) > 1 else (cls[0] if cls else None)
    passes = 0
    meat_seconds = None
    rss_gb = 0.0
    meats = {gp: {"rss": rss_gram[gp], "hetero": None, "crv": None} for gp, *_ in groups}
    if any(v in ("hetero", "CRV1") for v in vcov) and groups[0][2].shape[1]:
        passes = 1
        accs = {gp: _MeatAcc(A_.shape[1], H_.shape[1], vcov, cls) for gp, A_, H_ in groups}
        peak = {"rss": 0.0}

        def on(tid, r):
            peak["rss"] = max(peak["rss"], r["timing"].get("maxrss_gb", 0.0))
            for gp, a_ in accs.items():
                a_.add(r, gp)

        # each shard's prepared rows (before singletons), so the progress line's rate is in rows
        src_rows = {sh["id"]: sh["rows"] for sh in Dataset.open(m["dataset"]["root"]).shards}
        rows = {sid: src_rows.get(sid, 1) for sid in d.shards}
        prog = Progress(wd, label=label or "model", every=log_every)
        prog.add_stage("meat", rows)
        prog.start("meat")
        maps = [(gp, A_, H_) for gp, A_, H_ in groups]
        tasks = [(sid, 1, (d.files(sid, T), T, maps, vcov, cl)) for sid in d.shards]
        t_meat = time.time()
        executor.run(model_meat, tasks, on, prog, "meat")
        meat_seconds = time.time() - t_meat
        prog.finish("meat", log=False)
        meats = {gp: a_.finish() for gp, a_ in accs.items()}
        rss_gb = peak["rss"]
    rss = meats[""]["rss"]
    out = {}
    dfk_plain = _df_k(k, fe_levels, None)
    dfk_cl = _df_k(k, fe_levels, nested)
    kk = groups[0][2].shape[1]
    for j, y in enumerate(ys):
        bread_j = breads[j]
        V = {}
        if "iid" in vcov:
            V["iid"] = (vcov_iid(bread_j, rss[j], N, dfk_plain), N - dfk_plain)
        if "hetero" in vcov:
            mh = meats[""]["hetero"][j] if meats[""]["hetero"] is not None else np.zeros((kk, kk))
            V["hetero"] = (vcov_hetero(bread_j, mh, N, dfk_plain), N - dfk_plain)
        if "CRV1" in vcov:
            mc = meats[""]["crv"][j] if meats[""]["crv"] is not None else np.zeros((kk, kk))
            V["CRV1"] = (vcov_crv1(bread_j, mc, N, dfk_cl, G), G - 1)
        V = {t: (v_[np.ix_(meat_idx, meat_idx)], df) for t, (v_, df) in V.items()}
        out[y] = {
            "coef": coef[:, j],
            "vcov": V,
            "rss": float(rss[j]),
            "tss": float(yy[j]),
            "r2_within": float(1 - rss[j] / yy[j]) if yy[j] > 0 else float("nan"),
        }
    iv_info: dict = {}
    if is_iv:
        overid = _iv.overid(sol, rss, N)
        for j, y in enumerate(ys):
            out[y]["kappa"] = float(sol.kappa[j])
            out[y]["overid"] = overid[j]
        acc2 = {"fs_hetero": meats["fs_"]["hetero"], "fs_crv": meats["fs_"]["crv"]}
        iv_info = {
            "estimator": "fuller" if fuller is not None else estimator,
            "fuller": fuller,
            "endogenous": list(endog),
            "instruments": list(insts),
            "first_stage": _iv.first_stage(plan_iv, sol, acc_iv, acc2, N, fe_levels, nested, G),
        }
    fe_names = [fe["name"] for fe in m["fixed_effects"]]
    text = f"{' + '.join(ys)} ~ {' + '.join(xs) if xs else '1'} | {' + '.join(spec['fe'])}"
    if is_iv:
        text += f" | {' + '.join(endog)} ~ {' + '.join(insts)}"
    options = {
        "cluster": cluster,
        "vcov": list(vcov),
        "weights": spec["weights"],
        "drop_singletons": spec["drop_singletons"],
        "fe_tol": spec["fe_tol"],
        "fe_maxiter": m["fe_maxiter"],
        "fe_method": spec["fe_method"],
        "crossing_fe": (m["crossing"] or {}).get("method"),
        "max_dummy_levels": spec["max_dummy_levels"],
        "collin_tol": collin_tol,
        "cache": False,
        **({"estimator": estimator, "fuller": fuller} if is_iv else {}),
    }
    if (m["crossing"] or {}).get("method") == "iterate":  # plan C: the record's numerics say CG
        options.update(
            iter_tol=spec["iter_tol"],
            iter_maxiter=spec["iter_maxiter"],
            level_space_gb=spec["level_space_gb"],
            crossing_precond=spec["crossing_precond"],
            crossing_cg=m["crossing"]["cg"],
        )
    ds = Dataset.open(m["dataset"]["root"])
    run = _record.run_info(
        ds,
        executor,
        started=t_start,
        finished=time.time(),
        cpu_seconds=(time.process_time() - cpu_start) if isinstance(executor, SerialExecutor) else None,
    )
    if session:
        run["provenance"]["execution"]["session"] = session["id"]
    run["provenance"]["data"].append(
        {
            "role": "derived",
            "name": f"demeaned/{m['key']}",
            "location": str(d.root),
            "rows": N,
            "columns": list(d.declared),
        }
    )
    used_cols = (*ys, *xs, *endog, *insts)
    for c in d.created:
        if c["name"] in used_cols:
            run["provenance"]["data"].append(
                {
                    "role": "derived",
                    "name": f"demeaned/{m['key']}/{c['id']}",
                    "location": str(d.root / c["dir"]),
                    "rows": N,
                    "columns": [c["name"]],
                }
            )
    src_rows = (ds.meta.get("source") or {}).get("rows")
    run["sample"] = {
        "rows_input": int(m["load"]["rows"]),
        "dropped": {k_: int(m["load"][k_]) for k_ in ("missing", "non_finite", "zero_weights")},
    }
    if src_rows is not None and src_rows >= ds.n_rows:
        run["sample"]["rows_input"] = int(src_rows)
        run["sample"]["dropped"] = {"missing_keys": int(src_rows - ds.n_rows), **run["sample"]["dropped"]}
    run["fit_id"] = str(uuid.uuid4())
    if m.get("row_key"):
        run["row_key"] = m["row_key"]
    run["converged"] = bool(m["converged"])
    run["fe_nested_in_cluster"] = nested
    if len(names_cl) > 1:  # two-way: each count, and which clusters each nested FE is in
        run["n_clusters"] = {c: int(m["clusters"][c]["G"]) for c in names_cl}
        run["fe_nested_in"] = {
            fe["name"]: [c for c in names_cl if m["clusters"][c]["fe_nested"][i]]
            for i, fe in enumerate(m["fixed_effects"])
        }
    run["n_local_fes"] = sum(1 for fe in m["fixed_effects"] if fe["absorbed"] == "local")
    run["absorption"] = {
        fe["name"]: {
            "method": fe["absorbed"],
            "nested_in_partition": fe["absorbed"] == "local",
        }
        for fe in m["fixed_effects"]
    }
    if spec["weights"]:
        run["sum_weights"] = float(m["sum_weights"])
    run["session"] = {
        **(session or {}),
        "demeaned_key": m["key"],
        "declared": list(d.declared),
        "passes": passes,
        "formula": formula,
    }
    used = {v: d.expression(v) for v in used_cols if d.expression(v) is not None}
    if used:  # in the order created, so a replay can make them again
        run["session"]["created"] = [
            {"name": c["name"], "expression": c["expression"], "id": c["id"]}
            for c in d.created
            if c["name"] in used
        ]
    res = Result(
        formula=text,
        label=label or f.fingerprint(cluster=cluster, vcov=list(vcov)),
        coefnames=coefnames,
        outcomes=out,
        N=N,
        N_dropped=int(m["singletons"]),
        G=G,
        cluster=cluster,
        fe=fe_names,
        fe_levels=fe_levels,
        collinear=dropped,
        vcov_default=vcov[0],
        seconds=time.time() - t_start,
        workdir=str(wd),
        timing={
            "passes": passes,
            **({"meat_seconds": meat_seconds} if passes else {}),
            "n_shards": len(d.shards),
            **({"max_worker_rss_gb": rss_gb} if passes else {}),
        },
        options=options,
        terms=_record.term_rows(f, cat_levels or {}, has_fe=True),
        run=run,
        iv=iv_info,
    )
    res.save(wd / "result.json")
    return res


def session_record(res: Result, y: str | None = None, **kwargs) -> dict:
    """A model's estimation record (unencoded; write it with estrec.write), saying
    it came from a session: the sample rule is the session's, the call replays through it."""
    from . import record as _record

    rec = _record.to_estimation(res, y, **kwargs)
    s = res.run.get("session") or {}
    declared = s.get("declared") or []
    made = "".join(  # i() columns (year::2005) are made by the fit itself
        f".add_variable({c['name']!r}, {c['expression']!r})"
        for c in s.get("created", [])
        if "::" not in c["name"]
    )
    model = s.get("formula") or res.formula.split(" | ")[0]
    call = f"dfeols.Session.attach({s.get('name', '')!r}){made}.fit({model!r}"
    o = res.options
    if o.get("cluster"):
        call += f", cluster={o['cluster']!r}"
    call += f", vcov={o['vcov']!r}"
    if o.get("estimator") and o["estimator"] != "2sls":
        call += f", estimator={o['estimator']!r}"
    if o.get("fuller") is not None:
        call += f", fuller={o['fuller']!r}"
    call += ")"
    rec["specification"]["call"] = call
    rec["specification"]["sample"]["restriction"] = (
        "rows complete on every variable declared for the session (" + ", ".join(declared) + ")"
    )
    ext = rec["extensions"]["dfeols"]
    ext["replay"]["call"] = call
    if s.get("created"):  # the created variables' definitions, on the session's raw columns
        ext["created_variables"] = [{"name": c["name"], "expression": c["expression"]} for c in s["created"]]
    rec.setdefault("meta", {}).setdefault("tags", []).append("session")
    return rec


# ---------------------------------------------------------------- the manager


def sessions_root(profile=None, root=None) -> Path:
    if root:
        return Path(root)
    if env := os.environ.get("DFEOLS_SESSIONS"):
        return Path(env)
    from .site import profile as load_profile

    return load_profile(profile).scratch_dir("sessions")


def demeaned_root(profile=None, root=None) -> Path:
    if root:
        return Path(root)
    if env := os.environ.get("DFEOLS_DEMEANED"):
        return Path(env)
    from .site import profile as load_profile

    return load_profile(profile).scratch_dir("demeaned")


class Manager:
    """Runs one session: demean once, then answer requests until stopped, idle or out of
    time. `serve()` blocks; Session.start(..., background=True) runs it in a thread."""

    def __init__(
        self,
        name: str,
        spec: Spec,
        *,
        root=None,
        demeaned=None,
        executor=None,
        hours: float | None = None,
        idle_minutes: float | None = 30.0,
        persist: bool = True,
        poll: float = 0.5,  # Ted 2026-11-09: sessions are for long work; 0.5 s is enough
        profile=None,
    ):
        if not _NAME.match(name):
            raise ValueError(f"session name must match {_NAME.pattern}: {name!r}")
        self.name = name
        self.spec = spec
        self.dir = sessions_root(profile, root) / name
        self.demeaned_dir = demeaned_root(profile, demeaned)
        self.executor = executor or SerialExecutor()
        self.hours = hours
        self.ends: float | None = None  # set when ready: the hours count from then
        self.idle = idle_minutes * 60 if idle_minutes else None
        self.persist = persist
        self.poll = poll
        self.id = str(uuid.uuid4())
        self.started = _now()
        self.t0 = time.time()
        self.status = "starting"
        self.models: list[dict] = []  # {n, label, dir, record ids}
        self.clients: dict[str, dict] = {}
        self.events: list[dict] = []
        self.last_request = time.time()
        self.lock: DirLock | None = None
        self.data: Demeaned | None = None
        self._stop = threading.Event()
        self.state_extra: dict[str, Any] = {}  # kept in session.json (estimates, progress file)
        # row-columns per second of a model's pass, measured in this session: the first pass
        # reads files no node has cached yet (OSCER 35828091: 17 s for m1, 6 s for m2 with more
        # columns), so it is kept apart from the later passes
        self.pass_rate: float | None = None
        self.first_pass_rate: float | None = None
        self.calibration: tuple[float, int] | None = None  # (row-columns/s read, shards read)
        self.progress_every = 2.0

    # ---- state on disk
    def _state(self, **extra) -> dict:
        return {
            "format": SESSION_FORMAT,
            "name": self.name,
            "id": self.id,
            "status": self.status,
            "started": self.started,
            "ends": _dt.datetime.fromtimestamp(self.ends, _dt.timezone.utc).isoformat()
            if self.ends
            else None,
            "idle_minutes": self.idle / 60 if self.idle else None,
            "heartbeat": _now(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "job": os.environ.get("SLURM_JOB_ID"),
            "spec": dataclasses.asdict(self.spec),
            "demeaned": str(self.data.root) if self.data else None,
            "models": len(self.models),
            "workers": self._worker_counts(),
            "created": [{"name": c["name"], "expression": c["expression"]} for c in self.data.created]
            if self.data
            else [],
            **self.state_extra,
            **extra,
        }

    def _save_state(self, **extra) -> None:
        _write_json(self.dir / "session.json", self._state(**extra))

    def event(self, type: str, *, member: str | None = None, detail: dict | None = None, message=None):
        ev = {"when": _now(), "type": type}
        if member:
            ev["member"] = member
        if detail:
            ev["detail"] = json.loads(json.dumps(detail, default=_jsonable))
        if message:
            ev["message"] = message
        self.events.append(ev)
        _append_jsonl(self.dir / "events.jsonl", ev)
        return ev

    # ---- lifecycle
    def _archive_previous(self) -> None:
        """A session folder left by a manager that is gone (we hold the lock now): close its
        collection as crashed and move its files to previous/<id>/."""
        st = self.dir / "session.json"
        if not st.exists():
            return
        old = _read_json(st)
        prev = self.dir / "previous" / old.get("id", uuid.uuid4().hex)
        prev.mkdir(parents=True, exist_ok=True)
        if old.get("status") not in ("closed", "timed_out", "crashed", "failed"):
            try:
                _rewrite_collection(self.dir, old, status="crashed")
            except Exception:  # what it left may be partial; keep the files regardless
                log.exception("session %s: could not close the previous run's collection", self.name)
        for p in list(self.dir.iterdir()):
            if p.name != "previous":
                shutil.move(str(p), prev / p.name)

    def start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = DirLock(self.dir, what="session").acquire()
        self._archive_previous()
        for sub in ("requests", "responses", "models", "results"):
            (self.dir / sub).mkdir(exist_ok=True)
        self._save_state()
        try:
            self._load_data()
        except Exception as e:
            self.status = "failed"
            self.event("error", message=f"{type(e).__name__}: {e}")
            self._save_state(error=f"{type(e).__name__}: {e}")
            self.lock.release()
            raise
        self.status = "ready"
        if self.hours:
            self.ends = time.time() + self.hours * 3600
        self._save_state()
        log.info("session %s: ready (%s)", self.name, self.id)

    def _load_data(self) -> None:
        ds = Dataset.open(self.spec.data)
        _session_plan(self.spec, ds, None)  # check the spec before any work
        key = self.spec.key(ds)
        self.event(
            "data_loaded",
            detail={"dataset": str(ds.root), "rows": ds.n_rows, "shards": len(ds.shards), "key": key},
        )
        where = self.demeaned_dir / ds.root.name / key if self.persist else self.dir / "demeaned"
        if (where / "manifest.json").exists():
            self.data = Demeaned.open(where)
            self.event(
                "demeaned_loaded", detail={"key": key, "location": str(where), "bytes": self.data.bytes()}
            )
            return
        where.parent.mkdir(parents=True, exist_ok=True)
        with DirLock(where, what="demeaned data"):
            if (where / "manifest.json").exists():  # another session built it while we waited
                self.data = Demeaned.open(where)
                self.event("demeaned_loaded", detail={"key": key, "location": str(where)})
                return
            shutil.rmtree(where, ignore_errors=True)  # an interrupted build
            self._check_space(ds, where)
            est = estimate_seconds(self.demeaned_dir, ds.n_rows, len(self.spec.variables))
            self.state_extra.update(
                demean_estimate_seconds=list(est) if est else None,
                progress="progress/demean/status.json",
            )
            self._save_state()
            self.data = demean_once(
                self.spec,
                ds,
                where,
                self.executor,
                label=f"session {self.name}",
                log_every=self.progress_every,
                progress_dir=self.dir / "progress" / "demean",
            )
            self.state_extra.pop("progress", None)
        self.event(
            "demeaned_persisted" if self.persist else "demeaned",
            detail={
                "key": key,
                "location": str(where),
                "bytes": self.data.bytes(),
                "seconds": self.data.manifest["seconds"],
                "estimate_seconds": self.state_extra.get("demean_estimate_seconds"),
                "rows": self.data.manifest["rows"],
                "variables": self.data.names,
            },
        )
        _log_timing(self.demeaned_dir, self.data.manifest, self.data.bytes())

    def _check_space(self, ds: Dataset, where: Path) -> None:
        """Stop before demeaning when even the low end of the size estimate does not fit."""
        from .store import fmt_bytes, space

        se = size_estimate(
            self.demeaned_dir, ds.n_rows, _demeaned_columns(len(self.spec.variables), len(self.spec.clusters))
        )
        sp = space(where)
        if se and sp and se[0] > sp["free"]:
            raise RuntimeError(
                f"not enough scratch for the demeaned data: at least {fmt_bytes(se[0])}, "
                f"{fmt_bytes(sp['free'])} free in {sp['path']} (dfeols demeaned list / rm frees space)"
            )

    def serve(self) -> str:
        """Start (if not started), answer requests until the session ends; returns the end
        status (closed, timed_out)."""
        if self.status == "starting":
            self.start()
        end = "closed"
        try:
            beat = 0.0
            while not self._stop.is_set():
                handled = self._poll_once()
                now = time.time()
                if now - beat > 10:
                    self._save_state()
                    beat = now
                if self.ends and now > self.ends:
                    end = "timed_out"
                    self.event("note", message="walltime reached")
                    break
                if self.idle and now - self.last_request > self.idle:
                    end = "closed"  # Ted 2026-11-02: an idle end is a normal close
                    self.event("note", message=f"idle for {self.idle / 60:g} minutes")
                    break
                if not handled:
                    self._stop.wait(self.poll)
        finally:
            self.close(end)
        return end

    def stop(self) -> None:
        self._stop.set()

    def persist_data(self) -> dict:
        """Keep the demeaned data after the session (for one started with persist=False):
        move it from the session folder to <demeaned>/<dataset>/<key>/, where later sessions
        with the same key load it. Requests are handled one at a time, so no model is reading
        it during the move. Records written before keep the old location; the event says
        where it went."""
        if self.data is None:
            raise RuntimeError("no demeaned data yet")
        m = self.data.manifest
        if self.persist:
            return {"location": str(self.data.root), "already": True}
        key = m["key"]
        target = self.demeaned_dir / m["dataset"]["name"] / key
        target.parent.mkdir(parents=True, exist_ok=True)
        src = self.data.root
        created = list(self.data.created)
        with DirLock(target, what="demeaned data"):
            if (target / "manifest.json").exists():  # the same key was built by another session
                for dir_ in dict.fromkeys(c["dir"] for c in created):  # this session's batches,
                    if not (target / dir_ / "variable.json").exists():  # unless kept there too
                        (target / dir_).parent.mkdir(parents=True, exist_ok=True)
                        shutil.rmtree(target / dir_, ignore_errors=True)
                        shutil.move(str(src / dir_), str(target / dir_))
                old = self.data
                self.data = Demeaned.open(target)
                shutil.rmtree(src, ignore_errors=True)
                moved = False
            else:
                shutil.rmtree(target, ignore_errors=True)  # an interrupted build
                shutil.move(str(src), str(target))
                old = self.data
                self.data = Demeaned.open(target)
                moved = True
        for c in old.created:  # the same columns, now under target
            self.data.add_created(c["name"], c["info"], old_cross(old, c), c["expression"], c["dir"])
        self.persist = True
        detail = {"key": key, "location": str(target), "from": str(src), "bytes": self.data.bytes()}
        if not moved:
            detail["note"] = "already persisted by another session; this session's copy deleted"
        self.event("demeaned_persisted", detail=detail)
        self._save_state()
        return {"location": str(target), "already": False, "bytes": detail["bytes"]}

    def close(self, status: str = "closed") -> None:
        if self.status in ("closed", "timed_out", "crashed"):
            return
        self.status = status
        self.write_collection()
        if not self.persist and self.data is not None:
            shutil.rmtree(self.data.root, ignore_errors=True)
        self._save_state(ended=_now())
        if self.lock:
            self.lock.release()
        log.info(
            "session %s: %s after %s, %d models",
            self.name,
            status,
            fmt_td(time.time() - self.t0),
            len(self.models),
        )

    # ---- requests
    def _poll_once(self) -> bool:
        reqs = sorted((self.dir / "requests").glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name))
        for p in reqs:
            try:
                req = _read_json(p)
            except (OSError, json.JSONDecodeError):
                continue  # half-written by a client that didn't rename; next poll
            p.unlink(missing_ok=True)
            self.last_request = time.time()
            received = _now()
            rid = req.get("id") or p.stem
            try:
                resp = self.handle(req)
                resp = {"id": rid, "ok": True, **resp}
            except Exception as e:
                log.exception("session %s: request %s failed", self.name, rid)
                self.event(
                    "error",
                    message=f"{type(e).__name__}: {e}",
                    detail={"request": rid, "type": req.get("type")},
                )
                resp = {"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"}
            resp["sent"], resp["received"], resp["finished"] = req.get("sent"), received, _now()
            _write_json(self.dir / "responses" / f"{rid}.json", resp)
            if self._stop.is_set():
                break
        return bool(reqs)

    def handle(self, req: dict) -> dict:
        t = req.get("type")
        if t == "fit":
            return self.fit(req)
        if t == "refit":
            return self.refit(req)
        if t == "add_variable":
            return self.add_variable(req)
        if t == "add_variables":
            return self.add_variables(req)
        if t == "attach":
            cid = req.get("client") or str(uuid.uuid4())
            c = {
                "language": req.get("language", "python"),
                "version": req.get("version"),
                "host": req.get("host"),
                "connected": _now(),
            }
            self.clients[cid] = {k: v for k, v in c.items() if v is not None}
            self.event("client_connected", detail={"client": cid, "language": c["language"]})
            return {"client": cid, "session": self._state()}
        if t == "detach":
            cid = req.get("client")
            if cid in self.clients:
                self.clients[cid]["disconnected"] = _now()
                self.event("client_disconnected", detail={"client": cid})
            return {}
        if t == "status":
            return {"session": self._state()}
        if t == "export":
            if req.get("models"):
                return self.export_table(req)
            p = self.write_collection(req.get("path"))
            return {"path": str(p)}
        if t == "stop":
            self.stop()
            return {}
        if t == "persist":
            return self.persist_data()
        raise ValueError(f"unknown request type {t!r}")

    def fit(self, req: dict) -> dict:
        from estrec import write as ef_write

        n = len(self.models) + 1
        label = req.get("label") or f"m{n}"
        if not _NAME.match(label):
            raise ValueError(f"label must match {_NAME.pattern}: {label!r}")
        wd = self.dir / "models" / f"{n:03d}-{label}"
        factors, cat_levels = self._factors(req["formula"])
        req = {**req, "factors": factors}
        est = self.estimate(req)
        if req.get("id"):
            _write_json(
                self.dir / "responses" / f"{req['id']}.accepted.json",
                {
                    "model_dir": str(wd.relative_to(self.dir)),
                    **est,
                    "workers": self._worker_counts(),
                    "accepted": _now(),
                },
            )
        res = fit_model(
            self.data,
            req["formula"],
            vcov=req.get("vcov"),
            cluster=req.get("cluster"),
            label=label,
            executor=self.executor,
            workdir=wd,
            session={"id": self.id, "name": self.name},
            log_every=self.progress_every,
            estimator=req.get("estimator") or "2sls",
            fuller=req.get("fuller"),
            factors=factors,
            cat_levels=cat_levels,
        )
        if res.timing.get("meat_seconds"):
            self._log_pass(est["columns"], res.timing["meat_seconds"])
        recs = {}
        for y in res.outcomes:
            rec = session_record(res, y)
            name = "result.estrec.json" if len(res.outcomes) == 1 else f"result.{y}.estrec.json"
            rec = ef_write(
                rec,
                wd / name,
                by={"name": "dfeols", "version": _version()},
                level=_record_level(None, self.executor),
            )
            recs[y] = {"id": rec["meta"]["id"], "path": str((wd / name).relative_to(self.dir))}
        entry = {
            "n": n,
            "label": label,
            "formula": res.formula,
            "vcov": res.options["vcov"],
            "cluster": res.cluster,
            "dir": str(wd.relative_to(self.dir)),
            "records": recs,
            "seconds": res.seconds,
            "passes": res.timing["passes"],
            "estimate_seconds": est["estimate_seconds"],
            "finished": _now(),
        }
        if req.get("refit_of"):
            entry["refit_of"] = req["refit_of"]
        if res.iv:
            entry["estimator"] = res.options.get("estimator")
            entry["fuller"] = res.options.get("fuller")
        self.models.append(entry)
        _append_jsonl(self.dir / "results" / "index.jsonl", entry)
        for y, r in recs.items():
            self.event(
                "fit",
                member=r["id"],
                detail={
                    "label": label,
                    "formula": req["formula"],
                    "response": y,
                    "vcov": res.options["vcov"],
                    "cluster": res.cluster,
                    "seconds": res.seconds,
                    "passes": res.timing["passes"],
                    **(
                        {"estimator": res.options.get("estimator"), "fuller": res.options.get("fuller")}
                        if res.iv
                        else {}
                    ),
                },
            )
        self._save_state()
        return {"model": entry}

    def add_variable(self, req: dict) -> dict:
        """Create a variable from raw columns: computed on the session's rows, demeaned on its
        FEs, and kept in <demeaned>/created/ (a later session with the same data and
        expression reuses it). Asking again for the same name and expression does nothing."""
        r = self.add_variables({"id": req.get("id"), "variables": {req.get("name"): req.get("expression")}})
        return {"variable": r["variables"][0]}

    def add_variables(self, req: dict, *, internal: bool = False) -> dict:
        """Several created variables ({name: expression}), the new ones demeaned together in
        one batch (one demeaning however many: the i() levels of a factor, say), each column
        kept under its own digest so any of them is reused alone later. `internal` names
        (`year::2005`, from i()) skip the column-name rule."""
        from . import expr as _expr

        items = list((req.get("variables") or {}).items())
        if not items:
            raise ValueError("no variables to create")
        d = self.data
        ds = Dataset.open(self.spec.data)
        raw = {c for c in ds.columns if not c.startswith("__")}
        todo, out = [], {}
        seen: dict[str, str] = {}
        for name, text in items:
            if not isinstance(name, str) or (
                not internal and (not re.match(r"^[A-Za-z_]\w*$", name) or name.startswith("__"))
            ):
                raise ValueError(f"a created variable's name must be a column name: {name!r}")
            digest = _expr.digest(text)
            if digest in seen and seen[digest] != name:
                raise ValueError(f"{text!r} is asked for twice, as {seen[digest]} and {name}")
            seen[digest] = name
            hit = None
            for c in d.created:
                if c["name"] == name and c["info"]["digest"] == digest:
                    hit = c
                elif c["name"] == name:
                    raise ValueError(f"{name} is already created in this session, as {c['expression']!r}")
                elif c["info"]["digest"] == digest:
                    raise ValueError(f"{text!r} is already created in this session, as {c['name']}")
            if hit is not None:
                out[name] = self._var_entry(hit, reused=True, already=True)
                continue
            if name in d.declared:
                raise ValueError(f"{name} is declared for this session; created variables need a new name")
            unknown = [c for c in _expr.columns(text) if c not in raw]
            if unknown:
                raise KeyError(f"{text!r}: not columns of the prepared data: {unknown}")
            _expr.compile(text)  # every error in the expression before any work
            todo.append((name, text, digest))
        t0 = time.time()
        index = d.created_index()
        build = [(n, t, g) for n, t, g in todo if g not in index]
        built = set()
        if build:
            batch = hashlib.sha256(" ".join(sorted(g for *_, g in build)).encode()).hexdigest()[:16]
            where = d.root / "created" / batch
            where.parent.mkdir(parents=True, exist_ok=True)
            with DirLock(where, what="created variables"):
                index = d.created_index()  # another session may have built them meanwhile
                build = [(n, t, g) for n, t, g in build if g not in index]
                if build:
                    shutil.rmtree(where, ignore_errors=True)  # an interrupted build
                    what = ", ".join(f"{n} = {t}" for n, t, _ in build[:3]) + (
                        f" and {len(build) - 3} more" if len(build) > 3 else ""
                    )
                    prog_dir = self.dir / "progress" / f"create-{build[0][2]}"
                    if req.get("id"):
                        _write_json(
                            self.dir / "responses" / f"{req['id']}.accepted.json",
                            {
                                "model_dir": str(prog_dir.relative_to(self.dir)),
                                **self._create_estimate(len(build)),
                                "workers": self._worker_counts(),
                                "what": f"creating {what}",
                                "accepted": _now(),
                            },
                        )
                    info = demean_once(
                        self.spec,
                        ds,
                        where,
                        self.executor,
                        label=f"session {self.name}: {build[0][0]}"
                        + (f" (+{len(build) - 1})" if len(build) > 1 else ""),
                        log_every=self.progress_every,
                        progress_dir=prog_dir,
                        create=[t for _, t, _ in build],
                        base=d,
                    )
                    _write_json(where / "variable.json", info)
                    built = {g for *_, g in build}
            index = d.created_index()
        for name, text, digest in todo:
            dir_, info = index[digest]
            cross = d.known_cross(info)
            missing = [n for n, i in zip(d.names, d.ids) if i not in cross]
            if missing:  # reused, and crossed with some of this session's columns nowhere on file
                new = [(str(d.root / dir_), [(info["column"], name)])]
                groups = d.groups(missing)
                order = [nm for _, cols in groups for _, nm in cols]
                x = cross_pass(d, groups, new, self.executor, self.dir / "progress" / f"cross-{digest}")[0]
                cross.update(zip([d.ids[d.names.index(n)] for n in order], map(float, x)))
            d.add_created(name, info, cross, text, dir_)
            entry = self._var_entry(d.created[-1], reused=digest not in built, seconds=time.time() - t0)
            self.event("variable_created", detail=entry)
            out[name] = entry
        self._save_state()
        return {"variables": [out[n] for n, _ in items]}

    def _factors(self, formula: str) -> tuple[dict, dict]:
        """A formula's i() terms as created indicator columns, made (in one batch) if this
        session has not made them: `i(year)` is year == level for each level but the first,
        `i(year, ref=2000)` each level but 2000, named `year::level` as fit names them. The
        levels are the prepared data's (fit's `category_levels`, cached with the shards); a
        level absent from the sample gives an all-zero column, which is dropped as collinear."""
        from .estimate import category_levels

        f = parse(formula)
        cats = [t for t in (*f.terms, *f.endog, *f.instruments) if t.cat is not None]
        if not cats:
            return {}, {}
        for t in cats:
            if t.conts:
                raise NotImplementedError(
                    f"{t.label}: i() interacted with a variable in sessions; create the products with "
                    f"add_variable (e.g. '{t.cat}_x_{t.conts[0]}' as '({t.cat} == level) * {t.conts[0]}')"
                )
        ds = Dataset.open(self.spec.data)
        missing = sorted({t.cat for t in cats} - set(ds.columns))
        if missing:
            raise KeyError(f"i(): not columns of the prepared data: {missing}")
        levels = category_levels(ds, sorted({t.cat for t in cats}), self.executor)
        factors, want = {}, {}
        for t in cats:
            levs = list(levels[t.cat])
            bad = [l for l in levs if isinstance(l, bool) or not isinstance(l, int | float)]
            if bad:
                raise NotImplementedError(
                    f"{t.label}: i() in sessions takes numeric variables ({t.cat} has {bad[0]!r})"
                )
            if t.ref is not None:
                if t.ref not in levs:
                    raise ValueError(f"{t.label}: {t.ref!r} is not a level of {t.cat}")
                keep = [l for l in levs if l != t.ref]
            else:
                keep = levs[1:] if t.drop_first else levs
            names = [f"{t.cat}::{l}" for l in keep]
            factors[t.label] = names
            for nm, l in zip(names, keep):
                if nm not in self.data.names:
                    want[nm] = f"{t.cat} == {l!r}"
        if want:
            self.add_variables({"variables": want}, internal=True)
        return factors, {t.cat: list(levels[t.cat]) for t in cats}

    def _var_entry(self, c: dict, **kw) -> dict:
        info = c["info"]
        return {
            "name": c["name"],
            "expression": c["expression"],
            "columns": info["columns"],
            "digest": info["digest"],
            "location": str(self.data.root / c["dir"]),
            "rows": info["rows"],
            **kw,
        }

    def _create_estimate(self, columns: int = 1) -> dict:
        """Creating variables demeans their columns: about the session's demeaning per
        variable, up to all of it (the FEs' work, CG on crossing FEs, does not shrink with
        the columns)."""
        m = self.data.manifest
        sec = float(m.get("seconds") or 0.0)
        lo = min(sec * columns / max(len(self.data.declared), 1), sec)
        return {
            "passes": None,
            "columns": columns,
            "estimate_seconds": sec,
            "estimate_range": [lo, sec],
            "basis": f"demeaning {len(self.data.declared)} variables took {_secs(sec)}",
        }

    def refit(self, req: dict) -> dict:
        """An earlier model with another vcov or cluster: the coefficients again from the
        Gram matrix, and one pass for the new meat (none for iid)."""
        ref = req.get("model")
        old = next((m for m in reversed(self.models) if m["label"] == ref or str(m["n"]) == str(ref)), None)
        if old is None:
            raise KeyError(f"no model {ref!r} in this session ({', '.join(m['label'] for m in self.models)})")
        vcov = req.get("vcov") or old["vcov"]
        vcov = [vcov] if isinstance(vcov, str) else list(vcov)
        cluster = req.get("cluster")
        if cluster is None and "CRV1" in vcov:
            cluster = old["cluster"]
        if vcov == list(old["vcov"]) and cluster == old["cluster"]:
            raise ValueError(f"{old['label']} already has vcov {vcov} and cluster {cluster!r}")
        label = req.get("label") or f"{old['label']}.{'-'.join(v.lower() for v in vcov)}"
        if cluster and cluster != old["cluster"]:
            label = req.get("label") or f"{label}.{re.sub(r'[^A-Za-z0-9._-]', '_', cluster)}"
        self.event(
            "setting_changed",
            member=next(iter(old["records"].values()))["id"] if old["records"] else None,
            detail={
                "model": old["label"],
                "new_label": label,
                "vcov": {"from": old["vcov"], "to": vcov},
                "cluster": {"from": old["cluster"], "to": cluster},
            },
        )
        return self.fit(
            {
                "id": req.get("id"),
                "formula": old["formula"],
                "vcov": vcov,
                "cluster": cluster,
                "label": label,
                "refit_of": old["label"],
                "estimator": old.get("estimator"),
                "fuller": old.get("fuller"),
            }
        )

    def estimate(self, req: dict) -> dict:
        """A model's cost before it runs: rows x the columns its pass reads, at a rate from
        (in order) this session's later passes, its first pass, the first passes of earlier
        sessions on these demeaned data's file system, or a read of a few shard files. The
        last three give a range; `estimate_seconds` is its upper end."""
        _, ys, xs, endog, insts = _model_terms(req["formula"], self.spec.fe, req.get("factors"))
        xs, ys = [*xs, *insts], [*ys, *endog]  # IV's pass reads the instruments and endogenous
        vcov = req.get("vcov") or (["CRV1"] if req.get("cluster") else ["iid"])
        vcov = [vcov] if isinstance(vcov, str) else list(vcov)
        passes = int(any(v in ("hetero", "CRV1") for v in vcov) and bool(xs))
        # the columns model_meat reads: regressors, responses, the weight, the cluster codes
        cols = len(xs) + len(ys) + 1 + len(_cluster_names(req.get("cluster")))
        work = self.data.manifest["rows"] * cols
        out = {"passes": passes, "columns": cols, "estimate_seconds": 0.0, "estimate_range": None}
        if not passes:
            return {**out, "basis": "no pass"}
        if self.pass_rate:
            return {**out, "estimate_seconds": work / self.pass_rate, "basis": "measured in this session"}
        lo = self._parallel_read_seconds(work)
        if self.first_pass_rate:
            hi = work / self.first_pass_rate
            basis = "this session's first pass; later passes are usually faster"
        else:
            hist = pass_history(self.demeaned_dir)
            if hist:
                lo, hi = work / max(hist), work / min(hist)
                basis = f"first pass of {len(hist)} earlier session{'s' if len(hist) > 1 else ''}"
            else:
                rate, k = self.calibration or self._calibrate()
                hi = work / rate + TASK_OVERHEAD * len(self.data.shards)  # one reader, no overlap
                slots = self._workers()
                basis = (
                    f"read rate of {k} shard{'s' if k > 1 else ''}, 1 to {slots} "
                    f"worker{'s' if slots > 1 else ''}; first pass, no earlier sessions"
                )
        lo = min(lo, hi)
        return {**out, "estimate_seconds": hi, "estimate_range": [lo, hi], "basis": basis}

    def _worker_counts(self) -> dict | None:
        """Workers running and requested (None on one process): shown at ready and with each
        estimate, since a pass waits for the workers' jobs to start (Ted 2026-11-09)."""
        try:
            running, most = self.executor.slots()
        except Exception:
            return None
        if not most or most <= 1:
            return None
        return {"running": int(running or 0), "requested": int(most)}

    def _workers(self) -> int:
        running, most = self.executor.slots()
        return max(int(running or most or 1), 1)

    def _parallel_read_seconds(self, work: float) -> float:
        """The pass at the calibrated read rate on every running worker at once: a lower
        bound, as the workers share the file system."""
        rate, _ = self.calibration or self._calibrate()
        slots = self._workers()
        return work / (rate * slots) + TASK_OVERHEAD * len(self.data.shards) / slots

    def _log_pass(self, columns: int, seconds: float) -> None:
        rate = self.data.manifest["rows"] * columns / max(seconds, 1e-6)
        first = self.first_pass_rate is None and self.pass_rate is None
        if first:
            self.first_pass_rate = rate
        else:
            self.pass_rate = rate
        try:
            _append_jsonl(
                self.demeaned_dir / "passes.jsonl",
                {
                    "when": _now(),
                    "rows": self.data.manifest["rows"],
                    "columns": columns,
                    "seconds": seconds,
                    "workers": self._workers(),
                    "first": first,
                },
            )
        except OSError:
            pass

    def _calibrate(self, k: int = 3) -> tuple[float, int]:
        """Row-columns read per second by one process, from the largest few shard files
        (a model's pass is mostly reading its columns)."""
        d = self.data
        parts = sorted(d.shards, key=lambda s: -Path(d.part(s)).stat().st_size)[:k]
        cols = [*d.declared, WEIGHT_COL]
        t0 = time.perf_counter()
        n = sum(pl.read_parquet(d.part(s), columns=cols).height for s in parts)
        self.calibration = (max(n * len(cols), 1) / max(time.perf_counter() - t0, 1e-6), len(parts))
        return self.calibration

    # ---- the session collection
    def session_block(self) -> dict:
        m = self.data.manifest if self.data else {}
        ended = None if self.status in ("starting", "ready") else _now()
        block = {
            "id": self.id,
            "started": self.started,
            "status": "open" if self.status in ("starting", "ready") else self.status,
            "elapsed_seconds": time.time() - self.t0,
            "clients": list(self.clients.values()),
            "fixed": {
                "name": self.name,
                "fixed_effects": list(self.spec.fe),
                "weights": self.spec.weights,
                "declared_variables": list(self.spec.variables),
                "clusters": list(self.spec.clusters),
                "sample": "rows complete on every declared variable, singletons over all FEs dropped"
                if self.spec.drop_singletons
                else "rows complete on every declared variable",
                "drop_singletons": self.spec.drop_singletons,
                "fe_tolerance": self.spec.fe_tol,
                "demeaned_key": m.get("key"),
            },
        }
        if ended:
            block["ended"] = ended
        return block

    def provenance_block(self) -> dict:
        from .record import _environment, _scheduler, _software

        execution: dict[str, Any] = {
            "started": self.started,
            "elapsed_seconds": time.time() - self.t0,
            "idle_seconds": time.time() - self.last_request,
            "invocation": {"command": " ".join(sys.argv)},
        }
        if self.ends:
            execution["walltime_requested_seconds"] = self.ends - self.t0
        if sched := _scheduler():
            execution["scheduler"] = sched
        data = []
        if self.data:
            dm = self.data.manifest
            data.append(
                {
                    "role": "estimation",
                    "name": dm["dataset"]["name"],
                    "location": dm["dataset"]["root"],
                    "rows": dm["dataset"]["rows"],
                    "sha256": dm["dataset"]["meta_sha256"],
                }
            )
            data.append(
                {
                    "role": "derived",
                    "name": f"demeaned/{dm['key']}",
                    "location": str(self.data.root),
                    "rows": dm["rows"],
                    "columns": list(self.data.names),
                    "bytes": self.data.bytes(),
                }
            )
        return {"software": _software(), "environment": _environment(), "execution": execution, "data": data}

    def export_table(self, req: dict) -> dict:
        """A table of some models (labels or numbers) through the estrec table tool: the
        rendered file (format from its suffix), its table spec `<stem>.estrec-table.json`
        and the session collection `<stem>.estrec-collection.json` beside it, which the
        spec reads its records from, so the three travel together."""
        from estrec import read_collection
        from estrec.table import new_table, render_table, write_table

        out = Path(req.get("path") or self.dir / "exports" / "table.md")
        if not out.is_absolute():
            out = self.dir / "exports" / out
        fmt = EXPORT_FORMATS.get(out.suffix.lower())
        if fmt is None:
            raise ValueError(
                f"export to {', '.join(EXPORT_FORMATS)} (from the suffix), not {out.suffix or 'no suffix'!r}"
                + ("; .doc is a Word document (HTML Word opens)" if out.suffix.lower() == ".docx" else "")
            )
        ids, heads = [], []
        for ref in req["models"]:
            m = next((m for m in self.models if m["label"] == ref or str(m["n"]) == str(ref)), None)
            if m is None:
                raise KeyError(f"no model {ref!r} in this session")
            for y, r in m["records"].items():
                ids.append(r["id"])
                heads.append(m["label"] if len(m["records"]) == 1 else f"{m['label']}.{y}")
        out.parent.mkdir(parents=True, exist_ok=True)
        stem = out.name[: -len(out.suffix)] if out.suffix else out.name
        coll_path = self.write_collection(out.with_name(f"{stem}.estrec-collection.json"))
        by_id = {mm["record"]["meta"]["id"]: mm["record"] for mm in read_collection(coll_path)["members"]}
        opts = dict(req.get("table") or {})
        for k in ("stats", "rows", "keep", "drop"):  # R's jsonlite sends a 1-vector as a scalar
            if isinstance(opts.get(k), str | dict):
                opts[k] = [opts[k]]
        opts.setdefault("stats", ["nobs", "r2_within", {"vcov": "type"}])
        spec = new_table(
            [by_id[i] for i in ids],
            headers=req.get("headers") or heads,
            title=req.get("title"),
            sources=[{"id": "session", "path": coll_path.name}],
            **opts,
        )
        for col in spec["columns"]:
            col["source"] = "session"
        spec_path = write_table(spec, out.with_name(f"{stem}.estrec-table.json"))
        out.write_text(render_table(spec_path, fmt))
        self.event(
            "note",
            message=f"exported {len(ids)} column{'s' * (len(ids) > 1)} to {out}",
            detail={"path": str(out), "models": list(req["models"])},
        )
        return {"path": str(out), "table": str(spec_path), "collection": str(coll_path)}

    def write_collection(self, path=None) -> Path:
        path = Path(path) if path else self.dir / COLLECTION_NAME
        _write_session_collection(
            self.dir,
            self.models,
            self.events,
            self.session_block(),
            self.provenance_block(),
            path,
            level=_record_level(None, self.executor) or _default_level(),
        )
        return path


def old_cross(d: Demeaned, c: dict) -> dict[str, float]:
    """Created column c's cross-products with the columns before it in d."""
    j = d.ids.index(c["id"])
    return dict(zip(d.ids[:j], map(float, d.gram[j, :j])))


def cross_pass(d: Demeaned, name_groups, new_groups, executor, progress_dir: Path) -> np.ndarray:
    """One pass: the weighted cross-products of the columns in `new_groups` with those in
    `name_groups` (each as Demeaned.groups gives them)."""
    acc: dict[str, Any] = {"x": None}

    def on(tid, r):
        acc["x"] = r["x"] if acc["x"] is None else acc["x"] + r["x"]

    prog = Progress(progress_dir, label="cross-products", every=30.0)
    prog.add_stage("cross", {sid: 1 for sid in d.shards})
    prog.start("cross")
    tasks = [(sid, 1, (sid, str(d.root), new_groups, name_groups)) for sid in d.shards]
    executor.run(_cross_task, tasks, on, prog, "cross")
    prog.finish("cross", log=False)
    return acc["x"]


def _cross_task(sid: int, w_root: str, new_groups, name_groups) -> dict:
    t0 = time.perf_counter()
    w = pl.read_parquet(_part_path(Path(w_root), sid), columns=[WEIGHT_COL])[WEIGHT_COL].to_numpy()
    A = _read_columns(new_groups, sid, len(w))
    B = _read_columns(name_groups, sid, len(w))
    return {"x": (A * w[:, None]).T @ B, "timing": {"total": time.perf_counter() - t0}}


def _version() -> str:
    from . import __version__

    return __version__


def _record_level(level, executor):
    from .record import record_level

    return record_level(level, executor)


def _default_level() -> int:
    import estrec as ef

    return ef.DEFAULT_LEVEL


def _write_session_collection(
    sdir: Path,
    models,
    events,
    session: dict,
    provenance: dict | None,
    path: Path,
    level: int | None = None,
):
    """The session's collection, at `level` (the level its records were written at; when
    None, the highest level the records declare, so nothing is cut)."""
    import estrec as ef

    recs, labels = [], []
    for mdl in models:
        for y, r in mdl["records"].items():
            recs.append(ef.read(sdir / r["path"], nonfinite="string"))
            labels.append(mdl["label"] if len(mdl["records"]) == 1 else f"{mdl['label']}.{y}")
    if level is None:
        declared = [(r.get("meta") or {}).get("disclosure", {}).get("level") for r in recs]
        level = max([d for d in declared if isinstance(d, int)], default=ef.DEFAULT_LEVEL)
    level = max(level, 4)  # a session collection needs level 4 or more (spec 11.7)
    coll = ef.new_collection(
        recs,
        kind="session",
        labels=labels,
        session=session,
        events=list(events),
        provenance=provenance,
        title=f"session {session['fixed']['name']}",
        producer={"name": "dfeols", "version": _version()},
        level=level,
    )
    return ef.write_collection(coll, path, by={"name": "dfeols", "version": _version()})


def _rewrite_collection(sdir: Path, state: dict, status: str) -> None:
    """Close the collection of a run whose manager is gone, from what it left on disk."""
    models = (
        [json.loads(l) for l in (sdir / "results" / "index.jsonl").read_text().splitlines() if l.strip()]
        if (sdir / "results" / "index.jsonl").exists()
        else []
    )
    events = (
        [json.loads(l) for l in (sdir / "events.jsonl").read_text().splitlines() if l.strip()]
        if (sdir / "events.jsonl").exists()
        else []
    )
    events.append(
        {"when": _now(), "type": "note", "message": "manager gone; closed as crashed by the next run"}
    )
    spec = state.get("spec") or {}
    session = {
        "id": state["id"],
        "started": state.get("started"),
        "ended": state.get("heartbeat"),
        "status": status,
        "fixed": {
            "name": state.get("name"),
            "fixed_effects": spec.get("fe"),
            "declared_variables": spec.get("variables"),
            "weights": spec.get("weights"),
        },
    }
    _write_session_collection(sdir, models, events, session, None, sdir / COLLECTION_NAME)


def _demeaned_columns(variables: int, clusters: int) -> int:
    return variables + 1 + clusters  # the variables, the weight, the cluster codes


def _log_timing(root: Path, manifest: dict, nbytes: int | None = None) -> None:
    """One line per demeaning in <demeaned>/timings.jsonl, for start-up estimates."""
    try:
        _append_jsonl(
            root / "timings.jsonl",
            {
                "when": manifest["created"],
                "rows": manifest["rows"],
                "variables": len(manifest["variables"]),
                "crossing": bool(manifest["crossing"]),
                "local_fes": sum(1 for f in manifest["fixed_effects"] if f["absorbed"] == "local"),
                "shards": manifest["n_shards"],
                "seconds": manifest["seconds"],
                "columns": _demeaned_columns(len(manifest["variables"]), len(manifest["clusters"])),
                "bytes": nbytes,
            },
        )
    except OSError:
        pass


def size_estimate(root: Path, rows: int, columns: int) -> tuple[float, float, int] | None:
    """A range of bytes for demeaned data of `rows` x `columns`, from the compressed size of
    earlier runs (bytes per value), or None with no history. The third value is the number
    of runs it comes from. Compression depends on the variables (a dummy packs far smaller
    than a continuous variable), so the range can be wide."""
    p = Path(root) / "timings.jsonl"
    if not p.exists():
        return None
    per = []
    for line in p.read_text().splitlines():
        try:
            t = json.loads(line)
            if t.get("bytes") and t["rows"] > 0 and t["columns"] > 0:
                per.append(t["bytes"] / (t["rows"] * t["columns"]))
        except (ValueError, KeyError, TypeError):
            continue
    if not per:
        return None
    return rows * columns * min(per), rows * columns * max(per), len(per)


def pass_history(root: Path) -> list[float]:
    """Row-columns per second of the first model pass of earlier sessions (from
    <demeaned>/passes.jsonl)."""
    p = Path(root) / "passes.jsonl"
    if not p.exists():
        return []
    rates = []
    for line in p.read_text().splitlines():
        try:
            t = json.loads(line)
            if t.get("first") and t["seconds"] > 0:
                rates.append(t["rows"] * t["columns"] / t["seconds"])
        except (ValueError, KeyError, TypeError):
            continue
    return rates


def estimate_seconds(root: Path, rows: int, variables: int) -> tuple[float, float] | None:
    """A range for demeaning `rows` x `variables` from earlier runs (row-variables per
    second), or None with no history."""
    p = Path(root) / "timings.jsonl"
    if not p.exists():
        return None
    rates = []
    for line in p.read_text().splitlines():
        try:
            t = json.loads(line)
            if t["seconds"] > 0:
                rates.append(t["rows"] * t["variables"] / t["seconds"])
        except (ValueError, KeyError, TypeError):
            continue
    if not rates:
        return None
    work = rows * variables
    return work / max(rates), work / min(rates)


# ---------------------------------------------------------------- the client


class SessionError(RuntimeError):
    pass


class Session:
    """A client of a running session (Python). The R client speaks the same files."""

    def __init__(
        self,
        name: str,
        root=None,
        profile=None,
        *,
        manager: Manager | None = None,
        thread=None,
        progress: bool = True,
    ):
        self.name = name
        self.dir = sessions_root(profile, root) / name
        self.client: str | None = None
        self.progress = progress  # print estimates and progress lines to stderr while waiting
        self._shown: dict[str, Any] = {}
        self._manager = manager
        self._thread = thread

    # ---- start / attach
    @classmethod
    def start(
        cls,
        name: str,
        data,
        fe,
        variables,
        *,
        cluster=None,
        weights: str | None = None,
        root=None,
        demeaned=None,
        executor=None,
        hours: float | None = None,
        idle_minutes: float | None = 30.0,
        persist: bool = True,
        poll: float = 0.5,
        timeout: float | None = None,
        progress: bool = True,
        **spec_kw,
    ) -> Session:
        """Run a manager in a background thread of this process and attach to it (for a
        workstation or a job's own coordinator; on a cluster, `dfeols session start`
        submits the manager as a job). Returns once the data are demeaned."""
        spec = Spec.make(data, fe, variables, cluster, weights=weights, **spec_kw)
        mgr = Manager(
            name,
            spec,
            root=root,
            demeaned=demeaned,
            executor=executor,
            hours=hours,
            idle_minutes=idle_minutes,
            persist=persist,
            poll=poll,
        )
        mgr.start()
        th = threading.Thread(target=mgr.serve, name=f"dfeols-session-{name}", daemon=True)
        th.start()
        s = cls(name, root=root, manager=mgr, thread=th, progress=progress)
        s._attach(timeout)
        return s

    @classmethod
    def attach(
        cls, name: str, root=None, profile=None, timeout: float | None = None, progress: bool = True
    ) -> Session:
        """Attach to a running session (waits for it to be ready, showing the demeaning's
        progress)."""
        s = cls(name, root=root, profile=profile, progress=progress)
        s._attach(timeout)
        return s

    def _attach(self, timeout):
        t0 = time.time()
        while True:
            st = self.state()
            if st and st["status"] == "ready":
                if st.get("workers"):
                    self._say("workers: " + workers_text(st["workers"]))
                break
            if st and st["status"] in ("failed", "closed", "timed_out", "crashed"):
                raise SessionError(
                    f"session {self.name} is {st['status']}" + (f": {st['error']}" if st.get("error") else "")
                )
            if timeout is not None and time.time() - t0 > timeout:
                raise TimeoutError(f"session {self.name} not ready after {timeout:g} s")
            if st:
                est = st.get("demean_estimate_seconds")
                if est and self._once("demean_estimate"):
                    self._say(f"demeaning: estimated {fmt_td(est[0])} to {fmt_td(est[1])}")
                if st.get("progress"):
                    self._show_status(self.dir / st["progress"])
            time.sleep(0.5)
        r = self._request(
            "attach",
            language="python",
            version=sys.version.split()[0],
            host=socket.gethostname(),
        )
        self.client = r["client"]

    # ---- requests
    def state(self) -> dict | None:
        p = self.dir / "session.json"
        try:
            return _read_json(p)
        except (OSError, json.JSONDecodeError):
            return None

    def _say(self, msg: str) -> None:
        if self.progress:
            print(f"[session {self.name}] {msg}", file=sys.stderr, flush=True)

    def _once(self, key: str) -> bool:
        if key in self._shown:
            return False
        self._shown[key] = True
        return True

    def _show_status(self, path: Path) -> None:
        """Print a stage's status.json (Progress) when it has changed."""
        from .progress import status_line

        try:
            snap = _read_json(path)
        except (OSError, json.JSONDecodeError):
            return
        if snap.get("stages") and self._shown.get(str(path)) != snap.get("updated"):
            self._shown[str(path)] = snap.get("updated")
            self._say(status_line(snap))

    def _watch_fit(self, rid: str) -> None:
        acc = self.dir / "responses" / f"{rid}.accepted.json"
        if not acc.exists():
            return
        try:
            a = _read_json(acc)
        except (OSError, json.JSONDecodeError):
            return
        if self._once(f"est:{rid}"):
            rng = a.get("estimate_range")
            e = (
                f"{_secs(rng[0])} to {_secs(rng[1])}"
                if rng and _secs(rng[0]) != _secs(rng[1])
                else _secs(a["estimate_seconds"])
            )
            w = workers_note(a.get("workers"))
            if a.get("what"):
                self._say(f"{a['what']}: estimated {e} ({a['basis']}){w}")
            else:
                self._say(
                    "no pass over the data (from the Gram matrix)"
                    if not a["passes"]
                    else f"estimated {e}: {a['passes']} pass over {a['columns']} columns ({a['basis']}){w}"
                )
        self._show_status(self.dir / a["model_dir"] / "status.json")

    def _request(self, type: str, timeout: float | None = None, watch=None, **payload) -> dict:
        rid = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        req = {"id": rid, "type": type, "client": self.client, "sent": _now(), **payload}
        _write_json(self.dir / "requests" / f"{rid}.json", req)
        resp_path = self.dir / "responses" / f"{rid}.json"
        t0 = time.time()
        while not _appeared(resp_path):
            if timeout is not None and time.time() - t0 > timeout:
                raise TimeoutError(f"no response to {type} after {timeout:g} s")
            st = self.state()
            if st and st["status"] not in ("ready", "starting") and not _appeared(resp_path):
                raise SessionError(f"session {self.name} is {st['status']}")
            if watch:
                watch(rid)
            time.sleep(0.1)
        if watch:  # a quick request may finish before the first look
            watch(rid)
        resp = _read_json(resp_path)
        self.last_trip = {k: resp.get(k) for k in ("sent", "received", "finished")} | {"seen": _now()}
        if not resp.get("ok"):
            raise SessionError(resp.get("error", "request failed"))
        return resp

    def fit(
        self,
        formula: str,
        *,
        vcov=None,
        cluster: str | None = None,
        label: str | None = None,
        timeout: float | None = None,
        estimator: str | None = None,
        fuller: float | None = None,
    ) -> Result:
        """Fit `formula` on declared or created variables; the FEs are the session's. OLS
        (`y ~ x1 + x2`) or IV (`y ~ x | d ~ z1 + z2`; `estimator="liml"`, `fuller=`).
        `cluster` is one variable, or two ("a+b") for two-way clustering."""
        r = self._request(
            "fit",
            timeout=timeout,
            watch=self._watch_fit,
            formula=formula,
            vcov=vcov,
            cluster=cluster,
            label=label,
            estimator=estimator,
            fuller=fuller,
        )
        m = r["model"]
        if m["passes"] or m["seconds"] >= 1:
            self._say(f"{m['label']}: done in {_secs(m['seconds'])}")
        return Result.load(self.dir / m["dir"] / "result.json")

    def refit(
        self,
        model,
        *,
        vcov=None,
        cluster: str | None = None,
        label: str | None = None,
        timeout: float | None = None,
    ) -> Result:
        """An earlier model (a Result, or its label) with another vcov or cluster: the
        coefficients come from the Gram matrix again, the new SEs take one pass (iid none).
        A CRV1 refit keeps the model's cluster unless `cluster` names another."""
        ref = model.label if isinstance(model, Result) else model
        r = self._request(
            "refit",
            timeout=timeout,
            watch=self._watch_fit,
            model=ref,
            vcov=vcov,
            cluster=cluster,
            label=label,
        )
        m = r["model"]
        if m["passes"] or m["seconds"] >= 1:
            self._say(f"{m['label']}: done in {_secs(m['seconds'])}")
        return Result.load(self.dir / m["dir"] / "result.json")

    def add_variable(self, name: str, expression: str, *, timeout: float | None = None) -> Session:
        """Create `name` from raw columns of the dataset (`log(x)`, `a * b`, `x > 0`,
        `ifelse(x > 0, y, 0)`; R's `^`, `&`, `|`, `!` are fine), demeaned on the session's
        FEs; models can then use it like a declared variable. It must be defined on every row
        of the session's sample. Kept with the demeaned data, so another session on them
        reuses it. Returns the session, so calls chain."""
        r = self._request(
            "add_variable", timeout=timeout, watch=self._watch_fit, name=name, expression=expression
        )
        v = r["variable"]
        if not v.get("already"):
            how = "reused from an earlier session" if v["reused"] else f"in {_secs(v['seconds'])}"
            self._say(f"created {v['name']} = {v['expression']} ({how})")
        return self

    def add_variables(self, variables: dict, *, timeout: float | None = None) -> Session:
        """Several created variables ({name: expression}); the new ones are demeaned together,
        in one pass of the demeaning, however many there are."""
        r = self._request("add_variables", timeout=timeout, watch=self._watch_fit, variables=dict(variables))
        made = [v for v in r["variables"] if not v.get("already")]
        if made:
            reused = sum(bool(v["reused"]) for v in made)
            self._say(
                f"created {', '.join(v['name'] for v in made)}"
                + (f" ({reused} reused from an earlier session)" if reused else "")
            )
        return self

    def variables(self) -> dict:
        """The declared variables and the created ones (name -> expression)."""
        st = self.state() or {}
        return {
            "declared": list((st.get("spec") or {}).get("variables", [])),
            "created": {c["name"]: c["expression"] for c in st.get("created", [])},
        }

    def record(self, res: Result, y: str | None = None) -> dict:
        """The estimation record the session wrote for a model."""
        import estrec as ef

        name = (
            "result.estrec.json"
            if len(res.outcomes) == 1
            else f"result.{y or next(iter(res.outcomes))}.estrec.json"
        )
        return ef.read(Path(res.workdir) / name)

    def status(self) -> dict:
        return self._request("status")["session"]

    def export(self, path=None, table_path=None, **table) -> Path:
        """Write the session collection now (status open); returns its path. Given a list of
        models first, `export([m1, m2], "table2.tex", ...)` is `export_table`."""
        if isinstance(path, list | tuple):
            if table_path is None:
                raise TypeError("export(models, path): the table's path is needed")
            return self.export_table(path, table_path, **table)
        return Path(self._request("export", path=str(path) if path else None)["path"])

    def export_table(
        self,
        models,
        path,
        *,
        title: str | None = None,
        headers=None,
        **table,
    ) -> Path:
        """A table of `models` (Results or labels) through the estrec table tool, in the
        format of `path`'s suffix (.tex, .md, .html, .csv, .txt, .doc). Beside it: the table
        spec (`<stem>.estrec-table.json`) and the session collection it reads
        (`<stem>.estrec-collection.json`). `table` goes to estrec's `new_table` (`stats`,
        `rows`, `uncertainty`, `stars`, `keep`, `drop`, ...). Returns the rendered file."""
        refs = [m.label if isinstance(m, Result) else str(m) for m in models]
        r = self._request(
            "export",
            models=refs,
            path=str(Path(path).resolve()),
            title=title,
            headers=list(headers) if headers else None,
            table=table,
        )
        return Path(r["path"])

    def persist(self) -> Path:
        """Keep the demeaned data after the session ends (for a session started with
        persist=False / --no-persist); later sessions with the same spec load it. Returns
        where it is now; a no-op for a session that already persists."""
        r = self._request("persist")
        if not r.get("already"):
            self._say(f"demeaned data kept in {r['location']}")
        return Path(r["location"])

    def close(self) -> None:
        """Detach this client; the session keeps running."""
        if self.client:
            self._request("detach")
            self.client = None

    def stop(self, wait: float = 60) -> None:
        """End the session (writes the collection, releases the lock)."""
        self._request("stop")
        self.client = None
        if self._thread is not None:
            self._thread.join(wait)
        else:
            t0 = time.time()
            while (st := self.state()) and st["status"] == "ready" and time.time() - t0 < wait:
                time.sleep(0.2)

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc):
        if self._manager is not None:
            self.stop()
        else:
            self.close()


def replay(collection, session: Session, *, rtol: float = 1e-8, atol: float = 1e-12) -> dict:
    """Run a session collection's events again in `session` (started on the same data and
    fixed spec): each `variable_created` as add_variable, each model's `fit` (once per
    model, whatever its responses) as fit with the same formula, vcov, cluster and
    estimator, and compare every member's estimates and standard errors with the
    collection's record. Returns {"ok", "models": [...], "key_matches", "mismatches"}."""
    import estrec as ef
    from estrec import tidy

    coll = ef.read_collection(collection) if not isinstance(collection, dict) else collection
    members = {mm["record"]["meta"]["id"]: mm for mm in coll["members"]}
    key = ((coll.get("session") or {}).get("fixed") or {}).get("demeaned_key")
    st = session.state() or {}
    report: dict[str, Any] = {"models": [], "mismatches": 0, "key_matches": None}
    fits: dict[str, list] = {}
    order: list[tuple[str, Any]] = []
    for ev in coll.get("events") or []:
        det = ev.get("detail") or {}
        if ev["type"] == "variable_created":
            order.append(("var", det))
        elif ev["type"] == "fit":
            if det["label"] not in fits:
                fits[det["label"]] = []
                order.append(("fit", det))
            fits[det["label"]].append(ev.get("member"))
    for kind, det in order:
        if kind == "var":
            if "::" not in det["name"]:  # i() columns: the fit makes them again
                session.add_variable(det["name"], det["expression"])
            continue
        res = session.fit(
            det["formula"],
            vcov=det.get("vcov"),
            cluster=det.get("cluster"),
            label=det["label"],
            estimator=det.get("estimator"),
            fuller=det.get("fuller"),
        )
        if report["key_matches"] is None:
            report["key_matches"] = key == res.run.get("session", {}).get("demeaned_key")
        entry = {"label": det["label"], "responses": []}
        for mid, y in zip(fits[det["label"]], res.outcomes):
            old = members.get(mid, {}).get("record")
            new = session.record(res, y)
            a = {r_["term"]: r_ for r_ in tidy(old)} if old else {}
            b = {r_["term"]: r_ for r_ in tidy(new)}
            worst = 0.0
            same = bool(old) and set(a) == set(b)
            for t in b if same else []:
                for f_ in ("estimate", "std_error"):
                    x, z = a[t].get(f_), b[t].get(f_)
                    if x is None or z is None:
                        continue
                    err = abs(x - z) / max(abs(x), atol)
                    worst = max(worst, err)
                    same &= abs(x - z) <= atol + rtol * abs(x)
            entry["responses"].append({"response": y, "member": mid, "match": same, "max_rel_diff": worst})
            report["mismatches"] += not same
        report["models"].append(entry)
    report["ok"] = report["mismatches"] == 0
    report["session"] = st.get("name")
    return report


__all__ = [
    "replay",
    "Session",
    "Manager",
    "Spec",
    "Demeaned",
    "demean_once",
    "fit_model",
    "LockedError",
    "SessionError",
]


# ---------------------------------------------------------------- command line


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _spec_args(args) -> Spec:
    return Spec.make(
        args.data,
        args.fe,
        args.vars,
        args.cluster,
        weights=args.weights,
        drop_singletons=not args.keep_singletons,
        fe_tol=args.fe_tol,
    )


def _serve_words(args) -> list[str]:
    """`dfeols session serve` arguments that reproduce this start."""
    w = [
        "session",
        "serve",
        args.name,
        "--data",
        str(Path(args.data).resolve()),
        "--fe",
        args.fe,
        "--vars",
        args.vars,
    ]
    for flag, v in (
        ("--cluster", args.cluster),
        ("--weights", args.weights),
        ("--hours", args.hours),
        ("--idle-minutes", args.idle_minutes),
        ("--profile", args.profile),
        ("--root", args.root),
        ("--demeaned", args.demeaned),
        ("--fe-tol", args.fe_tol),
    ):
        if v is not None:
            w += [flag, str(v)]
    if args.no_persist:
        w.append("--no-persist")
    if args.keep_singletons:
        w.append("--keep-singletons")
    if getattr(args, "serial", False):
        w.append("--serial")
    if getattr(args, "worker_walltime", None):
        w += ["--worker-walltime", args.worker_walltime]
    return w


def cmd_serve(args) -> None:
    """Run the manager in this process (what the job runs)."""
    from .site import profile as load_profile

    ex = None
    if not args.serial and args.profile:
        p = load_profile(args.profile)
        if p.scheduler != "local" or p.worker.get("max_workers", 1) > 1:
            from .cluster import executor

            # the workers are held for the whole session, idle or not (Ted 2026-11-02), and
            # live as long as the manager's job
            ex = executor(p, hold=True, walltime=args.worker_walltime)
    mgr = Manager(
        args.name,
        _spec_args(args),
        root=args.root,
        demeaned=args.demeaned,
        executor=ex,
        hours=args.hours,
        idle_minutes=args.idle_minutes,
        persist=not args.no_persist,
        profile=args.profile,
    )
    import signal

    signal.signal(signal.SIGTERM, lambda *_: mgr.stop())  # scancel / walltime: close cleanly
    print(f"session {args.name}: {mgr.id}; folder {mgr.dir}", flush=True)
    end = mgr.serve()
    print(f"session {args.name}: {end}", flush=True)


TASK_COPIES = 4  # a demeaning task holds about this many copies of its shard's columns


def memory_warnings(spec: Spec, ds: Dataset, plan, profile) -> list[str]:
    """No cap on declared variables (Ted 2026-11-08), but a warning when a demeaning task's
    memory (the largest shard's rows x columns x 8 bytes x TASK_COPIES, plus dense crossing
    dummies) is over a worker thread's share of the profile's worker memory."""
    from .cluster import _memory_bytes

    out = []
    rows = max((s_["rows"] for s_ in ds.shards), default=0)
    cols = len(spec.variables) + len(spec.clusters) + 2 + (plan.n_dummy if not plan.crossing_iterate else 0)
    need = rows * cols * 8 * TASK_COPIES
    w = getattr(profile, "worker", None) or {}
    try:
        per_task = _memory_bytes(w["memory"]) / max(int(w.get("cores") or 1), 1)
    except (KeyError, TypeError, ValueError):
        per_task = None
    if per_task and need > per_task:
        out.append(
            f"a demeaning task needs ~{need / 1e9:.1f} GB ({rows:,} rows x {cols} columns in the largest "
            f"shard), over a worker thread's ~{per_task / 1e9:.1f} GB ({w['memory']} / {w.get('cores')} "
            "cores): fewer declared variables, smaller shards, or more worker memory"
        )
    return out


def _check_storage(p, dest: Path, lo: float | None, hi: float, yes: bool) -> None:
    """Scratch space for the demeaned data `session start` will write: the file system's free
    space, a stop when even the low end does not fit, and the profile's confirmation where
    storage is scarce (`demeaned_storage = "confirm"`, above `demeaned_confirm_gb`)."""
    from .store import fmt_bytes, space

    sp = space(dest)
    if sp:
        print(f"scratch: {fmt_bytes(sp['free'])} free of {fmt_bytes(sp['total'])} ({sp['path']})")
        if lo is not None and lo > sp["free"]:
            sys.exit(
                f"not enough scratch for the demeaned data: at least {fmt_bytes(lo)}, "
                f"{fmt_bytes(sp['free'])} free (dfeols demeaned list / rm frees space)"
            )
        if hi > sp["free"]:
            print(f"warning: the demeaned data may need up to {fmt_bytes(hi)}, more than is free")
    if p.demeaned_storage == "confirm" and hi > (p.demeaned_confirm_gb or 0) * 1e9:
        msg = (
            f"the demeaned data may take up to {fmt_bytes(hi)} "
            f"(profile asks above {p.demeaned_confirm_gb:g} GB)"
        )
        if yes:
            print(f"{msg}: confirmed by --yes")
        elif sys.stdin.isatty():
            if input(f"{msg}. Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                sys.exit("not started")
        else:
            sys.exit(f"{msg}: confirm with --yes")


def cmd_start(args) -> None:
    """Print the plan and estimates, then submit the manager (a job, or a local process)."""
    import shlex
    import subprocess

    from .site import profile as load_profile

    spec = _spec_args(args)
    ds = Dataset.open(spec.data)
    plan, _ = _session_plan(spec, ds, None)
    p = load_profile(args.profile)
    root = sessions_root(args.profile, args.root)
    dm_root = demeaned_root(args.profile, args.demeaned)
    key = spec.key(ds)
    where = dm_root / ds.root.name / key
    local = ", ".join(n for n, _ in plan.local_fes) or "none"
    cross = ", ".join(f"{n} ({L:,} levels)" for n, _, L in plan.global_fes) or "none"
    print(f"dataset: {ds.n_rows:,} rows, {len(ds.shards):,} shards; FEs local: {local}; crossing: {cross}")
    est = None
    if (where / "manifest.json").exists():
        print(f"demeaned data found ({key}): no demeaning needed")
        if p.scratch_purge_days:
            from .store import entries

            e = next((e for e in entries(dm_root) if e["path"] == str(where)), None)
            if e:
                print(f"  written {e['written']}; the site may purge it after ~{e['purge_after']}")
        est = (0.0, 0.0)
    else:
        est = estimate_seconds(dm_root, ds.n_rows, len(spec.variables))
        if est:
            print(
                f"demeaning {len(spec.variables)} variables: estimated {fmt_td(est[0])} to {fmt_td(est[1])}"
            )
        else:
            print(f"demeaning {len(spec.variables)} variables: no earlier runs to estimate from")
        cols = _demeaned_columns(len(spec.variables), len(spec.clusters))
        for w_ in memory_warnings(spec, ds, plan, p):
            print(f"warning: {w_}")
        most = ds.n_rows * cols * 8  # 8 bytes a value, before compression
        dest = where if not args.no_persist else root / args.name
        se = size_estimate(dm_root, ds.n_rows, cols)
        lo, hi = None, most
        if se:
            lo, hi, n = se
            hi = min(hi, most)
            rng = (
                f"~{lo / 1e9:.1f} GB"
                if f"{lo / 1e9:.1f}" == f"{hi / 1e9:.1f}"
                else f"{lo / 1e9:.1f} to {hi / 1e9:.1f} GB"
            )
            print(
                f"demeaned data: {rng} in {dest} (compressed size of {n} earlier "
                f"run{'s' if n > 1 else ''}; at most {most / 1e9:.1f} GB)"
            )
        else:
            print(f"demeaned data: at most {most / 1e9:.1f} GB in {dest} (before compression)")
        _check_storage(p, dest, lo, hi, args.yes)
    words = _serve_words(args)
    if p.scheduler == "local":
        root.mkdir(parents=True, exist_ok=True)
        logf = root / f"{args.name}.log"
        with open(logf, "ab") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-m", "dfeols.cli", *words],
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"manager pid {proc.pid}; log {logf}")
    else:
        from .cluster import coordinator_script, submit

        hours = args.hours or 2.0
        demean_h = (est[1] if est else 3600.0) / 3600
        wall = (hours + 1.5 * demean_h) * 3600
        p = dataclasses.replace(p, coordinator={**p.coordinator, "walltime": _hms(wall)})
        args.worker_walltime = args.worker_walltime or _hms(wall)  # workers last the session
        words = _serve_words(args)
        jobs = p.scratch_dir("jobs")
        jobs.mkdir(parents=True, exist_ok=True)
        cmd = "python -m dfeols.cli " + " ".join(shlex.quote(w) for w in words)
        out = str(jobs / f"session-{args.name}-%j.out")
        script = coordinator_script(p, cmd, job_name=f"dfeols-session-{args.name}", output=out)
        print(f"requesting {_hms(wall)} ({hours:g} h + 1.5 x the demeaning estimate)")
        print(submit(script, p, path=jobs / f"session-{args.name}-{time.strftime('%Y%m%d-%H%M%S')}.sh"))
        print(f"job output: {out.replace('%j', '<jobid>')}")
    if args.wait:
        s = Session(args.name, root=root)
        t0 = time.time()
        while True:
            st = s.state()
            if st and st.get("status") == "ready":
                print(f"ready after {fmt_td(time.time() - t0)}; session {st['id']}")
                if st.get("workers"):
                    print("workers: " + workers_text(st["workers"]))
                break
            if st and st.get("status") == "failed":
                sys.exit(f"session failed: {st.get('error')}")
            if st and st.get("progress"):
                s._show_status(s.dir / st["progress"])
            time.sleep(2)
    print(
        f"attach: dfeols.Session.attach({args.name!r}" + (f", root={str(root)!r}" if args.root else "") + ")"
    )


def cmd_status(args) -> None:
    root = sessions_root(args.profile, args.root)
    st = Session(args.name, root=root).state()
    if not st:
        sys.exit(f"no session {args.name!r} under {root}")
    print(
        json.dumps(
            {
                k: st[k]
                for k in (
                    "name",
                    "id",
                    "status",
                    "started",
                    "ends",
                    "heartbeat",
                    "host",
                    "job",
                    "models",
                    "workers",
                    "demeaned",
                )
                if k in st
            },
            indent=1,
        )
    )


def cmd_replay(args) -> None:
    s = Session.attach(args.name, root=args.root, profile=args.profile, timeout=args.timeout)
    try:
        r = replay(args.collection, s, rtol=args.rtol)
    finally:
        s.close()
    for mdl in r["models"]:
        for x in mdl["responses"]:
            tag = "same" if x["match"] else f"DIFFERENT (max relative difference {x['max_rel_diff']:.2e})"
            print(f"{mdl['label']} {x['response']}: {tag}")
    if r["key_matches"] is False:
        print("note: the demeaned data's key differs from the collection's (other data, spec or format)")
    print(f"{sum(len(m_['responses']) for m_ in r['models'])} members replayed, {r['mismatches']} different")
    if not r["ok"]:
        sys.exit(1)


def cmd_where(args) -> None:
    """The session's folder (for clients that don't read the site profile, such as R's)."""
    print(sessions_root(args.profile, args.root) / args.name)


def cmd_fit(args) -> None:
    s = Session.attach(args.name, root=args.root, profile=args.profile, timeout=args.timeout)
    try:
        vcov = [v.strip() for v in args.vcov.split(",")] if args.vcov else None
        r = s.fit(args.formula, vcov=vcov, cluster=args.cluster, label=args.label)
        print(r.summary())
    finally:
        s.close()


def cmd_refit(args) -> None:
    s = Session.attach(args.name, root=args.root, profile=args.profile, timeout=args.timeout)
    try:
        vcov = [v.strip() for v in args.vcov.split(",")] if args.vcov else None
        r = s.refit(args.model, vcov=vcov, cluster=args.cluster, label=args.label)
        print(r.summary())
    finally:
        s.close()


def cmd_add_variable(args) -> None:
    s = Session.attach(args.name, root=args.root, profile=args.profile, timeout=args.timeout)
    try:
        s.add_variable(args.variable, args.expression)
        print(f"{args.variable} = {s.variables()['created'][args.variable]}")
    finally:
        s.close()


def cmd_export(args) -> None:
    s = Session.attach(args.name, root=args.root, profile=args.profile, timeout=args.timeout)
    try:
        if args.models:
            if not args.path:
                sys.exit("--models needs --path (the table file: .tex, .md, .html, .csv, .txt, .doc)")
            models = [m.strip() for m in args.models.split(",") if m.strip()]
            print(s.export_table(models, args.path, title=args.title))
        else:
            print(s.export(args.path))
    finally:
        s.close()


def cmd_persist(args) -> None:
    s = Session.attach(args.name, root=args.root, profile=args.profile, timeout=args.timeout, progress=False)
    try:
        print(s.persist())
    finally:
        s.close()


def cmd_stop(args) -> None:
    s = Session(args.name, root=args.root, profile=args.profile)
    st = s.state()
    if not st or st.get("status") != "ready":
        sys.exit(f"session {args.name!r} is not running ({st.get('status') if st else 'not found'})")
    s.stop()
    print(f"session {args.name}: {s.state().get('status')}")


def add_parser(sub) -> None:
    """`dfeols session ...` on the CLI's subparsers."""
    sp = sub.add_parser("session", help="interactive sessions: absorb FEs once, fit many models")
    ss = sp.add_subparsers(dest="session_cmd", required=True)

    def common(q):
        q.add_argument("name")
        q.add_argument("--profile")
        q.add_argument("--root", help="sessions folder (default: <scratch>/sessions)")

    for nm, hlp in (
        ("start", "check, estimate and submit a session manager"),
        ("serve", "run the manager here"),
    ):
        q = ss.add_parser(nm, help=hlp)
        common(q)
        q.add_argument("--data", required=True, help="prepared shard folder")
        q.add_argument("--fe", required=True, help='fixed effects, e.g. "route + carrier^quarter"')
        q.add_argument("--vars", required=True, help="declared variables, comma-separated")
        q.add_argument("--cluster", help="cluster variables models may use, comma-separated")
        q.add_argument("--weights")
        q.add_argument("--hours", type=float, help="how long the session stays up")
        q.add_argument("--idle-minutes", type=float, default=30.0)
        q.add_argument("--no-persist", action="store_true", help="delete the demeaned data at the end")
        q.add_argument("--keep-singletons", action="store_true")
        q.add_argument("--fe-tol", type=float, default=1e-10)
        q.add_argument("--demeaned", help="demeaned-data folder (default: <scratch>/demeaned)")
        q.add_argument("--serial", action="store_true", help="no Dask workers: run in the manager")
        q.add_argument("--worker-walltime", help="workers' walltime (start sets the manager job's)")
        if nm == "start":
            q.add_argument("--wait", action="store_true", help="wait until the session is ready")
            q.add_argument("--yes", action="store_true", help="confirm a large demeaned-data write")
    q = ss.add_parser("status", help="the session's state")
    common(q)
    q = ss.add_parser("where", help="the session's folder")
    common(q)
    q = ss.add_parser("replay", help="run a session collection's models again here and compare")
    common(q)
    q.add_argument("collection", help="a session.estrec-collection.json")
    q.add_argument("--rtol", type=float, default=1e-8)
    q.add_argument("--timeout", type=float)
    q = ss.add_parser("fit", help="fit a model in a running session")
    common(q)
    q.add_argument("formula")
    q.add_argument("--vcov", help="iid, hetero, CRV1 (comma-separated)")
    q.add_argument("--cluster")
    q.add_argument("--label")
    q.add_argument("--timeout", type=float)
    q = ss.add_parser("refit", help="an earlier model with another vcov or cluster")
    common(q)
    q.add_argument("model", help="the model's label (m1, ...) or number")
    q.add_argument("--vcov", help="iid, hetero, CRV1 (comma-separated)")
    q.add_argument("--cluster")
    q.add_argument("--label")
    q.add_argument("--timeout", type=float)
    q = ss.add_parser("add-variable", help="create a variable from raw columns, demeaned on the FEs")
    common(q)
    q.add_argument("variable", help="its name")
    q.add_argument("expression", help="e.g. 'log(fare)', 'hhi * dist', 'pax > 0'")
    q.add_argument("--timeout", type=float)
    q = ss.add_parser("export", help="write the session collection now, or a table of models")
    common(q)
    q.add_argument("--path", help="the collection's path; with --models, the table's (.tex, .md, ...)")
    q.add_argument("--models", help="labels of the models to tabulate, comma-separated")
    q.add_argument("--title")
    q.add_argument("--timeout", type=float)
    q = ss.add_parser("persist", help="keep a --no-persist session's demeaned data after it ends")
    common(q)
    q.add_argument("--timeout", type=float)
    q = ss.add_parser("stop", help="end a session")
    common(q)


def main(args) -> None:
    {
        "start": cmd_start,
        "serve": cmd_serve,
        "status": cmd_status,
        "where": cmd_where,
        "replay": cmd_replay,
        "fit": cmd_fit,
        "refit": cmd_refit,
        "add-variable": cmd_add_variable,
        "export": cmd_export,
        "persist": cmd_persist,
        "stop": cmd_stop,
    }[args.session_cmd](args)
