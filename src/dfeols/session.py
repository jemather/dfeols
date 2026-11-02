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
      session.estimation-collection.json    on export and when the session ends

The demeaned data go to <demeaned>/<dataset>/<key>/ (key: a hash of what fixes them) and
are reused by a later session with the same inputs.

Not built yet (build order 3-6): created variables, a vcov change on an existing model,
persisted-data removal and size checks, exports, the R client, IV; crossing FEs on the
sparse or iterate path (sessions take crossing FEs as dense dummies, up to max_dummy_levels).
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
from .linalg import collinear_keep, solve_spd
from .lock import DirLock, LockedError
from .prep import Dataset
from .progress import Progress, fmt_td
from .results import Result
from .vcov import df_k as _df_k
from .vcov import vcov_crv1, vcov_hetero, vcov_iid
from .worker import Plan, _maxrss_gb, _meat, _prepare, _round1_dense

log = logging.getLogger("dfeols")

SESSION_FORMAT = "dfeols-session/1"
DEMEANED_FORMAT = "dfeols-demeaned/1"
COLLECTION_NAME = "session.estimation-collection.json"
WEIGHT_COL = "__w"
TASK_OVERHEAD = 0.005  # seconds a shard task costs beyond its reading (scheduling, results)
VCOVS = ("iid", "hetero", "CRV1")
_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write_json(path: Path, obj: Any) -> None:
    """Write atomically: readers on another node see the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable) + "\n")
    os.replace(tmp, path)


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
        return cls(
            data=str(Path(data).resolve()),
            fe=split(fe, "+"),
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


def _write_part(out: Path, sid: int, names: list[str], V: np.ndarray, w: np.ndarray, cls: dict) -> None:
    cols = {n: V[:, j] for j, n in enumerate(names)}
    cols[WEIGHT_COL] = w
    for name, codes in cls.items():
        cols[_cl_col(name)] = codes
    p = _part_path(out, sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    pl.DataFrame(cols).write_parquet(tmp)
    os.replace(tmp, p)


def demean_round1(plan: Plan, clusters: list[dict], sid: int, files: list[str], out: str) -> dict:
    """Per shard: the session sample, every declared variable demeaned by the local FEs and
    the crossing-FE dummies too, their cross-products, and the cluster facts. With no
    crossing FE the columns are final and written; else they wait in a stage file."""
    from .rowhash import rowset

    t0 = time.perf_counter()
    out_dir = Path(out)
    T: dict = {}
    loaded: dict = {}
    df, n0, local_codes, w = _prepare(plan, sid, files, loaded)
    n = df.height
    T["read"] = time.perf_counter() - t0
    aux = None
    if plan.crossing_iterate:  # plan C: per-level sums D'W V~ and the shard's part of A
        from .worker import _iter_store, _round1_sparse

        _, Yt, _, _, _, extra, converged, aux = _round1_sparse(plan, sid, df, n0, local_codes, w, T, t0)
        _iter_store(plan, sid, extra[0], extra[1], w)  # for CG steps with a pass each
        Zt = ZtZ = ZtY = None
    else:
        _, Yt, Zt, ZtZ, ZtY, converged, _ = _round1_dense(plan, df, n, local_codes, w, T)
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
    }
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
        _write_part(out_dir, sid, names, Yt, w, cls)
    T["total"] = time.perf_counter() - t0
    T["maxrss_gb"] = _maxrss_gb()
    r["timing"] = T
    return r


def demean_round2(
    sid: int, out: str, names: list[str], keep: np.ndarray | None, gamma: np.ndarray, plan: Plan | None = None
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
    _write_part(out_dir, sid, names, V, w, cls)
    p.unlink()
    return {
        "shard": sid,
        "gram": (V * w[:, None]).T @ V,
        "timing": {"total": time.perf_counter() - t0, "maxrss_gb": _maxrss_gb()},
    }


def model_meat(
    path: str, xs: list[str], ys: list[str], beta: np.ndarray, vcov_types: tuple, cluster: dict | None
) -> dict:
    """Per shard, for one model: residuals, RSS and the meats (hetero, CRV1)."""
    t0 = time.perf_counter()
    cols = [*xs, *ys, WEIGHT_COL] + ([_cl_col(cluster["name"])] if cluster else [])
    df = pl.read_parquet(path, columns=list(dict.fromkeys(cols)))
    X = df.select(xs).to_numpy().astype(np.float64) if xs else np.zeros((df.height, 0))
    Y = df.select(ys).to_numpy().astype(np.float64)
    w = df[WEIGHT_COL].to_numpy()
    cl = df[_cl_col(cluster["name"])].to_numpy() if cluster else None
    E = Y - X @ beta
    plan = SimpleNamespace(
        vcov_types=vcov_types,
        cluster=(cluster["name"], None, cluster["levels"]) if cluster else None,
        cluster_local=bool(cluster and cluster["local"]),
    )
    out: dict = {"n": df.height}
    _meat(plan, X, E, w, cl, out)
    out["timing"] = {"total": time.perf_counter() - t0, "maxrss_gb": _maxrss_gb()}
    return out


# ---------------------------------------------------------------- demean once


def _level_reduce(gram, plan: Plan):
    """Sum the shards' parts of A on the workers as they arrive (as fit's round 1)."""
    import functools

    from .crossing import combine_parts, part_of, without_part
    from .executors import Reduce

    return Reduce(
        get=part_of,
        strip=without_part,
        combine=functools.partial(combine_parts, offs=gram.offs, max_nnz=plan.level_nnz),
        done=lambda part: gram.add(part, None),
    )


class Demeaned:
    """The persisted demeaned data: residualised columns per shard and manifest.json."""

    def __init__(self, root: Path, manifest: dict):
        self.root = Path(root)
        self.manifest = manifest
        self.names = list(manifest["variables"])
        self.gram = np.asarray(manifest["gram"], float)
        self.shards = manifest["shards"]

    @classmethod
    def open(cls, root) -> Demeaned:
        return cls(Path(root), _read_json(Path(root) / "manifest.json"))

    def part(self, sid: int) -> str:
        return str(_part_path(self.root, sid))

    def bytes(self) -> int:
        return sum(p.stat().st_size for p in (self.root / "shards").glob("*.parquet"))


def demean_once(
    spec: Spec,
    ds: Dataset,
    out: Path,
    executor,
    label: str = "session",
    log_every: float = 30.0,
    progress_dir: Path | None = None,
):
    """Demean every declared variable on all the session's FEs; write out/shards and
    out/manifest.json (last, so a manifest means the data are complete)."""
    from . import rowhash
    from .estimate import _drop_singletons
    from .record import dfeols_source

    t_start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    plan, clusters = _session_plan(spec, ds, out / "singletons")
    names = list(spec.variables)
    prog = Progress(progress_dir or out, label=label, every=log_every)
    rows = {s["id"]: max(s["rows"], 1) for s in ds.shards}
    sing = {"passes": 0, "seconds": 0.0}
    if plan.keep_dir:
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
        "rss_gb": 0.0,
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
        from .crossing import LevelGram, level_space_budget

        plan.iter_token = f"{out}:{uuid.uuid4().hex}"
        plan.level_nnz = level_space_budget(executor, spec.level_space_gb)
        offs = np.r_[0, np.cumsum([L for *_, L in plan.global_fes])].astype(np.int64)
        acc["DR"] = np.zeros((int(offs[-1]), len(names)))
        acc["ddiag"] = np.zeros(int(offs[-1]))
        if plan.level_nnz:
            acc["gram_A"] = LevelGram(offs, [n for n, *_ in plan.global_fes], plan.level_nnz)
            reduce = _level_reduce(acc["gram_A"], plan)
    elif plan.global_fes:
        acc["ZtZ"] = np.zeros((k_tot, k_tot))
        acc["ZtY"] = np.zeros((k_tot, len(names)))
    local_of = {c["name"]: c["local"] for c in clusters}

    def on1(tid, r):
        acc["rss_gb"] = max(acc["rss_gb"], r["timing"].get("maxrss_gb", 0.0))
        for k in ("n", "rows_in", "sumw"):
            acc[k] += r[k]
        for k, v in r["load"].items():
            acc["load"][k] += v
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
        (s["id"], rows[s["id"]], (plan, clusters, s["id"], ds.shard_paths(s), str(out))) for s in ds.shards
    ]
    prog.add_stage("demean", rows)
    prog.start("demean")
    if reduce is not None and "reduce" in inspect.signature(executor.run).parameters:
        prog.task_gb = None  # a shard's part of A is not in any estimate: a whole worker
        executor.run(demean_round1, tasks, on1, prog, "demean", reduce=reduce)
    else:
        executor.run(demean_round1, tasks, on1, prog, "demean")
    prog.finish("demean")
    crossing: dict[str, Any] = {}
    extra: tuple = ()
    if plan.crossing_iterate:
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
        from .worker import _ITER_CACHE

        for k in [k for k in _ITER_CACHE if k[0] == plan.iter_token]:  # this process's (serial)
            _ITER_CACHE.pop(k, None)
    elif plan.global_fes:
        keep = collinear_keep(acc["ZtZ"], 1e-9)
        gamma, _ = solve_spd(acc["ZtZ"][np.ix_(keep, keep)], acc["ZtY"][keep])
        crossing = {"method": "dense", "dummies": int(k_tot), "kept": int(keep.sum())}
    if plan.global_fes:

        def on2(tid, r):
            acc["rss_gb"] = max(acc["rss_gb"], r["timing"].get("maxrss_gb", 0.0))
            acc["gram"] += r["gram"]

        tasks2 = [
            (s["id"], rows[s["id"]], (s["id"], str(out), names, keep, gamma, *extra)) for s in ds.shards
        ]
        prog.add_stage("partial_out", rows)
        prog.start("partial_out")
        executor.run(demean_round2, tasks2, on2, prog, "partial_out")
        prog.finish("partial_out")
        shutil.rmtree(out / "stage", ignore_errors=True)
    if plan.keep_dir:
        shutil.rmtree(plan.keep_dir, ignore_errors=True)
    if not acc["converged"]:
        log.warning("%s: FE demeaning did not converge in some shards", label)

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


# ---------------------------------------------------------------- one model


def _model_terms(formula: str, spec_fe: list[str]) -> tuple[Formula, list[str], list[str]]:
    f = parse(formula)
    if f.iv:
        raise NotImplementedError("IV in sessions is not built yet (sessions.md, build order 6)")
    if f.fes and {"^".join(fe_components(x)) for x in f.fes} != {"^".join(fe_components(x)) for x in spec_fe}:
        raise ValueError(
            f"the session's FEs are fixed ({' + '.join(spec_fe)}); leave the FE part out of the formula"
        )
    xs = []
    for t in f.terms:
        if t.cat is not None:
            raise NotImplementedError(f"factor terms (i()) in sessions are not built yet: {t.label}")
        if len(t.conts) != 1:
            raise ValueError(
                f"{t.label}: an interaction of demeaned columns is not the demeaned interaction; "
                "create the product from raw columns (add_variable, not built yet) and declare it"
            )
        xs.append(t.conts[0])
    return f, list(f.depvars), xs


def fit_model(
    d: Demeaned,
    formula: str,
    *,
    vcov=None,
    cluster: str | None = None,
    label: str | None = None,
    executor=None,
    workdir: Path,
    session: dict | None = None,
    collin_tol: float = 1e-9,
    log_every: float = 30.0,
) -> Result:
    """OLS of `formula` (y ~ x1 + x2, declared variables) on the demeaned data. Exact by
    Frisch-Waugh-Lovell; b and iid SEs from the Gram matrix, hetero/CRV1 with one pass."""
    from . import record as _record

    executor = executor or SerialExecutor()
    t_start = time.time()
    cpu_start = time.process_time()
    m = d.manifest
    spec = m["spec"]
    f, ys, xs = _model_terms(formula, spec["fe"])
    unknown = [v for v in [*ys, *xs] if v not in d.names]
    if unknown:
        raise KeyError(f"not declared in this session: {unknown} (declared: {', '.join(d.names)})")
    cluster = "^".join(fe_components(cluster)) if cluster else None
    if cluster and cluster not in m["clusters"]:
        raise KeyError(
            f"cluster {cluster!r} was not declared for this session ({', '.join(m['clusters']) or 'none'})"
        )
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
    ix, iy = [pos[x] for x in xs], [pos[y] for y in ys]
    Gm = d.gram
    XX = Gm[np.ix_(ix, ix)]
    keep = collinear_keep(XX, collin_tol) if xs else np.zeros(0, bool)
    kept = [xs[i] for i in np.flatnonzero(keep)]
    dropped = [xs[i] for i in np.flatnonzero(~keep)]
    ik = [pos[x] for x in kept]
    if kept:
        beta, bread = solve_spd(Gm[np.ix_(ik, ik)], Gm[np.ix_(ik, iy)])
    else:
        beta, bread = np.zeros((0, len(ys))), np.zeros((0, 0))
    yy = np.diag(Gm)[iy]
    rss = yy - np.einsum("kj,kj->j", Gm[np.ix_(ik, iy)], beta)
    N = int(m["rows"])
    k = len(kept)
    fe_levels = [int(fe["levels"]) for fe in m["fixed_effects"]]
    cl = dict(m["clusters"][cluster], name=cluster) if cluster else None
    nested = list(cl["fe_nested"]) if cl else []
    G = int(cl["G"]) if cl else None
    passes = 0
    meats = {"rss": None, "hetero": None, "crv": None}
    if any(v in ("hetero", "CRV1") for v in vcov) and kept:
        passes = 1
        acc: dict[str, Any] = {"rss": np.zeros(len(ys)), "hetero": None, "crv": None, "rss_gb": 0.0}
        if "hetero" in vcov:
            acc["hetero"] = np.zeros((len(ys), k, k))
        scores = None
        if "CRV1" in vcov:
            acc["crv"] = np.zeros((len(ys), k, k))
            if not cl["local"]:
                scores = np.zeros((len(ys), cl["levels"], k))

        def on(tid, r):
            acc["rss_gb"] = max(acc["rss_gb"], r["timing"].get("maxrss_gb", 0.0))
            acc["rss"] += r["rss"]
            if acc["hetero"] is not None:
                acc["hetero"] += r["meat_hetero"]
            if "meat_crv" in r:
                acc["crv"] += r["meat_crv"]
            if "cluster_scores" in r:
                for j in range(len(ys)):
                    np.add.at(scores[j], r["cluster_codes"], r["cluster_scores"][j])

        rows = {sid: 1 for sid in d.shards}
        prog = Progress(wd, label=label or "model", every=log_every)
        prog.add_stage("meat", rows)
        prog.start("meat")
        tasks = [(sid, 1, (d.part(sid), kept, ys, beta, vcov, cl)) for sid in d.shards]
        t_meat = time.time()
        executor.run(model_meat, tasks, on, prog, "meat")
        meat_seconds = time.time() - t_meat
        prog.finish("meat", log=False)
        if scores is not None:
            for j in range(len(ys)):
                acc["crv"][j] = scores[j].T @ scores[j]
        meats = acc
        rss = acc["rss"]
    out = {}
    dfk_plain = _df_k(k, fe_levels, None)
    for j, y in enumerate(ys):
        V = {}
        if "iid" in vcov:
            V["iid"] = (vcov_iid(bread, rss[j], N, dfk_plain), N - dfk_plain)
        if "hetero" in vcov:
            V["hetero"] = (vcov_hetero(bread, meats["hetero"][j], N, dfk_plain), N - dfk_plain)
        if "CRV1" in vcov:
            V["CRV1"] = (vcov_crv1(bread, meats["crv"][j], N, _df_k(k, fe_levels, nested), G), G - 1)
        out[y] = {
            "coef": beta[:, j],
            "vcov": V,
            "rss": float(rss[j]),
            "tss": float(yy[j]),
            "r2_within": float(1 - rss[j] / yy[j]) if yy[j] > 0 else float("nan"),
        }
    fe_names = [fe["name"] for fe in m["fixed_effects"]]
    text = f"{' + '.join(ys)} ~ {' + '.join(xs) if xs else '1'} | {' + '.join(spec['fe'])}"
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
            "columns": list(d.names),
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
        "declared": list(d.names),
        "passes": passes,
    }
    res = Result(
        formula=text,
        label=label or f.fingerprint(cluster=cluster, vcov=list(vcov)),
        coefnames=kept,
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
            **({"max_worker_rss_gb": meats["rss_gb"]} if passes else {}),
        },
        options=options,
        terms=_record.term_rows(f, {}, has_fe=True),
        run=run,
    )
    res.save(wd / "result.json")
    return res


def session_record(res: Result, y: str | None = None, **kwargs) -> dict:
    """A model's estimation record (unencoded; write it with estimation_format.write), saying
    it came from a session: the sample rule is the session's, the call replays through it."""
    from . import record as _record

    rec = _record.to_estimation(res, y, **kwargs)
    s = res.run.get("session") or {}
    declared = s.get("declared") or []
    call = f"dfeols.Session.attach({s.get('name', '')!r}).fit({res.formula.split(' | ')[0]!r}"
    o = res.options
    if o.get("cluster"):
        call += f", cluster={o['cluster']!r}"
    call += f", vcov={o['vcov']!r})"
    rec["specification"]["call"] = call
    rec["specification"]["sample"]["restriction"] = (
        "rows complete on every variable declared for the session (" + ", ".join(declared) + ")"
    )
    ext = rec["extensions"]["dfeols"]
    ext["replay"]["call"] = call
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
        poll: float = 1.0,
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
        self.pass_rate: float | None = None  # row-columns per second of a model's pass, measured
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
        _log_timing(self.demeaned_dir, self.data.manifest)

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
        with DirLock(target, what="demeaned data"):
            if (target / "manifest.json").exists():  # the same key was built by another session
                self.data = Demeaned.open(target)
                shutil.rmtree(src, ignore_errors=True)
                moved = False
            else:
                shutil.rmtree(target, ignore_errors=True)  # an interrupted build
                shutil.move(str(src), str(target))
                self.data = Demeaned.open(target)
                moved = True
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
            resp["finished"] = _now()
            _write_json(self.dir / "responses" / f"{rid}.json", resp)
            if self._stop.is_set():
                break
        return bool(reqs)

    def handle(self, req: dict) -> dict:
        t = req.get("type")
        if t == "fit":
            return self.fit(req)
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
            p = self.write_collection(req.get("path"))
            return {"path": str(p)}
        if t == "stop":
            self.stop()
            return {}
        if t == "persist":
            return self.persist_data()
        raise ValueError(f"unknown request type {t!r}")

    def fit(self, req: dict) -> dict:
        from estimation_format import write as ef_write

        n = len(self.models) + 1
        label = req.get("label") or f"m{n}"
        if not _NAME.match(label):
            raise ValueError(f"label must match {_NAME.pattern}: {label!r}")
        wd = self.dir / "models" / f"{n:03d}-{label}"
        est = self.estimate(req)
        if req.get("id"):
            _write_json(
                self.dir / "responses" / f"{req['id']}.accepted.json",
                {"model_dir": str(wd.relative_to(self.dir)), **est, "accepted": _now()},
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
        )
        if res.timing.get("meat_seconds"):
            self.pass_rate = self.data.manifest["rows"] * est["columns"] / res.timing["meat_seconds"]
        recs = {}
        for y in res.outcomes:
            rec = session_record(res, y)
            name = "result.estimation.json" if len(res.outcomes) == 1 else f"result.{y}.estimation.json"
            rec = ef_write(rec, wd / name, by={"name": "dfeols", "version": _version()})
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
                },
            )
        self._save_state()
        return {"model": entry}

    def estimate(self, req: dict) -> dict:
        """A model's cost before it runs: passes x columns x the pass throughput measured in
        this session (before the first pass: the demeaning's, which over-states the time)."""
        _, ys, xs = _model_terms(req["formula"], self.spec.fe)
        vcov = req.get("vcov") or (["CRV1"] if req.get("cluster") else ["iid"])
        vcov = [vcov] if isinstance(vcov, str) else list(vcov)
        passes = int(any(v in ("hetero", "CRV1") for v in vcov) and bool(xs))
        cols = len(xs) + len(ys) + 1 + bool(req.get("cluster"))
        rows = self.data.manifest["rows"]
        if not passes:
            return {"passes": 0, "columns": cols, "estimate_seconds": 0.0, "basis": "no pass"}
        if self.pass_rate:
            secs, basis = rows * cols / self.pass_rate, "measured in this session"
        else:
            rate, k = self.calibration or self._calibrate()
            slots = max(int(self.executor.slots()[0] or 1), 1)
            secs = rows * cols / (rate * slots) + TASK_OVERHEAD * len(self.data.shards) / slots
            basis = f"read rate of {k} shard{'s' if k > 1 else ''}, {slots} worker{'s' if slots > 1 else ''}"
        return {"passes": passes, "columns": cols, "estimate_seconds": secs, "basis": basis}

    def _calibrate(self, k: int = 3) -> tuple[float, int]:
        """Row-columns read per second by one process, from the largest few shard files
        (a model's pass is mostly reading its columns)."""
        d = self.data
        parts = sorted(d.shards, key=lambda s: -Path(d.part(s)).stat().st_size)[:k]
        cols = [*d.names, WEIGHT_COL]
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

    def write_collection(self, path=None) -> Path:
        path = Path(path) if path else self.dir / COLLECTION_NAME
        _write_session_collection(
            self.dir, self.models, self.events, self.session_block(), self.provenance_block(), path
        )
        return path


def _version() -> str:
    from . import __version__

    return __version__


def _write_session_collection(sdir: Path, models, events, session: dict, provenance: dict | None, path: Path):
    import estimation_format as ef

    recs, labels = [], []
    for mdl in models:
        for y, r in mdl["records"].items():
            recs.append(ef.read(sdir / r["path"], nonfinite="string"))
            labels.append(mdl["label"] if len(mdl["records"]) == 1 else f"{mdl['label']}.{y}")
    coll = ef.new_collection(
        recs,
        kind="session",
        labels=labels,
        session=session,
        events=list(events),
        provenance=provenance,
        title=f"session {session['fixed']['name']}",
        producer={"name": "dfeols", "version": _version()},
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


def _log_timing(root: Path, manifest: dict) -> None:
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
            },
        )
    except OSError:
        pass


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
        poll: float = 0.2,
        timeout: float | None = None,
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
        s = cls(name, root=root, manager=mgr, thread=th)
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
            e = a["estimate_seconds"]
            self._say(
                "no pass over the data (from the Gram matrix)"
                if not a["passes"]
                else f"estimated {_secs(e)}: {a['passes']} pass over {a['columns']} columns ({a['basis']})"
            )
        self._show_status(self.dir / a["model_dir"] / "status.json")

    def _request(self, type: str, timeout: float | None = None, watch=None, **payload) -> dict:
        rid = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        req = {"id": rid, "type": type, "client": self.client, "sent": _now(), **payload}
        _write_json(self.dir / "requests" / f"{rid}.json", req)
        resp_path = self.dir / "responses" / f"{rid}.json"
        t0 = time.time()
        while not resp_path.exists():
            if timeout is not None and time.time() - t0 > timeout:
                raise TimeoutError(f"no response to {type} after {timeout:g} s")
            st = self.state()
            if st and st["status"] not in ("ready", "starting") and not resp_path.exists():
                raise SessionError(f"session {self.name} is {st['status']}")
            if watch:
                watch(rid)
            time.sleep(0.1)
        if watch:  # a quick request may finish before the first look
            watch(rid)
        resp = _read_json(resp_path)
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
    ) -> Result:
        """Fit `formula` (y ~ x1 + x2, declared variables; the FEs are the session's)."""
        r = self._request(
            "fit",
            timeout=timeout,
            watch=self._watch_fit,
            formula=formula,
            vcov=vcov,
            cluster=cluster,
            label=label,
        )
        m = r["model"]
        if m["passes"] or m["seconds"] >= 1:
            self._say(f"{m['label']}: done in {_secs(m['seconds'])}")
        return Result.load(self.dir / m["dir"] / "result.json")

    def record(self, res: Result, y: str | None = None) -> dict:
        """The estimation record the session wrote for a model."""
        import estimation_format as ef

        name = (
            "result.estimation.json"
            if len(res.outcomes) == 1
            else f"result.{y or next(iter(res.outcomes))}.estimation.json"
        )
        return ef.read(Path(res.workdir) / name)

    def status(self) -> dict:
        return self._request("status")["session"]

    def export(self, path=None) -> Path:
        """Write the session collection now (status open); returns its path."""
        return Path(self._request("export", path=str(path) if path else None)["path"])

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


__all__ = [
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
        est = (0.0, 0.0)
    else:
        est = estimate_seconds(dm_root, ds.n_rows, len(spec.variables))
        if est:
            print(
                f"demeaning {len(spec.variables)} variables: estimated {fmt_td(est[0])} to {fmt_td(est[1])}"
            )
        else:
            print(f"demeaning {len(spec.variables)} variables: no earlier runs to estimate from")
        size = ds.n_rows * (len(spec.variables) + 1 + len(spec.clusters)) * 8
        print(f"demeaned data: ~{size / 1e9:.1f} GB in {where if not args.no_persist else root / args.name}")
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
                    "demeaned",
                )
                if k in st
            },
            indent=1,
        )
    )


def cmd_fit(args) -> None:
    s = Session.attach(args.name, root=args.root, profile=args.profile, timeout=args.timeout)
    try:
        vcov = [v.strip() for v in args.vcov.split(",")] if args.vcov else None
        r = s.fit(args.formula, vcov=vcov, cluster=args.cluster, label=args.label)
        print(r.summary())
    finally:
        s.close()


def cmd_export(args) -> None:
    s = Session.attach(args.name, root=args.root, profile=args.profile, timeout=args.timeout)
    try:
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
    q = ss.add_parser("status", help="the session's state")
    common(q)
    q = ss.add_parser("fit", help="fit a model in a running session")
    common(q)
    q.add_argument("formula")
    q.add_argument("--vcov", help="iid, hetero, CRV1 (comma-separated)")
    q.add_argument("--cluster")
    q.add_argument("--label")
    q.add_argument("--timeout", type=float)
    q = ss.add_parser("export", help="write the session collection now")
    common(q)
    q.add_argument("--path")
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
        "fit": cmd_fit,
        "export": cmd_export,
        "persist": cmd_persist,
        "stop": cmd_stop,
    }[args.session_cmd](args)
