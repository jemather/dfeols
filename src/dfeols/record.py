# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""dfeols results as `estimation` records (docs/design/estimation-format.md, Next step 5).

    run_info(ds, executor, ...)   what fit() knows about the run: provenance and the
                                  dfeols-only facts, stored on Result.run
    term_rows(formula, levels)    every reported term with its canonical name
    to_estimation(result, y, ...) the record (a dict) of one response of a Result
    write_estimation(result, path, y, ...), write_estimations(result, path, ...)

A record claims inference/0.1, linear-fe/0.1 and dfeols/0.1 (a fit without FEs claims
inference/0.1 only and keeps the dfeols block). A fit with several responses gives one
record per response (Ted, 2026-10-24), as fixest and pyfixest return one model per
response; the records share `extensions.dfeols.fit` (id and all responses).

Provenance is captured on the coordinator when the fit finishes. SLURM fields come
from the job's environment variables (no sacct call), so End, Elapsed, MaxRSS and
State, which only exist after the job, are absent.
"""

from __future__ import annotations

import datetime as _dt
import functools
import getpass
import hashlib
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import estimation_format as ef
import numpy as np

from .vcov import df_k as _df_k

PROFILES = ["inference/0.1", "linear-fe/0.1", "dfeols/0.1"]
# The fields the dfeols/0.1 profile marks identifying (tests/test_record.py keeps the two
# equal). Redaction always takes them, so a record that doesn't claim dfeols/0.1 (a fit
# without FEs) loses them too, and so does one redacted without the schemas.
IDENTIFYING = (
    "extensions.dfeols.shards.folder",
    "extensions.dfeols.dask.scheduler_address",
    "extensions.dfeols.dask.profile.account",
    "extensions.dfeols.dask.profile.project",
    "extensions.dfeols.dask.profile.scratch",
    "extensions.dfeols.dask.profile.source",
    "extensions.dfeols.dask.profile.prologue",
    "extensions.dfeols.dask.profile.env",
)
REPOSITORY = "https://github.com/jemather/dfeols"
# dfeols vcov names -> the format's vocabulary (hetero is N/(N-K) * sandwich: HC1)
VCOV_TYPES = {"iid": "iid", "hetero": "HC1", "CRV1": "CRV1"}
# Environment variables that change results or speed; never a full dump
ENV_ALLOWLIST = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "PYTHONHASHSEED",
    "TZ",
)
# Python packages listed when imported (role "dependency")
PACKAGES = ("numpy", "scipy", "polars", "pyarrow", "pandas", "pyfixest", "dask", "distributed", "jsonschema")
# SLURM job environment -> sacct field names
SLURM_ENV = {
    "SLURM_JOB_ID": "JobID",
    "SLURM_JOB_NAME": "JobName",
    "SLURM_JOB_PARTITION": "Partition",
    "SLURM_JOB_ACCOUNT": "Account",
    "SLURM_JOB_QOS": "QOS",
    "SLURM_JOB_NODELIST": "NodeList",
    "SLURM_JOB_USER": "User",
    "SLURM_SUBMIT_DIR": "WorkDir",
}
CPU_FLAGS = ("sse4_2", "avx", "avx2", "fma", "avx512f", "avx512dq", "avx512bw", "avx512vl")


def _utc(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def identifying_paths(record: dict) -> list[str]:
    """The fields `redact=True` removes: those marked identifying in the core and each
    profile the record claims (the format's DEFAULT_IDENTIFYING without the schemas), plus
    the dfeols block's."""
    schemas = ef.load_schemas()
    base = ef.record_identifying_paths(record, schemas) if schemas else ef.DEFAULT_IDENTIFYING
    return sorted({*base, *IDENTIFYING})


def check_profile(record: dict) -> list[ef.Issue]:
    """Built-in check of dfeols/0.1 (linear-fe's plus the dfeols block), for
    `estimation_format.validate(..., profiles={"dfeols/0.1": check_profile})` when
    jsonschema is missing; with it, the profile's schema is the check."""
    issues = ef.records.PROFILES["linear-fe/0.1"](record)
    if "dfeols" not in (record.get("extensions") or {}):
        issues.append(ef.Issue("error", "extensions.dfeols", "required"))
    return issues


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------- term names


def _level(v: Any) -> str:
    return str(v.item() if isinstance(v, np.generic) else v)


def term_rows(formula, cat_levels: dict[str, list], has_fe: bool) -> list[dict]:
    """Every reported regressor in model order (as worker.plan_columns names them), with
    its canonical form: R rendering, except factor x continuous in Stata form."""
    rows: list[dict] = []
    if formula.intercept and not has_fe:
        rows.append({"term": "Intercept", "canonical": {"name": "(Intercept)", "factors": []}})
    for t in formula.terms:
        conts = [{"variable": v} for v in t.conts]
        if t.cat is None:
            name = ":".join(t.conts)
            rows.append({"term": name, "canonical": {"name": name, "factors": conts}})
            continue
        levels = list(cat_levels[t.cat])
        ref = t.ref if t.ref is not None else (levels[0] if t.drop_first and levels else None)
        if t.ref is not None:
            levels = [lv for lv in levels if lv != t.ref]
        elif t.drop_first:
            levels = levels[1:]
        for lv in levels:
            fac = {"variable": t.cat, "level": _level(lv)}
            if ref is not None:
                fac["reference"] = _level(ref)
            term = f"{t.cat}::{lv}" + (":" + ":".join(t.conts) if t.conts else "")
            if t.conts:
                name = f"{_level(lv)}.{t.cat}" + "".join(f"#c.{v}" for v in t.conts)
            else:
                name = f"{t.cat}{_level(lv)}"
            rows.append({"term": term, "canonical": {"name": name, "factors": [fac, *conts]}})
    return rows


def factor_terms(formula, cat_levels: dict[str, list], max_levels: int = 1000) -> list[dict]:
    """specification.factor_terms: each i() term's variable, levels (when at most
    `max_levels`) and reference level."""
    out, seen = [], set()
    for t in formula.terms:
        if t.cat is None:
            continue
        levels = list(cat_levels[t.cat])
        ref = t.ref if t.ref is not None else (levels[0] if t.drop_first and levels else None)
        key = (t.cat, repr(ref))
        if key in seen:
            continue
        seen.add(key)
        d: dict[str, Any] = {"variable": t.cat, "n_levels": len(levels)}
        if len(levels) <= max_levels:
            d["levels"] = [_level(v) for v in levels]
        d["reference"] = None if ref is None else _level(ref)
        out.append(d)
    return out


# ---------------------------------------------------------------- provenance


@functools.lru_cache(maxsize=1)
def dfeols_source() -> dict:
    """dfeols's git commit and dirty flag: from the checkout it runs from, else from
    pip's direct_url.json (an install from git records the commit)."""
    here = Path(__file__).resolve().parent
    try:
        run = functools.partial(subprocess.run, capture_output=True, text=True, timeout=10, check=True)
        top = run(["git", "-C", str(here), "rev-parse", "--show-toplevel"]).stdout.strip()
        if (Path(top) / "src" / "dfeols" / "__init__.py").exists():
            commit = run(["git", "-C", top, "rev-parse", "HEAD"]).stdout.strip()
            dirty = bool(
                run(["git", "-C", top, "status", "--porcelain", "--untracked-files=no"]).stdout.strip()
            )
            return {"repository": REPOSITORY, "commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        from importlib.metadata import distribution

        d = json.loads(distribution("dfeols").read_text("direct_url.json") or "{}")
        vcs = d.get("vcs_info") or {}
        if vcs.get("commit_id"):
            return {"repository": d.get("url", REPOSITORY), "commit": vcs["commit_id"]}
    except Exception:
        pass
    return {}


def _software() -> list[dict]:
    from importlib.metadata import PackageNotFoundError, version

    from . import __version__

    src = dfeols_source()
    me: dict[str, Any] = {"name": "dfeols", "version": __version__, "role": "estimator"}
    if src.get("commit"):
        me["purl"] = f"pkg:github/jemather/dfeols@{src['commit']}"
        me["source"] = src
    out = [
        me,
        {
            "name": "python",
            "version": platform.python_version(),
            "role": "language",
            "attributes": {"implementation": platform.python_implementation()},
        },
    ]
    for name in PACKAGES:
        if name not in sys.modules:
            continue
        try:
            v = version(name)
        except PackageNotFoundError:
            continue
        out.append({"name": name, "version": v, "purl": f"pkg:pypi/{name}@{v}", "role": "dependency"})
    return out


def _cpu() -> dict:
    hw: dict[str, Any] = {}
    try:
        info = Path("/proc/cpuinfo").read_text()
        for line in info.splitlines():
            if line.startswith("model name") and "cpu_model" not in hw:
                hw["cpu_model"] = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and "cpu_flags" not in hw:
                have = set(line.split(":", 1)[1].split())
                hw["cpu_flags"] = [f for f in CPU_FLAGS if f in have]
        hw["sockets"] = len({ln for ln in info.splitlines() if ln.startswith("physical id")}) or None
    except OSError:
        hw["cpu_model"] = platform.processor() or platform.machine()
    hw["threads"] = os.cpu_count()
    try:
        hw["memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    return {k: v for k, v in hw.items() if v is not None}


def _blas() -> dict | None:
    try:
        cfg = np.show_config(mode="dicts")
        b = cfg["Build Dependencies"]["blas"]
        return {k: b[k] for k in ("name", "version") if k in b}
    except Exception:
        return None


def _environment() -> dict:
    env: dict[str, Any] = {"os": platform.system(), "kernel": platform.release()}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                env["os"] = line.split("=", 1)[1].strip('"')
    except OSError:
        pass
    env["machine"] = platform.machine()
    env["hostname"] = socket.gethostname()
    env["timezone"] = time.strftime("%Z")
    env["hardware"] = _cpu()
    blas = _blas()
    if blas:
        env["blas"] = blas
    try:
        from threadpoolctl import threadpool_info

        env["threadpools"] = threadpool_info()
    except ImportError:
        pass
    env["variables"] = {k: os.environ[k] for k in ENV_ALLOWLIST if k in os.environ}
    return env


def _invocation() -> dict:
    argv = list(sys.argv)
    inv: dict[str, Any] = {
        "command_line": shlex.join([Path(sys.executable).name, *argv]),
        "argv": argv,
        "working_directory": os.getcwd(),
        "mode": "interactive" if hasattr(sys, "ps1") or "ipykernel" in sys.modules else "batch",
    }
    script = Path(argv[0]) if argv and argv[0] not in ("", "-c", "-m") else None
    if script is not None and script.is_file():
        inv["script_path"] = str(script.resolve())
        digest = _sha256(script)
        if digest:
            inv["script_sha256"] = digest
    return inv


def _scheduler() -> dict | None:
    if "SLURM_JOB_ID" not in os.environ:
        return None
    s: dict[str, Any] = {"type": "slurm", "source": "environment"}
    s.update({f: os.environ[e] for e, f in SLURM_ENV.items() if e in os.environ})
    if "SLURM_CPUS_ON_NODE" in os.environ:
        s["AllocCPUS"] = os.environ["SLURM_CPUS_ON_NODE"]
    return s


def _dask(executor) -> tuple[dict | None, list[dict]]:
    """The extension's dask block (scheduler address, resolved profile) and the workers
    (host, threads, memory) for provenance.execution.workers."""
    client = getattr(executor, "client", None)
    if client is None:
        return None, []
    out: dict[str, Any] = {}
    workers: list[dict] = []
    try:
        info = client.scheduler_info()
        out["scheduler_address"] = info.get("address") or client.scheduler.address
        for w in (info.get("workers") or {}).values():
            workers.append(
                {
                    k: v
                    for k, v in {
                        "host": w.get("host"),
                        "cores": w.get("nthreads"),
                        "memory_bytes": w.get("memory_limit"),
                    }.items()
                    if v is not None
                }
            )
    except Exception:  # a closed client still leaves a result
        pass
    prof = getattr(executor, "profile", None)
    if prof is not None:
        out["profile"] = {
            k: v
            for k, v in vars(prof).items()
            if isinstance(v, (str, int, float, bool, dict, list, type(None)))
        }
    return out, workers


def _source_input(src: dict | None) -> list[dict]:
    """provenance.data entry for what prep read (dfeols_meta.json `source`): name, location
    (identifying), checksum, rows; per-file names, sizes and sha256 without their paths."""
    if not src or src.get("kind") == "lazyframe":
        return []
    entry: dict[str, Any] = {"role": "source"}
    files = src.get("files") or []
    if src.get("kind") == "frame":
        entry["name"] = "in-memory frame"
    elif len(files) == 1:
        entry.update(name=files[0]["name"], location=files[0]["path"])
    elif files:
        parent = os.path.commonpath([str(Path(f["path"]).parent) for f in files])
        entry.update(name=Path(parent).name, location=parent)
        entry["files"] = [{k: f[k] for k in ("name", "bytes", "sha256", "rows") if k in f} for f in files]
    if "checksum" in src:
        entry["checksum"] = src["checksum"]
    if "rows" in src:
        entry["rows"] = src["rows"]
    if "columns" in src:
        entry["columns"] = src["columns"]
    return [entry]


def run_info(ds, executor, *, started: float, finished: float, cpu_seconds: float | None) -> dict:
    """What fit() records about where and how it ran: `provenance` (core sections) and
    `dfeols` (dataset and executor facts for the extension). JSON-ready."""
    meta_sha = _sha256(Path(ds.root) / "dfeols_meta.json")
    dask, workers = _dask(executor)
    execution: dict[str, Any] = {
        "started": _utc(started),
        "finished": _utc(finished),
        "timezone": time.strftime("%Z"),
        "elapsed_seconds": finished - started,
        "invocation": _invocation(),
    }
    if cpu_seconds is not None:
        execution["processor_seconds"] = cpu_seconds
    sched = _scheduler()
    if sched:
        execution["scheduler"] = sched
    if workers:
        execution["workers"] = workers
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = None
    prov = {
        "software": _software(),
        "environment": _environment(),
        "execution": execution,
        "agent": {k: v for k, v in {"user": user, "host": socket.gethostname()}.items() if v},
        "data": [
            *_source_input(ds.meta.get("source")),
            {
                "role": "estimation",
                "name": Path(ds.root).name,
                "location": str(Path(ds.root).resolve()),
                "rows": int(ds.n_rows),
                "columns": list(ds.columns),
            },
        ],
    }
    shards: dict[str, Any] = {
        "folder": str(Path(ds.root).resolve()),
        "prep_version": str(ds.meta.get("prep_version", "")),
        "partition_key": ds.partition,
        "n_shards": len(ds.shards),
    }
    if ds.meta.get("target_rows") is not None:
        shards["target_rows"] = int(ds.meta["target_rows"])
    if meta_sha:
        shards["meta_sha256"] = meta_sha
    dfe: dict[str, Any] = {"shards": shards, "executor": type(executor).__name__}
    if dask:
        dfe["dask"] = dask
    return ef.to_json({"provenance": prov, "dfeols": dfe})


# ---------------------------------------------------------------- the record


def _t_stats(b, se, df):
    from scipy import stats

    with np.errstate(divide="ignore", invalid="ignore"):
        t = b / se
    return t, 2 * stats.t.sf(np.abs(t), df), stats.t.ppf(0.975, df)


def _call(result) -> str:
    o = result.options
    args = [repr(result.formula)]
    for k in (
        "cluster",
        "vcov",
        "weights",
        "drop_singletons",
        "fe_tol",
        "fe_maxiter",
        "fe_method",
        "crossing_fe",
        "max_dummy_levels",
        "collin_tol",
    ):
        if k in o:
            args.append(f"{k}={o[k]!r}")
    return f"dfeols.fit(ds, {', '.join(args)})"


def _ssc(vtype: str, N: int, k: int, dfk: int, G: int | None, nested: list[bool]) -> dict:
    K_counts = {
        "K": dfk,
        "regressors": k,
        "absorbed": dfk - k,
        "rule": "fixest k_fixef='nonnested': FE levels nested in the cluster are not counted "
        "(each such FE counts 1); one level per additional FE is redundant",
    }
    settings = {"k_adj": True, "k_fixef": "nonnested", "G_adj": True, "G_df": "min"}
    if vtype == "CRV1":
        return {
            "convention": "fixest",
            "formula": "G/(G-1) * (N-1)/(N-K)",
            "factors": {"G/(G-1)": G / (G - 1), "(N-1)/(N-K)": (N - 1) / (N - dfk), "settings": settings},
            "K_counts": {**K_counts, "fe_nested_in_cluster": nested},
        }
    if vtype == "hetero":
        return {
            "convention": "fixest",
            "formula": "N/(N-K)",
            "factors": {"N/(N-K)": N / (N - dfk), "settings": settings},
            "K_counts": K_counts,
        }
    return {"convention": "fixest", "formula": "RSS/(N-K)", "factors": {"N-K": N - dfk}, "K_counts": K_counts}


def _response(result, y: str | None) -> str:
    ys = list(result.outcomes)
    if y is None:
        if len(ys) > 1:
            raise ValueError(f"this fit has {len(ys)} responses ({', '.join(ys)}): pass y=, one record each")
        return ys[0]
    if y not in result.outcomes:
        raise KeyError(f"no response {y!r}; the fit has {', '.join(ys)}")
    return y


def to_estimation(
    result,
    y: str | None = None,
    *,
    publication: dict | None = None,
    title: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    provenance: bool = True,
) -> dict:
    """The `estimation` record of response `y` of `result` (optional when the fit has
    one response), unencoded (numpy arrays allowed).

    publication  a `publication` block: citation (CSL-JSON), locator, links
    provenance   False leaves out provenance (software, environment, execution, ...)

    Write it with `estimation_format.write` (or `Result.write_estimation`), which encodes it,
    adds history and the core digest, spills large structures and can redact."""
    from . import __version__

    r = result
    y = _response(r, y)
    o = r.outcomes[y]
    opts = r.options or {}
    run = r.run or {}
    k = len(r.coefnames)
    nested_cl = list(run.get("fe_nested_in_cluster") or [])
    dfk_plain = _df_k(k, r.fe_levels, None)
    dfk_cl = _df_k(k, r.fe_levels, nested_cl) if r.cluster else None
    df_resid = r.N - dfk_plain
    n_sing = int(r.N_dropped) if opts.get("drop_singletons", True) else 0
    has_fe = bool(r.fe)
    terms = r.terms or [{"term": t} for t in r.coefnames]

    messages = []
    status = "ok"
    if n_sing:
        messages.append(
            {
                "level": "info",
                "code": "singletons",
                "message": f"{n_sing:,} singleton rows dropped",
                "source": "dfeols",
            }
        )
    if r.collinear:
        messages.append(
            {
                "level": "info",
                "code": "collinear",
                "message": f"dropped as collinear: {', '.join(r.collinear)}",
                "source": "dfeols",
            }
        )
    if run.get("converged") is False:
        status = "warning"
        messages.append(
            {
                "level": "warning",
                "code": "not_converged",
                "message": "FE demeaning did not converge in some shards",
                "source": "dfeols",
            }
        )

    meta: dict[str, Any] = {
        "status": status,
        "estimator": {"family": "linear", "method": "wls" if opts.get("weights") else "ols"},
        "responses": [y],
        # linear-fe needs at least one FE; a fit without any is plain inference
        "profiles": list(PROFILES) if has_fe else ["inference/0.1"],
        "title": title or (r.label if len(r.outcomes) == 1 else f"{r.label} ({y})"),
    }
    if messages:
        meta["messages"] = messages
    if tags:
        meta["tags"] = list(tags)
    if notes:
        meta["notes"] = notes

    # ---- parameters and vcov
    b = np.asarray(o["coef"], float)
    V, df_t = o["vcov"][r.vcov_default]
    se = np.sqrt(np.diag(V))
    t, p, q = _t_stats(b, se, df_t)
    pos = {name: i for i, name in enumerate(r.coefnames)}
    params = []
    for row in terms:
        out = dict(row)
        if row["term"] in pos:
            i = pos[row["term"]]
            out.update(
                estimate=b[i],
                std_error=se[i],
                statistic=t[i],
                p_value=p[i],
                conf_low=b[i] - q * se[i],
                conf_high=b[i] + q * se[i],
            )
        elif row["term"] in r.collinear:
            out.update(estimate=None, std_error=None, statistic=None, p_value=None, dropped="collinear")
        else:
            continue
        params.append(out)
    vcovs = []
    for vt in [v for v in opts.get("vcov", []) if v in o["vcov"]] or list(o["vcov"]):
        Vt, dft = o["vcov"][vt]
        entry: dict[str, Any] = {
            "kind": "matrix",
            "labels": list(r.coefnames),
            "data": Vt,
            "type": VCOV_TYPES.get(vt, vt),
        }
        if vt == "CRV1":
            entry.update(cluster=[r.cluster], n_clusters={r.cluster: r.G}, n_clusters_min=r.G)
        entry["ssc"] = _ssc(vt, r.N, k, dfk_cl if vt == "CRV1" else dfk_plain, r.G, nested_cl)
        entry.update(df_t=dft, default=vt == r.vcov_default)
        vcovs.append(entry)

    # ---- metrics
    metrics: dict[str, Any] = {
        "nobs": r.N,
        "nobs_full": r.N + n_sing,
        "n_singletons": n_sing,
        "df_residual": df_resid,
        "df_model": k,
        "df_absorbed": dfk_plain - k,
    }
    if r.cluster and dfk_cl is not None and dfk_cl != dfk_plain:
        metrics["df_absorbed_nested"] = dfk_plain - dfk_cl
    if run.get("sum_weights") is not None:
        metrics["sum_weights"] = run["sum_weights"]
    metrics["rss"] = o["rss"]
    if "tss" in o:
        metrics["tss_within" if run.get("n_local_fes", 1) and has_fe else "tss"] = o["tss"]
    if "r2" in o:
        metrics["r2"] = o["r2"]
    if "r2_within" in o:
        metrics["r2_within"] = {
            "value": o["r2_within"],
            "definition": "1 - RSS / TSS of the response demeaned by the absorbed FEs "
            "(dummy FEs are regressors in RSS, not in TSS)",
        }
    metrics["sigma"] = float(np.sqrt(o["rss"] / df_resid)) if df_resid > 0 else None
    metrics["conf_level"] = 0.95

    structures: dict[str, Any] = {"vcov": vcovs}
    absorption = run.get("absorption") or {}
    structures["fixed_effects"] = [
        {
            "kind": "object",
            "name": name,
            "levels": int(L),
            **({"nested_in": [r.cluster]} if r.cluster and i < len(nested_cl) and nested_cl[i] else {}),
            **(
                {"redundant": 0 if i == 0 else 1, "redundant_method": "one_per_additional_fe"}
                if len(r.fe) > 1
                else {}
            ),
        }
        for i, (name, L) in enumerate(zip(r.fe, r.fe_levels))
    ]
    n_local = sum(1 for a in absorption.values() if a.get("method") == "local")
    numerics: dict[str, Any] = {
        "kind": "object",
        "algorithm": {0: "none", 1: "one-pass group demeaning"}.get(
            n_local,
            "exact two-way (Schur complement), alternating projections as fallback"
            if n_local == 2 and opts.get("fe_method", "auto") != "iterate"
            else "alternating projections (pyfixest core)",
        ),
        "solver": "Cholesky on the reduced normal equations",
        "tolerance": opts.get("fe_tol"),
        "max_iterations": opts.get("fe_maxiter"),
        "converged": run.get("converged"),
        "collinearity_tolerance": opts.get("collin_tol"),
        "collinear_terms": list(r.collinear),
    }
    # two FEs with fe_method auto/exact may fall back to iterating, so leave it unsaid
    if n_local != 2 or opts.get("fe_method") == "iterate":
        numerics["iterative"] = n_local >= 2
    structures["numerics"] = {k_: v for k_, v in numerics.items() if v is not None}

    # ---- specification
    sample = run.get("sample") or {}
    dropped = {k_: v for k_, v in (sample.get("dropped") or {}).items() if v or k_ == "missing"}
    spec: dict[str, Any] = {
        "call": _call(r),
        "formula": {"text": r.formula, "syntax": "fixest"},
        "terms": [row["term"] for row in terms],
        "weights": {"variable": opts["weights"], "type": "analytic"} if opts.get("weights") else None,
        "fixed_effects": list(r.fe),
        "clusters": [r.cluster] if r.cluster else [],
        "sample": {
            "restriction": None,
            **({"rows_input": sample["rows_input"]} if "rows_input" in sample else {}),
            "dropped": {**dropped, "singletons": n_sing},
        },
        "options": {
            "drop_singletons": bool(opts.get("drop_singletons", True)),
            **{k_: v for k_, v in opts.items() if k_ not in ("cluster", "weights", "drop_singletons")},
        },
    }
    if run.get("factor_terms"):
        spec["factor_terms"] = run["factor_terms"]

    record = ef.new_record(params, meta=meta, specification=spec, metrics=metrics, structures=structures)
    if provenance and run.get("provenance"):
        record["provenance"] = run["provenance"]
    if publication:
        record["publication"] = publication

    ext: dict[str, Any] = {
        "schema": "dfeols/0.1",
        "by": {"name": "dfeols", "version": __version__},
        "added": _utc(time.time()),
        # the fit this record came from; a fit with several responses writes one record each
        "fit": {"id": run.get("fit_id", r.label), "label": r.label, "responses": list(r.outcomes)},
    }
    dfe = run.get("dfeols") or {}
    ext.update({k_: v for k_, v in dfe.items() if k_ in ("shards", "dask", "executor")})
    if run.get("row_key"):
        ext["row_key"] = run["row_key"]
    if absorption:
        ext["absorption"] = absorption
    if r.timing:
        ext["timing"] = r.timing
    src = dfeols_source() if provenance else {}
    ext["replay"] = {"call": spec["call"], **({"commit": src["commit"]} if src.get("commit") else {})}
    record["extensions"] = {"dfeols": ext}
    return record


def record_paths(result, path: str | os.PathLike) -> dict[str, Path]:
    """Where each response's record goes: `path` itself for a one-response fit, else
    `<name>.<response>.estimation.json` next to it."""
    path = Path(path)
    ys = list(result.outcomes)
    if len(ys) == 1:
        return {ys[0]: path}
    stem = path.name.removesuffix(".json").removesuffix(".estimation")
    return {y: path.with_name(f"{stem}.{re.sub(r'[^A-Za-z0-9._-]', '_', y)}.estimation.json") for y in ys}


def write_estimation(
    result,
    path: str | os.PathLike,
    y: str | None = None,
    *,
    redact: bool = False,
    digest: bool = True,
    spill_large: bool = True,
    **kwargs: Any,
) -> dict:
    """Write the record of response `y` to `path` (`<name>.estimation.json`); returns it
    as written. `redact=True` removes the identifying fields; other keywords go to
    `to_estimation`. For every response of a fit, see `write_estimations`."""
    from . import __version__

    rec = to_estimation(result, y, **kwargs)
    return ef.write(
        rec,
        path,
        digest=digest,
        spill_large=spill_large,
        redact_identifying=identifying_paths(rec) if redact else False,
        by={"name": "dfeols", "version": __version__},
    )


def write_estimations(result, path: str | os.PathLike, **kwargs: Any) -> dict[str, dict]:
    """One record per response, at `record_paths(result, path)`; returns them by response."""
    return {y: write_estimation(result, p, y, **kwargs) for y, p in record_paths(result, path).items()}
