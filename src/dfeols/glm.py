# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""GLMs with high-dimensional fixed effects by IRLS: Poisson (log link) and binary cloglog.

    r = dfeols.fit_glm(ds, "unf ~ lfeat:i(tbin) + lposts | id_f + id_a + week",
                       family="poisson", cluster="id_a")

Each iteration is an ordinary weighted dfeols fit of the working response on the same
regressors and FEs,

    z = eta + (y - mu) / mu'(eta),     w = mu'(eta)^2 / V(mu),

so every FE path (local demeaning, crossing dummies, sparse sums, plan C) and every
executor work unchanged. The fit's round 2 already forms each row's residual net of every
FE, so the new linear predictor eta = z - resid (= xb + d, all FEs included) is computed in
the shard and written straight into the next iteration's working columns: per-shard
"sidecar" Parquet files aligned to the shard's raw rows, joined at load time. Nothing is
gathered on the coordinator but each shard's deviance.

Before the first iteration, rows in FE groups whose outcome never varies are dropped
(separation: a Poisson group with y = 0 throughout, a binary one with all 0 or all 1; their
FE runs to -inf/+inf and their rows carry no information), repeated until no group changes,
as ppmlhdfe and fixest do. Regressor-induced separation is not checked.

Standard errors are those of the converged weighted fit: its bread is (X~' W X~)^-1 and its
CRV1 / HC1 meat is built from w * (z - eta) = (y - mu) mu' / V, the GLM score, so the
clustered and heteroskedasticity-robust sandwiches are the GLM ones with fixest's small-sample
factors; the iid VCE is the bread times (N - 1)/(N - K) (dispersion 1 for both families).
All three equal pyfixest's fepois.

Convergence: |dev - dev_prev| / (0.1 + |dev|) < tol (fixest's rule), default 1e-8.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .formula import parse

log = logging.getLogger(__name__)

Z, W = "_glm_z", "_glm_w"


@dataclass(frozen=True)
class Family:
    name: str
    linkinv: Callable
    mu_eta: Callable  # d mu / d eta
    variance: Callable
    deviance: Callable  # per-row unit deviance
    start: Callable  # (y, ybar) -> eta0
    eta_bounds: tuple[float, float]
    binary: bool

    def working(self, y, eta, mu):
        d = self.mu_eta(eta)
        return eta + (y - mu) / d, d * d / self.variance(mu)


def _xlogy(x, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0) / y), 0.0)


_EPS = 1e-15

POISSON = Family(
    "poisson",
    linkinv=np.exp,
    mu_eta=np.exp,
    variance=lambda mu: mu,
    deviance=lambda y, mu: 2.0 * (_xlogy(y, mu) - (y - mu)),
    start=lambda y, ybar: np.log((y + ybar) / 2.0),
    eta_bounds=(-50.0, 50.0),
    binary=False,
)


def _cll_inv(eta):
    return np.clip(-np.expm1(-np.exp(eta)), _EPS, 1 - _EPS)


CLOGLOG = Family(
    "cloglog",
    linkinv=_cll_inv,
    mu_eta=lambda eta: np.maximum(np.exp(eta - np.exp(eta)), _EPS),
    variance=lambda mu: mu * (1.0 - mu),
    deviance=lambda y, mu: -2.0 * (y * np.log(mu) + (1.0 - y) * np.log1p(-mu)),
    start=lambda y, ybar: np.log(-np.log1p(-(y + 0.5) / 2.0)),
    eta_bounds=(-30.0, 3.6),  # mu within 2e-16 of 1 beyond
    binary=True,
)

FAMILIES = {"poisson": POISSON, "cloglog": CLOGLOG}


# ---------------------------------------------------------------------------- shard tasks
def shard_glm_init(sid: int, files: list[str], spec: dict) -> dict:
    """One pass over a shard's raw rows (the order dfeols's loader reads them in).

    mode "scan": per FE, the alive rows' levels with their outcome sums and counts
    (sparse), for the separation check. mode "write": the first working columns z0, w0
    (null on rows that are missing, invalid or separated) as the shard's sidecar."""
    t0 = time.perf_counter()
    fam = FAMILIES[spec["family"]]
    cols = sorted(set([spec["y"], *spec["vars"], *spec["fes"]]))
    df = pl.read_parquet(files, columns=cols)
    ok = pl.all_horizontal(
        [pl.col(c).is_not_null() & (pl.col(c).is_finite() if df.schema[c].is_float() else True) for c in cols]
    )
    valid = df.select(ok).to_series().to_numpy()
    y = df[spec["y"]].cast(pl.Float64).fill_null(0.0).to_numpy()
    bad = valid & (
        (y < 0) | ((y != np.round(y)) if fam.binary else False) | ((y > 1) if fam.binary else False)
    )
    if bad.any():
        raise ValueError(
            f"{spec['family']}: outcome {spec['y']!r} must be {'0 or 1' if fam.binary else '>= 0'} "
            f"({int(bad.sum())} rows in shard {sid} are not)"
        )
    alive = valid.copy()
    for c in spec["fes"]:
        dead = spec["dead"].get(c)
        if dead is not None and len(dead):
            alive &= ~np.isin(df[c].to_numpy(), dead)
    out = {"shard": sid, "n": int(alive.sum()), "sum_y": float(y[alive].sum())}
    if spec["mode"] == "scan":
        out["levels"] = {}
        for c in spec["fes"]:
            codes = df[c].to_numpy()[alive].astype(np.int64)
            lv, inv = np.unique(codes, return_inverse=True)
            out["levels"][c] = (lv, np.bincount(inv, weights=y[alive]), np.bincount(inv))
    else:
        eta = np.full(df.height, np.nan)
        eta[alive] = fam.start(y[alive], spec["ybar"])
        mu = fam.linkinv(eta[alive])
        z = np.full(df.height, np.nan)
        w = np.full(df.height, np.nan)
        z[alive], w[alive] = fam.working(y[alive], eta[alive], mu)
        from .worker import sidecar_path

        path = sidecar_path(spec["out"], sid)
        side = pl.DataFrame({Z: z, W: w}).with_columns(pl.all().fill_nan(None))
        side.write_parquet(path.with_suffix(".parquet.tmp"))
        path.with_suffix(".parquet.tmp").replace(path)
    out["timing"] = {"total": time.perf_counter() - t0}
    return out


def _run(executor, fn, ds, args_of, wd, label):
    from .progress import Progress

    prog = Progress(wd, label=label, every=60)
    rows = {s["id"]: max(s["rows"], 1) for s in ds.shards}
    prog.add_stage(label, rows)
    prog.start(label)
    res = {}
    tasks = [(s["id"], rows[s["id"]], (s["id"], ds.shard_paths(s), args_of(s["id"]))) for s in ds.shards]
    executor.run(fn, tasks, lambda tid, r: res.__setitem__(tid, r), prog, label)
    prog.finish(label, log=False)
    return res


# ---------------------------------------------------------------------------- driver
def fit_glm(
    ds,
    formula: str,
    family: str = "poisson",
    *,
    cluster=None,
    vcov=None,
    executor=None,
    workdir=None,
    tol: float = 1e-8,
    maxiter: int = 50,
    separation: bool = True,
    keep_sidecars: bool = False,
    **fit_kw,
):
    """Fit an HDFE GLM (`family` "poisson" or "cloglog") by IRLS over dfeols fits.

    `formula`, `cluster`, `vcov` and the remaining keywords (fe_tol, crossing_fe, ...) are
    `fit`'s. Returns the converged fit's Result, with `options["glm"]` holding the family,
    iterations, deviance, convergence and the rows dropped as separated."""
    from .estimate import fit
    from .executors import SerialExecutor
    from .prep import Dataset

    if family not in FAMILIES:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}, got {family!r}")
    fam = FAMILIES[family]
    if not isinstance(ds, Dataset):
        ds = Dataset.open(ds)
    executor = executor or SerialExecutor()
    for bad in ("weights", "predict", "predict_keep", "cells", "record"):
        if fit_kw.get(bad):
            raise NotImplementedError(f"fit_glm does not take {bad}= yet")
    fit_kw.pop("record", None)
    f = parse(formula)
    if f.iv or len(f.depvars) != 1:
        raise ValueError("fit_glm takes one outcome and no IV part")
    if any(t.slopes for t in f.fe_terms):
        raise NotImplementedError("fit_glm does not take varying slopes (fe[x]) yet")
    y = f.depvars[0]
    rhs, _, fe_part = formula.partition("~")[2].partition("|")
    fes = [ds.fes[t.base]["column"] for t in f.fe_terms]
    names = cluster.split("+") if isinstance(cluster, str) else list(cluster or [])
    clus_cols = [ds.clusters[c.strip()]["column"] for c in names if c.strip() in ds.clusters]
    xvars = sorted((set(f.variables) - {y} - {t.base for t in f.fe_terms}) | set(clus_cols))
    xvars = [v for v in xvars if v in ds.columns]

    tag = hashlib.sha1(f"{formula}|{family}|{cluster}".encode()).hexdigest()[:12]
    wd = Path(workdir) if workdir else ds.root / "runs" / f"glm-{tag}"
    sides = [wd / "side0", wd / "side1"]
    for d in sides:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    t0 = time.time()

    # separation: drop FE groups whose outcome never varies, until none changes
    dead: dict[str, np.ndarray] = {c: np.zeros(0, np.int64) for c in fes}
    passes, n0 = 0, None
    spec = {"family": family, "y": y, "vars": xvars, "fes": fes, "dead": dead, "mode": "scan"}
    while True:
        passes += 1
        res = _run(executor, shard_glm_init, ds, lambda sid: spec, wd, "glm_scan")
        n_alive = sum(r["n"] for r in res.values())
        if n0 is None:
            n0 = n_alive
        changed = False
        if not separation:
            break
        for c in fes:
            lv = np.concatenate([r["levels"][c][0] for r in res.values()])
            sy = np.concatenate([r["levels"][c][1] for r in res.values()])
            ct = np.concatenate([r["levels"][c][2] for r in res.values()])
            u, inv = np.unique(lv, return_inverse=True)
            S, C = np.bincount(inv, weights=sy), np.bincount(inv, weights=ct)
            bad = (S <= 0) | ((S >= C) if fam.binary else False)
            if bad.any():
                dead[c] = np.union1d(dead[c], u[bad])
                changed = True
        if not changed:
            break
    ybar = sum(r["sum_y"] for r in res.values()) / max(n_alive, 1)
    if n_alive == 0:
        raise ValueError(f"fit_glm: no rows left after the separation check ({passes} passes)")
    log.info("fit_glm %s: %s rows, %s dropped as separated (%d passes)", family, f"{n_alive:,}",
             f"{n0 - n_alive:,}", passes)  # fmt: skip
    wspec = dict(spec, mode="write", ybar=ybar, out=str(sides[0]))
    _run(executor, shard_glm_init, ds, lambda sid: wspec, wd, "glm_init")

    zf = f"{Z} ~{rhs}|{fe_part}" if fe_part else f"{Z} ~{rhs}"
    dev_prev, it, converged, r, history = None, 0, False, None, []
    for it in range(1, maxiter + 1):
        read, write = sides[(it - 1) % 2], sides[it % 2]
        for old in write.glob("glm-*.json"):
            old.unlink()
        r = fit(
            ds, zf, weights=W, cluster=cluster, vcov=vcov, executor=executor, workdir=wd / "fit",
            record=False, predict=wd / "pred", predict_keep=[y],
            _glm={"cols": (Z, W), "read": read, "write": write, "family": family, "y": y}, **fit_kw,
        )  # fmt: skip
        stats = [json.loads(p.read_text()) for p in write.glob("glm-*.json")]
        dev = sum(s["deviance"] for s in stats)
        history.append(dev)
        log.info("fit_glm %s: iteration %d, deviance %.10g", family, it, dev)
        if dev_prev is not None and abs(dev - dev_prev) / (0.1 + abs(dev)) < tol:
            converged = True
            break
        dev_prev = dev
    if not converged:
        log.warning("fit_glm %s: not converged after %d iterations (deviance %s)", family, it, history[-3:])
    if not keep_sidecars:
        for d in sides + [wd / "pred"]:
            shutil.rmtree(d, ignore_errors=True)

    # the converged weighted fit, relabelled as the GLM
    out = r.outcomes.pop(Z)
    _iid_to_glm(out, r)
    out["deviance"] = history[-1]
    for k in ("rss", "r2", "adj_r2", "r2_within", "adj_r2_within"):
        out.pop(k, None)
    r.outcomes = {y: out}
    r.formula = formula
    r.options = dict(r.options, weights=None)
    r.options["glm"] = {
        "family": family,
        "link": "log" if family == "poisson" else "cloglog",
        "iterations": it,
        "converged": converged,
        "deviance": history[-1],
        "deviance_history": history,
        "tol": tol,
        "separated_rows": int(n0 - n_alive),
        "separation_passes": passes,
        "seconds": round(time.time() - t0, 2),
    }
    return r


def _iid_to_glm(out: dict, r) -> None:
    """The weighted fit's iid VCE is bread * RSS/(N - K) (df_t = N - K); the GLM's has
    dispersion 1 and fixest's K adjustment: bread * (N - 1)/(N - K)."""
    v = out.get("vcov", {})
    if "iid" in v and out.get("rss"):
        V, df_t = v["iid"]
        v["iid"] = (V * (r.N - 1) / out["rss"], df_t)
