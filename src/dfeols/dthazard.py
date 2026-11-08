# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Discrete-time hazard linear probability models: the Farbmacher-Tauchmann IV estimator.

Farbmacher, H. and Tauchmann, H. (2021), "Linear fixed-effects estimation with non-repeated
outcomes", FAU Discussion Papers in Economics 03/2021. Design: docs/design/hazard.md.

The outcome is an absorbing state: each unit is observed over one spell at risk, y = 0 until
the period of the event (y = 1, the unit's last row) or censoring. Within and first-difference
estimators are biased here even without unobserved heterogeneity, because y can't be
differenced (y_{t-1} = 0 on every row). The estimator regresses y_it on the levels x_it by
just-identified IV with the j-th differences Δ^j x_it as instruments, a constant (or the
absorbed FEs) in both stages, on rows that have j predecessors. The unit FE is not absorbed:
Cov(a_i, Δ^j x_it) = 0 in the population is the identifying assumption; what remains is
survival bias.

`lpm_dthazard` builds that sample shard by shard (lags never cross shards: the data must be
partitioned by the unit), caches it as a derived dataset, and fits it with dfeols's IV path.
It also fits, on the same sample, the first-difference estimator (the IV's reduced form)
and pooled OLS for comparison, and reports the first-stage matrix G (b_FD ≈ G b_IV).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .executors import SerialExecutor
from .formula import Term, _term, fe_components, parse
from .lock import DirLock
from .prep import META, Dataset, remove_dir
from .progress import Progress

log = logging.getLogger("dfeols")

SAMPLE_VERSION = 1

# Why a row of the at-risk data is not in the estimation sample, in the order the rules
# are applied (a row is counted under the first that applies).
DROP_REASONS = {
    "after_event": "rows after the unit's first y = 1 (after_event='drop'; the default stops)",
    "missing": "y, a regressor or the weight missing or non-finite in the row itself",
    "first_periods": "the first j rows of each unit's observed spell: no j predecessors in the data",
    "gap": "a predecessor row t-1..t-j is not in the data (time is not consecutive)",
    "missing_lag": "a regressor missing or non-finite in a predecessor row t-1..t-j",
}


def _instrument_columns(terms: list[Term], instrumented: list[int], j: int, cat_levels: dict) -> list:
    """(term index, level or None, column name) for every instrument column the builder
    makes: one per continuous term, one per level of the categorical for an i() term (the
    levels of the prepared data; those absent from the sample are left out later)."""
    out, seen = [], set()
    for i in instrumented:
        t = terms[i]
        tail = "_x_".join(t.conts)
        for lv in cat_levels[t.cat] if t.cat else [None]:
            lvs = re.sub(r"\W", "_", str(lv))
            base = f"D{j}_" + (f"{t.cat}_{lvs}" if t.cat else "")
            name = base + ("_x_" if t.cat and tail else "") + tail
            k, cand = 1, name
            while cand in seen:
                k += 1
                cand = f"{name}_{k}"
            seen.add(cand)
            out.append((i, lv, cand))
    return out


def _term_text(t: Term) -> str:
    """The term as formula text, ref= included (Term.label leaves it out)."""
    parts = []
    if t.cat:
        parts.append(f"i({t.cat}, ref={t.ref!r})" if t.ref is not None else f"i({t.cat})")
    return ":".join([*parts, *t.conts])


def _binom(j: int) -> list[int]:
    return [(-1) ** k * math.comb(j, k) for k in range(j + 1)]


# ---------------------------------------------------------------- per-shard sample builder


def _build_task(paths: list[str], out_file: str, spec: dict) -> dict:
    """Build one shard's estimation sample: sort each spell, check the absorbing outcome,
    form Δ^j of every instrumented term from the unit's own predecessor rows, drop and count
    rows by reason, write the kept rows; return counts and per-FE predecessor summaries."""
    t0 = time.perf_counter()
    unit, tcol, y, j = spec["unit"], spec["time"], spec["y"], spec["diff"]
    df = pl.read_parquet(paths, columns=spec["read"])
    if df.schema[tcol] == pl.Date:
        df = df.with_columns(pl.col(tcol).cast(pl.Int32))
    df = df.sort([unit, tcol])
    n_in = df.height
    out: dict = {"rows_in": n_in, "dropped": dict.fromkeys(DROP_REASONS, 0), "errors": {}}

    same_unit = pl.col(unit) == pl.col(unit).shift(1)
    dup = df.filter(same_unit & (pl.col(tcol) == pl.col(tcol).shift(1)))
    if dup.height:
        out["errors"]["duplicate"] = {"rows": dup.height, "example": dup[unit][0]}
        return out
    yv = df[y]
    bad_y = df.filter(yv.is_not_null() & ~yv.is_in([0, 1]))
    if bad_y.height:
        out["errors"]["not_binary"] = {"rows": bad_y.height, "example": bad_y[y][0]}
        return out

    # rows after the first event
    ev = pl.when(pl.col(y) == 1).then(pl.col(tcol)).otherwise(None).min().over(unit)
    after = df.select((pl.col(tcol) > ev).fill_null(False)).to_series()
    n_after = int(after.sum())
    if n_after:
        if spec["after_event"] == "error":
            out["errors"]["after_event"] = {"rows": n_after, "example": df.filter(after)[unit][0]}
            return out
        df = df.filter(~after)
        out["dropped"]["after_event"] = n_after

    # term columns (the continuous part; products made here), their validity, and Δ^j
    tcols, valid = [], []
    for i, term in enumerate(spec["terms"]):
        e = pl.lit(1.0)
        for c in term["conts"]:
            e = e * pl.col(c).cast(pl.Float64)
        tcols.append(e.alias(f"__t{i}"))
        valid.append(pl.col(f"__t{i}").is_not_null() & pl.col(f"__t{i}").is_finite())
        if term["cat"]:
            valid.append(pl.col(term["cat"]).is_not_null())
    df = df.with_columns(tcols)
    ok_x = pl.all_horizontal(valid)
    df = df.with_columns(ok_x.alias("__okx"))
    own_ok = pl.col("__okx") & pl.col(y).is_not_null()
    if spec["weights"]:
        own_ok = own_ok & pl.col(spec["weights"]).is_not_null() & pl.col(spec["weights"]).is_finite()
    pos = pl.int_range(pl.len()).over(unit)
    window = (pl.col(unit).shift(j) == pl.col(unit)) & ((pl.col(tcol) - pl.col(tcol).shift(j)) == j)
    lag_ok = pl.all_horizontal([pl.col("__okx").shift(k) for k in range(1, j + 1)])
    reason = (
        pl.when(~own_ok)
        .then(pl.lit("missing"))
        .when(pos < j)
        .then(pl.lit("first_periods"))
        .when(~window.fill_null(False))
        .then(pl.lit("gap"))
        .when(~lag_ok.fill_null(False))
        .then(pl.lit("missing_lag"))
        .otherwise(None)
        .alias("__why")
    )
    w = _binom(j)
    diffs = []
    for i, lv, name in spec["inst_cols"]:
        cat = spec["terms"][i]["cat"]
        col = pl.col(f"__t{i}")
        if cat:
            col = col * (pl.col(cat) == lv).cast(pl.Float64)
        diffs.append(sum((col.shift(k) * w[k] for k in range(1, j + 1)), col * w[0]).alias(name))
    preds = [pl.col(c).shift(1).alias(f"__pred_{c}") for c in spec["fe_cols"]]
    df = df.with_columns([reason, *diffs, *preds])
    counts = df.group_by("__why").len()
    for r, n in zip(counts["__why"].to_list(), counts["len"].to_list()):
        if r is not None:
            out["dropped"][r] += int(n)
    kept = df.filter(pl.col("__why").is_null())

    # predecessor summaries per FE: is a cell's predecessor cell unique? is the FE within unit?
    out["fe_pred"] = {
        c: kept.group_by(c).agg(
            pl.col(f"__pred_{c}").min().alias("pmin"),
            pl.col(f"__pred_{c}").max().alias("pmax"),
            pl.len().alias("rows"),
            pl.col(unit).n_unique().alias("units"),
        )
        for c in spec["fe_cols"]
    }
    out["y_sum"] = float(kept[y].sum()) if kept.height else 0.0
    out["units"] = int(kept[unit].n_unique()) if kept.height else 0
    out["rows"] = kept.height
    out["cat_levels"] = {c: kept[c].drop_nulls().unique().to_list() for c in spec["cats"]}
    if kept.height:
        keep_cols = list(dict.fromkeys([*spec["write"], *spec["instruments_kept"]]))
        tmp = out_file + ".tmp"
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        kept.select(keep_cols).write_parquet(tmp)
        os.replace(tmp, out_file)
    out["timing"] = {"total": time.perf_counter() - t0}
    return out


# ---------------------------------------------------------------- result


@dataclass
class HazardResult:
    """lpm_dthazard's result: `iv` is the estimator (a dfeols Result, with its own record);
    `fd` (first differences, the reduced form) and `ols` (pooled OLS) are on the same sample;
    `G` the first-stage coefficients of each level on the instruments (b_FD ≈ G b_IV)."""

    iv: object
    fd: object | None
    ols: object | None
    G: dict
    sample: dict
    fe_checks: dict
    options: dict
    dataset: str
    warnings: list[str] = field(default_factory=list)

    def coef(self):
        return self.iv.coef()

    def tidy(self, **kw):
        return self.iv.tidy(**kw)

    def summary(self) -> str:
        s, o = self.sample, self.options
        lines = [
            "Discrete-time hazard LPM, Farbmacher-Tauchmann IV (lpm_dthazard)",
            f"  formula      {o['formula']}",
            f"  unit, time   {o['unit']}, {o['time']}    differences j = {o['diff']}",
            "  instruments  "
            + ", ".join(f"{e} <- {z}" for e, z in zip(self.iv.iv["endogenous"], self.iv.iv["instruments"])),
            "",
            "Sample (rows of the prepared data -> estimation sample)",
            f"  rows in the data                 {s['rows_input']:>14,}",
        ]
        for r, n in s["dropped"].items():
            lines.append(f"  - {r:<30} {n:>14,}   {DROP_REASONS.get(r, '')}")
        lines += [
            f"  = built sample                   {s['rows_built']:>14,}"
            f"  ({s['units']:,} units, {s['events']:,.0f} events)",
            f"  estimation sample (after fit's own drops: singletons etc.) {self.iv.N:,}",
            "",
            "IV (b_IV): y on levels, Δ^j as instruments",
            self.iv.summary(),
        ]
        if self.fd is not None:
            lines += ["", "First differences (b_FD, the reduced form; same sample)", self.fd.summary()]
        if self.ols is not None:
            lines += ["", "Pooled OLS (same sample)", self.ols.summary()]
        if self.G:
            lines += ["", "First stage G (rows: instruments, columns: levels; b_FD ≈ G b_IV)"]
            cols = list(self.G)
            rows = list(next(iter(self.G.values())))
            lines.append("  " + " " * 24 + "".join(f"{c:>14}" for c in cols))
            for r in rows:
                lines.append(f"  {r:<24}" + "".join(f"{self.G[c][r]:>14.6g}" for c in cols))
        for name, c in self.fe_checks.items():
            if c["ambiguous_rows"]:
                lines.append(
                    f"\nFE {name}: {c['ambiguous_share']:.2%} of rows are in cells whose predecessor cell"
                    " differs across units (movers); absorbed in levels"
                )
        for wmsg in self.warnings:
            lines.append(f"Warning: {wmsg}")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

    def to_dict(self) -> dict:
        return {
            "options": self.options,
            "sample": self.sample,
            "fe_checks": self.fe_checks,
            "G": self.G,
            "dataset": self.dataset,
            "warnings": self.warnings,
            "fits": {
                k: getattr(r, "workdir", None)
                for k, r in (("iv", self.iv), ("fd", self.fd), ("ols", self.ols))
            },
        }


# ---------------------------------------------------------------- the estimator


def lpm_dthazard(
    ds: Dataset | str,
    formula: str,
    *,
    unit: str,
    time: str,
    diff: int = 1,
    levels: list[str] | tuple[str, ...] = (),
    cluster: str | None = None,
    vcov=None,
    weights: str | None = None,
    after_event: str = "error",
    compare: bool = True,
    executor=None,
    workdir: str | os.PathLike | None = None,
    label: str | None = None,
    record: bool = True,
    **fit_options,
) -> HazardResult:
    """Discrete-time hazard linear probability model by the Farbmacher-Tauchmann IV
    (docs/design/hazard.md).

    formula   `y ~ x1 + x2 + x1:x2 + i(age, ref=55) + x1:i(g) | fe1 + fe2`: y is the
              absorbing-state indicator; every term is instrumented by its own j-th difference
              (unless named in `levels`), the FEs are absorbed as exogenous. An intercept is
              always included (the paper's constant) unless FEs absorb it. An i() term is
              estimated on the levels present in the sample (minus ref=, or the first level):
              each dummy is instrumented by Δ^j of itself. Levels that never occur in the
              sample (e.g. the youngest age, seen only in a unit's first period) get no
              coefficient and are reported (`excluded_levels`), as the paper drops age 53.
              Absorbing such a variable as a FE instead gives the same other coefficients
              (docs/design/hazard.md). A FE involving `unit` is refused (that is the within
              estimator, which is biased here).
    unit      the spell (unit) identifier; the dataset must be prepared with partition=unit
              (or a FE/cluster on unit nested in the partition), so a unit's rows are in one
              shard
    time      the period: integers, consecutive periods differ by 1 (a Date counts days)
    diff      j: instruments are Δ^j x (1 = first differences; 0 would be pooled OLS)
    levels    terms assumed uncorrelated with the unit effect in levels: they instrument
              themselves (default none)
    cluster, vcov, weights
              as in fit(); default vcov is "hetero" (the paper's), "CRV1" with a cluster
    after_event  rows after a unit's first y = 1: "error" (default) or "drop" (counted)
    compare   also fit first differences (the reduced form) and pooled OLS on the sample

    The sample. Starting from the prepared data, a row is dropped (and counted, under the
    first reason that applies; see DROP_REASONS) if it comes after the unit's event
    (after_event="drop"), if y, a regressor or the weight is missing or non-finite, if it is
    one of the first j rows of the unit's spell, if any of t-1..t-j is not in the data (a gap),
    or if a regressor is missing in one of those predecessor rows. fit() then drops FE
    singletons as usual. y must be 0/1, with at most one (unit, time) row each.

    Extra FEs: absorbing a FE as exogenous equals instrumenting its dummies by their
    differences exactly when a unit's cell at t-1 is determined by its cell at t (time, age,
    group x time with fixed groups). Where units move between groups that fails; the FE is
    still absorbed, with a warning giving the share of rows whose cell has more than one
    predecessor cell.
    """
    from .estimate import fit

    t_start = time_now()
    if isinstance(ds, (str, os.PathLike)):
        ds = Dataset.open(ds)
    executor = executor or SerialExecutor()
    f = parse(formula)
    if f.iv:
        raise ValueError("lpm_dthazard builds its own instruments: give y ~ x | fe without an IV part")
    if len(f.depvars) != 1:
        raise ValueError("lpm_dthazard takes one outcome, the absorbing-state indicator")
    if not f.intercept:
        raise ValueError("lpm_dthazard needs the constant (the paper's footnote 6); remove '0'")
    if not isinstance(diff, int) or diff < 1:
        raise ValueError("diff must be an integer >= 1 (diff=0 would be pooled OLS)")
    if after_event not in ("error", "drop"):
        raise ValueError("after_event must be 'error' or 'drop'")
    labels = [_term_text(t) for t in f.terms]
    lev_terms = []
    for lv in levels:
        lt = _term(lv)
        if lt not in f.terms:
            raise ValueError(f"levels names a term not in the formula: {lv!r}")
        lev_terms.append(lt)
    levels = [_term_text(t) for t in lev_terms]
    instrumented = [i for i, t in enumerate(f.terms) if t not in lev_terms]
    if not instrumented:
        raise ValueError("every term is in levels: nothing to instrument (that is pooled OLS)")
    y = f.depvars[0]

    # the unit's rows must be in one shard
    unit_ok = unit in ds.partition or any(
        info.get("components") == [unit] and info.get("nested_in_partition")
        for info in (*ds.fes.values(), *ds.clusters.values())
    )
    if not unit_ok:
        raise ValueError(
            f"lpm_dthazard needs each unit's rows in one shard: prep with partition={unit!r} "
            f"(this dataset is partitioned by {ds.partition})"
        )
    for fe in f.fes:
        if unit in fe_components(fe):
            raise ValueError(
                f"FE {fe!r} involves the unit {unit!r}: absorbing unit effects gives the within estimator, "
                "biased with an absorbing outcome (Farbmacher-Tauchmann 2021, appendix A.3)"
            )
        if fe not in ds.fes:
            raise KeyError(f"FE {fe!r} was not encoded by prep (have {list(ds.fes)})")
    cats = list(dict.fromkeys(t.cat for t in f.terms if t.cat))
    for c in (unit, time, y, *(c for t in f.terms for c in t.conts), *cats, *([weights] if weights else [])):
        if c not in ds.columns:
            raise KeyError(f"Column {c!r} is not in the dataset")
    if cluster is not None and cluster not in ds.clusters:
        raise KeyError(f"Cluster {cluster!r} was not encoded by prep (have {list(ds.clusters)})")
    tdtype = ds.scan([time]).collect_schema()[time]
    if not (tdtype.is_integer() or tdtype == pl.Date):
        raise TypeError(f"time column {time!r} is {tdtype}: give integer periods (or a Date, in days)")

    fe_cols = [ds.fes[fe]["column"] for fe in f.fes]
    cl_cols = [ds.clusters[cluster]["column"]] if cluster else []
    conts = list(dict.fromkeys(c for t in f.terms for c in t.conts))
    from .estimate import category_levels

    parent_levels = category_levels(ds, cats, executor) if cats else {}
    inst_cols = _instrument_columns(list(f.terms), instrumented, diff, parent_levels)
    write = list(
        dict.fromkeys([unit, time, y, *conts, *cats, *fe_cols, *cl_cols, *([weights] if weights else [])])
    )
    spec = {
        "unit": unit,
        "time": time,
        "y": y,
        "diff": diff,
        "after_event": after_event,
        "weights": weights,
        "conts": conts,
        "terms": [{"conts": list(t.conts), "cat": t.cat} for t in f.terms],
        "cats": cats,
        "instrumented": instrumented,
        "inst_cols": inst_cols,
        "instruments_kept": [name for *_, name in inst_cols],
        "fe_cols": fe_cols,
        "read": write,
        "write": write,
    }

    key = hashlib.sha256(
        json.dumps(
            {
                "v": SAMPLE_VERSION,
                "root": str(Path(ds.root).resolve()),
                "created": ds.meta.get("created"),
                "n_rows": ds.n_rows,
                **{k: v for k, v in spec.items() if k != "read"},
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:16]
    sample_dir = Path(ds.root) / "_dthazard" / key
    warnings: list[str] = []
    built = _build_sample(ds, spec, sample_dir, executor, label or "dthazard")
    sds = Dataset.open(sample_dir)
    info = sds.meta["dthazard"]

    # FE checks: nested in the unit (refused), movers (warned)
    fe_checks = {}
    for fe in f.fes:
        c = info["fe_checks"][ds.fes[fe]["column"]]
        fe_checks[fe] = c
        if c["nested_in_unit"]:
            raise ValueError(
                f"FE {fe!r} has one unit per level in the sample: it absorbs unit effects (the within "
                "estimator, biased with an absorbing outcome)"
            )
        if c["ambiguous_rows"]:
            msg = (
                f"FE {fe}: {c['ambiguous_share']:.2%} of sample rows are in cells whose predecessor cell "
                "differs across units (units move between groups). Absorbed in levels, which is not "
                "the same as "
                "instrumenting its dummies by their differences (docs/design/hazard.md)"
            )
            warnings.append(msg)
            log.warning("lpm_dthazard: %s", msg)

    # i() terms: fit expands them from the levels present in the sample (sorted, as
    # category_levels does), dropping ref= or the first; instrument exactly those levels
    sample_levels = {c: info["cat_levels"][c] for c in cats}
    instruments, excluded = [], {}
    for i in instrumented:
        t = f.terms[i]
        cols = [(lv, name) for ti, lv, name in inst_cols if ti == i]
        if not t.cat:
            instruments.append(cols[0][1])
            continue
        present = sample_levels[t.cat]
        if t.ref is not None and t.ref not in present:
            raise ValueError(f"{labels[i]}: the reference level {t.ref!r} is not in the estimation sample")
        if t.ref is not None:
            est = [lv for lv in present if lv != t.ref]
        else:
            est = present[1:] if t.drop_first else present
        by_level = dict(cols)
        instruments += [by_level[lv] for lv in est]
        gone = [lv for lv in parent_levels[t.cat] if lv not in present]
        if gone:
            excluded[labels[i]] = gone
            msg = (
                f"{labels[i]}: level(s) {gone} never occur in the estimation sample (e.g. the youngest age, "
                "seen only in a unit's first period), so they get no coefficient (the paper drops them too)"
            )
            warnings.append(msg)
            log.warning("lpm_dthazard: %s", msg)
        if t.ref is None and t.drop_first and present and parent_levels[t.cat][0] != present[0]:
            msg = f"{labels[i]}: reference level is {present[0]!r}, the first level in the sample"
            warnings.append(msg)
            log.warning("lpm_dthazard: %s", msg)
    endog = [labels[i] for i in instrumented]
    exog = list(levels) or ["1"]
    fe_part = f" | {' + '.join(f.fes)}" if f.fes else ""
    main_f = f"{y} ~ {' + '.join(exog)}{fe_part} | {' + '.join(endog)} ~ {' + '.join(instruments)}"
    fd_f = f"{y} ~ {' + '.join([*instruments, *levels])}{fe_part}"
    ols_f = f"{y} ~ {' + '.join(labels)}{fe_part}"
    vcov = vcov or ("CRV1" if cluster else "hetero")
    base = Path(workdir) if workdir else Path(ds.root) / "runs" / f"dthazard-{key}"
    label = label or "dthazard"
    common = dict(cluster=cluster, vcov=vcov, weights=weights, executor=executor, **fit_options)

    sample = {
        "rows_input": int(ds.n_rows),
        "dropped": {k: int(v) for k, v in info["dropped"].items() if v or k != "after_event"},
        "rows_built": int(info["rows"]),
        "units": int(info["units"]),
        "events": float(info["events"]),
    }
    options = {
        "formula": formula,
        "unit": unit,
        "time": time,
        "diff": diff,
        "levels": list(levels),
        "after_event": after_event,
        "cluster": cluster,
        "vcov": vcov,
        "weights": weights,
        "compare": compare,
        "fit_formula": main_f,
        "sample_key": key,
        **fit_options,
    }

    res = fit(sds, main_f, workdir=str(base / "iv"), label=f"{label}:iv", record=False, **common)
    fd = ols = None
    if compare:
        fd = fit(sds, fd_f, workdir=str(base / "fd"), label=f"{label}:fd", record=False, **common)
        ols = fit(sds, ols_f, workdir=str(base / "ols"), label=f"{label}:ols", record=False, **common)

    G = {}
    for e in res.iv["first_stage"]["per_endogenous"]:
        G[e["endogenous"]] = dict(zip(e["instruments"], e["coef"]))

    call_args = [repr(formula), f"unit={unit!r}", f"time={time!r}", f"diff={diff!r}"]
    for k, v in (("levels", list(levels)), ("cluster", cluster), ("weights", weights)):
        if v:
            call_args.append(f"{k}={v!r}")
    call_args.append(f"vcov={vcov!r}")
    if after_event != "error":
        call_args.append(f"after_event={after_event!r}")
    for k, v in fit_options.items():
        call_args.append(f"{k}={v!r}")
    restriction = (
        f"discrete-time hazard sample (lpm_dthazard): rows of each {unit} spell with {diff} consecutive "
        f"predecessor period(s) in {time} and complete regressors in them; see dropped"
    )
    hz = {
        "method": "Farbmacher-Tauchmann (2021) IV: y on levels, Δ^j levels as instruments, "
        "unit FE not absorbed",
        "reference": "Farbmacher, H. and Tauchmann, H. (2021). Linear fixed-effects estimation with "
        "non-repeated outcomes. FAU Discussion Papers in Economics 03/2021",
        "unit": unit,
        "time": time,
        "diff": diff,
        "levels": list(levels),
        "after_event": after_event,
        "sample": sample,
        "drop_reasons": {k: DROP_REASONS[k] for k in sample["dropped"]},
        "fe_checks": fe_checks,
        "excluded_levels": {k: [str(v) for v in vs] for k, vs in excluded.items()},
        "G": G,
        "fit_formula": main_f,
        "sample_key": key,
        "comparison": {
            name: {
                "formula": r.formula,
                "coefnames": list(r.coefnames),
                "coef": [float(v) for v in r.coef()],
                "std_error": [float(v) for v in np.sqrt(np.diag(r.vcov()))],
                "vcov_type": r.vcov_default,
                "nobs": int(r.N),
            }
            for name, r in (("first_differences", fd), ("pooled_ols", ols))
            if r is not None
        },
    }
    if warnings:
        hz["warnings"] = warnings
    # Record facts: the hazard rules are part of the sample, so they go in the record's
    # sample section (rows input = the prepared data, drops by reason) and the extension.
    run = res.run
    run["call"] = f"dfeols.lpm_dthazard(ds, {', '.join(call_args)})"
    run["restriction"] = restriction
    fit_dropped = dict(run.get("sample", {}).get("dropped", {}))
    merged = dict(sample["dropped"])
    for k, v in fit_dropped.items():
        merged[k] = merged.get(k, 0) + int(v)
    src_rows = (ds.meta.get("source") or {}).get("rows")
    run["sample"] = {"rows_input": int(ds.n_rows), "dropped": merged}
    if src_rows is not None and src_rows >= ds.n_rows:
        run["sample"] = {
            "rows_input": int(src_rows),
            "dropped": {"missing_keys": int(src_rows - ds.n_rows), **merged},
        }
    run["dthazard"] = hz
    res.save(Path(res.workdir) / "result.json")
    if record:
        rpath = base / "result.estrec.json" if record is True else Path(record)
        try:
            res.write_estimation(rpath)
            log.info("lpm_dthazard %s: record %s", label, rpath)
        except Exception:
            log.exception("lpm_dthazard %s: writing the estimation record failed", label)

    out = HazardResult(
        iv=res,
        fd=fd,
        ols=ols,
        G=G,
        sample=sample,
        fe_checks=fe_checks,
        options=options,
        dataset=str(sample_dir),
        warnings=warnings,
    )
    (base / "dthazard.json").write_text(json.dumps(out.to_dict(), indent=1, default=str))
    log.info(
        "lpm_dthazard %s: done in %.0fs (sample %s; %s)",
        label,
        time_now() - t_start,
        "reused" if not built else "built",
        sample_dir,
    )
    return out


def time_now() -> float:
    return time.time()


def _build_sample(ds: Dataset, spec: dict, out: Path, executor, label: str) -> bool:
    """Build (or reuse) the derived dataset at `out`; True if built now."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with DirLock(out, what="hazard sample"):
        if (out / META).exists():
            return False
        if out.exists():
            remove_dir(out)
        shards_dir = out / "shards"
        tasks = [
            (
                s["id"],
                max(s["rows"], 1),
                (ds.shard_paths(s), str(shards_dir / f"__shard={s['id']}" / "part-0.parquet"), spec),
            )
            for s in ds.shards
        ]
        results: dict = {}

        def on_result(tid, r):
            results[tid] = r

        prog = Progress(None, label=f"{label}:sample", every=60)
        prog.add_stage("sample", {tid: rows for tid, rows, _ in tasks})
        prog.start("sample")
        executor.run(_build_task, tasks, on_result, prog, "sample")
        prog.finish("sample")

        errors: dict = {}
        for r in results.values():
            for k, v in r["errors"].items():
                e = errors.setdefault(k, {"rows": 0, "example": v["example"]})
                e["rows"] += v["rows"]
        if errors:
            msgs = {
                "duplicate": "rows repeat a (unit, time) pair",
                "not_binary": "y takes values other than 0 and 1",
                "after_event": "rows come after the unit's first y = 1 (after_event='drop' drops them)",
            }
            raise ValueError(
                "lpm_dthazard: "
                + "; ".join(f"{v['rows']:,} {msgs[k]} (e.g. {v['example']!r})" for k, v in errors.items())
            )
        dropped = dict.fromkeys(DROP_REASONS, 0)
        for r in results.values():
            for k, v in r["dropped"].items():
                dropped[k] += v
        fe_checks = {}
        for c in spec["fe_cols"]:
            parts = [r["fe_pred"][c] for r in results.values() if r["fe_pred"][c].height]
            if not parts:
                fe_checks[c] = {
                    "cells": 0,
                    "ambiguous_cells": 0,
                    "ambiguous_rows": 0,
                    "ambiguous_share": 0.0,
                    "nested_in_unit": False,
                }
                continue
            g = (
                pl.concat(parts)
                .group_by(c)
                .agg(pl.col("pmin").min(), pl.col("pmax").max(), pl.col("rows").sum(), pl.col("units").sum())
            )
            amb = g.filter(pl.col("pmin") != pl.col("pmax"))
            n = int(g["rows"].sum())
            fe_checks[c] = {
                "cells": g.height,
                "ambiguous_cells": amb.height,
                "ambiguous_rows": int(amb["rows"].sum()),
                "ambiguous_share": float(amb["rows"].sum() / n) if n else 0.0,
                "nested_in_unit": bool((g["units"] == 1).all()),
            }
        rows = sum(r["rows"] for r in results.values())
        if not rows:
            raise ValueError(f"lpm_dthazard: no rows left in the sample (dropped: {dropped})")
        shards = []
        for tid, r in sorted(results.items()):
            if r["rows"]:
                d = shards_dir / f"__shard={tid}"
                shards.append(
                    {
                        "id": tid,
                        "files": sorted(str(p.relative_to(out)) for p in d.glob("*.parquet")),
                        "rows": r["rows"],
                    }
                )
        meta = dict(ds.meta)
        src = dict(meta.get("source") or {})
        src.pop("rows", None)  # lpm_dthazard writes the record's row accounting itself
        meta.update(
            shards=shards,
            n_rows=rows,
            columns=list(dict.fromkeys([*spec["write"], *spec["instruments_kept"]])),
            source=src,
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
            dthazard={
                "parent": str(Path(ds.root).resolve()),
                "parent_rows": ds.n_rows,
                "spec": {k: v for k, v in spec.items() if k != "read"},
                "dropped": dropped,
                "rows": rows,
                "units": sum(r["units"] for r in results.values()),
                "events": sum(r["y_sum"] for r in results.values()),
                "fe_checks": fe_checks,
                "cat_levels": {
                    c: sorted({lv for r in results.values() for lv in r["cat_levels"].get(c, [])})
                    for c in spec["cats"]
                },
            },
        )
        meta.pop("partition_plan", None)
        tmp = out / (META + ".tmp")
        tmp.write_text(json.dumps(meta, indent=1, default=str))
        os.replace(tmp, out / META)
        log.info("lpm_dthazard: sample built, %s of %s rows (%s)", f"{rows:,}", f"{ds.n_rows:,}", out)
        return True


__all__ = ["lpm_dthazard", "HazardResult", "DROP_REASONS"]
