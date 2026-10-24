# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared machinery for replicating published HDFE results with dfeols.

A replication is a set of `DatasetSpec`s (how to shard the data) and `Spec`s
(models, each fit on one or more of those datasets). For every spec this module

  1. fits it with dfeols on each listed dataset,
  2. fits the same model in memory with pyfixest (the exact reference),
  3. optionally reads reghdfe results written by a generated Stata do-file,
  4. compares all of them, plus the published numbers, and writes a report.

Checks and what counts as a failure:

  dfeols vs pyfixest         coefficients and SEs must agree (TOL_PYFIXEST)   -> FAIL
  dfeols across partitions   the same spec on differently sharded data must
                             give the same answer (TOL_INVARIANCE)            -> FAIL
  dfeols vs reghdfe          coefficients tight; SEs looser, because reghdfe
                             and fixest count absorbed degrees of freedom
                             differently in some designs (TOL_REGHDFE)        -> WARN
  dfeols vs published        agreement to the published rounding           -> INFO
  N                          must equal pyfixest's N                           -> FAIL

Nothing here is specific to one paper; see parsons2011/ for a worked example.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger("dfeols.replication")

TOL_PYFIXEST = dict(b_rtol=1e-7, b_atol=1e-10, se_rtol=1e-6)
TOL_INVARIANCE = dict(b_rtol=1e-9, b_atol=1e-12, se_rtol=1e-8)
TOL_REGHDFE = dict(b_rtol=1e-6, b_atol=1e-9, se_rtol=1e-3)


# --------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class DatasetSpec:
    """How to shard the analysis file for dfeols.

    partition      key whose groups never split across shards (str or list)
    fixed_effects  every FE any spec on this dataset uses
    clusters       every cluster variable any spec on this dataset uses
    """

    name: str
    partition: str | tuple[str, ...]
    fixed_effects: tuple[str, ...]
    clusters: tuple[str, ...] = ()

    def fingerprint(self, target_rows: int) -> str:
        s = json.dumps([asdict(self), target_rows], sort_keys=True, default=list)
        return hashlib.sha1(s.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Spec:
    """One model. Listing several datasets runs it on each and checks they agree.

    formula   dfeols syntax, e.g. "y ~ x + a:b + i(c) | fe1 + fe2^fe3"
    vcov      vcov types to compute and compare ("iid", "hetero", "CRV1")
    cluster   cluster variable for CRV1
    paper     where the published numbers live, e.g. "Table 3, col. 9"
    """

    label: str
    formula: str
    datasets: tuple[str, ...]
    vcov: tuple[str, ...] = ("hetero",)
    cluster: str | None = None
    paper: str = ""
    note: str = ""

    def key_terms(self) -> list[str]:
        """Terms worth comparing everywhere: continuous variables and products
        (not the i() dummies, whose names differ across packages)."""
        from dfeols.formula import parse

        return [":".join(t.conts) for t in parse(self.formula).terms if t.cat is None]

    def run_label(self, dataset: str) -> str:
        return self.label if len(self.datasets) == 1 else f"{self.label}@{dataset}"


# --------------------------------------------------------------------------- executors
def make_executor(spec: str | None):
    """serial | dask:N | <dfeols site profile name>. Returns (executor, close())."""
    import dfeols

    if spec in (None, "", "serial"):
        return None, lambda: None
    if spec.startswith("dask"):
        import distributed

        n = int(spec.split(":")[1]) if ":" in spec else 2
        lc = distributed.LocalCluster(n_workers=n, threads_per_worker=1, dashboard_address=None)
        client = distributed.Client(lc)
        ex = dfeols.DaskExecutor(client, max_workers=n, tick=1)

        def close():
            client.close()
            lc.close()

        return ex, close
    ex = dfeols.executor(spec)

    def close():
        ex.client.close()
        ex.cluster.close()

    return ex, close


# --------------------------------------------------------------------------- dfeols
def prep_dataset(source: Path, root: Path, dspec: DatasetSpec, target_rows: int):
    """Shard `source` (Parquet) for `dspec`, reusing shards from an identical earlier prep."""
    import dfeols

    root = Path(root)
    fp = dspec.fingerprint(target_rows)
    stamp = root / "replication_prep.json"
    if (root / "dfeols_meta.json").exists() and stamp.exists():
        if json.loads(stamp.read_text()).get("fingerprint") == fp:
            return dfeols.Dataset.open(root)
    part = list(dspec.partition) if isinstance(dspec.partition, tuple) else dspec.partition
    ds = dfeols.prep(
        str(source),
        shard_dir=root,
        partition=part,
        fixed_effects=list(dspec.fixed_effects),
        clusters=list(dspec.clusters),
        target_rows=target_rows,
        overwrite=True,
    )
    stamp.write_text(
        json.dumps(
            {"fingerprint": fp, "dataset": asdict(dspec), "target_rows": target_rows}, default=list, indent=1
        )
    )
    return ds


def crossing_levels(ds, formula: str) -> int:
    """Largest level count among this formula's FEs that cross the partition."""
    from dfeols.formula import fe_components, parse

    worst = 0
    for name in parse(formula).fes:
        m = ds.fes["^".join(fe_components(name))]
        if not m["nested_in_partition"]:
            worst = max(worst, int(m["n_levels"]))
    return worst


def fit_dfeols(
    ds, spec: Spec, dataset: str, executor=None, workdir: Path | None = None, max_dummy_levels: int = 2000
) -> dict:
    import dfeols

    need = crossing_levels(ds, spec.formula)
    if need > max_dummy_levels:
        log.warning(
            "%s: a crossing FE has %d levels; raising max_dummy_levels from %d. The sparse "
            "path handles this when exactly one FE is local; otherwise dummies are dense.",
            spec.run_label(dataset),
            need,
            max_dummy_levels,
        )
    r = dfeols.fit(
        ds,
        spec.formula,
        cluster=spec.cluster,
        vcov=list(spec.vcov),
        executor=executor,
        workdir=str(workdir) if workdir else None,
        label=spec.run_label(dataset),
        max_dummy_levels=max(max_dummy_levels, need),
        log_every=1e9,
    )
    coefs = {}
    for v in spec.vcov:
        t = r.tidy(None, v)
        coefs[v] = {name: (float(row["Estimate"]), float(row["Std. Error"])) for name, row in t.iterrows()}
    return {
        "engine": "dfeols",
        "label": spec.run_label(dataset),
        "spec": spec.label,
        "dataset": dataset,
        "N": int(r.N),
        "seconds": float(r.seconds),
        "fe": list(r.fe),
        "fe_levels": list(map(int, r.fe_levels)),
        "collinear": list(r.collinear),
        "timing": dict(r.timing),
        "coefs": coefs,
    }


# --------------------------------------------------------------------------- pyfixest
def _pyfixest_tol_kwargs(tol: float = 1e-13, maxiter: int = 1_000_000) -> dict:
    try:  # dfeols >= 0.7 ships the version-aware helper
        from dfeols.formula import pyfixest_tol_kwargs

        return pyfixest_tol_kwargs(tol, maxiter)
    except ImportError:
        pass
    try:
        from pyfixest.demeaners import MapDemeaner  # pyfixest >= 0.60

        return {"demeaner": MapDemeaner(fixef_tol=tol, fixef_maxiter=maxiter)}
    except ImportError:
        return {"fixef_tol": tol, "fixef_maxiter": maxiter}


def fit_pyfixest(data: pd.DataFrame, spec: Spec) -> dict:
    """The same model in memory with pyfixest, at a tight demeaning tolerance."""
    import warnings

    import pyfixest as pf

    from dfeols.formula import to_pyfixest

    fml = to_pyfixest(spec.formula)
    coefs, N, t0 = {}, None, time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for v in spec.vcov:
            vc = {"CRV1": spec.cluster} if v == "CRV1" else v
            m = pf.feols(fml, data=data, vcov=vc, fixef_rm="singleton", **_pyfixest_tol_kwargs())
            t = m.tidy()
            coefs[v] = {
                str(name): (float(row["Estimate"]), float(row["Std. Error"])) for name, row in t.iterrows()
            }
            N = int(m._N)
    return {
        "engine": "pyfixest",
        "label": spec.label,
        "spec": spec.label,
        "dataset": "memory",
        "N": N,
        "seconds": (time.time() - t0) / max(len(spec.vcov), 1),
        "formula": fml,
        "coefs": coefs,
    }


# --------------------------------------------------------------------------- Stata
def _stata_term(term: str) -> str:
    parts = term.split(":")
    return parts[0] if len(parts) == 1 else "#".join(f"c.{p}" for p in parts)


def formula_to_reghdfe(spec: Spec) -> tuple[str, list[tuple[str, str]]]:
    """reghdfe command for a spec, plus (dfeols name, Stata _b[] name) for key terms."""
    from dfeols.formula import parse

    f = parse(spec.formula)
    if len(f.depvars) != 1:
        raise ValueError("one outcome per reghdfe call")
    rhs = []
    for t in f.terms:
        if t.cat is None:
            rhs.append(_stata_term(":".join(t.conts)))
        elif not t.conts:
            rhs.append(f"ib{t.ref}.{t.cat}" if t.ref is not None else f"i.{t.cat}")
        else:
            base = "#".join(f"c.{c}" for c in t.conts)
            rhs.append(f"{base}#ib{t.ref}.{t.cat}" if t.ref is not None else f"{base}#ibn.{t.cat}")
    absorb = " ".join(fe.replace("^", "#") for fe in f.fes) or "_cons"
    v = spec.vcov[0]
    vce = {"hetero": "vce(robust)", "iid": "", "CRV1": f"vce(cluster {spec.cluster})"}[v]
    cmd = (
        f"reghdfe {f.depvars[0]} {' '.join(rhs)}, absorb({absorb}) {vce} tolerance(1e-12) iterate(100000)"
    ).replace("  ", " ")
    return cmd, [(k, _stata_term(k)) for k in spec.key_terms()]


def write_reghdfe_do(specs: list[Spec], data_dta: Path, out_csv: Path, path: Path, title: str) -> Path:
    """A do-file that fits every spec with reghdfe and writes key terms to a CSV."""
    lines = [
        f"* {title}: reghdfe reference results for the dfeols replication harness.",
        "* Generated by the harness; regenerate rather than edit (`harness.py stata-do`).",
        "* Requires: ssc install ftools, replace ; ssc install reghdfe, replace",
        "version 16",
        "set more off",
        f'use "{data_dta}", clear',
        "tempfile results",
        "capture postclose H",
        "postfile H str40 spec str60 term str8 vcov double b se N using `results', replace",
    ]
    for s in specs:
        cmd, terms = formula_to_reghdfe(s)
        lines += ["", f"* {s.label}  {s.paper}", cmd]
        for ours, stata in terms:
            lines.append(
                f'post H ("{s.label}") ("{ours}") ("{s.vcov[0]}") (_b[{stata}]) (_se[{stata}]) (e(N))'
            )
    lines += ["", "postclose H", "use `results', clear", f'export delimited using "{out_csv}", replace', ""]
    path.write_text("\n".join(lines))
    return path


def read_reghdfe_csv(path: Path) -> dict:
    """{spec label: record} in the same shape as fit_* records."""
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for label, g in df.groupby("spec"):
        coefs: dict = {}
        for _, row in g.iterrows():
            coefs.setdefault(row["vcov"], {})[row["term"]] = (float(row["b"]), float(row["se"]))
        out[label] = {
            "engine": "reghdfe",
            "label": label,
            "spec": label,
            "N": int(g["N"].iloc[0]),
            "coefs": coefs,
        }
    return out


# --------------------------------------------------------------------------- compare
def _close(a: float, b: float, rtol: float, atol: float = 0.0) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def compare(
    specs: list[Spec], ours: dict, pyf: dict, stata: dict, published: dict, published_n: dict | None = None
) -> pd.DataFrame:
    """One row per (run, vcov, term) with the reference values and a status per check.

    ours       {run label: dfeols record}
    pyf        {spec label: pyfixest record}
    stata      {spec label: reghdfe record}
    published  {spec label: {term: (b, se, decimals)}}
    """
    rows = []
    for s in specs:
        runs = [ours[s.run_label(d)] for d in s.datasets if s.run_label(d) in ours]
        if not runs:
            continue
        first = runs[0]
        ref = pyf.get(s.label)
        st = stata.get(s.label)
        pub = published.get(s.label, {})
        key = set(s.key_terms())
        for r in runs:
            for v in s.vcov:
                mine = r["coefs"].get(v, {})
                terms = list(mine) if ref is None else [t for t in mine if t in ref["coefs"].get(v, {})]
                missing = [t for t in key if t not in terms]
                for t in missing:
                    rows.append(
                        dict(
                            spec=s.label,
                            run=r["label"],
                            vcov=v,
                            term=t,
                            key=True,
                            status="FAIL",
                            detail="key term missing from dfeols or pyfixest",
                        )
                    )
                for t in terms:
                    b, se = mine[t]
                    row = dict(
                        spec=s.label,
                        run=r["label"],
                        dataset=r["dataset"],
                        vcov=v,
                        term=t,
                        key=t in key,
                        b=b,
                        se=se,
                        N=r["N"],
                        status="ok",
                        detail="",
                    )
                    flags = []
                    if ref is not None:
                        rb, rse = ref["coefs"][v][t]
                        row.update(
                            pyf_b=rb,
                            pyf_se=rse,
                            pyf_b_rel=_rel(b, rb),
                            pyf_se_rel=_rel(se, rse),
                            pyf_N=ref["N"],
                        )
                        if not (
                            _close(b, rb, TOL_PYFIXEST["b_rtol"], TOL_PYFIXEST["b_atol"])
                            and _close(se, rse, TOL_PYFIXEST["se_rtol"])
                        ):
                            flags.append(("FAIL", "differs from pyfixest"))
                        if r["N"] != ref["N"]:
                            flags.append(("FAIL", f"N {r['N']:,} vs pyfixest {ref['N']:,}"))
                    if r is not first and t in first["coefs"].get(v, {}):
                        fb, fse = first["coefs"][v][t]
                        row.update(inv_b_rel=_rel(b, fb), inv_se_rel=_rel(se, fse))
                        if not (
                            _close(b, fb, TOL_INVARIANCE["b_rtol"], TOL_INVARIANCE["b_atol"])
                            and _close(se, fse, TOL_INVARIANCE["se_rtol"])
                        ):
                            flags.append(("FAIL", f"differs across partitions from {first['label']}"))
                    if st is not None and t in st["coefs"].get(v, {}):
                        sb, sse = st["coefs"][v][t]
                        row.update(
                            reghdfe_b=sb,
                            reghdfe_se=sse,
                            reghdfe_b_rel=_rel(b, sb),
                            reghdfe_se_rel=_rel(se, sse),
                            reghdfe_N=st["N"],
                        )
                        if not (
                            _close(b, sb, TOL_REGHDFE["b_rtol"], TOL_REGHDFE["b_atol"])
                            and _close(se, sse, TOL_REGHDFE["se_rtol"])
                        ):
                            flags.append(("WARN", "differs from reghdfe"))
                        if r["N"] != st["N"]:
                            flags.append(("WARN", f"N {r['N']:,} vs reghdfe {st['N']:,}"))
                    if t in pub and v == s.vcov[0]:
                        pb, pse, dec = pub[t]
                        half = 0.5 * 10 ** (-dec) + 1e-12
                        row.update(pub_b=pb, pub_se=pse)
                        ok = abs(round(b, dec) - pb) <= half and (
                            pse is None or abs(round(se, dec) - pse) <= half
                        )
                        row["pub_match"] = ok
                        if not ok:
                            pn = (published_n or {}).get(s.label)
                            why = f" (published N {pn:,}, ours {r['N']:,})" if pn and pn != r["N"] else ""
                            flags.append(("INFO", "differs from published" + why))
                    for level in ("FAIL", "WARN", "INFO"):
                        hit = [d for lv, d in flags if lv == level]
                        if hit:
                            row["status"] = level
                            break
                    row["detail"] = "; ".join(d for _, d in flags)
                    rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report
def _fmt(x, nd=5):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{nd}f}"


def _fmt_rel(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return "0" if x == 0 else f"{x:.0e}"


def write_report(
    path: Path,
    title: str,
    specs: list[Spec],
    cmp: pd.DataFrame,
    ours: dict,
    pyf: dict,
    stata: dict,
    notes: list[str] = (),
) -> dict:
    counts = cmp["status"].value_counts().to_dict() if len(cmp) else {}
    lines = [f"# {title}", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    verdict = "FAIL" if counts.get("FAIL") else ("WARN" if counts.get("WARN") else "PASS")
    lines += [f"**Verdict: {verdict}**  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())), ""]
    lines += [
        "Checks: dfeols vs pyfixest (exact reference, FAIL), the same spec on different "
        "partitions (FAIL), dfeols vs reghdfe (WARN; SE degrees-of-freedom conventions can "
        "differ), dfeols vs published (INFO).",
        "",
    ]
    for n in notes:
        lines += [f"> {n}", ""]

    lines += [
        "## Runs",
        "",
        "| run | N | FEs (levels) | dfeols s | pyfixest s | reghdfe |",
        "|---|---:|---|---:|---:|---|",
    ]
    for s in specs:
        for d in s.datasets:
            r = ours.get(s.run_label(d))
            if not r:
                continue
            fes = ", ".join(f"{f} ({n:,})" for f, n in zip(r["fe"], r["fe_levels"]))
            p = pyf.get(s.label)
            lines.append(
                f"| {r['label']} | {r['N']:,} | {fes} | {r['seconds']:.1f} | "
                f"{p['seconds']:.1f} | {'yes' if s.label in stata else '—'} |"
                if p
                else f"| {r['label']} | {r['N']:,} | {fes} | {r['seconds']:.1f} | — | "
                f"{'yes' if s.label in stata else '—'} |"
            )
    lines.append("")

    if len(cmp):
        key = cmp[cmp["key"]]
        lines += [
            "## Key coefficients",
            "",
            "| run | vcov | term | b | se | published | Δb rel pyfixest | Δse rel pyfixest | "
            "Δb rel reghdfe | Δse rel reghdfe | status |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---:|---|",
        ]
        for _, r in key.iterrows():
            pub = f"{r['pub_b']} ({r['pub_se']})" if "pub_b" in r and not pd.isna(r.get("pub_b")) else ""
            lines.append(
                f"| {r['run']} | {r['vcov']} | `{r['term']}` | {_fmt(r.get('b'))} | {_fmt(r.get('se'))} | "
                f"{pub} | {_fmt_rel(r.get('pyf_b_rel'))} | {_fmt_rel(r.get('pyf_se_rel'))} | "
                f"{_fmt_rel(r.get('reghdfe_b_rel'))} | {_fmt_rel(r.get('reghdfe_se_rel'))} | "
                f"{r['status']}{': ' + r['detail'] if r['detail'] else ''} |"
            )
        lines.append("")
        allc = cmp.dropna(subset=["pyf_b_rel"]) if "pyf_b_rel" in cmp else cmp.iloc[0:0]
        if len(allc):
            lines += [
                "## All coefficients shared with pyfixest",
                "",
                f"{len(allc):,} coefficient-vcov pairs compared; largest relative differences: "
                f"b {allc['pyf_b_rel'].max():.1e}, se {allc['pyf_se_rel'].max():.1e}.",
                "",
            ]
        bad = cmp[cmp["status"].isin(["FAIL", "WARN"]) & ~cmp["key"]]
        if len(bad):
            lines += ["Non-key terms with problems:", ""]
            lines += [
                f"- {r['run']} `{r['term']}` ({r['vcov']}): {r['detail']}" for _, r in bad.head(30).iterrows()
            ]
            lines.append("")
    Path(path).write_text("\n".join(lines))
    return {"verdict": verdict, "counts": counts}
