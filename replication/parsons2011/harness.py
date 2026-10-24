# SPDX-License-Identifier: GPL-3.0-or-later
"""Replicate Parsons et al. (2011) with dfeols and check it against pyfixest,
reghdfe and the published tables.

    python harness.py inspect  --dta calledpitch.dta --do 20081366_statacode.do
    #   -> fill in VARMAP (and check SPECS) in config.py
    python harness.py build    --dta calledpitch.dta --out WORK
    python harness.py run      --data WORK [--executor serial|dask:4|<profile>] [--specs T3c9 T4]
    python harness.py stata-do --data WORK      # then run WORK/reference_reghdfe.do in Stata
    python harness.py report   --data WORK      # exit code 1 if any check FAILs
    python harness.py authors  --dta calledpitch.dta --data WORK
    #   -> the authors' approximate demeaning, to check the data against the published tables

WORK holds everything: analysis.parquet/.dta, shards/<dataset>, runs/<label>,
results/*.json, reghdfe_results.csv, comparison.csv and report.md.

calledpitch.dta is small, so this reads it with pandas in memory (agreed exception to the
project's lazy-Polars rule, 2026-10-23). Don't copy that pattern to large datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import config  # noqa: E402

log = logging.getLogger("dfeols.replication")

_EST = re.compile(
    r"^(?!\s*\*).*\b(reg|regress|areg|reghdfe|xtreg|felsdvreg|a2reg|probit|logit|dprobit|"
    r"xtlogit)\b",
    re.I,
)
_PREP = re.compile(r"^\s*(gen|generate|egen|replace|keep|drop|xi|recode)\b", re.I)


# --------------------------------------------------------------------------- inspect
def cmd_inspect(a):
    with pd.read_stata(a.dta, iterator=True) as rdr:
        labels = rdr.variable_labels()
    df = pd.read_stata(a.dta, convert_categoricals=False)
    print(f"{a.dta}: {len(df):,} rows, {df.shape[1]} columns\n")
    print(f"{'variable':<24}{'type':<10}{'distinct':>10}{'missing':>10}  label")
    for c in df.columns:
        s = df[c]
        print(f"{c:<24}{str(s.dtype):<10}{s.nunique():>10,}{int(s.isna().sum()):>10,}  {labels.get(c, '')}")
    if a.do:
        print(f"\nEstimation commands in {a.do} (line numbers):")
        text = Path(a.do).read_text(errors="replace").splitlines()
        for i, line in enumerate(text, 1):
            if _EST.search(line):
                print(f"  {i:>4}: {line.strip()}")
        print("\nVariable construction:")
        for i, line in enumerate(text, 1):
            if _PREP.search(line):
                print(f"  {i:>4}: {line.strip()}")
    print("\nNext: set VARMAP in config.py, check SPECS against the estimation lines, then `build`.")


# --------------------------------------------------------------------------- build
def _from_dummies(raw: pd.DataFrame, base: int, levels: dict[str, int]) -> pd.Series:
    """Categorical from a set of mutually exclusive dummies; rows with none get `base`."""
    d = raw[list(levels)]
    if (d.sum(axis=1) > 1).any():
        sys.exit(f"dummies {list(levels)} are not mutually exclusive")
    return pd.Series(
        np.select([d[c].eq(1) for c in levels], list(levels.values()), base), index=raw.index, dtype=np.int64
    )


def _derive(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes = []
    if "upm" not in df and {"ump_race", "pitcher_race"} <= set(df):
        df["upm"] = (df["ump_race"].astype(str) == df["pitcher_race"].astype(str)).astype(np.int8)
        df.loc[df["ump_race"].isna() | df["pitcher_race"].isna(), "upm"] = np.nan
        notes.append("upm = (ump_race == pitcher_race)")
    if "count" not in df and {"balls", "strikes"} <= set(df):
        df["count"] = df["balls"] * 10 + df["strikes"]
        notes.append("count = 10*balls + strikes")
    if "terminal" not in df and {"balls", "strikes"} <= set(df):
        df["terminal"] = ((df["strikes"] == 2) | (df["balls"] == 3)).astype(np.int8)
        notes.append("terminal = (strikes == 2) | (balls == 3)")
    if "well_att" not in df and {"attendance", "capacity"} <= set(df):
        df["well_att"] = (df["attendance"] / df["capacity"] >= config.WELL_ATTENDED_SHARE).astype(np.int8)
        notes.append(f"well_att = attendance/capacity >= {config.WELL_ATTENDED_SHARE}")
    return df, notes


def cmd_build(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mapped = {k: v for k, v in config.VARMAP.items() if v}
    if not mapped:
        sys.exit("VARMAP in config.py is empty: run `inspect` and fill it in first.")
    dummy_sets = getattr(config, "DUMMY_SETS", {})
    cols = set(mapped.values()) | {c for _, lv in dummy_sets.values() for c in lv}
    raw = pd.read_stata(a.dta, convert_categoricals=False, columns=sorted(cols))
    df = pd.DataFrame({k: raw[v] for k, v in mapped.items()})
    for name, (base, levels) in dummy_sets.items():
        df[name] = _from_dummies(raw, base, levels)
    del raw
    df, notes = _derive(df)
    notes += [f"{name} from dummies {', '.join(lv)} (base {base})" for name, (base, lv) in dummy_sets.items()]
    sample = getattr(config, "SAMPLE", None)
    if sample:
        n0 = len(df)
        df = df.query(sample)
        notes.append(f"sample {sample}: kept {len(df):,} of {n0:,}")
    missing = [c for c in config.REQUIRED if c not in df]
    if missing:
        sys.exit(f"Missing after mapping and derivation: {missing}. Map them in VARMAP.")
    if a.years:
        if "year" not in df:
            sys.exit("--years needs VARMAP['year']")
        lo, hi = a.years
        df = df[df["year"].between(lo, hi)]
        notes.append(f"seasons {lo}-{hi}")
    for c in config.IDS:
        if c in df and not pd.api.types.is_integer_dtype(df[c]):
            codes, _ = pd.factorize(df[c], sort=True)
            df[c] = np.where(codes < 0, np.nan, codes)
    n0 = len(df)
    keyed = [c for c in config.IDS + ["questec"] if c in df]
    df = df.dropna(subset=keyed)
    for c in keyed:
        df[c] = df[c].astype(np.int64)
    if len(df) < n0:
        notes.append(f"dropped {n0 - len(df):,} rows missing an id or questec")
    df = df.reset_index(drop=True)
    df.to_parquet(out / "analysis.parquet", index=False)
    df.to_stata(out / "analysis.dta", write_index=False, version=118)
    info = {
        "source": str(a.dta),
        "rows": len(df),
        "columns": list(df.columns),
        "levels": {c: int(df[c].nunique()) for c in keyed},
        "notes": notes,
    }
    (out / "build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


# --------------------------------------------------------------------------- run
def _plan(data: Path, only: list[str] | None):
    """Specs and datasets, minus anything that needs a column the data lacks."""
    cols = set(json.loads((data / "build.json").read_text())["columns"])
    from dfeols.formula import fe_components, parse

    specs = []
    for s in config.SPECS:
        if only and s.label not in only:
            continue
        f = parse(s.formula)
        need = set(f.variables) | {c for fe in f.fes for c in fe_components(fe)}
        need |= {s.cluster} if s.cluster else set()
        if need - cols:
            log.warning("skipping %s: data lacks %s", s.label, sorted(need - cols))
            continue
        specs.append(s)
    used = {d for s in specs for d in s.datasets}
    datasets = {}
    for d in config.DATASETS:
        if d.name in used:
            datasets[d.name] = replace(d, clusters=tuple(c for c in d.clusters if c in cols))
    return specs, datasets


def run(
    data: Path,
    only=None,
    executor: str = "serial",
    target_rows: int = 150_000,
    pyfixest: bool = True,
    max_dummy_levels: int = 2000,
) -> dict:
    data = Path(data)
    specs, datasets = _plan(data, only)
    src = data / "analysis.parquet"
    res_dir = data / "results"
    res_dir.mkdir(exist_ok=True)
    ex, close = common.make_executor(executor)
    try:
        prepped = {}
        for name, d in datasets.items():
            log.info("prep %s (partition %s)", name, d.partition)
            prepped[name] = common.prep_dataset(src, data / "shards" / name, d, target_rows)
        for s in specs:
            for d in s.datasets:
                lab = s.run_label(d)
                log.info("dfeols %s", lab)
                rec = common.fit_dfeols(prepped[d], s, d, ex, data / "runs" / lab, max_dummy_levels)
                rec["executor"] = executor
                (res_dir / f"dfeols_{lab}.json").write_text(json.dumps(rec, indent=1))
    finally:
        close()
    if pyfixest:
        pdf = pd.read_parquet(src)
        for s in specs:
            log.info("pyfixest %s", s.label)
            rec = common.fit_pyfixest(pdf, s)
            (res_dir / f"pyfixest_{s.label}.json").write_text(json.dumps(rec, indent=1))
    return check(data)


def cmd_run(a):
    out = run(Path(a.data), a.specs, a.executor, a.target_rows, not a.no_pyfixest, a.max_dummy_levels)
    print(f"{out['verdict']}: {out['counts']}  ->  {Path(a.data) / 'report.md'}")


# --------------------------------------------------------------------------- stata
def cmd_stata_do(a):
    data = Path(a.data).resolve()
    specs, _ = _plan(data, a.specs)
    p = common.write_reghdfe_do(
        specs,
        data / "analysis.dta",
        data / "reghdfe_results.csv",
        data / "reference_reghdfe.do",
        config.TITLE,
    )
    print(f"wrote {p}\nRun it in Stata (it writes {data / 'reghdfe_results.csv'}), then `report`.")


# --------------------------------------------------------------------------- report
def check(data: Path) -> dict:
    data = Path(data)
    res_dir = data / "results"
    ours = {r["label"]: r for r in map(lambda p: json.loads(p.read_text()), res_dir.glob("dfeols_*.json"))}
    pyf = {r["label"]: r for r in map(lambda p: json.loads(p.read_text()), res_dir.glob("pyfixest_*.json"))}
    stata = common.read_reghdfe_csv(data / "reghdfe_results.csv")
    specs = [s for s in config.SPECS if any(s.run_label(d) in ours for d in s.datasets)]
    cmp = common.compare(specs, ours, pyf, stata, config.PUBLISHED, config.PUBLISHED_N)
    cmp.to_csv(data / "comparison.csv", index=False)
    build = json.loads((data / "build.json").read_text())
    notes = [
        f"Data: {build['rows']:,} called pitches; levels {build['levels']}; "
        f"{'; '.join(build['notes']) or 'no derived variables'}.",
        f"Published values: {config.PUBLISHED_SOURCE}.",
    ]
    if not stata:
        notes.append("No reghdfe results yet: run `stata-do`, then the do-file in Stata.")
    return common.write_report(data / "report.md", config.TITLE, specs, cmp, ours, pyf, stata, notes)


def cmd_report(a):
    out = check(Path(a.data))
    print(f"{out['verdict']}: {out['counts']}  ->  {Path(a.data) / 'report.md'}")
    sys.exit(1 if out["verdict"] == "FAIL" else 0)


# --------------------------------------------------------------------------- authors
def _hc1(y: np.ndarray, X: np.ndarray, absorbed: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS on demeaned data with Stata's robust SEs, scaled by (N-1)/(N-p-a): a is areg's absorbed
    levels, or 1 for reg's constant. Data demeaned in one pass are not exactly mean zero, so
    reg gets its constant."""
    if not absorbed:
        X = np.column_stack([X, np.ones(len(X))])
    p = X.shape[1] - (0 if absorbed else 1)
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ (X.T @ y)
    e = y - X @ b
    n = len(y)
    V = xtx_inv @ ((X.T * e**2) @ X) @ xtx_inv * (n - 1) / (n - p - max(absorbed, 1))
    return b[:p], np.sqrt(np.diag(V))[:p]


def authors_procedure(dta: str) -> dict:
    """config.AUTHORS fit the way the do-file does: one-pass group demeaning on the full file,
    then areg (one absorbed FE) or reg on the sample. Returns {spec: {term: (b, se)}, "N": ...}."""
    out = {}
    raw = pd.read_stata(dta, convert_categoricals=False)
    for label, m in config.AUTHORS.items():
        ctrl = m.get("controls", config.AUTHORS_CONTROLS)
        cols = ["callstrike", *m["x"], *ctrl]
        d = raw[cols].astype(np.float64)
        for g in m["demean"]:
            d = d - d.groupby(raw[g]).transform("mean")
        keep = raw.eval(m["sample"]).to_numpy()
        d = d[keep]
        absorbed = 0
        if m["absorb"]:
            g = raw.loc[keep, m["absorb"]]
            d = d - d.groupby(g).transform("mean")
            absorbed = int(g.nunique())
        b, se = _hc1(d["callstrike"].to_numpy(), d[[*m["x"], *ctrl]].to_numpy(), absorbed)
        terms = {config.AUTHORS_TERMS.get(x, x): (float(b[i]), float(se[i])) for i, x in enumerate(m["x"])}
        out[label] = {"N": int(keep.sum()), "coefs": terms}
        log.info("authors %s: %s", label, {t: round(v[0], 5) for t, v in terms.items()})
    return out


def cmd_authors(a):
    res = authors_procedure(a.dta)
    data = Path(a.data)
    (data / "results").mkdir(parents=True, exist_ok=True)
    (data / "results" / "authors.json").write_text(json.dumps(res, indent=1))
    # the published tables mix rounding and truncation, so allow one unit in the last digit
    print(f"{'spec':<6} {'term':<14} {'authors b':>10} {'se':>8} {'published':>18}  within 1 unit")
    bad = 0
    for label, r in res.items():
        pub = config.PUBLISHED.get(label, {})
        for t, (b, se) in r["coefs"].items():
            p = pub.get(t)
            ok = ""
            if p:
                unit = 10.0 ** -p[2] + 1e-12
                good = abs(b - p[0]) <= unit and (p[1] is None or abs(se - p[1]) <= unit)
                ok, bad = ("yes" if good else "NO"), bad + (not good)
            ps = f"{p[0]} ({p[1]})" if p else ""
            print(f"{label:<6} {t:<14} {b:>10.6f} {se:>8.6f} {ps:>18}  {ok}")
    print("all published values reproduced" if not bad else f"{bad} published values not reproduced")
    sys.exit(1 if bad else 0)


# --------------------------------------------------------------------------- main
def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect", help="list the .dta variables and the do-file's estimation lines")
    s.add_argument("--dta", required=True)
    s.add_argument("--do")
    s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("build", help="standardized analysis file (Parquet + .dta)")
    s.add_argument("--dta", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"))
    s.set_defaults(fn=cmd_build)
    s = sub.add_parser("run", help="fit with dfeols and pyfixest, then compare")
    s.add_argument("--data", required=True)
    s.add_argument("--specs", nargs="*")
    s.add_argument("--executor", default="serial", help="serial | dask:N | <site profile>")
    s.add_argument(
        "--target-rows",
        type=int,
        default=150_000,
        help="rows per shard; small, so the ~1M-row file spans several shards",
    )
    s.add_argument("--max-dummy-levels", type=int, default=2000)
    s.add_argument("--no-pyfixest", action="store_true")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("stata-do", help="write the reghdfe reference do-file")
    s.add_argument("--data", required=True)
    s.add_argument("--specs", nargs="*")
    s.set_defaults(fn=cmd_stata_do)
    s = sub.add_parser("report", help="compare all results and write report.md")
    s.add_argument("--data", required=True)
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("authors", help="the authors' approximate procedure, vs the published values")
    s.add_argument("--dta", required=True)
    s.add_argument("--data", required=True)
    s.set_defaults(fn=cmd_authors)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
