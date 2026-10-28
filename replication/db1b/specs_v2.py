# SPDX-License-Identifier: GPL-3.0-or-later
"""Specs for db1b_gs_tickets_v2 and their references (STATE.md 2026-10-28).

References are pyfixest fits, or ivreghdfe where there are two endogenous regressors.

    python repo/replication/db1b/specs_v2.py reference \
        --data 'build/db1b/tickets_v2/*.parquet' --out reference.json

Every spec: FEs carrier^route_id + time, CRV1 by route_id, passenger weights. The weight column
picks the sample (zero outside it; dfeols and pyfixest drop zero weights). Instruments are the
Table 2 ones with IRUTHERF from first stages fit before and after 2006Q4 (`irutherf_pp`).

tail     IV effect of competition (-lnHERF) on ln price of the cell's top / bottom tail, before
         and after 2006Q4: the ticket-level version of the panel's ln P90 / ln P10 result
period   OLS -lnHERF by period (1993-99, 2000-2006Q3, 2006Q4-12, 2013-19, 2020-25), top / bottom tail
oneway   IV -lnHERF and its interaction with roundtrip, all estimation tickets, before / after
hub      IV -lnHERF and its interaction with the carrier's hub share, top / bottom tail after 2006Q4
v1_*     the db1b_gs_tickets_v1 README specs (unweighted, all rows), on the same rows
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import polars as pl

FE = "carrier^route_id + time"
Z = ["ln_passrte", "gensp", "irutherf_pp"]
IV = f"neg_ln_herf ~ {' + '.join(Z)}"


def _ix(v):
    return " + ".join(f"{z}:{v}" for z in Z)


SPECS = {
    "tail": {
        "formula": f"ln_price ~ bankrupt | {FE} | {IV}",
        "runs": {
            "top_pre": "w_top_pre",
            "top_post": "w_top_post",
            "bot_pre": "w_bot_pre",
            "bot_post": "w_bot_post",
        },
    },
    "period": {
        "formula": f"ln_price ~ neg_ln_herf:i(period) + bankrupt | {FE}",
        "runs": {"top": "w_top", "bot": "w_bot"},
    },
    "oneway": {
        "formula": f"ln_price ~ bankrupt + roundtrip | {FE} | "
        f"neg_ln_herf + neg_ln_herf:roundtrip ~ {' + '.join(Z)} + {_ix('roundtrip')}",
        "runs": {"pre": "w_pre", "post": "w_post"},
    },
    "hub": {
        "formula": f"ln_price ~ bankrupt + hubsh | {FE} | "
        f"neg_ln_herf + neg_ln_herf:hubsh ~ {' + '.join(Z)} + {_ix('hubsh')}",
        "runs": {"top_post": "w_top_post", "bot_post": "w_bot_post"},
    },
    "v1_counts": {"formula": f"ln_price ~ legn + lccn + bankrupt | {FE}", "runs": {"all": None}},
    "v1_herf": {"formula": f"ln_price ~ neg_ln_herf + ln_n + bankrupt | {FE}", "runs": {"all": None}},
}
# reference subsamples (pyfixest in memory); a run's subsample must hold rows of its weight
SUBSAMPLES = {"y2005": [2005], "y2015": [2015], "y2005_2007": [2005, 2006, 2007]}
REFERENCE = {
    "tail": {"top_pre": "y2005", "bot_pre": "y2005", "top_post": "y2015", "bot_post": "y2015"},
    "period": {"top": "y2005_2007", "bot": "y2005_2007"},
    "oneway": {"pre": "y2005", "post": "y2015"},
    "hub": {"top_post": "y2015", "bot_post": "y2015"},
    "v1_counts": {"all": "y2005"},
    "v1_herf": {"all": "y2005"},
}
PREP = {"partition": "route_id", "fixed_effects": ["carrier^route_id", "time"], "clusters": ["route_id"]}


def runs():
    for spec, s in SPECS.items():
        for run, w in s["runs"].items():
            yield f"{spec}_{run}", s["formula"], w, REFERENCE[spec][run]


def subsample(data: str, name: str) -> pl.DataFrame:
    return pl.scan_parquet(data).filter(pl.col("year").is_in(SUBSAMPLES[name])).collect(engine="streaming")


def _stata():
    import shutil

    for name in ("stata-mp", "stata-se", "stata"):
        if shutil.which(name):
            return shutil.which(name)
    return next((str(p) for p in sorted(Path("/usr/local").glob("stata*/stata-mp"), reverse=True)), None)


def ivreghdfe(d: pl.DataFrame, fml: str, w: str, work: Path) -> dict:
    """The run in Stata's ivreghdfe (aweights, absorb carrier^route_id and time, cluster route_id,
    small): for IV with two endogenous regressors, which pyfixest does not fit."""
    import subprocess

    lhs, rest = fml.split(" ~ ", 1)
    exog, _, iv = rest.split(" | ")
    endog, inst = iv.split(" ~ ")
    names, cols = {}, {}

    def var(term):  # a:b -> generated product column
        if ":" not in term:
            return term
        v = "x" + str(len(cols))
        a, b = term.split(":")
        cols[v] = pl.col(a) * pl.col(b)
        names[v] = term
        return v

    ex, en, zz = ([var(t.strip()) for t in part.split(" + ")] for part in (exog, endog, inst))
    raw = {c for t in (exog + " + " + endog + " + " + inst).split(" + ") for c in t.strip().split(":")}
    keep = [lhs, *sorted(raw), w, "carrier", "route_id", "time"]
    x = d.filter(pl.col(w) > 0).select(keep).with_columns(**cols)
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    x.write_csv(work / "d.csv")
    do = [
        f'import delimited using "{work / "d.csv"}", clear asdouble',  # default is float (single)
        "egen long cr = group(carrier route_id)",
        f"ivreghdfe {lhs} {' '.join(ex)} ({' '.join(en)} = {' '.join(zz)}) [aw={w}], "
        "absorb(cr time) cluster(route_id) small",
        f'file open fh using "{work / "out.txt"}", write replace',
        "matrix b = e(b)",
        "matrix V = e(V)",
        "local nm : colnames b",
        "forvalues i = 1/`=colsof(b)' {",
        "  local n : word `i' of `nm'",
        "  file write fh \"`n' \" %21.15g (b[1,`i']) \" \" %21.15g (sqrt(V[`i',`i'])) _n",
        "}",
        'file write fh "N " %21.15g (e(N)) _n',
        "file close fh",
    ]
    (work / "run.do").write_text("\n".join(do) + "\n")
    subprocess.run([_stata(), "-b", "do", "run.do"], cwd=work, capture_output=True, text=True, check=True)
    log = (work / "run.log").read_text()
    if "\nr(" in log:
        raise RuntimeError(log[-2000:])
    coef, n = {}, None
    for line in (work / "out.txt").read_text().splitlines():
        k, *v = line.split()
        if k == "N":
            n = int(float(v[0]))
        elif k != "_cons":
            coef[names.get(k, k)] = [float(v[0]), float(v[1])]
    # ivreghdfe's SEs differ from pyfixest's (and dfeols's) by a factor 1 + 1/(2N) on some samples:
    # one parameter more in its small-sample count (tail_top_post shows it against pyfixest)
    return {"N": n, "coef": coef, "tool": "ivreghdfe", "se_rtol": 1 / n}


def reference(data: str, work: Path, check: str | None = None) -> dict:
    """pyfixest for every run it can fit; ivreghdfe for two endogenous regressors (and for the
    `check` run as well, to show the two agree)."""
    import pyfixest as pf

    warnings.simplefilter("ignore")
    out = {}
    for sub in SUBSAMPLES:
        d = subsample(data, sub)
        for label, fml, w, s in runs():
            if s != sub:
                continue
            multi = fml.count(" | ") == 2 and " + " in fml.split(" | ")[2].split(" ~ ")[0]
            if multi:
                out[label] = ivreghdfe(d, fml, w, work / label)
            else:
                x = (d.filter(pl.col(w) > 0) if w else d).to_pandas()
                m = pf.feols(fml, data=x, vcov={"CRV1": "route_id"}, weights=w)
                t = m.tidy()
                out[label] = {
                    "N": int(m._N),
                    "coef": {
                        k: [float(t.loc[k, "Estimate"]), float(t.loc[k, "Std. Error"])] for k in t.index
                    },
                    "tool": "pyfixest",
                }
                if label in check.split(","):
                    out[label]["ivreghdfe"] = ivreghdfe(d, fml, w, work / label)
            out[label]["subsample"] = sub
            print(
                label,
                sub,
                out[label]["tool"],
                out[label]["N"],
                {k: [round(v, 5) for v in b] for k, b in out[label]["coef"].items()},
                flush=True,
            )
            if "ivreghdfe" in out[label]:
                print(
                    "   ivreghdfe", out[label]["ivreghdfe"]["N"], out[label]["ivreghdfe"]["coef"], flush=True
                )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["reference"])
    ap.add_argument("--data", default="build/db1b/tickets_v2/*.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="tmp/specs_v2", type=Path, help="Stata scratch")
    ap.add_argument(
        "--check", default="tail_top_pre,tail_top_post", help="also fit these runs with ivreghdfe"
    )
    a = ap.parse_args(argv)
    ref = {
        "prep": PREP,
        "subsamples": {k: {"year": v} for k, v in SUBSAMPLES.items()},
        "specs": {label: {"formula": f, "weights": w, "subsample": s} for label, f, w, s in runs()},
        "cluster": "route_id",
        "vcov": ["CRV1"],
        "reference": reference(a.data, a.work, a.check),
    }
    with open(a.out, "w") as f:
        json.dump(ref, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
