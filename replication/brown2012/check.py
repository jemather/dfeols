# SPDX-License-Identifier: GPL-3.0-or-later
"""Farbmacher & Tauchmann (2021) Table 3 with dfeols: Brown & Laschever's retirement model
(Tables_2_to_5.do column 7) by the within estimator, first differences and lpm_dthazard's IV,
school-clustered SEs; compared with the published numbers (3 decimals). docs/brown2012.md.

    python check.py build/brown2012/sample.parquet <workdir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

import dfeols

X = [
    "PW_0_100k",
    "PK_is_positive",
    "salary_10k",
    "serv_sqr",
    "age_aug_31_o55_minusi",
    "service_o55_minusi",
    "pupilteachratio",
    "pct_maplus",
    "pct_feml",
    "qmathmeanss",
    "retire_o55_sizeminusi",
]
AGES = [53, 54, *range(56, 67)]
# F&T Table 3 (coef, SE), rows in the paper's order; None where the column has no such row
PUB = {
    "PW_0_100k": [(0.038, 0.008), (0.014, 0.008), (0.021, 0.016)],
    "PK_is_positive": [(-0.088, 0.020), (-0.066, 0.020), (-0.100, 0.028)],
    "salary_10k": [(0.023, 0.016), (0.020, 0.017), (0.054, 0.043)],
    "serv_sqr": [(0.001, 0.000), (0.002, 0.000), (-0.000, 0.000)],
    "age_aug_31_o55_minusi": [(-0.001, 0.003), (-0.002, 0.003), (-0.000, 0.007)],
    "service_o55_minusi": [(-0.002, 0.001), (-0.002, 0.001), (-0.004, 0.002)],
    "pupilteachratio": [(0.001, 0.001), (0.001, 0.001), (0.010, 0.005)],
    "pct_maplus": [(-0.115, 0.074), (-0.139, 0.072), (-0.399, 0.173)],
    "pct_feml": [(0.142, 0.067), (0.145, 0.061), (0.310, 0.178)],
    "qmathmeanss": [(0.000, 0.003), (0.002, 0.003), (-0.004, 0.009)],
    "retire_o55_sizeminusi": [(-0.001, 0.001), (-0.002, 0.001), (0.002, 0.001)],
    "age53": [(0.306, 0.025), (0.350, 0.029), None],
    "age54": [(0.141, 0.013), (0.168, 0.015), (-0.000, 0.009)],
    "age56": [(-0.156, 0.013), (-0.178, 0.015), (-0.025, 0.009)],
    "age57": [(-0.310, 0.025), (-0.356, 0.029), (-0.026, 0.011)],
    "age58": [(-0.452, 0.036), (-0.522, 0.043), (-0.020, 0.013)],
    "age59": [(-0.567, 0.048), (-0.658, 0.057), (0.008, 0.016)],
    "age60": [(-0.631, 0.060), (-0.738, 0.071), (0.060, 0.019)],
    "age61": [(-0.694, 0.072), (-0.799, 0.086), (0.066, 0.022)],
    "age62": [(-0.686, 0.086), (-0.785, 0.101), (0.119, 0.027)],
    "age63": [(-0.727, 0.097), (-0.804, 0.114), (0.077, 0.032)],
    "age64": [(-0.788, 0.109), (-0.860, 0.127), (0.025, 0.031)],
    "age65": [(-0.794, 0.119), (-0.871, 0.141), (0.066, 0.033)],
    "age66": [(-0.825, 0.131), (-0.898, 0.154), (0.046, 0.036)],
    "year2000": [(0.092, 0.016), None, None],
    "year2001": [(0.197, 0.029), (0.030, 0.010), (0.015, 0.009)],
    "constant": [None, (0.103, 0.020), (-0.437, 0.472)],  # WI constant not reported by dfeols
}
N_PUB = {"wi": 21_290, "fd": 12_968, "iv": 12_968}
TEACHERS_PUB = {"wi": 8_320, "fd": 7_088, "iv": 7_088}
CLUSTERS_PUB = {"wi": 586, "fd": 578, "iv": 578}


def name(row: str, est: str) -> str:
    if row in X:
        return row if est != "fd" else f"D1_{row}"
    if row.startswith("age"):
        a = row[3:]
        return f"agecat::{a}" if est != "fd" else f"D1_agecat_{a}"
    if row.startswith("year"):
        y = row[4:]
        return f"ac_year::{y}" if est != "fd" else f"D1_ac_year_{y}"
    return "Intercept"


def main(src: str, workdir: str):
    wd = Path(workdir)
    df = pl.read_parquet(src).filter(pl.col("bad_age") == 0)
    ds = dfeols.prep(
        df,
        shard_dir=wd / "shards",
        partition="empno",
        fixed_effects=["empno"],
        clusters=["schoolcode", "empno"],
        target_rows=10_000,
        overwrite=True,
    )
    rhs = " + ".join(X) + " + i(agecat, ref=55) + i(ac_year)"
    wi = dfeols.fit(
        ds,
        f"retire ~ {rhs} | empno",
        cluster="schoolcode",
        drop_singletons=False,
        workdir=str(wd / "wi"),
        log_every=1e9,
    )
    h = dfeols.lpm_dthazard(
        ds,
        f"retire ~ {rhs}",
        unit="empno",
        time="ac_year",
        cluster="schoolcode",
        workdir=str(wd / "hazard"),
        log_every=1e9,
    )
    # the paper's FD column: Δ of every regressor incl. age 53 (not an IV regressor), Δ year 2001
    sds = dfeols.Dataset.open(h.dataset)
    fd_terms = [f"D1_{x}" for x in X] + [f"D1_agecat_{a}" for a in AGES] + ["D1_ac_year_2001"]
    fd = dfeols.fit(
        sds, "retire ~ " + " + ".join(fd_terms), cluster="schoolcode", workdir=str(wd / "fd"), log_every=1e9
    )
    fits = {"wi": wi, "fd": fd, "iv": h.iv}
    # F&T ran the IV with ivreg2 (their fn. 17), whose default cluster SEs omit fixest's (and
    # Stata reg's) (N-1)/(N-K) factor; compare the IV SEs on that convention
    n_iv, k_iv = h.iv.N, len(h.iv.coefnames)
    se_scale = {"wi": 1.0, "fd": 1.0, "iv": float(np.sqrt((n_iv - k_iv) / (n_iv - 1)))}
    rows, ok = [], True
    for row, pubs in PUB.items():
        for k, est in enumerate(("wi", "fd", "iv")):
            if pubs[k] is None:
                continue
            r = fits[est]
            n = name(row, est)
            if n not in r.coefnames:
                rows.append({"row": row, "est": est, "missing": True})
                ok = False
                continue
            i = r.coefnames.index(n)
            b, se = float(r.coef()[i]), float(np.sqrt(r.vcov()[i, i])) * se_scale[est]
            pb, ps = pubs[k]
            good = abs(round(b, 3) - pb) < 1e-9 and abs(round(se, 3) - ps) < 1e-9
            near = abs(b - pb) <= 0.0005 + 1e-9 and abs(se - ps) <= 0.0005 + 1e-9
            ok &= good
            rows.append(
                {
                    "row": row,
                    "est": est,
                    "ours": b,
                    "se": se,
                    "pub": pb,
                    "pub_se": ps,
                    "exact": good,
                    "within_rounding": near,
                }
            )
    counts = {
        est: {
            "N": int(fits[est].N),
            "N_pub": N_PUB[est],
            "clusters": int(fits[est].G),
            "clusters_pub": CLUSTERS_PUB[est],
        }
        for est in fits
    }
    print(f"{'row':<24}{'est':<4}{'ours':>9}{'se':>8}{'paper':>9}{'se':>8}")
    for r in rows:
        if r.get("missing"):
            print(f"{r['row']:<24}{r['est']:<4}  MISSING")
            continue
        flag = "" if r["exact"] else "  DIFF"
        print(
            f"{r['row']:<24}{r['est']:<4}{r['ours']:>9.4f}{r['se']:>8.4f}{r['pub']:>9.3f}{r['pub_se']:>8.3f}{flag}"
        )
    print(json.dumps(counts))
    print("excluded levels:", h.iv.run["dthazard"]["excluded_levels"], "sample:", h.sample)
    out = {"ok": bool(ok), "rows": rows, "counts": counts, "sample": h.sample, "warnings": h.warnings}
    (wd / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"IV SEs on ivreg2's convention (x {se_scale['iv']:.6f})")
    print("ALL MATCH THE PUBLISHED DIGITS" if ok else "SOME DIFFER")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
