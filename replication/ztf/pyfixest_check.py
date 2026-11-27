# SPDX-License-Identifier: GPL-3.0-or-later
"""Exactness check: dfeols vs pyfixest on one ZTF field (in memory).

    python -I repo/replication/ztf/pyfixest_check.py --data build/ztf/analysis.parquet \
        --root build/ztf/check --field 222

Writes the field's rows to <root>/field.parquet, fits the OLS specs S1-S3 and the IV specs S7-S8
(fit.py) with pyfixest and with dfeols, prints the relative differences, and writes
<root>/results_dfeols.json and <root>/results_pyfixest.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pyfixest as pf
from pyfixest.demeaners import MapDemeaner

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit import SPECS

CHECK = ["S1", "S2", "S3", "S7", "S8"]
# pyfixest 0.60 demeans to 1e-6 by default; tighten it so differences are dfeols's, not convergence
DEMEANER = MapDemeaner(fixef_tol=1e-13, fixef_maxiter=100_000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--field", type=int, default=222)
    a = ap.parse_args()

    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    sub = root / "field.parquet"
    pl.scan_parquet(a.data).filter(pl.col("fieldid") == a.field).sink_parquet(sub)
    d = pl.read_parquet(sub)
    print(f"field {a.field}: {d.height:,} rows")
    res = root / "results_dfeols.json"
    fit = str(Path(__file__).with_name("fit.py"))
    cmd = [sys.executable, "-I", fit, "--data", str(sub), "--root", str(root)]
    subprocess.run([*cmd, "--only", ",".join(CHECK), "--out", str(res)], check=True)
    got = json.loads(res.read_text())
    # Float32 columns (mag, w, cc_gr): pyfixest keeps float32 weights in lower precision (~1e-8 off on
    # the weighted specs), dfeols promotes to float64. Give pyfixest float64 so both see the same numbers.
    pdf = d.with_columns(pl.selectors.float().cast(pl.Float64)).to_pandas()
    ref = {}
    for sid in CHECK:
        _, f, kw = SPECS[sid]
        vc = {"CRV1": kw["cluster"]}
        m = pf.feols(f, data=pdf, vcov=vc, weights=kw.get("weights"), demeaner=DEMEANER)
        ref[sid] = {"coef": m.coef().to_dict(), "se": m.se().to_dict(), "nobs": int(m._N)}
        for v, b in m.coef().items():
            db = abs(got[sid]["coef"][v] / b - 1)
            ds = abs(got[sid]["se"][v] / m.se()[v] - 1)
            print(f"{sid} {v:11s} coef rel diff {db:.1e}  se rel diff {ds:.1e}")
    (root / "results_pyfixest.json").write_text(json.dumps(ref, indent=1))


if __name__ == "__main__":
    main()
