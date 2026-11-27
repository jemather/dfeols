# SPDX-License-Identifier: GPL-3.0-or-later
"""Exactness check: dfeols vs pyfixest on one ZTF field (in memory).

    python -I repo/replication/ztf/pyfixest_check.py --data build/ztf/analysis.parquet \
        --root build/ztf/check --field 222

Writes the field's rows to <root>/field.parquet, fits S1-S3 (fit.py) with pyfixest and with
dfeols, and prints the relative differences. On field 222 of the dial-in partition (3.7M rows):
coefficients agree to ~1e-14, SEs to ~1e-10.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit import SPECS

CHECK = ["S1", "S2", "S3"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--field", type=int, default=222)
    a = ap.parse_args()

    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    sub = root / "field.parquet"
    d = pl.read_parquet(a.data).filter(pl.col("fieldid") == a.field)
    d.write_parquet(sub)
    print(f"field {a.field}: {d.height:,} rows")
    res = root / "results_dfeols.json"
    fit = str(Path(__file__).with_name("fit.py"))
    cmd = [sys.executable, "-I", fit, "--data", str(sub), "--root", str(root)]
    subprocess.run([*cmd, "--only", ",".join(CHECK), "--out", str(res)], check=True)
    got = json.loads(res.read_text())
    pdf = d.to_pandas()
    for sid in CHECK:
        _, f, kw = SPECS[sid]
        vc = {"CRV1": kw["cluster"]}
        m = pf.feols(f, data=pdf, vcov=vc, weights=kw.get("weights"))
        for v, b in m.coef().items():
            db = abs(got[sid]["coef"][v] / b - 1)
            ds = abs(got[sid]["se"][v] / m.se()[v] - 1)
            print(f"{sid} {v:11s} coef rel diff {db:.1e}  se rel diff {ds:.1e}")


if __name__ == "__main__":
    main()
