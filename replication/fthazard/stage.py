# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage one simulated dataset for OSCER: outbox/<name>/ with data/, README.md,
reference.json, simulate.json and check.py (the manifest is written afterwards by the job).

    python stage.py build/fthazard/fthazard_st_v1 outbox/fthazard_st_v1 <commit>
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
DGP_TEXT = {
    "st": "stationary: x_it = a_i + 0.165 + 0.07 z_it",
    "rw": "random walk: x_it = x_i,t-1 + 0.1 n_it - 0.05, x_i1 = a_i + 0.2",
    "tr": "trend: x_it = a_i + 0.175 + 0.025 t h_it",
}


def readme(name: str, info: dict, ref: dict, stats: dict, commit: str) -> str:
    te = info["time_effects"]
    tables = "Table A4" if te else "Tables 1 and A3"
    lines = [
        f"# {name}",
        "",
        "Simulated discrete-time hazard panel: the large-sample Monte Carlo of Farbmacher & Tauchmann",
        "(2021),"
        + f' "Linear fixed-effects estimation with non-repeated outcomes", FAU DP 03/2021, {tables}.',
        "For checking `dfeols.lpm_dthazard` (the paper's IV: y on levels, Δx as instruments) at the paper's",
        "own size, against its published numbers, and as a scale test.",
        "",
        f"- **DGP:** {DGP_TEXT[info['dgp']]}; z, n, h iid Beta(0.2, 0.2); a_i ~ U(-0.05, 0.05);",
        "  P(y_it = 1) = 0.1 + a_i + x_it" + (" + tau_t, tau_3 = tau_5 = 0.2" if te else "") + "; T = 5;",
        "  each unit observed until its first event or t = 5 (no rows after the event)",
        f"- **Build:** `repo/replication/fthazard/simulate.py --dgp {info['dgp']} --units {info['units']}"
        + (" --time-effects" if te else "")
        + f" --seed {info['seed']}`, jemather/dfeols `{commit}`; deterministic (same arguments, same files)",
        f"- **Rows:** {info['rows']:,} in {len(info['parts'])} part(s); {info['units']:,} units; "
        f"{info['events']:,} events; no nulls",
        "- **Notes:** `docs/fthazard.md` on valiant",
        "",
        "## Schema",
        "",
        "| Column | Type | Notes |",
        "|---|---|---|",
        "| uid | Int64 | unit (spell) id, 0 .. units-1 |",
        "| t | Int8 | period 1-5; rows per t: "
        + ", ".join(f"{k}: {v:,}" for k, v in stats["per_t"].items())
        + " |",
        f"| x | Float64 | regressor; mean {stats['x_mean']:.4f}, sd {stats['x_sd']:.4f} |",
        f"| y | Int8 | event (absorbing); mean {stats['y_mean']:.4f} |",
        "| a | Float64 | the true a_i, for diagnostics only (never a regressor) |",
        "",
        "## FE / cluster levels",
        "",
        "| Variable | Distinct |",
        "|---|---:|",
        f"| uid | {stats['units']:,} |",
        "| t | 5 |",
        "",
        "## Suggested partition key and specs",
        "",
        "Partition `uid` (required by `lpm_dthazard`: a unit's rows in one shard); FE `uid` (within",
        "estimator only). HC1 SEs. Per `reference.json` → `specs`:",
        "",
        "- pooled OLS `y ~ x` (A3/A4: `y ~ x + i(t)`), all waves",
        "- within `y ~ x | uid` (A3/A4: `y ~ x + i(t) | uid`), all waves: the biased estimator",
        '- `dfeols.lpm_dthazard(ds, "y ~ x", unit="uid", time="t")` (A3/A4: `"y ~ x + i(t)"`): b_IV, and',
        "  b_FD and G from the same call",
        "",
        "Run everything and compare with the paper (|z| < 4 against each published number):",
        "",
        "    dfeols submit --profile oscer check.py -- <this folder>",
        "",
        "## Published values",
        "",
        "`reference.json` → `specs.<t1|a3|a4>.published`: (coefficient, SE) per estimator and term, typed",
        "from the paper's rendered tables; `n_iv` is the paper's FD/IV observation count at 4e7 units.",
    ]
    return "\n".join(lines) + "\n"


def main(src: Path, dst: Path, commit: str):
    info = json.loads((src / "simulate.json").read_text())
    ref = json.loads((src / "reference.json").read_text())
    lf = pl.scan_parquet(str(src / "data" / "*.parquet"))
    s = (
        lf.select(
            pl.col("x").mean().alias("x_mean"),
            pl.col("x").std().alias("x_sd"),
            pl.col("y").mean().alias("y_mean"),
            pl.col("uid").n_unique().alias("units"),
            pl.len().alias("rows"),
            *[pl.col(c).null_count().alias(f"null_{c}") for c in ("uid", "t", "x", "y", "a")],
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    if any(s[f"null_{c}"] for c in ("uid", "t", "x", "y", "a")) or s["rows"] != info["rows"]:
        raise SystemExit(f"{src}: nulls or row count mismatch: {s}")
    per_t = lf.group_by("t").len().sort("t").collect(engine="streaming")
    s["per_t"] = dict(zip(per_t["t"].to_list(), per_t["len"].to_list()))
    if (dst / "MANIFEST.sha256").exists():
        raise SystemExit(f"{dst} is already staged (never change a pushed dataset; build _v2)")
    (dst / "data").mkdir(parents=True, exist_ok=True)
    for p in info["parts"]:
        try:  # build/ and outbox/ share a file system: link, don't copy (~2 GB a dataset)
            os.link(src / p["file"], dst / p["file"])
        except OSError:
            shutil.copyfile(src / p["file"], dst / p["file"])
    for f in ("simulate.json", "reference.json"):
        shutil.copyfile(src / f, dst / f)
    shutil.copyfile(HERE / "check.py", dst / "check.py")
    (dst / "README.md").write_text(readme(dst.name, info, ref, s, commit))
    print(f"staged {dst} ({info['rows']:,} rows)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
