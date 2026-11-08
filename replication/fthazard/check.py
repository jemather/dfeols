# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the Farbmacher-Tauchmann Monte Carlo specs on one simulated dataset and compare with
the published large-sample estimates (reference.json next to the data). Nothing to edit.

    python check.py <dataset folder> [--profile oscer] [--max-workers 20] [--out summary.json]
    dfeols submit --profile oscer check.py -- /scratch/.../fthazard_st_v1

Ours and the paper's are independent draws from the same DGP, so each published number is
checked by z = (ours - paper) / sqrt(se_ours^2 + se_paper^2 + 2 r^2), r = rounding / sqrt(3)
(the paper prints 4 decimals); pass if |z| < 4. The IV/FD observation count is compared with
the paper's, scaled to our number of units. A smaller dataset (fewer units) checks the same
things with wider intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl

import dfeols

Z_MAX = 4.0


def _terms(res) -> dict:
    se = np.sqrt(np.diag(res.vcov()))
    return {n: (float(b), float(s)) for n, b, s in zip(res.coefnames, res.coef(), se)}


def compare(ours: dict, pub: dict, rounding: float) -> list[dict]:
    r2 = 2 * (rounding / math.sqrt(3)) ** 2
    rows = []
    for term, (pb, ps) in pub.items():
        if term not in ours:
            rows.append({"term": term, "published": pb, "ok": False, "note": "term missing"})
            continue
        b, s = ours[term]
        z = (b - pb) / math.sqrt(s * s + ps * ps + r2)
        rows.append(
            {
                "term": term,
                "ours": b,
                "se": s,
                "published": pb,
                "published_se": ps,
                "z": z,
                "ok": abs(z) < Z_MAX,
            }
        )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--target-rows", type=int, default=2_000_000)
    ap.add_argument("--shards", type=Path, default=None, help="where to prep (default: profile scratch)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    ref = json.loads((a.root / "reference.json").read_text())
    info = json.loads((a.root / "simulate.json").read_text())
    name = a.root.name
    if a.profile:
        prof = dfeols.profile(a.profile, **({"max_workers": a.max_workers} if a.max_workers else {}))
        ex = dfeols.executor(prof)
        shard_dir = a.shards or prof.scratch_dir("shards", name)
        run_dir = lambda s: prof.scratch_dir("runs", f"{name}_{s}")
    else:
        prof, ex = None, dfeols.SerialExecutor()
        shard_dir = a.shards or (a.root.parent / f"_shards_{name}")
        run_dir = lambda s: shard_dir.parent / f"_runs_{name}" / s

    t0 = time.time()
    p = ref["prep"]
    if (Path(shard_dir) / "dfeols_meta.json").exists():
        ds = dfeols.Dataset.open(shard_dir)
    else:
        ds = dfeols.prep(
            str(a.root / "data" / "*.parquet"),
            shard_dir=shard_dir,
            executor=ex,
            partition=p["partition"],
            fixed_effects=p["fixed_effects"],
            clusters=p["clusters"],
            target_rows=a.target_rows,
        )
    print(ds, flush=True)
    units = info["units"]
    scale = units / ref["units_published"]
    per_unit = (
        ds.scan(["uid"])
        .group_by("uid")
        .len()
        .select((pl.col("len") - 1).var())
        .collect(engine="streaming")
        .item()
    )

    summary = {"dataset": name, "units": units, "rows": ds.n_rows, "dfeols": dfeols.__version__, "specs": {}}
    ok_all = True
    for sname, s in ref["specs"].items():
        out = {"table": s["table"]}
        pub = s["published"]
        r = dfeols.fit(
            ds,
            s["ols"]["formula"],
            vcov=s["ols"]["vcov"],
            executor=ex,
            workdir=str(run_dir(f"{sname}_ols")),
            label=f"{sname}:ols",
        )
        out["ols"] = compare(_terms(r), pub["ols"], ref["rounding"])
        r = dfeols.fit(
            ds,
            s["wi"]["formula"],
            vcov=s["wi"]["vcov"],
            executor=ex,
            workdir=str(run_dir(f"{sname}_wi")),
            label=f"{sname}:wi",
        )
        out["wi"] = compare(_terms(r), pub["wi"], ref["rounding"])
        hz = s["hazard"]
        h = dfeols.lpm_dthazard(
            ds,
            hz["formula"],
            unit=hz["unit"],
            time=hz["time"],
            diff=hz["diff"],
            vcov=hz["vcov"],
            executor=ex,
            workdir=str(run_dir(f"{sname}_hazard")),
            label=f"{sname}:hazard",
        )
        out["fd"] = compare(_terms(h.fd), pub["fd"], ref["rounding"])
        out["iv"] = compare(_terms(h.iv), pub["iv"], ref["rounding"])
        n_exp = pub["n_iv"] * scale
        n_z = (h.sample["rows_built"] - n_exp) / math.sqrt(units * per_unit)
        out["n_iv"] = {
            "ours": h.sample["rows_built"],
            "fit_N": h.iv.N,
            "published_scaled": n_exp,
            "z": n_z,
            "ok": abs(n_z) < Z_MAX and h.iv.N == h.sample["rows_built"],
        }
        out["G"] = h.G
        out["sample"] = h.sample
        out["warnings"] = h.warnings
        summary["specs"][sname] = out
        print(f"\n{s['table']} ({name}, {units:,} units)")
        print(f"  {'est':<4}{'term':<11}{'ours':>10}{'se':>9}{'paper':>10}{'se':>8}{'z':>7}")
        for est in ("ols", "wi", "fd", "iv"):
            for row in out[est]:
                ok_all &= row["ok"]
                nan = float("nan")
                g = {k: row.get(k, nan) for k in ("ours", "se", "published_se", "z")}
                print(
                    f"  {est:<4}{row['term']:<11}{g['ours']:>10.4f}{g['se']:>9.4f}"
                    f"{row['published']:>10.4f}{g['published_se']:>8.4f}{g['z']:>7.2f}"
                    f"{'' if row['ok'] else '  FAIL'}"
                )
        ok_all &= out["n_iv"]["ok"]
        print(
            f"  IV/FD rows {out['n_iv']['ours']:,} vs paper scaled {n_exp:,.0f} (z {n_z:.2f})"
            f"{'' if out['n_iv']['ok'] else '  FAIL'}"
        )
    summary["ok"] = bool(ok_all)
    summary["seconds"] = round(time.time() - t0, 1)
    dest = a.out or Path(run_dir("summary")).with_suffix(".json")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text(json.dumps(summary, indent=1, default=float))
    print(f"\n{'ALL PASS' if ok_all else 'SOME CHECKS FAILED'} in {summary['seconds']:.0f}s; summary {dest}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
