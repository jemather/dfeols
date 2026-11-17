# SPDX-License-Identifier: GPL-3.0-or-later
"""D10 G reference check on R0 (391 genes x all 19,616 samples): pyfixest in memory against dfeols.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/gtex/reference.py \
        --tier-dir build/gtex/R0 --work tmp/gtex/R0 [--executor dask:8]

Specs (docs/gtex.md, Ted 2026-11-17): log_tpm ~ male:y_gene + male:xist + rin:log_gene_length
on the FE ladder g1-g3, CRV1 by donor and hetero. dfeols runs on shards partitioned by sample
(the plan) and by tissue (an invariance check: there sample and gene^tissue nest in the shard).
Writes <tier-dir>/reference.json and <tier-dir>/reference.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common

Y, X = "log_tpm", "male:y_gene + male:xist + rin:log_gene_length"
PARTS = {"sample": "sample", "tissue": "tissue"}
FES = ("sample", "gene", "gene^tissue", "gene^center")
LADDER = {
    "g1": "sample + gene",
    "g2": "sample + gene^tissue",
    "g3": "sample + gene^tissue + gene^center",
}
SPECS = [
    common.Spec(k, f"{Y} ~ {X} | {fe}", tuple(PARTS), ("CRV1", "hetero"), "donor") for k, fe in LADDER.items()
]
COLS = [Y, "male", "y_gene", "xist", "rin", "log_gene_length", "sample", "gene", "tissue", "center", "donor"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--executor", default="serial")
    ap.add_argument("--target-rows", type=int, default=1_000_000)
    ap.add_argument("--specs", default=",".join(LADDER))
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--skip-pyfixest", action="store_true")
    ap.add_argument(
        "--level-space-gb", type=float, default=0.0, help="0 (default) solves crossing FEs by CG, as in D11"
    )
    ap.add_argument("--iter-tol", type=float, default=1e-10, help="crossing-FE CG tolerance (D11: 1e-10)")
    a = ap.parse_args(argv)
    fit_kw = {"level_space_gb": a.level_space_gb, "iter_tol": a.iter_tol}
    specs = [s for s in SPECS if s.label in a.specs.split(",")]
    src = a.tier_dir / "data"
    a.work.mkdir(parents=True, exist_ok=True)

    # each finished fit is saved here, so a rerun after a kill skips it
    cache = a.work / "results.json"
    done = json.loads(cache.read_text()) if cache.exists() else {"dfeols": {}, "pyfixest": {}}
    ours, pyf = done["dfeols"], done["pyfixest"]

    def save():
        cache.write_text(json.dumps(done, indent=1, default=str))

    ex, close = common.make_executor(a.executor)
    try:
        for name, key in PARTS.items():
            if name not in a.parts.split(","):
                continue
            todo = [s for s in specs if s.run_label(name) not in ours]
            if not todo:
                continue
            dspec = common.DatasetSpec(f"gtex_{name}", key, FES, ("donor",))
            t0 = time.time()
            ds = common.prep_dataset(src, a.work / f"shards_{name}", dspec, a.target_rows)
            print(f"prep by {name}: {time.time() - t0:.0f} s", flush=True)
            for s in todo:
                r = common.fit_dfeols(ds, s, name, executor=ex, workdir=a.work / f"fit_{name}", **fit_kw)
                ours[r["label"]] = r
                save()
                print(
                    f"dfeols {r['label']}: N {r['N']:,}, {r['seconds']:.0f} s, fe levels {r['fe_levels']}",
                    flush=True,
                )
    finally:
        close()
    todo = [s for s in specs if s.label not in pyf]
    if todo and not a.skip_pyfixest:
        import polars as pl

        data = pl.read_parquet(src / "*.parquet", columns=COLS).to_pandas()
        for s in todo:
            print(f"pyfixest {s.label}: start", flush=True)
            r = common.fit_pyfixest(data, s)
            pyf[s.label] = r
            save()
            print(f"pyfixest {s.label}: N {r['N']:,}, {r['seconds']:.0f} s", flush=True)
    table = common.compare(specs, ours, pyf, {}, {})
    cols = [
        c
        for c in [
            "run",
            "vcov",
            "term",
            "b",
            "se",
            "N",
            "pyf_b_rel",
            "pyf_se_rel",
            "inv_b_rel",
            "inv_se_rel",
            "status",
            "detail",
        ]
        if c in table
    ]
    print(table[cols].to_string())
    (a.tier_dir / "reference.json").write_text(
        json.dumps({"dfeols": ours, "pyfixest": pyf}, indent=1, default=str)
    )
    (a.tier_dir / "reference.md").write_text(table[cols].to_markdown(index=False) + "\n")
    bad = (table.status == "FAIL").sum()
    print(f"{len(table)} checks, {bad} FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
