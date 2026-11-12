"""D11 reference check on a small tier (R0): pyfixest in memory against dfeols, both partitions.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/cellxgene/reference.py \
        --tier-dir build/cellxgene/R0 --work tmp/cellxgene/R0 [--executor dask:4]

Specs (docs/cellxgene.md): the three regressors on the FE ladder, clustered by dataset x donor
and heteroskedasticity-robust. dfeols runs on shards partitioned by cell (the plan) and by
donor (an invariance check). Writes <tier-dir>/reference.json and <tier-dir>/reference.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

X = "nucleus:mt_gene + nucleus:lnc_gene + full_length:log_gene_length"
PARTS = {"cell": "cell", "donor": "donor"}
SPECS = [
    common.Spec("L1_gene", f"log_count ~ {X} | cell + gene", tuple(PARTS), ("CRV1", "hetero"), "donor"),
    common.Spec("L2_gene_celltype", f"log_count ~ {X} | cell + gene^cell_type", tuple(PARTS), ("CRV1", "hetero"), "donor"),
    common.Spec("L3_plus_gene_dataset", f"log_count ~ {X} | cell + gene^cell_type + gene^dataset", tuple(PARTS), ("CRV1", "hetero"), "donor"),
]
COLS = ["log_count", "cell", "gene", "cell_type", "dataset", "donor", "nucleus", "mt_gene", "lnc_gene",
        "full_length", "log_gene_length"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--executor", default="serial")
    ap.add_argument("--target-rows", type=int, default=2_000_000)
    ap.add_argument("--specs", default=",".join(s.label for s in SPECS))
    ap.add_argument("--skip-pyfixest", action="store_true")
    ap.add_argument("--level-space-gb", type=float, default=0.0,
                    help="0 (default) solves crossing FEs by CG; the level-space Gram is dense here")
    ap.add_argument("--iter-tol", type=float, default=None, help="crossing-FE CG tolerance (dfeols default 1e-8)")
    ap.add_argument("--parts", default=",".join(PARTS))
    a = ap.parse_args(argv)
    fit_kw = {"level_space_gb": a.level_space_gb}
    if a.iter_tol is not None:
        fit_kw["iter_tol"] = a.iter_tol
    specs = [s for s in SPECS if s.label in a.specs.split(",")]
    src = a.tier_dir / "data"
    a.work.mkdir(parents=True, exist_ok=True)

    ex, close = common.make_executor(a.executor)
    ours, pyf = {}, {}
    try:
        for name, key in PARTS.items():
            if name not in a.parts.split(","):
                continue
            dspec = common.DatasetSpec(f"cellxgene_{name}", key, ("cell", "gene", "gene^cell_type", "gene^dataset"), ("donor",))
            t0 = time.time()
            ds = common.prep_dataset(src, a.work / f"shards_{name}", dspec, a.target_rows)
            print(f"prep by {name}: {time.time() - t0:.0f} s", flush=True)
            for s in specs:
                r = common.fit_dfeols(ds, s, name, executor=ex, workdir=a.work / f"fit_{name}",
                                     **fit_kw)
                ours[r["label"]] = r
                print(f"dfeols {r['label']}: N {r['N']:,}, {r['seconds']:.0f} s, fe levels {r['fe_levels']}", flush=True)
    finally:
        close()
    if not a.skip_pyfixest:
        import polars as pl

        data = pl.read_parquet(f"{src}/*.parquet", columns=COLS).to_pandas()
        for s in specs:
            r = common.fit_pyfixest(data, s)
            pyf[s.label] = r
            print(f"pyfixest {s.label}: N {r['N']:,}, {r['seconds']:.0f} s", flush=True)
    table = common.compare(specs, ours, pyf, {}, {})
    cols = [c for c in ["run", "vcov", "term", "b", "se", "N", "pyf_b_rel", "pyf_se_rel", "inv_b_rel",
                        "inv_se_rel", "status", "detail"] if c in table]
    print(table[cols].to_string())
    (a.tier_dir / "reference.json").write_text(json.dumps({"dfeols": ours, "pyfixest": pyf}, indent=1, default=str))
    (a.tier_dir / "reference.md").write_text(table[cols].to_markdown(index=False) + "\n")
    bad = (table.status == "FAIL").sum()
    print(f"{len(table)} checks, {bad} FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
