# SPDX-License-Identifier: GPL-3.0-or-later
"""D13: dfeols on the BYZ product-level R0 sample, against fixest (byz_ref.R), -> reference.json.

    p1: fit_glm("v ~ rta | i^k^t + j^k^t + i^j^k", family="poisson", cluster="pair"), all cells
    o1: fit("lv ~ rta | i^k^t + j^k^t + i^j^k", cluster="pair"), positive flows only

Partition k: every FE is nested in the product, so all three are shard-local and only the rta
coefficient (and the pair clusters) couple shards. Writes <out> (reference.json: fixest and dfeols
N, b, se per spec, relative differences) for the package's ref/.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/baci/byz_check.py \
        --data outbox/baci_byz_v1/ref/data/part-0000.parquet --fixest build/baci/byz/ref_fixest.json \
        --work build/baci/byz/check --out outbox/baci_byz_v1/ref/reference.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import dfeols
import polars as pl

FE = "i^k^t + j^k^t + i^j^k"
FES = ["i^k^t", "j^k^t", "i^j^k"]


def prep(src: Path, d: Path):
    return dfeols.prep(str(src), shard_dir=d, partition="k", fixed_effects=FES, clusters=["pair"],
                       target_rows=2_000_000, overwrite=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--fixest", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=["o1", "p1"])
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    pos = a.work / "positive.parquet"
    pl.scan_parquet(a.data).filter(pl.col("pos") == 1).sink_parquet(pos)
    fx = json.loads(a.fixest.read_text()) if a.fixest.exists() else {}
    out = {"data": str(a.data), "fixest": {k: fx.get(k) for k in ("p1", "o1")},
           "fixest_version": fx.get("fixest"), "dfeols_version": dfeols.__version__, "dfeols": {}}
    for spec in a.only:
        t0 = time.time()
        try:
            if spec == "o1":
                ds = prep(pos, a.work / "shards_o1")
                r = dfeols.fit(ds, f"lv ~ rta | {FE}", cluster="pair")
                formula = f"lv ~ rta | {FE} (pos == 1)"
            else:
                ds = prep(a.data, a.work / "shards_p1")
                r = dfeols.fit_glm(ds, f"v ~ rta | {FE}", family="poisson", cluster="pair", tol=1e-10,
                                   fe_tol=1e-12, workdir=str(a.work / "fit_p1"), log_every=1e9)
                formula = f"v ~ rta | {FE} (poisson)"
            t = r.tidy(None, "CRV1")
            rec = {"formula": formula, "N": int(r.N), "b": float(t.loc["rta", "Estimate"]),
                   "se_pair": float(t.loc["rta", "Std. Error"]), "sec": round(time.time() - t0, 1)}
        except Exception as e:
            rec = {"error": f"{type(e).__name__}: {e}", "sec": round(time.time() - t0, 1)}
        f = fx.get(spec)
        if f and "b" in rec:
            rec["rel_b"] = abs(rec["b"] - f["b"]) / max(abs(f["b"]), 1e-12)
            rec["rel_se"] = abs(rec["se_pair"] - f["se_pair"]) / f["se_pair"]
            rec["N_diff"] = rec["N"] - f["N"]
        out["dfeols"][spec] = rec
        print(spec, json.dumps(rec), flush=True)
    a.out.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
