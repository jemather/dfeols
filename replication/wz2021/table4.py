# SPDX-License-Identifier: GPL-3.0-or-later
"""D13: Weidner & Zylkin (2021) Table 4 PPML column with dfeols.fit_glm, against fixest.

One prep per industry (138,610 rows, partition exp_id in ~3 shards: exp x year and pair
nest in it, imp x year crosses; a separate prep per industry, because subset= on the pooled
5M rows rescans every row each IRLS iteration, ~2 min per industry), then

    fit_glm("trade ~ FTA | expyear + impyear + pair", family="poisson", cluster="pair")

compared with build/wz2021/fixest.csv (table4_fixest.R: fepois, same FEs and cluster) and
the package's published BACI_results.tex (PPML b and SE, ppmlhdfe2 tol 1e-6).

    cd /srv/dfeols && .venv/bin/python -I repo/replication/wz2021/table4.py \
        --data build/wz2021/baci_example.parquet --fixest build/wz2021/fixest.csv \
        --pub "tmp/wz_pkg/Empirical application (Table 4)/BACI_results.tex.tex" --work tmp/wz2021/dfeols \
        --out build/wz2021/table4.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import polars as pl

import dfeols

SKIPPED = {5, 6, 31, 34}  # isic_id the do-file skips (isiccode 11, 12, 40, 93)


def published(path: Path) -> dict[int, tuple[float, float]]:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^(\d+) &\s+(\S+) &\s+(\S+) &", line)
        if m:
            out[int(m[1])] = (float(m[2]), float(m[3]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--fixest", type=Path, required=True)
    ap.add_argument("--pub", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--crossing-fe", default="sparse")
    ap.add_argument("--only", type=int, nargs="*", help="isic_id values to run (default all)")
    a = ap.parse_args()
    pub = published(a.pub)
    fx = {r["isic_id"]: r for r in pl.read_csv(a.fixest).to_dicts()}
    data = pl.scan_parquet(a.data)
    rows = []
    for k in sorted(fx):
        if a.only and k not in a.only:
            continue
        kd = a.work / f"k{k}"
        kd.mkdir(parents=True, exist_ok=True)
        data.filter(pl.col("isic_id") == k).sink_parquet(kd / "data.parquet")
        ds = dfeols.prep(
            str(kd / "data.parquet"),
            shard_dir=kd / "shards",
            partition="exp_id",
            fixed_effects=["expyear", "impyear", "pair"],
            clusters=["pair"],
            target_rows=50_000,
            overwrite=True,
        )
        t0 = time.time()
        rec = {"isic_id": k, "isiccode": fx[k]["isiccode"], "skipped_in_paper": k in SKIPPED}
        try:
            r = dfeols.fit_glm(
                ds,
                "trade ~ FTA | expyear + impyear + pair",
                family="poisson",
                cluster="pair",
                tol=a.tol,
                fe_tol=1e-12,
                crossing_fe=a.crossing_fe,
                workdir=str(kd / "fit"),
                log_every=1e9,
            )
            t = r.tidy(None, "CRV1")
            rec.update(b=float(t.loc["FTA", "Estimate"]), se=float(t.loc["FTA", "Std. Error"]), N=int(r.N),
                       glm={kk: v for kk, v in r.options.get("glm", {}).items() if kk != "dead"})
        except Exception as e:  # report, keep going
            rec.update(error=f"{type(e).__name__}: {e}")
        rec["sec"] = round(time.time() - t0, 1)
        f = fx[k]
        rec["fixest"] = {"b": f["b"], "se": f["se"], "se_feK_none": f["se_feK_none"], "N": f["nobs"]}
        rec["published"] = pub.get(rec["isiccode"])
        if "b" in rec and f["b"] is not None and not (isinstance(f["b"], float) and math.isnan(f["b"])):
            rec["rel_b"] = abs(rec["b"] - f["b"]) / max(abs(f["b"]), 1e-12)
            rec["rel_se"] = abs(rec["se"] - f["se"]) / f["se"]
            # dfeols uses fixest's default small-sample factor; rescale to ppmlhdfe's (FEs not in K)
            rec["se_pub_convention"] = rec["se"] * f["se_feK_none"] / f["se"]
        rows.append(rec)
        print(json.dumps({kk: rec.get(kk) for kk in ("isic_id", "isiccode", "b", "se", "N", "rel_b", "rel_se", "sec", "error")}))
    a.out.write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
