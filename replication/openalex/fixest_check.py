# SPDX-License-Identifier: GPL-3.0-or-later
"""D14: R fixest on every fit already run by `all_fields.py` or `kmz_full.py`, against dfeols.

Each fit's data are in <run>/work/<a>/<b>/data.parquet (all_fields: <top>/<field_id or pooled>;
kmz_full: <variant>/<sample>). The formula is rebuilt from the file: y ~ t<decade> columns + cy +
cy2 + cy3 | author_id + (field_decade if present, else decade), CRV1 by author_id, for every
outcome in the result record. fixest (fixest_fit.R, mode default) fits all outcomes of a record in
one process; rel_b_fixest / rel_se_fixest / N_fixest are added to <run>/results.json (resumable:
records that already have them are skipped).

    cd /srv/dfeols && .venv/bin/python -I repo/replication/openalex/fixest_check.py build/openalex/all_fields build/openalex/kmz_full
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import pyarrow.parquet as pq

R = Path(__file__).with_name("fixest_fit.R")


def work_dir(run: Path, rec: dict) -> Path:
    if "top" in rec:
        return run / "work" / rec["top"] / (str(rec["field_id"]) if rec["field_id"] is not None else "pooled")
    return run / "work" / rec["variant"] / rec["label"]


def check(run: Path, rec: dict, rscript: str) -> dict:
    w = work_dir(run, rec)
    cols = pq.read_schema(w / "data.parquet").names
    ts = [c for c in cols if c.startswith("t") and c[1:].isdigit()]
    fe = "author_id + " + ("field_decade" if "field_decade" in cols else "decade")
    ys = [y for y, v in rec.items() if isinstance(v, dict) and "coef" in v]
    fmls = {y: f"{y} ~ {' + '.join(ts)} + cy + cy2 + cy3 | {fe}" for y in ys}
    out = w / "fixest.json"
    t0 = time.time()
    subprocess.run([rscript, str(R), str(w / "data.parquet"), str(out), "default", "author_id", *fmls.values()],
                   check=False, stdout=subprocess.DEVNULL)
    f = json.loads(out.read_text()) if out.exists() else {"status": "no output"}
    summ = {"status": f["status"], "sec": round(time.time() - t0, 1), "peak_gb": f.get("self_maxrss_gb"),
            "error": f.get("error")}
    if f["status"] == "ok":
        for y, fml in fmls.items():
            r, k = rec[y], f["fits"][fml]
            r["N_fixest"] = int(k["N"])
            for t, e in r["coef"].items():
                b, se = float(k["coef"][t]), float(k["se"][t])
                e["rel_b_fixest"] = abs(e["b"] - b) / max(abs(b), 1e-12)
                e["rel_se_fixest"] = abs(e["se"] - se) / se
    rec["fixest"] = summ
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--rscript", default=os.environ.get("RSCRIPT", "Rscript"))
    a = ap.parse_args()
    for run in a.runs:
        resf = run / "results.json"
        res = json.loads(resf.read_text())
        for rec in res["results"]:
            if rec.get("fixest", {}).get("status") == "ok":
                continue
            s = check(run, rec, a.rscript)
            resf.write_text(json.dumps(res, indent=1))
            print(f"{time.strftime('%H:%M:%S')} {work_dir(run, rec).relative_to(run)} N={rec['N']:,} {s}", flush=True)
        ok = [r for r in res["results"] if r.get("fixest", {}).get("status") == "ok"]
        es = [e for r in ok for v in r.values() if isinstance(v, dict) and "coef" in v for e in v["coef"].values()]
        nd = sum(v["N"] != v["N_fixest"] for r in ok for v in r.values() if isinstance(v, dict) and "N_fixest" in v)
        print(f"{run}: fixest ok on {len(ok)} of {len(res['results'])} records; max rel b {max(e['rel_b_fixest'] for e in es):.1e}, "
              f"se {max(e['rel_se_fixest'] for e in es):.1e}; outcomes with N != dfeols: {nd}", flush=True)


if __name__ == "__main__":
    main()
