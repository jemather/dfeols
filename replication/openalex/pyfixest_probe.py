# SPDX-License-Identifier: GPL-3.0-or-later
"""D14: where in-memory estimation stops working on the OpenAlex panel, and dfeols on the same fits.

Spec: the all-authors pooled KMZ fit on outbox/openalex_kmz_panel_v1, years 1970-2025, core
authors, all fields:  frac_works_n ~ top_kmz x decade (1970s-2020s) + cy + cy2 + cy3
| author_id + field_decade, CRV1 by author_id.
Rungs: authors with hash(author_id) % 16 < m, m = 1, 2, 4, 8, 16 (full: 206.9M rows, 12.7M authors).
Engines, each fit in its own process under a watchdog (peak RSS of the process tree sampled every
0.25 s; killed above --mem-cap-gb or --time-cap-min):
  pyfixest        pf.feols defaults (what a user would type), pandas input
  pyfixest_lean   lean=True, copy_data=False, store_data=False
  dfeols          dfeols.prep (partition author_id) + dfeols.fit on the local executor
A pyfixest engine stops at its first failed rung. Results: <out>/results.json.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/openalex/pyfixest_probe.py --out build/openalex/pyfixest_probe
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

DECS = (1970, 1980, 1990, 2000, 2010, 2020)
TS = [f"t{d}" for d in DECS]
FML = f"frac_works_n ~ {' + '.join(TS)} + cy + cy2 + cy3 | author_id + field_decade"


def make_rung(pkg: Path, out: Path, m: int, of: int = 16) -> dict:
    import polars as pl

    c = pl.col
    f = out / "data" / f"m{m:02d}_of{of}.parquet"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        (pl.scan_parquet(pkg / "data" / "*.parquet")
         .filter(c("year").is_between(1970, 2025) & c("field_id").is_not_null() & c("core")
                 & (c("author_id").hash(20261122) % of < m))
         .select("author_id", "field_decade", "frac_works_n",
                 *[(c("top_kmz") & (c("decade") == d)).cast(pl.Float64).alias(f"t{d}") for d in DECS], "cy", "cy2", "cy3")
         .sink_parquet(f, row_group_size=1_000_000))
    s = pl.scan_parquet(f).select(pl.len().alias("rows"), c("author_id").n_unique().alias("authors")).collect()
    return {"m": m, "of": of, "file": str(f), "bytes": f.stat().st_size, **s.to_dicts()[0]}


def worker(engine: str, data: str, work: str) -> None:
    rec: dict = {"engine": engine}
    t0 = time.time()
    try:
        if engine.startswith("pyfixest"):
            import polars as pl
            import pyfixest as pf

            df = pl.read_parquet(data).to_pandas()
            rec["load_sec"] = round(time.time() - t0, 1)
            kw = dict(lean=True, copy_data=False, store_data=False) if engine == "pyfixest_lean" else {}
            t1 = time.time()
            fit = pf.feols(FML, data=df, vcov={"CRV1": "author_id"}, **kw)
            td = fit.tidy()
            rec["fit_sec"] = round(time.time() - t1, 1)
            rec["N"] = int(fit._N)
            rec["coef"] = {t: {"b": float(td.loc[t, "Estimate"]), "se": float(td.loc[t, "Std. Error"])} for t in TS}
        else:
            import dfeols

            ds = dfeols.prep(data, shard_dir=Path(work) / "shards", partition="author_id",
                             fixed_effects=["author_id", "field_decade"], clusters=["author_id"],
                             target_rows=2_000_000, overwrite=True)
            rec["load_sec"] = round(time.time() - t0, 1)
            t1 = time.time()
            r = dfeols.fit(ds, FML, cluster="author_id", workdir=str(Path(work) / "fit"), log_every=1e9, record=False)
            td = r.tidy(None, "CRV1")
            rec["fit_sec"] = round(time.time() - t1, 1)
            rec["N"] = int(r.N)
            rec["coef"] = {t: {"b": float(td.loc[t, "Estimate"]), "se": float(td.loc[t, "Std. Error"])} for t in TS}
        rec["status"] = "ok"
    except BaseException as e:  # MemoryError and friends are the result here
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"[:2000]
        rec["traceback"] = traceback.format_exc()[-4000:]
    rec["sec"] = round(time.time() - t0, 1)
    rec["self_maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    Path(work, "worker.json").write_text(json.dumps(rec, indent=1))


def run(engine: str, data: str, work: Path, mem_cap: float, time_cap: float, log) -> dict:
    import psutil

    work.mkdir(parents=True, exist_ok=True)
    (work / "worker.json").unlink(missing_ok=True)
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-I", __file__, "--worker", engine, data, str(work)],
                         stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    pp = psutil.Process(p.pid)
    peak, killed = 0.0, None
    while p.poll() is None:
        try:
            rss = sum(x.memory_info().rss for x in [pp, *pp.children(recursive=True)]) / 2**30
        except psutil.Error:
            rss = 0.0
        peak = max(peak, rss)
        if rss > mem_cap:
            killed = f"memory cap {mem_cap:g} GB"
        elif time.time() - t0 > time_cap * 60:
            killed = f"time cap {time_cap:g} min"
        if killed:
            for x in [*pp.children(recursive=True), pp]:
                try:
                    x.kill()
                except psutil.Error:
                    pass
            p.wait()
            break
        time.sleep(0.25)
    rec = json.loads((work / "worker.json").read_text()) if (work / "worker.json").exists() else {"engine": engine}
    rec.update({"wall_sec": round(time.time() - t0, 1), "peak_rss_gb": round(peak, 2), "returncode": p.returncode})
    if killed:
        rec["status"] = "killed"
        rec["killed"] = killed
    elif "status" not in rec:
        rec["status"] = f"died (return code {p.returncode})"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, default=Path("outbox/openalex_kmz_panel_v1"))
    ap.add_argument("--out", type=Path, default=Path("build/openalex/pyfixest_probe"))
    ap.add_argument("--rungs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--of", type=int, default=16, help="rung m keeps hash %% of < m (testing: large of)")
    ap.add_argument("--engines", nargs="+", default=["dfeols", "pyfixest_lean", "pyfixest"])
    ap.add_argument("--mem-cap-gb", type=float, default=float(os.environ.get("MEM_CAP_GB", 96)))
    ap.add_argument("--time-cap-min", type=float, default=float(os.environ.get("TIME_CAP_MIN", 180)))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    resf = a.out / "results.json"
    res = json.loads(resf.read_text()) if resf.exists() else {"spec": FML, "runs": []}
    res.update({"mem_cap_gb": a.mem_cap_gb, "time_cap_min": a.time_cap_min, "cpus": os.cpu_count(),
                "polars_max_threads": os.environ.get("POLARS_MAX_THREADS")})
    done = {(r["engine"], r["m"]) for r in res["runs"] if r["status"] == "ok" and r["of"] == a.of}
    failed = {r["engine"] for r in res["runs"] if r["status"] != "ok" and r["of"] == a.of}
    with open(a.out / "workers.log", "a") as log:
        for m in a.rungs:
            rung = make_rung(a.package, a.out, m, a.of)
            print(f"{time.strftime('%H:%M:%S')} rung {rung}", flush=True)
            for e in a.engines:
                if (e, m) in done or e in failed:
                    continue
                log.write(f"\n===== {time.strftime('%H:%M:%S')} {e} m={m} =====\n")
                log.flush()
                r = {**run(e, rung["file"], a.out / "work" / f"{e}_m{m:02d}_of{a.of}", a.mem_cap_gb, a.time_cap_min, log),
                     "m": m, "of": a.of, "rows": rung["rows"], "authors": rung["authors"]}
                r.pop("traceback", None) if r["status"] == "ok" else None
                res["runs"].append(r)
                if r["status"] != "ok":
                    failed.add(e)
                resf.write_text(json.dumps(res, indent=1))
                print(f"{time.strftime('%H:%M:%S')} {e} m={m} rows={rung['rows']:,} {r['status']} {r.get('killed', r.get('error', ''))[:200]} "
                      f"wall {r['wall_sec']}s peak {r['peak_rss_gb']} GB", flush=True)
    # agreement with dfeols on each rung
    ok = {(r["engine"], r["m"], r["of"]): r for r in res["runs"] if r["status"] == "ok"}
    for (e, m, of), r in ok.items():
        if e != "dfeols" and ("dfeols", m, of) in ok:
            d = ok[("dfeols", m, of)]["coef"]
            r["max_rel_b_vs_dfeols"] = max(abs(r["coef"][t]["b"] - d[t]["b"]) / max(abs(d[t]["b"]), 1e-12) for t in TS)
            r["max_rel_se_vs_dfeols"] = max(abs(r["coef"][t]["se"] - d[t]["se"]) / d[t]["se"] for t in TS)
    resf.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:5])
    else:
        main()
