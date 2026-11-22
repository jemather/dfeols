# SPDX-License-Identifier: GPL-3.0-or-later
"""D14: where in-memory estimation (pyfixest, R fixest) stops working on the OpenAlex panel, and dfeols on the same fits.

Spec: the all-authors pooled KMZ fit on outbox/openalex_kmz_panel_v1, years 1970-2025, core
authors, all fields:  frac_works_n ~ top_kmz x decade (1970s-2020s) + cy + cy2 + cy3
| author_id + field_decade, CRV1 by author_id.
Rungs: authors with hash(author_id) % 16 < m, m = 1, 2, 4, 8, 16 (full: 206.9M rows, 12.7M authors).
Engines, each fit in its own process under a watchdog (peak RSS of the process tree sampled every
0.25 s; killed above --mem-cap-gb or --time-cap-min):
  pyfixest        pf.feols defaults (what a user would type), pandas input
  pyfixest_lean   lean=True, copy_data=False, store_data=False
  fixest          R fixest feols, defaults but fixef.rm = "singleton" (fixest_fit.R; $RSCRIPT, default Rscript)
  fixest_lean     also lean = TRUE, mem.clean = TRUE
  reghdfe         Stata/MP reghdfe with its defaults ($STATA, default /usr/local/stata18/stata-mp) on a
                  .dta of the rung (author_id recoded 1..K; written once per rung, not timed)
  dfeols          dfeols.prep (partition author_id) + dfeols.fit on a local Dask executor of
                  --dfeols-workers x --dfeols-cores (default $THREADS // 4 workers of 4 cores)
  dfeols_serial   the same on dfeols's default SerialExecutor (one process; the first runs, 2026-11-22)
An in-memory engine stops at its first failed rung. Results: <out>/results.json.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/openalex/pyfixest_probe.py --out build/openalex/pyfixest_probe
On OSCER: scripts/oscer_openalex_probe.sbatch (a whole normal node, cap = node memory - 24 GB).
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


def make_dta(rung: dict) -> str:
    """The rung as a Stata .dta (author_id recoded to dense 1..K, which fits Stata's long)."""
    import polars as pl

    f = Path(rung["file"]).with_suffix(".dta")
    if not f.exists():
        d = pl.read_parquet(rung["file"]).with_columns(pl.col("author_id").rank("dense").cast(pl.Int32),
                                                       pl.col("field_decade").cast(pl.Int32))
        tmp = f.with_suffix(".dta.part")
        d.to_pandas().to_stata(tmp, write_index=False, version=118)
        tmp.rename(f)
    return str(f.resolve())  # Stata runs in the fit's work folder


def reghdfe_do(dta: str) -> str:
    ts = " ".join(TS)
    return f"""set more off
timer clear
timer on 1
use "{dta}", clear
timer off 1
quietly timer list 1
local tload = r(t1)
timer on 2
capture noisily reghdfe frac_works_n {ts} cy cy2 cy3, absorb(author_id field_decade) vce(cluster author_id)
local rc = _rc
timer off 2
quietly timer list 2
local tfit = r(t2)
file open fh using "reghdfe.txt", write replace
file write fh "load" _tab "`tload'" _n "rc" _tab "`rc'" _n "fit" _tab "`tfit'" _n "processors" _tab "`c(processors)'" _n
if `rc' == 0 {{
    file write fh "N" _tab "`e(N)'" _n
    foreach v in {ts} {{
        file write fh "coef" _tab "`v'" _tab %21.0g (_b[`v']) _tab %21.0g (_se[`v']) _n
    }}
}}
file close fh
"""


def read_reghdfe(work: Path) -> dict:
    f = work / "reghdfe.txt"
    if not f.exists():
        log = (work / "fit.log").read_text(errors="replace").splitlines() if (work / "fit.log").exists() else []
        msg = [x for x in log if x.strip()][-4:]
        return {"engine": "reghdfe", "status": "error", "error": " | ".join(msg) or "no fit.log"}
    rec: dict = {"engine": "reghdfe", "coef": {}}
    for line in f.read_text().splitlines():
        k, *v = line.split("\t")
        if k == "coef":
            rec["coef"][v[0]] = {"b": float(v[1]), "se": float(v[2])}
        elif k in ("load", "fit"):
            rec[f"{k}_sec"] = round(float(v[0]), 1) if v and v[0] else None
        elif k in ("rc", "N", "processors"):
            rec[k] = int(float(v[0]))
    rec["status"] = "ok" if rec.get("rc") == 0 and len(rec["coef"]) == len(TS) else "error"
    if rec["status"] != "ok":
        rec["error"] = f"Stata rc {rec.get('rc')}; see {work / 'fit.log'}"
        rec.pop("coef")
    return rec


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
            ex = None
            if engine == "dfeols":
                nw, nc = int(os.environ["PROBE_DFEOLS_WORKERS"]), int(os.environ["PROBE_DFEOLS_CORES"])
                ex = dfeols.executor("local", cores=nc, max_workers=nw, min_workers=nw,
                                     memory=os.environ["PROBE_DFEOLS_MEMORY"])
                rec.update(workers=nw, cores_per_worker=nc)
            t1 = time.time()
            r = dfeols.fit(ds, FML, cluster="author_id", workdir=str(Path(work) / "fit"), log_every=1e9, record=False,
                           **({"executor": ex} if ex else {}))
            td = r.tidy(None, "CRV1")
            if ex is not None:
                ex.client.close()
                ex.cluster.close()
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
    cwd = None
    if engine == "reghdfe":
        (work / "fit.do").write_text(reghdfe_do(data))
        (work / "reghdfe.txt").unlink(missing_ok=True)
        cmd, cwd = [os.environ.get("STATA", "/usr/local/stata18/stata-mp"), "-b", "do", "fit.do"], work
    elif engine.startswith("fixest"):
        cmd = [os.environ.get("RSCRIPT", "Rscript"), str(Path(__file__).with_name("fixest_fit.R")), data,
               str(work / "worker.json"), "lean" if engine == "fixest_lean" else "default", "author_id", FML]
    else:
        cmd = [sys.executable, "-I", __file__, "--worker", engine, data, str(work)]
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), cwd=cwd)
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
    if engine == "reghdfe":
        rec = read_reghdfe(work)
    else:
        rec = json.loads((work / "worker.json").read_text()) if (work / "worker.json").exists() else {"engine": engine}
    if engine.startswith("fixest") and rec.get("status") == "ok":
        f = rec.pop("fits")[FML]
        rec["N"] = int(f["N"])
        rec["coef"] = {t: {"b": float(f["coef"][t]), "se": float(f["se"][t])} for t in TS}
    rec.update({"wall_sec": round(time.time() - t0, 1), "peak_rss_gb": round(peak, 2), "returncode": p.returncode,
                "mem_cap_gb": mem_cap, "host": os.uname().nodename})
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
    ap.add_argument("--engines", nargs="+", default=["dfeols", "fixest_lean", "fixest", "pyfixest_lean", "pyfixest", "reghdfe"])
    threads = int(os.environ.get("THREADS") or os.environ.get("POLARS_MAX_THREADS") or os.cpu_count())
    ap.add_argument("--dfeols-cores", type=int, default=4, help="cores per dfeols worker")
    ap.add_argument("--dfeols-workers", type=int, default=None, help="default: $THREADS // cores")
    ap.add_argument("--mem-cap-gb", type=float, default=float(os.environ.get("MEM_CAP_GB", 96)))
    ap.add_argument("--time-cap-min", type=float, default=float(os.environ.get("TIME_CAP_MIN", 180)))
    a = ap.parse_args()
    a.dfeols_workers = a.dfeols_workers or max(1, threads // a.dfeols_cores)
    os.environ.update(PROBE_DFEOLS_WORKERS=str(a.dfeols_workers), PROBE_DFEOLS_CORES=str(a.dfeols_cores),
                      PROBE_DFEOLS_MEMORY=f"{max(4, int(0.8 * a.mem_cap_gb / a.dfeols_workers))}GB")
    a.out.mkdir(parents=True, exist_ok=True)
    resf = a.out / "results.json"
    res = json.loads(resf.read_text()) if resf.exists() else {"spec": FML, "runs": []}
    res.update({"mem_cap_gb": a.mem_cap_gb, "time_cap_min": a.time_cap_min, "cpus": os.cpu_count(),
                "polars_max_threads": os.environ.get("POLARS_MAX_THREADS"),
                "dfeols_workers": a.dfeols_workers, "dfeols_cores": a.dfeols_cores})
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
                data = make_dta(rung) if e == "reghdfe" else rung["file"]
                r = {**run(e, data, a.out / "work" / f"{e}_m{m:02d}_of{a.of}", a.mem_cap_gb, a.time_cap_min, log),
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
        ref = next((k for k in ("dfeols", "dfeols_serial") if (k, m, of) in ok and k != e), None)
        if e not in ("dfeols", "dfeols_serial") and ref:
            d = ok[(ref, m, of)]["coef"]
            r["vs"] = ref
            r["max_rel_b_vs_dfeols"] = max(abs(r["coef"][t]["b"] - d[t]["b"]) / max(abs(d[t]["b"]), 1e-12) for t in TS)
            r["max_rel_se_vs_dfeols"] = max(abs(r["coef"][t]["se"] - d[t]["se"]) / d[t]["se"] for t in TS)
            r["N_minus_dfeols"] = r["N"] - ok[(ref, m, of)]["N"]
    resf.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:5])
    else:
        main()
