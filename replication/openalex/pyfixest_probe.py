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

estrec records (Ted 2026-11-22: "exactly the kind of thing we'd want to have records of"): every fit,
failed ones included, leaves <work>/record.estrec.json (record_r<k>.estrec.json for repeat k), and the run ends with
<out>/openalex_probe.estrec-collection.json (kind "batch", one member per fit, labelled
"<engine> m/of [repeat k]"). A finished fit's record is the engine's own, at disclosure level 5
(execution and host kept): dfeols fit(record=), pyfixest through estrec's converter
(converters/pyfixest/estimation_pyfixest.py: next to this file, in $ESTREC_PYFIXEST, or in the estrec
clone beside the dfeols repo), fixest through estrec R's write_estimation(), reghdfe through
`estrec write`. Where the R or Stata writer is missing the fit still runs and the record is the
probe's own, from the numbers the engine reported (status "warning", with a message saying so).
A killed or failed fit gets a status "failed" record written by the probe. Every record gets an
`extensions.openalex_probe` block: the watchdog's wall time and peak RSS of the process tree, the
engine's load and fit seconds, the rung, the caps and the host.

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
if `rc' == 0 {{
    capture which estrec
    if _rc == 0 {{
        capture noisily estrec write using "engine.estrec.json", replace level(5)
        local erc = _rc
    }}
    else local erc = "no estrec"
    file open fh using "reghdfe.txt", write append
    file write fh "estrec" _tab "`erc'" _n
    file close fh
}}
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
        elif k == "estrec" and v and v[0] != "0":
            rec["estrec_error"] = f"estrec write: {v[0]}"
    rec["status"] = "ok" if rec.get("rc") == 0 and len(rec["coef"]) == len(TS) else "error"
    if rec["status"] != "ok":
        rec["error"] = f"Stata rc {rec.get('rc')}; see {work / 'fit.log'}"
        rec.pop("coef")
    return rec


RECORD = "engine.estrec.json"  # what the engine writes; the probe files it as record[_r<repeat>].estrec.json
PROBE_VERSION = "2026-11-22"


def pyfixest_converter():
    """estrec's pyfixest writer, `to_record` (a converter in the estrec repo, not in the package)."""
    here = Path(__file__).resolve().parent
    for d in (os.environ.get("ESTREC_PYFIXEST"), here, here.parents[2] / "estimation" / "converters" / "pyfixest",
              here.parents[3] / "estimation" / "converters" / "pyfixest"):
        if d and (Path(d) / "estimation_pyfixest.py").exists():
            sys.path.insert(0, str(d))
            break
    from estimation_pyfixest import to_record

    return to_record


ENGINE = {  # engine -> (software, call as run)
    "dfeols": ("dfeols", f'dfeols.fit(ds, "{FML}", cluster="author_id", executor=<local Dask>)'),
    "dfeols_serial": ("dfeols", f'dfeols.fit(ds, "{FML}", cluster="author_id")'),
    "pyfixest": ("pyfixest", f'pf.feols("{FML}", data=df, vcov={{"CRV1": "author_id"}})'),
    "pyfixest_lean": ("pyfixest", f'pf.feols("{FML}", data=df, vcov={{"CRV1": "author_id"}}, lean=True, copy_data=False, store_data=False)'),
    "fixest": ("fixest", f'feols({FML}, d, cluster = ~author_id, fixef.rm = "singleton")'),
    "fixest_lean": ("fixest", f'feols({FML}, d, cluster = ~author_id, fixef.rm = "singleton", lean = TRUE, mem.clean = TRUE)'),
    "reghdfe": ("reghdfe", f"reghdfe frac_works_n {' '.join(TS)} cy cy2 cy3, absorb(author_id field_decade) vce(cluster author_id)"),
}


def probe_block(r: dict, rung: dict, a) -> dict:
    """extensions.openalex_probe: what the probe measured around the engine."""
    g = lambda k: r.get(k)  # noqa: E731
    b = {"schema": "openalex-probe/0.1", "by": {"name": "openalex pyfixest_probe.py", "version": PROBE_VERSION}, "added": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "engine": r["engine"], "status": r["status"], "repeat": r.get("repeat", 0),
         "rung": {k: rung[k] for k in ("m", "of", "rows", "authors", "bytes")},
         "wall_seconds": g("wall_sec"), "peak_rss_bytes": int(r["peak_rss_gb"] * 2**30),
         "peak_rss_sampling": "RSS of the process tree every 0.25 s",
         "load_seconds": g("load_sec"), "fit_seconds": g("fit_sec"), "self_maxrss_gb": g("self_maxrss_gb"),
         "mem_cap_bytes": int(a.mem_cap_gb * 2**30), "time_cap_minutes": a.time_cap_min, "host": g("host"),
         "threads": int(os.environ.get("THREADS") or 0) or None}
    for k in ("killed", "error", "estrec_error", "workers", "cores_per_worker", "processors", "nthreads"):
        if r.get(k) is not None:
            b[k] = r[k]
    return b


def write_record(work: Path, r: dict, rung: dict, a) -> str:
    """The fit's record at <work>/record.estrec.json: the engine's, else the probe's own; plus the probe block."""
    import estrec as ef

    eng, f = work / RECORD, work / ("record.estrec.json" if not r.get("repeat") else f"record_r{r['repeat']}.estrec.json")
    soft, call = ENGINE[r["engine"]]
    ext = probe_block(r, rung, a)
    if r["status"] == "ok" and eng.exists() and eng.stat().st_mtime >= r["_started"]:
        rec = json.loads(eng.read_text())  # as written (non-finite numbers stay encoded)
        rec.setdefault("extensions", {})["openalex_probe"] = ext
        rec.setdefault("history", []).append({"when": ext["added"], "by": ext["by"], "action": "extended",
                                              "sections": ["extensions"]})
        f.write_text(json.dumps(rec, indent=1) + "\n")  # extensions and history are outside the core digest
        eng.unlink()
        return str(f)
    eng.unlink(missing_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["_started"]))
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["_started"] + r["wall_sec"]))
    spec = {"call": call, "formula": {"text": FML, "syntax": "stata" if soft == "reghdfe" else "fixest"},
            "fixed_effects": ["author_id", "field_decade"], "clusters": ["author_id"],
            "sample": {"restriction": f"rung m={rung['m']} of {rung['of']}: authors with hash(author_id) % {rung['of']} < {rung['m']}",
                       "rows_input": rung["rows"]}}
    prov = {"software": [{"name": soft}],
            "data": [{"name": Path(rung["file"]).name, "rows": rung["rows"]}],
            "execution": {"started": started, "finished": finished, "elapsed_seconds": r["wall_sec"],
                          "peak_memory_bytes": ext["peak_rss_bytes"]},
            "environment": {"host": r.get("host")}}
    if r["status"] == "ok":  # the engine's writer is missing or failed: the numbers it reported
        params = [{"term": t, "estimate": c["b"], "std_error": c["se"]} for t, c in r["coef"].items()]
        meta = {"status": "warning", "messages": [{"level": "warning", "code": "probe_record",
                "message": f"written by the probe from the estimates {soft} reported (b and se only; no vcov): "
                           f"{r.get('estrec_error') or 'no estrec writer for ' + soft}", "source": "openalex probe"}]}
        extra = {"metrics": {"nobs": r["N"]}}
    else:
        params = []
        why = r.get("killed") or r.get("error") or r["status"]
        meta = {"status": "failed", "messages": [{"level": "error", "code": "killed" if r.get("killed") else "failed",
                "message": f"{soft} did not finish: {why}"[:2000], "source": "openalex probe"}]}
        extra = {}
    rec = ef.new_record(params, estimator={"family": "linear", "method": "ols"}, responses=["frac_works_n"],
                        meta=meta, specification=spec, provenance=prov, **extra)
    rec["extensions"] = {"openalex_probe": ext}
    ef.write(rec, f, level=5, by=ext["by"])
    return str(f)


def write_collection(out: Path, res: dict) -> None:
    import estrec as ef

    recs, labels = [], []
    for r in res["runs"]:
        f = Path(r.get("record") or "")
        if r.get("record") and f.exists():
            recs.append(ef.read(f))
            labels.append(f"{r['engine']} {r['m']}/{r['of']}" + (f" repeat {r['repeat']}" if r.get("repeat") else ""))
    if recs:
        coll = ef.new_collection(recs, kind="batch", labels=labels,
                                 title="OpenAlex KMZ all-authors pooled fit: where in-memory estimation stops",
                                 notes=f"{FML}, CRV1 by author_id; rungs of core authors; caps {res['mem_cap_gb']:g} GB / {res['time_cap_min']:g} min")
        ef.write_collection(coll, out / "openalex_probe.estrec-collection.json")


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
            try:
                import estrec as ef

                to_record = pyfixest_converter()
                kws = ", lean=True, copy_data=False, store_data=False" if kw else ""
                ef.write(to_record(fit, df, call=f'pf.feols("{FML}", data=df, vcov={{"CRV1": "author_id"}}{kws})'),
                         Path(work, RECORD), level=5)
            except Exception as e:  # the fit stands; the probe writes the record from rec
                rec["estrec_error"] = f"{type(e).__name__}: {e}"[:500]
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
            r = dfeols.fit(ds, FML, cluster="author_id", workdir=str(Path(work) / "fit"), log_every=1e9,
                           record=str(Path(work, RECORD)), record_level=5, **({"executor": ex} if ex else {}))
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
        os.environ["FIXEST_RECORD"] = str(work / RECORD)
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
    if engine.startswith("fixest") and rec.get("estrec_error"):
        rec["estrec_error"] = str(rec["estrec_error"])[:500]
    rec.update({"_started": t0, "wall_sec": round(time.time() - t0, 1), "peak_rss_gb": round(peak, 2), "returncode": p.returncode,
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
    ap.add_argument("--force-engines", nargs="*", default=[],
                    help="run these again even where they finished (repeat timings; every run is kept, numbered)")
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
    # a forced engine runs again from the first rung and stops at its first failure in this run
    failed = {r["engine"] for r in res["runs"] if r["status"] != "ok" and r["of"] == a.of} - set(a.force_engines)
    with open(a.out / "workers.log", "a") as log:
        for m in a.rungs:
            rung = make_rung(a.package, a.out, m, a.of)
            print(f"{time.strftime('%H:%M:%S')} rung {rung}", flush=True)
            for e in a.engines:
                if ((e, m) in done and e not in a.force_engines) or e in failed:
                    continue
                log.write(f"\n===== {time.strftime('%H:%M:%S')} {e} m={m} =====\n")
                log.flush()
                data = make_dta(rung) if e == "reghdfe" else rung["file"]
                r = {**run(e, data, a.out / "work" / f"{e}_m{m:02d}_of{a.of}", a.mem_cap_gb, a.time_cap_min, log),
                     "m": m, "of": a.of, "rows": rung["rows"], "authors": rung["authors"]}
                r.pop("traceback", None) if r["status"] == "ok" else None
                r["repeat"] = sum(1 for x in res["runs"] if x["engine"] == e and x["m"] == m and x["of"] == a.of)
                try:
                    r["record"] = write_record(a.out / "work" / f"{e}_m{m:02d}_of{a.of}", r, rung, a)
                except Exception as ex:  # a record problem never stops the probe
                    r["record_error"] = f"{type(ex).__name__}: {ex}"[:500]
                r.pop("_started")
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
    try:
        write_collection(a.out, res)
    except Exception as ex:
        print(f"estrec collection not written: {type(ex).__name__}: {ex}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:5])
    else:
        main()
