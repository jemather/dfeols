"""Benchmark dfeols on synthetic data with known parameters.

For each size it generates data (reusing what exists), preps one shard set per
partition key the specs need, fits each spec, and records:

  timing      generate / prep / round 1 / solve / round 2 / total seconds
  throughput  rows per wall-second and per worker-second (queue-independent)
  workers     average and max workers connected during the fit
  memory      peak worker RSS
  validity    max |z| against the truth, joint Wald p, and whether that is what
              the spec should find (unbiased / biased / se_wrong)
  ETA         predicted vs actual finish at 10/25/50/75% progress
  baseline    pyfixest time and max coefficient difference (small sizes only)

Examples
  python -m dfeols.bench --rows 1e6 1e7                         # local, generic panel
  python -m dfeols.bench --rows 1e7 1e8 --profile mycluster     # elastic cluster workers
  python -m dfeols.bench --dgp influencer --specs level shape --rows 1e8 --profile mycluster
  python -m dfeols.bench --rows 1e8 --set T=520 --set n_groups=2000 --profile mycluster
  python -m dfeols.bench --preset units10m_t100 --profile mycluster     # 1e9 rows, 10M units, T=100
  python -m dfeols.bench --preset units1m_t1000 --rows 1e8 --profile mycluster   # 1e8 first
  python -m dfeols.bench --preset units100m_t100 --profile mycluster   # 1e10 stress test
  python -m dfeols.bench --preset skewed_groups --profile mycluster    # oversized shards
  python -m dfeols.bench --preset units100m_t100 --rows 2e6 --profile local \
      --prep-opt hash_levels=0 --prep-opt range_groups=0 --prep-opt max_files=0   # 1e10 code paths, small
  dfeols bench --profile mycluster --rows 1e8 1e9

Writes results.csv, results.json and summary.md to --out (default <root>/results).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import time
from pathlib import Path

import numpy as np
import polars as pl

import dfeols
from .formula import to_pyfixest
from .synth import generate, load_truth, plan_for_rows
from .validate import recovery


# Large-scale presets: billion-row unit x time panels with crossing time FEs.
# Balanced, so units = rows / T. Specs: unit + time partitioned by unit (time
# FE crosses every shard: sparse path) and unit + group^time partitioned by
# group (both FEs local: exact two-way solve).
PRESETS = {
    "units10m_t100": {
        "rows": [1e9],
        "dgp": "panel",
        "specs": ["twfe", "grouptime"],
        "set": {"balanced": True, "T": 100, "n_groups": 500, "units_per_chunk": 20_000},
    },
    "units1m_t1000": {
        "rows": [1e9],
        "dgp": "panel",
        "specs": ["twfe", "grouptime"],
        "set": {"balanced": True, "T": 1000, "n_groups": 500, "units_per_chunk": 2_000},
    },
    # Stress tests.
    # 1e10 rows, 100M units: hash ids for the unit FE, range-based shards by unit,
    # a two-level shuffle when partitioning by group, 5,000 shards per set.
    # Needs ~1 TB scratch (raw ~370 GB + two shard sets) and a 64 GB coordinator.
    "units100m_t100": {
        "rows": [1e10],
        "dgp": "panel",
        "specs": ["twfe", "grouptime"],
        "gen_on_workers": True,
        "set": {"balanced": True, "T": 100, "n_groups": 5000, "units_per_chunk": 20_000},
    },
    # Skewed group sizes (Zipf 0.7 over 5,000 groups): the largest group holds
    # ~2.5% of rows (25M at 1e9) and becomes one oversized shard when
    # partitioning by group; tests uneven shards, worker memory and the ETA floor.
    "skewed_groups": {
        "rows": [1e9],
        "dgp": "panel",
        "specs": ["twfe", "grouptime"],
        "set": {"balanced": True, "T": 100, "n_groups": 5000, "group_zipf": 0.7, "units_per_chunk": 20_000},
    },
}


def _available_cores() -> int:
    """Cores this process may use (the job's allocation, not the whole node)."""
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except AttributeError:  # macOS
        return os.cpu_count() or 1


def eta_and_workers(workdir: Path) -> dict:
    p = workdir / "history.jsonl"
    if not p.exists():
        return {}
    h = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not h:
        return {}
    actual = h[-1]["t"]
    w = [x["workers"] for x in h]
    out = {"workers_avg": round(float(np.mean(w)), 1), "workers_max": int(max(w))}
    for q in (0.10, 0.25, 0.50, 0.75):
        pts = [x for x in h if x["frac_done"] >= q and x["eta_total"] is not None]
        if pts and actual > 0:
            x = pts[0]
            out[f"eta_err_{int(q * 100)}"] = round((x["t"] + x["eta_total"] - actual) / actual, 3)
    return out


def pyfixest_baseline(raw: Path, spec: dict, res) -> dict:
    import pyfixest as pf

    pdf = pl.read_parquet(str(raw / "*.parquet")).to_pandas()
    t = time.perf_counter()
    m = pf.feols(
        to_pyfixest(spec["formula"]),
        data=pdf,
        vcov={"CRV1": spec["cluster"]},
        fixef_tol=1e-10,
        fixef_maxiter=1_000_000,
    )
    secs = time.perf_counter() - t
    tab = m.tidy().loc[res.coefnames]
    return {
        "pyfixest_s": round(secs, 1),
        "pyfixest_max_coef_diff": float(np.max(np.abs(tab["Estimate"].to_numpy() - res.coef()))),
    }


# ---------------------------------------------------------------- layout
#
#   <root>/raw_<tag>/                     generated data (reused across runs)
#   <root>/shards_<tag>_by_<key>_<n>/     prepared shards (reused across runs)
#   <root>/runs/<run_id>/<tag>/<spec>/    one fit: result.json, bench.json,
#                                         status.json, history.jsonl  (never overwritten)
#   <root>/results/results.csv            every fit ever run here, rebuilt from runs/
#   <root>/results/<run_id>.md            summary of one benchmark invocation
#   <root>/results/latest.md              copy of the most recent summary
#
# `dfeols results [root]` prints the table; nothing is overwritten, so earlier
# results stay where they were written.


def _param_suffix(dgp: str, kw: dict | None) -> str:
    """Data folders are named by DGP, size and any non-default parameters, so
    different configurations never collide (e.g. T=100 vs T=1000 at 1e9)."""
    if not kw:
        return ""
    keep = {k: v for k, v in kw.items() if k not in ("units_per_chunk", "accounts_per_chunk")}
    if not keep:
        return ""
    short = {"balanced": "bal", "n_groups": "G", "n_units": "U"}
    parts = []
    for k in sorted(keep):
        v = keep[k]
        v = ("1" if v else "0") if isinstance(v, bool) else str(v).replace(" ", "")
        parts.append(f"{short.get(k, k)}{v}")
    return "_" + "_".join(parts)


def make_run_id() -> str:
    job = next(
        (os.environ[v] for v in ("SLURM_JOB_ID", "PBS_JOBID", "JOB_ID", "LSB_JOBID") if os.environ.get(v)),
        None,
    )
    rid = time.strftime("%Y%m%d-%H%M%S")
    return f"{rid}-{job.split('.')[0]}" if job else rid


def _verdict(expect: str, r: dict) -> str:
    if expect == "unbiased":
        return "ok" if (r["ok"] and r["p"] > 0.001) else "FAIL"
    if expect == "biased":
        return "ok" if r["p"] < 1e-4 else "FAIL"
    return "n/a"


def _row_from_result(res, truth, spec_name, *, n_rows=None, n_shards=None, wd=None) -> dict:
    s = truth["specs"][spec_name]
    tm = res.timing
    r = recovery(res, truth)
    return {
        "dgp": truth.get("kind", "?"),
        "rows": n_rows or res.N + res.N_dropped,
        "N": res.N,
        "spec": spec_name,
        "expect": s["expect"],
        "partition": s["partition"],
        "shards": n_shards or tm.get("n_shards"),
        "clusters": res.G,
        "round1_s": round(tm["round1_seconds"], 1),
        "round2_s": round(tm["round2_seconds"], 1),
        "fit_s": round(res.seconds, 1),
        "rows_per_s": int(res.N / max(res.seconds, 1e-9)),
        "rows_per_worker_s": int(res.N / max(tm["worker_seconds"], 1e-9)),
        "peak_worker_rss_gb": round(tm["max_worker_rss_gb"], 2),
        "max_abs_z": round(r["max_abs_z"], 2),
        "wald_p": round(r["p"], 4),
        "validity": _verdict(s["expect"], r),
        **(eta_and_workers(Path(wd)) if wd else {}),
    }


def collect(root) -> list[dict]:
    """Every benchmark fit under <root>/runs, newest first. Rows come from each
    fit's bench.json; fits from before bench.json existed (runs/<tag>/<spec>/)
    are rebuilt from result.json and the dataset's truth.json."""
    from .results import Result

    root = Path(root)
    rows = []
    for bj in root.glob("runs/*/*/*/bench.json"):
        try:
            rows.append(json.loads(bj.read_text()))
        except Exception:  # noqa: BLE001
            pass
    for rj in root.glob("runs/*/*/result.json"):  # legacy layout
        tag, spec = rj.parent.parent.name, rj.parent.name
        tp = root / f"raw_{tag}" / "truth.json"
        if not tp.exists():
            continue
        try:
            truth = json.loads(tp.read_text())
            res = Result.load(rj)
            row = _row_from_result(res, truth, spec, wd=rj.parent)
        except Exception:  # noqa: BLE001
            continue
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(rj.stat().st_mtime))
        rows.append(
            {
                "run_id": f"{ts}-legacy",
                "tag": tag,
                "finished": time.strftime("%Y-%m-%d %H:%M", time.localtime(rj.stat().st_mtime)),
                "workdir": str(rj.parent),
                **row,
            }
        )
    rows.sort(key=lambda r: (r.get("finished", ""), r.get("run_id", "")), reverse=True)
    return rows


def write_ledger(root) -> list[dict]:
    root = Path(root)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    if rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        pl.DataFrame([{c: r.get(c) for c in cols} for r in rows], infer_schema_length=None).write_csv(
            out / "results.csv"
        )
        (out / "results.json").write_text(json.dumps(rows, indent=1))
    return rows


def run(
    rows_list,
    *,
    root,
    out=None,
    dgp="panel",
    dgp_kw=None,
    specs=None,
    profile=None,
    target_rows=2e6,
    gen_jobs=1,
    pyfixest_max_rows=0,
    keep_data=True,
    gen_on_workers=False,
    prep_opts=None,
    log=print,
):
    root = Path(root)
    out = Path(out) if out else root / "results"
    out.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    summary = out / f"{run_id}.md"
    log(
        f"benchmark run {run_id}\n  this run's summary: {summary}\n  all results:        "
        f"{out / 'results.csv'}  (or: dfeols results {root})"
    )
    ex = dfeols.executor(profile) if profile else dfeols.SerialExecutor()
    host = socket.gethostname()
    results = []
    try:
        for rows in rows_list:
            tag = f"{dgp}_{rows:.0e}".replace("+", "") + _param_suffix(dgp, dgp_kw)
            raw = root / f"raw_{tag}"
            d = plan_for_rows(rows, dgp, **(dgp_kw or {}))
            t = time.perf_counter()
            # generate on the coordinator by default: it runs while the first
            # worker jobs wait in the queue (on workers it would wait for them)
            generate(
                raw, d, jobs=gen_jobs, executor=ex if (profile and gen_on_workers) else None, verbose=True
            )
            t_gen = time.perf_counter() - t
            truth = load_truth(raw)
            todo = specs or list(truth["specs"])
            missing = set(todo) - set(truth["specs"])
            if missing:
                raise KeyError(f"{dgp} DGP has no specs {sorted(missing)}; have {sorted(truth['specs'])}")

            datasets, t_prep = {}, {}
            for part in sorted({truth["specs"][s]["partition"] for s in todo}):
                sd = root / f"shards_{tag}_by_{part}_{int(target_rows)}"
                t = time.perf_counter()
                if (sd / "dfeols_meta.json").exists():
                    datasets[part] = dfeols.Dataset.open(sd)
                    t_prep[part] = float("nan")
                else:
                    datasets[part] = dfeols.prep(
                        str(raw / "*.parquet"),
                        shard_dir=sd,
                        partition=part,
                        target_rows=int(target_rows),
                        overwrite=True,
                        executor=ex if profile else None,
                        **truth["prep"],
                        **((prep_opts or {}) if profile else {}),
                    )
                    t_prep[part] = time.perf_counter() - t
            n_rows = next(iter(datasets.values())).n_rows
            log(
                f"\n== {dgp}: {n_rows:,} rows, {d.n_units:,} units (generate {t_gen:.0f}s, prep "
                + ", ".join(f"by {k} {v:.0f}s" for k, v in t_prep.items())
                + ")"
            )

            for name in todo:
                s = truth["specs"][name]
                ds = datasets[s["partition"]]
                wd = root / "runs" / run_id / tag / name
                res = dfeols.fit(
                    ds,
                    s["formula"],
                    cluster=s["cluster"],
                    executor=ex,
                    workdir=wd,
                    label=f"{tag}-{name}",
                    log_every=15,
                )
                row = {
                    "run_id": run_id,
                    "tag": tag,
                    "finished": time.strftime("%Y-%m-%d %H:%M"),
                    "profile": profile or "serial",
                    "coordinator": host,
                    **_row_from_result(res, truth, name, n_rows=ds.n_rows, n_shards=len(ds.shards), wd=wd),
                    "generate_s": round(t_gen, 1),
                    "prep_s": round(t_prep[s["partition"]], 1),
                    "workdir": str(wd),
                }
                if pyfixest_max_rows and ds.n_rows <= pyfixest_max_rows:
                    row.update(pyfixest_baseline(raw, s, res))
                (wd / "bench.json").write_text(json.dumps(row, indent=1))
                results.append(row)
                log(json.dumps(row))
                # after every fit, so partial runs are visible
                write_ledger(root)
                write_summary(results, summary, profile or "serial", host, target_rows, run_id)
                shutil.copyfile(summary, out / "latest.md")
            if not keep_data:
                for sd in {Path(ds.root) for ds in datasets.values()}:
                    shutil.rmtree(sd, ignore_errors=True)
    finally:
        if hasattr(ex, "client"):
            ex.client.close()
            ex.cluster.close()
    if results:
        write_ledger(root)
        write_summary(results, summary, profile or "serial", host, target_rows, run_id)
        shutil.copyfile(summary, out / "latest.md")
    return results


def write_summary(rows, path: Path, where: str, host: str, target_rows, run_id: str = ""):
    cols = [
        "dgp",
        "rows",
        "spec",
        "expect",
        "validity",
        "shards",
        "clusters",
        "workers_avg",
        "fit_s",
        "round1_s",
        "round2_s",
        "rows_per_worker_s",
        "rows_per_s",
        "peak_worker_rss_gb",
        "max_abs_z",
        "wald_p",
        "eta_err_25",
        "eta_err_50",
        "pyfixest_s",
        "pyfixest_max_coef_diff",
    ]
    cols = [c for c in cols if any(c in r for r in rows)]
    lines = [
        f"# dfeols benchmark {run_id} ({where}, coordinator {host}, {time.strftime('%Y-%m-%d %H:%M')})",
        "",
        f"Shard target {int(target_rows):,} rows. Fit directories: runs/{run_id}/.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            vals.append(
                f"{v:,}"
                if isinstance(v, int) and not isinstance(v, bool)
                else (f"{v:.1e}" if c.endswith("diff") and v != "" else str(v))
            )
        lines.append("| " + " | ".join(vals) + " |")
    lines += [
        "",
        "validity: ok if an 'unbiased' spec recovers the truth (every |z| under the "
        "Bonferroni bound and Wald p > 0.001) or a 'biased' spec is rejected (p < 1e-4).",
        "rows_per_worker_s is throughput per worker, independent of how many workers the "
        "queue provided; workers_avg shows how many there were.",
        "eta_err_q = (predicted - actual) / actual finish time when q% of rows were done.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m dfeols.bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="large-scale configuration; --rows/--set/--specs still override",
    )
    ap.add_argument("--rows", type=float, nargs="+", default=None, help="default 1e6 1e7")
    ap.add_argument("--dgp", default="panel")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="DGP parameter")
    ap.add_argument("--specs", nargs="+", default=None, help="spec names (default: all of the DGP's)")
    ap.add_argument(
        "--profile",
        default=os.environ.get("DFEOLS_PROFILE"),
        help="site profile for workers (default: the profile of the job this runs in, "
        "set by `dfeols submit`/`dfeols bench`; serial when run directly)",
    )
    ap.add_argument("--root", default=None, help="data/shards/runs (default: <profile scratch>/bench)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target-rows", type=float, default=2e6, help="rows per shard")
    ap.add_argument(
        "--gen-jobs",
        type=int,
        default=_available_cores(),
        help="local processes for data generation (default: cores allocated to this job)",
    )
    ap.add_argument(
        "--gen-on-workers", action="store_true", help="generate on cluster workers instead of the coordinator"
    )
    ap.add_argument("--pyfixest-max-rows", type=float, default=0)
    ap.add_argument("--drop-data", action="store_true", help="delete shards after each size")
    ap.add_argument(
        "--prep-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="distributed prep option, e.g. --prep-opt max_files=0 (forces the two-level "
        "shuffle); see dfeols.prep_dist.prep_distributed",
    )
    a = ap.parse_args(argv)
    logging.getLogger("dfeols").setLevel(logging.INFO)
    pre = PRESETS.get(a.preset, {})
    if pre:
        a.dgp = pre["dgp"] if a.dgp == "panel" else a.dgp
        a.rows = a.rows or pre["rows"]
        a.specs = a.specs or pre["specs"]
        a.gen_on_workers = a.gen_on_workers or pre.get("gen_on_workers", False)
    a.rows = a.rows or [1e6, 1e7]
    kw = dict(pre.get("set", {}))
    for s in a.set:
        k, v = s.split("=", 1)
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
    if "theta" in kw:
        kw["theta"] = tuple(kw["theta"])
    root = a.root or str(dfeols.profile(a.profile).scratch_dir("bench"))
    run(
        a.rows,
        root=root,
        out=a.out,
        dgp=a.dgp,
        dgp_kw=kw,
        specs=a.specs,
        profile=a.profile,
        target_rows=a.target_rows,
        gen_jobs=a.gen_jobs,
        pyfixest_max_rows=a.pyfixest_max_rows,
        keep_data=not a.drop_data,
        gen_on_workers=a.gen_on_workers,
        prep_opts={k: json.loads(v) for k, v in (o.split("=", 1) for o in a.prep_opt)},
    )
    latest = Path(a.out or Path(root) / "results") / "latest.md"
    if latest.exists():
        print(latest.read_text())


if __name__ == "__main__":
    main()
