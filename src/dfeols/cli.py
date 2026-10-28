# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Command line helpers.

  dfeols status RUN_DIR [--watch 30]   progress and ETA of a running fit
  dfeols status SHARD_DIR              all runs under <shard_dir>/runs
  dfeols info SHARD_DIR                summary of a prepared dataset
  dfeols plan SOURCE --formula F [--cluster C] [--partition K1,K2] [--profile P] [--json]
  dfeols plan SOURCE --fe A+B^C --vars Y,X1,X2 ...
                                       probe the FE structure before prep: levels,
                                       nesting, skew, solver path and memory per key

  dfeols config init [--force]         write a site config template
  dfeols config init --from-slurm [--profile NAME] [--write] [--sinfo F --scontrol F]
                                       draft a profile's node classes from sinfo/scontrol
  dfeols config show [--profile P]     print the resolved profile
  dfeols jobscript "CMD" [--profile P] print a coordinator job script for CMD
  dfeols submit SCRIPT.py [--profile P] [--name N] [-- ARGS]
  dfeols submit -m MODULE  [--profile P] [--name N] [-- ARGS]
                                       submit as a coordinator job
  dfeols bench [--profile P] [BENCH OPTIONS]
                                       submit a benchmark as a coordinator job, e.g.
                                       dfeols bench --profile oscer --preset skewed_groups
  dfeols check [--profile P]           start one worker, run a task, report
  dfeols results [ROOT] [-n 20] [--run RUN_ID]
                                       benchmark results (default ROOT: <scratch>/bench)

Where things go (under the profile's scratch directory):
  jobs/      coordinator job scripts and their output (dfeols submit)
  logs/      worker logs
  bench/     benchmark data, shards, runs/<run_id>/ and results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .progress import fmt_td


def _print_status(p: Path):
    s = json.loads(p.read_text())
    age = time.time() - p.stat().st_mtime
    print(
        f"{s['label']}  (updated {s['updated']}, {fmt_td(age)} ago)"
        f"  workers {s['workers']}" + (f"/{s['max_workers']}" if s.get("max_workers") else "")
    )
    for st in s["stages"]:
        state = "done" if st["finished"] else ("running" if st["elapsed_seconds"] else "waiting")
        line = (
            f"  {st['stage']:<7} {state:<8} {st['tasks_done']:>5}/{st['tasks']:<5} shards "
            f"{st['pct_rows']:>5.1f}% rows  elapsed {fmt_td(st['elapsed_seconds'])}"
        )
        if not st["finished"] and st["eta_seconds"] is not None:
            line += f"  ETA {fmt_td(st['eta_seconds'])}"
            if st.get("eta_band_seconds"):
                lo, hi = st["eta_band_seconds"]
                line += f" ({fmt_td(lo)}-{fmt_td(hi)})"
            if st.get("rate_projected"):
                line += " [projected from round 1]"
            if st.get("eta_seconds_at_max_workers") is not None:
                line += f"; {fmt_td(st['eta_seconds_at_max_workers'])} at max workers"
        print(line)
    if s.get("eta_total_seconds") is not None and not all(x["finished"] for x in s["stages"]):
        print(
            f"  total ETA {fmt_td(s['eta_total_seconds'])}  -> about {s['finish_time']}"
            "  (as of the update time above)"
        )
    if age > 600 and not all(x["finished"] for x in s["stages"]):
        print("  warning: status not updated for 10+ minutes; is the coordinator job still running?")


def status(path: str, watch: float | None):
    p = Path(path)
    files = [p / "status.json"] if (p / "status.json").exists() else sorted(p.glob("runs/*/status.json"))
    if not files:
        sys.exit(f"no status.json under {p}")
    while True:
        for f in files:
            _print_status(f)
        if not watch:
            return
        time.sleep(watch)
        print()


def info(path: str):
    from .prep import Dataset

    ds = Dataset.open(path)
    rows = sorted(s["rows"] for s in ds.shards)
    print(ds)
    print(f"shard rows: min {rows[0]:,}  median {rows[len(rows) // 2]:,}  max {rows[-1]:,}")
    for kind, d in (("FE", ds.fes), ("cluster", ds.clusters)):
        for name, m in d.items():
            extra = ""
            if "nested_in_cluster" in m:
                extra = "  nested in cluster: " + (
                    ", ".join(k for k, v in m["nested_in_cluster"].items() if v) or "none"
                )
            where = "local (nested in partition)" if m["nested_in_partition"] else "crosses partitions"
            print(f"{kind:>7} {name:<20} {m['n_levels']:>12,} levels  {where}{extra}")


def plan_cmd(args):
    from .planner import format_report, plan

    def split(s):
        return [x.strip() for x in s.split(",") if x.strip()] if s else None

    r = plan(
        args.source,
        args.formula,
        fe=[x.strip() for x in args.fe.split("+")] if args.fe else None,
        vars=split(args.vars),
        cluster=args.cluster,
        weights=args.weights,
        partition=split(args.partition),
        target_rows=args.target_rows,
        max_dummy_levels=args.max_dummy_levels,
        profile=args.profile,
    )
    print(json.dumps(r, indent=1) if args.json else format_report(r))


def config_cmd(args):
    from .site import config_path, init_config, profile

    if args.action == "init" and args.from_slurm:
        return slurm_init(args)
    if args.action == "init":
        cp = config_path()
        if cp.exists() and args.force:
            bak = cp.with_name(cp.name + time.strftime(".%Y%m%d-%H%M%S.bak"))
            bak.write_text(cp.read_text())
            print(f"backed up the existing config to {bak}")
        p = init_config(force=args.force)
        print(f"wrote {p}; edit the [profiles.cluster] table for your system")
    elif args.action == "path":
        print(config_path())
    else:
        import dataclasses
        import pprint

        pprint.pprint(dataclasses.asdict(profile(args.profile)), sort_dicts=False)


def slurm_init(args):
    """Draft a profile's node classes from sinfo and scontrol (this machine's, or saved output)."""
    from .site import config_path, init_config, load_config
    from .slurm_site import draft_profile, read_slurm

    if bool(args.sinfo) != bool(args.scontrol):
        sys.exit("--sinfo and --scontrol go together")
    if args.sinfo:
        sinfo, scontrol = Path(args.sinfo).read_text(), Path(args.scontrol).read_text()
    else:
        try:
            sinfo, scontrol = read_slurm()
        except RuntimeError as e:
            sys.exit(str(e))
    name = args.profile or "slurm"
    text = draft_profile(sinfo, scontrol, name)
    if not args.write:
        print(text)
        return
    cp = config_path()
    if name in load_config().get("profiles", {}):
        sys.exit(f"profile {name!r} is already in {cp}; print the draft (no --write) and merge it by hand")
    if not cp.exists():
        init_config()
    with open(cp, "a") as fh:
        fh.write("\n" + text)
    print(f"added profile {name!r} to {cp}; set its scratch and prologue, and check the policy lines")


def jobscript_cmd(args):
    from .cluster import coordinator_script
    from .site import profile

    print(coordinator_script(profile(args.profile), args.command, job_name=args.name))


def _submit_options(args):
    """Split the REMAINDER after SCRIPT: submit's own options before `--`, the script's after.

    argparse's REMAINDER swallows `--profile P` written after the script (the documented
    form), so pick those out here. Without `--`, unknown words go to the script as before.
    """
    words = list(args.args)
    if "--" in words:
        i = words.index("--")
        own, rest = words[:i], words[i + 1 :]
    else:
        own, rest = words, []
    op = argparse.ArgumentParser(prog="dfeols submit", add_help=False)
    op.add_argument("--profile", default=args.profile)
    op.add_argument("--name", default=args.name)
    op.add_argument("-m", "--module", default=args.module)
    known, extra = op.parse_known_args(own)
    args.profile, args.name, args.module = known.profile, known.name, known.module
    return extra + rest


def submit_cmd(args):
    import shlex as _sh

    from .cluster import coordinator_script
    from .site import profile

    rest = _submit_options(args)
    p = profile(args.profile)
    jobs = p.scratch_dir("jobs")
    jobs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.module:
        target = f"-m {args.module}"
        if args.script:
            rest = [args.script] + rest
    elif args.script:
        target = _sh.quote(str(Path(args.script).resolve()))
    else:
        sys.exit("give a script or -m module")
    cmd = f"python {target} " + " ".join(_sh.quote(a) for a in rest)
    output = str(jobs / f"{args.name}-%j.out")
    script = coordinator_script(
        p, f"cd {_sh.quote(str(Path.cwd()))}\n" + cmd.strip(), job_name=args.name, output=output
    )
    _submit_and_report(p, script, jobs / f"{args.name}-{stamp}.sh", output)


def _submit_and_report(p, script, path, output):
    from .cluster import submit

    mods = [l for l in p.prologue if l.strip().startswith(("module", "source", "conda"))]
    print(f"profile: {p.name} ({p.scheduler}, from {p.source})")
    for l in mods:
        print(f"  {l}")
    print(submit(script, p, path=path))
    shown = output.replace("%j", "<jobid>") if p.scheduler == "slurm" else str(Path(path).parent)
    print(f"job output: {shown}")


def bench_cmd(args, rest):
    import shlex as _sh

    from .cluster import coordinator_script
    from .site import profile

    p = profile(args.profile)
    jobs = p.scratch_dir("jobs")
    jobs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # the job exports DFEOLS_PROFILE, so the benchmark uses the same profile
    cmd = "python -m dfeols.bench " + " ".join(_sh.quote(a) for a in rest)
    output = str(jobs / f"{args.name}-%j.out")
    script = coordinator_script(
        p, f"cd {_sh.quote(str(Path.cwd()))}\n" + cmd.strip(), job_name=args.name, output=output
    )
    _submit_and_report(p, script, jobs / f"{args.name}-{stamp}.sh", output)


def check_cmd(args):
    """Start one worker through the profile's scheduler and run a task on it."""
    import socket
    import time as _t

    from .cluster import executor

    ex = executor(args.profile, max_workers=1, min_workers=1)
    print(
        f"coordinator {socket.gethostname()} at {ex.client.scheduler.address}; waiting for a worker "
        f"(up to {args.timeout}s; queue wait counts)...",
        flush=True,
    )
    t = _t.time()
    try:
        ex.client.wait_for_workers(1, timeout=args.timeout)
    except Exception as e:
        print(
            f"FAILED: no worker connected after {args.timeout}s ({e}).\n"
            "  - jobs still pending? check your scheduler queue\n"
            "  - jobs exited quickly? read the worker logs (profile scratch/logs)\n"
            "  - 'Timed out trying to connect'? set `interface` in the profile to the\n"
            "    compute nodes' internal network interface"
        )
        sys.exit(1)
    host = ex.client.submit(socket.gethostname, pure=False).result()
    print(f"OK: worker on {host} connected after {_t.time() - t:.0f}s and ran a task")
    ex.client.close()
    ex.cluster.close()


def results_cmd(args):
    from .bench import collect
    from .site import profile

    root = Path(args.root) if args.root else profile(args.profile).scratch_dir("bench")
    rows = collect(root)
    if args.run:
        rows = [r for r in rows if r.get("run_id", "").startswith(args.run)]
    if not rows:
        sys.exit(f"no benchmark fits under {root}/runs")
    cols = [
        "run_id",
        "tag",
        "spec",
        "validity",
        "N",
        "fit_s",
        "rows_per_worker_s",
        "workers_avg",
        "peak_worker_rss_gb",
        "max_abs_z",
        "wald_p",
    ]
    shown = rows[: args.n]
    w = {c: max(len(c), *(len(_fmt(r.get(c))) for r in shown)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in shown:
        print("  ".join(_fmt(r.get(c)).ljust(w[c]) for c in cols))
    print(f"\n{len(rows)} fits under {root}  (showing {len(shown)}, newest first)")
    print(f"table:     {root / 'results' / 'results.csv'}")
    print(f"summaries: {root / 'results'}/<run_id>.md  (latest.md = most recent)")
    print(f"fit files: {shown[0].get('workdir', root / 'runs')}  (result.json, status.json, history.jsonl)")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return str(v)


def main(argv=None):
    from . import __version__

    ap = argparse.ArgumentParser(prog="dfeols")
    ap.add_argument("--version", action="version", version=f"dfeols {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config", help="site configuration")
    c.add_argument("action", choices=["init", "show", "path"])
    c.add_argument("--profile")
    c.add_argument("--force", action="store_true")
    c.add_argument(
        "--from-slurm", action="store_true", help="init: draft a profile's node classes from Slurm"
    )
    c.add_argument("--sinfo", help="--from-slurm: saved sinfo output instead of running it")
    c.add_argument("--scontrol", help="--from-slurm: saved `scontrol show partition -o` output")
    c.add_argument("--write", action="store_true", help="--from-slurm: add the draft to the config file")
    j = sub.add_parser("jobscript", help="print a coordinator job script")
    j.add_argument("command")
    j.add_argument("--profile")
    j.add_argument("--name", default="dfeols")
    s_ = sub.add_parser("submit", help="submit a Python script (or -m module) as a coordinator job")
    s_.add_argument("-m", "--module", default=None)
    s_.add_argument("script", nargs="?")
    s_.add_argument("args", nargs=argparse.REMAINDER)
    s_.add_argument("--profile")
    s_.add_argument("--name", default="dfeols")
    bn = sub.add_parser("bench", help="submit a benchmark as a coordinator job (other options go to it)")
    bn.add_argument("--profile")
    bn.add_argument("--name", default="dfeols-bench")
    k = sub.add_parser("check", help="verify workers can start and connect")
    k.add_argument("--profile")
    k.add_argument("--timeout", type=int, default=3600)
    a = sub.add_parser("status")
    a.add_argument("path")
    a.add_argument("--watch", type=float, default=None, help="refresh every N seconds")
    b = sub.add_parser("info")
    b.add_argument("path")
    pl_ = sub.add_parser("plan", help="probe the FE structure and scale of a specification")
    pl_.add_argument("source", help="Parquet path or glob, or CSV")
    pl_.add_argument("--formula", help='dfeols formula, e.g. "y ~ x | a + b^c"')
    pl_.add_argument("--fe", help='FEs without a formula, e.g. "a + b^c"')
    pl_.add_argument("--vars", help="with --fe: outcome first, then regressors, comma-separated")
    pl_.add_argument("--cluster")
    pl_.add_argument("--weights")
    pl_.add_argument("--partition", help="candidate keys, comma-separated (a^b for a compound key)")
    pl_.add_argument("--target-rows", type=int, default=1_000_000)
    pl_.add_argument("--max-dummy-levels", type=int, default=2000)
    pl_.add_argument("--profile", help="check memory against this site profile")
    pl_.add_argument("--json", action="store_true", help="print the report as JSON")
    r = sub.add_parser("results", help="benchmark results table")
    r.add_argument("root", nargs="?")
    r.add_argument("--profile")
    r.add_argument("-n", type=int, default=20)
    r.add_argument("--run", default=None, help="only this run id (prefix)")
    sub.add_parser("estimation", help="validate, redact or digest estimation records", add_help=False)
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "bench":
        return bench_cmd(args, rest)
    if args.cmd == "estimation":
        from estimation_format import main as estimation_main

        sys.exit(estimation_main(rest))
    if rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    {
        "results": lambda: results_cmd(args),
        "status": lambda: status(args.path, args.watch),
        "info": lambda: info(args.path),
        "plan": lambda: plan_cmd(args),
        "config": lambda: config_cmd(args),
        "jobscript": lambda: jobscript_cmd(args),
        "submit": lambda: submit_cmd(args),
        "check": lambda: check_cmd(args),
    }[args.cmd]()


if __name__ == "__main__":
    main()
