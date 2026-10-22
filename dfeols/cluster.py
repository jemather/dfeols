"""Scheduler-agnostic execution: build a Dask executor, or a coordinator job
script, from a site profile (see dfeols.site).

    ex = dfeols.executor("mycluster")             # elastic workers via the scheduler
    ex = dfeols.executor("local", max_workers=8)  # this machine
    res = dfeols.fit(ds, "y ~ x | unit + time", cluster="unit", executor=ex)

Supported schedulers: Slurm, PBS/Torque, SGE/UGE, LSF (all through
dask-jobqueue) and local. Slurm is tested on a real cluster; the others use
dask-jobqueue's job templates and need a site check (`dfeols jobscript`).

Site-specific workarounds live here, not in user code:
  * Slurm: a /proc SLURM_CONF inside a job breaks nested sbatch (configless
    sites) -> cleared before submitting worker jobs.
  * All: the coordinator listens on a private (cluster-internal) interface,
    since a campus/public address is often unreachable from compute nodes.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path

from .site import Profile, profile as load_profile

log = logging.getLogger("dfeols")

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS", "POLARS_MAX_THREADS")


# --------------------------------------------------------------- utilities
def walltime_seconds(s: str, scheduler: str = "slurm") -> int:
    """Walltime string to seconds. Accepts D-HH:MM:SS, HH:MM:SS, and for LSF
    HH:MM (LSF's -W format); for others MM:SS with two fields."""
    s = str(s).strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(p) for p in s.split(":")]
    if len(parts) == 2 and scheduler == "lsf":
        parts = parts + [0]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def native_walltime(s: str, scheduler: str) -> str:
    """Convert a walltime in any accepted form to the scheduler's own format:
    Slurm D-HH:MM:SS, PBS/SGE HHH:MM:SS (hours may exceed 24), LSF HHH:MM."""
    secs = walltime_seconds(s, scheduler)
    h, rem = divmod(secs, 3600)
    m, sec = divmod(rem, 60)
    if scheduler == "slurm":
        d, h = divmod(h, 24)
        return (f"{d}-" if d else "") + f"{h:02d}:{m:02d}:{sec:02d}"
    if scheduler == "lsf":
        return f"{h:02d}:{m:02d}"
    return f"{h:02d}:{m:02d}:{sec:02d}"


def private_interface() -> str | None:
    """A network interface with a private IPv4 address, preferring 10.x."""
    import psutil
    best = []
    stats = psutil.net_if_stats()
    for name, addrs in psutil.net_if_addrs().items():
        st = stats.get(name)
        if st is not None and not st.isup:
            continue
        for a in addrs:
            if a.family != socket.AF_INET:
                continue
            ip = ipaddress.ip_address(a.address)
            if ip.is_loopback or ip.is_link_local or not ip.is_private:
                continue
            best.append((0 if a.address.startswith("10.") else 1, name, a.address))
    return sorted(best)[0][1] if best else None


def _interface_ips(name: str) -> list[str]:
    import psutil
    return [a.address for a in psutil.net_if_addrs().get(name, []) if a.family == socket.AF_INET]


def fix_scheduler_env(scheduler: str) -> None:
    if scheduler == "slurm":
        conf = os.environ.get("SLURM_CONF", "")
        if conf.startswith("/proc/"):
            os.environ.pop("SLURM_CONF")
            log.info("cluster: cleared SLURM_CONF=%s so sbatch works inside a job", conf)


def _memory_bytes(mem: str) -> int:
    from dask.utils import parse_bytes
    return int(parse_bytes(mem))


def _env_lines(p: Profile, cores: int, memory: str | None = None) -> list[str]:
    lines = [f"export {v}={cores}" for v in THREAD_VARS]
    # Polars sizes its memory budget from the node's RAM, not the job's limit,
    # and spills to /var/tmp, which is small on many compute nodes. Pin both.
    if memory:
        lines.append(f"export POLARS_OVERRIDE_TOTAL_MEMORY_MB={int(_memory_bytes(memory) * 0.9 / 2**20)}")
    spill = str(p.scratch_dir("tmp", "polars")) if p.scratch else "${TMPDIR:-/tmp}/polars-$USER"
    lines += [f"mkdir -p {spill}", f"export POLARS_TEMP_DIR={spill}"]
    lines += [f"export {k}={shlex.quote(str(v))}" for k, v in p.env.items()]
    return lines


# ------------------------------------------------- per-scheduler job kwargs
def _job_kwargs(p: Profile, cores: int, memory: str, walltime: str, *, worker: bool) -> dict:
    """dask-jobqueue keyword arguments that book `cores` CPUs and `memory`
    for one job, with a single Dask task slot (one shard at a time)."""
    s = p.scheduler
    kw = dict(cores=1, processes=1, memory=memory, walltime=native_walltime(walltime, s), queue=p.queue,
              job_extra_directives=list(p.extra_directives))
    if s in ("slurm", "pbs") and p.account:
        kw["account"] = p.account
    if p.project and s in ("pbs", "sge", "lsf"):
        kw["project"] = p.project
    if s == "slurm":
        kw["job_cpu"] = cores
    elif s == "pbs":
        spec = p.worker.get("resource_spec") or "select=1:ncpus={cores}:mem={memory}"
        kw["resource_spec"] = spec.format(cores=cores, memory=memory)
    elif s == "sge":
        pe = p.worker.get("pe", "smp")
        per_slot = max(_memory_bytes(memory) // max(cores, 1) // 2**20, 1)
        spec = p.worker.get("resource_spec") or "h_vmem={per_slot}M"
        kw["resource_spec"] = spec.format(cores=cores, memory=memory, per_slot=per_slot)
        kw["job_extra_directives"] = kw["job_extra_directives"] + [f"-pe {pe} {cores}"]
    elif s == "lsf":
        kw["ncpus"] = cores
        kw["mem"] = _memory_bytes(memory)
        if p.worker.get("lsf_units"):
            kw["lsf_units"] = p.worker["lsf_units"]
    return kw


_CLASSES = {"slurm": "SLURMCluster", "pbs": "PBSCluster", "sge": "SGECluster", "lsf": "LSFCluster"}
_JOBS = {"slurm": ("dask_jobqueue.slurm", "SLURMJob"), "pbs": ("dask_jobqueue.pbs", "PBSJob"),
         "sge": ("dask_jobqueue.sge", "SGEJob"), "lsf": ("dask_jobqueue.lsf", "LSFJob")}


# ------------------------------------------------------------- executors
def make_cluster(p: Profile, log_directory: str | None = None):
    """Create (not scale) a Dask cluster for profile p."""
    w = p.worker
    if p.scheduler == "local":
        from distributed import LocalCluster
        for v in THREAD_VARS:  # workers inherit the parent's environment
            os.environ.setdefault(v, str(w["cores"]))
        return LocalCluster(n_workers=0, threads_per_worker=1, processes=True,
                            dashboard_address=None)

    import dask_jobqueue
    fix_scheduler_env(p.scheduler)
    kw = _job_kwargs(p, w["cores"], w["memory"], w["walltime"], worker=True)
    if p.interface:
        name = private_interface() if p.interface == "auto" else p.interface
        if name:
            log.info("cluster: coordinator listens on %s (%s)", name, ", ".join(_interface_ips(name)))
            kw["scheduler_options"] = {"interface": name}
    wall = walltime_seconds(w["walltime"], p.scheduler)
    life = max(wall - min(60 * int(w["retire_minutes"]), wall // 4), 300)
    if log_directory or p.scratch:
        ld = Path(log_directory) if log_directory else p.scratch_dir("logs")
        ld.mkdir(parents=True, exist_ok=True)
        kw["log_directory"] = str(ld)
    cls = getattr(dask_jobqueue, _CLASSES[p.scheduler])
    return cls(
        job_name="dfeols-worker",
        local_directory=w.get("local_directory") or "${TMPDIR:-/tmp}",
        job_script_prologue=list(p.prologue) + _env_lines(p, w["cores"], w["memory"]),
        worker_extra_args=["--lifetime", f"{life}s", "--lifetime-stagger", "5m"],
        **kw)


def executor(profile: str | Profile | None = None, *, tick: float = 10.0,
             log_directory: str | None = None, **overrides):
    """Return a DaskExecutor whose workers come from the profile's scheduler.

    Worker settings (cores, memory, walltime, max_workers, min_workers,
    retire_minutes) and profile keys (queue, account, ...) can be overridden
    as keyword arguments. Workers join as the scheduler starts them.
    The client and cluster are available as ex.client and ex.cluster.
    """
    from distributed import Client

    from .executors import DaskExecutor

    p = profile if isinstance(profile, Profile) else load_profile(profile, **overrides)
    cl = make_cluster(p, log_directory)
    w = p.worker
    if p.scheduler == "local":
        cl.scale(w["max_workers"])
    else:
        # Scale up fast, scale down slowly. Dask's default releases a worker
        # after ~3 s idle, which throws away workers that waited minutes in the
        # queue whenever there is a pause (between rounds, between fits, while
        # the coordinator solves). Release only after `idle_minutes` idle.
        interval = 10.0
        wait = max(int(60 * float(w.get("idle_minutes", 15)) / interval), 1)
        cl.adapt(minimum_jobs=w["min_workers"], maximum_jobs=w["max_workers"],
                 interval=f"{interval}s", wait_count=wait)
    client = Client(cl)
    ex = DaskExecutor(client, max_workers=w["max_workers"], tick=tick)
    ex.cluster = cl
    ex.profile = p
    log.info("cluster: profile %r (%s), up to %d workers x %s cores, %s, walltime %s",
             p.name, p.scheduler, w["max_workers"], w["cores"], w["memory"], w["walltime"])
    return ex


# ---------------------------------------------------- coordinator scripts
def coordinator_script(p: Profile, command: str, job_name: str = "dfeols",
                       output: str | None = None) -> str:
    """A batch script that runs `command` as the long-lived coordinator."""
    c = p.coordinator
    lines = ["#!/usr/bin/env bash"]
    if p.scheduler != "local":
        import importlib
        mod, name = _JOBS[p.scheduler]
        Job = getattr(importlib.import_module(mod), name)
        kw = _job_kwargs(p, c["cores"], c["memory"], c["walltime"], worker=False)
        job = Job(scheduler="tcp://0.0.0.0:0", name="coordinator", job_name=job_name, **kw)
        lines.append(job.job_header)
        out = output or f"{job_name}-%j.out"
        if p.scheduler == "slurm":
            lines.append(f"#SBATCH -o {out}")
        elif p.scheduler == "pbs":
            lines.append("#PBS -j oe")       # output: <job_name>.o<jobid> in the submit dir
        elif p.scheduler == "sge":
            lines.append(f"#$ -o {out.replace('%j', '$JOB_ID')}")
        elif p.scheduler == "lsf":
            lines.append(f"#BSUB -o {out.replace('%j', '%J')}")
    lines += ["", "set -eo pipefail"]
    if p.scheduler == "pbs":
        lines.append('cd "${PBS_O_WORKDIR:-.}"')
    lines += list(p.prologue)
    lines += _env_lines(p, c["cores"], c["memory"])
    lines += ["export PYTHONUNBUFFERED=1", f"export DFEOLS_PROFILE={shlex.quote(p.name)}"]
    if p.scheduler == "slurm":
        lines.append('# configless Slurm: a /proc SLURM_CONF breaks nested sbatch')
        lines.append('if [[ "${SLURM_CONF:-}" == /proc/* ]]; then unset SLURM_CONF; fi')
    lines += ["", command, ""]
    return "\n".join(lines)


def submit(script: str, p: Profile, path: str | os.PathLike | None = None) -> str:
    """Write a job script and submit it; returns the scheduler's output."""
    path = Path(path or f"dfeols-job-{os.getpid()}.sh")
    path.write_text(script)
    path.chmod(0o755)
    if p.scheduler == "local":
        proc = subprocess.Popen(["bash", str(path)], stdout=open(Path(path).with_suffix(".out"), "w"),
                                stderr=subprocess.STDOUT, start_new_session=True)
        return f"started local process {proc.pid}, output in {Path(path).with_suffix('.out')}"
    fix_scheduler_env(p.scheduler)
    cmd = {"slurm": ["sbatch", str(path)], "pbs": ["qsub", str(path)],
           "sge": ["qsub", str(path)], "lsf": ["bsub"]}[p.scheduler]
    stdin = open(path) if p.scheduler == "lsf" else None
    r = subprocess.run(cmd, stdin=stdin, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr}")
    return r.stdout.strip()
