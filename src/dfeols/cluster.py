# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
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
  * interface = "ib": the coordinator and every worker use their own node's InfiniBand
    (IPoIB) address. OSCER's nodes default to Ethernet (eno1) and dfeols' traffic went over
    it: D6 r4's CG steps copied a 0.76 GB block to every worker and summed parts of that
    size, and waited 50-110 s a step before any task ran. IB names differ by node type
    (ibp4s0, ib0), so each job finds its own at start.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import shlex
import socket
import subprocess
import time
from pathlib import Path

from .lock import _job_id
from .site import Profile
from .site import profile as load_profile

log = logging.getLogger("dfeols")

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
TASK_THREAD_VARS = {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"}


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


def ib_interface() -> str | None:
    """The first InfiniBand (IPoIB) interface that is up and has an IPv4 address: a name
    starting with "ib" (ib0, ibp4s0, ...)."""
    import psutil

    stats = psutil.net_if_stats()
    for name in sorted(psutil.net_if_addrs()):
        st = stats.get(name)
        if name.startswith("ib") and (st is None or st.isup) and _interface_ips(name):
            return name
    return None


# worker job prologue for interface = "ib": the node's own IPoIB address, else its primary
# address (said in the worker's log); the worker listens there (--host)
IB_HOST_LINES = [
    "DFEOLS_WORKER_HOST=$(ip -br -4 addr 2>/dev/null | awk '$1 ~ /^ib/ && $2 == \"UP\" "
    "{ sub(\"/.*\", \"\", $3); print $3; exit }')",
    'if [ -z "$DFEOLS_WORKER_HOST" ]; then DFEOLS_WORKER_HOST=$(hostname -I | awk \'{print $1}\'); '
    'echo "dfeols: no InfiniBand address on $(hostname); worker listens on $DFEOLS_WORKER_HOST" >&2; fi',
    "export DFEOLS_WORKER_HOST",
]


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


def _env_lines(p: Profile, cores: int, memory: str | None = None, worker: bool = False) -> list[str]:
    # a worker runs several tasks at once, a thread each: BLAS and OpenMP get one thread per
    # task (no oversubscription); Polars and Rayon share one pool of the job's cores
    lines = [f"export {v}={1 if worker and v in TASK_THREAD_VARS else cores}" for v in THREAD_VARS]
    # Polars sizes its memory budget from the node's RAM, not the job's limit,
    # and spills to /var/tmp, which is small on many compute nodes. Pin both. Polars 2's
    # streaming engine spills to POLARS_OOC_SPILL_DIR (default /var/tmp/polars-$USER/spill)
    # and ignores POLARS_TEMP_DIR, so set both.
    if memory:
        lines.append(f"export POLARS_OVERRIDE_TOTAL_MEMORY_MB={int(_memory_bytes(memory) * 0.9 / 2**20)}")
    spill = str(p.scratch_dir("tmp", "polars")) if p.scratch else "${TMPDIR:-/tmp}/polars-$USER"
    lines += [f"mkdir -p {spill}", f"export POLARS_TEMP_DIR={spill}", f"export POLARS_OOC_SPILL_DIR={spill}/spill"]
    # Dask logs "Event loop was unresponsive" at INFO whenever a task holds the GIL over 3 s,
    # which on OSCER was 95% of a worker's .err (cellxgene H:L2, 36372921: 6521 messages, 26
    # over 30 s). Report only stalls long enough to matter; a profile's env can lower it.
    lines.append("export DASK_DISTRIBUTED__ADMIN__TICK__LIMIT=30s")
    lines += [f"export {k}={shlex.quote(str(v))}" for k, v in p.env.items()]
    return lines


# ------------------------------------------------- per-scheduler job kwargs
def _job_kwargs(p: Profile, cores: int, memory: str, walltime: str, *, worker: bool) -> dict:
    """dask-jobqueue keyword arguments that book `cores` CPUs and `memory` for one job: a
    worker gets a Dask thread per core (tasks ask for memory; see DaskExecutor), the
    coordinator job one."""
    s = p.scheduler
    kw = dict(
        cores=cores if worker else 1,
        processes=1,
        memory=memory,
        walltime=native_walltime(walltime, s),
        queue=p.queue,
        job_extra_directives=list(p.extra_directives),
    )
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
_JOBS = {
    "slurm": ("dask_jobqueue.slurm", "SLURMJob"),
    "pbs": ("dask_jobqueue.pbs", "PBSJob"),
    "sge": ("dask_jobqueue.sge", "SGEJob"),
    "lsf": ("dask_jobqueue.lsf", "LSFJob"),
}


# ------------------------------------------------------------- executors
def worker_log_tag() -> str:
    """Folder name for this coordinator's worker logs under <scratch>/logs/.

    The coordinator's own job id, so a run's worker logs (each named by the
    worker's job id) sit together and can be found from the parent job.
    Outside a batch job: local-<YYYYmmdd-HHMMSS>-<pid>.
    """
    jid = _job_id()
    if jid:
        return jid.replace("/", "_")
    return f"local-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"


def retire_spread_seconds(life: float, minutes: float | None = None) -> int:
    """Half-width of the window workers retire in (Dask's --lifetime-stagger): `minutes` if
    given, else 10% of the lifetime, at least 5 min; never more than a quarter of it."""
    s = 60 * float(minutes) if minutes is not None else max(0.1 * life, 300)
    return int(min(s, life / 4))


def allowed_failures(max_workers: int, n: int | None = None) -> int:
    """Dask's allowed-failures: a task fails (KilledWorker) once it has been on this many + 1
    workers that left while running it. `n` if given, else half of max_workers, so a task
    fails when it has outlived half the pool plus one; never below Dask's default 3."""
    if n is not None:
        return int(n)
    return max(int(0.5 * max_workers), 3)


def make_cluster(p: Profile, log_directory: str | None = None):
    """Create (not scale) a Dask cluster for profile p."""
    w = p.worker
    if p.scheduler == "local":
        from distributed import LocalCluster

        for v in THREAD_VARS:  # workers inherit the parent's environment
            os.environ.setdefault(v, "1" if v in TASK_THREAD_VARS else str(w["cores"]))
        return LocalCluster(
            n_workers=0,
            threads_per_worker=int(w["cores"]),
            processes=True,
            dashboard_address=None,
            resources={"MEMORY": _memory_bytes(w["memory"])},
        )

    import dask_jobqueue

    fix_scheduler_env(p.scheduler)
    kw = _job_kwargs(p, w["cores"], w["memory"], w["walltime"], worker=True)
    ib = p.interface == "ib"
    if p.interface:
        if ib:
            name = ib_interface()
            if name is None:
                # Workers on IB addresses are unreachable from a coordinator without IB (seen on
                # OSCER: the scheduler could not gather results), so all stay on Ethernet.
                name = private_interface()
                ib = False
                log.warning("cluster: interface = \"ib\" but this node has no InfiniBand address; "
                            "the coordinator listens on %s and workers stay on their default "
                            "addresses (Ethernet)", name)
        else:
            name = private_interface() if p.interface == "auto" else p.interface
        if name:
            log.info("cluster: coordinator listens on %s (%s)", name, ", ".join(_interface_ips(name)))
            kw.setdefault("scheduler_options", {})["interface"] = name
    wall = walltime_seconds(w["walltime"], p.scheduler)
    life = max(wall - min(60 * int(w["retire_minutes"]), wall // 4), 300)
    # Dask picks each worker's lifetime uniformly in center +/- stagger; the center sits a
    # stagger below `life`, so no worker outlives `life` (the walltime margin), and workers
    # that started together retire over a 2 x stagger window instead of all at once
    stagger = retire_spread_seconds(life, w.get("retire_spread_minutes"))
    center = life - stagger
    # tasks on a worker that leaves count against it; planned retirements and slow
    # connections are not the task's fault (OSCER H L2, 2026-11-20: KilledWorker after 4)
    kw.setdefault("scheduler_options", {})["allowed_failures"] = allowed_failures(
        w["max_workers"], w.get("allowed_failures"))
    if log_directory or p.scratch:
        ld = Path(log_directory) if log_directory else p.scratch_dir("logs", worker_log_tag())
        ld.mkdir(parents=True, exist_ok=True)
        kw["log_directory"] = str(ld)
        log.info("cluster: worker logs in %s", ld)
    kw["death_timeout"] = max(int(60 * float(w.get("start_minutes", 10))), 60)
    cls = getattr(dask_jobqueue, _CLASSES[p.scheduler])
    cl = cls(
        job_name="dfeols-worker",
        local_directory=w.get("local_directory") or "${TMPDIR:-/tmp}",
        job_script_prologue=list(p.prologue)
        + _env_lines(p, w["cores"], w["memory"], worker=True)
        + (IB_HOST_LINES if ib else []),
        worker_extra_args=(["--host", "$DFEOLS_WORKER_HOST"] if ib else [])
        + [
            "--lifetime",
            f"{center}s",
            "--lifetime-stagger",
            f"{stagger}s",
            "--resources",
            f"MEMORY={_memory_bytes(w['memory'])}",
        ],
        **kw,
    )
    cl.dfeols_log_directory = kw.get("log_directory")  # diag.attach_scheduler_log: next to them
    cl.dfeols_lifetime = (center, stagger)  # seconds, and the --lifetime-stagger either side
    return cl


def executor(
    profile: str | Profile | None = None,
    *,
    tick: float = 10.0,
    log_directory: str | None = None,
    hold: bool = False,
    **overrides,
):
    """Return a DaskExecutor whose workers come from the profile's scheduler.

    Workers join as the scheduler starts them. The Dask client and cluster are available as
    `ex.client` and `ex.cluster`.

    Parameters
    ----------
    profile : str or Profile, optional
        Site profile (see `dfeols.profile`). Default `$DFEOLS_PROFILE`, else the config's
        default, else `"local"`.
    tick : float, default 10.0
        Seconds between the coordinator's scheduler queries while tasks run.
    log_directory : str, optional
        Folder for the worker jobs' logs. Default `<scratch>/logs/...` when the profile has
        a scratch directory.
    hold : bool, default False
        Ask for `max_workers` and keep them while idle (sessions), instead of adapting
        between `min_workers` and `max_workers`.
    **overrides
        Worker settings (`cores`, `memory`, `walltime`, `max_workers`, `min_workers`,
        `retire_minutes`) and profile keys (`queue`, `account`, ...) for this executor only.

    Returns
    -------
    DaskExecutor
    """
    from distributed import Client

    from .executors import DaskExecutor

    if isinstance(profile, Profile):
        p = profile.with_overrides(**overrides)
    else:
        p = load_profile(profile, **overrides)
    cl = make_cluster(p, log_directory)
    w = p.worker
    if p.scheduler == "local" or hold:
        cl.scale(w["max_workers"])
    else:
        # Scale up fast, scale down slowly. Dask's default releases a worker
        # after ~3 s idle, which throws away workers that waited minutes in the
        # queue whenever there is a pause (between rounds, between fits, while
        # the coordinator solves). Release only after `idle_minutes` idle.
        interval = 10.0
        wait = max(int(60 * float(w.get("idle_minutes", 15)) / interval), 1)
        cl.adapt(
            minimum_jobs=w["min_workers"],
            maximum_jobs=w["max_workers"],
            interval=f"{interval}s",
            wait_count=wait,
        )
        if w.get("replace_ahead", True):
            _replace_ahead(cl, w["max_workers"], 60 * float(w.get("queue_wait_minutes", 30)),
                           max_spares=w.get("replace_ahead_max"),
                           quick_wait=60 * float(w.get("quick_wait_minutes", 3)))
    client = Client(cl)
    if p.scheduler == "slurm":
        _replace_lost_jobs(cl, w["max_workers"])
    if p.scheduler != "local" or log_directory:
        from .diag import attach_scheduler_log

        attach_scheduler_log(getattr(cl, "dfeols_log_directory", None) or log_directory)
    ex = DaskExecutor(
        client, max_workers=w["max_workers"], tick=tick, worker_memory=_memory_bytes(w["memory"])
    )
    ex.cluster = cl
    if p.scheduler != "local":
        _close_on_exit(cl)
    ex.profile = p
    log.info(
        "cluster: profile %r (%s), up to %d workers x %s cores, %s, walltime %s",
        p.name,
        p.scheduler,
        w["max_workers"],
        w["cores"],
        w["memory"],
        w["walltime"],
    )
    return ex


def _slurm_live_jobs() -> set[str] | None:
    """Job ids of this user's jobs Slurm still has (queued or running); None if squeue fails."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    try:
        r = subprocess.run(["squeue", "-h", "-u", user, "-o", "%i"], capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    if r.returncode:
        return None
    return {x.strip() for x in r.stdout.split() if x.strip()}


def lost_jobs(jobs: dict, connected: set, live: set) -> set:
    """Names of worker jobs that never connected and that the batch system no longer has.

    `jobs` maps a cluster's worker names to their batch job ids, `connected` holds the worker
    names the scheduler has seen, `live` the batch system's job ids. dask-jobqueue counts such a
    job as pending forever (it only drops workers that connected and then left), so adaptive
    scaling never replaces it."""
    return {n for n, j in jobs.items() if j and n not in connected and str(j) not in live}


def _replace_lost_jobs(cl, max_workers: int, every: float = 60.0, live_jobs=_slurm_live_jobs) -> None:
    """Every `every` s, drop worker jobs that ended without connecting (seen gone twice in a
    row), so the cluster submits replacements. Stops replacing, and says so, after
    3 x max_workers losses with no worker ever connected (a broken worker environment)."""
    import asyncio

    from tornado.ioloop import PeriodicCallback

    st = {"suspect": set(), "lost": 0, "ever": False, "stopped": False, "busy": False}

    async def check():
        if st["stopped"] or st["busy"] or cl.status.name in ("closing", "closed"):
            return
        st["busy"] = True
        try:
            seen = {d.get("name") for d in cl.scheduler_info.get("workers", {}).values()}
            st["ever"] = st["ever"] or bool(seen)
            jobs = {n: getattr(j, "job_id", None) for n, j in list(cl.workers.items())}
            cand = {n: j for n, j in jobs.items() if j and n not in seen}
            if not cand:
                st["suspect"] = set()
                return
            live = await asyncio.get_running_loop().run_in_executor(None, live_jobs)
            if live is None:
                return
            gone = lost_jobs(cand, seen, live)
            drop, st["suspect"] = gone & st["suspect"], gone - st["suspect"]
            if not drop:
                return
            for n in sorted(drop):
                log.warning("cluster: worker job %s ended without reaching the scheduler (see its log); "
                            "asking for a replacement", jobs[n])
                cl.workers.pop(n, None)
            st["lost"] += len(drop)
            if not st["ever"] and st["lost"] >= 3 * max_workers:
                st["stopped"] = True
                log.error("cluster: %d worker jobs ended without reaching the scheduler and none ever "
                          "has; not replacing more (check the worker logs)", st["lost"])
                return
            await cl._correct_state()
        except Exception as e:  # noqa: BLE001 (a watcher must not take the cluster down)
            log.warning("cluster: checking for lost worker jobs failed: %r", e)
        finally:
            st["busy"] = False

    pc = PeriodicCallback(check, every * 1000)
    cl.loop.add_callback(pc.start)
    cl.dfeols_lost_jobs = st
    cl.dfeols_lost_jobs_check = check


def queue_lead(waits: list[float], prior: float, life: float) -> float:
    """Seconds before a worker's retirement to ask for its replacement: 1.25 x the longest of the
    last 20 queue waits (submit to connect; `prior` until one is seen) plus 5 min, at most half a
    worker's lifetime. The longest, since waits vary a lot (OSCER: 5-16 min for the first pool,
    30-50 min for replacements) and asking early only costs some overlap."""
    q = max(waits[-20:]) if waits else prior
    return min(1.25 * q + 300.0, life / 2)


def retiring(connected: dict, now: float, life: float, stagger: float, lead: float,
             work_left: float | None, grace: float = 600.0) -> set:
    """Names of connected workers (name -> time first seen) that retire at their lifetime within
    `lead` s and that the work outlasts: replacements to ask for now.

    A worker retires between lifetime - stagger and lifetime + stagger after it started; it
    counts from `lead` before the earliest until `grace` after the latest (then it is late, and
    adaptive scaling treats any extra as surplus). `work_left` None (no run reporting) counts as
    work to come."""
    out = set()
    for n, t0 in connected.items():
        early, late = t0 + life - stagger, t0 + life + stagger + grace
        if early - lead <= now <= late and (work_left is None or work_left > max(early - now, 0) + 300):
            out.add(n)
    return out


def spares_cap(max_workers: int, n: int | None = None) -> int:
    """Most replacements asked for ahead at once: `n` if given, else 10% of max_workers, at
    least 1. Dask draws each worker's lifetime at random within the window, so every worker of
    a pool that started together comes due at once: uncapped, H L2 (OSCER 36358879, 2026-11-21)
    asked for 40 at 21:44 and ran 79 workers for 2 h, slower than 40 (/scratch-bound)."""
    if n is not None:
        return max(int(n), 0)
    return max(int(round(0.1 * max_workers)), 1)


def braked(waits: list[float], quick: float, k: int = 2) -> bool:
    """The last k queue waits were all within `quick` s: workers start at once, so none are
    asked for ahead (a retiring worker is replaced after it leaves)."""
    return len(waits) >= k and max(waits[-k:]) <= quick


def _replace_ahead(cl, max_workers: int, prior_wait: float, clock=time.time,
                   max_spares: int | None = None, quick_wait: float = 180.0) -> None:
    """Add, to adaptive scaling's target, one worker for each worker about to retire at its
    lifetime while the work outlasts it, so the replacement waits in the queue while the old
    worker still runs; at most `spares_cap` at once, and none while the last two queue waits
    were within `quick_wait` s (`braked`). It is computed on each adapt tick, so the extra drops
    the moment the old worker leaves. Queue waits are measured per job (first in the plan to
    first connected)."""
    ad = getattr(cl, "_adaptive", None)
    life, stagger = getattr(cl, "dfeols_lifetime", (None, None))
    if ad is None or life is None or not hasattr(ad, "safe_target"):
        log.warning("cluster: cannot ask for replacements ahead of retirement (no adaptive scaling)")
        return
    orig = ad.safe_target
    cap = spares_cap(max_workers, max_spares)
    st = {"planned": {}, "seen": {}, "waits": [], "extra": 0, "braked": False}

    async def safe_target():
        n = await orig()
        try:
            from . import progress

            now = clock()
            plan, seen = set(ad.plan), set(ad.observed)
            for x in plan:
                st["planned"].setdefault(x, now)
            for x in seen:
                if x not in st["seen"]:
                    st["seen"][x] = now
                    if x in st["planned"]:
                        st["waits"].append(now - st["planned"][x])
            for d in ("planned", "seen"):  # forget workers gone from both
                for x in [x for x in st[d] if x not in plan and x not in seen]:
                    del st[d][x]
            wl = progress.WORK_LEFT
            work_left = wl[1] - (now - wl[0]) if wl and now - wl[0] < 600 else None
            lead = queue_lead(st["waits"], prior_wait, life)
            due = retiring({x: st["seen"][x] for x in seen if x in st["seen"]}, now, life, stagger, lead, work_left)
            brake = braked(st["waits"], quick_wait)
            if brake != st["braked"]:
                log.info("cluster: the last 2 queue waits were %s %.0f min: %s", "within" if brake else "not all within",
                         quick_wait / 60, "replacements only after workers leave" if brake
                         else f"asking for replacements ahead again (at most {cap} at once)")
                st["braked"] = brake
            extra = 0 if brake else min(len(due), cap, max_workers)
            if extra > st["extra"]:
                log.info("cluster: %d worker(s) retire within %.0f min; asking for %d replacement(s) now, "
                         "%d ahead in all (at most %d; queue waits so far up to %.0f min)", len(due), lead / 60,
                         extra - st["extra"], extra, cap, max(st["waits"], default=prior_wait) / 60)
            st["extra"] = extra
            return n + extra
        except Exception as e:  # noqa: BLE001 (never take adaptive scaling down)
            log.warning("cluster: replacing retiring workers ahead failed: %r", e)
            return n

    ad.safe_target = safe_target
    cl.dfeols_replace_ahead = st


def _close_on_exit(cl) -> None:
    """Cancel the cluster's worker jobs, queued ones too, when the coordinator ends, including
    by SIGTERM (scancel, or the batch system at walltime). Python's default SIGTERM ends the
    process without cleanup, so the worker jobs still queued started after the coordinator had
    gone and sat until their death timeout."""
    import atexit
    import signal
    import sys
    import threading

    done = {"closed": False}

    def close():
        if done["closed"]:
            return
        done["closed"] = True
        try:
            cl.close(timeout=20)
        except Exception as e:  # pragma: no cover (best effort at exit)
            log.warning("cluster: closing the worker jobs at exit failed: %r", e)

    atexit.register(close)
    if threading.current_thread() is not threading.main_thread():
        return
    prev = signal.getsignal(signal.SIGTERM)
    if prev not in (signal.SIG_DFL, None):
        return  # the caller's own handler: leave it

    def on_term(signum, frame):
        log.warning("cluster: SIGTERM, cancelling the worker jobs")
        close()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)


# ---------------------------------------------------- coordinator scripts
def coordinator_script(p: Profile, command: str, job_name: str = "dfeols", output: str | None = None) -> str:
    """A batch script that runs `command` as the long-lived coordinator."""
    c = p.coordinator
    lines = ["#!/usr/bin/env bash"]
    if p.scheduler != "local":
        import importlib

        mod, name = _JOBS[p.scheduler]
        Job = getattr(importlib.import_module(mod), name)
        kw = _job_kwargs(p, c["cores"], c["memory"], c["walltime"], worker=False)
        if c.get("queue"):  # its own partition: a long one, while the workers use a short one
            kw["queue"] = c["queue"]
        if c.get("extra_directives") is not None:  # replace the profile's, e.g. without a node
            # constraint the long partition can't meet (OSCER longjobs has no hdr nodes)
            kw["job_extra_directives"] = list(c["extra_directives"])
        job = Job(scheduler="tcp://0.0.0.0:0", name="coordinator", job_name=job_name, **kw)
        lines.append(job.job_header)
        out = output or f"{job_name}-%j.out"
        if p.scheduler == "slurm":
            lines.append(f"#SBATCH -o {out}")
        elif p.scheduler == "pbs":
            lines.append("#PBS -j oe")  # output: <job_name>.o<jobid> in the submit dir
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
        lines.append("# configless Slurm: a /proc SLURM_CONF breaks nested sbatch")
        lines.append('if [[ "${SLURM_CONF:-}" == /proc/* ]]; then unset SLURM_CONF; fi')
    lines += ["", command, ""]
    return "\n".join(lines)


def submit(script: str, p: Profile, path: str | os.PathLike | None = None) -> str:
    """Write a job script and submit it; returns the scheduler's output."""
    path = Path(path or f"dfeols-job-{os.getpid()}.sh")
    path.write_text(script)
    path.chmod(0o755)
    if p.scheduler == "local":
        proc = subprocess.Popen(
            ["bash", str(path)],
            stdout=open(Path(path).with_suffix(".out"), "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return f"started local process {proc.pid}, output in {Path(path).with_suffix('.out')}"
    fix_scheduler_env(p.scheduler)
    cmd = {
        "slurm": ["sbatch", str(path)],
        "pbs": ["qsub", str(path)],
        "sge": ["qsub", str(path)],
        "lsf": ["bsub"],
    }[p.scheduler]
    stdin = open(path) if p.scheduler == "lsf" else None
    r = subprocess.run(cmd, stdin=stdin, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr}")
    return r.stdout.strip()
