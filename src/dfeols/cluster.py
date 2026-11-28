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
import math
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
        if w.get("io_ceiling", True):
            _pool_ceiling(cl, w)
    client = Client(cl)
    if p.scheduler == "slurm":
        _replace_lost_jobs(cl, w["max_workers"])
    if p.scheduler != "local":
        _restart_guard(cl, int(w.get("restart_limit", 3)), 60 * float(w.get("restart_window_minutes", 30)),
                       node_of=_slurm_node if p.scheduler == "slurm" else None,
                       max_excluded=max(3, w["max_workers"] // 4))
    if p.scheduler != "local" or log_directory:
        from .diag import attach_scheduler_log

        attach_scheduler_log(getattr(cl, "dfeols_log_directory", None) or log_directory)
    ex = DaskExecutor(
        client, max_workers=w["max_workers"], tick=tick, worker_memory=_memory_bytes(w["memory"]),
        retire_at_pass=p.scheduler != "local" and bool(w.get("retire_at_pass", True)),
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


def _slurm_node(job_id: str) -> str | None:
    """The node a Slurm job runs (or ran) on; None if neither squeue nor sacct says."""
    for cmd in (["squeue", "-h", "-j", str(job_id), "-o", "%N"],
                ["sacct", "-n", "-X", "-P", "-j", str(job_id), "-o", "NodeList"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except Exception:  # noqa: BLE001
            continue
        node = r.stdout.strip().splitlines()[0].strip() if not r.returncode and r.stdout.strip() else ""
        if node and node not in ("(null)", "None assigned"):
            return node
    return None


def exclude_directives(directives: list[str], nodes) -> list[str]:
    """Slurm job directives with `nodes` added to the node exclusion: an existing `--exclude`
    (or `-x`) directive is widened, else one is added. Other directives are kept as they are."""
    import re

    pat = re.compile(r"^\s*(?:--exclude[= ]|-x\s*=?\s*)\s*(\S+)\s*$")
    out, have = [], []
    for d in directives:
        m = pat.match(d)
        if m:
            have += [n for n in m.group(1).split(",") if n]
        else:
            out.append(d)
    allx = list(dict.fromkeys(have + [n for n in nodes if n]))
    return out + ([f"--exclude={','.join(allx)}"] if allx else [])


def exclude_nodes(cl, nodes) -> None:
    """Keep a dask-jobqueue cluster's later worker jobs off `nodes` (Slurm `--exclude`): the
    cluster's job options are shared by its spec for new workers, so this reaches every job
    submitted after it, and none already submitted."""
    opts = [getattr(cl, "new_spec", {}).get("options")]
    opts += [s.get("options") for s in getattr(cl, "worker_spec", {}).values()]
    for o in {id(o): o for o in opts if isinstance(o, dict)}.values():
        if "job_extra_directives" in o:
            o["job_extra_directives"] = exclude_directives(o["job_extra_directives"] or [], nodes)


def restarts_due(times: list[float], now: float, limit: int, window: float) -> bool:
    """Whether a worker job registered at `times` (its first start, then each restart of its
    Dask process) has restarted `limit` or more times within the last `window` s."""
    return limit > 0 and sum(1 for t in times[1:] if now - t <= window) >= limit


def _restart_guard(cl, limit: int = 3, window: float = 1800.0, node_of=_slurm_node,
                   max_excluded: int | None = None, clock=time.time) -> None:
    """Cancel a worker job whose Dask process keeps getting killed and restarted by its nanny, and,
    on Slurm, keep this fit's later worker jobs off its node.

    OSCER H L3 (36464754, 2026-11-27): the four worker jobs on node c880 were each SIGKILLed
    within a minute of every start, 3, 2, 23 and 288 times over 4.5 h, losing 316 workers' finished
    results. Slurm marked none OUT_OF_MEMORY and no Dask memory warning was logged: OSCER does not
    enforce jobs' memory requests, so another job on the node that used more than it asked for ran
    it out of memory. A worker job counts each registration with the scheduler after its first as
    a restart; at `limit` within `window` s it is cancelled and a new one submitted. At most
    `max_excluded` nodes are excluded (default 3; `executor` passes a quarter of max_workers, at
    least 3), so a problem of the fit's own doesn't exclude the whole cluster."""
    import asyncio

    from distributed.diagnostics.plugin import SchedulerPlugin

    st = {"starts": {}, "excluded": [], "cancelled": 0, "tasks": set()}
    cap = 3 if max_excluded is None else max_excluded

    async def act(name, jid, n):
        try:
            node = None
            if jid and node_of is not None:
                node = await asyncio.get_running_loop().run_in_executor(None, node_of, jid)
            if node and node not in st["excluded"] and len(st["excluded"]) < cap:
                st["excluded"].append(node)
                exclude_nodes(cl, [node])
                where = f"keeping this fit's new workers off node {node}"
            elif node in st["excluded"]:
                where = f"node {node} is already excluded"
            else:
                where = (f"not excluding node {node}: {len(st['excluded'])} are already" if node
                         else "its node is unknown")
            log.warning("cluster: worker %s (job %s) restarted %d times in %.0f min, its process killed each "
                        "time (on a shared node, usually another job using more memory than it asked for); "
                        "cancelling it and asking for a new one, %s", name, jid or "-", n, window / 60, where)
            job = cl.workers.pop(name, None)
            if job is not None:
                await job.close()
            st["cancelled"] += 1
            await cl._correct_state()
        except Exception as e:  # noqa: BLE001 (a watcher must not take the cluster down)
            log.warning("cluster: cancelling crash-looping worker %s failed: %r", name, e)

    class Guard(SchedulerPlugin):
        name = "dfeols-restart-guard"

        def add_worker(self, scheduler=None, worker=None, **kw):
            ws = scheduler.workers.get(worker) if scheduler is not None else None
            if ws is None:
                return
            name = ws.name
            job = cl.workers.get(name)
            ident = None if job is None else (getattr(job, "job_id", None) or id(job))
            rec = st["starts"].get(name)
            # a new job under the same name (a replacement) starts a new count; a first
            # registration may come before the cluster records its job (LocalCluster's nanny)
            if rec is None or (ident is not None and rec["job"] is not None and rec["job"] != ident):
                rec = st["starts"][name] = {"job": ident, "times": [], "done": False}
            rec["job"] = rec["job"] if ident is None else ident
            if rec["done"]:
                return
            now = clock()
            times = rec["times"]
            times.append(now)
            if restarts_due(times, now, limit, window):
                rec["done"] = True
                n = sum(1 for t in times[1:] if now - t <= window)
                t = asyncio.ensure_future(act(name, getattr(job, "job_id", None), n))
                st["tasks"].add(t)
                t.add_done_callback(st["tasks"].discard)

    if limit <= 0:
        return
    cl.scheduler.add_plugin(Guard())
    cl.dfeols_restart_guard = st


def queue_wait(waits: list[float], prior: float, pending: float = 0.0) -> float:
    """The queue wait to plan for: the longest of the last 20 (submit to connect; `prior` until
    one is seen), or the age of the oldest worker still waiting if longer. A wait in progress
    counts: H L1 (OSCER 36391325, 2026-11-23) kept planning for the 14 min it had seen while its
    replacements waited 16 min to 3 h, and the pool fell to 6 of 40 for an hour."""
    return max(max(waits[-20:]) if waits else prior, pending)


def queue_lead(waits: list[float], prior: float, life: float, pending: float = 0.0) -> float:
    """Seconds before a worker's retirement to ask for its replacement: 1.25 x `queue_wait` plus
    5 min, at most half a worker's lifetime. The longest, since waits vary a lot (OSCER: 5-16 min
    for the first pool, 30-50 min for replacements) and asking early only costs some overlap."""
    return min(1.25 * queue_wait(waits, prior, pending) + 300.0, life / 2)


def retiring(connected: dict, now: float, life: float, stagger: float, lead: float,
             work_left: float | None, grace: float = 600.0, early_by: float = 0.0) -> set:
    """Names of connected workers (name -> time first seen) that retire at their lifetime within
    `lead` s and that the work outlasts: replacements to ask for now.

    A worker retires between lifetime - stagger and lifetime + stagger after it started; it
    counts from `lead` before the earliest until `grace` after the latest (then it is late, and
    adaptive scaling treats any extra as surplus). `work_left` None (no run reporting) counts as
    work to come. `early_by`: workers may retire up to this much sooner (the executor's
    retire_at_pass retires them before a pass of about this length, not during it)."""
    out = set()
    for n, t0 in connected.items():
        early, late = t0 + life - stagger - early_by, t0 + life + stagger + grace
        if early - lead <= now <= late and (work_left is None or work_left > max(early - now, 0) + 300):
            out.add(n)
    return out


SPARES_WAIT_S = 900.0  # queue waits up to this keep the cap at 10%; longer ones scale it


def spares_cap(max_workers: int, n: int | None = None, wait: float = 0.0) -> int:
    """Most replacements asked for ahead at once: `n` if given, else 10% of max_workers (at
    least 1), scaled by `wait` / 15 min once the queue wait is longer, up to half of
    max_workers. Dask draws each worker's lifetime at random within the window, so every worker
    of a pool that started together comes due at once: uncapped, H L2 (OSCER 36358879,
    2026-11-21) asked for 40 at 21:44 and ran 79 workers for 2 h, slower than 40
    (/scratch-bound). But a spare that starts early costs its overlap, linear in time, while a
    worker short for a long wait costs more the longer the wait (Ted, 2026-11-23): so long
    waits get more spares, which mostly wait in the queue rather than run."""
    if n is not None:
        return max(int(n), 0)
    base = max(int(round(0.1 * max_workers)), 1)
    scaled = int(math.ceil(base * max(1.0, wait / SPARES_WAIT_S)))
    return min(scaled, max(max_workers // 2, base))


def braked(waits: list[float], quick: float, k: int = 2, pending: float = 0.0) -> bool:
    """The last k queue waits were all within `quick` s, and no worker has been waiting longer:
    workers start at once, so none are asked for ahead (a retiring worker is replaced after it
    leaves)."""
    return len(waits) >= k and max(waits[-k:]) <= quick and pending <= quick


def _replace_ahead(cl, max_workers: int, prior_wait: float, clock=time.time,
                   max_spares: int | None = None, quick_wait: float = 180.0) -> None:
    """Add, to adaptive scaling's target, one worker for each worker about to retire at its
    lifetime while the work outlasts it, so the replacement waits in the queue while the old
    worker still runs; at most `spares_cap` at once, and none while the last two queue waits
    were within `quick_wait` s (`braked`). A worker still waiting counts as a wait in progress,
    so a slow queue lengthens the lead and raises the cap while it lasts. It is computed on each adapt tick, so the extra drops
    the moment the old worker leaves. Queue waits are measured per job (first in the plan to
    first connected)."""
    ad = getattr(cl, "_adaptive", None)
    life, stagger = getattr(cl, "dfeols_lifetime", (None, None))
    if ad is None or life is None or not hasattr(ad, "safe_target"):
        log.warning("cluster: cannot ask for replacements ahead of retirement (no adaptive scaling)")
        return
    orig = ad.safe_target
    st = {"planned": {}, "seen": {}, "waits": [], "extra": 0, "braked": False, "cap": 0}

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
            # None (no run reporting, or its total not known yet, e.g. in the singleton passes)
            # counts as work to come
            work_left = wl[1] - (now - wl[0]) if wl and wl[1] is not None and now - wl[0] < 600 else None
            waiting = [now - st["planned"][x] for x in plan - seen if x in st["planned"]]
            pending = max(waiting, default=0.0)  # the oldest worker still in the queue
            lead = queue_lead(st["waits"], prior_wait, life, pending)
            cap = spares_cap(max_workers, max_spares, queue_wait(st["waits"], prior_wait, pending))
            # retire_at_pass retires a worker before a pass its lifetime would end in: up to a
            # pass sooner, and never for a pass over a quarter of its life (DaskExecutor)
            early_by = min(1.25 * float(getattr(cl, "dfeols_pass_seconds", 0.0) or 0.0), 0.25 * life)
            due = retiring({x: st["seen"][x] for x in seen if x in st["seen"]}, now, life, stagger, lead,
                           work_left, early_by=early_by)
            brake = braked(st["waits"], quick_wait, pending=pending)
            if brake != st["braked"]:
                log.info("cluster: the last 2 queue waits were %s %.0f min: %s", "within" if brake else "not all within",
                         quick_wait / 60, "replacements only after workers leave" if brake
                         else f"asking for replacements ahead again (at most {cap} at once)")
                st["braked"] = brake
            if cap > st["cap"] and st["cap"]:
                log.info("cluster: a worker has waited %.0f min in the queue: asking for replacements %.0f min "
                         "ahead, at most %d at once", pending / 60, lead / 60, cap)
            st["cap"] = cap
            extra = 0 if brake else min(len(due), cap, max_workers)
            target = n + extra
            ceil = getattr(cl, "dfeols_pool_ceiling", None)
            if ceil is not None:  # replacements only for workers that would take it below the ceiling
                ceil.observe(now, len(seen))
                target, extra = ceil.clip(n, extra, len(plan), len(due))
            if extra > st["extra"]:
                log.info("cluster: %d worker(s) retire within %.0f min; asking for %d replacement(s) now, "
                         "%d ahead in all (at most %d; queue waits so far up to %.0f min, oldest waiting %.0f min)",
                         len(due), lead / 60, extra - st["extra"], extra, cap,
                         max(st["waits"], default=prior_wait) / 60, pending / 60)
            st["extra"] = extra
            return target
        except Exception as e:  # noqa: BLE001 (never take adaptive scaling down)
            log.warning("cluster: replacing retiring workers ahead failed: %r", e)
            return n

    ad.safe_target = safe_target
    cl.dfeols_replace_ahead = st


def _progress_passes() -> dict | None:
    from . import progress

    return progress.PASSES


class PoolCeiling:
    """A soft ceiling on the pool once more workers stop adding rows/s (the shared filesystem
    saturates).

    D11 H on OSCER (/scratch-bound) ran at 26-27M rows/s with 40 workers, ~22M with 74-79 and
    28-30M with 23-24: past ~25-30 workers, more of them wait on reads and add contention.

    Each adapt tick `observe` notes the connected workers and the latest progress counters
    (progress.PASSES). Within one pass type (CG steps share one), a window of `window_s`
    becomes a sample (mean workers, rows/s) when the pool held steady (within 10%, or 1
    worker), at least MIN_TASKS tasks finished in it and tasks were waiting to run for 80% of
    it (not a stage's tail). The pool drifts on its own as workers retire and replacements
    arrive, so nothing is probed. When a sample with at least `larger` more workers than
    another of the same pass type, within `pair_s` of it, gained under `gain` rows/s, the
    smaller pool (at least `min_workers`) becomes the ceiling for the rest of the fit: workers
    retiring above it aren't replaced, and nothing is cancelled (`clip`). It comes off when a
    sample at or below it beats the saturated rate by `lift` (the filesystem got faster: other
    jobs on it ended), or a new fit starts."""

    MIN_TASKS = 20
    STEADY = 0.10
    BUSY = 0.8
    STALE_S = 180.0  # progress counters older than this: no run reporting

    def __init__(self, min_workers: int = 1, larger: float = 0.20, gain: float = 0.05,
                 window_s: float = 600.0, lift: float = 0.25, pair_s: float = 7200.0, passes=None):
        self.min_workers = max(int(min_workers), 1)
        self.larger, self.gain, self.window_s, self.lift, self.pair_s = larger, gain, window_s, lift, pair_s
        self._passes = passes or _progress_passes
        self.ceiling: int | None = None
        self.saturated: tuple[str, float] | None = None  # (pass type, rows/s) when it was set
        self.samples: list[dict] = []
        self.window: dict | None = None
        self.run = None

    # --- samples -----------------------------------------------------------
    def observe(self, now: float, workers: int) -> None:
        ps = self._passes()
        if not ps or now - ps["t"] > self.STALE_S or ps.get("stage") is None:
            self.window = None
            return
        if ps["run"] != self.run:  # a new fit: what the last one measured no longer applies
            if self.ceiling is not None:
                log.info("cluster: IO ceiling of %d workers lifted: a new fit started", self.ceiling)
            self.run, self.ceiling, self.saturated, self.samples, self.window = ps["run"], None, None, [], None
        kind = ps["stage"]
        rows, tasks = ps["kinds"].get(kind, (0, 0))
        w = self.window
        if w is None or w["kind"] != kind or rows < w["rows0"]:
            self.window = {"kind": kind, "t0": ps["t"], "rows0": rows, "tasks0": tasks, "t": ps["t"],
                           "rows": rows, "tasks": tasks, "obs": [], "busy": 0}
            w = self.window
        w["obs"].append(int(workers))
        w["busy"] += ps.get("queued", 0) > 0
        w["t"], w["rows"], w["tasks"] = ps["t"], rows, tasks
        if w["t"] - w["t0"] < self.window_s:
            return
        self.window = {"kind": kind, "t0": w["t"], "rows0": rows, "tasks0": tasks, "t": w["t"],
                       "rows": rows, "tasks": tasks, "obs": [], "busy": 0}
        s = self._sample(w, now)
        if s is not None:
            self.add(s)

    def _sample(self, w: dict, now: float) -> dict | None:
        obs = w["obs"]
        mean = sum(obs) / len(obs)
        if (mean < 1 or max(obs) - min(obs) > max(1.0, self.STEADY * mean)
                or w["tasks"] - w["tasks0"] < self.MIN_TASKS or w["busy"] < self.BUSY * len(obs)):
            return None
        return {"t": now, "kind": w["kind"], "workers": mean, "rate": (w["rows"] - w["rows0"]) / (w["t"] - w["t0"])}

    def add(self, s: dict) -> None:
        """Take one sample (also for tests): lift the ceiling, or set or lower it."""
        self.samples = [x for x in self.samples if s["t"] - x["t"] <= self.pair_s] + [s]
        if self.ceiling is not None and self.saturated and s["kind"] == self.saturated[0] \
                and s["workers"] <= self.ceiling + 0.5 and s["rate"] > (1 + self.lift) * self.saturated[1]:
            log.info("cluster: IO ceiling of %d workers lifted: %.0f workers now give %.1fM rows/s in %s, "
                     "over %.0f%% above the %.1fM it saturated at", self.ceiling, s["workers"], s["rate"] / 1e6,
                     s["kind"], 100 * self.lift, self.saturated[1] / 1e6)
            self.ceiling, self.saturated, self.samples = None, None, [s]
            return
        best = None
        same = [x for x in self.samples if x["kind"] == s["kind"]]
        for a in same:
            for b in same:
                if b["workers"] >= (1 + self.larger) * a["workers"] and b["rate"] < (1 + self.gain) * a["rate"]:
                    c = max(int(round(a["workers"])), self.min_workers)
                    if best is None or c < best[0]:
                        best = (c, a, b)
        if best is None or (self.ceiling is not None and best[0] >= self.ceiling):
            return
        c, a, b = best
        log.info("cluster: IO ceiling %d workers: in %s, %.0f workers gave %.1fM rows/s and %.0f gave %.1fM "
                 "(%.0f%%+ more workers, under %.0f%% more rows/s); workers retiring above %d aren't replaced",
                 c, s["kind"], a["workers"], a["rate"] / 1e6, b["workers"], b["rate"] / 1e6,
                 100 * self.larger, 100 * self.gain, c)
        self.ceiling, self.saturated = c, (s["kind"], max(a["rate"], b["rate"]))

    # --- the target ----------------------------------------------------------
    def clip(self, n: int, extra: int, planned: int, due: int) -> tuple[int, int]:
        """Adaptive scaling's target under the ceiling: (target, replacements ahead kept).

        `n` is adapt's own target, `extra` the replacements replace-ahead asks for ahead,
        `planned` the workers asked for (running or queued), `due` those about to retire.
        Never below `planned` because of the ceiling (nothing is cancelled; the pool shrinks as
        workers retire), and replacements ahead only for the ones that would take it below. The
        second value is the replacements ahead the target holds (0 while the pool is still above
        the ceiling plus them), for replace-ahead's log. A target of adapt's at or below the
        ceiling (scaling down while idle) is left alone."""
        lim = max(self.ceiling or 0, self.min_workers)
        if self.ceiling is None or n <= lim:
            return n + extra, extra
        keep = min(extra, max(0, lim - (planned - due)))
        target = max(lim + keep, min(planned, n + extra))
        return target, (keep if lim + keep >= planned else 0)


def _pool_ceiling(cl, p_worker: dict, clock=time.time, passes=None) -> PoolCeiling | None:
    """Install a PoolCeiling on an adaptive cluster from the profile's worker settings. With
    replace-ahead installed it consults the ceiling on its own tick; otherwise this wraps
    adaptive scaling's target itself."""
    ad = getattr(cl, "_adaptive", None)
    if ad is None or not hasattr(ad, "safe_target"):
        log.warning("cluster: cannot set an IO ceiling on the pool (no adaptive scaling)")
        return None
    w = p_worker
    ceil = PoolCeiling(
        min_workers=w.get("min_workers", 1),
        larger=float(w.get("io_ceiling_larger", 0.20)),
        gain=float(w.get("io_ceiling_gain", 0.05)),
        window_s=60 * float(w.get("io_ceiling_window_minutes", 10)),
        lift=float(w.get("io_ceiling_lift", 0.25)),
        pair_s=60 * float(w.get("io_ceiling_pair_minutes", 120)),
        passes=passes,
    )
    cl.dfeols_pool_ceiling = ceil
    if hasattr(cl, "dfeols_replace_ahead"):
        return ceil
    orig = ad.safe_target

    async def safe_target():
        n = await orig()
        try:
            ceil.observe(clock(), len(set(ad.observed)))
            return ceil.clip(n, 0, len(set(ad.plan)), 0)[0]
        except Exception as e:  # noqa: BLE001 (never take adaptive scaling down)
            log.warning("cluster: IO ceiling failed: %r", e)
            return n

    ad.safe_target = safe_target
    return ceil


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
