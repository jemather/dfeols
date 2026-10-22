"""Backward-compatible Slurm helpers. New code should use site profiles:

    ex = dfeols.executor("mycluster")      # see dfeols.site / `dfeols config init`
"""

from __future__ import annotations

from .cluster import fix_scheduler_env, private_interface, walltime_seconds  # noqa: F401
from .site import DEFAULTS, Profile


def _profile_from_kwargs(*, cores=8, memory="64GB", walltime="24:00:00", queue=None, account=None,
                         local_directory=None, retire_minutes=60, job_extra_directives=None,
                         job_script_prologue=None, interface=None, scheduler_interface="auto",
                         max_workers=20, min_workers=1, **_):
    return Profile(
        name="slurm-kwargs", scheduler="slurm", queue=queue, account=account, project=None,
        scratch=None, prologue=list(job_script_prologue or []),
        interface=interface or scheduler_interface,
        extra_directives=list(job_extra_directives or []), env={},
        worker={**DEFAULTS["worker"], "cores": cores, "memory": memory, "walltime": walltime,
                "retire_minutes": retire_minutes, "local_directory": local_directory,
                "max_workers": max_workers, "min_workers": min_workers},
        coordinator=dict(DEFAULTS["coordinator"]))


def make_cluster(**kw):
    from .cluster import make_cluster as _mk
    return _mk(_profile_from_kwargs(**kw), kw.get("log_directory"))


def slurm_executor(*, max_workers: int, min_workers: int = 1, tick: float = 10.0,
                   log_directory=None, **kw):
    """Deprecated: use dfeols.executor(profile). Returns (executor, client, cluster)."""
    from .cluster import executor
    ex = executor(_profile_from_kwargs(max_workers=max_workers, min_workers=min_workers, **kw),
                  tick=tick, log_directory=log_directory)
    return ex, ex.client, ex.cluster
