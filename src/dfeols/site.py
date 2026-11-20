# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Site profiles: everything that differs between HPC systems, in one file.

dfeols looks for a TOML config at $DFEOLS_CONFIG, else
~/.config/dfeols/config.toml. Each [profiles.<name>] table describes one
system (or one way of using it). `dfeols config init` writes a template.

    default = "mycluster"

    [profiles.mycluster]
    scheduler = "slurm"            # slurm | pbs | sge | lsf | local
    queue = "normal"               # partition / queue
    account = ""                   # optional
    scratch = "/scratch/$USER/dfeols"
    prologue = ["module load Python/3.12", "source $HOME/envs/dfeols/bin/activate"]
    interface = "auto"             # coordinator network interface ("auto" = private/internal;
                                   # "ib" = every job's InfiniBand address, coordinator and workers)
    extra_directives = []          # e.g. ["--qos=long"] (Slurm) or ["-l place=scatter"] (PBS)
    record_level = 5               # estimation records' disclosure level, 1-5 (default 4);
                                   # 5 keeps the scheduler, timing, hosts and every package

    [profiles.mycluster.worker]
    cores = 8
    memory = "32GB"
    walltime = "12:00:00"
    max_workers = 20
    min_workers = 1
    retire_minutes = 60
    idle_minutes = 15

    [profiles.mycluster.coordinator]
    cores = 4
    memory = "16GB"
    walltime = "7-00:00:00"

Everything is overridable per call, e.g. dfeols.executor("mycluster", max_workers=50).
Environment variables ($USER, $HOME, ...) in strings are expanded.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore

SCHEDULERS = ("slurm", "pbs", "sge", "lsf", "local")

DEFAULTS = {
    "scheduler": "local",
    "queue": None,
    "account": None,
    "project": None,
    "scratch": None,
    "prologue": [],
    "interface": "auto",
    "extra_directives": [],
    "env": {},
    "record_level": None,  # estimation records' disclosure level (estimation spec 4.9); None: 4
    "worker": {
        "cores": 8,
        "memory": "32GB",
        "walltime": "12:00:00",
        "max_workers": 20,
        "min_workers": 1,
        "retire_minutes": 60,
        "idle_minutes": 15,
        # how long a worker job's Dask process may take to start and reach the scheduler before
        # it gives up (dask-jobqueue's death-timeout, 60 s; on OSCER 16 of 40 workers starting
        # together from the shared filesystem missed it, 2026-11-19)
        "start_minutes": 10,
        # ask for a replacement this long (in queue waits seen so far) before a worker retires at
        # its lifetime, while the fit still has work past that point; false: only after it leaves
        # (on OSCER all 40 workers retired together and the pool took ~50 min to refill, 2026-11-20)
        "replace_ahead": True,
        # workers retire over lifetime minus up to 2 x this (None: 10% of the lifetime, at
        # least 5 min), so a pool that started together doesn't retire together
        "retire_spread_minutes": None,
        # times a task may be on a worker that leaves before Dask fails it (KilledWorker) on the
        # next; None: half of max_workers, at least Dask's 3, which counted planned retirements
        # and slow connections and failed H L2 (OSCER, 2026-11-20)
        "allowed_failures": None,
        "queue_wait_minutes": 30,  # the queue wait assumed until a worker job's own wait is seen
        "local_directory": None,
    },
    # queue: the coordinator's own partition (None: the profile's queue, as the workers);
    # extra_directives: its own, in place of the profile's (None: the profile's)
    "coordinator": {"cores": 4, "memory": "16GB", "walltime": "7-00:00:00", "queue": None,
                    "extra_directives": None},
    # site facts for `dfeols plan` (docs/design/planner.md); `dfeols config init --from-slurm`
    # drafts them
    "max_walltime": None,  # longest general job; a node class may override it
    "billing": None,  # none | cores | nodes | tres
    "scratch_purge_days": None,
    # persisted demeaned data (planner.md, Ted 2026-10-28): "plentiful" never asks; "confirm"
    # asks before `session start` writes more than demeaned_confirm_gb
    "demeaned_storage": "plentiful",
    "demeaned_confirm_gb": None,
    "nodes": {},  # node classes: {name: {partition, count, in_service, cores, memory, ...}}
}

TEMPLATE = """# dfeols site configuration. One [profiles.<name>] table per HPC system.
# Select a profile with dfeols.executor("<name>"), --profile <name>,
# $DFEOLS_PROFILE, or the `default` key below.

default = "cluster"

[profiles.local]
scheduler = "local"            # Dask workers on this machine
[profiles.local.worker]
cores = 1                      # cores per worker process
max_workers = 4                # number of worker processes
memory = "8GB"

[profiles.cluster]
scheduler = "slurm"            # slurm | pbs | sge | lsf
queue = "normal"               # partition (Slurm) / queue (PBS, SGE, LSF)
account = ""                   # allocation/account, if your site requires one
scratch = "/scratch/$USER/dfeols"   # shared filesystem for shards, caches, status
# commands run at the top of every job script, before Python starts
prologue = [
  "module load python",
  "source $HOME/envs/dfeols/bin/activate",
]
interface = "auto"             # coordinator NIC; "auto" picks a private (internal) address;
                               # "ib": coordinator and workers each use their node's InfiniBand
                               # (IPoIB) address, for clusters whose default NIC is slow Ethernet
extra_directives = []          # raw scheduler directives, e.g. ["--qos=long"]
# disclosure level of estimation records, 1-5 (default 4: no scheduler, timing, hosts or
# package list); 5 keeps them, for diagnosing runs on this system
# record_level = 5
# persisted demeaned data: "plentiful" (default) never asks; "confirm" makes `session start`
# ask before writing more than demeaned_confirm_gb (default 500) where scratch is scarce
# demeaned_storage = "confirm"
# demeaned_confirm_gb = 500
# scratch_purge_days = 14        # the site deletes scratch files this many days old

[profiles.cluster.worker]
cores = 8
memory = "32GB"
walltime = "12:00:00"          # shorter jobs usually start sooner
max_workers = 20
min_workers = 1
retire_minutes = 60            # stop taking shards this long before walltime
idle_minutes = 15              # release a worker only after this long with nothing to do
start_minutes = 10             # a worker job's Dask process may take this long to reach the scheduler
replace_ahead = true           # ask for replacements before workers retire, by the queue waits seen
queue_wait_minutes = 30        # the queue wait assumed until one is seen
# retire_spread_minutes = 60    # workers retire over this x 2 before their lifetime (default 10% of it)
# allowed_failures = 20        # a task fails on the next worker leaving under it (default max_workers / 2, at least 3)

[profiles.cluster.coordinator]
cores = 4
memory = "16GB"
walltime = "7-00:00:00"
# queue = "long"               # its own partition; default: the profile's queue
# extra_directives = []        # in place of the profile's, e.g. without a node constraint
"""


def config_path() -> Path:
    p = os.environ.get("DFEOLS_CONFIG")
    return Path(p).expanduser() if p else Path.home() / ".config" / "dfeols" / "config.toml"


def _expand(v):
    if isinstance(v, str):
        return os.path.expandvars(os.path.expanduser(v))
    if isinstance(v, list):
        return [_expand(x) for x in v]
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    return v


def _apply_overrides(d: dict, overrides: dict) -> dict:
    # flat overrides: worker keys may be given at top level for convenience
    for k, v in overrides.items():
        if v is None:
            continue
        if k in DEFAULTS["worker"]:
            d["worker"][k] = v
        else:
            d[k] = v
    return d


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Profile:
    name: str
    scheduler: str
    queue: str | None
    account: str | None
    project: str | None
    scratch: str | None
    prologue: list[str]
    interface: str | None
    extra_directives: list[str]
    env: dict
    worker: dict
    coordinator: dict
    max_walltime: str | None = None
    billing: str | None = None
    scratch_purge_days: int | None = None
    demeaned_storage: str = "plentiful"
    demeaned_confirm_gb: float | None = None
    record_level: int | None = None
    nodes: dict = field(default_factory=dict)
    source: str = ""

    def scratch_dir(self, *parts) -> Path:
        base = Path(self.scratch) if self.scratch else Path.cwd() / "dfeols-scratch"
        return base.joinpath(*parts)

    def with_overrides(self, **overrides) -> Profile:
        """Copy of this profile with the same flat overrides profile() accepts."""
        if not any(v is not None for v in overrides.values()):
            return self
        d = asdict(self)
        name, source = d.pop("name"), d.pop("source")
        d = _expand(_apply_overrides(d, overrides))
        return Profile(name=name, source=source, **d)


def load_config(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else config_path()
    if not p.exists():
        return {}
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def profile(name: str | None = None, path=None, **overrides) -> Profile:
    """Load a site profile: how to reach the cluster's scheduler and what workers to ask for.

    Parameters
    ----------
    name : str, optional
        Profile name in the config. Default `$DFEOLS_PROFILE`, else the config's `default`,
        else the built-in `"local"` profile (also used when there is no config).
    path : str or path, optional
        Config file. Default `$DFEOLS_CONFIG`, else `~/.config/dfeols/config.toml`
        (`dfeols config init` writes one).
    **overrides
        Profile keys to change for this call, e.g. `max_workers=40`.

    Returns
    -------
    Profile

    Raises
    ------
    KeyError
        If `name` is not in the config.
    """
    cfg = load_config(path)
    profiles = cfg.get("profiles", {})
    name = name or os.environ.get("DFEOLS_PROFILE") or cfg.get("default")
    if name is None:
        name = "local"
    if name not in profiles and name != "local":
        where = path or config_path()
        raise KeyError(
            f"profile {name!r} not found in {where} (have: {sorted(profiles) or 'none'}); "
            "run `dfeols config init` to create a config"
        )
    d = _expand(_apply_overrides(_merge(DEFAULTS, profiles.get(name, {})), overrides))
    if d["scheduler"] not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {d['scheduler']!r}")
    if d.get("record_level") is not None and d["record_level"] not in (1, 2, 3, 4, 5):
        raise ValueError(f"record_level must be 1 to 5, got {d['record_level']!r}")
    if d.get("demeaned_storage") not in ("plentiful", "confirm"):
        raise ValueError(f"demeaned_storage must be 'plentiful' or 'confirm', got {d['demeaned_storage']!r}")
    if d["demeaned_storage"] == "confirm" and d.get("demeaned_confirm_gb") is None:
        d["demeaned_confirm_gb"] = 500.0
    for k in ("account", "queue", "project"):
        if d.get(k) == "":
            d[k] = None
    known = set(Profile.__dataclass_fields__) - {"name", "source"}
    return Profile(name=name, source=str(path or config_path()), **{k: v for k, v in d.items() if k in known})


def init_config(path=None, force: bool = False) -> Path:
    p = Path(path) if path else config_path()
    if p.exists() and not force:
        raise FileExistsError(f"{p} exists (use --force to overwrite)")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(TEMPLATE)
    return p
