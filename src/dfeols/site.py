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
    interface = "auto"             # coordinator network interface ("auto" = private/internal)
    extra_directives = []          # e.g. ["--qos=long"] (Slurm) or ["-l place=scatter"] (PBS)

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
    "worker": {
        "cores": 8,
        "memory": "32GB",
        "walltime": "12:00:00",
        "max_workers": 20,
        "min_workers": 1,
        "retire_minutes": 60,
        "idle_minutes": 15,
        "local_directory": None,
    },
    "coordinator": {"cores": 4, "memory": "16GB", "walltime": "7-00:00:00"},
    # site facts for `dfeols plan` (docs/design/planner.md); `dfeols config init --from-slurm`
    # drafts them
    "max_walltime": None,  # longest general job; a node class may override it
    "billing": None,  # none | cores | nodes | tres
    "scratch_purge_days": None,
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
interface = "auto"             # coordinator NIC; "auto" picks a private (internal) address
extra_directives = []          # raw scheduler directives, e.g. ["--qos=long"]

[profiles.cluster.worker]
cores = 8
memory = "32GB"
walltime = "12:00:00"          # shorter jobs usually start sooner
max_workers = 20
min_workers = 1
retire_minutes = 60            # stop taking shards this long before walltime
idle_minutes = 15              # release a worker only after this long with nothing to do

[profiles.cluster.coordinator]
cores = 4
memory = "16GB"
walltime = "7-00:00:00"
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
    """Load a site profile. Missing config -> the built-in 'local' profile."""
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
