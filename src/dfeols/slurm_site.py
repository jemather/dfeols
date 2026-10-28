# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""`dfeols config init --from-slurm`: draft a site profile's node classes from Slurm.

Hardware, limits and billing come from two commands any user can run on a login node:

    sinfo -h -e -o "%R|%D|%c|%m|%l|%f|%G|%T|%N"     one line per partition x node type x state
    scontrol show partition -o                      one line of key=value per partition

Partitions over the same nodes with the same limits (OSCER's normal, normal_avx2, all ...)
collapse to one class. Each class records its total nodes and the nodes in service (state
allocated, mixed or idle) when the draft was made. Policy partitions (GPU, a node minimum,
short limits, down or hidden) and partitions that only span other classes are written with
`use = false` and the reason; policy Slurm can't tell (whether the site charges, purges
scratch, preempts) is left to the user (docs/design/planner.md).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import date

SINFO_FORMAT = "%R|%D|%c|%m|%l|%f|%G|%T|%N"
SINFO_CMD = ["sinfo", "-h", "-e", "-o", SINFO_FORMAT]
SCONTROL_CMD = ["scontrol", "show", "partition", "-o"]

IN_SERVICE = {"allocated", "mixed", "idle", "completing"}
# feature names that say which network a node is on
_NET = re.compile(r"^(hdr\d*|ndr\d*|edr\d*|fdr\d*|qdr|opa|omnipath|ib|infiniband|roce)$", re.I)
SHORT_LIMIT_MINUTES = 4 * 60


def expand_hostlist(s: str) -> list[str]:
    """Slurm's compressed host list ("c[001-003,010],gpu01") as host names."""
    out: list[str] = []
    for part in _split_outside_brackets(s):
        out.extend(_expand_one(part))
    return out


def _split_outside_brackets(s: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        depth += ch == "["
        depth -= ch == "]"
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p for p in parts if p and p != "(null)"]


def _expand_one(s: str) -> list[str]:
    m = re.search(r"\[([^\]]+)\]", s)
    if not m:
        return [s]
    head, tail = s[: m.start()], s[m.end() :]
    out = []
    for rng in m.group(1).split(","):
        lo, _, hi = rng.partition("-")
        width = len(lo)
        for i in range(int(lo), int(hi or lo) + 1):
            out.extend(_expand_one(f"{head}{i:0{width}d}{tail}"))
    return out


def _minutes(limit: str) -> float:
    """Slurm time limit ([days-]hours:minutes:seconds, or minutes) in minutes."""
    if limit in ("UNLIMITED", "infinite", "INFINITE", "NONE", ""):
        return float("inf")
    days, _, rest = limit.rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    if len(parts) == 1:  # minutes
        h, mi, se = 0, parts[0], 0
    elif len(parts) == 2:  # minutes:seconds
        h, (mi, se) = 0, parts
    else:
        h, mi, se = parts
    return (int(days or 0) * 24 + h) * 60 + mi + se / 60


@dataclass
class Partition:
    name: str
    nodes: set = field(default_factory=set)
    in_service: set = field(default_factory=set)
    cores: set = field(default_factory=set)
    memory_mb: set = field(default_factory=set)
    features: set = field(default_factory=set)
    gres: set = field(default_factory=set)
    limit: str = ""
    conf: dict = field(default_factory=dict)  # scontrol key=value

    @property
    def default(self) -> bool:
        return self.conf.get("Default") == "YES"

    @property
    def max_time(self) -> str:
        return self.conf.get("MaxTime") or self.limit

    @property
    def max_mem_mb(self) -> int:
        """The most a job can ask for on every node: MaxMemPerNode, else the smallest node."""
        node = min(self.memory_mb) if self.memory_mb else 0
        cap = self.conf.get("MaxMemPerNode", "UNLIMITED")
        return min(node, int(cap)) if cap.isdigit() else node


def parse_sinfo(text: str) -> dict[str, Partition]:
    parts: dict[str, Partition] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        name, _n, cores, mem, limit, feats, gres, state, nodelist = line.strip().split("|")
        p = parts.setdefault(name, Partition(name))
        hosts = expand_hostlist(nodelist)
        p.nodes.update(hosts)
        # states carry flags (idle~ powered off, down*, mixed-, drained@ ...)
        if re.sub(r"[^a-z]", "", state.lower()) in IN_SERVICE:
            p.in_service.update(hosts)
        p.cores.add(int(cores.rstrip("+")))
        p.memory_mb.add(int(mem.rstrip("+")))
        p.features.update(f for f in feats.split(",") if f and f != "(null)")
        p.gres.update(g for g in gres.split(",") if g and g != "(null)")
        p.limit = limit
    return parts


def parse_scontrol(text: str) -> dict[str, dict]:
    out = {}
    for line in text.splitlines():
        kv = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
        if "PartitionName" in kv:
            out[kv["PartitionName"]] = kv
    return out


def _excluded(p: Partition) -> list[str]:
    why = []
    if p.conf.get("State", "UP") != "UP":
        why.append(f"state {p.conf['State']}")
    if p.conf.get("Hidden") == "YES":
        why.append("hidden")
    if any("gpu" in g.lower() for g in p.gres):
        why.append("GPU nodes")
    min_nodes = p.conf.get("MinNodes", "0")
    if min_nodes.isdigit() and int(min_nodes) > 1:
        why.append(f"jobs need at least {min_nodes} nodes")
    if _minutes(p.max_time) <= SHORT_LIMIT_MINUTES:
        why.append(f"time limit {p.max_time}")
    if not p.in_service:
        why.append("no nodes in service")
    return why


def node_classes(parts: dict[str, Partition], conf: dict[str, dict]) -> list[dict]:
    """One class per group of partitions with the same nodes, limits and policy, best first."""
    for name, p in parts.items():
        p.conf = conf.get(name, {})
    groups: dict[tuple, list[Partition]] = {}
    for p in parts.values():
        key = (frozenset(p.nodes), p.max_time, p.max_mem_mb, tuple(_excluded(p)))
        groups.setdefault(key, []).append(p)
    classes = []
    for members in groups.values():
        rep = sorted(members, key=lambda p: (not p.default, len(p.name), p.name))[0]
        nets = sorted({f for f in rep.features if _NET.match(f)})
        weights = rep.conf.get("TRESBillingWeights", "")
        classes.append(
            {
                "name": re.sub(r"[^A-Za-z0-9_-]", "_", rep.name),
                "partition": rep.name,
                "aliases": sorted(p.name for p in members if p is not rep),
                "count": len(rep.nodes),
                "in_service": len(rep.in_service),
                "cores": min(rep.cores),
                "memory": f"{rep.max_mem_mb}MB",
                "mixed": len(rep.cores) > 1 or len(rep.memory_mb) > 1,
                "interconnect": "/".join(nets),
                "features": sorted(f for f in rep.features if not _NET.match(f)),
                "max_walltime": rep.max_time,
                "billing_weights": weights,
                "default": any(p.default for p in members),
                "nodes": rep.nodes,
                "excluded": _excluded(rep),
            }
        )
    # a partition whose nodes all sit in smaller partitions with at least its time limit adds
    # nothing but a choice of node types Slurm makes (OSCER's normal over 2650, 2670, ...).
    # Not "uniform" classes: on OSCER a few nodes of each hardware partition report less
    # memory, so none is uniform. Excluded smaller partitions count: their nodes are no more
    # usable through the bigger one
    for c in classes:
        if c["excluded"] or not c["nodes"]:
            continue
        limit = _minutes(c["max_walltime"])
        cover = [
            u
            for u in classes
            if u["nodes"] and u["nodes"] < c["nodes"] and _minutes(u["max_walltime"]) >= limit
        ]
        left = c["nodes"] - set().union(*(u["nodes"] for u in cover))
        if not left:
            names = [u["name"] for u in sorted(cover, key=lambda u: (-u["count"], u["name"]))]
            more = f" and {len(names) - 6} more" if len(names) > 6 else ""
            c["excluded"].append("spans " + ", ".join(names[:6]) + more)
        elif cover and len(left) <= len(c["nodes"]) // 2:
            # mostly covered: say which nodes only this partition reaches, so the user can judge
            c["only_here"] = sorted(left)
    classes.sort(key=lambda c: (bool(c["excluded"]), c["mixed"], -c["count"], c["name"]))
    return classes


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def draft_profile(sinfo: str, scontrol: str, name: str = "slurm", today: date | None = None) -> str:
    """The [profiles.<name>] TOML a site profile starts from."""
    classes = node_classes(parse_sinfo(sinfo), parse_scontrol(scontrol))
    if not classes:
        raise ValueError("sinfo listed no partitions")
    today = today or date.today()
    usable = [c for c in classes if not c["excluded"]]
    site = next((c for c in classes if c["default"]), usable[0] if usable else classes[0])
    site_wall = site["max_walltime"]
    weighted = any(c["billing_weights"] for c in classes)
    lines = [
        f"# Drafted by `dfeols config init --from-slurm` on {today.isoformat()} from sinfo and scontrol.",
        "# Slurm gives the hardware and limits; check the policy lines (billing, scratch purge,",
        "# which classes to use) against the site's documentation.",
        f"[profiles.{name}]",
        'scheduler = "slurm"',
        f"queue = {_toml_str(site['partition'])}",
        f"max_walltime = {_toml_str(site_wall)}  # {site['partition']}'s limit",
        f'billing = "{"tres" if weighted else "cores"}"'
        + (
            "  # TRESBillingWeights are set"
            if weighted
            else '  # Slurm bills allocated cores; "none" if the site charges nothing'
        ),
        "# scratch_purge_days = 14  # if the site deletes old scratch files",
        "",
    ]
    for c in classes:
        lines.append(f"[profiles.{name}.nodes.{c['name']}]")
        if c["excluded"]:
            lines.append(f"use = false  # {'; '.join(c['excluded'])}")
        lines.append(f"partition = {_toml_str(c['partition'])}")
        if c["aliases"]:
            lines.append(f"# same nodes and limits: {', '.join(c['aliases'])}")
        if c.get("only_here"):
            h = c["only_here"]
            lines.append(
                f"# {len(h)} of its nodes are in no smaller partition with its time limit: "
                + ", ".join(h[:8])
                + (" ..." if len(h) > 8 else "")
            )
        lines.append(f"count = {c['count']}")
        lines.append(f"in_service = {c['in_service']}  # allocated, mixed or idle on {today}")
        lines.append(f"cores = {c['cores']}" + ("  # smallest; mixed nodes" if c["mixed"] else ""))
        lines.append(f'memory = "{c["memory"]}"')
        lines.append(f"interconnect = {_toml_str(c['interconnect'])}")
        if c["features"]:
            lines.append("features = [" + ", ".join(_toml_str(f) for f in c["features"]) + "]")
        if c["max_walltime"] != site_wall:
            lines.append(f"max_walltime = {_toml_str(c['max_walltime'])}")
        if c["billing_weights"]:
            lines.append(f"billing_weights = {_toml_str(c['billing_weights'])}")
        lines.append("")
    return "\n".join(lines)


def read_slurm() -> tuple[str, str]:
    """sinfo and scontrol output on this machine."""
    try:
        sinfo = subprocess.run(SINFO_CMD, capture_output=True, text=True, check=True).stdout
        scontrol = subprocess.run(SCONTROL_CMD, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError as e:
        raise RuntimeError(
            "sinfo/scontrol not found: run this on a Slurm login node, or pass saved output with "
            f"--sinfo and --scontrol (from `{' '.join(SINFO_CMD[:-1])} '{SINFO_FORMAT}'` and "
            f"`{' '.join(SCONTROL_CMD)}`)"
        ) from e
    return sinfo, scontrol
