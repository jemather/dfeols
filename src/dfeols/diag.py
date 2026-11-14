# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Diagnostics for distributed runs: what the cluster and the coordinator were doing.

Job 35991256 (D11 R1, L2) spent 26 minutes in round 1 against L1's 2, with workers idle most
of the time, and nothing recorded showed why: no scheduler log and no per-task timings. A
DaskExecutor run now writes, when it has somewhere to write:

<fit workdir>/tasks.jsonl
    one line per task: stage, shard, when the coordinator took the result, the task's own
    timing (total seconds, read, peak RSS ...), seconds fetching the result and seconds in
    on_result (the coordinator's serial share). CG steps after the second are summarised
    only, so a 500-step fit stays a few MB. At each stage's end, a summary line: tasks, wall
    seconds, summed task seconds, mean tasks running at once (sum / wall), task-time
    quantiles, coordinator seconds.
<fit workdir>/cluster.jsonl
    every DFEOLS_DIAG_SECONDS (30) during a run: per worker the tasks assigned, threads,
    process and managed memory; the scheduler's task states (queued, processing, memory =
    results held on workers) and the bytes they hold; the coordinator's backlog of finished
    results not yet taken.
<fit workdir>/scheduler_profile.jsonl
    where the scheduler's event loop spent its time, from Dask's own statistical profile of
    that thread (sampled every 10 ms; the client shares the loop when the cluster runs in the
    coordinator, as dfeols.cluster's does). A line per stage (round 1, singletons, cg1, cg2
    and every PROFILE_EVERY_CG-th step), and every PROFILE_SECONDS within a long stage: the
    samples and the PROFILE_TOP functions by samples inside them (inclusive) and by samples
    in them alone (self). Job 36016180 (D11 L3) logged 877 "event loop was unresponsive"
    stalls in 12 h (median 5 s, 81 min in all), mid-step, with nothing saying what ran.
<worker log folder>/scheduler.log
    the coordinator process's `distributed` and `dask_jobqueue` log (the scheduler runs
    there): workers joining and leaving, lost and retried tasks, event-loop stalls.

DFEOLS_DIAG=0 turns the three .jsonl files off.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

log = logging.getLogger("dfeols")

DETAIL_CG_STEPS = 2  # per-task lines for cg1..cg2; later CG stages get the summary only
SCHEDULER_LOG_BYTES = 40_000_000  # under scripts/oscer_fetch.sh's 50 MB cap, 3 kept
PROFILE_EVERY_CG = 10  # scheduler profile for cg1, cg2, cg10, cg20 ...
PROFILE_SECONDS = 600.0  # within a stage; Dask keeps ~1000 one-second profile cycles
PROFILE_TOP = 25


def enabled() -> bool:
    return os.environ.get("DFEOLS_DIAG", "1") not in ("0", "false", "no", "")


def snapshot_seconds() -> float:
    try:
        return float(os.environ.get("DFEOLS_DIAG_SECONDS", "30"))
    except ValueError:
        return 30.0


def attach_scheduler_log(log_dir) -> Path | None:
    """Send this process's `distributed` and `dask_jobqueue` logging (INFO and up) to
    <log_dir>/scheduler.log, rotated at SCHEDULER_LOG_BYTES. Idempotent per file."""
    from logging.handlers import RotatingFileHandler

    if not log_dir:
        return None
    path = Path(log_dir) / "scheduler.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("diag: no scheduler log (%s)", e)
        return None
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for name in ("distributed", "dask_jobqueue"):
        lg = logging.getLogger(name)
        if any(getattr(h, "baseFilename", None) == str(path.resolve()) for h in lg.handlers):
            continue
        h = RotatingFileHandler(path, maxBytes=SCHEDULER_LOG_BYTES, backupCount=3)
        h.setLevel(logging.INFO)
        h.setFormatter(fmt)
        lg.addHandler(h)
        if lg.getEffectiveLevel() > logging.INFO:
            lg.setLevel(logging.INFO)
    log.info("cluster: scheduler log in %s", path)
    return path


def scheduler_snapshot(dask_scheduler=None) -> dict:
    """Runs on the scheduler (client.run_on_scheduler): workers and task states."""
    s = dask_scheduler
    states: Counter = Counter()
    held = 0
    for ts in s.tasks.values():
        states[ts.state] += 1
        if ts.state == "memory":
            held += int(getattr(ts, "nbytes", 0) or 0)
    workers = []
    for ws in s.workers.values():
        mem = getattr(ws, "memory", None)
        workers.append(
            {
                "name": str(ws.name),
                "host": ws.host,
                "threads": ws.nthreads,
                "assigned": len(ws.processing),
                "executing": len(getattr(ws, "executing", ()) or ()),
                "process_gb": round(getattr(mem, "process", 0) / 1e9, 2) if mem else None,
                "managed_gb": round(getattr(mem, "managed", 0) / 1e9, 2) if mem else None,
                "results": len(ws.has_what),
            }
        )
    return {
        "states": dict(states),
        "held_gb": round(held / 1e9, 3),
        "queued": len(getattr(s, "queued", ()) or ()),
        "unrunnable": len(getattr(s, "unrunnable", ()) or ()),
        "workers": workers,
    }


def _detail(stage: str) -> bool:
    if stage.startswith("cg") and stage[2:].isdigit():
        return int(stage[2:]) <= DETAIL_CG_STEPS
    return True


def _profiled(stage: str) -> bool:
    if stage.startswith("cg") and stage[2:].isdigit():
        n = int(stage[2:])
        return n <= DETAIL_CG_STEPS or n % PROFILE_EVERY_CG == 0
    return True


def profile_top(tree: dict, top: int = PROFILE_TOP) -> dict:
    """Dask's profile tree (distributed.profile: each node's count = the samples whose stack
    passes through it) to the functions with the most samples: inclusive (a function counted
    once per sample, however often it recurs in the stack) and self (the sample's own frame).
    Functions are file:name, so a function's lines add up. Inclusive leaves out the functions
    in every sample (the thread's bootstrap, the loop runner), which say nothing."""
    incl: Counter = Counter()
    own: Counter = Counter()
    stack = [(tree, frozenset())]
    while stack:
        node, above = stack.pop()
        kids = list((node.get("children") or {}).values())
        n = int(node.get("count", 0) or 0)
        d = node.get("description") or {}
        name = f"{Path(str(d.get('filename', ''))).name}:{d.get('name', '')}"
        here = above
        if node.get("identifier") != "root":
            if name not in above:
                incl[name] += n
            here = above | {name}
            mine = n - sum(int(k.get("count", 0) or 0) for k in kids)
            if mine > 0:
                own[name] += mine
        stack.extend((k, here) for k in kids)
    total = int(tree.get("count", 0) or 0)
    incl = Counter({k: v for k, v in incl.items() if v < total})
    return {
        "samples": total,
        "inclusive": incl.most_common(top),
        "self": own.most_common(top),
    }


class RunLog:
    """One executor.run's diagnostics (see the module docstring). Every write is best effort:
    a diagnostic never stops a fit."""

    def __init__(self, client, progress, stage: str, n_tasks: int):
        wd = getattr(progress, "workdir", None)
        self.on = bool(wd) and enabled()
        self.client, self.stage, self.n = client, stage, n_tasks
        self.label = getattr(progress, "label", "")
        self.detail = _detail(stage)
        self.t0 = time.time()
        self.next_snap = 0.0
        self.every = snapshot_seconds()
        self.task_s: list[float] = []
        self.fetch_s = self.on_s = 0.0
        self.rss = 0.0
        self.hits = self.hit_known = 0
        self.read_s = 0.0
        self.since = {"taken": 0, "fetch_s": 0.0, "on_result_s": 0.0}
        self.prof = self.on and _profiled(stage)
        self.prof_t = self.t0
        if self.on:
            self.tasks_path = Path(wd) / "tasks.jsonl"
            self.cluster_path = Path(wd) / "cluster.jsonl"
            self.profile_path = Path(wd) / "scheduler_profile.jsonl"

    def _write(self, path: Path, rec: dict) -> None:
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
        except OSError as e:
            if self.on:
                log.warning("diag: stopped writing %s (%s)", path.name, e)
            self.on = False

    def task(self, tid, timing: dict | None, fetch_s: float, on_s: float) -> None:
        timing = timing or {}
        total = float(timing.get("total", 0.0) or 0.0)
        self.task_s.append(total)
        self.fetch_s += fetch_s
        self.on_s += on_s
        self.rss = max(self.rss, float(timing.get("maxrss_gb", 0.0) or 0.0))
        if "cache_hit" in timing:  # plan C steps: did the shard come from the worker's cache
            self.hits += bool(timing["cache_hit"])
            self.hit_known += 1
        self.read_s += float(timing.get("read", 0.0) or 0.0)
        self.since["taken"] += 1
        self.since["fetch_s"] += fetch_s
        self.since["on_result_s"] += on_s
        if self.on and self.detail:
            rec = {
                "t": round(time.time(), 3),
                "label": self.label,
                "stage": self.stage,
                "shard": tid,
                "fetch_s": round(fetch_s, 4),
                "on_result_s": round(on_s, 4),
                **{k: v for k, v in timing.items() if isinstance(v, (int, float, bool, str))},
            }
            self._write(self.tasks_path, rec)

    def profile(self, now: float | None = None) -> None:
        """The scheduler loop's profile since the last one, as a scheduler_profile.jsonl line."""
        if not (self.on and self.prof):
            return
        now = time.time() if now is None else now
        rec = {
            "t": round(now, 3),
            "label": self.label,
            "stage": self.stage,
            "from_s": round(self.prof_t - self.t0, 1),
            "to_s": round(now - self.t0, 1),
        }
        try:
            import dask
            from dask.utils import parse_timedelta

            rec.update(profile_top(self.client.profile(scheduler=True, start=self.prof_t, stop=now)))
            # an idle loop waits in selectors.py, which the profiler leaves out: the samples
            # are the loop's busy time
            every = parse_timedelta(dask.config.get("distributed.worker.profile.interval"))
            rec["busy_s"] = round(rec["samples"] * every, 2)
        except Exception as e:  # no profiler (Python 3.11, profiling off) or a busy scheduler
            rec["profile_error"] = repr(e)[:200]
            self.prof = False  # once per stage is enough to say why
        self.prof_t = now
        self._write(self.profile_path, rec)

    def tick(self, backlog_fn=None, force: bool = False) -> None:
        """A cluster snapshot if one is due. backlog_fn: finished results not yet taken."""
        if not self.on:
            return
        now = time.time()
        if self.prof and now - self.prof_t >= PROFILE_SECONDS:
            self.profile(now)
        if not force and now < self.next_snap:
            return
        self.next_snap = now + self.every
        rec = {
            "t": round(now, 3),
            "label": self.label,
            "stage": self.stage,
            "elapsed_s": round(now - self.t0, 1),
            "taken": len(self.task_s),
            "of": self.n,
            "coordinator": {k: round(v, 3) if isinstance(v, float) else v for k, v in self.since.items()},
        }
        self.since = {"taken": 0, "fetch_s": 0.0, "on_result_s": 0.0}
        try:
            rec["backlog"] = backlog_fn() if backlog_fn else None
        except Exception:
            rec["backlog"] = None
        try:
            rec.update(self.client.run_on_scheduler(scheduler_snapshot))
        except Exception as e:  # a busy or older scheduler: the coordinator's side still helps
            rec["scheduler_error"] = repr(e)[:200]
        self._write(self.cluster_path, rec)

    def close(self) -> dict:
        wall = time.time() - self.t0
        t = np.asarray(self.task_s) if self.task_s else np.zeros(1)
        summ = {
            "tasks": len(self.task_s),
            "wall_s": round(wall, 2),
            "task_s_sum": round(float(t.sum()), 2),
            "mean_running": round(float(t.sum()) / wall, 2) if wall > 0 else None,
            "task_s_p50": round(float(np.quantile(t, 0.5)), 3),
            "task_s_p90": round(float(np.quantile(t, 0.9)), 3),
            "task_s_max": round(float(t.max()), 3),
            "fetch_s": round(self.fetch_s, 2),
            "on_result_s": round(self.on_s, 2),
            "max_rss_gb": round(self.rss, 2),
            "read_s_sum": round(self.read_s, 2),
        }
        if self.hit_known:
            summ["cache_hits"] = self.hits
            summ["cache_known"] = self.hit_known
        if self.on:
            self.tick(force=True)
            self.profile()
            self._write(
                self.tasks_path,
                {"t": round(time.time(), 3), "label": self.label, "stage": self.stage, "summary": summ},
            )
        return summ
