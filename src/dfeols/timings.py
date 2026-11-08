# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""The local record of earlier runs and the wall-time estimates made from it (planner build
steps 3 and 4, docs/design/planner.md).

Every `fit` appends a line to `<scratch>/timings/fits.jsonl` (the default site profile's
scratch, or `DFEOLS_TIMINGS`; nothing is written when neither names a folder): the path,
rows, columns, passes over the data, seconds and workers. A run's rate is
rows x columns x passes / (seconds x workers); an estimate is the planned work over the
range of earlier rates on the same path (all paths when it has none), times the workers the
fit will have. Session builds keep their own record (`<scratch>/demeaned/timings.jsonl`),
which `estimate_demean` reads.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("dfeols")

FITS = "fits.jsonl"
DEFAULT_SINGLETON_PASSES = (1, 4)  # with crossing FEs and no history: the passes before round 1
DEMEAN_FIRST_SHARE = 0.5  # of the walltime limit (Ted, 2026-10-28)


def history_dir(profile=None) -> Path | None:
    """Where the record lives: DFEOLS_TIMINGS, else the profile's scratch (None if unset)."""
    if env := os.environ.get("DFEOLS_TIMINGS"):
        return Path(env)
    try:
        if profile is None or not hasattr(profile, "scratch"):
            from .site import profile as load_profile

            profile = load_profile(profile)
    except Exception:
        return None
    return Path(profile.scratch) / "timings" if profile.scratch else None


def log_fit(entry: dict, profile=None) -> Path | None:
    """Append one run; never fails a fit."""
    d = history_dir(profile)
    if d is None:
        return None
    try:
        d.mkdir(parents=True, exist_ok=True)
        e = {"when": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **entry}
        with open(d / FITS, "a") as f:
            f.write(json.dumps(e) + "\n")
        return d / FITS
    except OSError as err:
        log.debug("timings: not recorded (%s)", err)
        return None


def history(profile=None) -> list[dict]:
    d = history_dir(profile)
    if d is None or not (d / FITS).exists():
        return []
    out = []
    for line in (d / FITS).read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("seconds", 0) > 0 and e.get("rows") and e.get("columns") and e.get("passes"):
            out.append(e)
    return out


def rate(e: dict) -> float:
    """Row-columns per second per worker, each pass counted."""
    return e["rows"] * e["columns"] * e["passes"] / (e["seconds"] * max(int(e.get("workers") or 1), 1))


def estimate_fit(
    rows: int,
    columns: int,
    path: str,
    *,
    solve: str | None = None,
    crossing: bool,
    workers: int,
    profile=None,
    runs: list[dict] | None = None,
) -> dict | None:
    """A range of seconds for one fit, from earlier runs; None with no history.

    Passes: two (the cross-products, then the residuals) plus the singleton passes (with
    crossing FEs; their range from earlier runs that had them, else 1-4) plus, on plan C's
    fallback, a pass per CG step (from earlier fallback runs; unknown without)."""
    runs = history(profile) if runs is None else runs
    if not runs:
        return None
    same = [e for e in runs if e.get("path") == path and (e.get("solve") == solve or solve is None)]
    use = same or runs
    rates = [rate(e) for e in use]
    sing = [e.get("singleton_passes", 0) for e in runs if e.get("crossing")]
    s_lo, s_hi = (min(sing), max(sing)) if sing else DEFAULT_SINGLETON_PASSES
    if not crossing:
        s_lo = s_hi = 0
    p_lo, p_hi = 2 + s_lo, 2 + s_hi
    unknown = None
    if solve == "fallback":
        steps = [e["cg_steps"] for e in runs if e.get("solve") == "fallback" and e.get("cg_steps")]
        if steps:
            p_lo, p_hi = p_lo + min(steps), p_hi + max(steps)
        else:
            unknown = "plus a pass per CG step (no earlier fallback run to say how many)"
    work = rows * columns
    w = max(int(workers or 1), 1)
    lo, hi = work * p_lo / (max(rates) * w), work * p_hi / (min(rates) * w)
    basis = (
        f"{len(use)} earlier {'run' if len(use) == 1 else 'runs'}"
        + (f" on this path ({path})" if same else " (none on this path)")
        + f", scaled to {w} worker{'s' if w > 1 else ''}; {p_lo}-{p_hi} passes"
    )
    return {"low": lo, "high": hi, "passes": [p_lo, p_hi], "basis": basis, "unknown": unknown}


def estimate_demean(rows: int, variables: int, profile=None) -> tuple[float, float] | None:
    """A session build's range (sessions' own record), or None."""
    from .session import demeaned_root, estimate_seconds

    try:
        return estimate_seconds(demeaned_root(profile), rows, variables)
    except Exception:
        return None


def shape(option: dict, est: dict | None, *, specs: int, max_walltime: float | None) -> dict:
    """One job or demean-first (docs/design/planner.md, Ted 2026-10-28), with the reasons."""
    why = []
    if option.get("solve") in ("level", "fallback", "maybe") or option.get("path") == "iterate":
        why.append("the crossing FEs are solved iteratively (plan C): demean once for every specification")
    if est and max_walltime and est["high"] > DEMEAN_FIRST_SHARE * max_walltime:
        why.append(
            f"the estimate's high end is over {DEMEAN_FIRST_SHARE:.0%} of the walltime limit "
            f"({_hours(max_walltime)}): demeaned data persist, so a new job resumes"
        )
    if specs > 1:
        why.append(f"{specs} specifications share these FEs: demean once, fit each from a session")
    return {"shape": "demean-first" if why else "one job", "why": why}


def _hours(s: float) -> str:
    return f"{s / 3600:.0f} h" if s >= 3600 else f"{s / 60:.0f} min"


def walltime_seconds(text: str | None) -> float | None:
    """A profile's walltime ('2-00:00:00') in seconds, or None."""
    if not text:
        return None
    from .cluster import walltime_seconds as _ws

    return float(_ws(str(text)))
