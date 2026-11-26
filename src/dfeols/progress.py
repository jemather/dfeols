# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Progress and ETA for elastic runs.

The ETA model:
  * Each completed task gives worker-seconds per row. The stage rate is the
    ratio of sums (total task seconds / total rows), so big shards weigh more.
  * Remaining work = rows not yet done x rate, minus half the expected work of
    tasks currently running (their start times are not observed exactly).
  * Wall-clock ETA = remaining work / worker slots *currently* connected, and
    never less than the time of the largest task still waiting (a long shard
    can't be split across workers). Because slots are re-read on every tick,
    the ETA falls as Slurm starts more workers.
  * A band comes from bootstrapping completed tasks' rates.
  * Before round 2 starts, its cost is projected from round 1's timing
    breakdown (read/write + cross-product share), then replaced by measured
    round-2 rates once tasks finish.

Status is written atomically to <workdir>/status.json every `every` seconds, which
`dfeols status <workdir>` prints from any login node. How much of it reaches the log is set
by the verbosity (set_verbosity, or DFEOLS_VERBOSITY in the environment):

  0  no progress lines (warnings and the fit's own notes still show)
  1  (default) a line when the stage, CG step or worker state changes, and otherwise once
     the stage has moved LOG_PCT% (and MIN_GAP_S seconds have passed) or HEARTBEAT_S
     seconds have passed since the last line;
     between those, a live line redrawn every `every` seconds and wiped by the next line
  2  a line every `every` seconds
"""

from __future__ import annotations

import fcntl
import io
import json
import logging
import os
import stat
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("dfeols")

LOG_PCT = 5.0  # verbosity 1: a line once a stage has moved this % of its rows since the last
HEARTBEAT_S = 600.0  # ... or this many seconds have passed, whichever comes first
SETTLE_MAX_S = 300.0  # Progress.settle_frac holds a stage's times back at most this long
MIN_GAP_S = 60.0  # a LOG_PCT line waits until this long after the last: a short stage
# (5% every few seconds) gets its first and last lines and the live line, not a scroll
_VERBOSITY: int | None = None


def set_verbosity(level: int) -> None:
    """Progress lines in the log: 0 none, 1 milestones and a heartbeat (default), 2 every
    tick. Applies to runs started after the call; DFEOLS_VERBOSITY sets it for a process."""
    global _VERBOSITY
    if int(level) not in (0, 1, 2):
        raise ValueError(f"verbosity must be 0, 1 or 2, not {level!r}")
    _VERBOSITY = int(level)


def verbosity() -> int:
    if _VERBOSITY is not None:
        return _VERBOSITY
    try:
        v = int(os.environ.get("DFEOLS_VERBOSITY", "1"))
    except ValueError:
        return 1
    return min(max(v, 0), 2)


# The live line: verbosity 1 shows the latest status between log lines, in place.
#   * terminal: it ends in a newline (a prompt after it gets its own line); the next redraw
#     or record moves the cursor up onto it and erases it (LIVE_UP).
#   * regular file (Slurm's .out): it is the file's last line, overwritten in place with
#     os.pwrite, padded with spaces so a shorter line covers a longer one, and a log record
#     is written over it the same way. So `tail`, `cat`, `less` and `grep` see the milestone
#     lines and one current live line, with no control codes. `tail -f` prints only what is
#     appended, so it shows a live line once after each log line; `watch -n 30 tail <file>`
#     or `dfeols status --watch` show it move. If anything else wrote to the file since
#     (print, a warning), the old live line stays as it was and the next one is appended.
#   * pipe, a file opened to append (>>), or anything else: no live line.
# LiveAwareHandler, the package's default handler, does the replacing before each record.
LIVE_UP = "\033[A\r\033[K"
_live = None  # {"stream", "kind": "tty" | "file", and for a file "fd", "off", "len", "end"}


def _stream_kind(stream) -> str | None:
    try:
        if stream.isatty():
            return "tty"
        fd = stream.fileno()
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        # opened to append (>>, Slurm's --open-mode=append): Linux's pwrite ignores the
        # offset there and appends, so the live line would pile up; the log lines only
        return None if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_APPEND else "file"
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        return None


def _file_slot_free(lv: dict) -> bool:
    """The live line is still the file's end: nothing has been written after it."""
    try:
        lv["stream"].flush()
        return os.fstat(lv["fd"]).st_size == lv["end"]
    except (OSError, ValueError):
        return False


def _pwrite_slot(lv: dict, text: str) -> bool:
    """Write `text` (newline-terminated) over the live line, padded to cover it; the file's
    offset goes to its new end, where the next ordinary write lands."""
    fd = lv["fd"]
    body = text.rstrip("\n")
    data = (body + " " * max(lv["len"] - 1 - len(body.encode()), 0) + "\n").encode()
    try:
        os.pwrite(fd, data, lv["off"])
        end = max(lv["end"], lv["off"] + len(data))
        os.lseek(fd, 0, os.SEEK_END)
    except OSError:
        return False
    lv["len"], lv["end"] = max(lv["len"], len(data)), end
    return True


def _clear_live():
    """Before a write that bypasses the handler (a warning): a terminal's live line is
    erased; a file's stays as it is and the next one is appended after the warning."""
    global _live
    lv, _live = _live, None
    if lv is not None and lv["kind"] == "tty":
        try:
            lv["stream"].write(LIVE_UP * lv.get("n", 1))
            lv["stream"].flush()
        except (OSError, ValueError):
            pass


class LiveAwareHandler(logging.StreamHandler):
    def emit(self, record):
        global _live
        lv = _live
        if lv is not None and lv["kind"] == "file" and lv["stream"] is self.stream:
            _live = None
            if _file_slot_free(lv):
                try:
                    msg = self.format(record)
                except Exception:  # noqa: BLE001 (logging's own handleError reports it)
                    self.handleError(record)
                    return
                if _pwrite_slot(lv, msg + "\n"):
                    return
        else:
            _clear_live()
        super().emit(record)


def _live_target():
    """The stream the default handler writes to, or None when the user configured their
    own handlers."""
    for h in log.handlers:
        if isinstance(h, LiveAwareHandler):
            return getattr(h, "stream", None)
    return None


def _draw_live(stream, text: str, stamp: bool = True):
    """`stamp`: lead with the time, as the default handler's records do; off where the
    printed lines around it carry none (the session client), so the two look the same.
    On a terminal `text` may be several lines (the client's header and status row), all
    redrawn and erased together."""
    global _live
    line = (time.strftime("%H:%M:%S ") if stamp else "") + text + "\n"
    lv = _live if _live is not None and _live["stream"] is stream else None
    kind = lv["kind"] if lv else _stream_kind(stream)
    try:
        if kind == "tty":
            stream.write((LIVE_UP * lv.get("n", 1) if lv else "") + line)
            stream.flush()
            _live = {"stream": stream, "kind": "tty", "n": line.count("\n")}
        elif kind == "file":
            if lv is not None and _file_slot_free(lv) and _pwrite_slot(lv, line):
                return
            stream.flush()
            fd = stream.fileno()
            off = os.fstat(fd).st_size
            stream.write(line)
            stream.flush()
            end = os.fstat(fd).st_size
            _live = {"stream": stream, "kind": "file", "fd": fd, "off": off, "len": end - off, "end": end}
    except (OSError, ValueError):
        _live = None


# warnings (SlowConvergenceWarning, ...) go straight to stderr, not through the handler:
# a terminal's live line is erased first so a warning does not land on it
_showwarning = warnings.showwarning


def _showwarning_clear(*args, **kwargs):
    _clear_live()
    _showwarning(*args, **kwargs)


warnings.showwarning = _showwarning_clear


def fmt_td(s: float | None) -> str:
    if s is None or not np.isfinite(s):
        return "?"
    s = int(round(s))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return (f"{d}d " if d else "") + f"{h:d}:{m:02d}:{s:02d}"


def _clock(ts: float) -> str | None:
    """Wall-clock time for a finish estimate, or None if it is not representable."""
    try:
        t = time.localtime(ts)
        # today: the clock alone; later: with the weekday
        return time.strftime("%H:%M" if t[:3] == time.localtime()[:3] else "%a %H:%M", t)
    except (OverflowError, OSError, ValueError):
        return None


THROUGHPUT_WINDOW_S = 600.0
THROUGHPUT_MIN_TASKS = 10
THROUGHPUT_MIN_S = 60.0


@dataclass
class StageProgress:
    name: str
    rows: dict[int, int]  # task id -> rows
    done: dict[int, float] = field(default_factory=dict)  # task id -> seconds
    started: float | None = None
    finished: float | None = None
    done_at: dict[int, float] = field(default_factory=dict)  # task id -> when its result came in
    title: str | None = None  # shown in the stage column instead of the name
    settle: bool = True  # Progress.settle_frac applies (not to a CG step: each would lose its ETA)

    def throughput(self, now: float, window: float = THROUGHPUT_WINDOW_S) -> float | None:
        """Rows a second whose results reached the coordinator over the last `window` seconds
        (the whole stage before that): what the stage actually achieves, which the tasks' own
        seconds overstate when results wait on the coordinator or the workers run fewer tasks
        than they could (D6 r4 35953385 round 1: 5.2M rows/s shown, 0.25M achieved)."""
        if self.started is None or len(self.done_at) < THROUGHPUT_MIN_TASKS:
            return None
        span = now - self.started
        if span < THROUGHPUT_MIN_S:
            return None
        lo = now - min(window, span)
        rows = sum(self.rows[k] for k, t in self.done_at.items() if t >= lo)
        return rows / (now - lo)

    @property
    def total_rows(self) -> int:
        return sum(self.rows.values())

    @property
    def done_rows(self) -> int:
        return sum(self.rows[k] for k in self.done)

    def rate(self) -> float | None:
        r = sum(self.rows[k] for k in self.done)
        return (sum(self.done.values()) / r) if r else None

    def rate_band(self, B: int = 200, rng: np.random.Generator | None = None):
        """10th-90th percentile of the seconds-per-row rate, by bootstrap over finished tasks."""
        if len(self.done) < 3:
            return None
        rng = rng if rng is not None else np.random.default_rng(0)
        rows = np.array([self.rows[k] for k in self.done], float)
        secs = np.array(list(self.done.values()))
        idx = rng.integers(0, len(rows), size=(B, len(rows)))
        rates = secs[idx].sum(1) / np.maximum(rows[idx].sum(1), 1)
        return np.percentile(rates, [10, 90])


class Progress:
    def __init__(
        self,
        workdir: str | os.PathLike | None,
        label: str = "",
        every: float = 30.0,
        round2_recomputes: bool = False,
    ):
        self.round2_recomputes = round2_recomputes
        self.workdir = Path(workdir) if workdir else None
        self.label = label
        self.every = every
        self.stages: dict[str, StageProgress] = {}
        self.order: list[str] = []
        self.t0 = time.time()
        self._last = 0.0
        self._header = HeaderEvery()
        self.verbosity = verbosity()
        self._printed: dict | None = None  # what the last log line showed (verbosity 1)
        self.stage2_prior_frac: float | None = None  # round2 secs / round1 secs per row
        self.round1_rate_prior: float | None = None  # round 1's rate from its checkpoint
        self.slots = 1
        self.max_slots: int | None = None
        self.running: dict[str, set[int]] = {}
        self.history: list[dict] = []  # (time, progress, ETA) per tick
        self.note = ""  # appended to the log line, e.g. the CG step and residual
        # GB each task asks of its worker (DaskExecutor runs as many at once as fit its memory);
        # None: a whole worker, one task at a time (prep, and stages without an estimate)
        self.task_gb: float | None = None
        # GB a task of `rows` rows asks (DaskExecutor's MEMORY), when tasks differ: else task_gb
        self.task_gb_of = None
        self.extra_eta: float | None = None  # seconds of work no stage holds yet (CG steps to come)
        # show a stage's times as "calibrating" until this share of its tasks is done (or
        # SETTLE_MAX_S has passed): the first tasks back are the quick ones, so the early
        # rate runs high (fpp_eng3 m6: 316M rows/s at 0.4%, 7M from 15%). Off (0) by
        # default; a fit's CG steps are stages too, added with settle=False so they keep it.
        self.settle_frac = 0.0
        # stages not added yet follow, whose times are unknown (the singleton passes until one
        # drops nothing, then the rounds): no total is shown, nor a stage count (cellxgene H:L3
        # singletons7 said "7/7" and 'all left' 2:28 for that pass alone)
        self.more = False

    # --- bookkeeping -----------------------------------------------------
    def add_stage(
        self, name: str, rows: dict[int, int], before: str | None = None, title: str | None = None,
        settle: bool = True,
    ):
        self.stages[name] = StageProgress(name, dict(rows), title=title, settle=settle)
        self.order.insert(self.order.index(before) if before in self.order else len(self.order), name)
        self.running[name] = set()

    def start(self, name: str):
        self.stages[name].started = time.time()

    def finish(self, name: str, log: bool = True):
        self.stages[name].finished = time.time()
        self.tick(force=log)

    def task_done(self, stage: str, tid: int, seconds: float, timing: dict | None = None):
        self.stages[stage].done[tid] = seconds
        self.stages[stage].done_at[tid] = time.time()
        self.running[stage].discard(tid)
        # the prior only describes a fit's round 2 relative to its round 1; prep
        # stages (scan, encode, split, merge) have unrelated costs and wait for
        # their own measured rate
        if stage == "round1" and timing and self.stage2_prior_frac is None:
            tot = timing.get("total", 0) or 1e-9
            # round 2 reads the cache (~ cost of writing it) and forms the
            # meat (~ one to two cross-products per vcov type).
            if self.round2_recomputes:
                # round 2 redoes round 1 (minus any cache write) and then forms
                # residuals and scores. Measured 0.3x-1.8x round 1 (faster at
                # scale on OSCER, where round 1 pays cold reads), so assume 1x
                # and let the measured rate take over after the first tasks.
                frac = (tot - timing.get("write", 0)) / tot
            else:
                frac = (timing.get("write", 0) + 1.5 * timing.get("crossprod", 0)) / tot
            self.stage2_prior_frac = float(min(max(frac, 0.05), 3.0))
        self.tick()

    def set_slots(self, slots: int, max_slots: int | None = None):
        self.slots = max(int(slots), 0)
        self.max_slots = max_slots

    # --- estimates -------------------------------------------------------
    def _stage_eta(self, st: StageProgress, rate: float | None, slots: int):
        if rate is None:
            return None, None
        pending = [r for k, r in st.rows.items() if k not in st.done]
        if not pending:
            return 0.0, 0.0
        running = [st.rows[k] for k in self.running.get(st.name, ()) if k not in st.done]
        work = sum(pending) * rate - 0.5 * sum(running) * rate
        if slots <= 0:
            return float("inf"), max(pending) * rate
        return max(work / slots, max(pending) * rate * (0.5 if running else 1.0)), max(pending) * rate

    def snapshot(self) -> dict:
        now = time.time()
        stages = []
        total_eta = 0.0
        total_known = True
        # round 2's prior is a share of round 1's rate (not the first stage's: singletons1)
        r1 = self.stages["round1"].rate() if "round1" in self.stages else None
        if r1 is None:  # round 1 came from a checkpoint: its rate as saved there
            r1 = self.round1_rate_prior
        for name in self.order:
            st = self.stages[name]
            rate = st.rate()
            projected = False
            if rate is None and name == "round2" and r1 is not None and self.stage2_prior_frac:
                rate, projected = r1 * self.stage2_prior_frac, True
            thr = None if st.finished else st.throughput(now)
            if st.finished:
                eta = 0.0
            else:
                eta, _ = self._stage_eta(st, rate, self.slots)
                if thr and eta is not None and np.isfinite(eta):  # the slower of the two
                    eta = max(eta, (st.total_rows - st.done_rows) / thr)
            band = None
            b = st.rate_band()
            if b is not None and not st.finished and eta is not None and rate:
                band = [eta * b[0] / rate, eta * b[1] / rate]
            eta_max = None
            if self.max_slots and self.max_slots > self.slots and rate and not st.finished:
                eta_max, _ = self._stage_eta(st, rate, self.max_slots)
            if eta is None or not np.isfinite(eta):  # no rate yet, or no workers connected
                total_known = False
            else:
                total_eta += eta
            stages.append(
                {
                    "stage": name,
                    "tasks_done": len(st.done),
                    "tasks": len(st.rows),
                    "rows_done": st.done_rows,
                    "rows": st.total_rows,
                    "pct_rows": round(100 * st.done_rows / max(st.total_rows, 1), 1),
                    "rows_per_sec_per_worker": (1 / rate) if rate else None,
                    "rows_per_sec": thr,  # achieved (StageProgress.throughput), None early on
                    "rate_projected": projected,
                    "eta_seconds": eta,
                    "eta_band_seconds": band,
                    "eta_seconds_at_max_workers": eta_max,
                    "elapsed_seconds": (now - st.started) if st.started else 0,
                    "finished": bool(st.finished),
                    "title": st.title,
                    "settling": bool(
                        not st.finished
                        and st.settle
                        and len(st.done) < self.settle_frac * len(st.rows)
                        and (now - (st.started or now)) < SETTLE_MAX_S
                    ),
                }
            )
        if self.extra_eta is not None:
            total_eta += self.extra_eta
        if self.more:
            total_known = False
        return {
            "label": self.label,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": now - self.t0,
            "workers": self.slots,
            "max_workers": self.max_slots,
            "stages": stages,
            "eta_total_seconds": total_eta if total_known else None,
            "finish_time": _clock(now + total_eta) if total_known else None,
            "more_stages": bool(self.more),
            "note": self.note,
        }

    def line(self, snap: dict) -> str:
        return status_line(snap)

    def tick(self, force: bool = False):
        now = time.time()
        if not force and now - self._last < self.every:
            return
        self._last = now
        snap = self.snapshot()
        global WORK_LEFT
        # stages still to be added: None, which the cluster counts as work to come (the last
        # known total, one singleton pass's, let workers go). A stage with no rate yet (a few
        # seconds at its start) keeps the last total, so replacements do not flicker
        tot = snap["eta_total_seconds"]
        if tot is not None:
            WORK_LEFT = (now, float(tot))
        elif snap["more_stages"]:
            WORK_LEFT = (now, None)
        self._report(snap, now, force)
        rows_all = sum(st.total_rows for st in self.stages.values())
        done_all = sum(st.done_rows for st in self.stages.values())
        self.history.append(
            {
                "t": now - self.t0,
                "workers": self.slots,
                "frac_done": done_all / max(rows_all, 1),
                "eta_total": snap["eta_total_seconds"],
            }
        )
        if self.workdir:
            self.workdir.mkdir(parents=True, exist_ok=True)
            with open(self.workdir / "history.jsonl", "a") as fh:
                fh.write(json.dumps(self.history[-1]) + "\n")
            self.workdir.mkdir(parents=True, exist_ok=True)
            p = self.workdir / "status.json"
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(snap, indent=1, default=float))
            os.replace(tmp, p)

    def _milestone(self, snap: dict, now: float) -> dict:
        return milestone(snap, now)

    def _report(self, snap: dict, now: float, force: bool):
        if self.verbosity <= 0:
            return
        if self.verbosity >= 2:
            self._log_line(snap)
            return
        m, last = self._milestone(snap, now), self._printed
        if force or milestone_due(m, last, now):
            shown = snap
            if last is not None and m["note"] == last["note"] and m["stage"] == last["stage"]:
                shown = {**snap, "note": ""}  # the note is on the line above; don't repeat it
            self._log_line(shown)
            self._printed = m
            return
        out = _live_target()
        if out is not None:
            _draw_live(out, self.line(snap))

    def _log_line(self, snap: dict):
        hdr = status_header(snap)
        if self._header(hdr):
            log.info(hdr)
        log.info(self.line(snap))


def milestone(snap: dict, now: float) -> dict:
    """What decides whether verbosity 1 logs a line: the current stage, its state (waiting
    for workers, calibrating, running), the share of its rows done and the note. Also used by
    the session client to choose between a printed line and the live line."""
    stages = snap["stages"]
    cur = next((s for s in stages if not s["finished"]), stages[-1])
    eta = cur["eta_seconds"]
    if snap["workers"] == 0 and not cur["finished"]:
        state = "waiting"
    elif eta is None or not np.isfinite(eta):
        state = "calibrating"
    else:
        state = "running"
    return {
        "stage": cur["stage"],
        "state": state,
        "pct": float(cur["pct_rows"]),
        "note": snap.get("note") or "",
        "t": now,
    }


def milestone_due(m: dict, last: dict | None, now: float) -> bool:
    """A printed line is due: the first, a new stage, state or note, LOG_PCT% more of the
    stage's rows since the last line (once MIN_GAP_S has passed), or HEARTBEAT_S seconds
    since it."""
    since = now - last["t"] if last is not None else 0.0
    return (
        last is None
        or any(m[k] != last[k] for k in ("stage", "state", "note"))
        or (m["pct"] - last["pct"] >= LOG_PCT and since >= MIN_GAP_S)  # all count from the
        or since >= HEARTBEAT_S  # last line printed: a line at whichever comes first
    )


# (time.time(), seconds of known work left) from the latest status line of any run in this
# process: the coordinator's cluster reads it to decide whether a retiring worker needs a
# replacement (cluster._replace_ahead)
WORK_LEFT: tuple[float, float | None] | None = None


HEADER_EVERY = 10  # the column header repeats every this many status lines


def _columns(snap: dict) -> list[tuple[str, str, str]]:
    """(header, value, align) per column of a status line. Widths depend only on what is fixed
    for a run (label, stage names, shard counts, max workers), so successive lines line up."""
    stages = snap["stages"]
    i = next((k for k, s in enumerate(stages) if not s["finished"]), len(stages) - 1)
    cur = stages[i]
    n = len(stages)
    more = snap.get("more_stages")
    wname = max(len(s.get("title") or s["stage"]) for s in stages)
    wn = len(str(max(s["tasks"] for s in stages)))
    mx = snap.get("max_workers")
    ww = len(str(mx or snap["workers"]))
    rps = cur["rows_per_sec_per_worker"]
    eta = cur["eta_seconds"]
    if snap["workers"] == 0 and not cur["finished"]:
        stage_left = "waiting for workers"
    elif eta is None or not np.isfinite(eta) or cur.get("settling"):
        stage_left = "calibrating"
    else:
        stage_left = fmt_td(eta)
    total = None if cur.get("settling") else snap["eta_total_seconds"]
    return [
        ("run", snap["label"], "<"),
        (
            "stage",
            f"{cur.get('title') or cur['stage']:<{wname}}"
            if more
            else f"{i + 1:>{len(str(n))}}/{n} {cur.get('title') or cur['stage']:<{wname}}",
            "<",
        ),
        ("shards", f"{cur['tasks_done']:>{wn}}/{cur['tasks']:<{wn}}", ">"),
        ("rows", f"{float(cur['pct_rows']):.1f}%", ">"),
        ("workers", f"{snap['workers']:>{ww}}/{mx}" if mx else f"{snap['workers']}", ">"),
        ("rows/s", "-" if cur.get("settling") else _rows_per_sec(cur, snap["workers"]), ">"),
        ("stage left", stage_left, ">"),
        ("all left", fmt_td(total) if total is not None else "-", ">"),
        ("done at", (None if cur.get("settling") else snap.get("finish_time")) or "-", ">"),
    ]


def _rows_per_sec(cur: dict, workers: int) -> str:
    """Achieved rows a second once known, else the tasks' own rate times the workers."""
    thr, rps = cur.get("rows_per_sec"), cur["rows_per_sec_per_worker"]
    if thr is not None:
        return f"{thr / 1e6:.2f}M"
    return f"{rps * max(workers, 1) / 1e6:.2f}M" if rps else "-"


_WIDTH = {"rows": 6, "rows/s": 8, "stage left": 10, "all left": 8, "done at": 9}


def _render(snap: dict, header: bool) -> str:
    out = []
    for h, v, al in _columns(snap):
        w = max(len(h), _WIDTH.get(h, 0), 0 if h == "stage left" else len(v))
        if not header and h == "stage left" and len(v) > w:  # "waiting for workers": no times
            out.append(v)
            break
        out.append(f"{(h if header else v):{al}{w}}")
    line = "  ".join(out)
    if not header:
        cur = next((s for s in snap["stages"] if not s["finished"]), snap["stages"][-1])
        at_max = None if cur.get("settling") else cur.get("eta_seconds_at_max_workers")
        mx = snap.get("max_workers")
        if at_max is not None and mx and snap["workers"] < mx:
            line += f"  ({fmt_td(at_max)} at {mx} workers)"
        if snap.get("note"):
            line += f"  {snap['note']}"
    return line.rstrip()


def status_line(snap: dict) -> str:
    """One log line from a status snapshot (Progress.snapshot, or a status.json read back):
    run, stage k of n, shards done, % rows, workers, throughput, time left in the stage and
    overall, finish clock. Columns are padded so lines of a run line up under status_header.
    The ETA band and elapsed time stay in status.json (the log's timestamps give elapsed)."""
    return _render(snap, header=False)


def status_header(snap: dict) -> str:
    """The column names, aligned with status_line for the same run."""
    return _render(snap, header=True)


class HeaderEvery:
    """Says when to repeat the column header: at the first line, every HEADER_EVERY lines,
    and when the columns change width (a new stage with a longer name)."""

    def __init__(self, every: int = HEADER_EVERY):
        self.every, self.n, self.last = every, 0, None

    def __call__(self, header: str) -> bool:
        due = self.n % self.every == 0 or header != self.last
        self.n = 1 if header != self.last else self.n + 1
        self.last = header
        return due
