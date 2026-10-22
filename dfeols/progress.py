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

Status is logged and written atomically to <workdir>/status.json, which
`dfeols status <workdir>` prints from any login node.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("dfeols")


def fmt_td(s: float | None) -> str:
    if s is None or not np.isfinite(s):
        return "?"
    s = int(round(s))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return (f"{d}d " if d else "") + f"{h:d}:{m:02d}:{s:02d}"


@dataclass
class StageProgress:
    name: str
    rows: dict[int, int]  # task id -> rows
    done: dict[int, float] = field(default_factory=dict)  # task id -> seconds
    started: float | None = None
    finished: float | None = None

    @property
    def total_rows(self) -> int:
        return sum(self.rows.values())

    @property
    def done_rows(self) -> int:
        return sum(self.rows[k] for k in self.done)

    def rate(self) -> float | None:
        r = sum(self.rows[k] for k in self.done)
        return (sum(self.done.values()) / r) if r else None

    def rate_band(self, B: int = 200, rng=np.random.default_rng(0)):
        if len(self.done) < 3:
            return None
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
        self.stage2_prior_frac: float | None = None  # round2 secs / round1 secs per row
        self.slots = 1
        self.max_slots: int | None = None
        self.running: dict[str, set[int]] = {}
        self.history: list[dict] = []  # (time, progress, ETA) per tick

    # --- bookkeeping -----------------------------------------------------
    def add_stage(self, name: str, rows: dict[int, int]):
        self.stages[name] = StageProgress(name, dict(rows))
        self.order.append(name)
        self.running[name] = set()

    def start(self, name: str):
        self.stages[name].started = time.time()

    def finish(self, name: str):
        self.stages[name].finished = time.time()
        self.tick(force=True)

    def task_done(self, stage: str, tid: int, seconds: float, timing: dict | None = None):
        self.stages[stage].done[tid] = seconds
        self.running[stage].discard(tid)
        if stage == self.order[0] and timing and self.stage2_prior_frac is None:
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
        r1 = self.stages[self.order[0]].rate() if self.order else None
        for name in self.order:
            st = self.stages[name]
            rate = st.rate()
            projected = False
            if rate is None and name != self.order[0] and r1 is not None and self.stage2_prior_frac:
                rate, projected = r1 * self.stage2_prior_frac, True
            if st.finished:
                eta = 0.0
            else:
                eta, _ = self._stage_eta(st, rate, self.slots)
            band = None
            b = st.rate_band()
            if b is not None and not st.finished and eta is not None and rate:
                band = [eta * b[0] / rate, eta * b[1] / rate]
            eta_max = None
            if self.max_slots and self.max_slots > self.slots and rate and not st.finished:
                eta_max, _ = self._stage_eta(st, rate, self.max_slots)
            if eta is None:
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
                    "rate_projected": projected,
                    "eta_seconds": eta,
                    "eta_band_seconds": band,
                    "eta_seconds_at_max_workers": eta_max,
                    "elapsed_seconds": (now - st.started) if st.started else 0,
                    "finished": bool(st.finished),
                }
            )
        return {
            "label": self.label,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": now - self.t0,
            "workers": self.slots,
            "max_workers": self.max_slots,
            "stages": stages,
            "eta_total_seconds": total_eta if total_known else None,
            "finish_time": time.strftime("%a %H:%M", time.localtime(now + total_eta))
            if total_known
            else None,
        }

    def line(self, snap: dict) -> str:
        cur = next((s for s in snap["stages"] if not s["finished"]), snap["stages"][-1])
        rps = cur["rows_per_sec_per_worker"]
        parts = [
            f"[{snap['label']} {cur['stage']}]",
            f"{cur['tasks_done']}/{cur['tasks']} shards",
            f"{cur['pct_rows']}% rows",
            f"{snap['workers']} workers" + (f" (max {snap['max_workers']})" if snap["max_workers"] else ""),
            f"{rps * max(snap['workers'], 1) / 1e6:.2f}M rows/s" if rps else "calibrating",
            f"elapsed {fmt_td(snap['elapsed_seconds'])}",
        ]
        if cur["eta_seconds"] is not None:
            e = f"stage ETA {fmt_td(cur['eta_seconds'])}"
            if cur["eta_band_seconds"]:
                lo, hi = cur["eta_band_seconds"]
                e += f" ({fmt_td(lo)}-{fmt_td(hi)})"
            parts.append(e)
        if snap["eta_total_seconds"] is not None:
            parts.append(f"total ETA {fmt_td(snap['eta_total_seconds'])} -> {snap['finish_time']}")
        if cur.get("eta_seconds_at_max_workers") is not None:
            parts.append(f"({fmt_td(cur['eta_seconds_at_max_workers'])} if all workers start)")
        return " | ".join(parts)

    def tick(self, force: bool = False):
        now = time.time()
        if not force and now - self._last < self.every:
            return
        self._last = now
        snap = self.snapshot()
        log.info(self.line(snap))
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
