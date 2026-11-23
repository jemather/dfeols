"""ETA and status reporting edge cases."""

import json
import math

from dfeols.progress import Progress


def test_no_workers_connected_does_not_crash(tmp_path):
    """With a measured rate but zero worker slots the stage ETA is infinite;
    the total ETA and finish time must be reported as unknown, not overflow
    (regression: OverflowError: timestamp out of range for platform time_t)."""
    prog = Progress(tmp_path, label="t", every=0)
    prog.add_stage("scan", {0: 100, 1: 100})
    prog.add_stage("encode", {0: 100, 1: 100})
    prog.start("scan")
    prog.task_done("scan", 0, 1.0)
    prog.task_done("scan", 1, 1.0)
    prog.finish("scan")
    prog.start("encode")
    prog.task_done("encode", 0, 1.0)
    prog.set_slots(0, 40)
    snap = prog.snapshot()
    assert math.isinf(snap["stages"][1]["eta_seconds"])
    assert snap["eta_total_seconds"] is None and snap["finish_time"] is None
    prog.tick(force=True)  # logs and writes status.json
    assert json.loads((tmp_path / "status.json").read_text())["finish_time"] is None
    assert "total ETA" not in prog.line(snap)


def test_prep_stages_are_not_projected_from_the_scan(tmp_path):
    """A prep stage without finished tasks has no ETA yet (regression: the
    split stage was projected from the fast scan rate, "8866M rows/s, ETA 0:00:01",
    while 200M-row split tasks ran for ~20 min)."""
    prog = Progress(tmp_path, label="prep", every=1e9)
    prog.add_stage("scan", {i: 2_000_000 for i in range(4)})
    prog.add_stage("split", {0: 200_000_000, 1: 200_000_000})
    prog.start("scan")
    for i in range(4):
        prog.task_done("scan", i, 0.1, {"total": 0.1})
    prog.finish("scan")
    prog.start("split")
    prog.set_slots(40, 40)
    snap = prog.snapshot()
    split = snap["stages"][1]
    assert split["eta_seconds"] is None and split["rows_per_sec_per_worker"] is None
    assert "calibrating" in prog.line(snap)


def test_fit_round2_still_projected_from_round1(tmp_path):
    prog = Progress(tmp_path, label="fit", every=1e9)
    prog.add_stage("round1", {i: 1000 for i in range(4)})
    prog.add_stage("round2", {i: 1000 for i in range(4)})
    prog.start("round1")
    for i in range(4):
        prog.task_done("round1", i, 1.0, {"total": 1.0, "write": 0.2, "crossprod": 0.2})
    prog.finish("round1")
    prog.start("round2")
    prog.set_slots(2, 2)
    r2 = prog.snapshot()["stages"][1]
    assert r2["rate_projected"] and r2["eta_seconds"] is not None


def test_cg_steps_show_in_the_line(tmp_path):
    """CG step stages go before round 2, so the log line names the step and its note
    (regression: OSCER showed 'round2 0/342 shards ... ETA 0:00:06' for 90 minutes of CG)."""
    from dfeols.estimate import _cg_steps_left

    prog = Progress(tmp_path, label="fit", every=1e9)
    prog.add_stage("round1", {0: 1000})
    prog.add_stage("round2", {0: 1000})
    prog.start("round1")
    prog.task_done("round1", 0, 1.0, {"total": 1.0})
    prog.finish("round1")
    prog.add_stage("cg1", {0: 1000}, before="round2")
    assert prog.order == ["round1", "cg1", "round2"]
    prog.start("cg1")
    prog.task_done("cg1", 0, 0.5)
    prog.note, prog.extra_eta = "CG step 1, residual 3.1e-01 (tol 1e-10)", 600.0
    line = prog.line(prog.snapshot())
    assert line.startswith("fit  2/3 cg1 ") and line.endswith("CG step 1, residual 3.1e-01 (tol 1e-10)")
    assert prog.snapshot()["eta_total_seconds"] >= 600
    assert _cg_steps_left([1.0, 0.5, 0.2], 1e-10) is None  # too few steps
    assert _cg_steps_left([1e-1, 1e-2, 1e-3, 1e-4], 1e-10) == 6  # a decade a step
    assert _cg_steps_left([1e-4, 2e-4, 1e-4, 1e-4], 1e-10) is None  # stalled


def test_status_lines_line_up_and_stay_short(tmp_path):
    """Successive lines of a run keep their columns under one header (padded fields) and fit
    a terminal: run, stage k/n, shards, % rows, workers, rows/s, time left, finish clock. The
    header repeats every HEADER_EVERY lines and when the columns change."""
    from dfeols.progress import HEADER_EVERY, HeaderEvery, status_header

    prog = Progress(tmp_path, label="eve2:all:r4", every=1e9)
    prog.add_stage("singletons1", {i: 1000 for i in range(4355)})
    prog.add_stage("round1", {i: 1000 for i in range(4355)})
    prog.start("singletons1")
    prog.set_slots(40, 40)
    lines, headers = [], set()
    for k in (9, 219, 3290):
        st = prog.stages["singletons1"]
        for _ in range(k - len(st.done)):
            prog.task_done("singletons1", len(st.done), 1.0)
        snap = prog.snapshot()
        lines.append(prog.line(snap))
        headers.add(status_header(snap))
    assert len(headers) == 1
    hdr = headers.pop()
    for col in ("shards", "rows/s", "stage left"):  # each value ends where its header ends
        end = hdr.index(col) + len(col)
        assert all(ln[end - 1] != " " and (len(ln) == end or ln[end] == " ") for ln in lines), (hdr, lines)
    assert all(len(ln) <= 100 for ln in lines + [hdr]), lines
    assert lines[0].split()[:4] == ["eve2:all:r4", "1/2", "singletons1", "9/4355"]
    prog.set_slots(0, 40)
    assert "waiting for workers" in prog.line(prog.snapshot())
    every = HeaderEvery()
    shown = [every(hdr) for _ in range(2 * HEADER_EVERY + 1)]
    assert shown.count(True) == 3 and shown[0] and shown[HEADER_EVERY]
    assert every(hdr + " ")  # columns changed: header again


def test_rows_per_sec_is_what_the_stage_achieves():
    # D6 r4 35953385 round 1: tasks of ~1 s each said 5.2M rows/s while results queued on the
    # coordinator and 0.25M rows/s came in; the line and the ETA now follow the arrivals
    import time

    from dfeols.progress import Progress, status_line

    p = Progress(None, label="t", every=1e9)
    p.add_stage("round1", {i: 1000 for i in range(100)})
    p.start("round1")
    now = time.time()
    st = p.stages["round1"]
    st.started = now - 100
    for i in range(20):
        p.task_done("round1", i, 0.01)
        st.done_at[i] = now - 100 + 5 * i
    p.set_slots(4)
    snap = p.snapshot()
    s = snap["stages"][0]
    assert 150 < s["rows_per_sec"] < 250
    assert s["eta_seconds"] >= 80_000 / 250
    assert "0.00M" in status_line(snap)  # ~200 rows/s, not 4 x 100k


def _replay(tmp_path, level, caplog, monkeypatch, ticks=480, every=30.0):
    """A 4-hour stage of 27011 shards ticked every 30 s, as in cellxgene H:L2 (36372921)."""
    import logging

    from dfeols import progress as pm

    pm.set_verbosity(level)
    try:
        clock = [1e6]
        monkeypatch.setattr(pm.time, "time", lambda: clock[0])
        prog = Progress(tmp_path, label="cx:H:L2", every=every)
        n = 27011
        prog.add_stage("cg13", {i: 5_000_000 for i in range(n)})
        prog.set_slots(40, 40)
        prog.start("cg13")
        prog.note = "CG step 12, residual 5.7e-08 (tol 1e-10)"
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="dfeols"):
            for k in range(ticks):
                clock[0] += every
                target = n * (k + 1) // ticks
                st = prog.stages["cg13"]
                while len(st.done) < target:
                    st.done[len(st.done)] = 1.0
                    st.done_at[len(st.done) - 1] = clock[0]
                prog.tick()
        return [r.getMessage() for r in caplog.records]
    finally:
        pm._VERBOSITY = None


def test_verbosity_1_logs_milestones_and_a_heartbeat(tmp_path, caplog, monkeypatch):
    msgs = _replay(tmp_path, 1, caplog, monkeypatch)
    lines = [m for m in msgs if m.startswith("cx:H:L2")]
    # 5% of this stage is 12 minutes, so the 10-minute heartbeat sets the pace: ~24 lines
    # in 4 hours, not 480
    assert 22 <= len(lines) <= 26, len(lines)
    assert sum("CG step 12" in m for m in lines) == 1  # the note shows once, not on every line
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 480  # status every tick


def test_verbosity_2_logs_every_tick_and_0_none(tmp_path, caplog, monkeypatch):
    lines = [m for m in _replay(tmp_path / "a", 2, caplog, monkeypatch) if m.startswith("cx:H:L2")]
    assert len(lines) == 480 and all("CG step 12" in m for m in lines)
    assert _replay(tmp_path / "b", 0, caplog, monkeypatch) == []


def test_heartbeat_when_nothing_moves(tmp_path, caplog, monkeypatch):
    import logging

    from dfeols import progress as pm

    clock = [1e6]
    monkeypatch.setattr(pm.time, "time", lambda: clock[0])
    prog = Progress(tmp_path, label="t", every=30.0)
    prog.verbosity = 1
    prog.add_stage("round1", {0: 10})
    prog.set_slots(0, 40)
    with caplog.at_level(logging.INFO, logger="dfeols"):
        for _ in range(41):  # 20 minutes waiting for workers
            clock[0] += 30
            prog.tick()
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("t ")]
    assert len(lines) == 3  # the first, then one each HEARTBEAT_S


def test_live_line_is_wiped_by_the_next_record(monkeypatch):
    import io
    import logging

    from dfeols import progress as pm

    class Tty(io.StringIO):
        def isatty(self):
            return True

    s = Tty()
    h = pm.LiveAwareHandler(s)
    h.setFormatter(logging.Formatter("%(message)s"))
    monkeypatch.setattr(pm.log, "handlers", [h])
    assert pm._live_target() is s
    pm._draw_live(s, "live 1")
    pm._draw_live(s, "live 2")
    pm.log.warning("a record")
    out = s.getvalue()
    # each live line ends in a newline (a prompt after `tail` starts on its own line); the
    # next one or the next record moves up onto it and erases it
    assert out.endswith("live 1\n" + pm.LIVE_UP + out.split(pm.LIVE_UP)[1].split("live 2")[0]
                        + "live 2\n" + pm.LIVE_UP + "a record\n")
    assert pm._live_stream is None
    f = io.StringIO()  # a file (Slurm's .out) gets the live line too, so tail -f shows it move
    h.stream = f
    assert pm._live_target() is f
    monkeypatch.setattr(pm.log, "handlers", [logging.StreamHandler(f)])
    assert pm._live_target() is None  # a handler the user set up: left alone


def test_a_warning_wipes_the_live_line(monkeypatch):
    import io
    import warnings

    from dfeols import progress as pm

    s = io.StringIO()
    pm._draw_live(s, "live status")
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        monkeypatch.setattr(pm, "_showwarning", lambda *a, **k: s.write("WARN\n"))
        warnings.showwarning = pm._showwarning_clear
        warnings.warn("slow")
    assert s.getvalue().endswith("live status\n" + pm.LIVE_UP + "WARN\n")
