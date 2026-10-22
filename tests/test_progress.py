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
