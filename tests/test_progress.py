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
