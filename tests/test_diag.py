# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""dfeols.diag: the scheduler-loop profile and the stage summaries' cache counts."""

import json
import time
from types import SimpleNamespace

import pytest

from dfeols import diag


def _node(name, count, *kids, filename="m.py"):
    return {
        "identifier": f"{name};{filename};1",
        "description": {"filename": f"/x/{filename}", "name": name, "line_number": 1, "line": ""},
        "count": count,
        "children": {k["identifier"]: k for k in kids},
    }


def test_profile_top_counts_inclusive_once_and_self():
    # root 10: loop 10 -> [handle 6 -> [handle 4 (recursion) -> leaf 3], idle 3]
    leaf = _node("leaf", 3)
    inner = _node("handle", 4, leaf)
    outer = _node("handle", 6, inner)
    tree = {
        "identifier": "root",
        "description": {},
        "count": 10,
        "children": {"l": _node("loop", 10, outer, _node("idle", 3))},
    }
    got = diag.profile_top(tree)
    incl, own = dict(got["inclusive"]), dict(got["self"])
    assert got["samples"] == 10
    assert "m.py:loop" not in incl  # in every sample
    assert incl["m.py:handle"] == 6  # not 6 + 4
    assert incl["m.py:leaf"] == 3 and incl["m.py:idle"] == 3
    assert own == {"m.py:leaf": 3, "m.py:idle": 3, "m.py:handle": 2 + 1, "m.py:loop": 1}


def test_profiled_stages():
    assert all(diag._profiled(s) for s in ("round1", "singletons3", "cg1", "cg2", "cg10", "cg120"))
    assert not any(diag._profiled(s) for s in ("cg3", "cg9", "cg121"))


def _inc(i):
    return i + 1


def test_runlog_writes_scheduler_profile_and_cache_counts(tmp_path, monkeypatch):
    distributed = pytest.importorskip("distributed")
    monkeypatch.setenv("DFEOLS_DIAG", "1")
    prog = SimpleNamespace(workdir=str(tmp_path), label="t")
    with (
        distributed.LocalCluster(n_workers=1, threads_per_worker=1, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        rl = diag.RunLog(c, prog, "cg10", 3)
        t = time.time()
        while time.time() - t < 2.5:  # keep the scheduler busy over a few one-second cycles
            c.gather(c.map(_inc, range(300), pure=False))
        for i, hit in enumerate((True, False, True)):
            rl.task(i, {"total": 1.0, "read": 0.0 if hit else 0.5, "cache_hit": hit}, 0.01, 0.0)
        summ = rl.close()
        quiet = diag.RunLog(c, prog, "cg11", 1)  # not a profiled step
        quiet.task(0, {"total": 1.0}, 0.0, 0.0)
        assert "cache_hits" not in quiet.close()
    assert summ["cache_hits"] == 2 and summ["cache_known"] == 3 and summ["read_s_sum"] == 0.5
    lines = [json.loads(x) for x in (tmp_path / "scheduler_profile.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in lines] == ["cg10"]
    rec = lines[0]
    assert "profile_error" not in rec, rec
    assert rec["samples"] > 0 and rec["self"]
    assert rec["busy_s"] == pytest.approx(rec["samples"] * 0.01)
    assert rec["to_s"] >= rec["from_s"] >= 0
