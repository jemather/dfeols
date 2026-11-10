"""Planner build steps 3-4: fits record their timings (timings.py), the planner estimates wall
time from them, and recommends one job or demean-first (Ted 2026-10-28)."""

import json
import logging
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import dfeols
from dfeols import timings
from dfeols.planner import format_report
from simdata import sim_panel

warnings.filterwarnings("ignore")


@pytest.fixture
def record(tmp_path, monkeypatch):
    monkeypatch.setenv("DFEOLS_TIMINGS", str(tmp_path / "timings"))
    return tmp_path / "timings" / timings.FITS


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    data = sim_panel().drop_nulls().filter(pl.col("eng").is_finite())
    return dfeols.prep(
        data,
        shard_dir=tmp_path_factory.mktemp("shards"),
        partition="id_a",
        fixed_effects=["spell", "week"],
        clusters=["id_a"],
        target_rows=3000,
    )


def _fit(ds, tmp_path, formula, **kw):
    return dfeols.fit(ds, formula, workdir=tmp_path, log_every=1e9, record=False, **kw)


def test_fits_are_recorded_and_a_repeat_suggests_a_session(ds, tmp_path, record, caplog):
    _fit(ds, tmp_path / "a", "eng ~ feat | spell + week")
    _fit(ds, tmp_path / "b", "eng ~ feat | spell + week", crossing_fe="iterate")
    with caplog.at_level(logging.WARNING, logger="dfeols"):
        _fit(ds, tmp_path / "c", "eng ~ feat + posts | spell + week")
    runs = [json.loads(l) for l in record.read_text().splitlines()]
    assert [r["path"] for r in runs] == ["sparse", "iterate", "sparse"]  # fit picks sparse with a local FE
    assert runs[1]["solve"] == "level" and runs[1]["cg_steps"] > 0
    a, c = runs[0], runs[2]
    assert a["fe_key"] == c["fe_key"] and a["rows"] == c["rows"] > 0
    assert c["columns"] == a["columns"] + 1  # one more regressor
    assert a["passes"] == 2 + a["singleton_passes"] and a["seconds"] > 0 and a["workers"] == 1
    assert "dfeols session start" in caplog.text and "eng ~ feat | spell + week" in caplog.text
    assert len(timings.history()) == 3


def test_no_record_without_a_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("DFEOLS_TIMINGS", raising=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text('default = "x"\n[profiles.x]\nscheduler = "local"\n')
    monkeypatch.setenv("DFEOLS_CONFIG", str(cfg))
    assert timings.history_dir() is None and timings.log_fit({"rows": 1}) is None


def _run(path, rows=1_000_000, columns=4, passes=2, seconds=10.0, workers=10, **kw):
    return {
        "path": path,
        "rows": rows,
        "columns": columns,
        "passes": passes,
        "seconds": seconds,
        "workers": workers,
        **kw,
    }


def test_estimate_fit():
    runs = [
        _run("sparse", seconds=10.0, crossing=True, singleton_passes=2, passes=4),
        _run("sparse", seconds=20.0, crossing=True, singleton_passes=3, passes=5),
        _run(
            "iterate",
            seconds=50.0,
            solve="fallback",
            cg_steps=30,
            passes=32,
            crossing=True,
            singleton_passes=0,
        ),
    ]
    assert timings.estimate_fit(1, 1, "local", crossing=False, workers=1, runs=[]) is None
    # same path: rates 1e6*4*4/(10*10)=1.6e5 and 1e6*4*5/(20*10)=1e5 per worker
    e = timings.estimate_fit(2_000_000, 4, "sparse", crossing=True, workers=20, runs=runs)
    assert e["passes"] == [2, 5]  # 2 + singleton passes 0..3 over every crossing run
    assert e["low"] == pytest.approx(2e6 * 4 * 2 / (1.6e5 * 20))
    assert e["high"] == pytest.approx(2e6 * 4 * 5 / (1e5 * 20))
    assert "on this path (sparse)" in e["basis"]
    # no crossing FEs: no singleton passes; no run on this path: all runs
    e = timings.estimate_fit(1_000_000, 4, "local", crossing=False, workers=10, runs=runs)
    assert e["passes"] == [2, 2] and "none on this path" in e["basis"]
    # the fallback adds a pass per CG step, from the earlier fallback runs
    e = timings.estimate_fit(1_000_000, 4, "iterate", solve="fallback", crossing=True, workers=10, runs=runs)
    assert e["passes"] == [2 + 0 + 30, 2 + 3 + 30] and e["unknown"] is None
    e = timings.estimate_fit(
        1_000_000, 4, "iterate", solve="fallback", crossing=True, workers=10, runs=runs[:2]
    )
    assert "CG step" in e["unknown"]


def test_shape():
    one = timings.shape({"path": "sparse", "solve": None}, {"high": 3600.0}, specs=1, max_walltime=48 * 3600)
    assert one == {"shape": "one job", "why": []}
    long = timings.shape({"path": "sparse"}, {"high": 30 * 3600.0}, specs=1, max_walltime=48 * 3600)
    assert long["shape"] == "demean-first" and "50% of the walltime limit (48 h)" in long["why"][0]
    assert (
        timings.shape({"path": "iterate", "solve": "level"}, None, specs=1, max_walltime=None)["shape"]
        == "demean-first"
    )
    many = timings.shape({"path": "local"}, None, specs=3, max_walltime=None)
    assert many["shape"] == "demean-first" and "3 specifications" in many["why"][0]
    assert timings.walltime_seconds("2-00:00:00") == 48 * 3600 and timings.walltime_seconds(None) is None


def test_plan_reports_wall_time_and_shape(record):
    data = sim_panel().filter(pl.col("eng").is_finite())
    r = dfeols.plan(data, "eng ~ feat | spell + week", cluster="id_a")
    assert all(o["estimate"] is None for o in r["options"]) and r["history_runs"] == 0
    assert "no earlier fits recorded" in format_report(r)
    record.parent.mkdir(parents=True)
    record.write_text(
        "\n".join(
            json.dumps(_run(p, crossing=True, singleton_passes=2, passes=4))
            for p in ("sparse", "dense", "local")
        )
    )
    prof = types.SimpleNamespace(
        name="t",
        worker={"memory": "32GB", "cores": 8, "max_workers": 10},
        coordinator={"memory": "16GB"},
        max_walltime="48:00:00",
    )
    r = dfeols.plan(data, "eng ~ feat | spell + week", cluster="id_a", profile=prof, specs=2)
    o = r["options"][0]
    assert o["estimate"]["low"] > 0 and o["estimate"]["high"] >= o["estimate"]["low"]
    assert o["shape"]["shape"] == "demean-first" and "2 specifications" in o["shape"]["why"][-1]
    text = format_report(r)
    assert "wall time ~" in text and "-> demean-first" in text and "3 earlier fits" in text
    r = dfeols.plan(data, "eng ~ feat | spell + week", cluster="id_a", profile=prof)
    if r["options"][0]["path"] != "C":
        assert "-> one job" in format_report(r)


def test_estimate_fit_model():
    """With enough earlier runs the range comes from a regression of log seconds on the fit's
    size, parallelism and path: on runs that follow such a law it is narrow and holds a new
    run; with a short history, or none on the path, the rate range as before."""
    rng = np.random.default_rng(0)

    def law(rows, cols, w, path):
        return 3.0 * (rows * cols / 1e6) ** 0.8 / min(w, 40) ** 0.6 * (4.0 if path == "sparse" else 1.0)

    runs = []
    for _ in range(40):
        rows, cols, w = int(10 ** rng.uniform(5, 9)), int(rng.integers(2, 20)), int(rng.integers(1, 40))
        path = ["local", "sparse"][rng.integers(2)]
        sec = law(rows, cols, w, path) * float(np.exp(rng.normal(0, 0.2)))
        runs.append(
            _run(
                path,
                rows=rows,
                columns=cols,
                workers=w,
                shards=40,
                seconds=sec,
                crossing=path == "sparse",
                singleton_passes=1 if path == "sparse" else 0,
            )
        )
    want = law(5e7, 8, 20, "sparse")
    e = timings.estimate_fit(50_000_000, 8, "sparse", crossing=True, workers=20, shards=40, runs=runs)
    assert "fit to 40 earlier runs" in e["basis"] and e["unknown"] is None
    assert e["low"] < want < e["high"] and e["high"] / e["low"] < 3
    assert e["passes"] == [3, 3]  # still the rate range's count
    # fewer than MODEL_MIN_RUNS, or no run on the path: the rate range
    short = timings.estimate_fit(50_000_000, 8, "sparse", crossing=True, workers=20, runs=runs[:5])
    assert "fit to" not in short["basis"]
    other = timings.estimate_fit(50_000_000, 8, "dense", crossing=True, workers=20, runs=runs)
    assert "none on this path" in other["basis"]
