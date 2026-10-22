"""Only one prep may write a shard_dir at a time (regression: a second
benchmark job for the same preset cleared the first job's _codes while it
was still scanning, and the first failed with FileNotFoundError)."""

import importlib
import json
import os
import time

import pytest

import dfeols

prep_mod = importlib.import_module("dfeols.prep")  # dfeols.prep is the function
from dfeols.synth import PanelDGP, generate, load_truth


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    generate(d, PanelDGP(n_units=400, units_per_chunk=200, T=10, n_groups=10), verbose=False)
    return d, load_truth(d)


def _prep(raw, sd, **kw):
    d, truth = raw
    return dfeols.prep(
        str(d / "*.parquet"),
        shard_dir=sd,
        partition="group",
        target_rows=2000,
        executor=dfeols.SerialExecutor(),
        **truth["prep"],
        **kw,
    )


def test_second_prep_refuses_and_leaves_first_alone(raw, tmp_path):
    sd = tmp_path / "sd"
    (sd / "_codes").mkdir(parents=True)
    marker = sd / "_codes" / "in_progress.parquet"
    marker.write_text("x")
    with prep_mod.prep_lock(sd):  # a prep that is still running
        with pytest.raises(dfeols.PrepInProgress, match="another prep is writing"):
            _prep(raw, sd)
        with pytest.raises(dfeols.PrepInProgress):
            _prep(raw, sd, overwrite=True)
        assert marker.exists()  # nothing of the running prep was touched


def test_lock_released_after_prep(raw, tmp_path):
    sd = tmp_path / "sd"
    _prep(raw, sd)
    assert not prep_mod._lock_path(sd).exists()
    _prep(raw, sd, overwrite=True)  # and can be taken again


def test_stale_lock_is_taken_over(raw, tmp_path):
    sd = tmp_path / "sd"
    lock = prep_mod._lock_path(sd)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"host": "elsewhere", "pid": 1, "job": "123", "started": 0}))
    old = time.time() - prep_mod.LOCK_STALE_S - 60
    os.utime(lock, (old, old))  # its holder stopped refreshing it
    ds = _prep(raw, sd)
    assert ds.n_rows > 0 and not lock.exists()


def test_dead_local_process_lock_is_taken_over(raw, tmp_path):
    sd = tmp_path / "sd"
    lock = prep_mod._lock_path(sd)
    lock.parent.mkdir(parents=True, exist_ok=True)
    import socket

    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": 2**22 + 12345, "started": 0}))
    _prep(raw, sd)
    assert not lock.exists()


def test_lock_is_refreshed(tmp_path, monkeypatch):
    monkeypatch.setattr(prep_mod, "LOCK_REFRESH_S", 0.05)
    sd = tmp_path / "sd"
    with prep_mod.prep_lock(sd):
        lock = prep_mod._lock_path(sd)
        old = time.time() - 1000
        os.utime(lock, (old, old))
        time.sleep(0.3)
        assert time.time() - lock.stat().st_mtime < 5
