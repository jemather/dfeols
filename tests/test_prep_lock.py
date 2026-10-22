"""Only one job may write a shard folder (or benchmark data set) at a time
(regression: a second job for the same preset cleared the first job's _codes
while it was still scanning; another hit "Directory not empty: '_buckets'")."""

import json
import os
import socket
import time

import pytest

import dfeols
from dfeols.lock import DirLock, LockedError
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
    with DirLock(sd):  # a prep that is still running
        with pytest.raises(dfeols.LockedError, match="in use by"):
            _prep(raw, sd)
        with pytest.raises(LockedError):
            _prep(raw, sd, overwrite=True)
        assert marker.exists()  # nothing of the running prep was touched


def test_lock_released_after_prep(raw, tmp_path):
    sd = tmp_path / "sd"
    _prep(raw, sd)
    assert not DirLock(sd).path.exists()
    _prep(raw, sd, overwrite=True)  # and can be taken again
    assert not list(tmp_path.glob(".sd.deleting-*"))  # the old folder was removed


def test_stale_lock_is_taken_over(raw, tmp_path):
    sd = tmp_path / "sd"
    lock = DirLock(sd).path
    lock.write_text(json.dumps({"host": "elsewhere", "pid": 1, "job": "123"}))
    old = time.time() - 3600
    os.utime(lock, (old, old))  # its holder stopped refreshing it
    assert _prep(raw, sd).n_rows > 0 and not lock.exists()


def test_dead_local_process_lock_is_taken_over(raw, tmp_path):
    sd = tmp_path / "sd"
    lock = DirLock(sd).path
    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": 2**22 + 12345}))
    _prep(raw, sd)
    assert not lock.exists()


def test_lock_is_refreshed_and_only_owner_releases(tmp_path):
    a = DirLock(tmp_path / "x", heartbeat=0.05).acquire()
    old = time.time() - 1000
    os.utime(a.path, (old, old))
    time.sleep(0.3)
    assert time.time() - a.path.stat().st_mtime < 5
    a.release()
    b = DirLock(tmp_path / "x").acquire()
    a.release()  # a second release by a former holder must not drop b's lock
    assert b.path.exists()
    b.release()
    assert not b.path.exists()


def test_bench_rebuilds_overflowed_shard_sets():
    from types import SimpleNamespace

    from dfeols.bench import stale_dataset

    def ds(shards, n_rows, **meta):
        return SimpleNamespace(meta=meta, shards=[{"rows": r} for r in shards], n_rows=n_rows)

    good = ds([2_000_000] * 500, 10**9, shard_assignment="range", prep_version="0.7.0")
    assert stale_dataset(good, 2e6) == ""
    assert "overflow" in stale_dataset(ds([2_000_000] * 500, 10**9, shard_assignment="range"), 2e6)
    broken = ds([993_842_500, 6_157_500], 10**9, shard_assignment="lpt", prep_version="0.7.0")
    assert "far from the target" in stale_dataset(broken, 2e6)
    assert "add up" in stale_dataset(ds([10, 10], 30), 2e6)
