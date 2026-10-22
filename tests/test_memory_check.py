"""fit() refuses a shard that cannot fit in the profile's worker memory."""

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("dask")
from dfeols.estimate import _check_worker_memory


def _ex(mem):
    return SimpleNamespace(profile=SimpleNamespace(name="site", worker={"memory": mem}))


def test_refuses_far_over_worker_memory():
    with pytest.raises(MemoryError, match="993,842,500 rows"):
        _check_worker_memory(_ex("24GB"), "twfe", 993_842_500, 379.2)


def test_warns_slightly_over(caplog):
    with caplog.at_level(logging.WARNING, logger="dfeols"):
        _check_worker_memory(_ex("24GB"), "twfe", 50_000_000, 30.0)
    assert "may run out of memory" in caplog.text


def test_no_profile_no_check():
    _check_worker_memory(object(), "twfe", 10**9, 1e6)  # serial / plain Dask client: nothing known
