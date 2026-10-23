"""Command-line entry point: argument handling that needs no scheduler."""

import pytest

import dfeols
from dfeols.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert dfeols.__version__ in capsys.readouterr().out


def test_unknown_args_rejected_outside_bench():
    # `dfeols bench` forwards unknown options to the benchmark; other commands must not
    with pytest.raises(SystemExit) as e:
        main(["status", ".", "--no-such-option"])
    assert e.value.code == 2


def test_profile_object_takes_overrides(monkeypatch, tmp_path):
    # executor(prof, max_workers=...) used to drop overrides when given a Profile
    monkeypatch.setenv("DFEOLS_CONFIG", str(tmp_path / "none.toml"))
    p = dfeols.profile("local")
    q = p.with_overrides(max_workers=3, scratch="/x", queue=None)
    assert q.worker["max_workers"] == 3 and q.scratch == "/x"
    assert p.worker["max_workers"] != 3 and p.scratch is None
    assert p.with_overrides() is p
    assert dfeols.profile("local", max_workers=3, scratch="/x") == q
