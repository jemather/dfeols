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
