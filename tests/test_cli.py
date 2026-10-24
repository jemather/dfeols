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


@pytest.mark.parametrize("argv, profile, name, cmd_end", [
    # the documented form: options after the script, script args after --
    (["submit", "run.py", "--profile", "oscer", "--", "/data/x"], "oscer", "dfeols", "run.py /data/x"),
    (["submit", "--profile", "oscer", "run.py", "--", "/data/x"], "oscer", "dfeols", "run.py /data/x"),
    (["submit", "run.py", "--name", "job", "--profile", "p", "--", "--profile", "q"], "p", "job",
     "run.py --profile q"),
    (["submit", "run.py", "a", "--flag"], None, "dfeols", "run.py a --flag"),
])
def test_submit_options_after_script(monkeypatch, tmp_path, argv, profile, name, cmd_end):
    # REMAINDER used to pass `--profile oscer` to the script as its first argument
    import dfeols.cli as cli
    import dfeols.cluster
    import dfeols.site

    seen = {}
    real = dfeols.site.profile
    monkeypatch.setenv("DFEOLS_CONFIG", str(tmp_path / "none.toml"))

    def fake_profile(n=None, **k):
        seen["p"] = n
        return real("local", scratch=str(tmp_path))

    monkeypatch.setattr(dfeols.site, "profile", fake_profile)
    monkeypatch.setattr(dfeols.cluster, "coordinator_script", lambda p, cmd, **k: seen.update(cmd=cmd, **k) or cmd)
    monkeypatch.setattr(cli, "_submit_and_report", lambda *a: None)
    main(argv)
    assert seen["p"] == profile and seen["job_name"] == name
    assert seen["cmd"].endswith(cmd_end)
