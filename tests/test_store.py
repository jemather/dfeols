"""Persisted demeaned data: list, rm, and the scratch checks at session start."""

import json
import types
import warnings

import polars as pl
import pytest

import dfeols
from dfeols import store
from dfeols.cli import main
from dfeols.lock import DirLock
from dfeols.session import Session, _check_storage
from simdata import sim_panel

pytest.importorskip("jsonschema")
warnings.filterwarnings("ignore")


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


def _roots(tmp_path):
    return {"root": tmp_path / "s", "demeaned": tmp_path / "d"}


def _words(r):
    return ["--demeaned", str(r["demeaned"]), "--sessions", str(r["root"])]


def test_list_rm_and_rebuild(ds, tmp_path, capsys):
    r = _roots(tmp_path)
    s = Session.start("a", ds.root, "spell + week", ["eng", "feat"], cluster="id_a", poll=0.05, **r)
    try:
        (e,) = store.entries(r["demeaned"], r["root"], purge_days=14)
        assert e["complete"] and e["current"] and e["bytes"] > 0 and e["rows"] > 0
        assert e["sessions"] == ["a"] and e["building"] is None
        assert e["variables"] == ["eng", "feat"] and e["fixed_effects"] == ["spell", "week"]
        assert e["purge_after"] is not None
        with pytest.raises(RuntimeError, match="in use by running session"):
            store.remove(r["demeaned"], e["key"], r["root"])
    finally:
        s.stop()
    (e,) = store.entries(r["demeaned"], r["root"])
    assert e["sessions"] == []
    capsys.readouterr()
    main(["demeaned", "list", *_words(r)])
    out = capsys.readouterr().out
    assert f"{e['dataset']}/{e['key']}" in out and "1 folder," in out
    with pytest.raises(SystemExit, match="--yes"):  # no tty under pytest
        main(["demeaned", "rm", e["key"][:6], *_words(r)])
    assert store.entries(r["demeaned"], r["root"])
    main(["demeaned", "rm", e["key"][:6], "--yes", *_words(r)])
    assert f"removed {e['dataset']}/{e['key']}" in capsys.readouterr().out
    assert store.entries(r["demeaned"], r["root"]) == []
    assert not (r["demeaned"] / e["dataset"]).exists()
    assert (r["demeaned"] / "timings.jsonl").exists()  # the estimates' history stays
    with Session.start("b", ds.root, "spell + week", ["eng", "feat"], cluster="id_a", poll=0.05, **r):
        pass
    ev = [json.loads(l)["type"] for l in (r["root"] / "b" / "events.jsonl").read_text().splitlines()]
    assert "demeaned_persisted" in ev and "demeaned_loaded" not in ev


def _fake(root, dataset, key, manifest=None):
    kd = root / dataset / key
    (kd / "shards").mkdir(parents=True)
    (kd / "shards" / "part-000000.parquet").write_bytes(b"x" * 100)
    if manifest is not None:
        (kd / "manifest.json").write_text(json.dumps(manifest))
    return kd


def test_incomplete_old_format_locked_and_prefixes(tmp_path, capsys):
    root = tmp_path / "d"
    _fake(root, "ds1", "abc111", None)
    old = {"format": "dfeols-demeaned/1", "created": "2026-11-04T06:04:07Z", "rows": 10, "variables": ["y"]}
    _fake(root, "ds1", "abc222", old)
    kd = _fake(root, "ds2", "def333", None)
    (root / "timings.jsonl").write_text("")
    es = {e["key"]: e for e in store.entries(root, None, purge_days=14)}
    assert set(es) == {"abc111", "abc222", "def333"}
    assert not es["abc111"]["complete"] and es["abc222"]["complete"] and not es["abc222"]["current"]
    assert es["abc222"]["purge_after"] == "2026-11-18"
    with pytest.raises(KeyError, match="matches 2 folders"):
        store.find(root, "abc")
    assert store.find(root, "ds1/abc2")["key"] == "abc222"
    with DirLock(kd, what="demeaned data"):  # a build in progress
        assert store.entries(root)[-1]["building"] is not None
        main(["demeaned", "list", "--demeaned", str(root)])
        assert "being built" in capsys.readouterr().out
        with pytest.raises(RuntimeError, match="being built"):
            store.remove(root, "def")
    main(["demeaned", "list", "--demeaned", str(root)])
    out = capsys.readouterr().out
    assert "incomplete" in out and "old format dfeols-demeaned/1" in out
    store.remove(root, "abc1")
    store.remove(root, "def")
    assert [e["key"] for e in store.entries(root)] == ["abc222"]
    assert not (root / "ds2").exists() and (root / "ds1").exists()


def test_check_storage(tmp_path, capsys):
    plenty = types.SimpleNamespace(demeaned_storage="plentiful", demeaned_confirm_gb=None)
    _check_storage(plenty, tmp_path / "x" / "y", 1.0, 2.0, False)
    assert "free of" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not enough scratch"):
        _check_storage(plenty, tmp_path, 1e30, 1e30, True)
    _check_storage(plenty, tmp_path, None, 1e30, False)
    assert "more than is free" in capsys.readouterr().out
    scarce = types.SimpleNamespace(demeaned_storage="confirm", demeaned_confirm_gb=1.0)
    _check_storage(scarce, tmp_path, 1e8, 5e8, False)  # under the threshold: no question
    with pytest.raises(SystemExit, match="confirm with --yes"):
        _check_storage(scarce, tmp_path, 1e9, 2e9, False)
    _check_storage(scarce, tmp_path, 1e9, 2e9, True)
    assert "confirmed by --yes" in capsys.readouterr().out


def test_profile_storage_settings(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[profiles.hpc]\nscheduler = "slurm"\ndemeaned_storage = "confirm"\n')
    p = dfeols.site.profile("hpc", path=cfg)
    assert p.demeaned_storage == "confirm" and p.demeaned_confirm_gb == 500
    cfg.write_text('[profiles.hpc]\nscheduler = "slurm"\n')
    assert dfeols.site.profile("hpc", path=cfg).demeaned_storage == "plentiful"
    cfg.write_text('[profiles.hpc]\nscheduler = "slurm"\ndemeaned_storage = "scarce"\n')
    with pytest.raises(ValueError, match="demeaned_storage"):
        dfeols.site.profile("hpc", path=cfg)


def test_start_asks_where_storage_is_scarce(ds, tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        'default = "tiny"\n[profiles.tiny]\nscheduler = "local"\n'
        'demeaned_storage = "confirm"\ndemeaned_confirm_gb = 0.000001\n'
    )
    monkeypatch.setenv("DFEOLS_CONFIG", str(cfg))
    r = _roots(tmp_path)
    words = ["--data", str(ds.root), "--fe", "spell + week", "--vars", "eng,feat"]
    words += ["--root", str(r["root"]), "--demeaned", str(r["demeaned"])]
    with pytest.raises(SystemExit, match="confirm with --yes"):
        main(["session", "start", "q", *words])
    assert not (r["root"] / "q").exists()  # nothing started


def test_manager_stops_before_demeaning_when_it_cannot_fit(ds, tmp_path):
    r = _roots(tmp_path)
    r["demeaned"].mkdir()
    huge = {
        "when": "2026-11-01T00:00:00Z",
        "rows": 1,
        "variables": 2,
        "columns": 3,
        "bytes": 10**18,
        "seconds": 1,
    }
    (r["demeaned"] / "timings.jsonl").write_text(json.dumps(huge) + "\n")
    with pytest.raises(RuntimeError, match="not enough scratch"):
        Session.start("big", ds.root, "spell + week", ["eng", "feat"], poll=0.05, progress=False, **r)
    assert json.loads((r["root"] / "big" / "session.json").read_text())["status"] == "failed"


def test_memory_warning_for_many_variables(ds):
    """No cap on declared variables (Ted 2026-11-08); a warning when a demeaning task would
    not fit a worker thread's share of the profile's memory."""
    from dfeols.session import Spec, _session_plan, memory_warnings

    spec = Spec.make(ds.root, "spell + week", ["eng", "feat"], "id_a")
    plan, _ = _session_plan(spec, ds, None)
    roomy = types.SimpleNamespace(worker={"memory": "32GB", "cores": 8})
    assert memory_warnings(spec, ds, plan, roomy) == []
    tight = types.SimpleNamespace(worker={"memory": "1MB", "cores": 4})
    (w,) = memory_warnings(spec, ds, plan, tight)
    assert "a demeaning task needs" in w and "1MB / 4 cores" in w
    assert memory_warnings(spec, ds, plan, types.SimpleNamespace()) == []
