"""Sessions (dfeols.session): demean once, fit many. Every session model must equal
dfeols.fit on the same formula and sample (Frisch-Waugh-Lovell), its record and the session
collection must validate, and persisted demeaned data must give the same results."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import estimation_format as estimation

import dfeols
from dfeols.session import Manager, Session, SessionError, Spec
from simdata import sim_panel

pytest.importorskip("jsonschema")
warnings.filterwarnings("ignore")

VARS = ["eng", "eng2", "posts", "feat", "wt"]


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    # the session sample is rows complete on every declared variable; with eng missing on
    # some rows, compare with fit on data that drop them first
    data = sim_panel().drop_nulls().filter(pl.col("eng").is_finite())
    return dfeols.prep(
        data,
        shard_dir=tmp_path_factory.mktemp("shards"),
        partition="id_a",
        fixed_effects=["spell", "week", "id_a"],
        clusters=["id_a", "week"],
        target_rows=3000,
    )


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    base = tmp_path_factory.mktemp("sess")
    return {"root": base / "sessions", "demeaned": base / "demeaned"}


def _start(name, ds, roots, fe="spell + week", **kw):
    return Session.start(name, ds.root, fe, VARS, cluster="id_a, week", poll=0.05, **roots, **kw)


def _fit(ds, tmp_path, formula, **kw):
    return dfeols.fit(ds, formula, workdir=tmp_path, log_every=1e9, record=False, **kw)


def _same(a, b, rtol=1e-9):
    assert a.N == b.N
    assert a.coefnames == b.coefnames
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=rtol, atol=1e-12)
    for t in a.outcomes[next(iter(a.outcomes))]["vcov"]:
        np.testing.assert_allclose(a.vcov(type=t), b.vcov(type=t), rtol=1e-7, atol=1e-14)
        assert (
            a.outcomes[next(iter(a.outcomes))]["vcov"][t][1]
            == b.outcomes[next(iter(b.outcomes))]["vcov"][t][1]
        )


@pytest.fixture(scope="module")
def sess(ds, roots):
    s = _start("crossing", ds, roots)
    yield s
    s.stop()


def test_fwl_crossing_fe(sess, ds, tmp_path):
    """Local FE (spell) + crossing FE (week, dense dummies): every vcov, both clusters."""
    for kw in (
        {"vcov": ["iid", "hetero"]},
        {"cluster": "id_a", "vcov": ["CRV1", "iid", "hetero"]},
        {"cluster": "week"},
    ):
        a = sess.fit("eng ~ posts + feat", **kw)
        b = _fit(ds, tmp_path / str(len(kw)), "eng ~ posts + feat | spell + week", weights=None, **kw)
        _same(a, b)
        np.testing.assert_allclose(a.outcomes["eng"]["r2_within"], b.outcomes["eng"]["r2_within"], rtol=1e-9)


def test_fwl_other_models(sess, ds, tmp_path):
    a = sess.fit("eng2 ~ feat", cluster="id_a")
    _same(a, _fit(ds, tmp_path / "a", "eng2 ~ feat | spell + week", cluster="id_a"))
    # the FE part may be written out if it is the session's
    a = sess.fit("posts ~ feat + eng | week + spell")
    _same(a, _fit(ds, tmp_path / "b", "posts ~ feat + eng | spell + week"))


def test_no_pass_for_iid(sess):
    r = sess.fit("eng ~ feat", vcov="iid")
    assert r.timing["passes"] == 0
    assert sess.fit("eng ~ feat", vcov="hetero").timing["passes"] == 1


def test_local_fes_only_and_weights(ds, roots, tmp_path):
    with Session.start(
        "local", ds.root, "spell", ["eng", "feat", "posts"], cluster="id_a", weights="wt", poll=0.05, **roots
    ) as s:
        a = s.fit("eng ~ feat + posts", cluster="id_a", vcov=["CRV1", "hetero"])
    b = _fit(
        ds, tmp_path, "eng ~ feat + posts | spell", cluster="id_a", vcov=["CRV1", "hetero"], weights="wt"
    )
    _same(a, b)


def test_records_validate(sess):
    r = sess.fit("eng ~ posts + feat", cluster="id_a", label="base")
    rec = sess.record(r)
    issues = [i for i in estimation.validate(rec) if i.level != "info"]
    assert not issues, "\n".join(map(str, issues))
    assert estimation.check_digest(rec) is True
    assert rec["provenance"]["execution"]["session"] == sess.state()["id"]
    assert rec["specification"]["call"].startswith(
        "dfeols.Session.attach('crossing').fit('eng ~ posts + feat'"
    )
    assert "declared for the session" in rec["specification"]["sample"]["restriction"]
    assert rec["meta"]["profiles"] == ["inference/0.1", "linear-fe/0.1", "dfeols/0.1"]


def test_errors(sess):
    with pytest.raises(SessionError, match="not declared"):
        sess.fit("eng ~ tenure")
    with pytest.raises(SessionError, match="interaction"):
        sess.fit("eng ~ feat:posts")
    with pytest.raises(SessionError, match="FEs are fixed"):
        sess.fit("eng ~ feat | spell")
    with pytest.raises(SessionError, match="not declared for this session"):
        sess.fit("eng ~ feat", cluster="spell")
    assert sess.status()["status"] == "ready"  # an error doesn't end the session


def test_second_manager_same_name_fails(sess, ds, roots):
    m = Manager("crossing", Spec.make(ds.root, "spell", ["eng"]), **roots)
    with pytest.raises(dfeols.LockedError):
        m.start()


def test_collection_and_reuse(ds, roots, tmp_path):
    s = _start("coll", ds, roots)
    a1 = s.fit("eng ~ posts + feat", cluster="id_a", label="base")
    s.fit("eng + eng2 ~ feat", label="two")
    p = s.export()
    coll = estimation.read_collection(p)
    assert coll["session"]["status"] == "open"
    s.close()
    s.stop()

    coll = estimation.read_collection(p)
    issues = [i for i in estimation.validate_collection(coll, base_dir=p.parent) if i.level != "info"]
    assert not issues, "\n".join(map(str, issues))
    assert coll["meta"]["kind"] == "session"
    assert coll["session"]["status"] == "closed"
    assert [m["label"] for m in coll["members"]] == ["base", "two.eng", "two.eng2"]
    types = [e["type"] for e in coll["events"]]
    assert types[:2] == ["data_loaded", "demeaned_loaded"]  # the crossing fixture built them
    assert types.count("fit") == 3 and "client_connected" in types and "client_disconnected" in types
    fits = [e for e in coll["events"] if e["type"] == "fit"]
    assert [e["member"] for e in fits] == [m["id"] for m in coll["members"]]
    sid = coll["session"]["id"]
    assert all(m["record"]["provenance"]["execution"]["session"] == sid for m in coll["members"])
    assert coll["session"]["fixed"]["declared_variables"] == VARS
    assert coll["session"]["clients"][0]["language"] == "python"

    # the same spec again: demeaned data reused, same answer
    s2 = _start("coll", ds, roots)
    b1 = s2.fit("eng ~ posts + feat", cluster="id_a", label="base")
    s2.stop()
    _same(a1, b1, rtol=0)
    ev = [json.loads(l) for l in (roots["root"] / "coll" / "events.jsonl").read_text().splitlines()]
    assert ev[1]["type"] == "demeaned_loaded"
    assert (roots["root"] / "coll" / "previous" / sid / "session.estimation-collection.json").exists()


def test_crashed_run_closed_by_next(ds, roots):
    m = Manager("crash", Spec.make(ds.root, "spell + week", VARS, "id_a"), **roots)
    m.start()
    m.fit({"formula": "eng ~ feat", "label": "one"})
    old = m.id
    m.lock.release()  # the manager dies: lock gone, nothing closed
    m2 = Manager("crash", Spec.make(ds.root, "spell + week", VARS, "id_a"), **roots)
    m2.start()
    m2.close()
    p = roots["root"] / "crash" / "previous" / old / "session.estimation-collection.json"
    coll = estimation.read_collection(p)
    assert coll["session"]["status"] == "crashed"
    assert len(coll["members"]) == 1
    issues = [i for i in estimation.validate_collection(coll, base_dir=p.parent) if i.level == "error"]
    assert not issues, "\n".join(map(str, issues))


def test_no_persist_removes_demeaned(ds, roots, tmp_path):
    r = {"root": tmp_path / "s", "demeaned": tmp_path / "d"}
    s = Session.start("np", ds.root, "spell", ["eng", "feat"], persist=False, poll=0.05, **r)
    where = Path(s.state()["demeaned"])
    assert where.is_relative_to(r["root"])
    s.fit("eng ~ feat")
    s.stop()
    assert not where.exists()
    assert not r["demeaned"].exists()


def test_cli_serve_fit_stop(ds, tmp_path, capsys):
    """The manager as its own process (`dfeols session serve`), driven from the CLI."""
    import subprocess

    from dfeols.cli import main

    root = tmp_path / "s"
    words = ["--root", str(root)]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "dfeols.cli",
            "session",
            "serve",
            "cli",
            "--data",
            str(ds.root),
            "--fe",
            "spell + week",
            "--vars",
            "eng,feat,posts",
            "--cluster",
            "id_a",
            "--demeaned",
            str(tmp_path / "d"),
            "--serial",
            *words,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        main(["session", "fit", "cli", "eng ~ feat + posts", "--cluster", "id_a", "--timeout", "120", *words])
        out = capsys.readouterr().out
        assert "feat" in out and "Clusters (id_a)" in out
        main(["session", "status", "cli", *words])
        assert json.loads(capsys.readouterr().out)["models"] == 1
        main(["session", "stop", "cli", *words])
        assert proc.wait(60) == 0
    finally:
        proc.kill()
    assert json.loads((root / "cli" / "session.json").read_text())["status"] == "closed"
    assert (root / "cli" / "session.estimation-collection.json").exists()


def test_idle_end_is_closed_and_hours_count_from_ready(ds, roots, tmp_path):
    import time

    r = {"root": tmp_path / "s", "demeaned": roots["demeaned"]}
    m = Manager(
        "idle", Spec.make(ds.root, "spell", ["eng", "feat"]), idle_minutes=0.02, hours=1, poll=0.05, **r
    )
    m.start()
    ready = time.time()
    assert ready - 1 <= m.ends - 3600 <= ready + 1
    assert m.serve() == "closed"
    st = json.loads((r["root"] / "idle" / "session.json").read_text())
    assert st["status"] == "closed"
    coll = estimation.read_collection(r["root"] / "idle" / "session.estimation-collection.json")
    assert coll["session"]["status"] == "closed"
    assert coll["events"][-1]["message"].startswith("idle for")


@pytest.mark.parametrize("level_space_gb", [None, 0])
def test_crossing_fe_iterate(ds, tmp_path, level_space_gb):
    """Crossing FEs above max_dummy_levels: plan C, one CG solve for every declared variable
    (in level space, or with a pass per step when level_space_gb=0). Equal to the dense
    solve's fit up to CG's tolerance."""
    r = {"root": tmp_path / "s", "demeaned": tmp_path / "d"}
    with pytest.warns(Warning) if level_space_gb == 0 else warnings.catch_warnings():
        s = Session.start(
            "it",
            ds.root,
            "spell + week",
            VARS,
            cluster="id_a, week",
            max_dummy_levels=10,
            level_space_gb=level_space_gb,
            iter_tol=1e-10,
            poll=0.05,
            **r,
        )
    try:
        m = json.loads((Path(s.state()["demeaned"]) / "manifest.json").read_text())
        assert m["crossing"]["method"] == "iterate"
        assert m["crossing"]["cg"]["mode"] == ("pass" if level_space_gb == 0 else "level")
        assert [f["absorbed"] for f in m["fixed_effects"]] == ["local", "crossing_iterate"]
        for kw in ({"cluster": "id_a", "vcov": ["CRV1", "hetero", "iid"]}, {"cluster": "week"}):
            a = s.fit("eng ~ posts + feat", **kw)
            b = _fit(ds, tmp_path / "f", "eng ~ posts + feat | spell + week", crossing_fe="dense", **kw)
            assert a.N == b.N and a.coefnames == b.coefnames
            np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-7)
            for t in a.outcomes["eng"]["vcov"]:
                np.testing.assert_allclose(a.vcov(type=t), b.vcov(type=t), rtol=1e-6)
        rec = s.record(a)
        issues = [i for i in estimation.validate(rec) if i.level != "info"]
        assert not issues, "\n".join(map(str, issues))
        num = rec["structures"]["numerics"]
        assert "conjugate gradient" in num["algorithm"] and num["iterations"] == m["crossing"]["cg"]["steps"]
    finally:
        s.stop()


def test_crossing_fe_only_iterate(ds, tmp_path):
    """No local FE: the crossing FEs alone, by CG (they also span the constant)."""
    r = {"root": tmp_path / "s", "demeaned": tmp_path / "d"}
    with Session.start(
        "w", ds.root, "week", ["eng", "feat"], max_dummy_levels=10, iter_tol=1e-10, poll=0.05, **r
    ) as s:
        a = s.fit("eng ~ feat", vcov="hetero")
    b = _fit(ds, tmp_path / "f", "eng ~ feat | week", vcov="hetero")
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-7)
    np.testing.assert_allclose(a.vcov(), b.vcov(), rtol=1e-6)


def test_dense_refuses_large_crossing_fe(ds, tmp_path):
    m = Manager(
        "d",
        Spec.make(ds.root, "spell + week", ["eng"], max_dummy_levels=10, crossing_fe="dense"),
        root=tmp_path / "s",
        demeaned=tmp_path / "d",
    )
    with pytest.raises(ValueError, match="crossing_fe='iterate'"):
        m.start()


def test_estimates_and_progress(ds, tmp_path, capsys):
    r = {"root": tmp_path / "s", "demeaned": tmp_path / "d"}
    with Session.start("p1", ds.root, "spell + week", ["eng", "feat"], cluster="id_a", poll=0.05, **r):
        pass
    assert (r["demeaned"] / "timings.jsonl").exists()
    # a new key: the start-up estimate comes from the earlier run
    s = Session.start("p2", ds.root, "spell + week", ["eng", "feat", "posts"], cluster="id_a", poll=0.05, **r)
    try:
        lo, hi = s.state()["demean_estimate_seconds"]
        assert 0 < lo <= hi
        capsys.readouterr()
        s.fit("eng ~ feat", vcov="iid", label="a")
        assert "no pass over the data" in capsys.readouterr().err
        s.fit("eng ~ feat + posts", cluster="id_a", label="b")
        err = capsys.readouterr().err
        assert "estimated" in err and "read rate of" in err and "b: done in" in err
        s.fit("eng ~ posts", vcov="hetero", label="c")
        assert "measured in this session" in capsys.readouterr().err
        idx = [json.loads(l) for l in (r["root"] / "p2" / "results" / "index.jsonl").read_text().splitlines()]
        assert [m["estimate_seconds"] > 0 for m in idx] == [False, True, True]
    finally:
        s.stop()
