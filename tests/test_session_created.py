"""Sessions, build order step 4: created variables (add_variable) and refit. A model with a
created variable must equal dfeols.fit with that variable built before prep (or as fit's own
interaction), on dense and iterated crossing FEs; created variables are reused by a later
session on the same demeaned data; refit equals a fresh fit with the new vcov."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import estrec as estimation

import dfeols
from dfeols import store
from dfeols.session import Session, SessionError
from simdata import sim_panel

pytest.importorskip("jsonschema")
warnings.filterwarnings("ignore")

VARS = ["eng", "posts", "feat"]


def _data():
    return sim_panel().drop_nulls().filter(pl.col("eng").is_finite())


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return dfeols.prep(
        _data(),
        shard_dir=tmp_path_factory.mktemp("shards"),
        partition="id_a",
        fixed_effects=["spell", "week"],
        clusters=["id_a", "week"],
        target_rows=3000,
    )


@pytest.fixture(scope="module")
def ds_made(tmp_path_factory):
    """The same data with the created variables made before prep, for fit to compare with."""
    data = _data().with_columns(lwt=pl.col("wt").log(), pos=(pl.col("posts") > 0).cast(pl.Float64))
    return dfeols.prep(
        data,
        shard_dir=tmp_path_factory.mktemp("shards_made"),
        partition="id_a",
        fixed_effects=["spell", "week"],
        clusters=["id_a", "week"],
        target_rows=3000,
    )


def _roots(tmp_path):
    return {"root": tmp_path / "s", "demeaned": tmp_path / "d"}


def _start(name, ds, r, **kw):
    kw.setdefault("progress", False)
    return Session.start(name, ds.root, "spell + week", VARS, cluster="id_a, week", poll=0.05, **r, **kw)


def _fit(ds, tmp_path, formula, **kw):
    return dfeols.fit(ds, formula, workdir=tmp_path, log_every=1e9, record=False, **kw)


def _same(a, b, rtol=1e-9, vtol=1e-7):
    """Same estimates; term names may differ (fp against fit's feat:posts)."""
    assert a.N == b.N
    assert len(a.coefnames) == len(b.coefnames)
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=rtol, atol=1e-12)
    ya, yb = next(iter(a.outcomes)), next(iter(b.outcomes))
    for t in a.outcomes[ya]["vcov"]:
        np.testing.assert_allclose(a.vcov(type=t), b.vcov(type=t), rtol=vtol, atol=1e-14)
        assert a.outcomes[ya]["vcov"][t][1] == b.outcomes[yb]["vcov"][t][1]


@pytest.mark.parametrize(
    "spec",
    [{}, {"max_dummy_levels": 10, "iter_tol": 1e-12}, {"max_dummy_levels": 10, "level_space_gb": 0}],
    ids=["dense", "iterate", "iterate-no-level-space"],
)
def test_created_matches_fit(ds, ds_made, tmp_path, spec):
    r = _roots(tmp_path)
    tol = {} if not spec else {"rtol": 1e-6, "vtol": 1e-6}
    with _start("c", ds, r, **spec) as s:
        s.add_variable("fp", "feat * posts").add_variable("lwt", "log(wt)")
        s.add_variable("pos", "posts > 0")
        assert s.variables()["created"] == {"fp": "feat * posts", "lwt": "log(wt)", "pos": "posts > 0"}
        kw = {"cluster": "id_a", "vcov": ["CRV1", "hetero", "iid"]}
        a = s.fit("eng ~ posts + fp", **kw)
        b = _fit(ds, tmp_path / "f1", "eng ~ posts + feat:posts | spell + week", **kw)
        _same(a, b, **tol)
        assert a.coefnames == ["posts", "fp"]
        a = s.fit("eng ~ lwt + pos + feat", cluster="week")
        b = _fit(ds_made, tmp_path / "f2", "eng ~ lwt + pos + feat | spell + week", cluster="week")
        _same(a, b, **tol)
        a = s.fit("lwt ~ feat + fp", vcov="hetero")  # a created response
        b = _fit(ds_made, tmp_path / "f3", "lwt ~ feat + feat:posts | spell + week", vcov="hetero")
        _same(a, b, **tol)


def test_created_must_cover_the_sample(ds, tmp_path):
    r = _roots(tmp_path)
    with _start("m", ds, r) as s:
        with pytest.raises(SessionError, match=r"missing or not finite on [\d,]+ of the session"):
            s.add_variable("lp", "log(posts)")  # posts is 0 on some rows
        with pytest.raises(SessionError, match="not columns of the prepared data"):
            s.add_variable("z", "log(nope)")
        with pytest.raises(SessionError, match="unknown function"):
            s.add_variable("z", "system(feat)")
        with pytest.raises(SessionError, match="must be a column name"):
            s.add_variable("1z", "feat")
        with pytest.raises(SessionError, match="is declared"):
            s.add_variable("eng", "feat * 2")
        s.add_variable("f2", "feat^2")
        s.add_variable("f2", "feat ^ 2")  # the same again: nothing to do
        with pytest.raises(SessionError, match="already created in this session, as 'feat \\*\\* 2'|already"):
            s.add_variable("f2", "feat^3")
        with pytest.raises(SessionError, match="already created in this session, as f2"):
            s.add_variable("g", "feat**2")
        with pytest.raises(SessionError, match="add_variable"):
            s.fit("eng ~ feat:posts")
        with pytest.raises(SessionError, match="not declared or created"):
            s.fit("eng ~ fp")
        s.fit("eng ~ f2")  # the session is still fine
    lp = s.dir / "events.jsonl"
    ev = [json.loads(l) for l in lp.read_text().splitlines()]
    assert [e["type"] for e in ev].count("variable_created") == 1  # f2 (again: nothing logged)
    assert sum(e["type"] == "error" for e in ev) == 9


def test_reused_by_a_later_session(ds, tmp_path, capsys):
    r = _roots(tmp_path)
    with _start("one", ds, r) as s:
        s.add_variable("fp", "feat * posts").add_variable("f2", "feat^2")
        a = s.fit("eng ~ fp + f2 + posts", cluster="id_a")
    with _start("two", ds, r, progress=True) as s:
        # p2 is new; fp is reused, with no cross-product with p2 on file: one pass for it
        s.add_variable("p2", "posts^2").add_variable("featxposts", "posts*feat")
        err = capsys.readouterr().err
        assert "created featxposts = posts*feat (reused from an earlier session)" in err
        b = s.fit("eng ~ featxposts + posts + p2", cluster="id_a")
        s.add_variable("ff", "feat ^ 2")  # reused, crossed with fp's on file and p2 by a pass
        c = s.fit("eng ~ featxposts + ff + posts", cluster="id_a")
        d = s.fit("eng ~ featxposts + ff + posts + p2", cluster="id_a")
    ev = [json.loads(l) for l in (r["root"] / "two" / "events.jsonl").read_text().splitlines()]
    made = {e["detail"]["name"]: e["detail"]["reused"] for e in ev if e["type"] == "variable_created"}
    assert made == {"p2": False, "featxposts": True, "ff": True}
    _same(a, c, rtol=1e-12, vtol=1e-12)
    data_fit = _fit(ds, tmp_path / "f", "eng ~ feat:posts + posts | spell + week", cluster="id_a")
    with _start("three", ds, r) as s:
        e = s.add_variable("p2", "posts^2").fit("eng ~ posts + p2", cluster="id_a")
    with _start("four", ds, r) as s:
        s.add_variable("x", "feat * posts")
        _same(s.fit("eng ~ x + posts", cluster="id_a"), data_fit)
    assert b.N == d.N == e.N
    (k,) = store.entries(r["demeaned"])
    assert sorted(k["created"]) == ["feat * posts", "feat ** 2", "posts ** 2"]


def test_refit(ds, tmp_path):
    r = _roots(tmp_path)
    with _start("r", ds, r) as s:
        s.add_variable("fp", "feat * posts")
        m = s.fit("eng ~ posts + fp", vcov="iid", label="base")
        h = s.refit(m, vcov="hetero")
        assert h.label == "base.hetero" and h.timing["passes"] == 1
        np.testing.assert_array_equal(h.coef(), m.coef())
        _same(h, _fit(ds, tmp_path / "h", "eng ~ posts + feat:posts | spell + week", vcov="hetero"))
        c = s.refit("base", vcov="CRV1", cluster="week")
        assert c.label == "base.crv1.week" and c.cluster == "week"
        c2 = s.refit(c, vcov=["CRV1", "iid"])  # keeps the cluster
        assert c2.cluster == "week" and c2.label == "base.crv1.week.crv1-iid"
        with pytest.raises(SessionError, match="already has"):
            s.refit(c2, vcov=["CRV1", "iid"])
        with pytest.raises(SessionError, match="no model 'nope'"):
            s.refit("nope", vcov="hetero")
        p = s.export()
    coll = estimation.read_collection(p)
    issues = [i for i in estimation.validate_collection(coll, base_dir=p.parent) if i.level != "info"]
    assert not issues, "\n".join(map(str, issues))
    ch = [e for e in coll["events"] if e["type"] == "setting_changed"]
    assert [e["detail"]["model"] for e in ch] == ["base", "base", "base.crv1.week"]
    assert ch[0]["detail"]["vcov"] == {"from": ["iid"], "to": ["hetero"]}
    assert ch[0]["member"] == coll["members"][0]["id"]
    idx = [json.loads(l) for l in (s.dir / "results" / "index.jsonl").read_text().splitlines()]
    assert [m.get("refit_of") for m in idx] == [None, "base", "base", "base.crv1.week"]


@pytest.mark.usefixtures("full_records")
def test_records_with_created_variables(ds, tmp_path):
    r = _roots(tmp_path)
    with _start("rec", ds, r) as s:
        s.add_variable("fp", "feat * posts").add_variable("f2", "feat^2")
        m = s.fit("eng ~ posts + fp", cluster="id_a")
        rec = s.record(m)
        p = s.export()
    issues = [i for i in estimation.validate(rec) if i.level != "info"]
    assert not issues, "\n".join(map(str, issues))
    call = rec["specification"]["call"]
    assert call == (
        "dfeols.Session.attach('rec').add_variable('fp', 'feat * posts').fit('eng ~ posts + fp', "
        "cluster='id_a', vcov=['CRV1'])"
    )
    assert rec["extensions"]["dfeols"]["created_variables"] == [{"name": "fp", "expression": "feat * posts"}]
    derived = [d for d in rec["provenance"]["data"] if d["role"] == "derived"]
    assert [d["columns"] for d in derived] == [VARS, ["fp"]]
    coll = estimation.read_collection(p)
    issues = [i for i in estimation.validate_collection(coll, base_dir=p.parent) if i.level != "info"]
    assert not issues, "\n".join(map(str, issues))
    made = [e["detail"]["name"] for e in coll["events"] if e["type"] == "variable_created"]
    assert made == ["fp", "f2"]


def test_persist_takes_created_variables(ds, tmp_path):
    r = _roots(tmp_path)
    s = _start("np", ds, r, persist=False)
    s.add_variable("fp", "feat * posts")
    a = s.fit("eng ~ fp + posts", vcov="hetero")
    where = s.persist()
    assert (where / "created").is_dir()
    b = s.fit("eng ~ fp + posts", vcov="hetero")
    _same(a, b, rtol=0, vtol=0)
    s.stop()
    with _start("np2", ds, r) as s2:
        s2.add_variable("fp", "feat * posts")
        _same(a, s2.fit("eng ~ fp + posts", vcov="hetero"), rtol=0, vtol=0)
    ev = [json.loads(l) for l in (r["root"] / "np2" / "events.jsonl").read_text().splitlines()]
    assert [e["detail"]["reused"] for e in ev if e["type"] == "variable_created"] == [True]


def test_persist_into_a_kept_key_keeps_created(ds, tmp_path):
    r = _roots(tmp_path)
    s = _start("q1", ds, r, persist=False)
    s.add_variable("fp", "feat * posts")
    a = s.fit("eng ~ fp", vcov="hetero")
    _start("q2", ds, r).stop()  # keeps the same key, without fp
    where = s.persist()
    assert (where / "created").is_dir()
    _same(a, s.fit("eng ~ fp", vcov="hetero"), rtol=0, vtol=0)
    s.stop()


def test_cli_add_variable_and_refit(ds, tmp_path, capsys):
    from dfeols.cli import main

    r = _roots(tmp_path)
    s = _start("cli", ds, r)
    try:
        w = ["--root", str(r["root"])]
        main(["session", "add-variable", "cli", "fp", "feat * posts", *w])
        assert "fp = feat * posts" in capsys.readouterr().out
        main(["session", "fit", "cli", "eng ~ fp", "--label", "a", *w])
        main(["session", "refit", "cli", "a", "--vcov", "CRV1", "--cluster", "id_a", *w])
        out = capsys.readouterr().out
        assert "fp" in out
        assert [m["label"] for m in s._manager.models] == ["a", "a.crv1.id_a"]
    finally:
        s.stop()
