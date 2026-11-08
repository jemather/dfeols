"""IV in sessions (build order step 6): 2SLS, LIML and Fuller from the Gram matrix of the
residualised columns, equal to dfeols.fit's IV (itself checked against pyfixest, fixest,
linearmodels and ivreghdfe) on the same sample: coefficients, every vcov, kappa, first-stage
statistics and the overidentification tests; dense and iterated crossing FEs; created
instruments; two-way clustering against pyfixest; records validate."""

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import estrec as estimation

import dfeols
from dfeols.formula import pyfixest_tol_kwargs
from dfeols.session import Session, SessionError
from simdata import sim_panel

pytest.importorskip("jsonschema")
warnings.filterwarnings("ignore")
VARS = ["eng", "eng2", "posts", "feat", "tenure"]


def _data():
    return sim_panel().drop_nulls().filter(pl.col("eng").is_finite())


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return dfeols.prep(
        _data().with_columns(f2=pl.col("feat") ** 2),
        shard_dir=tmp_path_factory.mktemp("shards"),
        partition="id_a",
        fixed_effects=["spell", "week"],
        clusters=["id_a", "week"],
        target_rows=3000,
    )


def _fit(ds, tmp_path, formula, **kw):
    return dfeols.fit(ds, formula, workdir=tmp_path, log_every=1e9, record=False, fe_tol=1e-12, **kw)


def _same_iv(a, b, rtol):
    assert a.N == b.N and a.coefnames == b.coefnames
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=rtol, atol=1e-12)
    oa, ob = a.outcomes[next(iter(a.outcomes))], b.outcomes[next(iter(b.outcomes))]
    for t in oa["vcov"]:
        np.testing.assert_allclose(a.vcov(type=t), b.vcov(type=t), rtol=rtol * 10, atol=1e-14)
        assert oa["vcov"][t][1] == ob["vcov"][t][1]
    np.testing.assert_allclose(oa["kappa"], ob["kappa"], rtol=1e-10)
    np.testing.assert_allclose(oa["rss"], ob["rss"], rtol=rtol)
    for ta, tb in zip(oa["overid"], ob["overid"]):
        assert ta["name"] == tb["name"]
        np.testing.assert_allclose(ta["statistic"], tb["statistic"], rtol=rtol * 10)
    fa, fb = a.iv["first_stage"]["per_endogenous"], b.iv["first_stage"]["per_endogenous"]
    for ea, eb in zip(fa, fb):
        np.testing.assert_allclose(ea["coef"], eb["coef"], rtol=rtol)
        np.testing.assert_allclose(ea["partial_r2"], eb["partial_r2"], rtol=rtol * 10)
        for sa, sb in zip(ea["f_stat"], eb["f_stat"]):
            assert sa["name"] == sb["name"]
            np.testing.assert_allclose(sa["statistic"], sb["statistic"], rtol=rtol * 10)


@pytest.mark.parametrize("max_dummy_levels", [2000, 10], ids=["dense", "iterate"])
def test_iv_equals_fit(ds, tmp_path, max_dummy_levels):
    r = {"root": tmp_path / "s", "demeaned": tmp_path / "d"}
    rtol = 1e-9 if max_dummy_levels == 2000 else 1e-6
    with Session.start(
        "iv",
        ds.root,
        "spell + week",
        VARS,
        cluster="id_a, week",
        poll=0.05,
        progress=False,
        max_dummy_levels=max_dummy_levels,
        iter_tol=1e-12,
        fe_tol=1e-12,
        **r,
    ) as s:
        s.add_variable("f2", "feat^2")
        for kw in (
            {"vcov": ["iid", "hetero"]},
            {"cluster": "id_a", "vcov": ["CRV1", "iid"]},
            {"cluster": "week", "estimator": "liml"},
            {"vcov": "hetero", "estimator": "liml", "fuller": 1.0},
        ):
            a = s.fit("eng ~ eng2 | posts ~ feat + tenure + f2", **kw)
            b = _fit(
                ds, tmp_path / str(len(kw)), "eng ~ eng2 | spell + week | posts ~ feat + tenure + f2", **kw
            )
            _same_iv(a, b, rtol)
        # just identified, no exogenous regressors; two endogenous
        a = s.fit("eng ~ 1 | posts ~ feat", vcov="hetero")
        _same_iv(a, _fit(ds, tmp_path / "j", "eng ~ 1 | spell + week | posts ~ feat", vcov="hetero"), rtol)
        a = s.fit("eng ~ 1 | posts + eng2 ~ feat + tenure + f2", cluster="id_a")
        b = _fit(
            ds, tmp_path / "two", "eng ~ 1 | spell + week | posts + eng2 ~ feat + tenure + f2", cluster="id_a"
        )
        _same_iv(a, b, rtol)
        assert s.fit("eng ~ eng2 | posts ~ feat + tenure", vcov="iid").timing["passes"] == 0
        m = s.fit("eng ~ eng2 | posts ~ feat + tenure", vcov="iid", estimator="liml", label="liml")
        h = s.refit(m, vcov="hetero")
        assert h.iv["estimator"] == "liml" and h.timing["passes"] == 1
        np.testing.assert_array_equal(h.coef(), m.coef())
        rec = s.record(h)
        with pytest.raises(SessionError, match="not identified"):
            s.fit("eng ~ eng2 | posts + feat ~ tenure")
        with pytest.raises(SessionError, match="apply to IV"):
            s.fit("eng ~ eng2", estimator="liml")
    issues = [i for i in estimation.validate(rec) if i.level != "info"]
    assert not issues, "\n".join(map(str, issues))
    assert "iv/0.1" in rec["meta"]["profiles"]
    assert (
        "posts ~ feat + tenure" in rec["specification"]["call"]
        and "estimator='liml'" in rec["specification"]["call"]
    )


def test_iv_twoway_equals_pyfixest(ds, tmp_path):
    pf = pytest.importorskip("pyfixest")
    r = {"root": tmp_path / "s", "demeaned": tmp_path / "d"}
    with Session.start(
        "iv2",
        ds.root,
        "spell + week",
        VARS,
        cluster="id_a, week",
        poll=0.05,
        progress=False,
        fe_tol=1e-12,
        **r,
    ) as s:
        a = s.fit("eng ~ eng2 | posts ~ feat + tenure", cluster="id_a+week")
    b = pf.feols(
        "eng ~ eng2 | spell + week | posts ~ feat + tenure",
        data=_data().to_pandas(),
        vcov={"CRV1": "id_a+week"},
        **pyfixest_tol_kwargs(1e-12),
    )
    np.testing.assert_allclose(a.coef(), b.coef().values, rtol=1e-9)
    np.testing.assert_allclose(np.sqrt(np.diag(a.vcov(type="CRV1"))), b.se().values, rtol=1e-8)


def test_replay(ds, tmp_path, capsys):
    """A session collection's events run again in a new session give the same members; a
    changed estimate is reported."""
    import copy

    from dfeols.cli import main
    from dfeols.session import replay

    r = {"root": tmp_path / "s", "demeaned": tmp_path / "d"}
    kw = dict(cluster="id_a, week", poll=0.05, progress=False, **r)
    with Session.start("orig", ds.root, "spell + week", VARS, **kw) as s:
        s.add_variable("f2", "feat^2")
        m = s.fit("eng ~ eng2 + f2", cluster="id_a")
        s.refit(m, vcov="hetero")
        s.fit("eng + eng2 ~ posts", label="two")
        s.fit("eng ~ eng2 | posts ~ feat + f2", estimator="liml", cluster="id_a+week", label="iv")
        p = s.export()
    coll = estimation.read_collection(p)
    with Session.start("again", ds.root, "spell + week", VARS, **kw) as s2:
        rep = replay(coll, s2, rtol=1e-12)
        assert rep["ok"] and rep["key_matches"], rep
        assert [m_["label"] for m_ in rep["models"]] == ["m1", "m1.hetero", "two", "iv"]
        assert sum(len(m_["responses"]) for m_ in rep["models"]) == len(coll["members"]) == 5
        bad = copy.deepcopy(coll)
        bad["members"][0]["record"]["parameters"][0]["estimate"] *= 1.001
        rep = replay(bad, s2)
        assert not rep["ok"] and rep["mismatches"] == 1
        assert rep["models"][0]["responses"][0]["max_rel_diff"] == pytest.approx(1e-3 / 1.001)
        main(["session", "replay", "again", str(p), "--root", str(r["root"])])
        assert "5 members replayed, 0 different" in capsys.readouterr().out
