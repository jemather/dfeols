"""Two-way clustering in dfeols.fit (Ted's list, 2026-11-09; sessions had it since 2026-11-08):
CRV1 on two clusters by Cameron, Gelbach and Miller (V_a + V_b - V_ab, each with the smaller
cluster count, t df G_min - 1), equal to pyfixest's `vcov={"CRV1": "a+b"}` for each kind of
pair (one cluster nested in the partition, or both crossing it, the intersection's scores then
summed by code) on each crossing-FE path (dense, sparse, iterate), with and without the
demeaned cache, and for IV; the record names both clusters and validates."""

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
from simdata import sim_panel

pf = pytest.importorskip("pyfixest")
warnings.filterwarnings("ignore")

# id_a and spell are nested in the partition (id_a), week and id_f cross it: each kind of pair
PAIRS = ["id_a+week", "week+id_f", "spell+id_f", ["id_f", "id_a"], "spell+id_a"]


def _data():
    return sim_panel().drop_nulls().filter(pl.col("eng").is_finite())


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return dfeols.prep(
        _data(),
        shard_dir=tmp_path_factory.mktemp("shards"),
        partition="id_a",
        fixed_effects=["spell", "week"],
        clusters=["id_a", "week", "id_f", "spell"],
        target_rows=3000,
    )


@pytest.fixture(scope="module")
def pdf():
    return _data().to_pandas()


def _fit(ds, tmp_path, formula, **kw):
    kw.setdefault("record", False)
    return dfeols.fit(ds, formula, workdir=tmp_path, log_every=1e9, fe_tol=1e-12, iter_tol=1e-13, **kw)


def _check(a, b, name):
    assert a.N == b._N
    np.testing.assert_allclose(a.coef(), b.coef().values, rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(np.sqrt(np.diag(a.vcov(type="CRV1"))), b.se().values, rtol=1e-8)
    assert a.outcomes[next(iter(a.outcomes))]["vcov"]["CRV1"][1] == b._df_t
    assert a.cluster == name


@pytest.mark.parametrize(
    "crossing_fe,cache",
    [("dense", False), ("dense", True), ("sparse", False), ("iterate", False)],
    ids=["dense", "dense-cache", "sparse", "iterate"],
)
def test_twoway_equals_pyfixest(ds, pdf, tmp_path, crossing_fe, cache):
    for i, cl in enumerate(PAIRS):
        name = cl if isinstance(cl, str) else "+".join(cl)
        a = _fit(
            ds,
            tmp_path / str(i),
            "eng ~ posts + feat | spell + week",
            cluster=cl,
            vcov=["CRV1", "hetero"],
            crossing_fe=crossing_fe,
            cache=cache,
        )
        b = pf.feols(
            "eng ~ posts + feat | spell + week", data=pdf, vcov={"CRV1": name}, **pyfixest_tol_kwargs(1e-12)
        )
        _check(a, b, name)
        h = pf.feols(
            "eng ~ posts + feat | spell + week", data=pdf, vcov="hetero", **pyfixest_tol_kwargs(1e-12)
        )
        np.testing.assert_allclose(np.sqrt(np.diag(a.vcov(type="hetero"))), h.se().values, rtol=1e-8)


def test_twoway_iv_equals_pyfixest(ds, pdf, tmp_path):
    a = _fit(ds, tmp_path, "eng ~ 1 | spell + week | posts ~ feat", cluster="id_a+week")
    b = pf.feols(
        "eng ~ 1 | spell + week | posts ~ feat",
        data=pdf,
        vcov={"CRV1": "id_a+week"},
        **pyfixest_tol_kwargs(1e-12),
    )
    _check(a, b, "id_a+week")
    assert a.iv["first_stage"]["per_endogenous"]


def test_one_way_unchanged(ds, pdf, tmp_path):
    """One cluster: same spec id as before two-way existed, and still equal to pyfixest."""
    a = _fit(ds, tmp_path, "eng ~ posts | spell + week", cluster="week")
    b = pf.feols("eng ~ posts | spell + week", data=pdf, vcov={"CRV1": "week"}, **pyfixest_tol_kwargs(1e-12))
    _check(a, b, "week")
    assert "n_clusters" not in a.run


def test_twoway_record_and_summary(ds, tmp_path):
    pytest.importorskip("jsonschema")
    a = _fit(ds, tmp_path, "eng ~ posts | spell + week", cluster="id_a+week", record=True)
    assert set(a.run["n_clusters"]) == {"id_a", "week"} and a.G == min(a.run["n_clusters"].values())
    assert a.run["fe_nested_in"]["spell"] == ["id_a"] and a.run["fe_nested_in"]["week"] == ["week"]
    assert f"id_a {a.run['n_clusters']['id_a']:,}, week" in a.summary()
    path = tmp_path / "result.estrec.json"
    rec = estimation.read(path, nonfinite="string")
    issues = [i for i in estimation.validate(rec, base_dir=path.parent) if i.level != "info"]
    assert not issues, "\n".join(map(str, issues))
    (v,) = [v for v in rec["structures"]["vcov"] if v["type"] == "CRV1"]
    assert v["cluster"] == ["id_a", "week"] and v["n_clusters_min"] == a.G
    assert "Cameron" in v["ssc"]["formula"]
    assert rec["specification"]["clusters"] == ["id_a", "week"]


def test_refusals(ds, tmp_path):
    with pytest.raises(NotImplementedError, match="one or two"):
        _fit(ds, tmp_path, "eng ~ posts | spell + week", cluster="id_a+week+id_f")
    with pytest.raises(ValueError, match="twice"):
        _fit(ds, tmp_path, "eng ~ posts | spell + week", cluster="id_a+id_a")
    with pytest.raises(KeyError, match="not encoded"):
        _fit(ds, tmp_path, "eng ~ posts | spell + week", cluster="id_a+nope")
    with pytest.raises(NotImplementedError, match="two-way needs cells=False"):
        _fit(ds, tmp_path, "eng ~ posts | spell + week", cluster="id_a+week", cells=True)
