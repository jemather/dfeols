"""Two-way clustering in sessions (Ted 2026-11-08): CRV1 on two declared clusters by
Cameron, Gelbach and Miller (V_a + V_b - V_ab, each with the smaller cluster count, t df
G_min - 1), equal to pyfixest's `vcov={"CRV1": "a+b"}` for each kind of pair: one cluster
nested in the partition, or both crossing it (the intersection's scores summed sparsely)."""

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

pf = pytest.importorskip("pyfixest")
pytest.importorskip("jsonschema")
warnings.filterwarnings("ignore")


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


@pytest.mark.parametrize("max_dummy_levels", [2000, 10], ids=["dense", "iterate"])
def test_twoway_equals_pyfixest(ds, tmp_path, max_dummy_levels):
    pdf = _data().to_pandas()
    r = {"root": tmp_path / "s", "demeaned": tmp_path / "d"}
    with Session.start(
        "tw",
        ds.root,
        "spell + week",
        ["eng", "posts", "feat"],
        cluster="id_a, week, id_f, spell",
        poll=0.05,
        progress=False,
        max_dummy_levels=max_dummy_levels,
        iter_tol=1e-12,
        **r,
    ) as s:
        s.add_variable("fp", "feat * posts")
        for cl in ("id_a+week", "week+id_f", "spell+id_f", ["id_f", "id_a"]):
            a = s.fit("eng ~ posts + fp", cluster=cl, vcov=["CRV1", "hetero"])
            name = cl if isinstance(cl, str) else "+".join(cl)
            b = pf.feols(
                "eng ~ posts + feat:posts | spell + week",
                data=pdf,
                vcov={"CRV1": name},
                **pyfixest_tol_kwargs(1e-12),
            )
            assert a.N == b._N
            np.testing.assert_allclose(a.coef(), b.coef().values, rtol=1e-9, atol=1e-13)
            np.testing.assert_allclose(np.sqrt(np.diag(a.vcov(type="CRV1"))), b.se().values, rtol=1e-8)
            assert a.outcomes["eng"]["vcov"]["CRV1"][1] == b._df_t
            assert a.cluster == name
        m = s.fit("eng ~ posts", cluster="id_a", label="one")
        t = s.refit(m, vcov="CRV1", cluster="id_a+week")
        assert t.cluster == "id_a+week" and t.label == "one.crv1.id_a_week"
        rec = s.record(t)
        with pytest.raises(SessionError, match="one or two"):
            s.fit("eng ~ posts", cluster="id_a+week+id_f")
        with pytest.raises(SessionError, match="twice"):
            s.fit("eng ~ posts", cluster="id_a+id_a")
    issues = [i for i in estimation.validate(rec) if i.level != "info"]
    assert not issues, "\n".join(map(str, issues))
    (v,) = [v for v in rec["structures"]["vcov"] if v["type"] == "CRV1"]
    assert v["cluster"] == ["id_a", "week"]
    assert set(v["n_clusters"]) == {"id_a", "week"} and v["n_clusters_min"] == min(v["n_clusters"].values())
    assert "Cameron" in v["ssc"]["formula"]
    assert rec["specification"]["clusters"] == ["id_a", "week"]
