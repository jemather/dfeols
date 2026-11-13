"""dfeols.stack: stacked per-equation-FE systems and group subsamples.

Coefficients of the stacked fit equal each equation fitted alone (pyfixest), and the
stacked fit equals pyfixest on the same stacked rows (coefficients and CRV1 SEs).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import dfeols
from dfeols.formula import pyfixest_tol_kwargs, to_pyfixest
from simdata import sim_panel

pf = pytest.importorskip("pyfixest")
warnings.filterwarnings("ignore")
TOL = dict(rtol=1e-7, atol=1e-10)


@pytest.fixture(scope="module")
def data():
    d = sim_panel(seed=3, missing=0.0)
    rng = np.random.default_rng(7)
    n = d.height
    d = d.sort("spell", "week").with_columns(
        lfeat=pl.col("feat").shift(1).over("spell"),
        lposts=pl.col("posts").shift(1).over("spell"),
    )
    risk = rng.uniform(size=n) < 0.8
    unf = 0.02 - 0.03 * d["feat"].to_numpy() * (d["tbin"].to_numpy() - 2.5) + rng.normal(0, 0.1, n)
    leng = 0.4 * d["feat"].to_numpy() + rng.normal(0, 1, n)
    return d.with_columns(risk=pl.Series(risk), unf=pl.Series(unf), leng=pl.Series(leng))


def _pf(pdf, fml, cluster="id_a"):
    return pf.feols(fml, data=pdf, vcov={"CRV1": cluster}, **pyfixest_tol_kwargs(1e-13))


def _fit_stacked(st, spec, tmp_path, cluster="id_a"):
    df = st.frame.collect()
    ds = dfeols.prep(
        df, shard_dir=tmp_path / "sh", partition="id_f", fixed_effects=st.fixed_effects,
        clusters=[cluster], target_rows=2000,
    )  # fmt: skip
    r = dfeols.fit(ds, st.formula(spec), cluster=cluster, log_every=1e9)
    return r, df


def test_t1_stack_equals_separate_fits(data, tmp_path):
    eqs = [
        dfeols.Equation(1, y="eng", x={"x": "feat", "p": "posts"}, fe=["spell", "week"]),
        dfeols.Equation(
            2, y="unf", x={"x": "lfeat", "p": "lposts"}, fe=["id_f", "id_a", "week"], rows=pl.col("risk")
        ),
    ]
    st = dfeols.stack(data, eqs, partition="id_f", keep=["tbin", "id_a"])
    # spell and id_f are local (nested in id_f); week and id_a cross; no local filler needed
    assert st.slots["_fe_l0"] == {1: "spell", 2: "id_f"}
    assert st.slots["_fe_c1"][1] == "(filler: constant)"
    r, df = _fit_stacked(st, "x:i(tbin) + p:i(tbin)", tmp_path)

    pdf = data.to_pandas()
    m1 = _pf(pdf, "eng ~ i(tbin, feat) + i(tbin, posts) | spell + week")
    m2 = _pf(pdf[pdf["risk"]], "unf ~ i(tbin, lfeat) + i(tbin, lposts) | id_f + id_a + week")
    b = r.coef()
    for k, m, xv in [(1, m1, "feat"), (2, m2, "lfeat")]:
        ix = st.coef_index(r.coefnames, "x", k)
        names = [n for n in m.coef().index if f", {xv})" in n or n.endswith(f":{xv}")]
        assert len(ix) == len(names) == 6
        np.testing.assert_allclose(b[ix], m.coef()[names].to_numpy(), **TOL)
    assert r.N == m1._N + m2._N

    # the stacked fit is an ordinary fit: equal to pyfixest on the stacked rows, SEs too
    sp = df.to_pandas()
    ms = _pf(sp, to_pyfixest(st.formula("x:i(tbin) + p:i(tbin)")))
    t = r.tidy()
    np.testing.assert_allclose(t["Estimate"].to_numpy(), ms.coef()[t.index].to_numpy(), **TOL)
    np.testing.assert_allclose(t["Std. Error"].to_numpy(), ms.se()[t.index].to_numpy(), rtol=1e-7)


def test_mediation_system_same_rows(data, tmp_path):
    # T1M shape: a-path, total, direct + b-path; the mediator role exists only in k=3
    d = data.with_columns(pl.when(pl.col("id_a") % 7 == 0).then(None).otherwise(pl.col("leng")).alias("leng"))
    rows = pl.col("risk")
    eqs = [
        dfeols.Equation(1, y="leng", x={"x": "lfeat", "p": "lposts"}, fe=["id_f", "id_a"], rows=rows),
        dfeols.Equation(2, y="unf", x={"x": "lfeat", "p": "lposts"}, fe=["id_f", "id_a"], rows=rows),
        dfeols.Equation(
            3, y="unf", x={"x": "lfeat", "p": "lposts", "m": "leng"}, fe=["id_f", "id_a"], rows=rows
        ),
    ]
    st = dfeols.stack(d, eqs, partition="id_f", keep=["tbin", "id_a"], same_rows=True)
    assert st.terms("x:i(tbin) + m") == ("x_k1:i(tbin) + x_k2:i(tbin) + x_k3:i(tbin) + m_k3")
    r, df = _fit_stacked(st, "x:i(tbin) + p:i(tbin) + m", tmp_path)
    # every equation has the same rows
    counts = df.group_by("k").len().sort("k")["len"].to_list()
    assert counts[0] == counts[1] == counts[2]
    base = d.filter(rows).drop_nulls(["leng", "lfeat", "lposts", "unf"]).to_pandas()
    m3 = _pf(base, "unf ~ i(tbin, lfeat) + i(tbin, lposts) + leng | id_f + id_a")
    ib = st.coef_index(r.coefnames, "m", 3)
    np.testing.assert_allclose(r.coef()[ib], m3.coef()[["leng"]].to_numpy(), **TOL)


def test_interacted_fe_and_local_filler(data, tmp_path, caplog):
    # eq 2 has one local FE fewer: its second local slot is filled with id_f (redundant)
    eqs = [
        dfeols.Equation(1, y="eng", x={"x": "feat"}, fe=["spell", "id_f^week"]),
        dfeols.Equation(2, y="unf", x={"x": "lfeat"}, fe=["spell", "week"]),
    ]
    st = dfeols.stack(data, eqs, partition="id_f", keep=["tbin", "id_a"])
    assert st.slots["_fe_l1"] == {1: "id_f^week", 2: "(filler: id_f)"}
    assert "redundant" in caplog.text
    r, df = _fit_stacked(st, "x:i(tbin)", tmp_path)
    pdf = data.to_pandas()
    m1 = _pf(pdf, "eng ~ i(tbin, feat) | spell + id_f^week")
    m2 = _pf(pdf, "unf ~ i(tbin, lfeat) | spell + week")
    for k, m in [(1, m1), (2, m2)]:
        ix = st.coef_index(r.coefnames, "x", k)
        np.testing.assert_allclose(r.coef()[ix], m.coef().to_numpy(), **TOL)


def test_no_local_fe_for_needed_slot_errors(data):
    eqs = [
        dfeols.Equation(1, y="eng", x={"x": "feat"}, fe=["spell"]),
        dfeols.Equation(2, y="unf", x={"x": "lfeat"}, fe=["week"]),
    ]
    with pytest.raises(ValueError, match="no FE nested"):
        dfeols.stack(data, eqs, partition="id_f")


def test_sample_groups(data):
    s15 = data.filter(dfeols.sample_groups("id_f", 0.15, seed=1))
    s05 = data.filter(dfeols.sample_groups("id_f", 0.05, seed=1))
    nf = data["id_f"].n_unique()
    assert abs(s15["id_f"].n_unique() / nf - 0.15) < 0.03
    # whole groups: every row of a kept follower is kept
    kept = set(s15["id_f"].unique().to_list())
    assert data.filter(pl.col("id_f").is_in(list(kept))).height == s15.height
    # nested, and the same lazily/streaming and on any split of the rows
    assert set(s05["id_f"].unique().to_list()) <= kept
    lazy = data.lazy().filter(dfeols.sample_groups("id_f", 0.15, seed=1)).collect(engine="streaming")
    assert lazy.sort("spell", "week").equals(s15.sort("spell", "week"))
    half = pl.concat([data[: data.height // 2], data[data.height // 2 :]]).filter(
        dfeols.sample_groups("id_f", 0.15, seed=1)
    )
    assert half.height == s15.height
    other = data.filter(dfeols.sample_groups("id_f", 0.15, seed=2))
    assert set(other["id_f"].unique().to_list()) != kept
