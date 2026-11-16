"""fit(subset=...): a row condition at fit time equals prepping the filtered rows, on every path
(local FEs, dense, sparse and iterated crossing FEs, two-way clusters, cells, IV, predict,
GLM), and is recorded (restriction, dropped counts, spec id)."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import dfeols
from dfeols.expr import ExpressionError, condition
from simdata import sim_panel

warnings.filterwarnings("ignore")
SUB = 'grp != 2 & region == "a"'
FES = ["spell", "week", "id_a^week", "id_f"]


@pytest.fixture(scope="module")
def data():
    d = sim_panel(seed=7)
    rng = np.random.default_rng(3)
    grp = rng.integers(0, 4, d.height).astype(float)
    grp[rng.uniform(size=d.height) < 0.02] = np.nan  # a NaN input: outside the subset
    region = np.where(rng.uniform(size=d.height) < 0.7, "a", "b").astype(object)
    region[rng.uniform(size=d.height) < 0.02] = None  # a null input: outside too
    z = d["posts"].to_numpy() + rng.normal(size=d.height)
    return d.with_row_index("rid").with_columns(
        pl.col("rid").cast(pl.Int64), grp=pl.Series(grp), region=pl.Series(region, dtype=pl.String),
        z=pl.Series(z),
    )


def _prep(df, path):
    return dfeols.prep(df, shard_dir=path, partition="id_a", fixed_effects=FES, clusters=["id_a", "id_f"],
                       target_rows=3000)  # fmt: skip


@pytest.fixture(scope="module")
def ds_full(data, tmp_path_factory):
    return _prep(data, tmp_path_factory.mktemp("full"))


@pytest.fixture(scope="module")
def ds_sub(data, tmp_path_factory):
    return _prep(data.filter(condition(SUB)), tmp_path_factory.mktemp("sub"))


def _same(a, b, rtol=1e-9):
    assert a.N == b.N
    assert a.coefnames == b.coefnames
    for y in a.outcomes:
        np.testing.assert_allclose(a.coef(y), b.coef(y), rtol=rtol, atol=1e-12)
        for t in a.options["vcov"]:
            np.testing.assert_allclose(a.vcov(y, type=t), b.vcov(y, type=t), rtol=max(rtol, 1e-8), atol=1e-14)


CASES = {
    "local": dict(formula="eng ~ posts + feat | spell", cluster="id_a"),
    "dense": dict(formula="eng ~ posts + feat | spell + week", cluster="id_a", crossing_fe="dense"),
    "sparse": dict(formula="eng ~ posts + feat | spell + id_a^week + id_f", cluster="id_a", crossing_fe="sparse"),
    "iterate": dict(formula="eng ~ posts + feat | spell + week + id_f", cluster="id_a", crossing_fe="iterate",
                    iter_tol=1e-12),
    "twoway": dict(formula="eng ~ posts + feat | spell + week", cluster="id_a+id_f"),
    "hetero_weights": dict(formula="eng ~ posts | spell + week", vcov=["hetero", "iid"], weights="wt"),
    "cells": dict(formula="eng ~ posts | spell + week", cluster="id_a", cells=True),
    "iv": dict(formula="eng ~ feat | spell + week | posts ~ z", cluster="id_a"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_subset_equals_filtered_prep(case, ds_full, ds_sub, tmp_path):
    kw = dict(CASES[case])
    f = kw.pop("formula")
    a = dfeols.fit(ds_full, f, subset=SUB, workdir=str(tmp_path / "a"), log_every=1e9, **kw)
    b = dfeols.fit(ds_sub, f, workdir=str(tmp_path / "b"), log_every=1e9, **kw)
    _same(a, b, rtol=1e-8 if case == "iterate" else 1e-9)


def test_record_and_counts(ds_full, ds_sub, data, tmp_path):
    a = dfeols.fit(ds_full, "eng ~ posts | spell + week", cluster="id_a", subset=SUB,
                   workdir=str(tmp_path / "a"), log_every=1e9)  # fmt: skip
    b = dfeols.fit(ds_sub, "eng ~ posts | spell + week", cluster="id_a", workdir=str(tmp_path / "b"),
                   log_every=1e9)  # fmt: skip
    outside = data.height - data.filter(condition(SUB)).height
    s = a.run["sample"]
    assert s["dropped"]["subset"] == outside
    assert s["dropped"]["missing"] == b.run["sample"]["dropped"]["missing"]  # counted within the subset
    assert "subset" not in b.run["sample"]["dropped"]
    assert SUB in a.run["restriction"]  # as written
    assert a.options["subset"] == SUB
    rec = json.loads((tmp_path / "a" / "result.estrec.json").read_text())
    assert rec["specification"]["sample"]["dropped"]["subset"] == outside
    assert SUB in rec["specification"]["sample"]["restriction"]
    assert f"subset={SUB!r}" in rec["specification"]["call"]
    # canonical text: spacing doesn't change the spec id, the condition does
    c = dfeols.fit(ds_full, "eng ~ posts | spell + week", cluster="id_a", subset='grp!=2&region=="a"',
                   log_every=1e9)  # fmt: skip
    d = dfeols.fit(ds_full, "eng ~ posts | spell + week", cluster="id_a", subset="grp != 2", log_every=1e9)
    e = dfeols.fit(ds_full, "eng ~ posts | spell + week", cluster="id_a", log_every=1e9)
    assert Path(c.workdir).name != Path(d.workdir).name != Path(e.workdir).name
    np.testing.assert_allclose(c.coef("eng"), a.coef("eng"), rtol=1e-12)


def test_predict(ds_full, ds_sub, tmp_path):
    f = "eng ~ posts + feat | spell + week"
    dfeols.fit(ds_full, f, cluster="id_a", subset=SUB, predict=tmp_path / "pa", predict_keep=["rid"],
               workdir=str(tmp_path / "a"), log_every=1e9)  # fmt: skip
    dfeols.fit(ds_sub, f, cluster="id_a", predict=tmp_path / "pb", predict_keep=["rid"],
               workdir=str(tmp_path / "b"), log_every=1e9)  # fmt: skip
    pa = pl.read_parquet(tmp_path / "pa" / "part-*.parquet").sort("rid")
    pb = pl.read_parquet(tmp_path / "pb" / "part-*.parquet").sort("rid")
    np.testing.assert_array_equal(pa["rid"].to_numpy(), pb["rid"].to_numpy())
    for c in ("xb", "d", "resid"):
        np.testing.assert_allclose(pa[c].to_numpy(), pb[c].to_numpy(), rtol=1e-8, atol=1e-10)


def test_glm(data, tmp_path):
    rng = np.random.default_rng(1)
    cnt = rng.poisson(np.exp(-1 + 0.5 * data["feat"].to_numpy()))
    d = data.with_columns(cnt=pl.Series(cnt.astype(float)))
    full = _prep(d, tmp_path / "full")
    sub = _prep(d.filter(condition(SUB)), tmp_path / "sub")
    f = "cnt ~ feat + posts | spell + week"
    a = dfeols.fit_glm(full, f, "poisson", cluster="id_a", subset=SUB, workdir=tmp_path / "a")
    b = dfeols.fit_glm(sub, f, "poisson", cluster="id_a", workdir=tmp_path / "b")
    assert a.N == b.N
    np.testing.assert_allclose(a.coef("cnt"), b.coef("cnt"), rtol=1e-8)
    np.testing.assert_allclose(a.vcov("cnt"), b.vcov("cnt"), rtol=1e-7)
    assert a.options["glm"]["separated_rows"] == b.options["glm"]["separated_rows"]


def test_errors(ds_full):
    with pytest.raises(ExpressionError, match="comparison"):
        dfeols.fit(ds_full, "eng ~ posts | spell", subset="grp + 1")
    with pytest.raises(KeyError, match="nope"):
        dfeols.fit(ds_full, "eng ~ posts | spell", subset="nope > 0")
    with pytest.raises(ValueError, match="outside subset="):
        dfeols.fit(ds_full, "eng ~ posts | spell", subset="grp > 100")


def test_factor_levels_from_the_subset(data, ds_full, tmp_path):
    """i()'s levels are the subset's: when it removes the reference level, the next one is the
    reference, as in a prep of the filtered rows (and fixest), not a collinear drop at the end."""
    cond = SUB + " & tbin != 0"
    sub = _prep(data.filter(condition(cond)), tmp_path / "sub")
    f = "eng ~ i(tbin) + posts | spell + week"
    a = dfeols.fit(ds_full, f, cluster="id_a", subset=cond, workdir=str(tmp_path / "a"), log_every=1e9)
    b = dfeols.fit(sub, f, cluster="id_a", workdir=str(tmp_path / "b"), log_every=1e9)
    assert "tbin::0" not in a.coefnames and "tbin::1" not in a.coefnames  # 1 is now the reference
    _same(a, b)
    # the whole dataset's levels still serve a fit without a subset (separate cache entries)
    c = dfeols.fit(ds_full, f, cluster="id_a", log_every=1e9)
    assert "tbin::1" in c.coefnames
