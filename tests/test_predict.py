"""fit(predict=...): in-sample xb, d (FE sum) and residuals equal pyfixest's on every FE path."""

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import dfeols
from dfeols.formula import pyfixest_tol_kwargs
from simdata import sim_panel

pf = pytest.importorskip("pyfixest")
warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def data():
    d = sim_panel(seed=5)
    return d.with_row_index("rid").with_columns(pl.col("rid").cast(pl.Int64))


@pytest.fixture(scope="module")
def ds(data, tmp_path_factory):
    return dfeols.prep(
        data,
        shard_dir=tmp_path_factory.mktemp("shards"),
        partition="id_a",
        fixed_effects=["spell", "week", "id_a^week", "id_f"],
        clusters=["id_a"],
        target_rows=3000,
    )


def _check(ds, data, fml, pf_fml, tmp_path, weights=None, **kw):
    out = tmp_path / "pred"
    r = dfeols.fit(ds, fml, cluster="id_a", weights=weights, predict=out, predict_keep=["rid"],
                   log_every=1e9, **kw)  # fmt: skip
    got = pl.read_parquet(out / "part-*.parquet").sort("rid")
    assert got.height == r.N
    m = pf.feols(pf_fml, data=data.to_pandas(), weights=weights, **pyfixest_tol_kwargs(1e-13))
    rid = m._data["rid"].to_numpy()
    order = np.argsort(rid)
    np.testing.assert_array_equal(got["rid"].to_numpy(), rid[order])
    np.testing.assert_allclose(got["resid"].to_numpy(), np.asarray(m.resid())[order], rtol=1e-7, atol=1e-8)
    yhat = np.asarray(m.predict())[order]
    np.testing.assert_allclose((got["xb"] + got["d"]).to_numpy(), yhat, rtol=1e-7, atol=1e-8)
    # xb is the reported regressors only: equal to X b on the raw rows
    return r, got


def test_local_fe_and_week_dummies(ds, data, tmp_path):
    _check(ds, data, "eng ~ feat:i(tbin) + posts | spell + week",
           "eng ~ i(tbin, feat) + posts | spell + week", tmp_path)  # fmt: skip


def test_two_local_fes_weighted(ds, data, tmp_path):
    _check(ds, data, "eng ~ feat:i(tbin) + posts | spell + id_a^week",
           "eng ~ i(tbin, feat) + posts | spell + id_a^week", tmp_path, weights="wt")  # fmt: skip


@pytest.mark.parametrize("crossing_fe", ["sparse", "iterate"])
def test_crossing_fe_paths(ds, data, tmp_path, crossing_fe):
    # id_f crosses the id_a partition: per-level sums, or plan C's CG
    _check(ds, data, "eng ~ feat + posts | spell + id_f", "eng ~ feat + posts | spell + id_f",
           tmp_path, crossing_fe=crossing_fe, iter_tol=1e-12)  # fmt: skip


def test_no_fe_intercept_in_xb(ds, data, tmp_path):
    r, got = _check(ds, data, "eng ~ feat + posts", "eng ~ feat + posts", tmp_path)
    np.testing.assert_allclose(got["d"].to_numpy(), 0.0, atol=1e-9)


def test_several_outcomes_and_errors(ds, data, tmp_path):
    out = tmp_path / "p2"
    dfeols.fit(ds, "eng + eng2 ~ feat | spell + week", predict=out, log_every=1e9)
    cols = pl.read_parquet(out / "part-*.parquet").columns
    assert {"xb_eng", "d_eng", "resid_eng", "xb_eng2", "resid_eng2"} <= set(cols)
    with pytest.raises(ValueError, match="cells"):
        dfeols.fit(ds, "eng ~ feat | spell + week", predict=out, cells=True, log_every=1e9)
    with pytest.raises(KeyError):
        dfeols.fit(ds, "eng ~ feat | spell", predict=out, predict_keep=["nope"], log_every=1e9)
