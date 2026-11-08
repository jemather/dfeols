"""IV with crossing_fe="iterate" (plan C): the CG solution covers every column, so the k-class
solve runs on cross-products net of the crossing FEs; equal to the dense path and to
pyfixest, in level space and with a pass per CG step, with and without a local FE."""

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

warnings.filterwarnings("ignore")
FM = "eng ~ eng2 | spell + week | posts ~ feat + tenure"


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


def _fit(ds, tmp_path, fm=FM, **kw):
    return dfeols.fit(ds, fm, workdir=tmp_path, log_every=1e9, record=False, fe_tol=1e-12, **kw)


def _same(a, b, rtol):
    assert a.N == b.N and a.coefnames == b.coefnames
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=rtol, atol=1e-13)
    oa, ob = a.outcomes["eng"], b.outcomes["eng"]
    for t in oa["vcov"]:
        np.testing.assert_allclose(a.vcov(type=t), b.vcov(type=t), rtol=rtol, atol=1e-15)
        assert oa["vcov"][t][1] == ob["vcov"][t][1]
    np.testing.assert_allclose(oa["kappa"], ob["kappa"], rtol=1e-10)
    fa, fb = a.iv["first_stage"]["per_endogenous"], b.iv["first_stage"]["per_endogenous"]
    for ea, eb in zip(fa, fb):
        np.testing.assert_allclose(ea["coef"], eb["coef"], rtol=rtol)
        for sa, sb in zip(ea["f_stat"], eb["f_stat"]):
            np.testing.assert_allclose(sa["statistic"], sb["statistic"], rtol=rtol)


@pytest.mark.parametrize("level_space_gb", [None, 0], ids=["level", "pass-per-step"])
@pytest.mark.parametrize(
    "kw",
    [
        {"vcov": ["iid", "hetero"]},
        {"cluster": "id_a", "estimator": "liml"},
        {"cluster": "week", "estimator": "liml", "fuller": 1.0},
    ],
    ids=["2sls", "liml", "fuller"],
)
def test_iterate_equals_dense(ds, tmp_path, kw, level_space_gb):
    a = _fit(ds, tmp_path / "d", crossing_fe="dense", **kw)
    b = _fit(ds, tmp_path / "i", crossing_fe="iterate", iter_tol=1e-12, level_space_gb=level_space_gb, **kw)
    assert b.options["crossing_fe"] == "iterate" and b.iv["estimator"] == a.iv["estimator"]
    _same(a, b, 1e-8)


def test_two_endogenous(ds, tmp_path):
    fm = "eng ~ 1 | spell + week | posts + eng2 ~ feat + tenure + wt"
    a = _fit(ds, tmp_path / "d", fm, crossing_fe="dense", cluster="id_a")
    b = _fit(ds, tmp_path / "i", fm, crossing_fe="iterate", iter_tol=1e-12, cluster="id_a")
    _same(a, b, 1e-8)


def test_crossing_only_against_pyfixest(tmp_path):
    pf = pytest.importorskip("pyfixest")
    data = _data()
    ds = dfeols.prep(
        data,
        shard_dir=tmp_path / "sh",
        partition="tbin",
        fixed_effects=["spell", "week"],
        clusters=["id_a"],
        target_rows=3000,
    )
    a = _fit(ds, tmp_path / "f", crossing_fe="iterate", iter_tol=1e-12, cluster="id_a")
    assert a.options["crossing_fe"] == "iterate"  # no FE nested in tbin
    b = pf.feols(FM, data=data.to_pandas(), vcov={"CRV1": "id_a"}, **pyfixest_tol_kwargs(1e-12))
    np.testing.assert_allclose(a.coef(), b.coef().values, rtol=1e-7)
    np.testing.assert_allclose(np.sqrt(np.diag(a.vcov(type="CRV1"))), b.se().values, rtol=1e-6)


def test_cg_checkpoint_resumes_a_fit(ds, tmp_path, caplog):
    """Plan C with a pass per CG step: a fit stopped by iter_maxiter leaves its CG state in the
    workdir; the same fit again resumes from it and gives the uninterrupted answer."""
    import logging

    kw = dict(crossing_fe="iterate", iter_tol=1e-12, level_space_gb=0, cluster="id_a")
    clean = _fit(ds, tmp_path / "clean", **kw)
    wd = tmp_path / "w"
    with pytest.raises(RuntimeError, match="did not converge in 3 steps"):
        _fit(ds, wd, iter_maxiter=3, **kw)
    assert (wd / "cg_checkpoint.npz").exists()
    with caplog.at_level(logging.INFO, logger="dfeols"):
        again = _fit(ds, wd, **kw)
    assert "resumes at step 3" in caplog.text
    assert not (wd / "cg_checkpoint.npz").exists()
    cg = again.options["crossing_cg"]
    assert cg["resumed_from"] == 3 and cg["passes"] == cg["steps"] - 3
    _same(clean, again, 1e-10)
