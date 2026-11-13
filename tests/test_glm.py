"""dfeols.fit_glm: Poisson equals pyfixest's fepois, cloglog equals R fixest's feglm."""

import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import dfeols
from simdata import sim_panel

pf = pytest.importorskip("pyfixest")
warnings.filterwarnings("ignore")
TOL = dict(rtol=1e-6, atol=1e-9)


@pytest.fixture(scope="module")
def data():
    d = sim_panel(seed=2, missing=0.0)
    rng = np.random.default_rng(1)
    a = d["id_a"].to_numpy()
    ae = 0.3 * rng.normal(size=a.max() + 1)[a]
    feat = d["feat"].to_numpy()
    cnt = rng.poisson(np.exp(-1 + 0.5 * feat + ae))
    p = -np.expm1(-np.exp(-1.5 + 0.6 * feat + ae))
    exit_ = (rng.uniform(size=len(p)) < p).astype(float)
    return d.with_columns(cnt=pl.Series(cnt.astype(float)), exit=pl.Series(exit_))


@pytest.fixture(scope="module")
def ds(data, tmp_path_factory):
    return dfeols.prep(
        data, shard_dir=tmp_path_factory.mktemp("sh"), partition="id_a",
        fixed_effects=["spell", "week", "id_f"], clusters=["id_a"], target_rows=3000,
    )  # fmt: skip


@pytest.mark.parametrize(
    "fml,kw",
    [
        ("cnt ~ feat + posts | spell + week", {}),
        ("cnt ~ feat + posts | spell + id_f", {"crossing_fe": "sparse"}),
        ("cnt ~ feat + posts | spell + id_f", {"crossing_fe": "iterate", "iter_tol": 1e-12}),
    ],
)
def test_poisson_equals_fepois(ds, data, tmp_path, fml, kw):
    vc = ["CRV1", "hetero", "iid"]
    r = dfeols.fit_glm(ds, fml, "poisson", cluster="id_a", vcov=vc, workdir=tmp_path, log_every=1e9, **kw)
    g = r.options["glm"]
    assert g["converged"] and g["separated_rows"] > 0
    for v, pv in [("CRV1", {"CRV1": "id_a"}), ("hetero", "hetero"), ("iid", "iid")]:
        m = pf.fepois(fml, data=data.to_pandas(), vcov=pv, iwls_tol=1e-12, fixef_tol=1e-13)
        assert r.N == m._N
        np.testing.assert_allclose(r.coef(), m.coef().to_numpy(), **TOL)
        np.testing.assert_allclose(np.sqrt(np.diag(r.vcov(type=v))), m.se().to_numpy(), rtol=1e-5)
    assert list(r.outcomes) == ["cnt"] and r.formula == fml
    assert not (tmp_path / "side0").exists()  # sidecars removed


def _fixest(csv, fml, family):
    code = f"""
suppressMessages(library(fixest)); suppressMessages(library(jsonlite))
d <- read.csv("{csv}")
m <- feglm({fml}, data = d, family = {family}, cluster = ~id_a, glm.tol = 1e-12, fixef.tol = 1e-11)
cat(toJSON(list(b = unname(coef(m)), se = unname(se(m)), n = nobs(m)), digits = NA))
"""
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True, timeout=300)
    if out.returncode:
        pytest.skip(f"R fixest not usable: {out.stderr[-300:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs R with fixest")
def test_cloglog_equals_fixest(ds, data, tmp_path):
    fml = "exit ~ feat + posts | spell + week"
    r = dfeols.fit_glm(ds, fml, "cloglog", cluster="id_a", workdir=tmp_path, tol=1e-12, log_every=1e9)
    assert r.options["glm"]["converged"]
    csv = tmp_path / "d.csv"
    data.select("exit", "feat", "posts", "spell", "week", "id_a").write_csv(csv)
    ref = _fixest(csv, fml, 'binomial(link = "cloglog")')
    assert r.N == ref["n"][0]
    np.testing.assert_allclose(r.coef(), ref["b"], **TOL)
    np.testing.assert_allclose(np.sqrt(np.diag(r.vcov())), ref["se"], rtol=1e-5)


def test_bad_outcome_and_options(ds, tmp_path):
    with pytest.raises(ValueError, match="0 or 1"):
        dfeols.fit_glm(ds, "cnt ~ feat | spell", "cloglog", workdir=tmp_path, log_every=1e9)
    with pytest.raises(NotImplementedError):
        dfeols.fit_glm(ds, "cnt ~ feat | spell", "poisson", weights="wt", workdir=tmp_path)
    with pytest.raises(ValueError, match="family"):
        dfeols.fit_glm(ds, "cnt ~ feat | spell", "logit", workdir=tmp_path)
