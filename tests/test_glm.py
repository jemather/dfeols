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



def test_singletons_and_separation_iterate_jointly(tmp_path):
    """A singleton drop can leave an FE group with only zeros; that group must go too, as in
    fixest (Weidner & Zylkin Table 4, isic 10, kept 8 extra rows before this was one loop).

    Group a=0 has rows (b=0, y=1), (b=1, y=0), (b=1, y=0). b=0 is a singleton; once it goes, a=0
    is all zeros and is separated, so N = 400 (a=1..100, each with b=1..4 and y >= 1)."""
    rng = np.random.default_rng(3)
    a = np.r_[0, 0, 0, np.repeat(np.arange(1, 101), 4)]
    b = np.r_[0, 1, 1, np.tile(np.arange(1, 5), 100)]
    x = rng.normal(size=len(a))
    y = np.r_[1.0, 0.0, 0.0, 1.0 + rng.poisson(np.exp(0.3 * x[3:]))]
    df = pl.DataFrame({"a": a, "b": b, "x": x, "y": y})
    ds = dfeols.prep(df, shard_dir=tmp_path / "sh", partition="a", fixed_effects=["a", "b"], target_rows=200)
    r = dfeols.fit_glm(ds, "y ~ x | a + b", "poisson", workdir=tmp_path / "w", log_every=1e9)
    assert r.N == 400
    m = pf.fepois("y ~ x | a + b", data=df.filter(pl.col("a") > 0).to_pandas(), iwls_tol=1e-12, fixef_tol=1e-13)
    np.testing.assert_allclose(r.coef(), m.coef().to_numpy(), **TOL)


def test_warm_start_and_step_halving(data, tmp_path, monkeypatch):
    """The IRLS warm start (worker._warm_start) only moves the demeaning's starting point, so
    the fit equals a cold one; and a step that raises the deviance is halved back (glm
    shard_glm_halve) and the fit still converges to the same estimates."""
    from dfeols import glm as G
    from dfeols import worker as Wk

    ds2 = dfeols.prep(
        data, shard_dir=tmp_path / "sh", partition="id_a", fixed_effects=["spell", "week"],
        clusters=["id_a"], target_rows=3000,
    )  # fmt: skip
    fml = "cnt ~ feat + posts | spell + week"
    kw = dict(cluster="id_a", log_every=1e9, fe_method="iterate", fe_tol=1e-13, tol=1e-12)
    warm = dfeols.fit_glm(ds2, fml, "poisson", workdir=tmp_path / "w", keep_sidecars=True, **kw)
    assert list((tmp_path / "w" / "side1").glob("warm-*.npz")), "round 1 saved no warm start"
    monkeypatch.setattr(G, "_WARM", False)
    cold = dfeols.fit_glm(ds2, fml, "poisson", workdir=tmp_path / "c", **kw)
    np.testing.assert_allclose(warm.coef(), cold.coef(), rtol=1e-9)
    assert warm.options["glm"]["step_halvings"] == [0] * warm.options["glm"]["iterations"]
    monkeypatch.setattr(G, "_WARM", True)

    # a bad second step: eta overshoots by 4x the change, which raises the deviance
    real, calls = Wk._write_glm_step, {"n": 0}

    def overshoot(plan, shard_id, df, n0, eta):
        calls["n"] += 1
        if 13 <= calls["n"] <= 24:  # the shards' writes in iteration 2 (12 shards or fewer)
            prev = pl.read_parquet(Wk.sidecar_path(plan.sidecar_dir, shard_id), columns=[G.ETA])[G.ETA]
            p = prev.to_numpy()[df["__row"].to_numpy()]
            eta = p + 5.0 * (eta - p)
        return real(plan, shard_id, df, n0, eta)

    monkeypatch.setattr(Wk, "_write_glm_step", overshoot)
    nshards = len(ds2.shards)
    calls["n"] = 12 - nshards  # so iteration 2's writes are calls 13 .. 12 + nshards
    bad = dfeols.fit_glm(ds2, fml, "poisson", workdir=tmp_path / "b", **kw)
    h = bad.options["glm"]["step_halvings"]
    assert h[1] > 0 and bad.options["glm"]["converged"], h
    np.testing.assert_allclose(bad.coef(), cold.coef(), rtol=1e-8)
