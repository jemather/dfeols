"""IV (2SLS, LIML) in dfeols against pyfixest, and its own consistency checks
(docs/design/iv.md)."""

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

TOL = dict(rtol=1e-8, atol=1e-10)


@pytest.fixture(scope="module")
def data():
    return iv_panel()


def iv_panel():
    """sim_panel plus endogenous d, d2 (correlated with the error u) and instruments z1-z3."""
    df = sim_panel()
    rng = np.random.default_rng(1)
    n = df.height
    u = rng.normal(size=n)
    z = rng.normal(size=(n, 3))
    wk = rng.normal(size=df["week"].max() + 1)[df["week"].to_numpy()]
    d = 0.6 * z[:, 0] + 0.4 * z[:, 1] + 0.2 * z[:, 2] + 0.5 * u + wk + rng.normal(size=n)
    d2 = 0.5 * z[:, 2] - 0.3 * z[:, 0] + 0.4 * u + rng.normal(size=n)
    feat = df["feat"].to_numpy()
    y = 0.8 * d - 0.4 * d2 + 0.05 * df["posts"].to_numpy() + wk + u * (1 + 0.5 * feat)
    return df.with_columns(
        pl.Series("d", d),
        pl.Series("d2", d2),
        pl.Series("z1", z[:, 0]),
        pl.Series("z2", z[:, 1]),
        pl.Series("z3", z[:, 2]),
        pl.Series("yiv", y),
    )


@pytest.fixture(scope="module")
def ds(data, tmp_path_factory):
    return dfeols.prep(
        data,
        shard_dir=tmp_path_factory.mktemp("shards"),
        partition="id_a",
        fixed_effects=["spell", "week", "id_a"],
        clusters=["id_a", "week"],
        target_rows=3000,
    )


def _pf_vcov(v, cl):
    return {"CRV1": cl} if v == "CRV1" else v


def _compare(ds, data, fml, vcov, cluster=None, weights=None, **kw):
    r = dfeols.fit(ds, fml, cluster=cluster, vcov=vcov, weights=weights, log_every=1e9, **kw)
    pdf = data.to_pandas()
    for v in [vcov] if isinstance(vcov, str) else vcov:
        m = pf.feols(fml, data=pdf, vcov=_pf_vcov(v, cluster), weights=weights, **pyfixest_tol_kwargs(1e-13))
        t = m.tidy()
        ours = r.tidy(None, v)
        np.testing.assert_allclose(ours["Estimate"], t.loc[r.coefnames, "Estimate"], **TOL)
        np.testing.assert_allclose(ours["Std. Error"], t.loc[r.coefnames, "Std. Error"], rtol=1e-8)
        np.testing.assert_allclose(ours["Pr(>|t|)"], t.loc[r.coefnames, "Pr(>|t|)"], rtol=1e-5, atol=1e-12)
        assert r.N == m._N
    return r


@pytest.mark.parametrize(
    "fml,crossing_fe",
    [
        ("yiv ~ posts | spell | d ~ z1 + z2", "auto"),  # local FE only
        ("yiv ~ posts | spell + week | d ~ z1 + z2 + z3", "dense"),  # crossing FE as dummies
        ("yiv ~ posts | spell + week | d ~ z1 + z2 + z3", "auto"),  # sparse crossing-FE path
        ("yiv ~ posts | d ~ z1 + z2", "auto"),  # no FE, intercept
        ("yiv ~ 1 | spell | d ~ z1", "auto"),  # just identified, no exogenous regressors
    ],
)
@pytest.mark.parametrize("vcov,cluster", [(["iid", "hetero"], None), ("CRV1", "id_a"), ("CRV1", "week")])
def test_2sls_matches_pyfixest(ds, data, fml, crossing_fe, vcov, cluster):
    _compare(ds, data, fml, vcov, cluster=cluster, crossing_fe=crossing_fe)


@pytest.mark.parametrize("crossing_fe", ["dense", "auto"])
def test_2sls_weights(ds, data, crossing_fe):
    _compare(
        ds,
        data,
        "yiv ~ posts | spell + week | d ~ z1 + z2",
        ["iid", "hetero"],
        weights="wt",
        crossing_fe=crossing_fe,
    )


def test_first_stage_f_matches_pyfixest(ds, data):
    """pyfixest's first-stage F under the fit's vcov, and its effective F (which it
    computes with hetero errors)."""
    fml = "yiv ~ 1 | spell | d ~ z1 + z2"
    pdf = data.to_pandas()
    for v in ("iid", "hetero"):
        r = dfeols.fit(ds, fml, vcov=v, log_every=1e9)
        m = pf.feols(fml, data=pdf, vcov=v, **pyfixest_tol_kwargs(1e-13))
        m.IV_Diag()
        fs = r.iv["first_stage"]["per_endogenous"][0]
        f = {t["name"]: t["statistic"] for t in fs["f_stat"]}
        np.testing.assert_allclose(
            f["f_iid" if v == "iid" else "wald_hetero"], m._f_stat_1st_stage, rtol=1e-8
        )
        if v == "hetero":
            eff = {e["vcov"]: e["statistic"] for e in fs["effective_f"]}
            np.testing.assert_allclose(eff["hetero"], m._eff_F, rtol=1e-8)


def test_iid_wald_equals_partial_f(ds):
    r = dfeols.fit(ds, "yiv ~ posts | spell + week | d ~ z1 + z2 + z3", vcov=["iid", "hetero"], log_every=1e9)
    fs = r.iv["first_stage"]["per_endogenous"][0]
    assert 0 < fs["partial_r2"] < fs["r2"] < 1
    eff = {e["vcov"]: e["statistic"] for e in fs["effective_f"]}
    assert eff["iid"] == pytest.approx(fs["f_stat"][0]["statistic"], rel=1e-10)


def test_sparse_equals_dense_two_endogenous(ds):
    fml = "yiv ~ posts | spell + week | d + d2 ~ z1 + z2 + z3"
    kw = dict(vcov=["iid", "hetero"], log_every=1e9)
    a = dfeols.fit(ds, fml, crossing_fe="dense", **kw)
    b = dfeols.fit(ds, fml, crossing_fe="auto", **kw)
    assert b.options["crossing_fe"] == "sparse"
    for v in ("iid", "hetero"):
        np.testing.assert_allclose(a.tidy(None, v).to_numpy(), b.tidy(None, v).to_numpy(), rtol=1e-8)
    for fa, fb in zip(a.iv["first_stage"]["per_endogenous"], b.iv["first_stage"]["per_endogenous"]):
        for ta, tb in zip(fa["f_stat"], fb["f_stat"]):
            assert ta["statistic"] == pytest.approx(tb["statistic"], rel=1e-8)


def test_liml_just_identified_equals_2sls(ds):
    fml = "yiv ~ posts | spell | d ~ z1"
    a = dfeols.fit(ds, fml, vcov="hetero", log_every=1e9)
    b = dfeols.fit(ds, fml, vcov="hetero", estimator="liml", log_every=1e9)
    assert b.outcomes["yiv"]["kappa"] == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(a.tidy().to_numpy(), b.tidy().to_numpy(), rtol=1e-8)


def test_not_identified(ds):
    with pytest.raises(ValueError, match="not identified"):
        dfeols.fit(ds, "yiv ~ posts | spell | d + d2 ~ z1", log_every=1e9)


def test_estimator_needs_iv(ds):
    with pytest.raises(ValueError, match="IV formulas"):
        dfeols.fit(ds, "yiv ~ posts | spell", estimator="liml", log_every=1e9)


@pytest.mark.parametrize(
    "fml,kw",
    [
        ("yiv ~ posts | spell + week | d + d2 ~ z1 + z2 + z3", dict(cluster="id_a")),
        ("yiv ~ posts | d ~ z1 + z2", dict(vcov="hetero", estimator="liml", fuller=1.0)),
    ],
)
def test_record_claims_iv_and_validates(ds, tmp_path, fml, kw):
    est = pytest.importorskip("estimation_format")
    pytest.importorskip("jsonschema")
    dfeols.fit(ds, fml, workdir=tmp_path, log_every=1e9, **kw)
    path = tmp_path / "result.estimation.json"
    rec = est.read(path, nonfinite="string")
    bad = [i for i in est.validate(rec, base_dir=tmp_path) if i.level != "info"]
    assert not bad, "\n".join(map(str, bad))
    assert "iv/0.1" in rec["meta"]["profiles"]
    assert rec["specification"]["instruments"][0] == "z1"
    fs = [p for p in rec["parameters"] if p.get("component") == "first_stage"]
    assert {p["response"] for p in fs} == set(rec["specification"]["endogenous"])
    main = [p["term"] for p in rec["parameters"] if "component" not in p]
    assert main[-len(rec["specification"]["endogenous"]) :] == rec["specification"]["endogenous"]
    per = rec["structures"]["first_stage"]["per_endogenous"]
    assert all(e["f_stat"]["statistic"] > 0 for e in per)
    assert {t["name"] for t in rec["structures"]["first_stage"]["overid"]} >= {"sargan"}
    if kw.get("estimator") == "liml":
        assert rec["meta"]["estimator"]["method"] == "fuller" and rec["metrics"]["fuller"] == 1.0
        assert rec["metrics"]["kappa"] > 1  # LIML's kappa >= 1; Fuller subtracts only 1 / (N - L)


def test_liml_matches_textbook(ds, data):
    """LIML coefficients and kappa against a direct computation on spell-demeaned data:
    kappa = min eig of (W'M_X1 W)(W'M_Z W)^-1, W = [y, X2], then the k-class formula."""
    r = dfeols.fit(ds, "yiv ~ posts | spell | d + d2 ~ z1 + z2 + z3", estimator="liml", log_every=1e9)
    pdf = data.to_pandas()
    cols = ["yiv", "posts", "d", "d2", "z1", "z2", "z3"]
    dm = pdf[cols] - pdf.groupby("spell")[cols].transform("mean")
    y, x1, x2 = dm[["yiv"]].to_numpy(), dm[["posts"]].to_numpy(), dm[["d", "d2"]].to_numpy()
    Z = np.column_stack([x1, dm[["z1", "z2", "z3"]].to_numpy()])

    def resid(a, b):
        return a - b @ np.linalg.lstsq(b, a, rcond=None)[0]

    W = np.column_stack([y, x2])
    kappa = min(np.linalg.eigvals(np.linalg.solve(resid(W, Z).T @ W, resid(W, x1).T @ W)).real)
    X = np.column_stack([x1, x2])
    Xk = X - kappa * resid(X, Z)
    beta = np.linalg.solve(Xk.T @ X, Xk.T @ y)[:, 0]
    assert r.outcomes["yiv"]["kappa"] == pytest.approx(kappa, rel=1e-9)
    np.testing.assert_allclose(r.coef(), beta, rtol=1e-8)


def test_two_endogenous_match_fixest(ds, data, tmp_path):
    """Coefficients, clustered SEs and the first-stage F statistics (fitstat ivf, ivwald)
    against fixest, which (unlike pyfixest) takes several endogenous regressors.

    fixest's iid ivf takes its denominator df from the clustered fit's K, where an FE nested
    in the cluster counts once; dfeols's iid F counts every absorbed level (as pyfixest does
    for an iid fit), so the two differ by the ratio of the df."""
    import json
    import shutil
    import subprocess

    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript not found")
    fml = "yiv ~ posts | spell + week | d + d2 ~ z1 + z2 + z3"
    data.write_csv(tmp_path / "d.csv")
    script = f"""
    suppressMessages({{library(fixest); library(jsonlite)}})
    df <- read.csv("{tmp_path / "d.csv"}")
    m <- feols({fml}, df, cluster = ~id_a, fixef.tol = 1e-11)
    ct <- coeftable(m)
    out <- list(names = rownames(ct), b = unname(ct[, 1]), se = unname(ct[, 2]), n = nobs(m),
                ivf = lapply(fitstat(m, ~ivf), function(x) x$stat),
                ivf_df2 = lapply(fitstat(m, ~ivf), function(x) x$df2),
                ivwald = lapply(fitstat(m, ~ivwald), function(x) x$stat),
                sargan = fitstat(m, ~sargan)$sargan$stat)
    cat(toJSON(out, digits = NA, auto_unbox = TRUE))
    """
    p = subprocess.run([rscript, "-e", script], capture_output=True, text=True)
    if p.returncode or "there is no package" in p.stderr:
        pytest.skip(f"fixest/jsonlite not usable: {p.stderr[-300:]}")
    fx = json.loads(p.stdout)
    r = dfeols.fit(ds, fml, cluster="id_a", log_every=1e9)
    names = [n.removeprefix("fit_") for n in fx["names"]]
    t = r.tidy().loc[names]
    np.testing.assert_allclose(t["Estimate"], fx["b"], rtol=1e-7)
    np.testing.assert_allclose(t["Std. Error"], fx["se"], rtol=1e-7)
    assert r.N == fx["n"]
    for e in r.iv["first_stage"]["per_endogenous"]:
        f = {x["name"]: x["statistic"] for x in e["f_stat"]}
        key = next(k for k in fx["ivf"] if k.split("::")[1] == e["endogenous"])
        wkey = next(k for k in fx["ivwald"] if k.split("::")[1] == e["endogenous"])
        f_iid = next(x for x in e["f_stat"] if x["name"] == "f_iid")
        assert f_iid["statistic"] * fx["ivf_df2"][key] / f_iid["df2"] == pytest.approx(
            fx["ivf"][key], rel=1e-7
        )
        assert f["wald_CRV1"] == pytest.approx(fx["ivwald"][wkey], rel=1e-7)
    sargan = {t["name"]: t for t in r.outcomes["yiv"]["overid"]}["sargan"]
    # small here (valid instruments): a difference of large sums, limited by fixest's FE tolerance
    assert sargan["statistic"] == pytest.approx(fx["sargan"], rel=1e-5) and sargan["df"] == 1


@pytest.mark.parametrize("fuller", [None, 1.0])
@pytest.mark.parametrize("vcov,cluster", [("hetero", None), ("CRV1", "id_a")])
def test_liml_matches_linearmodels(ds, data, fuller, vcov, cluster):
    """LIML and Fuller without FEs: kappa, coefficients and robust/clustered SEs with
    linearmodels' debiased small-sample factors (fixest's N/(N-K), G/(G-1)(N-1)/(N-K))."""
    lm = pytest.importorskip("linearmodels")
    fml = "yiv ~ posts | d + d2 ~ z1 + z2 + z3"
    r = dfeols.fit(ds, fml, vcov=vcov, cluster=cluster, estimator="liml", fuller=fuller, log_every=1e9)
    pdf = data.to_pandas()
    pdf["Intercept"] = 1.0
    mod = lm.IVLIML(
        pdf["yiv"], pdf[["Intercept", "posts"]], pdf[["d", "d2"]], pdf[["z1", "z2", "z3"]], fuller=fuller or 0
    )
    kw = {"cov_type": "robust"} if vcov == "hetero" else {"cov_type": "clustered", "clusters": pdf["id_a"]}
    res = mod.fit(debiased=True, **kw)
    assert r.outcomes["yiv"]["kappa"] == pytest.approx(res.kappa, rel=1e-9)
    ov = {x["name"]: x["statistic"] for x in r.outcomes["yiv"]["overid"]}
    assert ov["sargan"] == pytest.approx(res.sargan.stat, rel=1e-8)
    if fuller is None:
        assert ov["anderson_rubin"] == pytest.approx(res.anderson_rubin.stat, rel=1e-8)
    t = r.tidy()
    np.testing.assert_allclose(t["Estimate"], res.params[r.coefnames], rtol=1e-8)
    np.testing.assert_allclose(t["Std. Error"], res.std_errors[r.coefnames], rtol=1e-7)


def _stata():
    import os
    import shutil

    if os.environ.get("DFEOLS_STATA"):
        return os.environ["DFEOLS_STATA"]
    for name in ("stata-mp", "stata-se", "stata"):
        if shutil.which(name):
            return shutil.which(name)
    for path in sorted(Path("/usr/local").glob("stata*/stata-mp"), reverse=True):
        return str(path)
    return None


IVREGHDFE = {
    # name: (dfeols formula and fit options, ivreghdfe command)
    "liml_robust": (
        ("yiv ~ posts | spell | d + d2 ~ z1 + z2 + z3", dict(vcov="hetero", estimator="liml")),
        "ivreghdfe yiv posts (d d2 = z1 z2 z3), absorb(spell) liml robust small",
    ),
    "liml_cluster": (
        ("yiv ~ posts | spell | d + d2 ~ z1 + z2 + z3", dict(cluster="id_a", estimator="liml")),
        "ivreghdfe yiv posts (d d2 = z1 z2 z3), absorb(spell) liml cluster(id_a) small",
    ),
    "fuller_robust": (
        ("yiv ~ posts | spell | d ~ z1 + z2 + z3", dict(vcov="hetero", estimator="liml", fuller=1.0)),
        "ivreghdfe yiv posts (d = z1 z2 z3), absorb(spell) fuller(1) robust small",
    ),
    "2sls_first": (
        ("yiv ~ posts | spell | d ~ z1 + z2 + z3", dict(vcov="iid")),
        "ivreghdfe yiv posts (d = z1 z2 z3), absorb(spell) small ffirst",
    ),
}


@pytest.fixture(scope="module")
def ivreghdfe(data, tmp_path_factory):
    """Run every IVREGHDFE command in one Stata batch; per spec, a dict of b/se by name,
    N, lambda, kclass, fuller and e(first) rows."""
    import subprocess

    stata = _stata()
    if stata is None:
        pytest.skip("Stata not found (set DFEOLS_STATA)")
    out = tmp_path_factory.mktemp("ivreghdfe")
    data.write_csv(out / "d.csv")
    probe = out / "probe.do"
    probe.write_text("sysuse auto, clear\nivreghdfe price (weight = length), absorb(turn)\n")
    subprocess.run([stata, "-b", "do", probe.name], cwd=out, capture_output=True, text=True)
    tail = (out / "probe.log").read_text().split("ivreghdfe price")[-1]
    if "r(" in tail:
        err = [ln for ln in tail.splitlines() if ln.strip() and not ln.startswith(("(", ".", ">"))]
        pytest.skip(
            f"ivreghdfe doesn't run in this Stata ({' / '.join(err[:2])}); "
            "it needs ivreg2 (with livreg2.mlib), ranktest and reghdfe"
        )
    lines = [f'import delimited using "{out / "d.csv"}", clear']
    for name, (_, cmd) in IVREGHDFE.items():
        lines += [
            cmd,
            f'file open fh using "{out / name}.txt", write replace',
            "matrix b = e(b)",
            "matrix V = e(V)",
            "local names : colnames b",
            "forvalues i = 1/`=colsof(b)' {",
            "  local nm : word `i' of `names'",
            "  file write fh \"b `nm' \" %21.15g (b[1,`i']) _n",
            "  file write fh \"se `nm' \" %21.15g (sqrt(V[`i',`i'])) _n",
            "}",
            'file write fh "N " %21.15g (e(N)) _n',
            "foreach s in lambda kclass fuller sargan arubin arubin_lin {",
            "  capture file write fh \"`s' \" %21.15g (e(`s')) _n",
            "}",
            "capture confirm matrix e(first)",
            "if !_rc {",
            "  matrix F = e(first)",
            "  local rows : rownames F",
            "  local cols : colnames F",
            "  forvalues i = 1/`=rowsof(F)' {",
            "    forvalues j = 1/`=colsof(F)' {",
            "      local r : word `i' of `rows'",
            "      local c : word `j' of `cols'",
            "      file write fh \"first `r'.`c' \" %21.15g (F[`i',`j']) _n",
            "    }",
            "  }",
            "}",
            "file close fh",
        ]
    (out / "run.do").write_text("\n".join(lines) + "\n")
    subprocess.run([stata, "-b", "do", "run.do"], cwd=out, capture_output=True, text=True)
    log = (out / "run.log").read_text()
    assert "\nr(" not in log, log[-3000:]
    res = {}
    for name in IVREGHDFE:
        d = {}
        for line in (out / f"{name}.txt").read_text().splitlines():
            *key, val = line.split()
            d[" ".join(key)] = float("nan") if val == "." else float(val)  # . is Stata's missing
        res[name] = d
    return res


@pytest.mark.parametrize("name", list(IVREGHDFE))
def test_matches_ivreghdfe(ds, ivreghdfe, name):
    (fml, kw), _ = IVREGHDFE[name]
    st = ivreghdfe[name]
    r = dfeols.fit(ds, fml, log_every=1e9, **kw)
    assert r.N == st["N"]
    t = r.tidy()
    for c in r.coefnames:
        assert t.loc[c, "Estimate"] == pytest.approx(st[f"b {c}"], rel=1e-7), c
        assert t.loc[c, "Std. Error"] == pytest.approx(st[f"se {c}"], rel=1e-6), c
    ov = {x["name"]: x["statistic"] for x in r.outcomes["yiv"]["overid"]}
    if not np.isnan(st["sargan"]):  # ivreg2 reports Hansen's J instead under robust errors
        assert ov["sargan"] == pytest.approx(st["sargan"], rel=1e-7)
    if kw.get("estimator") == "liml":
        assert r.outcomes["yiv"]["kappa"] == pytest.approx(st["kclass"], rel=1e-9)
        assert ov["anderson_rubin"] == pytest.approx(st["arubin"], rel=1e-7)
        assert ov["anderson_rubin_linear"] == pytest.approx(st["arubin_lin"], rel=1e-7)
    if name == "2sls_first":
        e = r.iv["first_stage"]["per_endogenous"][0]
        f = {x["name"]: x["statistic"] for x in e["f_stat"]}
        assert f["f_iid"] == pytest.approx(st["first F.d"], rel=1e-6)
        assert e["partial_r2"] == pytest.approx(st["first pr2.d"], rel=1e-7)
