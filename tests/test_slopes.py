"""Varying slopes, fe[x] (docs/design/slopes.md; Decisions 2026-11-10): fixest's syntax
(fe[x], fe[[x]] slope only, fe[x, z], a^b[x]), on FEs nested in the partition (one FE exactly,
several by alternating projections) and on crossing FEs as dense dummies. pyfixest has no
slopes, so the reference is fixest 0.14.2 through Rscript (skipped without R or fixest); the
demeaning is also checked against dummy regressions in NumPy, which always runs.

fixest counts every slope coefficient in K, identified or not, and dfeols does the same. One
difference from fixest is older than slopes: with two FEs nested in the cluster (id_a and spell,
clustered by id_a) fixest counts the second one's levels and pyfixest does not; dfeols follows
pyfixest, so the cases here do not cluster on such a pair."""

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
import estrec as estimation

import dfeols
from dfeols.demean import demean
from dfeols.formula import parse, split_fe
from simdata import sim_panel

warnings.filterwarnings("ignore")

# (formula, vcov, weights); vcov is iid, hetero or the cluster (a+b: two-way)
CASES = [
    ("eng ~ feat + posts | spell[tenure]", "iid", None),
    ("eng ~ feat + posts | spell[tenure]", "id_a", None),
    ("eng ~ feat + posts | spell[tenure]", "week", "wt"),
    ("eng ~ feat + posts | spell[[tenure]]", "iid", None),
    ("eng ~ feat + posts | spell[[tenure]]", "week", None),
    ("eng ~ feat + posts | id_a[week] + spell", "iid", None),
    ("eng ~ feat + posts | id_a[week] + spell", "week", None),
    ("eng ~ feat + posts | id_a[week] + spell", "hetero", "wt"),
    ("eng ~ feat + posts | spell + week[feat]", "iid", None),
    ("eng ~ feat + posts | spell + week[feat]", "id_a", None),
    ("eng ~ feat + posts | spell + week[tenure]", "id_a", "wt"),
    ("eng ~ feat + posts | spell + week^tbin[[posts]]", "week", None),
    ("eng ~ posts | id_a[feat, week] + week", "id_a", None),
    ("eng ~ posts | id_a[feat, week] + week", "id_a+week", None),
    ("eng ~ feat | week[[posts]]", "iid", None),
    ("eng ~ feat | week[[posts]]", "week", None),
    ("eng ~ feat + posts | id_a[[week]] + spell", "week", None),
    ("eng ~ posts | spell[tenure] | feat ~ tbin", "id_a", None),
    ("eng ~ posts | spell[tenure] + week | feat ~ tbin", "hetero", None),
]

R_SCRIPT = r"""
suppressMessages({library(fixest); library(jsonlite)})
a <- commandArgs(TRUE)
d <- read.csv(a[1]); cases <- fromJSON(a[2], simplifyVector = FALSE)
out <- lapply(cases, function(cs) {
  vc <- if (cs$vcov %in% c("iid", "hetero")) cs$vcov else as.formula(paste0("~", cs$vcov))
  w <- if (is.null(cs$weights)) NULL else as.formula(paste0("~", cs$weights))
  m <- feols(as.formula(cs$formula), d, vcov = vc, weights = w, fixef.tol = 1e-11, notes = FALSE)
  list(names = I(names(coef(m))), coef = I(unname(coef(m))), se = I(unname(se(m))), N = nobs(m),
       df_t = unname(attr(vcov(m, attr = TRUE), "df.t")), wr2 = unname(fitstat(m, "wr2")[[1]]))
})
write_json(out, a[3], digits = NA, auto_unbox = TRUE)
"""


def _has_fixest() -> bool:
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(["Rscript", "-e", "library(fixest); library(jsonlite)"], capture_output=True)
    return r.returncode == 0


needs_fixest = pytest.mark.skipif(not _has_fixest(), reason="needs Rscript with fixest and jsonlite")


def _data():
    return sim_panel().drop_nulls().filter(pl.col("eng").is_finite())


def _prep(tmp_path_factory, partition):
    return dfeols.prep(
        _data(),
        shard_dir=tmp_path_factory.mktemp(f"shards_{partition}"),
        partition=partition,
        fixed_effects=["spell", "week", "id_a", "week^tbin"],
        clusters=["id_a", "week"],
        target_rows=3000,
    )


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return _prep(tmp_path_factory, "id_a")


@pytest.fixture(scope="module")
def ds_week(tmp_path_factory):
    return _prep(tmp_path_factory, "week")


def _fit(ds, tmp_path, formula, vcov="iid", weights=None, **kw):
    kw.setdefault("record", False)
    if vcov in ("iid", "hetero"):
        kw["vcov"] = vcov
    else:
        kw["cluster"] = vcov
    return dfeols.fit(ds, formula, weights=weights, workdir=tmp_path, log_every=1e9, fe_tol=1e-13, **kw)


def _vtype(vcov):
    return vcov if vcov in ("iid", "hetero") else "CRV1"


@pytest.fixture(scope="module")
def fixest(tmp_path_factory):
    d = tmp_path_factory.mktemp("fixest")
    _data().write_csv(d / "panel.csv")
    (d / "cases.json").write_text(json.dumps([{"formula": f, "vcov": v, "weights": w} for f, v, w in CASES]))
    (d / "fit.R").write_text(R_SCRIPT)
    subprocess.run(
        ["Rscript", str(d / "fit.R"), str(d / "panel.csv"), str(d / "cases.json"), str(d / "out.json")],
        check=True,
        capture_output=True,
    )
    return json.loads((d / "out.json").read_text())


# ---------------------------------------------------------------- parsing


def test_parse_fe_terms():
    assert split_fe("unit[t]") == split_fe("unit[ t ]")
    t = split_fe("a ^ b[[x, z]]")
    assert (t.base, t.slopes, t.intercept, t.label) == ("a^b", ("x", "z"), False, "a^b[[x, z]]")
    f = parse("y ~ x | unit[t] + year")
    assert f.variables == {"y", "x", "t"}
    assert [t.base for t in f.fe_terms] == ["unit", "year"]
    for bad in ["y ~ x | unit[t] + unit", "y ~ x | unit[]", "y ~ x | unit[t, t]", "y ~ x | u[t]]"]:
        with pytest.raises(ValueError):
            parse(bad)


# ---------------------------------------------------------------- demeaning


def _dummies(c, L):
    D = np.zeros((len(c), L))
    D[np.arange(len(c)), c] = 1.0
    return D


@pytest.mark.parametrize("case", ["g[z]", "g[[z]]", "g[z, z2]", "g[z] + h", "g[[z]] + h", "g[z] + h[z2]"])
def test_demean_equals_dummy_regression(case):
    """Residuals on [D_g, D_g z, ...] by weighted least squares, including a group whose z is
    constant (its slope is not identified: the pseudo-inverse leaves the same residuals)."""
    rng = np.random.default_rng(0)
    n, G, H = 3000, 80, 15
    g, h = rng.integers(0, G, n), rng.integers(0, H, n)
    z = rng.normal(size=n) + g / 10
    z[g == 3] = 2.0
    z2 = rng.normal(size=n)
    w = rng.uniform(0.5, 2, n)
    X = rng.normal(size=(n, 4))
    Dg, Dh = _dummies(g, G), _dummies(h, H)
    spec = {
        "g[z]": ([g], [z[:, None]], [True], [Dg, Dg * z[:, None]]),
        "g[[z]]": ([g], [z[:, None]], [False], [Dg * z[:, None]]),
        "g[z, z2]": ([g], [np.c_[z, z2]], [True], [Dg, Dg * z[:, None], Dg * z2[:, None]]),
        "g[z] + h": ([g, h], [z[:, None], None], [True, True], [Dg, Dg * z[:, None], Dh]),
        "g[[z]] + h": ([g, h], [z[:, None], None], [False, True], [Dg * z[:, None], Dh]),
        "g[z] + h[z2]": (
            [g, h],
            [z[:, None], z2[:, None]],
            [True, True],
            [Dg, Dg * z[:, None], Dh, Dh * z2[:, None]],
        ),
    }[case]
    codes, S, ic, B = spec
    B = np.hstack(B)
    sw = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(B * sw, X * sw, rcond=None)
    got, ok = demean(X, codes, w, 1e-13, 100_000, slopes=S, intercept=ic)
    assert ok
    np.testing.assert_allclose(got, X - B @ coef, atol=1e-11)


# ---------------------------------------------------------------- fit against fixest


@needs_fixest
@pytest.mark.parametrize("i", range(len(CASES)), ids=[f"{f} [{v}{', w' if w else ''}]" for f, v, w in CASES])
def test_fit_equals_fixest(ds, fixest, tmp_path, i):
    formula, vcov, weights = CASES[i]
    r = _fit(ds, tmp_path, formula, vcov, weights)
    ref = fixest[i]
    assert r.N == ref["N"]
    # fixest: (Intercept), IV's endogenous regressors first as fit_<name>
    names = [{"(Intercept)": "Intercept"}.get(n, n.removeprefix("fit_")) for n in ref["names"]]
    assert sorted(r.coefnames) == sorted(names)
    at = [names.index(n) for n in r.coefnames]
    np.testing.assert_allclose(r.coef(), np.array(ref["coef"])[at], rtol=1e-9, atol=1e-12)
    se = np.sqrt(np.diag(r.vcov(type=_vtype(vcov))))
    np.testing.assert_allclose(se, np.array(ref["se"])[at], rtol=1e-8)
    y = next(iter(r.outcomes))
    assert r.outcomes[y]["vcov"][_vtype(vcov)][1] == ref["df_t"]
    if formula.count("|") == 1:  # OLS: within R2
        np.testing.assert_allclose(r.outcomes[y]["r2_within"], ref["wr2"], rtol=1e-8)


def test_partition_invariance(ds, ds_week, tmp_path):
    """Local slopes (id_a[week], partition id_a) equal crossing slopes as dense dummies
    (partition week)."""
    f = "eng ~ feat + posts | id_a[week] + week"
    a = _fit(ds, tmp_path / "a", f, "week")
    b = _fit(ds_week, tmp_path / "b", f, "week")
    assert a.run["absorption"]["id_a[week]"]["method"] == "local"
    assert b.run["absorption"]["id_a[week]"]["method"] == "crossing_dense"
    assert a.N == b.N and dict(zip(a.fe, a.fe_levels)) == dict(zip(b.fe, b.fe_levels))
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-9)
    np.testing.assert_allclose(a.vcov(type="CRV1"), b.vcov(type="CRV1"), rtol=1e-8)


def test_cached_round2(ds, tmp_path):
    """The demeaned cache (round 2 reads round 1's arrays) gives the same fit."""
    f = "eng ~ feat + posts | id_a[week] + spell"
    a = _fit(ds, tmp_path / "a", f, "week", cache=True)
    b = _fit(ds, tmp_path / "b", f, "week", cache=False)
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-12)
    np.testing.assert_allclose(a.vcov(type="CRV1"), b.vcov(type="CRV1"), rtol=1e-10)


def test_record_validates(ds, tmp_path):
    r = _fit(ds, tmp_path, "eng ~ feat + posts | spell[tenure] + week[[feat]]", "id_a", record=True)
    rec = estimation.read(tmp_path / "result.estrec.json", nonfinite="string")
    bad = [i for i in estimation.validate(rec, base_dir=tmp_path) if i.level != "info"]
    assert not bad, "\n".join(map(str, bad))
    fes = {fe["name"]: fe for fe in rec["structures"]["fixed_effects"]}
    assert fes["spell[tenure]"]["components"] == ["spell", "tenure"]
    assert fes["week[[feat]]"]["components"] == ["week", "feat"]
    assert r.fe == ["spell[tenure]", "week[[feat]]"]


# ---------------------------------------------------------------- what is not there yet


def test_refusals(ds, ds_week, tmp_path):
    f = "eng ~ feat + posts | spell + week[feat]"
    for how in ("sparse", "iterate"):
        with pytest.raises(NotImplementedError, match="varying slopes"):
            _fit(ds, tmp_path, f, crossing_fe=how)
    with pytest.raises(NotImplementedError, match="varying slopes"):
        _fit(ds, tmp_path, "eng ~ feat + posts | spell[tenure]", cells=True)
    with pytest.raises(NotImplementedError, match="max_dummy_levels"):
        _fit(ds_week, tmp_path, "eng ~ feat + posts | week + spell[tenure]", max_dummy_levels=100)


def test_sessions_refuse_slopes(tmp_path):
    from dfeols.session import Spec

    with pytest.raises(NotImplementedError, match="sessions do not take varying slopes"):
        Spec.make(tmp_path, "unit[t] + year", "y, x")
