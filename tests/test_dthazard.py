"""lpm_dthazard: the Farbmacher-Tauchmann discrete-time hazard IV (docs/design/hazard.md).

The reference sample is built here independently (pandas, per-unit shifts) and fitted by
pyfixest's IV; dfeols must match it, count every dropped row by reason, and keep the
algebra of the paper (b_FD = G b_IV exactly, just identified)."""

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import dfeols
from dfeols.dthazard import DROP_REASONS, lpm_dthazard
from dfeols.formula import pyfixest_tol_kwargs

pf = pytest.importorskip("pyfixest")
warnings.filterwarnings("ignore")

TOL = dict(rtol=1e-8, atol=1e-10)


def hazard_panel(n=3000, T=6, seed=1, movers=0.0, gaps=0.0, missing=0.0):
    """The paper's section 4 DGP (stationary x1, a_i ~ U(-.05, .05), alpha = .1, b = 1) plus a
    random-walk x2 (b = .5), a group g (fixed per unit unless movers > 0), time effects; rows
    after the event are not generated. gaps drops random non-first rows, missing blanks x1."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        a = rng.uniform(-0.05, 0.05)
        g = int(rng.integers(0, 4))
        x2 = a + 0.1
        for t in range(1, T + 1):
            if t > 1 and rng.random() < movers:
                g = int(rng.integers(0, 4))
            x1 = a + 0.165 + 0.07 * rng.beta(0.2, 0.2)
            x2 = x2 + 0.05 * (rng.beta(0.2, 0.2) - 0.5)
            p = 0.1 + a + x1 - 0.165 + 0.5 * (x2 - 0.1) + 0.01 * (t % 2) + 0.005 * g
            y = int(rng.random() < p)
            rows.append((i, t, g, x1, x2, y))
            if y:
                break
    df = pl.DataFrame(rows, schema=["uid", "t", "g", "x1", "x2", "y"], orient="row")
    if gaps:
        df = df.filter((pl.col("t") == 1) | (pl.Series(rng.random(df.height)) >= gaps))
    if missing:
        df = df.with_columns(
            pl.when(pl.Series(rng.random(df.height)) < missing).then(None).otherwise(pl.col("x1")).alias("x1")
        )
    return df.with_columns(gt=pl.col("g") * 100 + pl.col("t"))


def reference_sample(df: pl.DataFrame, j: int, terms=("x1", "x2")):
    """The estimation sample built the slow, obvious way: per unit, a row is kept if the
    rows t-1..t-j exist with complete regressors and the row itself is complete."""
    pdf = df.sort(["uid", "t"]).to_pandas()
    w = [(-1) ** k * math.comb(j, k) for k in range(j + 1)]
    out, why = [], dict.fromkeys(DROP_REASONS, 0)
    for _, u in pdf.groupby("uid", sort=False):
        u = u.reset_index(drop=True)
        tt = u["t"].tolist()
        for r in range(len(u)):
            own = all(np.isfinite(u.loc[r, c]) for c in terms) and not np.isnan(u.loc[r, "y"])
            if not own:
                why["missing"] += 1
            elif r < j:
                why["first_periods"] += 1
            elif tt[r] - tt[r - j] != j:
                why["gap"] += 1
            elif not all(np.isfinite(u.loc[r - k, c]) for k in range(1, j + 1) for c in terms):
                why["missing_lag"] += 1
            else:
                row = u.loc[r].to_dict()
                for c in terms:
                    row[f"D{j}_{c}"] = sum(w[k] * u.loc[r - k, c] for k in range(j + 1))
                out.append(row)
    import pandas as pd

    return pd.DataFrame(out), why


@pytest.fixture(scope="module")
def data():
    return hazard_panel()


@pytest.fixture(scope="module")
def ds(data, tmp_path_factory):
    return dfeols.prep(
        data,
        shard_dir=tmp_path_factory.mktemp("hz"),
        partition="uid",
        fixed_effects=["t", "g^t", "uid"],
        clusters=["uid", "g"],
        target_rows=2000,
    )


def _check(r, m, names):
    t = m.tidy()
    ours = r.tidy()
    np.testing.assert_allclose(ours["Estimate"], t.loc[names, "Estimate"], **TOL)
    np.testing.assert_allclose(ours["Std. Error"], t.loc[names, "Std. Error"], rtol=1e-8)
    assert r.N == m._N


@pytest.mark.parametrize("fe", ["", "t", "g^t"])
@pytest.mark.parametrize("j", [1, 2])
@pytest.mark.parametrize("vcov,cluster", [("hetero", None), ("CRV1", "uid")])
def test_matches_pyfixest_on_hand_built_sample(ds, data, tmp_path, fe, j, vcov, cluster):
    fe_part = f" | {fe}" if fe else ""
    h = lpm_dthazard(
        ds,
        f"y ~ x1 + x2{fe_part}",
        unit="uid",
        time="t",
        diff=j,
        vcov=vcov,
        cluster=cluster,
        workdir=tmp_path,
        log_every=1e9,
    )
    ref, why = reference_sample(data, j)
    assert {k: v for k, v in h.sample["dropped"].items()} == {
        k: v for k, v in why.items() if k != "after_event"
    }
    pv = {"CRV1": "uid"} if vcov == "CRV1" else vcov
    pfe = f" | {fe}" if fe else ""
    tol = pyfixest_tol_kwargs(1e-13)
    # two endogenous: dfeols.fit (checked against fixest in test_iv) on the hand-built sample
    rds = dfeols.prep(
        pl.from_pandas(ref),
        shard_dir=tmp_path / "ref",
        partition="uid",
        fixed_effects=["t", "g^t"],
        clusters=["uid"],
        target_rows=100_000,
    )
    r = dfeols.fit(
        rds,
        f"y ~ 1{pfe} | x1 + x2 ~ D{j}_x1 + D{j}_x2",
        vcov=vcov,
        cluster=cluster,
        log_every=1e9,
        workdir=tmp_path / "rfit",
        record=False,
    )
    np.testing.assert_allclose(h.iv.coef(), r.coef(), **TOL)
    np.testing.assert_allclose(h.iv.vcov(), r.vcov(), rtol=1e-8, atol=1e-14)
    assert h.iv.N == r.N
    m = pf.feols(f"y ~ D{j}_x1 + D{j}_x2{pfe}", data=ref, vcov=pv, **tol)
    _check(h.fd, m, h.fd.coefnames)
    m = pf.feols(f"y ~ x1 + x2{pfe}", data=ref, vcov=pv, **tol)
    _check(h.ols, m, h.ols.coefnames)
    # one endogenous: pyfixest directly
    h1 = lpm_dthazard(
        ds,
        f"y ~ x1{fe_part}",
        unit="uid",
        time="t",
        diff=j,
        vcov=vcov,
        cluster=cluster,
        workdir=tmp_path / "one",
        log_every=1e9,
    )
    ref1, _ = reference_sample(data, j, terms=("x1",))
    m = pf.feols(f"y ~ 1{pfe} | x1 ~ D{j}_x1", data=ref1, vcov=pv, **tol)
    _check(h1.iv, m, h1.iv.coefnames)


def test_reduced_form_is_G_times_b(ds, tmp_path):
    """Just identified: the first-difference (reduced form) coefficients equal G b_IV."""
    h = lpm_dthazard(ds, "y ~ x1 + x2 | t", unit="uid", time="t", workdir=tmp_path, log_every=1e9)
    b = dict(zip(h.iv.coefnames, h.iv.coef()))
    fd = dict(zip(h.fd.coefnames, h.fd.coef()))
    for inst in ("D1_x1", "D1_x2"):
        assert fd[inst] == pytest.approx(sum(h.G[e][inst] * b[e] for e in ("x1", "x2")), rel=1e-9, abs=1e-12)


def test_gaps_and_missing_are_dropped_and_counted(tmp_path):
    df = hazard_panel(n=1500, seed=3, gaps=0.15, missing=0.05)
    ds = dfeols.prep(df, shard_dir=tmp_path / "s", partition="uid", fixed_effects=["t"], target_rows=1500)
    h = lpm_dthazard(ds, "y ~ x1 + x2 | t", unit="uid", time="t", workdir=tmp_path / "w", log_every=1e9)
    ref, why = reference_sample(df, 1)
    assert why["gap"] > 0 and why["missing_lag"] > 0 and why["missing"] > 0
    assert h.sample["dropped"] == {k: v for k, v in why.items() if k != "after_event"}
    assert h.sample["rows_built"] == len(ref) == df.height - sum(why.values())
    s = h.summary()
    h1 = lpm_dthazard(ds, "y ~ x1 | t", unit="uid", time="t", workdir=tmp_path / "w1", log_every=1e9)
    ref1, _ = reference_sample(df, 1, terms=("x1",))
    m = pf.feols("y ~ 1 | t | x1 ~ D1_x1", data=ref1, vcov="hetero", **pyfixest_tol_kwargs(1e-13))
    _check(h1.iv, m, h1.iv.coefnames)
    for reason in ("gap", "missing_lag", "first_periods"):
        assert reason in s


def test_movers_warn_fixed_groups_do_not(tmp_path, ds):
    h = lpm_dthazard(ds, "y ~ x1 | g^t", unit="uid", time="t", workdir=tmp_path / "a", log_every=1e9)
    assert h.fe_checks["g^t"]["ambiguous_rows"] == 0 and not h.warnings
    df = hazard_panel(n=1500, seed=4, movers=0.3)
    ds2 = dfeols.prep(df, shard_dir=tmp_path / "s", partition="uid", fixed_effects=["g^t"], target_rows=1500)
    h = lpm_dthazard(ds2, "y ~ x1 | g^t", unit="uid", time="t", workdir=tmp_path / "b", log_every=1e9)
    c = h.fe_checks["g^t"]
    assert c["ambiguous_rows"] > 0 and 0 < c["ambiguous_share"] <= 1
    assert any("move between groups" in w for w in h.warnings)


def test_after_event_rows(tmp_path):
    df = hazard_panel(n=500, seed=5)
    extra = df.filter(pl.col("y") == 1).head(7).with_columns(pl.col("t") + 1, y=pl.lit(0, dtype=pl.Int64))
    df2 = pl.concat([df, extra])
    ds = dfeols.prep(df2, shard_dir=tmp_path / "s", partition="uid", fixed_effects=["t"], target_rows=1000)
    with pytest.raises(ValueError, match="7 rows come after"):
        lpm_dthazard(ds, "y ~ x1 | t", unit="uid", time="t", workdir=tmp_path / "w", log_every=1e9)
    h = lpm_dthazard(
        ds, "y ~ x1 | t", unit="uid", time="t", after_event="drop", workdir=tmp_path / "w2", log_every=1e9
    )
    assert h.sample["dropped"]["after_event"] == 7


def test_refusals(ds, data, tmp_path):
    with pytest.raises(ValueError, match="within estimator"):
        lpm_dthazard(ds, "y ~ x1 | uid", unit="uid", time="t", workdir=tmp_path)
    with pytest.raises(ValueError, match="IV part"):
        lpm_dthazard(ds, "y ~ 1 | x1 ~ x2", unit="uid", time="t", workdir=tmp_path)
    with pytest.raises(NotImplementedError, match="absorb"):
        lpm_dthazard(ds, "y ~ x1 + i(g)", unit="uid", time="t", workdir=tmp_path)
    with pytest.raises(ValueError, match="constant"):
        lpm_dthazard(ds, "y ~ 0 + x1", unit="uid", time="t", workdir=tmp_path)
    with pytest.raises(ValueError, match="nothing to instrument"):
        lpm_dthazard(ds, "y ~ x1", unit="uid", time="t", levels=["x1"], workdir=tmp_path)
    by_g = dfeols.prep(data, shard_dir=tmp_path / "g", partition="g", fixed_effects=["t"], target_rows=5000)
    with pytest.raises(ValueError, match="partition='uid'"):
        lpm_dthazard(by_g, "y ~ x1", unit="uid", time="t", workdir=tmp_path)
    bad = data.with_columns(y=pl.col("y") * 2)
    dsb = dfeols.prep(bad, shard_dir=tmp_path / "b", partition="uid", fixed_effects=["t"], target_rows=5000)
    with pytest.raises(ValueError, match="other than 0 and 1"):
        lpm_dthazard(dsb, "y ~ x1", unit="uid", time="t", workdir=tmp_path)


def test_levels_instrument_themselves(ds, data, tmp_path):
    h = lpm_dthazard(
        ds, "y ~ x1 + x2 | t", unit="uid", time="t", levels=["x2"], workdir=tmp_path, log_every=1e9
    )
    ref, _ = reference_sample(data, 1)
    m = pf.feols("y ~ x2 | t | x1 ~ D1_x1", data=ref, vcov="hetero", **pyfixest_tol_kwargs(1e-13))
    _check(h.iv, m, h.iv.coefnames)
    assert list(h.G) == ["x1"]


def test_sample_reused(ds, tmp_path, caplog):
    lpm_dthazard(ds, "y ~ x1 + x2", unit="uid", time="t", workdir=tmp_path / "a", log_every=1e9)
    with caplog.at_level("INFO", logger="dfeols"):
        lpm_dthazard(ds, "y ~ x1 + x2", unit="uid", time="t", workdir=tmp_path / "b", log_every=1e9)
    assert "sample reused" in caplog.text


def test_record_validates_and_carries_the_sample_rules(ds, tmp_path):
    est = pytest.importorskip("estrec")
    pytest.importorskip("jsonschema")
    lpm_dthazard(ds, "y ~ x1 + x2 | t", unit="uid", time="t", cluster="uid", workdir=tmp_path, log_every=1e9)
    path = tmp_path / "result.estrec.json"
    rec = est.read(path, nonfinite="string")
    bad = [i for i in est.validate(rec, base_dir=tmp_path) if i.level != "info"]
    assert not bad, "\n".join(map(str, bad))
    assert "iv/0.1" in rec["meta"]["profiles"]
    sp = rec["specification"]
    assert sp["call"].startswith("dfeols.lpm_dthazard(ds, 'y ~ x1 + x2 | t', unit='uid', time='t', diff=1")
    assert "hazard" in sp["sample"]["restriction"]
    assert sp["sample"]["dropped"]["first_periods"] > 0
    assert sp["instruments"] == ["D1_x1", "D1_x2"]
    hz = rec["extensions"]["dfeols"]["dthazard"]
    assert hz["diff"] == 1 and set(hz["G"]) == {"x1", "x2"}
    assert set(hz["comparison"]) == {"first_differences", "pooled_ols"}
    assert "first_periods" in hz["drop_reasons"]


def test_monte_carlo_sanity(tmp_path):
    """Stationary x (paper Table 1, first row): b_IV near 1, b_FD near 1/2, b_OLS above 1."""
    rng = np.random.default_rng(7)
    n, T = 40_000, 5
    a = rng.uniform(-0.05, 0.05, n)
    x = a[:, None] + 0.165 + 0.07 * rng.beta(0.2, 0.2, (n, T))
    u = rng.random((n, T))
    ev = u < (0.1 + a[:, None] + x)
    first = np.where(ev.any(1), ev.argmax(1), T - 1)
    keep = np.arange(T)[None, :] <= first[:, None]
    ii, tt = np.nonzero(keep)
    df = pl.DataFrame({"uid": ii, "t": tt + 1, "x": x[ii, tt], "y": ev[ii, tt].astype(np.int64)})
    ds = dfeols.prep(df, shard_dir=tmp_path / "s", partition="uid", target_rows=50_000)
    h = lpm_dthazard(ds, "y ~ x", unit="uid", time="t", workdir=tmp_path / "w", log_every=1e9)
    b, se = h.iv.coef()[-1], np.sqrt(h.iv.vcov()[-1, -1])
    assert abs(b - 1) < 4 * se
    assert abs(h.fd.coef()[-1] - 0.5) < 4 * np.sqrt(h.fd.vcov()[-1, -1])
    assert h.ols.coef()[-1] > 1.2
