"""Validity tests on a large synthetic panel with known parameters.

What each test establishes:

  recovery        estimates are centred on the truth: every coefficient is
                  within a Bonferroni z bound and the joint Wald test of
                  b = beta does not reject (account-clustered SEs).
  negative ctrl   dropping a confounded FE makes the same tests reject,
                  so the recovery tests have power.
  clustering      the DGP has account-correlated errors, so CRV1 SEs exceed
                  iid SEs; iid inference over-rejects.
  invariance      results do not depend on how the data are sharded or which
                  executor runs them (to ~1e-10).
  pyfixest        results equal pyfixest on the same data (to ~1e-8).
  coverage (slow) Monte Carlo: 95% CRV1 intervals cover the truth ~95% of
                  the time; iid intervals undercover.
"""

import math
import shutil

import numpy as np
import pytest
from scipy import stats

import dfeols
from dfeols.formula import pyfixest_tol_kwargs

LEVEL = "eng ~ feat:i(tbin) + posts:i(tbin) | spell + week"
SHAPE = "eng ~ feat:i(tbin) + posts:i(tbin) | spell + id_a^week"


def _truth_vec(res, truth, scale=1.0):
    """True values for the reported coefficients.

    With account x week FE, feat and posts (account-week variables) are
    absorbed, so one tenure bin of each is dropped as collinear and the others
    are identified relative to it (the 'shape'). The truth is then
    beta_b - beta_dropped for that variable.
    """
    tr = truth["coefs"]
    ref = {}
    for c in res.collinear:
        if c in tr:
            ref[c.split(":")[-1]] = tr[c]
    out = []
    for c in res.coefnames:
        var = c.split(":")[-1]
        out.append((tr.get(c, np.nan) - ref.get(var, 0.0)) * scale)
    return np.array(out)


def _sel(res):
    """Indices of coefficients that have a known truth (skip Intercept etc.)."""
    return [i for i, c in enumerate(res.coefnames) if c.startswith("tbin::")]


def _check_recovery(res, beta, y=None, vtype="CRV1", alpha=0.001):
    """Bonferroni z bound per coefficient and a joint Wald test.

    With clustered SEs the Wald statistic uses the Hotelling-type small-G
    correction F = W (G-k) / (k (G-1)) ~ F(k, G-k), since the cluster
    covariance is estimated from only G clusters.
    """
    idx = _sel(res)
    b = res.coef(y)[idx]
    V = res.vcov(y, vtype)[np.ix_(idx, idx)]
    beta = beta[idx]
    se = np.sqrt(np.diag(V))
    z = (b - beta) / se
    k = len(b)
    zcrit = stats.norm.isf(alpha / (2 * k))  # Bonferroni
    d = b - beta
    wald = float(d @ np.linalg.solve(V, d))
    G = res.G
    if vtype == "CRV1" and G and G > k + 1:
        p = stats.f.sf(wald * (G - k) / (k * (G - 1)), k, G - k)
    else:
        p = stats.chi2.sf(wald, k)
    return {
        "max_abs_z": float(np.abs(z).max()),
        "zcrit": zcrit,
        "wald": wald,
        "p": p,
        "k": k,
        "worst": res.coefnames[idx[int(np.abs(z).argmax())]],
    }


def _fit(synth, fml, executor, **kw):
    kw.setdefault("cluster", "id_a")
    return dfeols.fit(synth["ds"], fml, executor=executor, log_every=1e9, **kw)


# ---------------------------------------------------------------- recovery
@pytest.mark.parametrize("fml", [LEVEL, SHAPE], ids=["spell+week", "spell+acct^week"])
def test_recovery(synth, executor, fml):
    res = _fit(synth, fml, executor)
    r = _check_recovery(res, _truth_vec(res, synth["truth"]))
    print(
        f"\n{fml}\n  N={res.N:,} G={res.G:,}  max|z|={r['max_abs_z']:.2f} "
        f"(crit {r['zcrit']:.2f}, {r['worst']})  Wald={r['wald']:.1f} on {r['k']} df, p={r['p']:.3f}"
    )
    assert r["max_abs_z"] < r["zcrit"], r
    assert r["p"] > 0.001, r


def test_recovery_weighted_two_outcomes(synth, executor):
    res = _fit(synth, LEVEL.replace("eng ~", "eng + eng2 ~"), executor, weights="wt")
    for y, scale in (("eng", 1.0), ("eng2", 0.5)):
        r = _check_recovery(res, _truth_vec(res, synth["truth"], scale), y=y)
        print(f"\n{y} (weighted): max|z|={r['max_abs_z']:.2f}  p={r['p']:.3f}")
        assert r["max_abs_z"] < r["zcrit"] and r["p"] > 0.001, (y, r)


def test_fe_counts_match_data(synth, executor):
    import polars as pl

    res = _fit(synth, "eng ~ feat + posts | spell + week", executor)
    raw = pl.scan_parquet(str(synth["raw"] / "*.parquet"))
    n_spell_multi = raw.group_by("spell").len().filter(pl.col("len") > 1).select(pl.len()).collect().item()
    n_rows_multi = (
        raw.group_by("spell").len().filter(pl.col("len") > 1).select(pl.col("len").sum()).collect().item()
    )
    assert res.fe_levels[0] == n_spell_multi  # spells after dropping singletons
    assert res.N == n_rows_multi
    assert res.G == raw.select(pl.col("id_a").n_unique()).collect().item()


# -------------------------------------------------------- negative controls
@pytest.mark.parametrize(
    "fml",
    [
        "eng ~ feat:i(tbin) + posts:i(tbin) | spell",  # drops week FE (trend confounding)
        "eng ~ feat:i(tbin) + posts:i(tbin) | week",  # drops spell FE (propensity confounding)
    ],
    ids=["no-week-FE", "no-spell-FE"],
)
def test_negative_control_rejects(synth, executor, fml):
    res = _fit(synth, fml, executor)
    r = _check_recovery(res, _truth_vec(res, synth["truth"]))
    print(f"\n{fml}: Wald={r['wald']:.0f} p={r['p']:.2e}")
    assert r["p"] < 1e-4, "omitting a confounded FE should be detected"


def test_clustering_matters(synth, executor):
    res = _fit(synth, LEVEL, executor, vcov=["CRV1", "iid", "hetero"])
    se_c = np.sqrt(np.diag(res.vcov(type="CRV1")))
    se_i = np.sqrt(np.diag(res.vcov(type="iid")))
    ratio = np.median(se_c / se_i)
    feat = [i for i, c in enumerate(res.coefnames) if c.endswith(":feat")]
    ratio_feat = np.median(se_c[feat] / se_i[feat])
    print(f"\nmedian SE ratio CRV1/iid = {ratio:.2f} (feature coefs {ratio_feat:.2f})")
    # feat varies at the account-week level and the DGP has account-week shocks
    assert ratio_feat > 1.1


# --------------------------------------------------------------- invariance
def test_shard_invariance(synth, executor, tmp_path):
    ds2 = dfeols.prep(
        str(synth["raw"] / "*.parquet"),
        shard_dir=tmp_path / "s2",
        partition="id_a",
        fixed_effects=["spell", "week"],
        clusters=["id_a", "id_f"],
        target_rows=max(synth["ds"].n_rows // 29, 500),
    )
    assert len(ds2.shards) != len(synth["ds"].shards)
    a = _fit(synth, LEVEL, executor, vcov=["CRV1", "hetero"])
    b = dfeols.fit(ds2, LEVEL, cluster="id_a", vcov=["CRV1", "hetero"], executor=executor, log_every=1e9)
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-10, atol=1e-12)
    for t in ("CRV1", "hetero"):
        np.testing.assert_allclose(a.vcov(type=t), b.vcov(type=t), rtol=1e-9, atol=1e-15)
    # follower clusters cross shards differently in the two layouts
    a = _fit(synth, "eng ~ feat:i(tbin) | spell + week", executor, cluster="id_f")
    b = dfeols.fit(ds2, "eng ~ feat:i(tbin) | spell + week", cluster="id_f", executor=executor, log_every=1e9)
    np.testing.assert_allclose(a.vcov(), b.vcov(), rtol=1e-9, atol=1e-15)


def test_executor_invariance(synth):
    distributed = pytest.importorskip("distributed")
    a = _fit(synth, LEVEL, dfeols.SerialExecutor())
    with (
        distributed.LocalCluster(n_workers=2, threads_per_worker=1, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        b = _fit(synth, LEVEL, dfeols.DaskExecutor(c, max_workers=2, tick=1))
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a.vcov(), b.vcov(), rtol=1e-9, atol=1e-15)


# ----------------------------------------------------------------- pyfixest
def test_matches_pyfixest(synth, executor, request, tmp_path):
    """Exact agreement with pyfixest on a subset of whole accounts (pyfixest
    needs the data in memory, so the subset is capped by --pyfixest-rows)."""
    pf = pytest.importorskip("pyfixest")
    import gc

    import polars as pl

    cap = int(request.config.getoption("--pyfixest-rows"))
    raw = pl.scan_parquet(str(synth["raw"] / "*.parquet"))
    sizes = raw.group_by("id_a").len().sort("id_a").collect()
    keep = sizes.filter(pl.col("len").cum_sum() <= cap)["id_a"]
    cols = ["eng", "tbin", "feat", "posts", "spell", "week", "id_a", "id_f"]
    sub = raw.filter(pl.col("id_a").is_in(keep.implode())).select(cols).collect()
    ds = dfeols.prep(
        sub,
        shard_dir=tmp_path / "pf",
        partition="id_a",
        fixed_effects=["spell", "week"],
        clusters=["id_a", "id_f"],
        target_rows=max(sub.height // 5, 1000),
    )
    pdf = sub.to_pandas()
    del sub
    gc.collect()
    for cl in ("id_a", "id_f"):
        ours = dfeols.fit(ds, LEVEL, cluster=cl, vcov=["CRV1", "hetero"], executor=executor, log_every=1e9)
        for v in ("CRV1", "hetero"):
            m = pf.feols(
                "eng ~ i(tbin, feat) + i(tbin, posts) | spell + week",
                data=pdf,
                vcov={"CRV1": cl} if v == "CRV1" else v,
                **pyfixest_tol_kwargs(1e-12),
            )
            t = m.tidy().loc[ours.coefnames]
            o = ours.tidy(type=v)
            np.testing.assert_allclose(o["Estimate"], t["Estimate"], rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(o["Std. Error"], t["Std. Error"], rtol=1e-6)
            assert ours.N == m._N
            del m
            gc.collect()


# ------------------------------------------------------- Monte Carlo (slow)
@pytest.mark.slow
def test_coverage_monte_carlo(request, tmp_path):
    """Repeated small panels: CI coverage of CRV1 should be ~95%."""

    from dfeols.synth import DGP, generate_chunk

    R = request.config.getoption("--mc-reps")
    pick = ["tbin::2:feat", "tbin::12:feat", "tbin::24:feat", "tbin::5:posts"]
    cover = {"CRV1": [], "iid": []}
    for rep in range(R):
        d = DGP(
            n_accounts=300, accounts_per_chunk=300, spells_logmean=2.5, weeks=60, n_bins=24, seed=1000 + rep
        )
        df = generate_chunk(d, 0)
        root = tmp_path / f"r{rep}"
        ds = dfeols.prep(
            df,
            shard_dir=root,
            partition="id_a",
            fixed_effects=["spell", "week"],
            clusters=["id_a"],
            target_rows=max(df.height // 4, 1000),
        )
        res = dfeols.fit(ds, LEVEL, cluster="id_a", vcov=["CRV1", "iid"], log_every=1e9, cache=False)
        tr = d.truth()["coefs"]
        idx = [res.coefnames.index(c) for c in pick if c in res.coefnames]
        for v in cover:
            tab = res.tidy(type=v)
            lo, hi = tab["2.5%"].to_numpy()[idx], tab["97.5%"].to_numpy()[idx]
            beta = np.array([tr[res.coefnames[i]] for i in idx])
            cover[v].extend(((lo <= beta) & (beta <= hi)).tolist())
        shutil.rmtree(root, ignore_errors=True)
    c, i = np.mean(cover["CRV1"]), np.mean(cover["iid"])
    n = len(cover["CRV1"])
    se = math.sqrt(0.95 * 0.05 / R)  # conservative: per-coefficient draws
    print(f"\ncoverage over {R} reps x {len(pick)} coefs: CRV1 {c:.3f}, iid {i:.3f} (n={n})")
    assert abs(c - 0.95) < 3 * se + 0.01, c
    assert i < c - 0.05, "iid intervals should undercover under clustered errors"
