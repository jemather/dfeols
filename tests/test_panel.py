"""Validity tests on the generic unit x time panel (dfeols.synth.PanelDGP).

Two synthetic panels:
  classic     group-time shocks are noise: unit + time FEs recover the truth,
              SEs must be clustered by group
  confounded  group-time shocks also drive x: only unit + group^time recovers
              the truth; unit + time is biased

Every spec in the DGP's truth.json is checked against what it should find
(unbiased -> recovered; biased -> rejected), and the same model fitted under
two partition keys (unit vs group) must give identical results.
"""

import numpy as np
import polars as pl
import pytest

import dfeols
from dfeols.formula import to_pyfixest
from dfeols.synth import PanelDGP, generate, load_truth
from dfeols.validate import recovery


def _make(base, name, rows, **kw):
    d = PanelDGP(**kw)
    per_unit = (d.T + d.min_periods) / 2 if not d.balanced else d.T
    d = d.scaled(max(int(rows / per_unit), 200))
    d = PanelDGP(**{**d.__dict__, "units_per_chunk": max(d.n_units // 6, 50)})
    raw = base / f"panel_{name}_{int(rows)}"
    generate(raw, d, verbose=False)
    truth = load_truth(raw)
    shards = {}
    n = pl.scan_parquet(str(raw / "*.parquet")).select(pl.len()).collect().item()
    for part in ("unit", "group"):
        sd = base / f"panel_{name}_{int(rows)}_by_{part}"
        if (sd / "dfeols_meta.json").exists():
            shards[part] = dfeols.Dataset.open(sd)
        else:
            shards[part] = dfeols.prep(str(raw / "*.parquet"), shard_dir=sd, partition=part,
                                       target_rows=max(n // 10, 1000), overwrite=True,
                                       **truth["prep"])
    return {"raw": raw, "truth": truth, "ds": shards, "dgp": d}


@pytest.fixture(scope="session")
def panels(request, tmp_path_factory):
    rows = request.config.getoption("--rows")
    base = request.config.getoption("--data-dir")
    from pathlib import Path
    base = Path(base) if base else tmp_path_factory.mktemp("panel")
    # few groups so group clustering has bite; enough for a valid small-G test
    common = dict(n_groups=150, T=40)
    return {"classic": _make(base, "classic", rows, gt_confound=0.0, **common),
            "confounded": _make(base, "confounded", rows, gt_confound=1.0, **common)}


def _fit(p, spec, executor, **kw):
    s = p["truth"]["specs"][spec]
    return dfeols.fit(p["ds"][s["partition"]], s["formula"], cluster=kw.pop("cluster", s["cluster"]),
                      executor=executor, log_every=1e9, **kw)


CASES = [(panel, spec) for panel in ("classic", "confounded")
         for spec in ("twfe", "twfe_by_grp", "grouptime", "no_unit_fe")]


@pytest.mark.parametrize("panel,spec", CASES, ids=[f"{a}-{b}" for a, b in CASES])
def test_spec_finds_what_it_should(panels, executor, panel, spec):
    p = panels[panel]
    expect = p["truth"]["specs"][spec]["expect"]
    res = _fit(p, spec, executor)
    r = recovery(res, p["truth"])
    print(f"\n{panel}/{spec} ({expect}): N={res.N:,} G={res.G} max|z|={r['max_abs_z']:.2f} "
          f"(crit {r['zcrit']:.2f}) p={r['p']:.3g}")
    if expect == "unbiased":
        assert r["ok"] and r["p"] > 0.001, r
    else:
        assert r["p"] < 1e-4, r


def test_partition_invariance(panels, executor):
    """unit + time partitioned by unit (time as dummies) and by group
    (unit FE local, time as dummies) are the same model."""
    p = panels["classic"]
    a = _fit(p, "twfe", executor, vcov=["CRV1", "hetero"])
    b = _fit(p, "twfe_by_grp", executor, vcov=["CRV1", "hetero"])
    assert a.N == b.N
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-9, atol=1e-12)
    for t in ("CRV1", "hetero"):
        np.testing.assert_allclose(a.vcov(type=t), b.vcov(type=t), rtol=1e-8, atol=1e-15)


def test_group_clustering_matters(panels, executor):
    p = panels["classic"]
    g = _fit(p, "twfe", executor)                      # cluster by group
    u = _fit(p, "twfe", executor, cluster="unit")
    ratio = np.median(np.sqrt(np.diag(g.vcov())) / np.sqrt(np.diag(u.vcov())))
    print(f"\nmedian SE ratio group/unit clustering = {ratio:.2f}")
    assert ratio > 1.15


def test_weights(panels, executor):
    p = panels["classic"]
    res = _fit(p, "grouptime", executor, weights="w")
    r = recovery(res, p["truth"])
    assert r["ok"] and r["p"] > 0.001, r


def test_matches_pyfixest(panels, executor, tmp_path, request):
    pf = pytest.importorskip("pyfixest")
    p = panels["confounded"]
    cap = request.config.getoption("--pyfixest-rows")
    raw = pl.scan_parquet(str(p["raw"] / "*.parquet"))
    sizes = raw.group_by("group").len().sort("group").collect()
    keep = sizes.filter(pl.col("len").cum_sum() <= max(cap, sizes["len"][0]))["group"]
    sub = raw.filter(pl.col("group").is_in(keep.implode())).collect()
    for spec in ("twfe", "grouptime"):
        s = p["truth"]["specs"][spec]
        ds = dfeols.prep(sub, shard_dir=tmp_path / spec, partition=s["partition"],
                         target_rows=max(sub.height // 5, 1000), **p["truth"]["prep"])
        ours = dfeols.fit(ds, s["formula"], cluster="group", vcov=["CRV1", "hetero"],
                          executor=executor, log_every=1e9)
        for v in ("CRV1", "hetero"):
            m = pf.feols(to_pyfixest(s["formula"]), data=sub.to_pandas(),
                         vcov={"CRV1": "group"} if v == "CRV1" else v,
                         fixef_tol=1e-12, fixef_maxiter=1_000_000)
            t = m.tidy().loc[ours.coefnames]
            o = ours.tidy(type=v)
            np.testing.assert_allclose(o["Estimate"], t["Estimate"], rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(o["Std. Error"], t["Std. Error"], rtol=1e-6)
            assert ours.N == m._N
