"""Validate dfeols against pyfixest on simulated panels.

Run:  pytest -q tests/        (add -k dask to run only the Dask tests)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
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
    return sim_panel()


@pytest.fixture(scope="module")
def ds(data, tmp_path_factory):
    root = tmp_path_factory.mktemp("shards")
    return dfeols.prep(
        data,
        shard_dir=root,
        partition="id_a",
        fixed_effects=["spell", "week", "id_a^week", "id_a"],
        clusters=["id_a", "id_f", "week"],
        target_rows=3000,
    )


def _pf_vcov(v, cl):
    return {"CRV1": cl} if v == "CRV1" else v


def _compare(
    ds, data, ours_fml, pf_fml, vcov, cluster=None, weights=None, rename=None, executor=None, y=None
):
    r = dfeols.fit(
        ds, ours_fml, cluster=cluster, vcov=vcov, weights=weights, executor=executor, log_every=1e9
    )
    pdf = data.to_pandas()
    # dfeols uses one common sample across outcomes (listwise deletion)
    pdf = pdf.dropna(subset=list(r.outcomes))
    for v in [vcov] if isinstance(vcov, str) else vcov:
        m = pf.feols(
            pf_fml, data=pdf, vcov=_pf_vcov(v, cluster), weights=weights, **pyfixest_tol_kwargs(1e-13)
        )
        mods = m.all_fitted_models if hasattr(m, "all_fitted_models") else {"": m}
        for mod in mods.values():
            yy = mod._depvar
            t = mod.tidy()
            names = [rename.get(c, c) if rename else c for c in r.coefnames]
            b_pf = t.loc[names, "Estimate"].to_numpy()
            se_pf = t.loc[names, "Std. Error"].to_numpy()
            ours = r.tidy(yy, v)
            np.testing.assert_allclose(ours["Estimate"].to_numpy(), b_pf, **TOL)
            np.testing.assert_allclose(ours["Std. Error"].to_numpy(), se_pf, rtol=1e-8)
            assert r.N == mod._N, (r.N, mod._N)
            # p-values use the same t degrees of freedom
            np.testing.assert_allclose(
                ours["Pr(>|t|)"].to_numpy(), t.loc[names, "Pr(>|t|)"].to_numpy(), rtol=1e-5, atol=1e-12
            )
    return r


def test_one_local_fe_plus_week_dummies(ds, data):
    _compare(
        ds,
        data,
        "eng ~ feat:i(tbin) + posts | spell + week",
        "eng ~ i(tbin, feat) + posts | spell + week",
        ["CRV1", "iid", "hetero"],
        cluster="id_a",
    )


def test_two_local_fes(ds, data):
    # spell and account-week FEs, both nested in id_a (iterative demeaning)
    _compare(
        ds,
        data,
        "eng ~ feat:i(tbin) + posts:i(tbin) | spell + id_a^week",
        "eng ~ i(tbin, feat) + i(tbin, posts) | spell + id_a^week",
        ["CRV1", "iid"],
        cluster="id_a",
    )


def test_exact_two_way_equals_iteration(ds, data):
    # the Schur-complement solver and alternating projections give the same fit
    f = "eng ~ feat:i(tbin) + posts:i(tbin) | spell + id_a^week"
    a = dfeols.fit(ds, f, cluster="id_a", fe_method="exact", log_every=1e9)
    b = dfeols.fit(ds, f, cluster="id_a", fe_method="iterate", log_every=1e9)
    assert a.collinear == b.collinear and a.N == b.N
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(a.vcov(), b.vcov(), rtol=1e-7, atol=1e-14)


@pytest.mark.parametrize("cluster,weights", [("id_a", None), ("id_f", "wt"), ("week", None)])
def test_sparse_crossing_fe_equals_dense(ds, data, cluster, weights):
    # time dummies via per-level sums (no dense dummies) give the same fit
    f = "eng + eng2 ~ feat:i(tbin) + posts | spell + week"
    kw = dict(cluster=cluster, weights=weights, vcov=["CRV1", "hetero", "iid"], log_every=1e9)
    a = dfeols.fit(ds, f, crossing_fe="sparse", **kw)
    b = dfeols.fit(ds, f, crossing_fe="dense", **kw)
    assert a.N == b.N and a.coefnames == b.coefnames and a.fe_levels == b.fe_levels
    for y in ("eng", "eng2"):
        np.testing.assert_allclose(a.coef(y), b.coef(y), rtol=1e-9, atol=1e-12)
        for t in ("CRV1", "hetero", "iid"):
            np.testing.assert_allclose(a.vcov(y, t), b.vcov(y, t), rtol=1e-8, atol=1e-15)
        assert abs(a.outcomes[y]["rss"] - b.outcomes[y]["rss"]) < 1e-8 * b.outcomes[y]["rss"]


def test_sparse_two_crossing_fes(data, tmp_path):
    # regression: the second crossing FE's columns were offset twice ("index exceeds matrix
    # dimension"); the test above has only one crossing FE, whose offset is 0
    ds = dfeols.prep(
        data,
        shard_dir=tmp_path / "s",
        partition="id_a",
        fixed_effects=["spell", "week", "tenure"],
        clusters=["id_a"],
        target_rows=3000,
    )
    f = "eng ~ feat + posts | spell + week + tenure"
    kw = dict(cluster="id_a", vcov=["CRV1", "hetero"], log_every=1e9)
    a = dfeols.fit(ds, f, crossing_fe="sparse", **kw)
    b = dfeols.fit(ds, f, crossing_fe="dense", **kw)
    assert a.N == b.N and a.fe_levels == b.fe_levels
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-9, atol=1e-12)
    for t in ("CRV1", "hetero"):
        np.testing.assert_allclose(a.vcov(None, t), b.vcov(None, t), rtol=1e-8, atol=1e-15)
    m = pf.feols(
        "eng ~ feat + posts | spell + week + tenure",
        data=data.to_pandas(),
        vcov={"CRV1": "id_a"},
        **pyfixest_tol_kwargs(1e-13),
    )
    np.testing.assert_allclose(a.coef(), m.coef().to_numpy(), **TOL)


def test_sparse_crossing_fe_above_max_dummy_levels(data, tmp_path):
    # a crossing FE with 3,000 levels (> max_dummy_levels and SPARSE_GRAM_LEVELS): the sparse
    # path takes it and shards return their Gram matrices sparse (D4: carrier x quarter)
    import polars as pl

    from dfeols.worker import SPARSE_GRAM_LEVELS

    d = data.with_columns(cw=(pl.col("id_f") % 100) * 100 + pl.col("week"))
    assert d["cw"].n_unique() > SPARSE_GRAM_LEVELS
    ds = dfeols.prep(
        d,
        shard_dir=tmp_path / "s",
        partition="id_a",
        fixed_effects=["spell", "week", "cw"],
        clusters=["id_a"],
        target_rows=3000,
    )
    f = "eng ~ feat + posts | spell + cw"
    kw = dict(cluster="id_a", vcov=["CRV1", "hetero", "iid"], log_every=1e9)
    a = dfeols.fit(ds, f, **kw)
    assert a.options["crossing_fe"] == "sparse"
    b = dfeols.fit(ds, f, crossing_fe="dense", max_dummy_levels=10_000, **kw)
    assert a.N == b.N and a.fe_levels == b.fe_levels
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-9, atol=1e-12)
    for t in ("CRV1", "hetero", "iid"):
        np.testing.assert_allclose(a.vcov(None, t), b.vcov(None, t), rtol=1e-8, atol=1e-15)
    pdf = d.to_pandas()
    for v in ("CRV1", "hetero"):
        m = pf.feols(f, data=pdf, vcov=_pf_vcov(v, "id_a"), **pyfixest_tol_kwargs(1e-13))
        assert a.N == m._N
        np.testing.assert_allclose(a.coef(), m.coef().to_numpy(), **TOL)
        np.testing.assert_allclose(np.sqrt(np.diag(a.vcov(None, v))), m.se().to_numpy(), rtol=1e-7)
    # with crossing_fe="dense", a crossing FE this large is refused
    with pytest.raises(NotImplementedError, match="crossing_fe='dense'"):
        dfeols.fit(ds, f, crossing_fe="dense", **kw)


@pytest.mark.parametrize(
    "f",
    [
        "eng ~ feat + posts | spell + id_a^week + cw",  # two local FEs crossing within an account
        "eng ~ feat + posts | spell + id_a^week + id_a + cw",  # id_a redundant with both
        "eng ~ feat:i(tbin) + posts | spell + id_a + week",  # spell nested in id_a, few dummies
    ],
)
def test_sparse_several_local_fes(data, tmp_path, f):
    # plan B: several FEs nested in the partition on the sparse path (per-component Schur
    # solves for the dummy block), against dense dummies and pyfixest
    import polars as pl

    d = data.with_columns(cw=(pl.col("id_f") % 100) * 100 + pl.col("week"))
    ds = dfeols.prep(
        d,
        shard_dir=tmp_path / "s",
        partition="id_a",
        fixed_effects=["spell", "id_a^week", "id_a", "week", "cw"],
        clusters=["id_a"],
        target_rows=3000,
    )
    kw = dict(cluster="id_a", vcov=["CRV1", "hetero", "iid"], log_every=1e9)
    a = dfeols.fit(ds, f, crossing_fe="sparse", **kw)
    b = dfeols.fit(ds, f, crossing_fe="dense", max_dummy_levels=10_000, **kw)
    assert a.options["crossing_fe"] == "sparse" and b.options["crossing_fe"] == "dense"
    assert a.N == b.N and a.fe_levels == b.fe_levels
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-8, atol=1e-10)
    for t in ("CRV1", "hetero", "iid"):
        np.testing.assert_allclose(a.vcov(None, t), b.vcov(None, t), rtol=1e-7, atol=1e-14)
    m = pf.feols(
        f.replace("feat:i(tbin)", "i(tbin, feat)"),
        data=d.to_pandas(),
        vcov={"CRV1": "id_a"},
        **pyfixest_tol_kwargs(1e-13),
    )
    assert a.N == m._N
    np.testing.assert_allclose(a.coef(), m.coef().to_numpy(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.sqrt(np.diag(a.vcov(None, "CRV1"))), m.se().to_numpy(), rtol=1e-7)
    if "cw" in f:  # above max_dummy_levels, auto takes the sparse path
        assert dfeols.fit(ds, f, **kw).options["crossing_fe"] == "sparse"


@pytest.fixture(scope="module")
def ds_cw(data, tmp_path_factory):
    import polars as pl

    d = data.with_columns(
        cw=(pl.col("id_f") % 100) * 100 + pl.col("week"), fw=(pl.col("id_f") % 37) * 100 + pl.col("week")
    )
    ds = dfeols.prep(
        d,
        shard_dir=tmp_path_factory.mktemp("cw"),
        partition="id_a",
        fixed_effects=["spell", "id_a^week", "week", "cw", "fw", "id_f"],
        clusters=["id_a"],
        target_rows=3000,
    )
    return d, ds


@pytest.mark.parametrize(
    "f",
    [
        "eng ~ feat + posts | spell + cw",  # one local FE
        "eng ~ feat + posts | spell + id_a^week + cw",  # two local FEs
        "eng ~ feat + posts | cw + id_f",  # no local FE: 5,300 crossing levels
        "eng ~ feat:i(tbin) + posts | spell + week + id_f",  # two crossing FEs
    ],
)
def test_iterate_crossing_fes(ds_cw, f):
    # plan C: crossing-FE coefficients by conjugate gradient across shards, against pyfixest
    d, ds = ds_cw
    kw = dict(cluster="id_a", vcov=["CRV1", "hetero", "iid"], log_every=1e9)
    a = dfeols.fit(ds, f, crossing_fe="iterate", **kw)
    assert a.options["crossing_fe"] == "iterate"
    pdf = d.to_pandas().dropna(subset=["eng"])
    for v in ("CRV1", "hetero"):
        m = pf.feols(
            f.replace("feat:i(tbin)", "i(tbin, feat)"),
            data=pdf,
            vcov=_pf_vcov(v, "id_a"),
            **pyfixest_tol_kwargs(1e-13),
        )
        assert a.N == m._N
        np.testing.assert_allclose(a.coef(), m.coef().to_numpy(), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(np.sqrt(np.diag(a.vcov(None, v))), m.se().to_numpy(), rtol=1e-7)


@pytest.mark.parametrize(
    "f", ["eng ~ feat + posts | spell + fw + cw", "eng + eng2 ~ feat + posts | id_a^week + fw + cw"]
)
def test_iterate_level_space_and_fallback(ds_cw, f):
    # plan C's two modes: A = D~'W D~ summed in round 1 and CG on the coordinator (level space),
    # or a pass per step when A is over level_space_gb (with A's diagonal blocks as block-Jacobi
    # if those fit, else Jacobi), which warns; every mode and preconditioner equals pyfixest
    from dfeols.crossing import NNZ_BYTES, SlowConvergenceWarning

    d, ds = ds_cw
    kw = dict(cluster="id_a", crossing_fe="iterate", log_every=1e9)
    a = dfeols.fit(ds, f, **kw)
    cg = a.options["crossing_cg"]
    assert (cg["mode"], cg["preconditioner"], cg["passes"]) == ("level", "sgs", 0)
    assert sum(cg["block_nnz"]) < cg["A_nnz"]
    mid = (sum(cg["block_nnz"]) + cg["A_nnz"]) // 2 * NNZ_BYTES / 1e9  # blocks fit, A does not
    runs = {"level": a}
    for name, opt, want in [
        ("bjacobi", dict(crossing_precond="bjacobi"), ("level", "bjacobi")),
        ("jacobi", dict(crossing_precond="jacobi"), ("level", "jacobi")),
        ("blocks", dict(level_space_gb=mid), ("pass", "bjacobi")),
        ("off", dict(level_space_gb=0), ("pass", "jacobi")),
    ]:
        if want[0] == "pass":
            with pytest.warns(SlowConvergenceWarning, match="convergence may be slow"):
                runs[name] = dfeols.fit(ds, f, **kw, **opt)
        else:
            runs[name] = dfeols.fit(ds, f, **kw, **opt)
        c2 = runs[name].options["crossing_cg"]
        assert (c2["mode"], c2["preconditioner"]) == want
        assert c2["passes"] == (c2["steps"] if want[0] == "pass" else 0)
    assert "nonzeros" in runs["blocks"].options["crossing_cg"]["fallback_reason"]
    # block-Jacobi beats Jacobi on the same passes
    assert runs["blocks"].options["crossing_cg"]["steps"] < runs["off"].options["crossing_cg"]["steps"]
    pdf = d.to_pandas().dropna(subset=["eng"])
    for y in f.split("~")[0].split("+"):
        y = y.strip()
        m = pf.feols(
            f"{y} ~ {f.split('~')[1]}", data=pdf, vcov={"CRV1": "id_a"}, **pyfixest_tol_kwargs(1e-13)
        )
        for r in runs.values():
            assert r.N == m._N
            np.testing.assert_allclose(r.coef(y), m.coef().to_numpy(), rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(np.sqrt(np.diag(r.vcov(y))), m.se().to_numpy(), rtol=1e-7)


def test_iterate_equals_sparse_two_outcomes(ds_cw):
    _, ds = ds_cw
    f = "eng + eng2 ~ feat + posts | spell + week + id_f"
    kw = dict(cluster="id_a", vcov=["CRV1", "hetero", "iid"], weights="wt", log_every=1e9)
    a = dfeols.fit(ds, f, crossing_fe="iterate", **kw)
    b = dfeols.fit(ds, f, crossing_fe="sparse", **kw)
    assert a.N == b.N and a.fe_levels == b.fe_levels
    for y in ("eng", "eng2"):
        np.testing.assert_allclose(a.coef(y), b.coef(y), rtol=1e-8, atol=1e-10)
        for t in ("CRV1", "hetero", "iid"):
            np.testing.assert_allclose(a.vcov(y, t), b.vcov(y, t), rtol=1e-7, atol=1e-14)
        assert a.outcomes[y]["rss"] == pytest.approx(b.outcomes[y]["rss"], rel=1e-9)


def test_iterate_on_dask_equals_serial(ds_cw):
    # CG blocks go to the workers by scatter, shard state stays in each worker's cache
    distributed = pytest.importorskip("distributed")
    _, ds = ds_cw
    f = "eng ~ feat + posts | cw + id_f"
    kw = dict(cluster="id_a", crossing_fe="iterate", log_every=1e9)
    a = dfeols.fit(ds, f, **kw)
    with distributed.LocalCluster(n_workers=2, threads_per_worker=1, processes=False) as lc:
        with distributed.Client(lc) as client:
            ex = dfeols.DaskExecutor(client, tick=0.2)
            ex.scatter_min_bytes = 1  # scatter even this test's small blocks
            b = dfeols.fit(ds, f, executor=ex, workdir=str(ds.root / "runs" / "dask"), **kw)
    assert b.options["crossing_cg"]["steps"] == a.options["crossing_cg"]["steps"]
    np.testing.assert_allclose(a.coef(), b.coef(), rtol=1e-10)
    np.testing.assert_allclose(a.vcov(), b.vcov(), rtol=1e-9)


@pytest.mark.parametrize("crossing_fe", ["dense", "sparse", "iterate"])
def test_regressor_absorbed_by_crossing_fe_is_dropped(ds_cw, crossing_fe):
    # wk depends only on week, a crossing FE: fixest drops wk; the direct solves used to keep
    # it (dropping a dummy) and report an unidentified coefficient
    import polars as pl

    d, _ = ds_cw
    d = d.with_columns(wk=(pl.col("week") * 0.1) ** 2)
    ds = dfeols.prep(
        d,
        shard_dir=ds_cw[1].root.parent / f"wk_{crossing_fe}",
        partition="id_a",
        fixed_effects=["spell", "week"],
        clusters=["id_a"],
        target_rows=3000,
    )
    f = "eng ~ feat + wk + posts | spell + week"
    a = dfeols.fit(ds, f, cluster="id_a", crossing_fe=crossing_fe, log_every=1e9)
    assert a.collinear == ["wk"] and a.coefnames == ["feat", "posts"]
    m = pf.feols(
        f, data=d.to_pandas().dropna(subset=["eng"]), vcov={"CRV1": "id_a"}, **pyfixest_tol_kwargs(1e-13)
    )
    np.testing.assert_allclose(a.coef(), m.coef().to_numpy(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.sqrt(np.diag(a.vcov())), m.se().to_numpy(), rtol=1e-7)


def test_iterate_auto_and_limits(ds_cw):
    _, ds = ds_cw
    kw = dict(cluster="id_a", log_every=1e9)
    # no FE nested in the partition and a crossing FE above max_dummy_levels: auto iterates
    assert dfeols.fit(ds, "eng ~ feat | cw", **kw).options["crossing_fe"] == "iterate"
    with pytest.raises(NotImplementedError, match="IV with crossing_fe='iterate'"):
        dfeols.fit(ds, "eng ~ posts | cw | feat ~ eng2", **kw)
    with pytest.raises(RuntimeError, match="did not converge in 2 steps"):
        dfeols.fit(ds, "eng ~ feat | cw + id_f", crossing_fe="iterate", iter_maxiter=2, **kw)


@pytest.fixture(scope="module")
def singleton_data(data):
    # a crossing FE with many one-row levels; dropping those rows leaves new singletons in
    # spell (local) and tenure, so finding them all takes several passes across shards
    import polars as pl

    rng = np.random.default_rng(7)
    n = data.height
    rare = np.where(rng.random(n) < 0.03, np.arange(n) + 1000, rng.integers(0, 40, n))
    return data.with_columns(rare=pl.Series(rare))


# sparse needs an FE nested in the partition; by week, none is
@pytest.mark.parametrize("partition,crossing_fe", [("id_a", "sparse"), ("id_a", "dense"), ("week", "dense")])
def test_singletons_over_all_fes_match_fixest(singleton_data, tmp_path, partition, crossing_fe):
    # fixest/reghdfe drop singletons in every FE, iterating; dfeols must too, whatever the
    # partition (it used to drop only the local FE's singletons, so N depended on the key)
    ds = dfeols.prep(
        singleton_data,
        shard_dir=tmp_path / "s",
        partition=partition,
        fixed_effects=["spell", "week", "tenure", "rare"],
        clusters=["id_a"],
        target_rows=3000,
    )
    f = "eng ~ feat + posts | spell + tenure + rare"
    r = dfeols.fit(
        ds,
        f,
        cluster="id_a",
        vcov=["CRV1", "hetero"],
        crossing_fe=crossing_fe,
        max_dummy_levels=10_000,
        log_every=1e9,
    )
    pdf = singleton_data.to_pandas()
    for v in ("CRV1", "hetero"):
        vc = _pf_vcov(v, "id_a")
        m = pf.feols(f, data=pdf, vcov=vc, fixef_rm="singleton", **pyfixest_tol_kwargs(1e-13))
        assert r.N == m._N
        np.testing.assert_allclose(r.coef(), m.coef().to_numpy(), **TOL)
        np.testing.assert_allclose(np.sqrt(np.diag(r.vcov(None, v))), m.se().to_numpy(), rtol=1e-7)
    assert r.timing["singleton_passes"] >= 2


def test_exact_two_way_residuals_orthogonal():
    from dfeols.demean import exact_two_way

    rng = np.random.default_rng(1)
    n = 5000
    g = rng.integers(0, 40, n)  # 40 groups (components)
    a = g * 100 + rng.integers(0, 30, n)  # e.g. units within group
    b = g * 100 + rng.integers(0, 12, n)  # e.g. periods within group
    from dfeols.demean import densify

    a, b = densify(a)[0].astype(np.int64), densify(b)[0].astype(np.int64)
    x = rng.normal(size=(n, 3))
    w = rng.uniform(0.5, 2, n)
    r = exact_two_way(x, a, b, w)
    for codes in (a, b):
        for j in range(3):
            assert np.abs(np.bincount(codes, weights=w * r[:, j])).max() < 1e-10


def test_cluster_crossing_partition(ds, data):
    # follower clusters span shards -> per-cluster scores reduced on coordinator
    _compare(
        ds,
        data,
        "eng ~ feat:i(tbin) + posts | spell + week",
        "eng ~ i(tbin, feat) + posts | spell + week",
        "CRV1",
        cluster="id_f",
    )


def test_weights_and_multiple_outcomes(ds, data):
    _compare(
        ds,
        data,
        "eng + eng2 ~ feat:i(tbin) + posts | spell + week",
        "eng + eng2 ~ i(tbin, feat) + posts | spell + week",
        ["CRV1", "hetero"],
        cluster="id_a",
        weights="wt",
    )


def test_no_fe_with_intercept_and_dummies(ds, data):
    _compare(
        ds,
        data,
        "eng ~ feat + posts + i(tbin)",
        "eng ~ feat + posts + i(tbin)",
        ["iid", "hetero", "CRV1"],
        cluster="id_a",
    )


def test_fe_nested_in_cluster_dof(ds, data):
    # account FE nested in account clusters: nonnested small-sample rule
    _compare(
        ds,
        data,
        "eng ~ feat:i(tbin) + posts | id_a + week",
        "eng ~ i(tbin, feat) + posts | id_a + week",
        "CRV1",
        cluster="id_a",
    )


def test_week_cluster_with_global_fe(ds, data):
    # week FE (dummies) nested in week clusters; clusters cross shards
    _compare(
        ds,
        data,
        "eng ~ feat + posts | spell + week",
        "eng ~ feat + posts | spell + week",
        "CRV1",
        cluster="week",
    )


def test_only_dummy_fes(ds, data):
    # FEs that all cross shards: dummies plus a hidden constant, no Intercept reported
    r = _compare(
        ds,
        data,
        "eng ~ feat + posts | week",
        "eng ~ feat + posts | week",
        ["CRV1", "iid", "hetero"],
        cluster="id_a",
    )
    assert "Intercept" not in r.coefnames


def test_no_cache_path(ds, data):
    r1 = dfeols.fit(ds, "eng ~ feat + posts | spell", cluster="id_a", log_every=1e9)
    r2 = dfeols.fit(ds, "eng ~ feat + posts | spell", cluster="id_a", cache=False, log_every=1e9)
    np.testing.assert_allclose(r1.coef(), r2.coef(), rtol=1e-12)
    np.testing.assert_allclose(r1.vcov(), r2.vcov(), rtol=1e-10)


def test_dask_local_cluster(ds, data):
    distributed = pytest.importorskip("distributed")
    with (
        distributed.LocalCluster(
            n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None
        ) as lc,
        distributed.Client(lc) as client,
    ):
        ex = dfeols.DaskExecutor(client, max_workers=2, tick=0.5)
        _compare(
            ds,
            data,
            "eng ~ feat:i(tbin) + posts | spell + week",
            "eng ~ i(tbin, feat) + posts | spell + week",
            ["CRV1", "iid"],
            cluster="id_a",
            executor=ex,
        )
        _compare(
            ds,
            data,
            "eng ~ feat + posts | spell + week",
            "eng ~ feat + posts | spell + week",
            "CRV1",
            cluster="id_f",
            executor=ex,
        )


def test_status_file_and_result_roundtrip(ds, tmp_path):
    r = dfeols.fit(ds, "eng ~ feat + posts | spell", cluster="id_a", workdir=tmp_path, log_every=0)
    import json

    st = json.loads((tmp_path / "status.json").read_text())
    assert [s["stage"] for s in st["stages"]] == ["round1", "round2"]
    assert all(s["finished"] for s in st["stages"])
    r2 = dfeols.Result.load(tmp_path / "result.json")
    np.testing.assert_allclose(r.coef(), r2.coef())
    np.testing.assert_allclose(r.vcov(), r2.vcov())


def test_category_levels_on_workers_and_cached(tmp_path):
    """i() levels are read by executor tasks and cached in the shard folder
    (regression: one coordinator process read the column from all 5,000 shards,
    55 minutes at 1e10 rows before round 1 started)."""
    import json

    from dfeols.estimate import category_levels

    data = sim_panel(n_accounts=30, followers_per=20, weeks=12, seed=3)
    ds = dfeols.prep(
        data, shard_dir=tmp_path / "s", partition="id_a", fixed_effects=["spell", "week"], target_rows=500
    )
    calls = []

    class Counting(dfeols.SerialExecutor):
        def run(self, fn, tasks, on_result, progress, stage, local_help=False):
            calls.append((stage, len(tasks)))
            return super().run(fn, tasks, on_result, progress, stage, local_help)

    lv = category_levels(ds, ["tbin"], Counting())
    assert lv["tbin"] == sorted(data["tbin"].drop_nulls().unique().to_list())
    assert calls == [("levels", min(len(ds.shards), 4))]
    assert json.loads((ds.root / "_levels" / "tbin.json").read_text())["levels"] == lv["tbin"]
    assert category_levels(ds, ["tbin"], Counting()) == lv and len(calls) == 1  # cached
    r = dfeols.fit(ds, "eng ~ feat:i(tbin) | spell + week", executor=Counting(), log_every=1e9)
    assert calls[1:] and all(stage != "levels" for stage, _ in calls[1:])  # fit reused the cache
    assert len(r.coefnames) == len(lv["tbin"])
