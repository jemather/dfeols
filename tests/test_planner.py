"""`dfeols plan` (planner step 1) agrees with what prep and build_plan do for each key."""

import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import dfeols
from dfeols.cli import main
from dfeols.estimate import estimate_coordinator_gb
from dfeols.planner import format_report
from simdata import sim_panel


@pytest.fixture(scope="module")
def data():
    return sim_panel()


SPECS = [
    # one FE nested in every key -> sparse path or local
    ("eng ~ feat + posts | spell + week", "id_a", 2000),
    # spell nested in id_f, so id_f has two local FEs and dense week dummies
    ("eng ~ feat | spell + id_f + week", "id_a", 2000),
    # two local FEs and a crossing FE above max_dummy_levels: plan B
    ("eng ~ feat | spell + id_a + id_f", "id_a", 100),
    # no FE nested in tbin, crossing FEs above max_dummy_levels: plan C
    ("eng ~ feat | spell + week", "tbin", 20),
]


@pytest.mark.parametrize("formula, cluster, max_dummy", SPECS)
def test_plan_matches_prep_and_build_plan(data, tmp_path, formula, cluster, max_dummy):
    # prep counts levels over all rows, the planner over the sample: compare without NaNs
    assert dfeols.plan(data, formula)["sample_rows"] == data.filter(pl.col("eng").is_finite()).height
    data = data.filter(pl.col("eng").is_finite())
    r = dfeols.plan(data, formula, cluster=cluster, max_dummy_levels=max_dummy, target_rows=3000)
    assert r["rows"] == r["sample_rows"] == data.height
    fes = list(r["fes"])
    assert format_report(r)
    for o in r["options"]:
        key = o["key"].split("^")
        ds = dfeols.prep(
            data,
            shard_dir=tmp_path / o["key"],
            partition=key,
            fixed_effects=fes,
            clusters=[cluster],
            target_rows=3000,
            checksum=False,
        )
        for a in fes:
            assert r["fes"][a]["levels"] == ds.fes[a]["n_levels"]
            assert r["fes"][a]["nested_in"][o["key"]] == ds.fes[a]["nested_in_partition"]
            assert r["fes"][a]["nested_in_cluster"] == ds.fes[a]["nested_in_cluster"][cluster]
        assert o["cluster_nested"] == ds.clusters[cluster]["nested_in_partition"]
        assert o["largest_group"] == data.group_by(key).len()["len"].max()
        if o["status"] != "exists":
            with pytest.raises(NotImplementedError):
                dfeols.build_plan(ds, formula, cluster=cluster, max_dummy_levels=max_dummy)
            continue
        p = dfeols.build_plan(ds, formula, cluster=cluster, max_dummy_levels=max_dummy)
        assert sorted(o["local_fes"]) == sorted(k for k, _ in p.local_fes)
        assert o["crossing_fes"] == {k: L for k, _, L in p.global_fes}
        assert o["n_dummy"] == p.n_dummy
        assert p.sparse_global == (o["path"] in ("sparse", "B", "C"))
        assert p.crossing_iterate == (o["path"] == "C")
        assert o["coordinator_gb"] == round(estimate_coordinator_gb(p), 1)


def test_plan_paths(data):
    paths = {
        f: {o["key"]: o["path"] for o in dfeols.plan(data, f, cluster=c, max_dummy_levels=m)["options"]}
        for f, c, m in SPECS
    }
    assert paths[SPECS[0][0]] == {"spell": "sparse", "week": "sparse", "id_a": "sparse"}
    assert paths[SPECS[1][0]]["id_f"] == "dense"
    assert paths[SPECS[2][0]]["id_a"] == "B"
    assert paths[SPECS[3][0]]["tbin"] == "C"


def test_plan_missing_and_recommendation(data):
    d = data.with_columns(feat=data["feat"].scatter([0, 1, 2], None))
    r = dfeols.plan(d, fe="spell + week", vars=["eng", "feat"], partition=["id_a", "week"])
    nan = int(d["eng"].is_nan().sum())
    assert r["nulls"] == {"feat": 3, "eng": nan}  # NaN counts as missing, as in fit
    assert r["sample_rows"] == d.filter(pl.col("feat").is_not_null() & pl.col("eng").is_finite()).height
    assert [o["key"] for o in r["options"]] == ["id_a", "week"]  # smaller shards rank first
    assert r["recommended"] == "id_a"
    with pytest.raises(KeyError):
        dfeols.plan(d, "eng ~ feat | nosuch")


def test_plan_cli_json(data, tmp_path, capsys):
    src = tmp_path / "d.parquet"
    data.write_parquet(src)
    main(["plan", str(src), "--fe", "spell + week", "--vars", "eng,feat", "--cluster", "id_a", "--json"])
    r = json.loads(capsys.readouterr().out)
    assert r["recommended"] in {"spell", "week", "id_a"}
    main(["plan", str(src), "--formula", "eng ~ feat | spell + week", "--partition", "id_a^week"])
    assert "id_a^week" in capsys.readouterr().out


@pytest.mark.parametrize(
    "formula, key, gcols, exact",
    [
        ("eng ~ feat | spell + sw", "id_a", ("spell",), True),  # one local FE: B'B is A's pattern
        ("eng ~ feat | spell + id_a + sw", "id_a", ("id_a",), False),  # two: an upper bound
        ("eng ~ feat | spell + week", "tbin", (), True),  # none: the (spell, week) pairs
    ],
)
def test_a_size_against_fit(data, tmp_path, formula, key, gcols, exact):
    # the planner's count of A's nonzeros against the A that fit builds in round 1; sw is a
    # sparse crossing FE (each spell touches a few of its 997 levels, groups share pairs)
    from dfeols.planner import _a_size

    d = data.with_columns(sw=(pl.col("id_f") * 31 + pl.col("week")) % 997)
    f = dfeols.formula.parse(formula)
    ds = dfeols.prep(d, shard_dir=tmp_path, partition=key, fixed_effects=list(f.fes), target_rows=3000)
    a = dfeols.fit(ds, formula, crossing_fe="iterate", drop_singletons=False, log_every=1e9)
    fit_nnz = a.options["crossing_cg"]["A_nnz"]
    sample = d.lazy().filter(pl.col("eng").is_finite())
    cross = {c: (c,) for c in f.fes if not ds.fes[c]["nested_in_partition"]}
    L = {c: sample.select(c).unique().collect().height for c in cross}  # levels in the sample, as plan
    big = _a_size(sample, gcols, cross, L, budget=10**12)
    if gcols:
        assert big["bound"] > fit_nnz and not big["exact"]  # the bound overstates; not counted
        counted = _a_size(sample, gcols, cross, L, budget=big["bound"] - 1)
    else:
        counted = big
    assert counted["exact"] and not counted["over"]
    assert counted["nnz"] == fit_nnz if exact else counted["nnz"] >= fit_nnz
    over = _a_size(sample, gcols, cross, L, budget=fit_nnz // 2)
    assert over["over"] and over["nnz"] > fit_nnz // 2


def test_plan_level_space_or_fallback(data):
    # plan C keys report A against the level-space budget; over it, the pre-run warning
    f, c, m = SPECS[3]
    o = {x["key"]: x for x in dfeols.plan(data, f, cluster=c, max_dummy_levels=m)["options"]}["tbin"]
    assert o["solve"] == "level" and o["A"]["exact"] and "level space" in o["path_text"]
    r = dfeols.plan(data, f, cluster=c, max_dummy_levels=m, level_space_gb=0)
    o = {x["key"]: x for x in r["options"]}["tbin"]
    assert o["solve"] == "fallback" and "pass over the shards per CG step" in o["path_text"]
    (msg,) = [w for w in o["warnings"] if "may converge slowly" in w]
    assert "spell x week" in msg and "spell " in msg
    assert "may converge slowly" in format_report(r)


def test_partition_rule_smallest_crossing_subject_to_balance():
    # by firm, the crossing system is year's 10 levels; by year, firm's 100. A firm with half
    # the rows makes the firm key unbalanced, and the year key is recommended instead
    import numpy as np

    rng = np.random.default_rng(0)
    n = 10_000
    firm = rng.integers(0, 100, n)
    d = pl.DataFrame({"firm": firm, "year": rng.integers(0, 10, n), "x": rng.normal(size=n)})
    d = d.with_columns(y=pl.col("x") + pl.col("firm") * 0.01)
    r = dfeols.plan(d, "y ~ x | firm + year", target_rows=1100)
    assert r["recommended"] == "firm" and r["options"][0]["crossing_levels"] == 10
    skew = d.with_columns(firm=pl.when(pl.arange(0, n) < n // 2).then(0).otherwise(pl.col("firm")))
    r = dfeols.plan(skew, "y ~ x | firm + year", target_rows=1100)
    assert r["recommended"] == "year"
    o = {x["key"]: x for x in r["options"]}
    assert not o["firm"]["balanced"] and any("unbalanced" in w for w in o["firm"]["warnings"])


def test_plan_cells(data):
    r = dfeols.plan(data, "eng ~ feat | id_a + week", cluster="id_a")
    c = r["cells"]
    cells = data.filter(pl.col("eng").is_finite()).select("id_a", "week").n_unique()
    assert c["key"] == ["id_a", "week"] and c["cells"] == cells
    assert c["recommended"] == (r["sample_rows"] >= 10 * c["cells"]) and "cells" in format_report(r)
    assert not dfeols.plan(data, "eng ~ posts | spell | feat ~ eng2")["cells"]["eligible"]
