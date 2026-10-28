"""`dfeols plan` (planner step 1) agrees with what prep and build_plan do for each key."""

import json
import sys
from pathlib import Path

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
    # no FE nested in tbin, crossing FEs above max_dummy_levels: not built (fit refuses)
    ("eng ~ feat | spell + week", "tbin", 20),
]


@pytest.mark.parametrize("formula, cluster, max_dummy", SPECS)
def test_plan_matches_prep_and_build_plan(data, tmp_path, formula, cluster, max_dummy):
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
        assert p.sparse_global == (o["path"] in ("sparse", "B"))
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
    assert r["sample_rows"] == d.height - 3 and r["nulls"] == {"feat": 3}
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
