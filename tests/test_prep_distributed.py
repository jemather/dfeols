"""Distributed prep (map/reduce on workers) must produce the same dataset as
single-node prep: same rows, FE level counts, nesting flags, and identical
fits."""

import os
import shutil

import numpy as np
import polars as pl
import pytest

import dfeols
from dfeols.prep_dist import _nrows
from dfeols.synth import InfluencerDGP, PanelDGP, generate, load_truth


def _gen(tmp_path, d, name):
    raw = tmp_path / name
    generate(raw, d, verbose=False)
    return raw, load_truth(raw)


CASES = [
    (
        "panel",
        PanelDGP(n_units=6000, units_per_chunk=1000, T=30, n_groups=60, gt_confound=1.0),
        [
            ("unit", "y ~ x1:i(cat) + x2 + x3 | unit + time", "group"),
            ("group", "y ~ x1:i(cat) + x2 + x3 | unit + group^time", "group"),
        ],
    ),
    (
        "influencer",
        InfluencerDGP(n_accounts=120, accounts_per_chunk=30, spells_logmean=2.5, weeks=40),
        [
            ("id_a", "eng ~ feat:i(tbin) + posts | spell + week", "id_f"),
            ("id_a", "eng ~ feat:i(tbin) | spell + id_a^week", "id_a"),
        ],
    ),
]


@pytest.mark.parametrize("name,d,fits", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("execname", ["serial", "dask"])
def test_distributed_prep_matches(tmp_path, name, d, fits, execname):
    raw, truth = _gen(tmp_path, d, name)
    src = str(raw / "*.parquet")
    if execname == "dask":
        distributed = pytest.importorskip("distributed")
        lc = distributed.LocalCluster(n_workers=2, threads_per_worker=1, dashboard_address=None)
        client = distributed.Client(lc)
        ex = dfeols.DaskExecutor(client, max_workers=2, tick=0.5)
    else:
        ex = dfeols.SerialExecutor()
    try:
        for part, fml, cl in fits:
            kw = dict(partition=part, target_rows=8000, **truth["prep"])
            a = dfeols.prep(src, shard_dir=tmp_path / f"single_{part}", overwrite=True, **kw)
            b = dfeols.prep(src, shard_dir=tmp_path / f"dist_{part}", overwrite=True, executor=ex, **kw)
            assert b.meta.get("prep_mode") == "distributed"
            assert a.n_rows == b.n_rows
            assert sum(s["rows"] for s in b.shards) == b.n_rows
            for k in a.fes:
                assert a.fes[k]["n_levels"] == b.fes[k]["n_levels"], k
                assert a.fes[k]["nested_in_partition"] == b.fes[k]["nested_in_partition"], k
                assert a.fes[k].get("nested_in_cluster") == b.fes[k].get("nested_in_cluster"), k
            for k in a.clusters:
                assert a.clusters[k]["n_levels"] == b.clusters[k]["n_levels"]
            # every partition group lands in exactly one shard
            pk = a.partition
            g = (
                pl.concat(
                    [
                        pl.read_parquet(b.shard_paths(s), columns=pk).with_columns(
                            pl.lit(s["id"]).alias("_s")
                        )
                        for s in b.shards
                    ]
                )
                .group_by(pk)
                .agg(pl.col("_s").n_unique().alias("k"))
            )
            assert g["k"].max() == 1
            ra = dfeols.fit(a, fml, cluster=cl, vcov=["CRV1", "hetero"], log_every=1e9)
            rb = dfeols.fit(b, fml, cluster=cl, vcov=["CRV1", "hetero"], executor=ex, log_every=1e9)
            assert ra.N == rb.N and ra.coefnames == rb.coefnames and ra.fe_levels == rb.fe_levels
            np.testing.assert_allclose(ra.coef(), rb.coef(), rtol=1e-9, atol=1e-12)
            for t in ("CRV1", "hetero"):
                np.testing.assert_allclose(ra.vcov(type=t), rb.vcov(type=t), rtol=1e-8, atol=1e-15)
    finally:
        if execname == "dask":
            client.close()
            lc.close()


def test_forced_merge_and_flat_layout(tmp_path):
    """max_pieces=0 merges every shard; the result equals the unmerged layout."""
    d = PanelDGP(n_units=3000, units_per_chunk=300, T=20, n_groups=40)
    raw, truth = _gen(tmp_path, d, "p")
    src = str(raw / "*.parquet")
    kw = dict(partition="group", target_rows=6000, executor=dfeols.SerialExecutor(), **truth["prep"])
    a = dfeols.prep(src, shard_dir=tmp_path / "pieces", rows_per_task=1e4, **kw)
    b = dfeols.prep(src, shard_dir=tmp_path / "merged", rows_per_task=1e4, max_pieces=0, **kw)
    assert any(len(s["files"]) > 1 for s in a.shards)  # groups span files -> several pieces
    assert all(len(s["files"]) == 1 and "_merged_" in s["files"][0] for s in b.shards)
    import os

    assert not [n for n in os.listdir(tmp_path / "merged" / "shards") if "_t" in n]  # pieces removed
    left = {p.name for p in (tmp_path / "pieces").iterdir() if p.is_dir()}
    assert left <= {"shards", "_prep_status"}, left  # no temp dirs left
    f = "y ~ x1:i(cat) + x2 + x3 | unit + group^time"
    ra = dfeols.fit(a, f, cluster="group", log_every=1e9)
    rb = dfeols.fit(b, f, cluster="group", log_every=1e9)
    np.testing.assert_allclose(ra.coef(), rb.coef(), rtol=1e-10)
    np.testing.assert_allclose(ra.vcov(), rb.vcov(), rtol=1e-9)


def _square(x):
    return {"v": x * x, "timing": {"total": 0.0}}


def test_coordinator_helps_with_no_workers(tmp_path):
    """local_help: with zero workers connected the coordinator runs the tasks."""
    distributed = pytest.importorskip("distributed")
    from dfeols.progress import Progress

    with distributed.LocalCluster(n_workers=0, dashboard_address=None) as lc, distributed.Client(lc) as c:
        ex = dfeols.DaskExecutor(c, max_workers=2, tick=0.2)
        prog = Progress(tmp_path, label="t", every=1e9)
        prog.add_stage("s", {i: 1 for i in range(6)})
        got = {}
        ex.run(
            _square,
            [(i, 1, (i,)) for i in range(6)],
            lambda t, r: got.__setitem__(t, r["v"]),
            prog,
            "s",
            local_help=True,
        )
    assert got == {i: i * i for i in range(6)}


@pytest.mark.parametrize("name,d,fits", CASES, ids=[c[0] for c in CASES])
def test_hash_ids_and_range_shards_match(tmp_path, name, d, fits):
    """Forcing the 1e10-scale paths on small data (hash ids for nested FEs and
    clusters, hash-range shard assignment) gives the same fits as dense codes."""
    raw, truth = _gen(tmp_path, d, name)
    src = str(raw / "*.parquet")
    ex = dfeols.SerialExecutor()
    for part, fml, cl in fits:
        kw = dict(partition=part, target_rows=8000, **truth["prep"])
        a = dfeols.prep(src, shard_dir=tmp_path / f"codes_{part}", overwrite=True, **kw)
        b = dfeols.prep(
            src,
            shard_dir=tmp_path / f"hash_{part}",
            overwrite=True,
            executor=ex,
            hash_levels=0,
            range_groups=0,
            **kw,
        )
        assert b.meta["shard_assignment"] == "range"
        assert any(m.get("id") == "hash" for m in list(b.fes.values()) + list(b.clusters.values()))
        assert a.n_rows == b.n_rows and sum(s["rows"] for s in b.shards) == b.n_rows
        rows = [s["rows"] for s in b.shards]
        assert max(rows) <= 3 * 8000 + max(1, 0)  # range bins stay near target
        for k in a.fes:
            assert a.fes[k]["n_levels"] == b.fes[k]["n_levels"]
        ra = dfeols.fit(a, fml, cluster=cl, vcov=["CRV1", "hetero"], log_every=1e9)
        rb = dfeols.fit(b, fml, cluster=cl, vcov=["CRV1", "hetero"], log_every=1e9)
        assert ra.N == rb.N and ra.G == rb.G and ra.fe_levels == rb.fe_levels
        np.testing.assert_allclose(ra.coef(), rb.coef(), rtol=1e-9, atol=1e-12)
        for t in ("CRV1", "hetero"):
            np.testing.assert_allclose(ra.vcov(type=t), rb.vcov(type=t), rtol=1e-8, atol=1e-15)


@pytest.mark.parametrize(
    "part,fml",
    [
        ("group", "y ~ x1:i(cat) + x2 + x3 | unit + group^time"),  # groups scattered over files
        ("unit", "y ~ x1:i(cat) + x2 + x3 | unit + time"),  # units contiguous in files
    ],
)
def test_two_level_shuffle_matches(tmp_path, part, fml):
    d = PanelDGP(n_units=3000, units_per_chunk=300, T=20, n_groups=40)
    raw, truth = _gen(tmp_path, d, "p")
    src = str(raw / "*.parquet")
    kw = dict(partition=part, target_rows=6000, **truth["prep"])
    a = dfeols.prep(src, shard_dir=tmp_path / "single", **kw)
    b = dfeols.prep(
        src,
        shard_dir=tmp_path / "two",
        executor=dfeols.SerialExecutor(),
        rows_per_task=1e4,
        max_files=0,
        **kw,
    )
    assert b.meta["shuffle"].startswith("two-level")
    assert not (tmp_path / "two" / "_buckets").exists()
    assert all(len(s["files"]) <= 2 for s in b.shards)  # one piece per bucket per shard
    ra = dfeols.fit(a, fml, cluster="group", vcov=["CRV1", "hetero"], log_every=1e9)
    rb = dfeols.fit(b, fml, cluster="group", vcov=["CRV1", "hetero"], log_every=1e9)
    assert ra.N == rb.N and ra.fe_levels == rb.fe_levels
    np.testing.assert_allclose(ra.coef(), rb.coef(), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ra.vcov(type="CRV1"), rb.vcov(type="CRV1"), rtol=1e-8, atol=1e-15)


def test_range_shards_on_hashed_multicolumn_key(tmp_path):
    d = PanelDGP(n_units=2000, units_per_chunk=400, T=15, n_groups=30)
    raw, truth = _gen(tmp_path, d, "p")
    src = str(raw / "*.parquet")
    kw = dict(partition=["group", "cat"], target_rows=5000, fixed_effects=["time"], clusters=["group"])
    a = dfeols.prep(src, shard_dir=tmp_path / "single", **kw)
    b = dfeols.prep(src, shard_dir=tmp_path / "range", executor=dfeols.SerialExecutor(), range_groups=0, **kw)
    assert b.meta["shard_assignment"] == "range"
    g = (
        pl.concat(
            [
                pl.read_parquet(b.shard_paths(s), columns=["group", "cat"]).with_columns(
                    pl.lit(s["id"]).alias("_s")
                )
                for s in b.shards
            ]
        )
        .group_by(["group", "cat"])
        .agg(pl.col("_s").n_unique().alias("k"))
    )
    assert g["k"].max() == 1
    f = "y ~ x2 + x3 | time"
    ra = dfeols.fit(a, f, cluster="group", log_every=1e9)
    rb = dfeols.fit(b, f, cluster="group", log_every=1e9)
    np.testing.assert_allclose(ra.coef(), rb.coef(), rtol=1e-9)
    np.testing.assert_allclose(ra.vcov(), rb.vcov(), rtol=1e-8)


class _ZombieExecutor(dfeols.SerialExecutor):
    """Every prep task also leaves the files of a second, unaccepted attempt,
    as when the scheduler gives up on a busy worker and reruns its task
    elsewhere while the first attempt keeps writing."""

    def __init__(self, root):
        super().__init__()
        self.dirs = [root / "shards", root / "_buckets"]
        self.planted = 0

    def _listing(self):
        return {d / n for d in self.dirs if d.exists() for n in os.listdir(d)}

    def run(self, fn, tasks, on_result, progress, stage, local_help=False):
        def twice(*args):
            before = self._listing()
            res = fn(*args)
            for f in self._listing() - before:
                if f.suffix == ".parquet":
                    shutil.copy(f, f.with_name(f.stem + "_zombie.parquet"))
                    self.planted += 1
            return res

        return super().run(twice, tasks, on_result, progress, stage, local_help)


@pytest.mark.parametrize(
    "opts",
    [dict(), dict(max_pieces=0), dict(max_files=0)],
    ids=["direct", "merge", "two-level"],
)
def test_duplicate_task_attempts_are_ignored(tmp_path, opts):
    """Files from task attempts whose result was not accepted never enter the
    dataset (regression: 'encoded 2,000,400,000 rows, expected 2,000,000,000')."""
    d = PanelDGP(n_units=3000, units_per_chunk=300, T=20, n_groups=40)
    raw, truth = _gen(tmp_path, d, "p")
    src = str(raw / "*.parquet")
    kw = dict(partition="group", target_rows=6000, **truth["prep"])
    a = dfeols.prep(src, shard_dir=tmp_path / "single", **kw)
    ex = _ZombieExecutor(tmp_path / "z")
    b = dfeols.prep(src, shard_dir=tmp_path / "z", executor=ex, rows_per_task=1e4, **opts, **kw)
    assert ex.planted > 0
    on_disk = sum(_nrows(str(tmp_path / "z" / f)) for s in b.shards for f in s["files"])
    assert on_disk == b.n_rows == a.n_rows
    assert not [n for n in os.listdir(tmp_path / "z" / "shards") if "zombie" in n]
    f = "y ~ x1:i(cat) + x2 + x3 | unit + group^time"
    ra = dfeols.fit(a, f, cluster="group", log_every=1e9)
    rb = dfeols.fit(b, f, cluster="group", log_every=1e9)
    assert ra.N == rb.N
    np.testing.assert_allclose(ra.coef(), rb.coef(), rtol=1e-10)


def test_interrupted_prep_is_not_reused(tmp_path):
    """A shard_dir left by a prep that died (no metadata file) is cleared, so
    its stale pieces cannot end up in the new dataset."""
    d = PanelDGP(n_units=1000, units_per_chunk=250, T=10, n_groups=20)
    raw, truth = _gen(tmp_path, d, "p")
    src = str(raw / "*.parquet")
    kw = dict(partition="group", target_rows=4000, executor=dfeols.SerialExecutor(), **truth["prep"])
    first = dfeols.prep(src, shard_dir=tmp_path / "sd", **kw)
    (tmp_path / "sd" / "dfeols_meta.json").unlink()  # as if the first prep had crashed
    stale = tmp_path / "sd" / "shards" / "s000000_t00099_stale_0.parquet"
    shutil.copy(tmp_path / "sd" / first.shards[0]["files"][0], stale)
    b = dfeols.prep(src, shard_dir=tmp_path / "sd", **kw)
    assert not stale.exists()
    assert sum(_nrows(str(tmp_path / "sd" / f)) for s in b.shards for f in s["files"]) == b.n_rows
