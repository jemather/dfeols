"""dfeols writes its own `estimation` records (dfeols.record); every record a fit writes
here is validated against the packaged schemas (core + inference, linear-fe, dfeols)."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import dfeols
from dfeols import estimation
from simdata import sim_panel

pytest.importorskip("jsonschema")
warnings.filterwarnings("ignore")

PUBLICATION = {
    "citation": {
        "type": "article-journal",
        "title": "A paper",
        "author": [{"family": "Doe", "given": "J."}],
        "issued": {"date-parts": [[2026]]},
    },
    "locator": {"table": "3", "column": "9", "label": "Table 3, column 9"},
    "status": "working-paper",
}


@pytest.fixture(scope="module")
def data():
    import polars as pl

    return sim_panel().with_columns(feat2=2.0 * pl.col("feat"))  # collinear with feat


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


def _check(path: Path) -> dict:
    """The record at `path` has no validation errors, and no warnings either."""
    rec = estimation.read(path, nonfinite="string")
    issues = estimation.validate(rec, base_dir=path.parent)
    bad = [i for i in issues if i.level != "info"]
    assert not bad, "\n".join(map(str, bad))
    assert estimation.check_digest(rec) is True
    return rec


def _fit(ds, tmp_path, formula, **kw):
    r = dfeols.fit(ds, formula, workdir=tmp_path, log_every=1e9, **kw)
    return r, _check(tmp_path / "result.estimation.json")


def test_default_record_fe_cluster(ds, tmp_path):
    r, rec = _fit(
        ds, tmp_path, "eng ~ feat:i(tbin) + posts | spell + week", cluster="id_a", vcov=["CRV1", "iid"]
    )
    assert rec["meta"]["profiles"] == ["inference/0.1", "linear-fe/0.1", "dfeols/0.1"]
    assert rec["meta"]["responses"] == ["eng"]
    # parameters match Result.tidy, in model order, with canonical names
    tidy = r.tidy()
    rows = rec["parameters"]
    assert [p["term"] for p in rows] == list(tidy.index)
    np.testing.assert_allclose([p["estimate"] for p in rows], tidy["Estimate"], rtol=1e-15)
    np.testing.assert_allclose([p["std_error"] for p in rows], tidy["Std. Error"], rtol=1e-15)
    np.testing.assert_allclose([p["p_value"] for p in rows], tidy["Pr(>|t|)"], rtol=1e-12)
    np.testing.assert_allclose([p["conf_low"] for p in rows], tidy["2.5%"], rtol=1e-12)
    by = {p["term"]: p["canonical"] for p in rows}
    t1 = next(t for t in by if t.startswith("tbin::"))
    lvl = t1.split("::")[1].split(":")[0]
    assert by[t1]["name"] == f"{lvl}.tbin#c.feat"
    assert by[t1]["factors"] == [{"variable": "tbin", "level": lvl}, {"variable": "feat"}]
    assert by["posts"] == {"name": "posts", "factors": [{"variable": "posts"}]}
    # vcov: both types, CRV1 default and full, ssc written out
    vc = rec["structures"]["vcov"]
    assert [v["type"] for v in vc] == ["CRV1", "iid"] and vc[0]["default"] and not vc[1]["default"]
    np.testing.assert_allclose(vc[0]["data"], r.vcov(), rtol=1e-15)
    assert vc[0]["n_clusters"] == {"id_a": r.G} and vc[0]["df_t"] == r.G - 1
    assert vc[0]["ssc"]["convention"] == "fixest" and "G/(G-1)" in vc[0]["ssc"]["factors"]
    # metrics and sample
    m = rec["metrics"]
    assert m["nobs"] == r.N and m["n_singletons"] == r.N_dropped and m["nobs_full"] == r.N + r.N_dropped
    assert m["df_residual"] == r.outcomes["eng"]["vcov"]["iid"][1] == vc[1]["df_t"]
    assert m["r2_within"]["value"] == r.outcomes["eng"]["r2_within"]
    smp = rec["specification"]["sample"]
    assert smp["rows_input"] == ds.n_rows
    assert smp["rows_input"] - sum(smp["dropped"].values()) == r.N
    assert [fe["name"] for fe in rec["structures"]["fixed_effects"]] == ["spell", "week"]
    assert [fe["levels"] for fe in rec["structures"]["fixed_effects"]] == r.fe_levels
    assert rec["specification"]["options"]["drop_singletons"] is True
    # provenance and the dfeols extension
    prov = rec["provenance"]
    assert prov["software"][0]["name"] == "dfeols" and prov["software"][0]["role"] == "estimator"
    assert prov["agent"]["host"] and prov["environment"]["hardware"]["threads"]
    assert prov["execution"]["elapsed_seconds"] > 0 and prov["data"][0]["rows"] == ds.n_rows
    ext = rec["extensions"]["dfeols"]
    assert ext["shards"]["n_shards"] == len(ds.shards) and len(ext["shards"]["meta_sha256"]) == 64
    assert ext["absorption"] == {
        "spell": {"method": "local", "nested_in_partition": True},
        "week": {"method": "crossing_sparse", "nested_in_partition": False},
    }
    assert ext["replay"]["call"].startswith("dfeols.fit(ds, 'eng ~ feat:i(tbin) + posts | spell + week'")
    assert rec["history"][0]["by"]["name"] == "dfeols"


def test_fe_nested_in_cluster(ds, tmp_path):
    r, rec = _fit(ds, tmp_path, "eng ~ feat + posts | id_a + week", cluster="id_a")
    fe = {f["name"]: f for f in rec["structures"]["fixed_effects"]}
    assert fe["id_a"]["nested_in"] == ["id_a"] and "nested_in" not in fe["week"]
    assert rec["metrics"]["df_absorbed_nested"] > 0


def test_weights_and_multiple_outcomes(ds, tmp_path):
    """One record per response (Ted, 2026-10-24), sharing the fit id."""
    r = dfeols.fit(
        ds,
        "eng + eng2 ~ feat + posts | spell",
        cluster="id_a",
        vcov=["CRV1", "hetero"],
        weights="wt",
        workdir=tmp_path,
        log_every=1e9,
    )
    assert not (tmp_path / "result.estimation.json").exists()
    recs = {y: _check(tmp_path / f"result.{y}.estimation.json") for y in ("eng", "eng2")}
    fit_ids = {rec["extensions"]["dfeols"]["fit"]["id"] for rec in recs.values()}
    assert len(fit_ids) == 1 and len({rec["meta"]["id"] for rec in recs.values()}) == 2
    for y, rec in recs.items():
        assert rec["meta"]["responses"] == [y]
        assert rec["meta"]["estimator"]["method"] == "wls"
        assert rec["specification"]["weights"] == {"variable": "wt", "type": "analytic"}
        assert [p["term"] for p in rec["parameters"]] == ["feat", "posts"]
        np.testing.assert_allclose([p["estimate"] for p in rec["parameters"]], r.coef(y), rtol=1e-15)
        assert [v["type"] for v in rec["structures"]["vcov"]] == ["CRV1", "HC1"]
        np.testing.assert_allclose(rec["structures"]["vcov"][1]["data"], r.vcov(y, "hetero"), rtol=1e-15)
        assert rec["metrics"]["r2_within"]["value"] == r.outcomes[y]["r2_within"]
        assert rec["metrics"]["sum_weights"] > 0
        assert rec["extensions"]["dfeols"]["fit"]["responses"] == ["eng", "eng2"]
    with pytest.raises(ValueError, match="pass y="):
        r.to_estimation()
    assert r.to_estimation("eng2")["meta"]["responses"] == ["eng2"]


def test_no_fe_intercept_and_collinear(ds, tmp_path):
    r, rec = _fit(ds, tmp_path, "eng ~ feat + feat2 + posts + i(tbin)", vcov="hetero")
    terms = [p["term"] for p in rec["parameters"]]
    assert terms[:4] == ["Intercept", "feat", "feat2", "posts"]
    assert rec["parameters"][0]["canonical"] == {"name": "(Intercept)", "factors": []}
    assert rec["meta"]["profiles"] == ["inference/0.1"] and "dfeols" in rec["extensions"]
    drop = [p for p in rec["parameters"] if p.get("dropped")]
    assert [p["term"] for p in drop] == r.collinear == ["feat2"]
    t = next(p for p in rec["parameters"] if p["term"].startswith("tbin::"))
    assert t["canonical"]["name"] == "tbin" + t["term"].split("::")[1]
    assert t["canonical"]["factors"][0]["reference"] == rec["specification"]["factor_terms"][0]["reference"]
    assert rec["structures"]["fixed_effects"] == [] and rec["metrics"]["df_absorbed"] == 0
    assert "r2" in rec["metrics"] and rec["meta"]["messages"][0]["code"] == "collinear"


def test_redacted_record(ds, tmp_path):
    _fit(ds, tmp_path, "eng ~ feat + posts | spell", cluster="id_a", redact=True)
    rec = _check(tmp_path / "result.estimation.json")
    text = json.dumps(rec)
    assert "agent" not in rec["provenance"]
    assert "hostname" not in rec["provenance"]["environment"]
    assert "location" not in rec["provenance"]["data"][0]
    assert "folder" not in rec["extensions"]["dfeols"]["shards"]
    assert "working_directory" not in rec["provenance"]["execution"]["invocation"]
    assert str(tmp_path) not in text and str(Path.cwd()) not in text
    assert [h["action"] for h in rec["history"]] == ["created", "redacted"]


def test_publication_and_record_path(ds, tmp_path):
    out = tmp_path / "t3c9.estimation.json"
    dfeols.fit(
        ds,
        "eng ~ feat + posts | spell",
        workdir=tmp_path / "wd",
        record=out,
        publication=PUBLICATION,
        log_every=1e9,
    )
    rec = _check(out)
    assert rec["publication"]["locator"]["label"] == "Table 3, column 9"
    assert not (tmp_path / "wd" / "result.estimation.json").exists()
    dfeols.fit(ds, "eng ~ feat + posts | spell", workdir=tmp_path / "off", record=False, log_every=1e9)
    assert not (tmp_path / "off" / "result.estimation.json").exists()


def test_from_saved_result(ds, tmp_path):
    r, rec = _fit(ds, tmp_path, "eng ~ feat + posts | spell", cluster="id_a")
    loaded = dfeols.Result.load(tmp_path / "result.json")
    again = loaded.write_estimation(tmp_path / "again.estimation.json", title="again")["eng"]
    _check(tmp_path / "again.estimation.json")
    assert again["parameters"] == rec["parameters"]
    assert again["metrics"] == rec["metrics"]
    assert again["meta"]["title"] == "again"


def test_result_saved_before_records(ds, tmp_path):
    """A result.json from before records existed still gives a valid record (no
    provenance or sample counts, so a warning for the missing expected section only)."""
    dfeols.fit(
        ds, "eng ~ feat + posts | spell", cluster="id_a", workdir=tmp_path, record=False, log_every=1e9
    )
    d = json.loads((tmp_path / "result.json").read_text())
    for k in ("options", "terms", "run"):
        del d[k]
    for o in d["outcomes"].values():
        del o["tss"]
    (tmp_path / "old.json").write_text(json.dumps(d))
    old = dfeols.Result.load(tmp_path / "old.json")
    old.write_estimation(tmp_path / "old.estimation.json")
    rec = estimation.read(tmp_path / "old.estimation.json")
    issues = estimation.validate(rec)
    assert not [i for i in issues if i.level == "error"], issues
    assert [i.path for i in issues if i.level == "warning"] == ["provenance"]


def test_record_failure_does_not_fail_fit(ds, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no")

    monkeypatch.setattr(dfeols.record, "write_estimation", boom)
    r = dfeols.fit(ds, "eng ~ feat | spell", workdir=tmp_path, log_every=1e9)
    assert (tmp_path / "result.json").exists() and r.N > 0


def test_slurm_fields_and_their_redaction(ds, tmp_path, monkeypatch):
    env = {
        "SLURM_JOB_ID": "35448540",
        "SLURM_JOB_NAME": "parsons",
        "SLURM_JOB_PARTITION": "normal",
        "SLURM_JOB_ACCOUNT": "acct-secret",
        "SLURM_JOB_NODELIST": "c[001-004]",
        "SLURM_JOB_USER": "ted-secret",
        "SLURM_SUBMIT_DIR": "/scratch/ted-secret/run",
        "SLURM_CPUS_ON_NODE": "8",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _, rec = _fit(ds, tmp_path / "a", "eng ~ feat | spell")
    s = rec["provenance"]["execution"]["scheduler"]
    assert s["type"] == "slurm" and s["JobID"] == "35448540" and s["Partition"] == "normal"
    assert s["Account"] == "acct-secret" and s["AllocCPUS"] == "8"
    _, red = _fit(ds, tmp_path / "b", "eng ~ feat | spell", redact=True)
    assert "secret" not in json.dumps(red) and "35448540" not in json.dumps(red)
    assert red["provenance"]["execution"]["scheduler"]["Partition"] == "normal"


def test_dask_executor_and_profile(ds, tmp_path):
    from types import SimpleNamespace

    distributed = pytest.importorskip("distributed")
    prof = SimpleNamespace(
        name="oscer",
        scheduler="slurm",
        queue="normal",
        account="acct-secret",
        project=None,
        scratch="/scratch/secret",
        prologue=["module load secret"],
        env={"X": "secret"},
        worker={"cores": 1, "memory": "4GB"},
        source="/home/secret/site.toml",
    )
    with (
        distributed.LocalCluster(n_workers=2, threads_per_worker=1, dashboard_address=None) as lc,
        distributed.Client(lc) as c,
    ):
        ex = dfeols.DaskExecutor(c, max_workers=2, tick=0.5)
        ex.profile = prof
        _, rec = _fit(ds, tmp_path / "a", "eng ~ feat + posts | spell", cluster="id_a", executor=ex)
        _, red = _fit(
            ds, tmp_path / "b", "eng ~ feat + posts | spell", cluster="id_a", executor=ex, redact=True
        )
    ext = rec["extensions"]["dfeols"]
    assert ext["executor"] == "DaskExecutor" and ext["dask"]["scheduler_address"].startswith("tcp://")
    assert ext["dask"]["profile"]["account"] == "acct-secret" and "workers" not in ext["dask"]
    assert len(rec["provenance"]["execution"]["workers"]) == 2
    assert "processor_seconds" not in rec["provenance"]["execution"]  # workers' CPU time isn't known
    text = json.dumps(red)
    assert "secret" not in text and "tcp://" not in text and "127.0.0.1" not in text
    assert red["extensions"]["dfeols"]["dask"]["profile"]["worker"] == {"cores": 1, "memory": "4GB"}


# ---------------------------------------------------------------- row_key and source checksums


def test_rowset_hash_properties():
    import polars as pl

    from dfeols import rowhash

    df = pl.DataFrame({"x": [1.5, -0.0, float("nan"), 2.0], "g": ["a", "b", None, "a"], "k": [1, 2, 3, 4]})
    full = rowhash.rowset(df, ["x", "g", "k"])
    # order- and split-invariant; column order doesn't matter
    parts = [rowhash.rowset(df[2:], ["k", "g", "x"]), rowhash.rowset(df[:2].reverse(), ["x", "g", "k"])]
    assert rowhash.combine(parts) == full and full["rows"] == 4
    # -0.0 == 0.0 and every NaN alike; a changed value or a dropped row changes it
    same = df.with_columns(x=pl.Series([1.5, 0.0, float("nan"), 2.0]))
    assert rowhash.rowset(same, ["x", "g", "k"]) == full
    assert rowhash.rowset(df.with_columns(k=pl.Series([1, 2, 3, 5])), ["x", "g", "k"]) != full
    assert rowhash.rowset(df[:3], ["x", "g", "k"]) != full
    # pinned: a change of algorithm must change its name (dfeols-rowset-v1)
    assert rowhash.hexdigest(full) == PINNED_ROWSET == _rowset_reference(df.to_dicts())


PINNED_ROWSET = "23d77b231955dab9fc3757c5a2c33543"


def _rowset_reference(rows: list[dict]) -> str:
    """dfeols-rowset-v1 in plain Python (no numpy), to check the vectorised one."""
    import hashlib
    import math
    import struct

    M = (1 << 64) - 1

    def mix(x):
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & M
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & M
        return x ^ (x >> 31)

    def h8(b):
        return int.from_bytes(hashlib.sha256(b).digest()[:8], "little")

    def bits(v):
        if v is None:
            return 0x6A09E667F3BCC908
        if isinstance(v, float):
            v = math.nan if math.isnan(v) else v + 0.0
            return struct.unpack("<Q", struct.pack("<d", v))[0]
        if isinstance(v, str):
            return h8(v.encode())
        return v & M

    lanes = []
    for seed in (0x9E3779B97F4A7C15, 0xD1B54A32D192ED03):
        total = 0
        for row in rows:
            h = seed
            for c in sorted(row):
                h = mix(h ^ mix(bits(row[c]) ^ h8(c.encode())))
            total = (total + h) & M
        lanes.append(total)
    return "".join(f"{v:016x}" for v in lanes)


def test_row_key_same_sample_across_partitions(data, ds, tmp_path):
    other = dfeols.prep(
        data,
        shard_dir=tmp_path / "s2",
        partition="spell",
        fixed_effects=["spell", "week", "id_a"],
        clusters=["id_a", "week"],
        target_rows=5000,
    )
    f = "eng ~ feat + posts | spell"
    _, a = _fit(ds, tmp_path / "a", f, cluster="id_a")
    _, b = _fit(other, tmp_path / "b", f, cluster="id_a")
    _, c = _fit(ds, tmp_path / "c", "eng ~ feat + posts | spell", cluster="id_a", drop_singletons=False)
    ka, kb, kc = (r["extensions"]["dfeols"]["row_key"] for r in (a, b, c))
    assert ka == kb and ka["rows"] == a["metrics"]["nobs"]
    assert ka["columns"] == ["eng", "feat", "id_a", "posts", "spell"]
    assert kc["value"] != ka["value"]  # singletons kept: another sample


def test_source_checksum_frame(ds, data, tmp_path):
    src = ds.meta["source"]
    assert src["kind"] == "frame" and src["rows"] == data.height
    assert src["checksum"]["of"] == "in-memory" and src["checksum"]["method"] == "dfeols-rowset-v1"
    _, rec = _fit(ds, tmp_path, "eng ~ feat | spell")
    inp = rec["provenance"]["data"]
    assert [d["role"] for d in inp] == ["source", "estimation"]
    assert inp[0]["checksum"] == src["checksum"]


def test_source_checksum_files_and_prep_drops(data, tmp_path):
    import hashlib

    import polars as pl

    raw = tmp_path / "raw"
    raw.mkdir()
    d = data.with_columns(spell=pl.when(pl.int_range(pl.len()) < 7).then(None).otherwise(pl.col("spell")))
    d[: d.height // 2].write_parquet(raw / "part-0.parquet")
    d[d.height // 2 :].write_parquet(raw / "part-1.parquet")
    ds2 = dfeols.prep(
        raw, shard_dir=tmp_path / "s", partition="id_a", fixed_effects=["spell"], target_rows=5000
    )
    src = ds2.meta["source"]
    shas = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    assert {f["name"]: f["sha256"] for f in src["files"]} == shas
    lines = "".join(f"{shas[n]}  {n}\n" for n in sorted(shas))
    assert src["checksum"] == {
        **src["checksum"],
        "of": "manifest",
        "value": hashlib.sha256(lines.encode()).hexdigest(),
    }
    assert src["rows"] == data.height and ds2.n_rows == data.height - 7
    _, rec = _fit(ds2, tmp_path / "wd", "eng ~ feat | spell")
    smp = rec["specification"]["sample"]
    assert smp["rows_input"] == data.height and smp["dropped"]["missing_keys"] == 7
    assert smp["rows_input"] - sum(smp["dropped"].values()) == rec["metrics"]["nobs"]
    assert rec["provenance"]["data"][0]["location"] == str(raw.resolve())
    one = dfeols.prep(
        raw / "part-0.parquet",
        shard_dir=tmp_path / "s1",
        partition="id_a",
        fixed_effects=["spell"],
        target_rows=5000,
    )
    assert one.meta["source"]["checksum"] == {
        "algorithm": "sha256",
        "value": shas["part-0.parquet"],
        "of": "file",
    }
    off = dfeols.prep(
        raw, shard_dir=tmp_path / "s0", partition="id_a", fixed_effects=["spell"], checksum=False
    )
    assert "checksum" not in off.meta["source"] and "sha256" not in off.meta["source"]["files"][0]
