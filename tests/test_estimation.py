"""The `estimation` record library: canonical JSON, digest, writer/reader, spilling,
validator and redactor."""

import json
import math
import warnings

import numpy as np
import pytest

from dfeols import estimation as est
from dfeols.cli import main as dfeols_main


def lm_record():
    """The base R `lm` example from the design doc."""
    return {
        "schema": "estimation/0.1",
        "meta": {
            "id": "6f1c0000-0000-4000-8000-000000000001",
            "created": "2026-10-24T15:02:11Z",
            "status": "ok",
            "estimator": {"family": "linear", "method": "ols"},
            "responses": ["mpg"],
            "profiles": ["inference/0.1"],
        },
        "specification": {
            "call": "lm(mpg ~ wt, data = mtcars)",
            "formula": {"text": "mpg ~ wt", "syntax": "R"},
        },
        "metrics": {"nobs": 32, "df_residual": 30, "r2": 0.7528, "sigma": 3.046, "conf_level": 0.95},
        "parameters": [
            {
                "term": "(Intercept)",
                "estimate": 37.285,
                "std_error": 1.8776,
                "statistic": 19.858,
                "p_value": 8.24e-19,
            },
            {
                "term": "wt",
                "estimate": -5.3445,
                "std_error": 0.5591,
                "statistic": -9.559,
                "p_value": 1.29e-10,
            },
        ],
        "structures": {
            "vcov": [
                {
                    "kind": "matrix",
                    "labels": ["(Intercept)", "wt"],
                    "data": [[3.525, -1.0], [-1.0, 0.3126]],
                    "type": "iid",
                    "ssc": {"convention": "R lm"},
                    "default": True,
                }
            ]
        },
        "provenance": {
            "software": [
                {"name": "R", "version": "4.5.1", "role": "language"},
                {"name": "stats", "version": "4.5.1", "role": "estimator"},
            ],
            "environment": {"os": "Ubuntu 24.04", "blas": "OpenBLAS 0.3.26", "host": "valiant"},
            "agent": {"user": "ted", "host": "valiant"},
            "execution": {
                "invocation": {"working_directory": "/srv/dfeols", "command_line": "Rscript a.R"},
                "scheduler": {"type": "slurm", "JobID": "35448540", "Partition": "normal"},
            },
            "data": [
                {
                    "role": "input",
                    "name": "mtcars",
                    "location": "/home/ted/mtcars.csv",
                    "checksum": {"algorithm": "sha256", "value": "ab" * 32, "of": "file"},
                }
            ],
        },
    }


def fe_record(k=3, fe_levels=(1172, 26)):
    """A linear-fe/dfeols record with numpy values, a NaN and an iid + CRV1 vcov."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(k, k))
    V = a @ a.T
    terms = [f"x{i}" for i in range(k)]
    return {
        "meta": {
            "status": "ok",
            "estimator": {"family": "linear", "method": "ols"},
            "responses": ["y"],
            "profiles": ["linear-fe/0.1", "dfeols/0.1"],
        },
        "specification": {"options": {"drop_singletons": True, "tolerance": 1e-8}},
        "metrics": {
            "nobs": np.int64(1_838_676),
            "n_singletons": 12,
            "df_residual": 1_837_000,
            "r2_within": math.nan,
        },
        "parameters": [
            {"term": t, "estimate": b, "std_error": s, "p_value": 0.5}
            for t, b, s in zip(terms, rng.normal(size=k), np.sqrt(np.diag(V)))
        ],
        "structures": {
            "vcov": [
                {
                    "kind": "matrix",
                    "labels": terms,
                    "data": V,
                    "type": "iid",
                    "ssc": {"convention": "fixest"},
                },
                {
                    "kind": "matrix",
                    "labels": terms,
                    "data": V * 2,
                    "type": "CRV1",
                    "cluster": ["pitcher"],
                    "ssc": {"convention": "fixest", "factors": {"G_adj": 1172 / 1171}},
                    "default": True,
                },
            ],
            "fixed_effects": [
                {"kind": "object", "name": f"fe{i}", "levels": n} for i, n in enumerate(fe_levels)
            ],
        },
        "provenance": {"software": [{"name": "dfeols", "version": "0.7.0", "role": "estimator"}]},
        "extensions": {"dfeols": {"schema": "dfeols/0.1", "shards": {"folder": "/scratch/x", "count": 4}}},
    }


def errors(issues):
    return [i for i in issues if i.level == "error"]


def schemas():
    """The schemas in schema/; skips the test where they or jsonschema are missing."""
    pytest.importorskip("jsonschema")
    sc = est.load_schemas()
    if sc is None:
        pytest.skip("no schema/ folder")
    return sc


@pytest.fixture(params=["schema", "builtin"])
def mode(request):
    """Validate with the schemas in schema/, or with the built-in fallback checks."""
    return schemas() if request.param == "schema" else None


# ---------------------------------------------------------------- RFC 8785


@pytest.mark.parametrize(
    "x, s",
    [
        (0.0, "0"),
        (-0.0, "0"),
        (1.0, "1"),
        (4.5, "4.5"),
        (2e-3, "0.002"),
        (0.000001, "0.000001"),
        (1e-7, "1e-7"),
        (1e-27, "1e-27"),
        (1e21, "1e+21"),
        (1e30, "1e+30"),
        (1e23, "1e+23"),
        (123e18, "123000000000000000000"),
        (2.0**68, "295147905179352830000"),
        (333333333.33333329, "333333333.3333333"),
        (9007199254740992.0, "9007199254740992"),
        (5e-324, "5e-324"),
        (-1.7976931348623157e308, "-1.7976931348623157e+308"),
        (-0.5, "-0.5"),
        (1.5e-6, "0.0000015"),
    ],
)
def test_es_number(x, s):
    assert est._es_number(x) == s


def test_canonical_json_rfc8785_example():
    """RFC 8785 section 3.2.2."""
    obj = {
        "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 0.000000000000000000000000001],
        # \u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/ in the RFC's JavaScript
        "string": chr(0x20AC)
        + "$"
        + chr(0x0F)
        + chr(0x0A)
        + "A'B"
        + chr(0x22)
        + chr(0x5C) * 2
        + chr(0x22)
        + "/",
        "literals": [None, True, False],
    }
    esc = chr(0x5C)
    want = (
        '{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"string":"'
        + chr(0x20AC)
        + "$"
        + esc
        + "u000f"
        + esc
        + "nA'B"
        + esc
        + '"'
        + esc * 4
        + esc
        + '"/"}'
    )
    assert est.canonical_json(obj) == want.encode()


def test_canonical_json_sorts_keys_by_utf16():
    """RFC 8785 section 3.2.3: the emoji (a surrogate pair) sorts before U+FB33."""
    keys = [chr(c) for c in (0x20AC, 0x0D, 0xFB33, 0x31, 0x1F600, 0x80, 0xF6)]
    out = json.loads(est.canonical_json({k: 0 for k in keys}))
    assert list(out) == [chr(c) for c in (0x0D, 0x31, 0x80, 0xF6, 0x20AC, 0x1F600, 0xFB33)]


def test_canonical_json_rejects_what_it_cannot_represent():
    with pytest.raises(est.EstimationError):
        est.canonical_json(math.nan)
    with pytest.raises(est.EstimationError):
        est.canonical_json(2**60)


# ---------------------------------------------------------------- digest


def test_digest_ignores_order_and_annotations_but_not_results():
    rec = est.add_digest(lm_record())
    assert est.check_digest(rec) is True
    reordered = json.loads(json.dumps(rec, sort_keys=True))
    assert est.check_digest(reordered) is True
    rec["extensions"] = {"harness": {"schema": "h/0.1", "pass": True}}
    rec["history"] = [{"when": "x", "by": {"name": "h"}, "action": "extended"}]
    rec["provenance"]["environment"]["os"] = "other"
    assert est.check_digest(rec) is True
    rec["parameters"][1]["estimate"] = -5.3446
    assert est.check_digest(rec) is False
    rec = lm_record()
    assert est.check_digest(rec) is None


def test_digest_same_for_numpy_and_python_values():
    a = lm_record()
    b = lm_record()
    b["structures"]["vcov"][0]["data"] = np.array(b["structures"]["vcov"][0]["data"])
    b["metrics"]["nobs"] = np.int32(32)
    assert est.core_digest(a) == est.core_digest(b)


# ---------------------------------------------------------------- writer and reader


def test_write_read_round_trip(tmp_path):
    p = tmp_path / "m.estimation.json"
    out = est.write(fe_record(), p)
    assert out["schema"] == est.SCHEMA and out["meta"]["id"] and out["meta"]["created"]
    assert out["history"][0]["action"] == "created"
    assert "attachments" not in out  # small: nothing spilled
    text = p.read_text()
    assert '"NaN"' in text and "NaN," not in text.replace('"NaN"', "")
    rec = est.read(p, check=True)
    assert math.isnan(rec["metrics"]["r2_within"]) and rec["metrics"]["nobs"] == 1_838_676
    assert est.check_digest(rec) is True
    assert errors(est.validate(rec, base_dir=tmp_path)) == []
    raw = est.read(p, nonfinite="string")
    assert raw["metrics"]["r2_within"] == "NaN"


def test_write_does_not_change_the_input(tmp_path):
    rec = lm_record()
    before = json.dumps(rec, sort_keys=True)
    est.write(rec, tmp_path / "a.estimation.json", redact_identifying=True)
    assert json.dumps(rec, sort_keys=True) == before


def test_text_fields_named_like_nonfinite_stay_strings(tmp_path):
    rec = lm_record()
    rec["parameters"][1]["term"] = "Inf"
    p = tmp_path / "a.estimation.json"
    est.write(rec, p)
    assert est.read(p)["parameters"][1]["term"] == "Inf"


def test_digest_can_be_turned_off(tmp_path):
    out = est.write(lm_record(), tmp_path / "a.estimation.json", digest=False)
    assert "core_digest" not in out["meta"]
    assert any(i.level == "info" and "no digest" in i.message for i in est.validate(out))


def test_bare_nan_is_rejected():
    with pytest.raises(est.EstimationError):
        est.loads('{"schema": "estimation/0.1", "metrics": {"x": NaN}}')


def test_naive_datetime_is_rejected():
    import datetime

    with pytest.raises(est.EstimationError):
        est.to_json({"t": datetime.datetime(2026, 1, 1)})
    t = datetime.datetime(2026, 1, 1, 12, tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))
    assert est.to_json(t) == "2026-01-01T17:00:00Z"


# ---------------------------------------------------------------- spilling


def test_large_vcov_spills_to_parquet_and_loads_back(tmp_path):
    rec = fe_record(k=150)  # 22,500 elements per vcov
    p = tmp_path / "big.estimation.json"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # spilled, so no size warning
        out = est.write(rec, p)
    links = out["structures"]["vcov"]
    assert [v["kind"] for v in links] == ["link", "link"]
    assert links[1]["type"] == "CRV1" and links[1]["ssc"]["convention"] == "fixest" and links[1]["default"]
    assert "data" not in links[1] and "labels" not in links[1]
    att = est.attachment(out, links[1]["attachment"])
    assert att["media_type"] == "application/vnd.apache.parquet" and att["rows"] == 150
    assert (tmp_path / att["path"]).exists() and att["path"].startswith("big.")
    rec2 = est.read(p)
    s = est.load_structure(p, rec2, rec2["structures"]["vcov"][1])
    np.testing.assert_array_equal(np.array(s["data"]), np.asarray(rec["structures"]["vcov"][1]["data"]))
    assert s["labels"] == rec["structures"]["vcov"][1]["labels"] and s["kind"] == "matrix"
    assert errors(est.validate(rec2, base_dir=tmp_path)) == []  # profiles still see type and ssc
    assert est.check_digest(rec2) is True
    # a file that changed afterwards is caught
    (tmp_path / att["path"]).write_bytes(b"x")
    assert any("sha256" in i.message for i in est.validate(rec2, base_dir=tmp_path))
    with pytest.raises(est.EstimationError):
        est.load_structure(p, rec2, rec2["structures"]["vcov"][1])


def test_spill_by_size_and_json_fallback(tmp_path):
    rec = est.to_json(lm_record())
    rec["structures"]["tree"] = {"kind": "tree", "nodes": [{"id": i, "split": "x" * 40} for i in range(5000)]}
    ids = est.spill(rec, tmp_path, "t", max_bytes=100_000)
    assert ids == ["tree"]
    att = est.attachment(rec, "tree")
    assert att["media_type"] == "application/json"
    assert rec["structures"]["vcov"][0]["kind"] == "matrix"  # small one stays
    back = est.load_structure(tmp_path / "t.json", rec, rec["structures"]["tree"])
    assert back["kind"] == "tree" and len(back["nodes"]) == 5000


def test_size_limits(tmp_path):
    rec = lm_record()
    rec["parameters"] = [
        {"term": f"x{i}", "estimate": 0.1, "std_error": 0.1, "p_value": 0.5} for i in range(20_000)
    ]
    with pytest.warns(UserWarning, match="elements"):
        est.write(rec, tmp_path / "w.estimation.json")
    rec["parameters"] = [{"term": "t" * 200 + str(i), "estimate": 0.1} for i in range(80_000)]
    with pytest.raises(est.EstimationError, match="limit"):
        est.write(rec, tmp_path / "f.estimation.json")
    assert not (tmp_path / "f.estimation.json").exists()


def test_count_elements():
    assert est.count_elements([[1, 2], [3, 4]]) == 4
    assert est.count_elements([{"a": [1, 2, 3]}, {"a": 1}]) == 2


# ---------------------------------------------------------------- validator


def test_lm_example_is_valid():
    issues = est.validate(lm_record())
    assert errors(issues) == []
    assert [i.message for i in issues if i.level == "warning"] == []


def test_missing_sections_and_bad_shapes(mode):
    rec = lm_record()
    del rec["parameters"], rec["metrics"]
    rec["meta"]["status"] = "great"
    rec["structures"]["x"] = {"kind": "link", "attachment": "nope"}
    rec["oops"] = 1
    issues = est.validate(rec, schema=mode)
    paths = {(i.level, i.path) for i in issues}
    assert ("error", "parameters") in paths
    assert ("warning", "metrics") in paths
    assert ("error", "meta.status") in paths
    assert ("error", "structures.x.attachment") in paths
    assert ("warning", "oops") in paths
    assert errors(est.validate({"schema": "other/1"}, schema=mode))


def test_profiles(mode):
    rec = est.prepare(fe_record())
    assert errors(est.validate(rec, schema=mode)) == []
    del rec["metrics"]["n_singletons"]
    del rec["specification"]["options"]["drop_singletons"]
    del rec["structures"]["vcov"][1]["ssc"]
    rec["structures"]["fixed_effects"][1].pop("levels")
    del rec["extensions"]["dfeols"]
    del rec["parameters"][0]["p_value"]
    found = errors(est.validate(rec, schema=mode))
    paths = {i.path for i in found}
    expected = {"metrics.n_singletons", "specification.options.drop_singletons", "extensions.dfeols"}
    if mode is None:
        expected |= {"structures.vcov.ssc", "structures.fixed_effects[1]", "parameters[0]"}
    else:
        expected |= {"structures.vcov", "structures.fixed_effects[1].levels", "parameters[0].p_value"}
    assert expected <= paths
    # dfeols/0.1 includes the linear-fe rules: each problem is reported once
    assert len(found) == len(paths)
    rec["meta"]["profiles"] = ["glm/0.1"]
    assert any(i.level == "info" and "glm/0.1" in i.message for i in est.validate(rec, schema=mode))


def test_digest_mismatch_is_a_warning():
    rec = est.prepare(lm_record())
    rec["metrics"]["r2"] = 0.99
    w = [i for i in est.validate(rec) if i.path == "meta.core_digest"]
    assert w and w[0].level == "warning"


def test_schema_plugin():
    calls = []

    def check(rec):
        calls.append(rec["meta"]["id"])
        return [est.Issue("error", "meta", "from plug-in")]

    issues = est.validate(lm_record(), schema=check)
    assert calls and any(i.message == "from plug-in" for i in issues)


def test_default_uses_schema_dir_or_says_it_fell_back(monkeypatch, tmp_path):
    rec = est.prepare(fe_record())
    del rec["metrics"]["n_singletons"]
    monkeypatch.setenv("ESTIMATION_SCHEMA_DIR", str(tmp_path))  # no schemas there
    issues = est.validate(rec)
    assert any(i.level == "info" and "built-in" in i.message for i in issues)
    assert "metrics.n_singletons" in {i.path for i in errors(issues)}
    monkeypatch.delenv("ESTIMATION_SCHEMA_DIR")
    sc = schemas()
    assert est.find_schema_dir() == sc.directory
    issues = est.validate(rec)
    assert not any("built-in" in i.message for i in issues)
    assert [i.path for i in errors(issues)] == ["metrics.n_singletons"]


def test_profile_file_resolves_the_core_offline():
    """A profile's allOf refers to ../estimation-0.1.json; with the other schemas as
    documents it resolves locally, and the core's rules apply."""
    sc = schemas()
    check = est.jsonschema_check(sc.profiles["linear-fe/0.1"], sc.documents())
    rec = est.prepare(fe_record())
    assert check(rec) == []
    rec["meta"]["status"] = "great"  # a core rule
    del rec["metrics"]["n_singletons"]  # a profile rule
    assert {i.path for i in check(rec)} >= {"meta.status", "metrics.n_singletons"}


def test_size_limits_match_schema():
    lim = schemas().core["x-size-limits"]
    assert (lim["warn_bytes"], lim["warn_elements"], lim["fail_bytes"]) == (
        est.WARN_BYTES,
        est.WARN_ELEMENTS,
        est.FAIL_BYTES,
    )


def test_jsonschema_plugin():
    pytest.importorskip("jsonschema")
    schema = {"type": "object", "required": ["meta"], "properties": {"meta": {"required": ["nope"]}}}
    assert any("nope" in i.message for i in est.validate(lm_record(), schema=schema))


# ---------------------------------------------------------------- redactor


def test_redact_removes_identifying_fields_and_keeps_digest():
    rec = est.prepare(lm_record())
    out = est.redact(rec)
    prov = out["provenance"]
    assert "agent" not in prov and "host" not in prov["environment"]
    assert "location" not in prov["data"][0] and prov["data"][0]["checksum"]["value"] == "ab" * 32
    assert prov["execution"]["scheduler"] == {"type": "slurm", "Partition": "normal"}
    h = out["history"][-1]
    assert h["action"] == "redacted" and h["method"] == "removed"
    assert "provenance.agent" in h["fields"] and "provenance.data[0].location" in h["fields"]
    assert est.check_digest(out) is True
    assert "agent" in rec["provenance"]  # input unchanged
    assert "valiant" not in json.dumps({k: v for k, v in out.items() if k != "history"})


def test_redact_hash_compares_equal_with_same_salt():
    a = est.redact(lm_record(), salt="s1")
    b = est.redact(lm_record(), salt=b"s1")
    c = est.redact(lm_record(), salt="s2")
    host = lambda r: r["provenance"]["environment"]["host"]
    assert host(a) == host(b) != host(c)
    assert host(a).startswith("hmac-sha256:")
    assert a["history"][-1]["salt_id"] == b["history"][-1]["salt_id"] != c["history"][-1]["salt_id"]


def test_redact_takes_marks_from_core_and_claimed_profiles():
    sc = schemas()
    rec = est.prepare(fe_record())
    rec["provenance"]["agent"] = {"user": "ted"}
    # claims dfeols/0.1: the core's marks and the dfeols profile's (its extension block)
    assert est.record_identifying_paths(rec, sc) == sorted(est.DEFAULT_IDENTIFYING)
    out = est.redact(rec)
    assert "folder" not in out["extensions"]["dfeols"]["shards"] and "agent" not in out["provenance"]
    # without the claim, the dfeols block isn't covered by any schema it follows
    rec["meta"]["profiles"] = ["linear-fe/0.1"]
    out = est.redact(rec)
    assert (
        out["extensions"]["dfeols"]["shards"]["folder"] == "/scratch/x" and "agent" not in out["provenance"]
    )
    assert "extensions.dfeols.shards.folder" not in est.record_identifying_paths(rec, sc)


def test_identifying_paths_follow_refs_across_files():
    sc = schemas()
    alone = set(est.identifying_paths(sc.profiles["dfeols/0.1"]))
    across = set(est.identifying_paths(sc.profiles["dfeols/0.1"], sc.documents()))
    assert alone < across and "provenance.agent" in across - alone


def test_redact_refuses_digest_sections():
    with pytest.raises(est.EstimationError):
        est.redact(lm_record(), paths=["specification.call"])


def test_identifying_paths_from_schema():
    schema = {
        "$defs": {
            "host": {"type": "string", "x-identifying": True},
            "input": {
                "type": "object",
                "properties": {"location": {"x-identifying": True}, "rows": {"type": "integer"}},
            },
        },
        "properties": {
            "provenance": {
                "properties": {
                    "agent": {"type": "object", "x-identifying": True},
                    "environment": {"properties": {"host": {"$ref": "#/$defs/host"}}},
                    "data": {"type": "array", "items": {"$ref": "#/$defs/input"}},
                }
            },
            "extensions": {"additionalProperties": {"properties": {"path": {"x-identifying": True}}}},
        },
    }
    paths = est.identifying_paths(schema)
    assert paths == [
        "extensions.*.path",
        "provenance.agent",
        "provenance.data[*].location",
        "provenance.environment.host",
    ]
    rec = lm_record()
    rec["extensions"] = {"a": {"schema": "a/1", "path": "/x"}, "b": {"schema": "b/1"}}
    out = est.redact(rec, schema=schema)
    assert "path" not in out["extensions"]["a"] and "location" not in out["provenance"]["data"][0]


# ---------------------------------------------------------------- command line


def test_cli(tmp_path, capsys, monkeypatch):
    p = tmp_path / "a.estimation.json"
    est.write(lm_record(), p)
    assert est.main(["validate", str(p)]) == 0
    assert "valid" in capsys.readouterr().out

    o = tmp_path / "r.estimation.json"
    monkeypatch.setenv("ESTIMATION_REDACT_SALT", "pepper")
    assert est.main(["redact", str(p), "-o", str(o), "--hash"]) == 0
    r = est.read(o)
    assert r["provenance"]["agent"].startswith("hmac-sha256:") and est.check_digest(r) is True

    assert est.main(["digest", str(o)]) == 0
    assert "matches" in capsys.readouterr().out

    bad = json.loads(p.read_text())
    bad["parameters"][0]["estimate"] = 0
    del bad["meta"]["status"]
    (tmp_path / "b.estimation.json").write_text(json.dumps(bad))
    assert est.main(["digest", str(tmp_path / "b.estimation.json")]) == 1
    with pytest.raises(SystemExit) as e:
        dfeols_main(["estimation", "validate", "-q", str(tmp_path / "b.estimation.json")])
    assert e.value.code == 1
    assert "meta.status" in capsys.readouterr().out


def test_cli_schema_options(tmp_path, capsys):
    p = tmp_path / "a.estimation.json"
    rec = est.prepare(fe_record())
    del rec["metrics"]["n_singletons"]
    p.write_text(json.dumps(rec))
    assert est.main(["validate", "--no-schema", str(p)]) == 1
    assert "built-in" not in capsys.readouterr().out
    assert est.main(["validate", "--schema-dir", str(schemas().directory), str(p)]) == 1
    assert "[linear-fe/0.1] 'n_singletons' is a required property" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        est.main(["validate", "--schema-dir", str(tmp_path), str(p)])
    o = tmp_path / "r.json"
    assert est.main(["redact", str(p), "-o", str(o), "--schema-dir", str(schemas().directory)]) == 0
    assert "folder" not in json.loads(o.read_text())["extensions"]["dfeols"]["shards"]


def test_cli_hash_needs_salt(tmp_path, monkeypatch):
    p = tmp_path / "a.estimation.json"
    est.write(lm_record(), p)
    monkeypatch.delenv("ESTIMATION_REDACT_SALT", raising=False)
    with pytest.raises(SystemExit):
        est.main(["redact", str(p), "-o", str(tmp_path / "o.json"), "--hash"])


@pytest.mark.parametrize("name", ["lm-minimal", "dfeols-linear-fe", "iv"])
def test_schema_examples_validate(name):
    sc = schemas()
    rec = est.read(sc.directory / "examples" / f"{name}.estimation.json")
    assert errors(est.validate(rec)) == []
