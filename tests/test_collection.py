"""dfeols.write_collection: several fits as one estrec-collection/0.1 file, validated
with estrec (each member as a record)."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import estrec as estimation

import dfeols
from simdata import sim_panel

pytest.importorskip("jsonschema")
warnings.filterwarnings("ignore")

CITATION = {"type": "article-journal", "title": "A paper", "issued": {"date-parts": [[2026]]}}


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return dfeols.prep(
        sim_panel(),
        shard_dir=tmp_path_factory.mktemp("shards"),
        partition="id_a",
        fixed_effects=["spell", "week", "id_a"],
        clusters=["id_a", "week"],
        target_rows=3000,
    )


@pytest.fixture(scope="module")
def fits(ds, tmp_path_factory):
    wd = tmp_path_factory.mktemp("fits")
    base = dfeols.fit(
        ds, "eng ~ posts | spell", cluster="id_a", workdir=wd / "a", label="base", log_every=1e9
    )
    full = dfeols.fit(ds, "eng ~ posts + feat | spell + week", workdir=wd / "b", label="full", log_every=1e9)
    two = dfeols.fit(ds, "eng + posts ~ feat | spell", workdir=wd / "c", label="two", log_every=1e9)
    return base, full, two


def _issues(path):
    coll = estimation.read_collection(path)
    issues = estimation.validate_collection(coll, base_dir=Path(path).parent)
    return coll, [i for i in issues if i.level != "info"]


def test_batch_one_member_per_response(fits, tmp_path):
    base, full, two = fits
    p = tmp_path / "runs.estrec-collection.json"
    out = dfeols.write_collection([base, full, two], p)
    coll, bad = _issues(p)
    assert not bad, "\n".join(map(str, bad))
    assert coll["meta"]["kind"] == "batch"
    assert coll["meta"]["producer"]["name"] == "dfeols"
    assert [m["label"] for m in coll["members"]] == ["base", "full", "two.eng", "two.posts"]
    assert [m["record"]["meta"]["responses"] for m in coll["members"]][2:] == [["eng"], ["posts"]]
    # members are the fits' records: same estimates, valid digests and hashes
    rec = coll["members"][0]["record"]
    np.testing.assert_allclose([q["estimate"] for q in rec["parameters"]], base.coef(), rtol=1e-15)
    for m in coll["members"]:
        assert estimation.check_digest(m["record"]) is True
        assert m["sha256"] == estimation.member_sha256(m["record"])
    assert out["members"][0]["id"] == coll["members"][0]["id"]


def test_paper_with_locators_and_mapping(fits, tmp_path):
    base, full, _ = fits
    p = tmp_path / "table3.estrec-collection.json"
    dfeols.write_collection(
        {"col1": base, "col2": full},
        p,
        kind="paper",
        publication={"citation": CITATION},
        locators=["Table 3, col. 1", {"table": "3", "column": "2"}],
        title="Table 3",
    )
    coll, bad = _issues(p)
    assert not bad, "\n".join(map(str, bad))
    assert [m["label"] for m in coll["members"]] == ["col1", "col2"]
    assert coll["members"][0]["locator"] == {"label": "Table 3, col. 1"}
    assert coll["members"][1]["locator"] == {"table": "3", "column": "2"}
    # the collection's locators are not copied into the records
    assert "publication" not in coll["members"][0]["record"]
    assert coll["meta"]["title"] == "Table 3"


def test_records_pass_through_and_redact(fits, tmp_path):
    base, full, _ = fits
    rec = estimation.read(Path(base.workdir) / "result.estrec.json")
    p = tmp_path / "mixed.estrec-collection.json"
    coll = dfeols.write_collection([rec, full], p, redact=True)
    _, bad = _issues(p)
    assert not bad, "\n".join(map(str, bad))
    assert coll["members"][0]["id"] == rec["meta"]["id"]
    for m in coll["members"]:
        r = m["record"]
        assert "agent" not in r.get("provenance", {})
        assert "folder" not in r["extensions"]["dfeols"]["shards"]


def test_members_file(fits, tmp_path):
    p = tmp_path / "big.estrec-collection.json"
    dfeols.write_collection(list(fits), p, members="jsonl")
    on_disk = json.loads(p.read_text())
    assert all("attachment" in m and "record" not in m for m in on_disk["members"])
    assert (tmp_path / "big.members.jsonl").exists()
    _, bad = _issues(p)
    assert not bad, "\n".join(map(str, bad))


def test_label_count_must_match(fits, tmp_path):
    with pytest.raises(ValueError, match="one per response"):
        dfeols.write_collection(list(fits), tmp_path / "x.estrec-collection.json", labels=["a", "b", "c"])
