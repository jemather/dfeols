"""The estimation format's catalog (src/dfeols/schema/catalog.json): the checks a pull request
that adds or changes a profile must pass. See CONTRIBUTING.md, "Estimation profiles"."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from dfeols import estimation as est

ROOT = est.PACKAGED_SCHEMAS
CATALOG = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))
ENTRIES = [CATALOG["core"], *CATALOG["profiles"]]
PROFILES = {e["name"]: e for e in CATALOG["profiles"]}
NAME = re.compile(r"^[a-z0-9][a-z0-9-]*/\d+\.\d+$")
ID_BASE = "https://raw.githubusercontent.com/jemather/dfeols/main/src/dfeols/schema/"


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def ids(entries):
    return [e["name"] for e in entries]


def test_license():
    assert CATALOG["license"] == "CC0-1.0"
    assert "CC0" in (ROOT / "LICENSE").read_text(encoding="utf-8")


def test_entries_are_complete_and_unique():
    names = ids(ENTRIES)
    assert len(names) == len(set(names)), "duplicate names"
    for e in ENTRIES:
        assert NAME.match(e["name"]), f"{e['name']}: use <name>/<major>.<minor>"
        assert e["status"] in ("draft", "published"), e["name"]
        assert e.get("maintainer"), f"{e['name']}: maintainer (GitHub handle) required"
        assert e.get("examples"), f"{e['name']}: at least one example record"
    for e in CATALOG["profiles"]:
        assert e["kind"] in ("standard", "tool"), e["name"]
        assert e.get("description"), e["name"]


def test_every_schema_file_is_listed():
    listed = {e["file"] for e in ENTRIES}
    on_disk = {"estimation-0.1.json", *(f"profiles/{p.name}" for p in (ROOT / "profiles").glob("*.json"))}
    assert on_disk == listed, "files in profiles/ and catalog.json entries must match"
    examples = {x for e in ENTRIES for x in e["examples"]}
    assert examples <= {f"examples/{p.name}" for p in (ROOT / "examples").glob("*.json")}


@pytest.mark.parametrize("entry", ENTRIES, ids=ids(ENTRIES))
def test_schema_file(entry):
    jsonschema = pytest.importorskip("jsonschema")
    doc = load(entry["file"])
    jsonschema.Draft202012Validator.check_schema(doc)
    assert doc["title"] == entry["name"]
    assert doc.get("x-license") == "CC0-1.0", "schemas in the catalog are CC0 (see LICENSE)"
    assert doc["$id"] == ID_BASE + entry["file"], "$id must match the file's place in the repository"
    if entry is not CATALOG["core"]:
        assert "rules" in doc.get("$defs", {}), "a profile keeps its own rules in $defs/rules"


@pytest.mark.parametrize("entry", CATALOG["profiles"], ids=ids(CATALOG["profiles"]))
def test_extends_and_deviations_are_well_formed(entry):
    doc = load(entry["file"])
    for base in doc.get("x-extends", []):
        assert base in PROFILES and base != entry["name"], f"x-extends: unknown profile {base!r}"
    for d in doc.get("x-deviations", []):
        assert d.get("path") and d.get("reason"), "each deviation needs a path and a reason"
    if entry["kind"] == "standard":
        assert not doc.get("x-deviations"), "standard profiles don't deviate from each other"


@pytest.mark.parametrize("entry", ENTRIES, ids=ids(ENTRIES))
def test_examples_validate_and_departures_are_declared(entry):
    """Each example is valid under its profile, and wherever it fails a base the profile
    extends, the profile declares that deviation (the validator would warn otherwise)."""
    pytest.importorskip("jsonschema")
    sc = est.load_schemas()
    for rel in entry["examples"]:
        rec = est.read(ROOT / rel)
        name = entry["name"]
        if entry is not CATALOG["core"]:
            assert name in rec["meta"].get("profiles", []), f"{rel} must claim {name}"
        issues = est.validate(rec, schema=sc)
        bad = [str(i) for i in issues if i.level == "error" or "without a declaration" in i.message]
        assert bad == [], rel


@pytest.mark.parametrize("entry", ENTRIES, ids=ids(ENTRIES))
def test_published_files_never_change(entry):
    """A published version is fixed: its file must still hash to the sha256 recorded when it
    was published. A fix is a new version (`/0.2`), not an edit."""
    if entry["status"] != "published":
        assert "sha256" not in entry, f"{entry['name']}: sha256 is recorded when publishing"
        return
    digest = hashlib.sha256((ROOT / entry["file"]).read_bytes()).hexdigest()
    assert digest == entry.get("sha256"), f"{entry['name']} is published and its file changed"


def test_package_includes_the_schemas():
    """pyproject.toml's package-data must cover every catalog file."""
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    for pattern in ("schema/*.json", "schema/profiles/*.json", "schema/examples/*.json", "schema/LICENSE"):
        assert f'"{pattern}"' in text
    assert '"src/dfeols/schema/LICENSE"' in text  # license-files
