"""The R client (r/, package dfeols) against a running session: it attaches, creates
variables, fits, refits and exports through the same request files as the Python client,
and gets each model back as an estrec record that tidy()/glance()/modelsummary read. Skipped
without Rscript or without the estrec R package source (estimation/r) to install from."""

import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import dfeols
from dfeols.session import Session
from simdata import sim_panel

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
ESTREC_R = Path(os.environ.get("ESTREC_R_SOURCE", REPO.parent / "estimation" / "r"))

pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None or not (ESTREC_R / "DESCRIPTION").exists(),
    reason="needs Rscript and the estrec R package source (ESTREC_R_SOURCE)",
)


@pytest.fixture(scope="module")
def rlib(tmp_path_factory):
    """estrec and dfeols (R) installed into a temporary library, as R CMD check does."""
    lib = tmp_path_factory.mktemp("rlib")
    for src in (ESTREC_R, REPO / "r"):
        r = subprocess.run(
            ["R", "CMD", "INSTALL", "--no-test-load", "-l", str(lib), str(src)],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            pytest.skip(f"R CMD INSTALL {src.name} failed: {r.stderr[-500:]}")
    return lib


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    data = sim_panel().drop_nulls().filter(pl.col("eng").is_finite())
    return dfeols.prep(
        data,
        shard_dir=tmp_path_factory.mktemp("shards"),
        partition="id_a",
        fixed_effects=["spell", "week"],
        clusters=["id_a", "week"],
        target_rows=3000,
    )


def test_r_client(ds, rlib, tmp_path):
    r = {"root": tmp_path / "s", "demeaned": tmp_path / "d"}
    s = Session.start(
        "rs",
        ds.root,
        "spell + week",
        ["eng", "posts", "feat"],
        cluster="id_a, week",
        poll=0.05,
        progress=False,
        **r,
    )
    try:
        out, table = tmp_path / "r.json", tmp_path / "tables" / "from_r.md"
        env = {**os.environ, "R_LIBS": str(rlib), "DFEOLS_SESSIONS": str(r["root"])}
        script = REPO / "r" / "inst" / "tests" / "session_client.R"
        p = subprocess.run(
            ["Rscript", str(script), "rs", str(out), str(table)],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert p.returncode == 0, p.stderr[-3000:]
        got = json.loads(out.read_text())
        py = s.fit("eng ~ posts + fp", cluster="id_a")  # the same models from Python
        py_iv = s.fit("eng ~ f2 | posts ~ feat + p2", estimator="liml", cluster="id_a")
        py_tw = s.fit("eng ~ posts + i(tbin)", cluster="id_a+week")
        models = [m["label"] for m in s._manager.models]
    finally:
        s.stop()
    td = {row["term"]: row for row in got["tidy"]}
    np.testing.assert_allclose([td[t]["estimate"] for t in py.coefnames], py.coef(), rtol=1e-12)
    np.testing.assert_allclose(
        [td[t]["std.error"] for t in py.coefnames], np.sqrt(np.diag(py.vcov(type="CRV1"))), rtol=1e-12
    )
    assert got["nobs"] == py.N
    assert got["labels"] == ["r1", "r1.hetero"]
    assert models[:4] == ["r1", "m2", "r1.hetero", "two"]
    for rows, ref in ((got["iv"], py_iv), (got["twoway"], py_tw)):
        t_ = {row["term"]: row for row in rows}
        np.testing.assert_allclose([t_[t]["estimate"] for t in ref.coefnames], ref.coef(), rtol=1e-12)
        se = np.sqrt(np.diag(ref.vcov(type="CRV1")))
        np.testing.assert_allclose([t_[t]["std.error"] for t in ref.coefnames], se, rtol=1e-12)
    assert got["two"] == ["eng", "f2"]
    assert "posts" in got["modelsummary"] and "fp" in got["modelsummary"]
    assert Path(got["table"]) == table and "From R" in table.read_text()
    assert (table.parent / "from_r.estrec-table.json").exists()
    assert "not declared or created" in got["error"]
    assert {k: v for k, v in got["created"].items() if "::" not in k} == {
        "fp": "feat * posts",
        "f2": "feat^2",
        "p2": "posts^2",
        "p3": "posts^3",
    }
    assert got["declared"] == ["eng", "posts", "feat"]
    ev = [json.loads(l) for l in (r["root"] / "rs" / "events.jsonl").read_text().splitlines()]
    langs = [e["detail"]["language"] for e in ev if e["type"] == "client_connected"]
    assert langs == ["R", "python"] or langs == ["python", "R"]
