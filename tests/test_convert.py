"""dfeols convert-dta: Parquet parts equal the .dta, with one schema across chunks."""

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dfeols.cli import main as cli_main
from dfeols.convert import convert_dta


@pytest.fixture
def dta(tmp_path):
    rng = np.random.default_rng(0)
    n = 10_007
    unf = pd.array(rng.integers(0, 2, n), dtype="Int8")
    unf[:50] = pd.NA  # missing only in the first chunk: pandas upcasts that chunk to float
    df = pd.DataFrame(
        {
            "id_f": np.arange(n, dtype=np.int32),
            "week": pd.Timestamp("2010-01-04") + pd.to_timedelta(7 * rng.integers(0, 156, n), unit="D"),
            "unf": unf,
            "eng": rng.normal(size=n).astype(np.float32),
            "feat": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)),
            "grp": rng.choice(["a", "bb", ""], n),
        }
    )
    p = tmp_path / "panel.dta"
    df.to_stata(
        p,
        write_index=False,
        version=118,
        convert_dates={"week": "tw"},
        variable_labels={"eng": "engagement"},
    )
    return p, df


def test_parts_equal_source(dta, tmp_path):
    p, df = dta
    out = tmp_path / "pq"
    m = convert_dta(p, out, chunk_rows=1000, part_gb=20e3 / 2**30)  # ~20 KB parts
    assert m["rows"] == len(df) and len(m["parts"]) > 1
    assert sum(x["rows"] for x in m["parts"]) == len(df)
    got = pl.read_parquet(out / "part-*.parquet")
    assert got.schema["unf"] == pl.Int8 and got.schema["week"].is_numeric()  # %tw code, not a date
    assert got.schema["eng"] == pl.Float32 and got.schema["id_f"] == pl.Int32
    assert got["unf"].null_count() == 50 and m["nulls"]["unf"] == 50
    assert got["feat"].null_count() == int(df["feat"].isna().sum())
    assert (got["feat"].is_nan().sum() or 0) == 0  # Stata missing is null, not NaN
    np.testing.assert_array_equal(got["id_f"].to_numpy(), df["id_f"].to_numpy())
    assert got["week"].n_unique() == df["week"].nunique()
    np.testing.assert_allclose(got["eng"].to_numpy(), df["eng"].to_numpy())
    np.testing.assert_array_equal(got["unf"].to_numpy()[50:], df["unf"].to_numpy()[50:].astype(int))
    meta = json.loads((out / "_dta_meta.json").read_text())
    cols = {c["name"]: c for c in meta["columns"]}
    assert cols["week"]["format"] == "%tw" and cols["eng"]["label"] == "engagement"
    assert not list(out.glob("*.tmp"))


def test_columns_and_overwrite(dta, tmp_path):
    p, df = dta
    out = tmp_path / "pq"
    convert_dta(p, out, columns=["id_f", "eng"])
    assert pl.read_parquet(out / "part-*.parquet").columns == ["id_f", "eng"]
    with pytest.raises(FileExistsError):
        convert_dta(p, out)
    with pytest.raises(KeyError):
        convert_dta(p, tmp_path / "x", columns=["nope"])


def test_cli(dta, tmp_path, capsys):
    p, df = dta
    cli_main(["convert-dta", str(p), str(tmp_path / "c"), "--chunk-rows", "3000"])
    assert f"{len(df):,} rows" in capsys.readouterr().out
