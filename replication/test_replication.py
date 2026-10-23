# SPDX-License-Identifier: GPL-3.0-or-later
"""Published-data replications as a pytest suite. Skipped unless the data is built.

    export DFEOLS_PARSONS_DIR=/scratch/$USER/parsons2011     # output of `harness.py build`
    pytest replication/ -q                                     # serial, all specs
    DFEOLS_REPLICATION_EXECUTOR=dask:4 pytest replication/ -q
"""

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "parsons2011"))

PARSONS = os.environ.get("DFEOLS_PARSONS_DIR")


@pytest.fixture(scope="module")
def parsons():
    if not PARSONS or not (Path(PARSONS) / "analysis.parquet").exists():
        pytest.skip("set DFEOLS_PARSONS_DIR to a directory prepared with `harness.py build`")
    import harness

    harness.run(Path(PARSONS), executor=os.environ.get("DFEOLS_REPLICATION_EXECUTOR", "serial"))
    import pandas as pd

    return pd.read_csv(Path(PARSONS) / "comparison.csv")


def test_parsons_matches_pyfixest(parsons):
    bad = parsons[parsons["status"].eq("FAIL") & parsons["detail"].str.contains("pyfixest|N ", na=False)]
    assert bad.empty, bad[["run", "vcov", "term", "detail"]].to_string()


def test_parsons_partition_invariance(parsons):
    assert parsons["inv_b_rel"].notna().any(), "no spec ran on two partitions"
    bad = parsons[parsons["detail"].str.contains("across partitions", na=False)]
    assert bad.empty, bad[["run", "vcov", "term", "inv_b_rel", "inv_se_rel"]].to_string()


def test_parsons_key_terms_present(parsons):
    bad = parsons[parsons["detail"].str.contains("key term missing", na=False)]
    assert bad.empty, bad[["run", "term"]].to_string()
