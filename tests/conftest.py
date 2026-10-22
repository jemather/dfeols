"""Shared fixtures and options for the synthetic-data suite.

    pytest tests/                                    # quick: ~0.5M rows
    pytest tests/ --rows 2e7 --data-dir /scratch/$USER/dfeols-synth
    pytest tests/ -m slow --mc-reps 200              # Monte Carlo coverage
    pytest tests/ --executor dask:4                  # run fits on a local Dask cluster
"""

import os
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore", category=UserWarning)


def pytest_addoption(parser):
    g = parser.getgroup("dfeols")
    g.addoption("--rows", type=float, default=float(os.environ.get("DFEOLS_TEST_ROWS", 5e5)),
                help="rows in the synthetic validation panel")
    g.addoption("--data-dir", default=os.environ.get("DFEOLS_TEST_DATA"),
                help="reuse generated data here (default: a pytest temp dir)")
    g.addoption("--shards", type=int, default=12, help="approximate number of shards")
    g.addoption("--executor", default="serial", help="serial | dask:N")
    g.addoption("--mc-reps", type=int, default=100, help="Monte Carlo replications")
    g.addoption("--pyfixest-rows", type=float, default=2e5,
                help="rows (whole accounts) used for the in-memory pyfixest comparison")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Monte Carlo and other long tests (run with -m slow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="slow: run with -m slow")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)


@pytest.fixture(scope="session")
def executor(request):
    spec = request.config.getoption("--executor")
    if spec == "serial":
        import dfeols
        yield dfeols.SerialExecutor()
        return
    if spec.startswith("dask"):
        import distributed
        import dfeols
        n = int(spec.split(":")[1]) if ":" in spec else 2
        with distributed.LocalCluster(n_workers=n, threads_per_worker=1,
                                      dashboard_address=None) as lc, distributed.Client(lc) as c:
            yield dfeols.DaskExecutor(c, max_workers=n, tick=1)
        return
    raise ValueError(spec)


@pytest.fixture(scope="session")
def synth(request, tmp_path_factory):
    """Generate (or reuse) the validation panel and prep it into shards."""
    import dfeols
    from dfeols.synth import generate, load_truth, plan_for_rows

    rows = request.config.getoption("--rows")
    base = request.config.getoption("--data-dir")
    base = Path(base) if base else tmp_path_factory.mktemp("synth")
    raw = base / f"raw_{int(rows)}"
    # small chunks so even small panels span several files
    # smaller audiences than the default DGP, so the test panel has many
    # accounts (clusters) even at modest sizes: ~600 rows per account
    d = plan_for_rows(rows, "influencer", spells_logmean=2.5,
                      accounts_per_chunk=max(50, int(rows / 600 / 8)))
    generate(raw, d, jobs=min(4, os.cpu_count() or 1), verbose=False)
    truth = load_truth(raw)

    n_shards = request.config.getoption("--shards")
    shards = base / f"shards_{int(rows)}_{n_shards}"
    if (shards / "dfeols_meta.json").exists():
        ds = dfeols.Dataset.open(shards)
    else:
        import polars as pl
        n = pl.scan_parquet(str(raw / "*.parquet")).select(pl.len()).collect().item()
        ds = dfeols.prep(str(raw / "*.parquet"), shard_dir=shards, partition="id_a",
                         fixed_effects=["spell", "week", "id_a^week"], clusters=["id_a", "id_f"],
                         target_rows=max(n // n_shards, 1000), overwrite=True)
    return {"raw": raw, "ds": ds, "truth": truth, "base": base, "dgp": d}
