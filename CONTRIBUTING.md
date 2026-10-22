# Contributing to dfeols

Thanks for your interest. Bug reports, site profiles for new clusters, and
pull requests are all welcome.

## Reporting a problem

Open an issue with:

- `dfeols --version`, `python --version`, and the versions of polars,
  pyfixest and (if used) dask / dask-jobqueue;
- the scheduler (Slurm, PBS, …) and the output of `dfeols config show`
  with account names removed;
- for wrong results: the formula, `dfeols info <shard_dir>`, and if possible
  a small synthetic reproduction (`python -m dfeols.synth`);
- for crashes on a cluster: the coordinator log and one worker log.

## Development setup

```bash
git clone https://github.com/jemather/dfeols.git
cd dfeols
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # optional: runs ruff on each commit
```

## Checks

```bash
ruff check . && ruff format --check .
pytest                       # ~2 minutes on 2 cores, 0.5M-row synthetic panels
pytest --executor dask:2     # the same fits on a local Dask cluster
pytest -m slow               # Monte Carlo coverage (long)
mypy                         # advisory for now
```

Larger test data: `pytest --rows 5e6 --data-dir /scratch/$USER/dfeols-tests`.

## What a change needs

- **Correctness first.** Anything that touches estimation must keep agreement
  with pyfixest (`tests/test_vs_pyfixest.py`) and recovery on the synthetic
  DGPs (`tests/test_panel.py`, `tests/test_influencer.py`). New model
  features need a pyfixest/fixest comparison test.
- **Invariance.** Results must not depend on the partition key, the number
  of shards, or the executor; add to the invariance tests where relevant.
- **No site-specific code.** Queues, modules, paths and accounts belong in
  site profiles (`examples/profiles/`), never in the package.
- Update `CHANGELOG.md` under *Unreleased*.

## New site profiles

If you run dfeols on a new cluster, a tested profile in
`examples/profiles/` (with the site name, scheduler and date tested in a
comment) is a very useful contribution. PBS, SGE and LSF are currently
untested on real systems.

## Releasing (maintainers)

1. Update the version in `src/dfeols/__init__.py`, `CITATION.cff` and the
   changelog.
2. Tag `vX.Y.Z` and push the tag. The release workflow builds the sdist and
   wheel, publishes to TestPyPI, then to PyPI (trusted publishing).

## License

By contributing you agree that your contributions are licensed under the
project's license, GPL-3.0-or-later.
