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

## Documentation

The docs site (https://jemather.github.io/dfeols/) is built with MkDocs Material from
`docs/` and `mkdocs.yml`, and deployed to GitHub Pages from `main` by
`.github/workflows/docs.yml`. Build it locally with

```bash
pip install -e ".[docs]"
mkdocs serve                 # live preview at http://127.0.0.1:8000
mkdocs build --strict        # what CI runs: broken links and anchors fail it
```

User-facing explanations go in `docs/` (the README is a short landing page that links
there). The API reference is generated from docstrings: public functions and classes use
NumPy style (`Parameters`, `Returns`, ...), and every parameter is described.

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

## Estimation profiles

The `estimation` result format, its schemas and profile catalog, and the
`estrec` validator dfeols uses live in their own repository,
https://github.com/estrec/estrec. Propose new profiles (and
changes to `dfeols/0.1`) there; its CONTRIBUTING.md explains how. To try a
draft profile against dfeols records, point `ESTREC_SCHEMA_PATH` (or
`dfeols estrec validate --schema-path`) at its folder.

## Releasing (maintainers)

1. Update the version in `src/dfeols/__init__.py`, `CITATION.cff` and the
   changelog. PyPI rejects dependencies given as git URLs, so every
   dependency, `estrec` included, must be required by version. If dfeols
   needs an unreleased estrec change, release estrec first.
2. Tag `vX.Y.Z` and push the tag. The release workflow builds the sdist and
   wheel, publishes to TestPyPI, then to PyPI (trusted publishing).

## License

By contributing you agree that your contributions are licensed under the
project's license, GPL-3.0-or-later.
