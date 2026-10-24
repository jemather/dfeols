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

## Estimation profiles

The `estimation` result format (`docs/design/estimation-format.md`) is meant
for any software. Its schemas live in `src/dfeols/schema/`, listed in
`catalog.json`, and ship with the package. A *profile* adds rules for one kind
of model (`linear-fe/0.1`, `iv/0.1`) or one tool (`dfeols/0.1`).

To add a profile, open a pull request with:

- `profiles/<name>-<version>.json`: a JSON Schema (Draft 2020-12) whose
  `title` is the profile name (`reghdfe/0.1`) and whose `$id` matches its path
  in this repository. Its rules go in `$defs/rules`, and the file is
  `allOf: [core, #/$defs/rules]`, like the existing profiles.
- At least one example record in `examples/` that claims the profile.
- An entry in `catalog.json`: `name`, `file`, `kind` (`tool` for one
  program's records, `standard` for a model type shared across tools),
  `status: "draft"`, `maintainer` (your GitHub handle), `description`,
  `examples`.

Tool profiles are encouraged to stay close to a standard one:

- Name the profiles you build on in `x-extends` (e.g. `["linear-fe/0.1"]`).
  The validator checks records against them too, and reports "also meets
  linear-fe/0.1" when they do.
- Including the base's rules (`allOf: [{"$ref": ".../linear-fe-0.1.json#/$defs/rules"}]`)
  keeps your records fully conforming, which is the simplest choice.
- Where your tool can't or won't follow the base, list each departure in
  `x-deviations` as `{"path": "metrics.n_singletons", "reason": "..."}`.
  Declared departures are reported as information; undeclared ones as
  warnings.

`pytest tests/test_catalog.py tests/test_schema.py` runs the checks CI
applies: the schema is valid, names and files match, examples validate, every
departure from a base is declared, and published files are unchanged. A
published version never changes; a fix is a new version (`/0.2`). To try a
profile before submitting it, put it in a folder and point
`ESTIMATION_SCHEMA_PATH` (or `dfeols estimation validate --schema-path`) at
it; a local profile can't replace one in the catalog.

## Releasing (maintainers)

1. Update the version in `src/dfeols/__init__.py`, `CITATION.cff` and the
   changelog. To publish a draft profile, set its catalog entry's `status` to
   `published` and add the file's `sha256`.
2. Tag `vX.Y.Z` and push the tag. The release workflow builds the sdist and
   wheel, publishes to TestPyPI, then to PyPI (trusted publishing).

## License

By contributing you agree that your contributions are licensed under the
project's license, GPL-3.0-or-later.
