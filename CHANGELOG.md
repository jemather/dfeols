# Changelog

All notable changes to dfeols are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/) (before 1.0, minor versions may
break the API).

## [Unreleased]

## [0.7.0] - 2026-10-22

First public release.

### Added
- `dfeols --version`.
- CI on GitHub Actions: Linux and macOS, Python 3.10–3.14, a Dask job, a
  minimum-versions job and a nightly job against the newest polars and
  pyfixest.

### Changed
- Source moved to a `src/` layout; `examples/` and `tests/` are not installed.
- Licensed under GPL-3.0-or-later.
- Dependency ranges are now tested at both ends: `polars>=1.37,<3`,
  `pyfixest>=0.50.1,<1`, `numpy>=1.25`; Python 3.10–3.13.
- Dense FE codes are `int64` (were `uint64`), which fixes `np.bincount`
  failures under NumPy 1.x.
- pyfixest comparisons (tests, `dfeols.bench` baseline) use the typed
  `demeaner=MapDemeaner(...)` argument on pyfixest ≥ 0.60 and fall back to
  `fixef_tol`/`fixef_maxiter` on older releases
  (`dfeols.formula.pyfixest_tol_kwargs`).
- pyfixest's compiled demeaner (a private API) is imported defensively; if
  its location or signature changes, dfeols logs a warning and uses the NumPy
  fallback. A test fails if the compiled path is not found, so the nightly CI
  job catches upstream changes.
- The ETA bootstrap no longer shares one random generator across calls.
- Code formatted and linted with ruff.

### Fixed
- Distributed prep could count (and keep) rows twice when a task ran twice
  at once, e.g. after the scheduler gave up on a busy worker that kept
  writing ("prep: encoded 2,000,400,000 rows, expected 2,000,000,000"). Each
  task attempt now writes uniquely named files; the coordinator keeps only
  the files listed by the attempt it accepted, removes the rest, and passes
  explicit file lists to the split and merge steps. Merge inputs are deleted
  by the coordinator after the merge is accepted.
- A `shard_dir` left by an interrupted prep (no `dfeols_meta.json`) is
  cleared before prepping again, so its stale pieces cannot join the new
  dataset.
- The ETA no longer crashes with `OverflowError` when no workers are
  connected (an infinite ETA is reported as unknown).
- Range-based shard assignment (more than `range_groups` partition groups,
  new in 0.6.0) computed bins from polars' `UInt32` row counts, which wrap
  at 2**32: at 1e9 rows, 10M units and 500 bins it produced 4 shards, one
  with 994M rows. Row counts are now Int64 throughout prep, and prep refuses
  to write a shard larger than twice the target plus the largest partition
  group. **Shard sets prepared with range assignment by 0.6.0–0.6.1 should
  be re-prepped** (`"shard_assignment": "range"` in `dfeols_meta.json`).
- `fit` checks the largest shard's memory estimate against the profile's
  worker memory: it warns above it and refuses above 1.5×, instead of
  letting the scheduler kill worker after worker on the same shard.
- prep warns when one partition group is more than 4× the shard target.

### Removed
- `dfeols.slurm` (deprecated shim). Use `dfeols.executor(profile)` with a
  site profile (`dfeols config init`).
- `benchmarks/bench.py` stub; use `python -m dfeols.bench`.

## [0.6.1]

### Added
- `dfeols bench`: submits a benchmark as a coordinator job with one
  `--profile`; the job's profile is passed through to the benchmark
  (`$DFEOLS_PROFILE`).

### Changed
- `dfeols submit` and `dfeols bench` print the profile and module lines used.
- `dfeols config init --force` backs up the existing config first.
- OSCER example coordinator raised to 16 cores / 64 GB (data generation runs
  on the coordinator).

## [0.6.0]

### Added
- Hash ids for nested FEs with more than 2e6 levels (collisions checked
  exactly, with a fallback to dense codes) and range-based shard assignment
  for more than 5e6 partition groups, for 1e10-row datasets.
- Two-level shuffle in distributed prep when there would be more than 5e4
  piece files.
- Stress presets `units100m_t100` (1e10 rows) and `skewed_groups`.
- Round-2 ETA prior.

### Fixed
- Nested-in-cluster detection with signed ids.
- Worker memory estimate.

## [0.5.1]

### Changed
- Flat piece files; shards are merged only when fragmented.
- Adaptive task size in distributed prep.
- The coordinator runs read-only scan tasks itself while no workers are
  connected.

## [0.5.0]

### Added
- Distributed prep on workers (scan → reduce → encode → merge/split).
- Polars spill directory and memory pinning to the job allocation.
- Collision-free data folders.

## [0.4.0]

### Added
- Sparse crossing FEs: with one local FE, crossing-FE blocks of X'WX come
  from per-level sums and no dummy columns are built.
- Billion-row benchmark presets.
- Lazy worker scale-down (`idle_minutes`).
- Append-only results layout and `dfeols results`.

## [0.3.0]

### Added
- General HDFE with a generic unit × time panel DGP.
- Scheduler-agnostic TOML site profiles (Slurm, PBS, SGE, LSF, local).
- Exact two-way demeaning (Schur complement per connected component).

## [0.2.0]

### Added
- Synthetic DGP with known parameters, validity suite, benchmark runner.

## [0.1.0]

### Added
- Two-round map/reduce fit, Serial and Dask executors, ETA, agreement with
  pyfixest.

[Unreleased]: https://github.com/jemather/dfeols/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/jemather/dfeols/releases/tag/v0.7.0
