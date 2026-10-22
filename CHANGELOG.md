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
  `pyfixest>=0.50.1,<1`, `numpy>=1.25`; Python 3.10–3.14.
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
These include the fixes from the 0.6.2 snapshot (below), merged into the new
layout, plus:
- The coordinator merges per-file scan results as they arrive instead of
  holding them until the scan ends. At 1e10 rows it kept 5,000 copies of the
  group × time level set and its nesting-check tables (well over 100 GB) and
  was killed; merged, these stay near the number of distinct keys.
- `fit` checks the largest shard's memory estimate against the profile's
  worker memory: it warns above it and refuses above 1.5×, instead of
  letting the scheduler kill worker after worker on the same shard.
- prep refuses to write a shard larger than twice the target plus the
  largest partition group (only a bug can produce one), and warns when one
  partition group is more than 4× the target.
- A `shard_dir` left by an interrupted prep (no `dfeols_meta.json`) is
  cleared before prepping again, so its stale pieces cannot join the new
  dataset.
- Prep stages are no longer given an ETA projected from the scan rate (the
  split stage showed "8866M rows/s, ETA 0:00:01" while its 200M-row tasks ran
  for ~20 minutes); the round-1-to-round-2 prior now applies to fits only.
- The ETA no longer crashes with `OverflowError` when no workers are
  connected (an infinite ETA is reported as unknown).
- Prep tasks counted their output by listing the folder they had written
  into. On Lustre, listing a folder that 40 workers are writing into can
  return an entry twice, so a bucket file was counted twice per repeated
  entry: "encoded 10,002,000,000 rows, expected 10,000,000,000" at 1e10, with
  every file on disk correct (10 entries of 200,000 rows; the earlier +2.8M
  and +400k were the same). Tasks now count exactly the files they handed to
  Polars, and the input file list is deduplicated.
- Every encode and split task checks that it wrote exactly the rows the scan
  counted for its input; on a mismatch it deletes its own files and reruns
  (up to 3 attempts), and otherwise fails naming the task and keeping its
  files for inspection.
- Dask task keys include a per-run id; every prep used the same keys
  (`dfeols-encode-prep-17`), so a second prep in the same job could be handed
  a result Dask still held for the first.
- Folder locks also take over immediately when their holder ran on this
  host and is gone, and a release never removes another holder's lock.

### Removed
- `dfeols.slurm` (deprecated shim). Use `dfeols.executor(profile)` with a
  site profile (`dfeols config init`).
- `benchmarks/bench.py` stub; use `python -m dfeols.bench`.

## [0.6.2]

Fixes from the first 1e10 attempts on OSCER (a snapshot, not published).

### Fixed
- Range-based shard assignment (more than `range_groups` partition groups,
  new in 0.6.0) used polars' `UInt32` row counts, which wrap at 2**32: at
  2e9 rows it produced 2 shards instead of 1,000 (at 1e9 rows with 10M units,
  4 shards). Counts are Int64 and the bins and boundaries are checked to be
  monotone. **Shard sets prepared with range assignment before 0.6.2 must be
  rebuilt**; `dfeols bench` detects them (no `prep_version` in their
  metadata) and re-preps them.
- A task that ran twice at once (a rerun overlapping its first attempt, or
  another job writing the same folder) could be counted twice ("encoded
  2,000,400,000 rows, expected 2,000,000,000"). Every task attempt writes
  uniquely named files and reports them; the coordinator keeps one attempt
  per task, ignores duplicate results, sweeps the other files, and gives the
  split and merge steps explicit file lists. Merge inputs are deleted by the
  coordinator once the merge is accepted.
- Two jobs could write the same folder. Shard folders (during prep) and
  benchmark data sets (during generation and prep) are locked with a sibling
  `<folder>.lock` (O_EXCL, refreshed every 30 s, stale after 5 min); a
  second job fails at once with `dfeols.LockedError` naming the holder.
- Overwrites rename the old folder before deleting it, so a straggling
  writer cannot make the delete fail ("Directory not empty: '_buckets'").

### Added
- `prep_version` in `dfeols_meta.json`.

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
