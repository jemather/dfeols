# Changelog

All notable changes to dfeols are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/) (before 1.0, minor versions may
break the API).

## [Unreleased]

### Added
- `dfeols plan SOURCE --formula ... [--cluster] [--partition K1,K2] [--profile P] [--json]` and
  `dfeols.plan()`: probe a specification before prep (planner step 1, docs/design/planner.md).
  Streaming aggregates on the source report rows and missingness, levels and singletons of
  every FE, nesting of each FE and the cluster in each candidate key (default: every FE,
  interaction component and the cluster), the largest groups per key, and for each key the
  solver path fit would take (or the unbuilt plan B / C it needs), shard count and rough
  worker and coordinator memory, checked against a profile; keys are ranked and one
  recommended. 32 s and 7.5 GB peak on DB1B's 683M-row market table.
- Plan B: crossing FEs on the sparse path with several FEs nested in the partition. The
  dummy block's projection on the local FEs is a Schur complement solved per connected
  component (pseudo-inverse, so redundant local FEs are fine); round 2 demeans the dummy
  terms with all local FEs. `crossing_fe="auto"` takes it when a crossing FE exceeds
  `max_dummy_levels` (below that, several local FEs keep dense dummies); `"sparse"` forces it.
  Equal to dense dummies and pyfixest (OLS, 2SLS, LIML; iid, hetero, CRV1).
- Plan C: `crossing_fe="iterate"`, crossing-FE coefficients by Jacobi-preconditioned conjugate
  gradient across shards (one map/reduce round per step, levels x columns blocks only;
  `shard_matvec`), then OLS by Frisch-Waugh-Lovell and the sparse path's round 2. Works with
  any number of local FEs, including none. `"auto"` takes it above 30,000 crossing-FE dummies
  or for a crossing FE above `max_dummy_levels` with no FE nested in the partition. Options
  `iter_tol`, `iter_maxiter`; `Result.options["crossing_cg"]` has the steps. Workers keep shard
  codes between steps (`DFEOLS_ITER_CACHE_GB`). Not for IV yet.
- `DaskExecutor` scatters large arrays that every task takes (once per round, broadcast)
  instead of sending them with each task.
- `dfeols config init --from-slurm [--profile NAME] [--write] [--sinfo F --scontrol F]`: draft a
  profile's node classes from `sinfo` and `scontrol show partition` (on a login node, or saved
  output). Partitions over the same nodes, limits and policy collapse to one class; each
  records partition, total nodes, nodes in service, cores, memory (MaxMemPerNode or the
  smallest node), interconnect and CPU features, and its walltime when it differs from the
  default partition's. GPU, node-minimum, short-limit, down or hidden partitions, and mixed
  partitions that only span other classes, get `use = false` with the reason. Prints by
  default; `--write` appends to the config and never replaces a profile. Profiles now load
  `max_walltime`, `billing`, `scratch_purge_days` and `nodes`.
- Instrumental variables: `fit(ds, "y ~ x | fe | d1 + d2 ~ z1 + z2")` (fixest syntax) with
  `estimator="2sls"` (default) or `"liml"`, and `fuller=` for Fuller's modified LIML; any number
  of endogenous regressors; iid, hetero and CRV1 standard errors on every FE path (local,
  dense and sparse crossing FEs), from the same two passes over the shards. First-stage
  statistics per endogenous regressor (`Result.iv["first_stage"]`): coefficients on the
  instruments with their vcovs, R², partial R², the iid partial F, the Wald F under each vcov
  and, with one endogenous regressor, the Montiel Olea-Pflueger effective F. Per outcome,
  Sargan's overidentification test and, for LIML, the Anderson-Rubin tests
  (`Result.outcomes[y]["overid"]`). Records claim `iv/0.1`. Tested against pyfixest, fixest,
  linearmodels and a direct LIML computation (docs/design/iv.md).
- dfeols writes its own `estimation` records: `fit(..., record=True)` (the default) writes
  `<workdir>/result.estimation.json` (one record per response) with provenance (software and
  git commit, environment, invocation, SLURM job fields, Dask workers), the sample by drop
  reason, canonical term names, each vcov's small-sample correction and the `dfeols/0.1`
  extension; `publication=` and `redact=` options. `Result.to_estimation(y)` and
  `Result.write_estimation(path)` build them later. Every record is validated against
  estimation-format's schemas in `tests/test_record.py`.
- `prep` records the source's sha256 (per file, plus a manifest hash for several files) or, for
  an in-memory frame, a content hash, and its row count, in `dfeols_meta.json` (`checksum=False`
  skips it). Records carry it, and `extensions.dfeols.row_key` identifies the exact sample with
  a partition-independent hash of the rows used (`dfeols.rowhash`).
- dfeols depends on [estimation-format](https://github.com/estimation-format/estimation) for
  reading, writing, validating and redacting `estimation` records and for the format's schemas
  and profile catalog (including `dfeols/0.1`), pinned to a commit until it is on PyPI;
  `dfeols estimation {validate,redact,digest}` runs its command line. `dfeols.estimation`
  re-exports it, deprecated. `redact=True` also removes the dfeols block's identifying fields
  from records that don't claim `dfeols/0.1` (fits without FEs).

### Changed
- Crossing FEs above `max_dummy_levels` are fitted when the spec has exactly one FE nested in
  the partition (the sparse path); `max_dummy_levels` now limits only dense dummy columns.
  Above 2,000 levels shards return their Gram matrix sparse, and collinearity detection is
  blocked (same columns kept; a fit with 3,000 dummies went from 26 s to 2 s). Coordinator
  memory is logged above 4 GB (about 11 GB at 15,000 levels).
- Singletons are dropped over all fixed effects, iterating across shards, as fixest and reghdfe
  do (previously only in the FE local to the partition, so N could depend on the partition).

### Fixed
- Folder locks: two jobs that found the same stale lock could both take it, the second
  deleting the first's fresh lock (seen on OSCER, where the second job then crashed). A
  takeover now sets the stale lock aside and deletes it only if it is the file it judged
  stale, and new locks are hard-linked into place fully written, so a fresh lock is never
  read as empty.
- `dfeols submit SCRIPT.py --profile P -- ARGS` passed `--profile P` to the script instead of
  using it; options after the script and before `--` now go to `submit`.
- The sparse path crashed with two or more crossing FEs (offsets applied twice).
- `executor(profile, **overrides)` ignored the overrides when given a Profile.

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
- `fit` reads the levels of `i()` variables with executor tasks and caches
  them in `<shard_dir>/_levels/`; it read the column from every shard on
  the coordinator, 55 minutes before round 1 at 1e10 rows.
- The prep scan runs over batches of files (about 4 tasks per worker); each
  task reduces its level sets and nesting checks before returning, so the
  coordinator receives one copy per batch instead of one per file. At 1e10
  rows the per-file results made the by-unit scan coordinator-bound
  (50 minutes, against 16 for the by-group scan).
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
  counted for its input; on a mismatch it logs which shards or buckets carry
  the difference, moves its files to `<shard_dir>/_rejected/` (the first 5
  attempts) and reruns, up to 3 attempts, and otherwise fails naming the task.
  At 1e10 rows on OSCER, 4 of 1,000 encode attempts wrote 200k–800k extra
  rows (whole multiples of one bucket's share) and were correct when rerun.
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
