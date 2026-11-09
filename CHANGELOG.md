# Changelog

All notable changes to dfeols are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/) (before 1.0, minor versions may
break the API).

## [Unreleased]

### Fixed

- Plan C in level space no longer runs workers out of memory. Shards cap their part of A by
  the worker's memory, not the coordinator's, so four parts fit in one combine. Combines add
  their inputs one at a time, and keep only the diagonal blocks (then nothing) before passing
  60% of the worker's memory, so the fit falls back to a pass per CG step instead of losing
  workers. A 162M-row fit with 51,844 crossing levels killed 24 GB workers in round 1.
- Round 2 without a cache no longer rebuilds each shard's part of A when it recomputes
  round 1. The rebuild sat outside the task memory estimate, so several such tasks on one
  worker killed it.
- `dfeols plan` caps A's budget at what the workers can sum (60% of a worker's memory), as
  fit does.
- Arguments shared by every task that are dicts of large arrays (round 2's linear maps, with
  plan C's FE solution) go to the workers once instead of inside every task: a 758,405-level
  fit sent a 46 MiB task graph.

## [0.8.0] - 2026-11-09

Instrumental variables, two-way clustering, crossing FEs of any size, interactive sessions
with an R client, the planner, estrec records and discrete-time hazard models. dfeols is not
on PyPI: it depends on [estrec](https://github.com/estrec/estrec) by a pinned git commit
until estrec is there, so install from the tag
(`pip install "dfeols[dask] @ git+https://github.com/jemather/dfeols@v0.8.0"`).

Results can differ from 0.7.0 on the same data: singletons are now dropped over all FEs
(N no longer depends on the partition), `r2_within` nets y of every FE, and regressors an FE
absorbs are dropped as collinear (see Changed and Fixed).

### Added

#### Estimation
- Instrumental variables: `fit(ds, "y ~ x | fe | d1 + d2 ~ z1 + z2")` (fixest syntax) with
  `estimator="2sls"` (default) or `"liml"`, and `fuller=` for Fuller's modified LIML; any
  number of endogenous regressors; iid, hetero and CRV1 standard errors on every FE path
  (local, dense, sparse and iterated crossing FEs), from the same two passes over the shards.
  First-stage statistics per endogenous regressor (`Result.iv["first_stage"]`): coefficients
  on the instruments with their vcovs, R², partial R², the iid partial F, the Wald F under
  each vcov and, with one endogenous regressor, the Montiel Olea-Pflueger effective F. An
  endogenous regressor the instruments fit exactly (time dummies instrumented by their own
  differences) is marked `exact_fit` and gets no F tests. Per outcome, Sargan's
  overidentification test and, for LIML, the Anderson-Rubin tests
  (`Result.outcomes[y]["overid"]`). Tested against pyfixest, fixest, linearmodels,
  ivreghdfe and a direct LIML computation (docs/design/iv.md).
- Two-way clustering: `fit(..., cluster="a+b")` (or `["a", "b"]`) and in sessions.
  Cameron-Gelbach-Miller (V_a + V_b - V_ab) with pyfixest's and fixest's conventions: each
  term scaled with G the smaller count, which is also the t df. Round 2 sums the scores of
  a, b and their intersection (codes made per shard; summed by code when both clusters cross
  the partition). Equal to pyfixest for each kind of pair (nested in the partition or
  crossing it) on every crossing-FE path, with or without the cache, and for IV. Records list
  both clusters and their counts; the summary shows each count. Not with `cells=True`.
- `dfeols.lpm_dthazard`: discrete-time hazard linear probability models (an absorbing-state
  outcome) by the Farbmacher-Tauchmann (2021) IV, levels instrumented by their differences
  (Δ^j with `diff=`), with the first-difference and pooled OLS estimates and the G matrix
  linking them (docs/design/hazard.md). The sample is built per shard, partitioned by unit,
  and cached as a derived dataset; rows with gaps or missing lags are dropped and counted by
  reason in the result and the record. Extra FEs are absorbed, with a warning and counts where
  that is not equivalent to differencing their dummies (units that move between cells, or
  FEs constant within a unit). `i()` terms: each level's dummy is instrumented by its own
  difference, and levels absent from the sample are reported as `excluded_levels`.
  Reproduces Farbmacher and Tauchmann's Table 3 (Brown and Laschever 2012) to the printed
  digits.
- Plan B: crossing FEs on the sparse path with several FEs nested in the partition. The
  dummy block's projection on the local FEs is a Schur complement solved per connected
  component (pseudo-inverse, so redundant local FEs are fine); round 2 demeans the dummy
  terms with all local FEs. `crossing_fe="auto"` takes it when a crossing FE exceeds
  `max_dummy_levels` (below that, several local FEs keep dense dummies); `"sparse"` forces it.
- Plan C: `crossing_fe="iterate"`, crossing-FE coefficients by preconditioned conjugate
  gradient, then OLS or IV by Frisch-Waugh-Lovell and the sparse path's round 2. Works with
  any number of local FEs, including none; `"auto"` takes it above 30,000 crossing-FE dummies
  or for a crossing FE above `max_dummy_levels` with no FE nested in the partition.
  - In level space (the default): round 1 also returns each shard's part of A = D~'W D~,
    summed on the workers in a tree as parts finish (only the root reaches the coordinator),
    and CG runs on the coordinator with no further passes. The preconditioner follows the
    connected components of each crossing FE's block: exact pinned or pseudo-inverse solves
    (AP-sweep, i.e. symmetric block Gauss-Seidel, when all are exact), AMG V-cycles on large
    sparse components (optional `dfeols[amg]`, pyamg), else Jacobi. DB1B 2005Q2-Q3 (10.3M
    rows): 7 steps in 0.14 s against 16 passes in 23 s.
  - Above `level_space_gb` (default a quarter of the coordinator's memory) each CG step is a
    pass over the shards, summed in a tree on the workers, and dfeols warns that convergence
    may be slow (`crossing.SlowConvergenceWarning`), naming what made A too large. The CG
    state is kept after each step (`cg_checkpoint.npz`), so a fit stopped by walltime or
    `iter_maxiter` resumes where it stopped.
  - Options `iter_tol` (default 1e-8), `iter_maxiter`, `level_space_gb`, `crossing_precond`;
    `Result.options["crossing_cg"]` gives the mode, preconditioner, steps and sizes, and the
    progress line shows the step, the residual and roughly how many steps are left.
- `fit(..., cells=True)`: the fit from per-cell sums (`dfeols.cells`). One pass groups each
  shard's rows by every FE key and the cluster; the coordinator drops singletons on the
  cells, absorbs the FE with most levels by group means and solves the others with plan C's
  level-space CG, then b by Frisch-Waugh-Lovell, RSS and CRV1 scores from the cell moments.
  Exact. OLS; vcov CRV1 (one-way) and iid. DB1B 2005Q2-Q3: 10.3M rows in 329k cells, 9.8 s
  against 27 s (level space) and 43 s (dense solve); b within 1e-11.

#### Interactive sessions
- `dfeols.Session` and `dfeols session start|serve|status|where|fit|refit|add-variable|export|persist|replay|stop`
  (docs/design/sessions.md). A manager demeans the declared variables on the session's FEs
  once (local FEs, dense crossing FEs, or plan C with one CG solve for every declared
  variable), keeps the residualised columns per shard and their Gram matrix, and fits models
  from request files: b and iid SEs with no pass over the data, hetero and CRV1 (one- or
  two-way) with one. OLS and IV (2SLS, LIML, Fuller) share the pass. Results equal
  `dfeols.fit` on the same sample.
- Created variables: `Session.add_variable(name, expression)` and `add_variables({...})`, on
  raw columns of the prepared data (`feat * posts`, `log(wt)`, `posts > 0`,
  `ifelse(x > 0, y, 0)`; R's `^`, `&`, `|`, `!` with R's precedence), parsed by `dfeols.expr`
  through an allowlist of `ast` nodes (no eval). Each is computed on the session's rows,
  demeaned like a declared variable in one batch, kept by digest and reused by later sessions
  on the same demeaned data. `i()` terms in session models become indicator columns created
  the same way.
- `Session.refit(model, vcov=, cluster=)`: an earlier model with another vcov or cluster, one
  pass for the new SEs (none for iid).
- Exports through the estrec table tool: `s.export([m1, m2], "table2.tex")` writes the table
  (.tex, .md, .html, .csv, .txt, .doc), its `estrec-table/0.1` spec and the session
  collection, so it renders again from the files alone. `dfeols session replay` runs a
  session collection's events again and compares each model with its record.
- The R client, package `dfeols` in `r/`: `attach_session`, `fit`, `refit`, `add_variable`,
  `export_table`, ...; it speaks the request files (no network path to the compute node, no
  Python) and returns estrec records, so `tidy()`, `glance()` and modelsummary work. Equal to
  the Python client to 1e-12 against a local session.
- Estimates and progress: the demeaning's estimate from earlier runs and its live progress
  while a client waits; per model, an estimate before it runs (a range until a pass has been
  measured, from this session's first pass and earlier sessions' in `passes.jsonl`) and its
  progress lines, with how many of the requested workers are running yet.
- Demeaned data under `<scratch>/demeaned/<dataset>/<key>/`, reused by a session with the
  same inputs; `Session.persist()` keeps a `--no-persist` session's. `dfeols demeaned list`
  (size, when written, purge date from `scratch_purge_days`, sessions using each key, created
  variables) and `dfeols demeaned rm <key>` (asks, refuses while in use). `session start`
  checks free space on scratch against the size estimate; profile settings
  `demeaned_storage` and `demeaned_confirm_gb` make it ask before a large write.

#### Planning
- `dfeols plan SOURCE --formula ... [--cluster] [--partition K1,K2] [--profile P] [--workers N] [--specs N] [--json]`
  and `dfeols.plan()`: probe a specification before prep (docs/design/planner.md). Streaming
  aggregates on the source report rows and missingness, levels and singletons of every FE,
  nesting of each FE and the cluster in each candidate key, the largest groups, and per key
  the solver path fit would take, shard count, and worker and coordinator memory against a
  profile (on plan C, A's size against the level-space budget). It recommends the key that
  makes the crossing system smallest among keys whose largest group is within `target_rows`
  or half a worker's share of the rows, and `fit(cells=True)` from 2 rows per cell. DB1B's
  683M-row market table probes in 72 s with 8.8 GB.
- Wall-time estimates: every `fit` appends its path, rows, columns, passes, seconds and
  workers to `<scratch>/timings/fits.jsonl` (`dfeols.timings`), and `dfeols plan` gives each
  key a wall-time range from earlier runs on the same path. It recommends demeaning first in
  a session for plan C, for estimates over half the profile's `max_walltime`, or with
  `--specs N` > 1, else one job. `fit` warns when another formula was fitted on the same
  data, FEs and weights before, naming the session command.
- `prep(partition="auto", formula=...)` shards by the key `dfeols plan` recommends and keeps
  the report in `dfeols_meta.json`.
- `dfeols config init --from-slurm [--profile NAME] [--exclude PATTERN] [--write] [--sinfo F --scontrol F]`:
  draft a profile's node classes from `sinfo` and `scontrol show partition`. Partitions over
  the same nodes, limits and policy collapse to one class with its partition, nodes (total
  and in service), cores (capped at MaxCPUsPerNode), memory (MaxMemPerNode or the smallest
  node), interconnect, CPU features and walltime. GPU, node-minimum, short-limit, down,
  hidden, redundant and `--exclude`d partitions get `use = false` with the reason. Prints by
  default; `--write` appends and never replaces a profile. Profiles load `max_walltime`,
  `billing`, `scratch_purge_days` and `nodes`.

#### Records
- Every fit writes an [estrec](https://github.com/estrec/estrec) record
  (`fit(record=True)`, the default): `<workdir>/result.estrec.json`, one per response
  (`result.<y>.estrec.json`), with provenance (software and git commit, environment,
  invocation, scheduler fields, workers), the sample by drop reason, canonical term names,
  each vcov's small-sample correction, IV's first-stage and overidentification tests, and
  the `dfeols/0.1` extension; `publication=` and `redact=`. `Result.write_estimation(path)`
  writes one later. Validated against estrec's schemas in the tests.
- Disclosure levels (estrec spec 4.9): `fit(record_level=, purpose=)`, from the argument,
  else the profile's `record_level`, else `$DFEOLS_RECORD_LEVEL`, else 4: no scheduler,
  timing, hosts or package list unless `record_level = 5`. Purpose profiles such as
  `dcas/0.1`.
- `prep` records the source's sha256 (per file, plus a manifest hash) or an in-memory frame's
  content hash, and its row count (`checksum=False` skips it); records carry it, and
  `extensions.dfeols.row_key` identifies the exact sample with a partition-independent hash
  of the rows used (`dfeols.rowhash`).
- `dfeols.write_collection(fits, path, kind=...)`: several fits as one
  `estrec-collection/0.1` file; sessions write theirs (`kind: "session"`, with events).
- `dfeols estrec {validate,redact,digest}` runs estrec's command line; extra
  `dfeols[estrec]` adds JSON Schema validation.

#### Execution
- Workers run several shards at once when they fit: a worker job gets a Dask thread per core
  and declares its memory as the resource MEMORY, and each fit task asks for its memory
  estimate, so a 24 GB, 4-core worker runs up to 4 tasks together. Prep tasks and plan C's
  round 1 in level space still take a whole worker.
- `DaskExecutor` scatters large arrays that every task takes once per round instead of
  sending them with each task. Executors take an optional `reduce` (`executors.Reduce`).
- `fit` logs which dfeols runs: version, commit (uncommitted changes flagged) and the folder
  it was imported from.

### Changed
- Singletons are dropped over all fixed effects, iterating across shards, as fixest and
  reghdfe do (previously only in the FE local to the partition, so N could depend on the
  partition).
- `r2_within` (and `tss`) take y net of all the FEs, as fixest does, on every path; with a
  crossing FE they were net of the local FEs only. Fits whose FEs all cross the partition
  report `r2_within` instead of `r2`.
- Crossing FEs above `max_dummy_levels` are fitted on the sparse path when the spec has one
  FE nested in the partition; `max_dummy_levels` now limits only dense dummy columns. Above
  2,000 levels shards return their Gram matrix sparse (a fit with 3,000 dummies went from
  26 s to 2 s).
- The NumPy demeaning fallback sweeps all columns at once with Irons-Tuck acceleration
  (419 s to 97 s on a 500k-row panel with 1% movers). pyfixest's compiled demeaner stays
  the default and a requirement.
- `prep` uses a two-level shuffle through buckets when one encode task would open more than
  `max_open` (128) shards (one 69M-row file into 3,434 shards peaked at 23.5 GB; now 5.8 GB).

### Fixed
- Regressors an FE absorbs are dropped as collinear, as fixest does, on every path. A column
  constant within a local FE is rounding noise net of it, and was kept (b ~ 4e11, plan C and
  the cell path failed to converge); a regressor a crossing FE absorbs (carrier x quarter
  with `carrier^time`) was kept while a dummy was dropped. Each column is now tested against
  its sum of squares before the FEs, dummies first (OLS and IV).
- IV round 2 builds one equation's scores at a time, and none for exact first stages: with
  ~100 endogenous regressors all at once was tens of GB per shard (EVE order-days: now
  3:54 and 7.9 GB for 2.6M rows).
- An empty estimation sample (every row a singleton, missing or zero-weight) is a ValueError
  saying where the rows went; `fit` returned an empty result.
- The sparse path crashed with two or more crossing FEs (offsets applied twice).
- Folder locks: two jobs that found the same stale lock could both take it, the second
  deleting the first's fresh lock. A takeover now sets the stale lock aside and deletes it
  only if it is the file it judged stale, and new locks are hard-linked into place fully
  written.
- `dfeols submit SCRIPT.py --profile P -- ARGS` passed `--profile P` to the script; options
  after the script and before `--` now go to `submit`.
- `executor(profile, **overrides)` ignored the overrides when given a Profile.
- `fit(cells=True)` sized workers as if it built dense dummy columns (the full DB1B fit was
  refused at "4601 GB" per shard) and warned of a dense coordinator Gram matrix it never
  builds.
- `dfeols plan` crashed when a candidate key shared a column with an FE, and counted NaN
  outcomes that fit drops.
- Plan C's progress line showed round 2 with a 6-second ETA for the whole CG run.
- The CLI exits 141 without a traceback when its output is piped into `head`.
- Distributed prep with a relative `shard_dir` wrote its pieces under the PartitionBy base
  folder (Polars 2 reads relative paths against it) and failed its row check; task roots
  are now absolute.

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

[Unreleased]: https://github.com/jemather/dfeols/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/jemather/dfeols/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/jemather/dfeols/releases/tag/v0.7.0
