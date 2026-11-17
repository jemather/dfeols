# Changelog

All notable changes to dfeols are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/) (before 1.0, minor versions may
break the API).

## [Unreleased]

### Added

- Documentation site at https://jemather.github.io/dfeols/ (built with Zensical from
  `mkdocs.yml`, deployed to GitHub Pages by `.github/workflows/docs.yml`): getting started, a
  user guide, an API reference generated from the docstrings, benchmarks, validation and the
  design notes. `pip install -e ".[docs]"` and `zensical serve` build it locally. It was first
  built with MkDocs Material, which is now in maintenance mode.

### Changed

- Public docstrings are in NumPy style, and every parameter of the public API is described.
- The README is a short landing page; its sections moved to the docs site. `HOWTO.md` moved to
  `docs/guide/workflow.md` ("Workflow on a cluster").
- The benchmarks page compares dfeols with pyfixest, fixest and reghdfe in speed and memory on
  four public datasets (9.5M to 1.6B rows), and dfeols on one large-memory node with dfeols on
  10-80 cluster workers.
- User-facing text no longer uses the internal names "plan B" and "plan C": `dfeols plan`'s
  report, `fit`'s and `plan`'s docstrings, `--level-space-gb`'s help and the docs say "the
  sparse path" and "conjugate gradient" (the CG solve, `crossing_fe="iterate"`). The JSON
  report's `path` keys (`"B"`, `"C"`) are unchanged.
- Faster crossing-FE solves on the coordinator (EVE contracts market r1-r4: 42 to 8 minutes
  locally, results equal to 1e-10):
  - `crossing_fe="auto"` takes conjugate gradient above 5,000 crossing-FE dummies (was 30,000).
    Between the two, the dense solve's Gram (up to 43 GB) took minutes where CG took seconds
    (28,879 dummies: 306 s dense, 15 s by CG). `crossing_fe="sparse"` still forces the dense
    solve. Fits in that range now meet `iter_tol` rather than being exact.
  - In level space, `crossing_precond="auto"` runs a short trial of Jacobi and of the
    block-structure choice (block-Jacobi or the AP-sweep) and keeps the one that reaches the
    tolerance sooner by the clock. The block preconditioner can cut steps 7-30x (DB1B, NFIP)
    or barely at all at 4-5x the cost a step (EVE). The trial is in the run record
    (`crossing_cg.precond_trial`).
  - The product with A in level space runs on up to 16 threads (`DFEOLS_MATVEC_THREADS`), with
    the same result to the bit.
- An acknowledgement of OSCER, where the computing for this project was performed, and of
  the OSCER staff who supported it.

## [0.9.0] - 2026-11-18

The first release on PyPI. Late-task copies in round 1 and CG steps, row conditions at fit time
(`subset=`), and the scale fixes from the full-size runs.

### Added

- Late tasks copied: in the iterated crossing-FE stages (round 1 and each CG step), once no
  task is left to hand out, a task out for over `DFEOLS_COPY_AFTER_S` (default 30 s) and 4x
  its expected time (`DFEOLS_COPY_FACTOR`) is copied to the fastest idle worker; the first of
  the two to finish counts and the other is cancelled, so each part is summed once. In D11 L3
  (36130493) 65 of 310 CG steps each waited on one task of 60-430 s (median 3-4 s) on two
  nodes shared with other jobs: 3.2 of 8.1 h. The log says how many were copied and how many
  copies won. `DFEOLS_COPY_AFTER_S=0` turns it off.
- `fit(subset=...)` (and `fit_glm`): a row condition on the dataset's columns, e.g.
  `"vanished == 1"` or `"duration < 365 & is_buy == 0"` (comparisons joined by `&`, `|`, `!`,
  in `dfeols.expr` syntax; `expr.condition`). Rows where it does not hold, or where a column
  it reads is null or not finite, are left out in each shard before listwise deletion and
  singletons, so one prep serves every sample (D6 v2's `duration < 365` sample was a 96 GB
  filtered copy with its own 61-minute prep). Every path takes it (local, dense, sparse and
  iterated crossing FEs, two-way clusters, cells, IV, predict, GLM's separation scan); each
  equals a prep of the filtered rows (tests/test_subset.py). The record keeps the condition
  as written as the sample's restriction, counts the rows outside it as `dropped.subset`, and
  replays it in `call`; its canonical form is part of the spec id. i() levels come from the
  subset's rows (cached per condition in `_levels/`), so a subset that removes the reference
  level makes the next one the reference, as a prep of those rows and fixest do.
- Profile `interface = "ib"`: the coordinator and every worker use their own node's
  InfiniBand (IPoIB) address, found at job start (names differ by node type: `ib0`,
  `ibp4s0`); a worker node without one falls back to its primary address and says so in its
  log. A coordinator without one keeps every worker on its default address, with a warning:
  on OSCER a coordinator on a node without IB could not gather results from workers on IB
  addresses and the fit stalled. On OSCER dfeols' traffic went over `eno1` Ethernet while
  each node also has IPoIB: D6 r4's CG steps copy a 0.76 GB block to every worker and sum
  parts of that size, and waited 50-110 s a step before any task ran. Pair it with a
  scheduler constraint that keeps jobs on IB nodes (OSCER: `extra_directives =
  ["--constraint=hdr"]`).
- Run diagnostics (`dfeols.diag`): a Dask fit writes `tasks.jsonl` in its workdir (a line per
  task: the task's timing, seconds the coordinator spent fetching its result and in
  `on_result`; a summary per stage with wall time, summed task time, mean tasks running at
  once and task-time quantiles; CG steps after the second summarised only) and
  `cluster.jsonl` (every `DFEOLS_DIAG_SECONDS`, default 30: per worker the tasks assigned,
  threads and memory, the scheduler's task states and the bytes of results held, the
  coordinator's backlog of finished results). `dfeols.executor` sends the coordinator's
  `distributed` and `dask_jobqueue` log (the scheduler runs there) to `scheduler.log` next
  to the worker logs. `scheduler_profile.jsonl` (per stage, and for CG steps 1, 2 and every
  10th): the scheduler event loop's busy seconds and its top functions, from Dask's own
  profile of that thread (the client shares it). Stage summaries count CG cache hits
  and the seconds tasks spent reading shards. `DFEOLS_DIAG=0` turns the JSONL files off.

- HDFE GLMs by IRLS: `dfeols.fit_glm(ds, formula, family="poisson" | "cloglog", cluster=...)`.
  Each iteration is an ordinary weighted `fit` of the working response, so every FE path
  and executor applies; round 2 computes the new linear predictor (xb + every FE) in the
  shard and writes the next working response and weights to per-shard sidecar files, so
  only deviances reach the coordinator. FE groups whose outcome never varies (Poisson: all
  zero; binary: all 0 or all 1) are dropped first, repeated until stable. Poisson equals
  pyfixest's `fepois` (N, coefficients, CRV1, HC1 and iid SEs) with local FEs, the sparse
  crossing path and the CG solve; cloglog equals R fixest's `feglm(binomial("cloglog"))`.
  Convergence on relative deviance change (`tol`, default 1e-8 as fixest; tighten it for
  agreement beyond about 1e-5). Not yet: prior weights, varying slopes, `cells=True`.
- In-sample prediction including FEs: `fit(..., predict=folder, predict_keep=["row_id"])`
  writes each estimation row's `xb` (reported regressors), `d` (the sum of its FE terms,
  local and crossing) and `resid`, one Parquet part per shard, during round 2 (no extra
  pass over the shards beyond re-reading each shard's rows). Equal to pyfixest's
  `predict()` and `resid()` with local FEs, dense dummies, the sparse crossing path and
  the CG solve, with weights and several outcomes. OLS only; not with `cells=True`.
- Stacked regressions with per-equation FEs (`dfeols.stack`, `dfeols.Equation`): each
  equation's rows tagged by `k`, its outcome in `y`, regressor roles as per-equation
  columns (`x_k1`, `x_k2`, ... zero off the equation), and every FE interacted with `k`
  in slots lined up across equations (local FEs with local, crossing with crossing;
  missing slots filled with a level the equation's other FEs span). One fit gives every
  equation's coefficients and one joint clustered VCE for cross-equation tests.
  `Stacked.formula("x:i(tbin) + p")` expands roles per equation; `same_rows=True` gives
  every equation the same observations; `Stacked.coef_index` finds an equation's terms.
  Coefficients equal each equation fitted alone, and the stacked fit equals pyfixest on
  the stacked rows. `dfeols.sample_groups(by, frac, seed)`: a filter keeping a fixed
  share of groups with all their rows, from a hash stable across machines and Polars
  versions (nested fractions nest).
- `dfeols convert-dta SRC OUT` (`dfeols.convert.convert_dta`): streams a Stata `.dta` file
  into 1 GB Parquet parts with bounded memory. One Arrow schema comes from the header, so
  integer columns stay int8/16/32 even in chunks with missing values. Stata missing becomes
  null, value labels are kept as codes, dates as Stata codes, and labels, formats, value
  labels and null counts go to `_dta_meta.json` (written last, rows checked against the
  header).
- Varying slopes in `fit`, in fixest's syntax: `unit[t]` (unit FE plus a slope on t per
  unit), `unit[[t]]` (slopes only), `unit[t, z]`, `a^b[x]`. Slopes on an FE nested in the
  partition key are projected out in the shard, exactly with one local FE and by
  alternating projections with several. With slopes, crossing FEs are dense dummy columns
  (each slope variable times every level), up to `max_dummy_levels`. As fixest: only FEs with
  an intercept drop singletons, K counts every slope coefficient (also unidentified ones,
  and with no discount for nesting in the cluster), and a model with only slope-only FEs
  reports its intercept. Equal to fixest 0.14.2 (coefficients 1e-9, SEs 1e-8) on 19 specs:
  local and crossing slopes, weights, iid, HC1, one- and two-way CRV1, IV
  (`tests/test_slopes.py`). The sparse path, the CG solve, `cells=True` and sessions refuse
  slopes with a message; `dfeols plan` counts slope columns and marks such keys not built.
- Sessions take `i()` on text variables and interacted, `x:i(c)` and `x:i(c, ref=...)`, as
  `fit` does (columns `c::level:x`), equal to `fit` on dense and iterated crossing FEs.
- Created-variable expressions compare a column with a quoted string, `region == "west"`
  (`==` and `!=` only; the column is read as text, categoricals too).
- `fit(cells=True)` takes IV (2SLS, LIML, Fuller, with first stages and overidentification
  tests) and two-way CRV1. The cells give round 1's Gram matrices net of every FE exactly, so
  the row path's own solvers run on them, and round 2's RSS and meats are the solve's linear
  maps applied to each cell's sums. Equal to the row path in every number of the result.

### Changed

- On PyPI: `pip install "dfeols[dask]"`. estrec is now required by version
  (`estrec>=0.1,<0.2`, from PyPI) instead of a pinned git commit, which PyPI does not
  accept. The README has the dfeols mark (`docs/dfeols.svg`).
- CG steps with a pass per step (D6 r4 on 2.7-4.2B rows: ~10 min a pass; D11 L3):
  each task goes back to the worker that cached its shard in round 1 (`DaskExecutor.run(...,
  affinity=)`; job 36011018 hit the cache on 2.5% of tasks and reread every shard); a cached
  shard keeps its dummies on its own levels, so a step's product is over those, not a dense
  P x c block per task; the CG block reaches every worker as copies in a tree rather than
  each worker fetching it from one (~2.5 min a step at 0.68 GB x 36 workers); step tasks ask
  for their own memory, not round 1's level arrays; the step sums are merged by position
  (no `np.add.at`, dense above a quarter of the levels) and a combine asks for its estimate
  instead of a whole worker (whole-worker combines at the head of a worker's queue held it
  to one task at a time); the CG checkpoint is written at most every 5 min. The shard cache
  keeps what it holds when full instead of evicting (LRU on a loop that does not fit hit
  nothing, D11 L3: 2 of 1,443) and takes 40% of a worker (was 25%).
- CG steps on OSCER after the above (D6 lt365 r4 job 36097628, D11 L3 36097608): task
  time fell 3-5x but passes did not get faster. Now every step task is placed by the
  coordinator, largest first, on the worker where it would finish first (its shard cached
  there counts once, elsewhere 4x), so a worker holding many shards no longer queues 51-68
  tasks while most sit idle: a loose restriction stays on its worker however busy, and
  stealing will not move a task whose input is that worker's copy of the block. Each task
  takes its own worker's copy, never the single root copy (each miss fetched the block from
  one worker again). The affinity map follows a task only to a worker that kept its state
  (results say `cached`), and after a step where tasks moved, workers drop the stale copies
  (`worker.iter_cache_keep`); hits had fallen from 2,241 of 2,902 to 485. Finished results
  are fetched by 16 threads (`DFEOLS_FETCH_THREADS`) instead of one at a time: one fetch
  from a busy worker held up the rest (cg5: 148 s, with 1,078 results waiting), and
  fetching took 300-700 s of each 500-900 s pass.
- CG steps (with the placement above, D11 L3 job 36113341, cg836): a placement fixed
  at the start of a step still left six workers with 18-29 tasks queued at 100 s of a
  ~190 s step while the other 17 ran nothing. Step tasks are now handed out as workers free
  up: each worker holds at most 2x its threads submitted (`DFEOLS_QUEUE_DEPTH`), takes its
  own cached shards first, then unowned ones, then the tail of the worker that would finish
  last, when that worker would finish it later than this one reading the shard again
  (4x). Each worker's pace is learnt from its tasks' times and kept across steps, so a slow
  node sheds its queue. Workers that join mid-step get tasks at once; a gone worker's queue
  goes back to the shared one.
- CG steps asked far more worker memory than they use: every step task asked for the
  largest shard's estimate, with the cache share applied twice (round 1's estimate was
  already scaled for it), so D6 r4 (job 36113346) asked 14.7 GB of a 24 GB worker per task
  and ran one step task per worker: 4-5 tasks running on 34 workers, 400-900 s a step for
  ~2,000 task-seconds. Each step task now asks for its own shard's rows
  (`Progress.task_gb_of`, which `DaskExecutor` uses per task), the cache counted once:
  ~3.7 GB for a 1M-row D6 shard.
- `np.unique` without `return_*` in NumPy 2.5 hashes, 30-50x slower than a sort on many
  distinct integers (1M level codes: 0.69 s against 0.02 s). Shard level sets, present
  levels and cluster codes and the CG step combines use `crossing.sorted_unique`: a D6 step
  combine of 4 x 300k levels 0.77 -> 0.16 s (~970 combines a step). Combines switch to the
  dense P-row sum when the parts hold half the levels (was a quarter), where it is faster.
- The CG solve without level space: round 1's per-level sums are summed in a tree on the workers
  (D11 L3 36016180: 27 min fetching ~98 MB results one at a time). With level space, once a
  combine finds the summed parts of A past the budget, the shards still to run build only
  their diagonal blocks, or nothing (D6 r4 35948257 built ~4.5e10 nonzeros for a 4e8 budget).
- The progress line's rows/s and stage ETA follow the rows whose results have arrived (over
  the last 10 min), not the tasks' own seconds (D6 r4 35953385 round 1: 5.2M rows/s shown,
  0.25M achieved, ETA 7 min for 4 h).
- A fit using the CG solve logs its CG mode (level space budget or off) before round 1, not only after.
- `dfeols submit` refuses a script path that does not exist, and takes `--coordinator-memory`.

- Worker logs go in `<scratch>/logs/<coordinator job id>/` (outside a batch job,
  `logs/local-<time>-<pid>/`), not all in `<scratch>/logs/`. Each file is still named by the
  worker's own job id, so a run's worker logs could not be told apart from every other run's.
  The coordinator logs the folder at start.

- `dfeols plan` recommends `cells=True` only when the cells fit the coordinator's memory;
  otherwise it gives the need and the smallest node class that holds it. IV specs are eligible.

- `dfeols plan`'s wall-time ranges come from a regression on earlier runs (log time on size,
  parallelism and path; the range is 5-95% of its misses) once there are a dozen of them. On
  OSCER's record it held 80% of fits at a median width of 10x, where the old range of earlier
  runs' rates was 57x wide (up to 5,600x) and 3x off at its midpoint.

- The R client is tested on a cluster: on OSCER (R 4.5.2, login node, a 40-worker session on
  DB1B market flags) fit, refit, a created variable, two outcomes and LIML IV equal the same
  fits from Python to 1e-10; requests wait a median 0.6 s beyond the session's work. Its
  README and DESCRIPTION name broom, which modelsummary needs for the records.

### Fixed

- A resumed CG solve (`cg_checkpoint.npz`) estimated the steps left from a fall
  measured against step 0's residual of 1, so for its first 30 steps it said "about 10 more
  steps" when hundreds were left (D11 L3 resumed at step 808). The estimate now uses only the
  resumed steps.
- The DaskExecutor's result loop: `distributed.wait(FIRST_COMPLETED)` over every pending
  future, called after each completion, left a waiter per future on the event loop the
  scheduler shares, so the loop stalled more as a stage went on (D6 36011018, D11 36016180:
  877 stalls, median 5 s; found by the D11 chat). Done callbacks now feed a queue, once per
  future; the scheduler is queried once a tick.
- A result that cannot be fetched (a worker out of local disk for spilled data: job 35937527,
  53 min blocked in Dask's gather retries) fails the fit after `DFEOLS_GATHER_TIMEOUT_S`
  (900 s) with the worker named.
- A coordinator ended by SIGTERM (scancel, walltime) cancels its worker jobs, queued ones too.

- The CG solve's per-process shard cache (`worker._ITER_CACHE`) is now guarded by a lock. A worker's
  threads clearing the previous fit's entries together raised `KeyError` (OSCER job 35991256:
  43 round-1 tasks failed and were retried), and a CG step's check-then-read could lose its
  entry to another thread's eviction.
- The CG solve's worker-memory estimate counts round 1's arrays over every crossing level
  (P x (2(k+m)+4) doubles, ~0.6 GB a task at 6.24M levels) and the sparse level matrices, and
  a round-1 task's memory request reserves the worker's CG cache (scaled by W / (W - cache)),
  so fewer round-1 tasks share a worker when the cache is large.

- Round 1 with a large crossing FE no longer sends every shard's per-level arrays dense. Each
  shard returned a present mask and each level's smallest and largest cluster over all of the
  crossing FE's levels, ~265 MB a shard at 15.5M location-days (D6 r4 on OSCER: ~1.15 TB
  merged one shard at a time on the coordinator, round 1 ~4 h). Shards now send only their own
  levels (~18 MB there); results are unchanged, and dense arrays from older shard results
  still merge.
- The IV first-stage R² is net of every FE, as the outcomes' within R² is. It was net of the
  FEs nested in the partition only (or of none), so it changed with the partition key.
- The CG solve in level space no longer runs workers out of memory. Shards cap their part of A by
  the worker's memory, not the coordinator's, so four parts fit in one combine. Combines add
  their inputs one at a time, and keep only the diagonal blocks (then nothing) before passing
  60% of the worker's memory, so the fit falls back to a pass per CG step instead of losing
  workers. A 162M-row fit with 51,844 crossing levels killed 24 GB workers in round 1.
- Round 2 without a cache no longer rebuilds each shard's part of A when it recomputes
  round 1. The rebuild sat outside the task memory estimate, so several such tasks on one
  worker killed it.
- Shards count their part of A before building it, not after. A part over the cap, and
  diagonal blocks over it too, is never built. Blocks that fit are built one crossing FE at a
  time. Building first took 110 GB on two days of NYC rideshare trips, where A is nearly
  dense over 60,720 zone pairs; counting first takes 0.6 GB and 2 s.
- `dfeols plan` caps A's budget at what the workers can sum (60% of a worker's memory), as
  fit does.
- Arguments shared by every task that are dicts of large arrays (round 2's linear maps, with
  the CG solve's FE solution) go to the workers once instead of inside every task: a 758,405-level
  fit sent a 46 MiB task graph.
- Distributed prep's reduce no longer gathers the scan's level sets and nesting checks in the
  coordinator's memory. Scan tasks write those frames to `<shard_dir>/_scan/` and return only
  their paths; the coordinator counts levels, checks hash ids for collisions and nesting, and
  writes code tables in streaming passes of ~50M frame rows (`REDUCE_ROWS`), each over one
  bucket of the key's hash, so the counts stay exact. D6 v2's lt365 sample (2.9B rows, 424M
  type x region x day levels) had the 61 GB coordinator killed twice right after the scan
  (OSCER 35988182, 36008269): on the same frames, concat + `unique()` peaked at 71 GB (80 GB
  with the nesting check) and the streaming passes at 5 GB, 18 s instead of 93 s.

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
- Crossing FEs on the sparse path with several FEs nested in the partition. The
  dummy block's projection on the local FEs is a Schur complement solved per connected
  component (pseudo-inverse, so redundant local FEs are fine); round 2 demeans the dummy
  terms with all local FEs. `crossing_fe="auto"` takes it when a crossing FE exceeds
  `max_dummy_levels` (below that, several local FEs keep dense dummies); `"sparse"` forces it.
- `crossing_fe="iterate"` (the CG solve): crossing-FE coefficients by preconditioned conjugate
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
  cells, absorbs the FE with most levels by group means and solves the others with the CG solve's
  level-space CG, then b by Frisch-Waugh-Lovell, RSS and CRV1 scores from the cell moments.
  Exact. OLS; vcov CRV1 (one-way) and iid. DB1B 2005Q2-Q3: 10.3M rows in 329k cells, 9.8 s
  against 27 s (level space) and 43 s (dense solve); b within 1e-11.

#### Interactive sessions
- `dfeols.Session` and `dfeols session start|serve|status|where|fit|refit|add-variable|export|persist|replay|stop`
  (docs/design/sessions.md). A manager demeans the declared variables on the session's FEs
  once (local FEs, dense crossing FEs, or one CG solve for every declared
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
  profile (with the CG solve, A's size against the level-space budget). It recommends the key that
  makes the crossing system smallest among keys whose largest group is within `target_rows`
  or half a worker's share of the rows, and `fit(cells=True)` from 2 rows per cell. DB1B's
  683M-row market table probes in 72 s with 8.8 GB.
- Wall-time estimates: every `fit` appends its path, rows, columns, passes, seconds and
  workers to `<scratch>/timings/fits.jsonl` (`dfeols.timings`), and `dfeols plan` gives each
  key a wall-time range from earlier runs on the same path. It recommends demeaning first in
  a session for the CG solve, for estimates over half the profile's `max_walltime`, or with
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
  estimate, so a 24 GB, 4-core worker runs up to 4 tasks together. Prep tasks and the CG solve's
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
  constant within a local FE is rounding noise net of it, and was kept (b ~ 4e11, the CG solve and
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
- The CG solve's progress line showed round 2 with a 6-second ETA for the whole CG run.
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

[Unreleased]: https://github.com/jemather/dfeols/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/jemather/dfeols/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/jemather/dfeols/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/jemather/dfeols/releases/tag/v0.7.0
