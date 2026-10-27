# dfeols architecture

A guide for contributors. The [README](../README.md) covers usage and the
[HOWTO](../HOWTO.md) the workflow on a real dataset.

## Overview

```python
ex  = dfeols.executor("mycluster")        # elastic Dask workers from a site profile
ds  = dfeols.prep("/data/*.parquet", shard_dir=..., partition="unit",
                  fixed_effects=["unit", "time"], clusters=["unit"], executor=ex)
res = dfeols.fit(ds, "y ~ x1 + x2:i(cat) | unit + time", cluster="unit", executor=ex)
```

`prep` writes Parquet shards once per dataset and partition key; every model
on that dataset reuses them. `fit` runs two map/reduce rounds over the shards.
Both run their tasks through an *executor*, which is serial in-process, a
local Dask cluster, or an adaptive dask-jobqueue cluster on an HPC scheduler.

## Prep

`prep.py` (single node) and `prep_dist.py` (distributed; the default when an
`executor=` is given).

Prep encodes FEs and clusters as integer codes, checks which FEs are nested in
the partition key and in each cluster, and writes Parquet shards in which no
partition group is split. The distributed version runs as
scan → reduce → encode → (merge | split) rounds on workers; the coordinator
holds only small tables (level sets and group counts), merging the per-file
scan results as they arrive (`_Compactor`) so overlapping tables such as
group × time levels never pile up. At scale it switches
automatically to:

- hash ids for nested FEs with more than 2e6 levels (`hash_levels`; collisions
  are checked exactly, with a fallback to dense codes);
- range-based shard assignment with more than 5e6 groups (`range_groups`);
- a two-level shuffle when there would be more than 5e4 piece files
  (`max_files`).

Each strategy can be forced on small data (`--prep-opt hash_levels=0`, etc.),
and `tests/test_prep_distributed.py` checks that every strategy produces the
same dataset as single-node prep.

## Fit

`estimate.py` (coordinator) and `worker.py` (shard tasks).

0. **Singletons** (only when an FE crosses the partition): passes over the
   shards drop singletons in every FE, iterating, as fixest and reghdfe do.
   Pass 1 drops local-FE singletons and counts each crossing FE's levels; each
   later pass drops rows whose level count from the previous pass is 1 (an upper
   bound, since shards only drop rows), until a pass drops nothing. Keep masks
   go to `<workdir>/singletons/`, so the workdir must be on shared scratch.
   With only local FEs, each shard finds its singletons in round 1.
1. **Round 1** (per shard): build the design, drop singletons, demean, return
   `X'WX` and `X'Wy`.
2. **Solve** (coordinator): reduce the cross-products and solve, with
   collinearity detection (`linalg.py`).
3. **Round 2** (per shard): residuals and the sandwich meat, either per
   cluster when clusters are nested in the partition key, or as cluster scores
   reduced on the coordinator otherwise.

Small-sample corrections follow fixest/pyfixest (`vcov.py`).

### Fixed effects by type

| FE type | How it's handled |
|---|---|
| FEs nested in the partition key ("local") | demeaned inside shards (`demean.py`) |
| one local FE | exact, one pass |
| two local FEs | exact Schur-complement solve per connected component (dense Cholesky, LAPACK); 37× faster than alternating projections on the influencer `shape` spec |
| three or more local FEs | iterated (pyfixest's compiled alternating projections; NumPy fallback) |
| crossing FEs with ≤ `max_dummy_levels` (2000) levels | dummies. With exactly one local FE they never materialize: cross-products come from per-level sums (`D'W Q D = D'WD − C'diag(1/W_a)C`), and the vcov uses Frisch–Waugh–Lovell on the reported coefficients. This makes T=1000 feasible |

### Robustness

- Every task attempt writes uniquely named files and reports them; the
  coordinator keeps one attempt per task and sweeps the rest, because a task
  can run twice at once (a rerun overlapping a worker the scheduler gave up on).
- Shard folders and benchmark data sets are locked (`lock.DirLock`: a sibling
  `<folder>.lock`, hard-linked into place fully written, heartbeat every 30 s,
  stale after 5 min; a takeover renames the stale lock aside and deletes it only
  if it is still the file it judged stale).
- Overwrites rename a folder before deleting it; an interrupted prep's folder
  (no metadata file) is cleared before reuse.
- Row counts are Int64 everywhere (polars' `len()` is UInt32 and wraps).

## Execution

`executors.py`, `cluster.py`, `site.py`.

- `SerialExecutor` runs in-process; `DaskExecutor` uses any
  `dask.distributed` client, including an adaptive dask-jobqueue cluster whose
  workers join as the scheduler starts them.
- Tasks are submitted largest-first (LPT), after one small calibration task.
- Retries handle lost workers.
- Scale-down is lazy (`idle_minutes`), so workers that waited in the queue
  are not thrown away between fits.
- The coordinator runs read-only scan tasks itself while no workers are
  connected.
- Site profiles (TOML) hold everything scheduler-specific. Built-in fixes:
  - clear a `/proc` `SLURM_CONF` (configless Slurm breaks nested `sbatch`);
  - bind the coordinator to a private/internal interface;
  - set `POLARS_TEMP_DIR` to scratch;
  - pin `POLARS_OVERRIDE_TOTAL_MEMORY_MB` to the job allocation.

## Progress and ETA

`progress.py`.

- Rate = worker-seconds per row from completed tasks.
- Remaining time = remaining work divided by the workers connected now, never
  less than the largest pending shard, with a bootstrap band.
- Round 2 is projected from round 1's timing until measured.
- Writes `status.json` and `history.jsonl`; `dfeols status` reads them.

## Validation tooling

`synth/`, `validate.py`, `bench.py`.

- Two DGPs with known parameters, generated chunk by chunk, each chunk seeded
  from (seed, chunk id):
  - `panel`: generic unit × time with groups, confounding, AR(1) errors and
    optional Zipf group sizes;
  - `influencer`: an example follower × account × week DGP.
- Each dataset's `truth.json` carries the true coefficients plus specs
  labeled `unbiased`, `biased` or `se_wrong`.
- `dfeols.bench` generates, preps, fits and checks recovery, and keeps an
  append-only results layout (`dfeols results`). Presets: `units10m_t100`,
  `units1m_t1000` (1e9 rows each), `units100m_t100` (1e10) and
  `skewed_groups`.

## CLI

`cli.py`: `dfeols config init|show|path`, `check`, `jobscript`,
`submit [-m module]`, `bench`, `status`, `info`, `results`.

## Module map

| File | Role |
|---|---|
| `prep_dist.py` | distributed prep (scan/reduce/encode/merge/split) |
| `worker.py` | shard tasks: design, demean, sparse crossing FEs, meat |
| `bench.py` | benchmark runner, presets, results ledger |
| `estimate.py` | `fit()`: planning, reduce, solve, inference, memory estimate |
| `cluster.py` | scheduler-agnostic executors, job scripts, submit |
| `prep.py` | single-node prep, `Dataset` |
| `progress.py` | ETA model, status files |
| `cli.py` | command line |
| `synth/` | DGPs and the chunked generator |
| `site.py` | TOML site profiles |
| `formula.py` | parser, pyfixest translation |
| `demean.py` | one-way exact, two-way Schur, alternating-projections fallback |
| `executors.py` | Serial and Dask executors |
| `results.py`, `vcov.py`, `linalg.py`, `validate.py` | results object, small-sample rules, collinearity, recovery checks |

## Extending

The executor interface (`run(fn, tasks, on_result, progress, stage)`) is all
a new estimation step needs: map over shards, reduce on the coordinator,
broadcast back. High-cardinality crossing FEs, for example, would be
alternating projections as extra rounds through the same interface.
