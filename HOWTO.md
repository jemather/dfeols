# dfeols: how to use it with a real dataset

This is the short version. The README covers the design; `python -m dfeols.bench --help`,
`python -m dfeols.synth --help` and `dfeols --help` cover the tools.

## 1. One-time setup on a machine

```bash
python -m venv ~/envs/dfeols && source ~/envs/dfeols/bin/activate
pip install "dfeols[dask]"    # or: pip install "dfeols[dask] @ git+https://github.com/jemather/dfeols"
dfeols config init            # writes ~/.config/dfeols/config.toml
```

Create the venv on a filesystem the compute nodes can see (usually your home directory), with
the same Python module the profile's `prologue` loads.

Edit the profile in that file:
- `scheduler`, `queue`, `account`
- `prologue`: the module loads and venv activation every job runs
- `scratch`: a shared filesystem every node can see
- `[worker]` cores, memory, walltime and `max_workers`

Smaller, shorter worker jobs start sooner on a busy queue. Workers are released only after
`idle_minutes` (default 15) with nothing to do, so a pause between fits doesn't hand back
workers that waited in the queue. `examples/profiles/oscer.toml`
is a tested Slurm profile; `examples/profiles/templates.toml` covers PBS, SGE, LSF and local.

Set `default = "<name>"` at the top of the file so commands use your profile without
`--profile`. `dfeols config init --force` replaces the file with the template (it saves a backup
first). Every `submit`/`bench` prints the profile and module lines it used.

Then check:

```bash
dfeols check                  # one worker starts through the scheduler and runs a task
```

If it fails:
- jobs pending: queue limits
- jobs exit at once: check the worker logs in `<scratch>/logs`
- "Timed out trying to connect": set `interface` in the profile

## 2. Prep the data once

Prep encodes the FE and cluster variables, checks nesting and writes Parquet shards.
Every model on that dataset reuses the shards.

```python
import dfeols
prof = dfeols.profile()
ex = dfeols.executor(prof)
ds = dfeols.prep("/data/panel/*.parquet",                # Parquet files (or a frame / CSV without executor)
                 shard_dir=prof.scratch_dir("shards", "panel"),
                 executor=ex,                               # prep on the workers
                 partition="unit",                          # groups never split across shards
                 fixed_effects=["unit", "time", "unit^quarter"],   # every FE any model uses
                 clusters=["unit", "state"],                # every cluster variable any model uses
                 target_rows=2_000_000)                     # rows per shard
```

**Choose the partition key** so that the most FEs are nested in it.
- Unit × time: partition by unit. The unit FE is then handled inside each shard, and time
  crosses shards.
- `unit + group^time` with units nested in groups: partition by group, so both are local.
- Run `dfeols info <shard_dir>` to see how each FE was classified.

Crossing FEs need up to `max_dummy_levels` (2000) levels; high-cardinality crossing FEs are
not supported yet.

**Shard size:** worker memory peaks at roughly 3.5 × rows × (outcomes + regressors) × 8 bytes.
On OSCER, 2M-row shards peaked at about 15 GB with 74 regressors. Aim for at least 5 shards per
worker so the load balances and the ETA has data.

**Run prep on the workers** with `executor=ex`. This needs Parquet input; split a very large
single file into many, which also parallelizes. Prep then runs as map/reduce rounds:

1. scan each file for its FE levels and group sizes; while no workers have connected yet, the
   coordinator scans files itself
2. build the code tables on the coordinator
3. encode and bucket rows on the workers, in about 4 tasks per worker so the last wave is
   short; each task writes one flat piece file per shard it touches
4. merge only shards that ended up in more than 8 pieces (when partition groups are spread
   across many input files); other shards keep their pieces as is

The coordinator holds only the code tables: 0.3 GB for 30M rows in testing, and roughly
8 bytes per distinct FE level (10M units ≈ 80 MB). Without `executor`, prep streams on the
coordinator alone. That's fine up to a few hundred million rows if the coordinator has the
memory.

At very large scale prep switches strategy on its own (each threshold can be set as a `prep()`
keyword):

| Situation | What prep does | Threshold |
|---|---|---|
| a nested FE or cluster with millions of levels (e.g. 100M units) | 64-bit hash ids instead of a code table join; collisions are checked exactly, with a fallback to codes | `hash_levels=2e6` |
| millions of partition groups | shards are contiguous ranges of the partition key (or its hash for multi-column keys), so tasks need only the boundaries | `range_groups=5e6` |
| partition groups scattered across input files, so one piece per (task, shard) would mean too many files | two-level shuffle: tasks write to buckets, then each bucket is split into its shards | `max_files=5e4` |

Every job script sets two Polars variables:
- `POLARS_TEMP_DIR=<scratch>/tmp/polars`, so spill files go to scratch instead of the node's
  small `/var/tmp`
- `POLARS_OVERRIDE_TOTAL_MEMORY_MB` = the job's memory, so Polars budgets for the job's limit
  rather than the whole node

## 3. Validate the specification before the full run

**Compare against a trusted tool on a subsample.** Prep a subset of whole units and fit it with
dfeols and with your Stata/pyfixest code. Results should agree to about 1e-8.

```python
sub = pl.scan_parquet(src).filter(pl.col("unit") % 50 == 0).collect()
ds_small = dfeols.prep(sub, shard_dir=..., partition="unit", ...)
```

When comparing to `reghdfe`/pyfixest with two FEs, use a tight tolerance (1e-10). At looser
defaults they can stop early.

**Check recovery on synthetic data with the same structure.** Generate a panel with known
parameters and confirm the spec recovers them with valid SEs:

```bash
python -m dfeols.bench --rows 1e7 --set T=104 --set n_groups=300 --profile <name>
```

## 4. Write the run script

Copy `examples/run_fit.py`, set `SOURCE`, `PREP` and `SPECS`:

```python
SPECS = {
  "main": dict(formula="y ~ x1 + x2:i(size) | unit + time", cluster="state"),
  "alt":  dict(formula="y + y2 ~ x1 + x2 | unit + group^time", cluster="unit", weights="w",
               vcov=["CRV1", "hetero"]),
}
```

**Formula syntax:**

| Term | Meaning | Stata |
|---|---|---|
| `x` | continuous variable | `x` |
| `a:b` | product | `c.a#c.b` |
| `i(c)` | dummies, first level dropped | `i.c` |
| `x:i(c)` | x times every level of c | `c.x#ibn.c` |
| `x:i(c, ref=0)` | same, level 0 dropped | `c.x#i.c` |
| `\| fe1 + fe2^fe3` | absorbed FEs | `absorb(fe1 fe2#fe3)` |

**Other options:**
- `vcov`: `"iid"`, `"hetero"` or `"CRV1"`; a list computes several in one pass.
- `weights`: analytic weights.
- Multiple outcomes share one pass and a common sample.

## 5. Submit and monitor

```bash
dfeols submit run_fit.py --profile <name>    # coordinator as a batch job (default profile if omitted)
dfeols status <scratch>/runs --watch 60     # shards done, workers, rows/s, ETA
```

Always submit the coordinator as a batch job. In an interactive allocation, the run dies when
the allocation expires.

**The ETA** is based on the workers connected right now. It drops as the queue starts more
workers. The "if all workers start" figure is the lower bound.

## 6. Where everything goes

Everything lives under the profile's `scratch` directory, in fixed places:

| Path | Contents |
|---|---|
| `jobs/` | coordinator job scripts and their output (`dfeols submit` prints the path) |
| `logs/` | worker logs |
| `runs/<label>/` | your own fits (from `run_fit.py`): result.json, coefs.csv, status.json |
| `bench/raw_<tag>/`, `bench/shards_*` | benchmark data and shards, reused across runs |
| `bench/runs/<run_id>/<tag>/<spec>/` | one benchmark fit; never overwritten |
| `bench/results/results.csv` | every benchmark fit ever run, newest first |
| `bench/results/<run_id>.md` | summary of one benchmark run; `latest.md` is the most recent |

`<run_id>` is the start time plus the scheduler job id, e.g. `20261021-171800-35355433`.
`dfeols results` prints the benchmark table from any login node, and
`dfeols results --run <run_id>` shows one run. The table is rebuilt from `runs/` each time,
so it can't lose earlier fits. Fits from before this layout are picked up too, and their
`run_id` ends in `-legacy`.

## 7. Results

Each spec writes to `<scratch>/runs/<label>/`:

- `result.json`: coefficients, full vcov for every requested type, N, singletons dropped,
  cluster and FE level counts, collinear terms, timings. Load it with
  `dfeols.Result.load(path)`, then use `.tidy()`, `.summary()`, `.coef()` and `.vcov()`.
- `coefs.csv`: the tidy table.
- `status.json` and `history.jsonl`: progress and ETA history.

## 8. What is fast, and what is not

| Model structure | How it's handled | Relative cost |
|---|---|---|
| 1 FE local to the shard | exact, one pass | fastest |
| 1 local FE + crossing FEs (unit + time) | crossing FEs via per-level sums, no dummy columns | fast; T up to 2000 |
| 2 local FEs (unit + group^time, spell + account×week) | exact Schur-complement solve per connected group | fast |
| 2 local FEs + crossing FEs | crossing FEs as dense dummies | memory grows with the levels |
| 3+ local FEs | iterated (alternating projections) | slowest; demeaning dominates |
| crossing FE with > 2000 levels | not supported yet | — |

## Scaling to a billion rows and beyond

The two presets generate balanced panels of 1e9 rows with known parameters. Each fits
`unit + time` (partitioned by unit, time crossing) and `unit + group^time` (partitioned by
group):

```bash
# run 1e8 first to check timing and memory, then the full size
dfeols bench --profile <name> --preset units10m_t100 --rows 1e8
dfeols bench --profile <name> --preset units10m_t100
dfeols bench --profile <name> --preset units1m_t1000
```

One job per data set at a time: a benchmark locks its data set (`<scratch>/bench/locks/`) while it
generates and preps, and a second job for the same preset and size stops at once with an error
naming the job that holds it. Different presets can run side by side. A lock left by a job that
died is taken over after 5 minutes. Shard sets built by versions before 0.6.2 with range-based
shards are rebuilt automatically ("re-preparing …").

Stress tests:

```bash
# 1e10 rows, 100M units, 5,000 shards per set: hash ids, range shards, two-level shuffle
dfeols bench --profile <name> --preset units100m_t100 --rows 2e9   # step up first
dfeols bench --profile <name> --preset units100m_t100
# skewed group sizes: the largest group (~25M rows) becomes one oversized shard
dfeols bench --profile <name> --preset skewed_groups
# all of the 1e10 code paths on a tiny panel, on this machine
python -m dfeols.bench --preset units100m_t100 --rows 2e6 --profile local \
    --prep-opt hash_levels=0 --prep-opt range_groups=0 --prep-opt max_files=0
```

For the 1e10 preset:
- **Scratch:** about 1 TB (raw ≈ 370 GB plus two shard sets).
- **Coordinator:** 64 GB. It holds the 100M-unit level set and group counts during the reduce
  step.
- **Workers:** generation runs on the workers for this preset.
- **Throughput:** at the 1e9 rates, expect roughly 30 min of fitting per spec on 40 workers.

Budget for each 1e9-row preset:

- **Scratch:** about 37 GB of raw data, plus about 40 GB per shard set. The presets need two
  shard sets (by unit and by group), and prep briefly needs another copy as staging, so plan
  on about 160 GB.
- **Coordinator:** it generates the data itself (while the first worker jobs wait in the queue;
  `--gen-on-workers` moves generation to the workers). Prep then runs on the workers. Give it
  8–16 cores and 32 GB, e.g. a `[profiles.<name>.coordinator]` with `cores = 16`,
  `memory = "32GB"`. Most of that is for generation, not prep.
- **Data folders** are named by DGP, size and non-default parameters, for example
  `raw_panel_1e09_T100_bal1_G500`, so the two presets never collide.
- **Measured on OSCER at 1e9 rows** (both presets, up to 40 four-core workers, about 36 on
  average; distributed prep):

  | | 10M units × T=100 | 1M units × T=1000 |
  |---|---|---|
  | generate (coordinator) | 48 s | 337 s |
  | prep by unit / by group | 700 s / 543 s | 524 s / 1300 s |
  | fit `unit + time` | 178 s | 91 s |
  | fit `unit + group^time` | 158 s | 157 s |
  | peak worker memory | 3.4 GB | 2.8 GB |
  | recovers the truth | yes (max \|z\| 1.09, 2.77) | yes (max \|z\| 0.50, 1.36) |

  End to end: about 27 min for T=100 and 40 min for T=1000. Prep is most of it, and part of
  that is queue wait for the first workers.
