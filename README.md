# dfeols

See [HOWTO.md](HOWTO.md) for the step-by-step workflow with a real dataset.

Sharded, elastic high-dimensional fixed-effects OLS for HPC clusters.

- **General HDFE.** Any `y ~ x | fe1 + fe2^fe3` model with one-way clustered,
  heteroskedasticity-robust or iid SEs, weights and multiple outcomes. Unit x
  time panels are the typical case; nothing is tied to a particular dataset.
- **Any scheduler.** Site details (scheduler, queue, modules, scratch, worker
  sizes, network interface) live in one profile file. Slurm, PBS/Torque,
  SGE/UGE, LSF and local are supported through dask-jobqueue.
- **Elastic.** Workers join as the scheduler starts them; the fit starts with
  the first one and the ETA shortens as more arrive. Lost workers' shards are
  retried.
- **Exact.** Matches pyfixest/fixest to about 1e-8 (coefficients, SEs, small-
  sample corrections), tested on synthetic data with known parameters.

## How it works

The data are split into shards by a **partition key**: groups of that key
never span shards. Each FE is classified:

| FE relative to the partition key | handled as |
|---|---|
| nested in it (e.g. `unit` when partitioning by unit) | demeaned inside each shard |
| crossing, few levels (e.g. `time`, up to `max_dummy_levels`=2000) | dummy regressors; with one local FE their cross-products come from per-level sums, so no dummy columns are built |
| crossing, many levels | not yet supported (see Roadmap) |

A fit is two map/reduce rounds over shards: (1) demean and return `X'WX`,
`X'Wy`; solve for β on the coordinator; (2) residuals and the sandwich meat.
Worker memory scales with shard size, not total N.

**Choosing the partition key.** Pick the key the most FEs are nested in.
Unit x time: partition by unit (unit FE local, time as dummies). With
`unit + group^time` and units nested in groups: partition by group (both FEs
local). The same model under different keys gives identical results
(tested). `dfeols info <shard_dir>` shows how each FE was classified.

## Install

```bash
python -m venv ~/envs/dfeols && source ~/envs/dfeols/bin/activate
pip install -e ".[dask,test]"
pytest -q tests/
```

## Use

```python
import dfeols

ds = dfeols.prep(
    "/data/panel/*.parquet",
    shard_dir="/scratch/me/shards/panel",
    partition="unit",
    fixed_effects=["unit", "time"],
    clusters=["unit", "state"],
)

ex = dfeols.executor("mycluster")  # or "local", or omit for serial
res = dfeols.fit(ds, "y ~ x1 + x2:i(size) | unit + time", cluster="state", executor=ex)
print(res.summary())
res.tidy()
res.coef()
res.vcov()
```

### Formula syntax

`y1 + y2 ~ x + a:b + i(c) + x:i(c) + x:i(c, ref=0) | fe1 + fe2^fe3`

| Term | Meaning | Stata | fixest/pyfixest |
|---|---|---|---|
| `x` | continuous | `x` | `x` |
| `a:b` | product | `c.a#c.b` | `a:b` |
| `i(c)` | dummies, first level dropped | `i.c` | `i(c)` |
| `x:i(c)` | x times every level of c | `c.x#ibn.c` | `i(c, x)` |
| `x:i(c, ref=0)` | same, level 0 dropped | `c.x#i.c` | `i(c, x, ref=0)` |
| `\| a + b^d` | absorbed FEs | `absorb(a b#d)` | `\| a + b^d` |

Multiple outcomes share one pass and a common sample (rows missing any
outcome are dropped).

## Site profiles (any HPC system)

```bash
dfeols config init          # writes ~/.config/dfeols/config.toml; edit [profiles.cluster]
dfeols config show          # resolved profile
dfeols check                # start one worker through the scheduler, run a task
dfeols jobscript "python run.py"         # print the coordinator job script
dfeols submit run.py                     # submit run.py as the coordinator job
dfeols bench --profile cluster --rows 1e8
```

A profile holds everything site-specific; code never mentions a queue or a
module. Examples: `examples/profiles/oscer.toml` (tested Slurm site) and
`examples/profiles/templates.toml` (PBS, SGE, LSF, local). Select one with
`dfeols.executor("name")`, `--profile name`, `$DFEOLS_PROFILE`, or `default`.
Walltimes are written once in any form (`12:00:00`, `2-00:00:00`) and
converted to each scheduler's native format.

Built-in site workarounds: on configless Slurm sites, a `/proc` `SLURM_CONF`
inside a job breaks `sbatch` and is cleared; the coordinator listens on a
private (cluster-internal) interface by default, because a campus-facing
address is often unreachable from compute nodes (`interface = "auto"`).

Run the coordinator as a batch job (`dfeols submit`), not in an interactive
allocation: when the allocation expires, the coordinator and the fit go with it.

## Progress and ETA

Each fit writes `status.json` and `history.jsonl` to its run directory and logs
lines like

```
[main round1] | 312/1200 shards | 26.1% rows | 18 workers (max 40) | 2.31M rows/s |
elapsed 0:42:10 | stage ETA 1:58:03 (1:47:12-2:09:40) | total ETA 2:31:20 -> Wed 09:14
```

`dfeols status <run dir or shard dir> --watch 60` shows it from any login node.
The ETA divides remaining work (from measured seconds per row) by the workers
connected now, never less than the largest shard still waiting, with a
bootstrap band; round 2 is projected from round 1's timing until measured.

## Validation and benchmarks on synthetic data

`dfeols.synth` generates data with known parameters at any size, chunk by
chunk (in parallel locally or on cluster workers):

- **panel** (default): generic unit x time panel, units nested in groups,
  unbalanced, regressors confounded with unit, time and (optionally)
  group-time effects; AR(1), heteroskedastic, group-correlated errors.
- **influencer**: follower x account x week panel with tenure bins and a
  piecewise-linear true response.

Each dataset's `truth.json` lists the true coefficients and a set of specs
with what each should find: `unbiased`, `biased` (omits a confounded FE) or
`se_wrong` (clustering ignores the error structure). Tests and benchmarks are
driven by these, so a new DGP brings its own checks.

```bash
python -m dfeols.synth /scratch/me/raw --rows 1e8 --jobs 16 --set T=104
pytest tests/ --rows 5e6 --data-dir /scratch/me/dfeols-tests
python -m dfeols.bench --rows 1e7 1e8 --profile cluster
python -m dfeols.bench --dgp influencer --specs level shape --rows 1e8 --profile cluster
```

| Test file | Checks |
|---|---|
| `test_vs_pyfixest.py` | equals pyfixest on hard FE structures, weights, multiple outcomes, clusters crossing shards, only-dummy FEs, Dask executor |
| `test_panel.py` | generic panel: each spec finds what it should, unit- vs group-partitioned fits identical, group clustering matters, weights, pyfixest agreement |
| `test_influencer.py` | influencer DGP: recovery, negative controls, shard and executor invariance, Monte Carlo coverage (`-m slow`) |

The benchmark's `summary.md` reports, per size and spec: validity verdict,
fit and round times, rows per worker-second (queue-independent), workers
available, peak worker memory, ETA error, and an optional pyfixest baseline.

## Performance notes

- Python is not the bottleneck: cross-products run in BLAS, demeaning in
  pyfixest's compiled core, I/O in Arrow/Polars.
- FEs local to the shard are demeaned **exactly, without iteration** when
  there are one or two of them. Two FEs use a Schur-complement solve: the FE
  with more levels per connected component is eliminated in closed form, and
  the other is solved with a small dense Cholesky (LAPACK) per component.
  On the influencer `shape` spec (spell + account x week) this cut demeaning
  from 168s to 6s per 600k-row shard and agrees with iteration to 1e-14.
  Where alternating projections already converge in a few sweeps (e.g.
  unit + group^time) the two are about equally fast. `fe_method="iterate"`
  forces iteration; three or more local FEs always iterate (pyfixest's core).
- Crossing FEs (e.g. time in unit x time, partitioned by unit) never build
  dummy columns when there is one local FE: their blocks of X'WX come from
  per-level sums, and the vcov uses Frisch-Waugh-Lovell on the reported
  coefficients. 1.6x faster at T=104; at T=1000 the dense path ran out of
  memory on a 2M-row panel, the sparse path took 5s.
- Demeaned arrays are cached between rounds only when demeaning is iterative
  (`cache="auto"`); otherwise round 2 recomputes, which beats re-reading.
- Keep `fe_tol` tight (default 1e-10). At pyfixest's default 1e-6 a two-FE
  within-account model stopped early and was off by up to 2.2 in testing.

## Roadmap

- Sparse crossing FEs alongside two local FEs (currently dense dummies).
- High-cardinality crossing FEs: alternating projections as map/reduce rounds.
- Two-way clustering, IV, heterogeneous slopes `fe[x]`.

## Layout

```
dfeols/formula.py    formula parser (+ pyfixest translation)
dfeols/prep.py       single-node streaming prep: FE codes, nesting checks, LPT sharding
dfeols/prep_dist.py  distributed prep: the same as map/reduce rounds on workers
dfeols/worker.py     shard tasks (rounds 1 and 2)
dfeols/demean.py     within-shard demeaning
dfeols/estimate.py   coordinator: plan, reduce, solve, inference
dfeols/vcov.py       small-sample rules
dfeols/executors.py  Serial and Dask executors
dfeols/progress.py   ETA model, status.json, history.jsonl
dfeols/site.py       site profiles (TOML)
dfeols/cluster.py    scheduler-agnostic executors and job scripts
dfeols/synth/        synthetic DGPs (panel, influencer)
dfeols/validate.py   compare fits with known truth
dfeols/bench.py      benchmark runner
dfeols/cli.py        dfeols status | info | config | check | jobscript | submit
```
