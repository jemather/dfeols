# dfeols

[![CI](https://github.com/jemather/dfeols/actions/workflows/ci.yml/badge.svg)](https://github.com/jemather/dfeols/actions/workflows/ci.yml)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](https://github.com/jemather/dfeols/blob/main/LICENSE)
![Python 3.10–3.14](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue.svg)

**Sharded, elastic high-dimensional fixed-effects OLS for HPC clusters.**

dfeols estimates `y ~ x | fe1 + fe2^fe3` models on datasets far larger than
one machine's memory (benchmarked at 1e9 rows) by splitting the data into
shards and running two map/reduce rounds over a pool of cluster workers that
grows as the scheduler starts them.

- **General HDFE.** fixest-style formulas; iid, heteroskedasticity-robust
  (HC1) or one-way cluster-robust (CRV1) SEs; analytic weights; several
  outcomes on a common sample. Unit × time panels are the typical case;
  nothing is tied to a particular dataset.
- **Any scheduler.** Site details (scheduler, queue, modules, scratch, worker
  sizes, network interface) live in one profile file. Slurm, PBS/Torque,
  SGE/UGE, LSF and local are supported through dask-jobqueue.
- **Elastic.** Workers join as the scheduler starts them; the fit starts with
  the first one and the ETA shortens as more arrive. Lost workers' shards are
  retried.
- **Exact.** Matches pyfixest/fixest to about 1e-8 or better (coefficients,
  SEs, small-sample corrections), and recovers known parameters on synthetic
  data.

See **[HOWTO.md](https://github.com/jemather/dfeols/blob/main/HOWTO.md)** for the step-by-step workflow with a real
dataset, and [docs/ARCHITECTURE.md](https://github.com/jemather/dfeols/blob/main/docs/ARCHITECTURE.md) for how it works
inside.

## Install

```bash
pip install "dfeols[dask]"          # [dask] adds dask, distributed and dask-jobqueue
# development version:
pip install "dfeols[dask] @ git+https://github.com/jemather/dfeols"
```

Python 3.10–3.14. Linux HPC clusters are the target; CI also runs the test
suite on macOS for local use. Without `[dask]`,
fits run serially in one process, which is useful for testing a model on a
subsample.

## Quick start

```python
import dfeols

ex = dfeols.executor("mycluster")          # a site profile; "local"; or omit for serial

ds = dfeols.prep("/data/panel/*.parquet", shard_dir="/scratch/me/shards/panel",
                 partition="unit", fixed_effects=["unit", "time"],
                 clusters=["unit", "state"], executor=ex)

res = dfeols.fit(ds, "y ~ x1 + x2:i(size) | unit + time", cluster="state", executor=ex)
print(res.summary())
res.tidy(); res.coef(); res.vcov()
```

On a cluster, run that script as a batch job so it outlives your login
session:

```bash
dfeols config init          # writes ~/.config/dfeols/config.toml; edit it for your site
dfeols check                # start one worker through the scheduler and run a task
dfeols submit run.py        # run.py becomes the coordinator job
dfeols status /scratch/me/shards/panel --watch 60
```

## How it works

The data are split into shards by a **partition key**: groups of that key
never span shards. Each FE is classified:

| FE relative to the partition key | handled as |
|---|---|
| nested in it (e.g. `unit` when partitioning by unit) | demeaned inside each shard: exactly for one or two such FEs, iterated for three or more |
| crossing, few levels (e.g. `time`, up to `max_dummy_levels`=2000) | dummy regressors; with one local FE their cross-products come from per-level sums, so no dummy columns are built |
| crossing, many levels | not yet supported (see [Limitations](#limitations)) |

A fit is two map/reduce rounds over shards: (1) demean and return `X'WX`,
`X'Wy`; solve for β on the coordinator; (2) residuals and the sandwich meat.
Worker memory scales with shard size, not total N.

**Choosing the partition key.** Pick the key the most FEs are nested in.
Unit × time: partition by unit (unit FE local, time as dummies). With
`unit + group^time` and units nested in groups: partition by group (both FEs
local). The same model under different keys gives identical results
(tested). `dfeols info <shard_dir>` shows how each FE was classified.

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
module. Examples: [`examples/profiles/oscer.toml`](https://github.com/jemather/dfeols/blob/main/examples/profiles/oscer.toml)
(a tested Slurm site) and
[`examples/profiles/templates.toml`](https://github.com/jemather/dfeols/blob/main/examples/profiles/templates.toml) (PBS,
SGE, LSF, local; untested on real systems). Select one with
`dfeols.executor("name")`, `--profile name`, `$DFEOLS_PROFILE`, or `default`.
Walltimes are written once in any form (`12:00:00`, `2-00:00:00`) and
converted to each scheduler's native format.

Built-in site workarounds: on configless Slurm sites, a `/proc` `SLURM_CONF`
inside a job breaks `sbatch` and is cleared; the coordinator listens on a
private (cluster-internal) interface by default, because a campus-facing
address is often unreachable from compute nodes (`interface = "auto"`);
Polars' spill directory goes to scratch and its memory limit is pinned to the
job allocation.

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

## Benchmarks

Synthetic panels of 1e9 rows with known parameters, on a Slurm cluster (OU
OSCER) with up to 40 four-core workers (about 36 on average):

| | 10M units × T=100 | 1M units × T=1000 |
|---|---|---|
| fit `unit + time` | 178 s | 91 s |
| fit `unit + group^time` | 158 s | 157 s |
| peak worker memory | 3.4 GB | 2.8 GB |
| prep by unit / by group (distributed) | 700 s / 543 s | 524 s / 1300 s |
| recovers the truth | yes | yes |

At **1e10 rows** (100M units × T=100, 5,000 shards per set, up to 40
workers), both specs recovered the truth (`unit + time`: max |z| 0.35;
`unit + group^time`: max |z| 1.79). The fits took 47 and 59 minutes, peak
worker memory was 4.3 GB, and the shard sets took 39 minutes (by unit) and
74 minutes (by group, through the two-level shuffle) to prepare: about 3 hours
40 minutes end to end.

Reproduce with `dfeols bench --profile <name> --preset units10m_t100` (or
`units1m_t1000`); see [HOWTO.md](https://github.com/jemather/dfeols/blob/main/HOWTO.md#scaling-to-a-billion-rows-and-beyond)
for resource budgets and the 1e10-row stress presets.

## Validation

`dfeols.synth` generates data with known parameters at any size, chunk by
chunk (in parallel locally or on cluster workers):

- **panel** (default): generic unit × time panel, units nested in groups,
  unbalanced, regressors confounded with unit, time and (optionally)
  group-time effects; AR(1), heteroskedastic, group-correlated errors.
- **influencer** (example): follower × account × week panel with tenure bins
  and a piecewise-linear true response; clusters cross shards.

Each dataset's `truth.json` lists the true coefficients and a set of specs
with what each should find: `unbiased`, `biased` (omits a confounded FE) or
`se_wrong` (clustering ignores the error structure). Tests and benchmarks are
driven by these, so a new DGP brings its own checks.

```bash
python -m dfeols.synth /scratch/me/raw --rows 1e8 --jobs 16 --set T=104
pytest tests/ --rows 5e6 --data-dir /scratch/me/dfeols-tests
python -m dfeols.bench --rows 1e7 1e8 --profile cluster
```

| Test file | Checks |
|---|---|
| `test_vs_pyfixest.py` | equals pyfixest on hard FE structures, weights, multiple outcomes, clusters crossing shards, only-dummy FEs, Dask executor |
| `test_panel.py` | generic panel: each spec finds what it should, unit- vs group-partitioned fits identical, group clustering matters, weights, pyfixest agreement |
| `test_influencer.py` | influencer DGP: recovery, negative controls, shard and executor invariance, Monte Carlo coverage (`-m slow`) |
| `test_prep_distributed.py` | distributed prep equals single-node prep under every strategy (merge, two-level shuffle, hash ids, range shards, multi-column keys) |
| `test_demean.py` | compiled and NumPy demeaning agree; exact two-way equals iteration |

In a Monte Carlo of 40 replications, CRV1 intervals covered the truth 95.8%
of the time (iid: 85.8%, as expected with clustered errors).

## Performance notes

- Python is not the bottleneck: cross-products run in BLAS, demeaning in
  pyfixest's compiled core, I/O in Arrow/Polars.
- FEs local to the shard are demeaned **exactly, without iteration** when
  there are one or two of them. Two FEs use a Schur-complement solve: the FE
  with more levels per connected component is eliminated in closed form, and
  the other is solved with a small dense Cholesky (LAPACK) per component.
  On the influencer `shape` spec (spell + account × week) this cut demeaning
  from 168 s to 6 s per 600k-row shard and agrees with iteration to 1e-14.
  `fe_method="iterate"` forces iteration; three or more local FEs always
  iterate.
- Crossing FEs (e.g. time in unit × time, partitioned by unit) never build
  dummy columns when there is one local FE: their blocks of X'WX come from
  per-level sums, and the vcov uses Frisch–Waugh–Lovell on the reported
  coefficients. At T=1000 the dense path ran out of memory on a 2M-row panel;
  the sparse path took 5 s.
- Keep `fe_tol` tight (default 1e-10). At pyfixest's default 1e-6 a two-FE
  within-account model stopped early and was off by up to 2.2 in testing.

## Limitations

- **Not yet supported:** IV, two-way clustering, heterogeneous slopes
  (`fe[x]`), and crossing FEs with more than `max_dummy_levels` levels (for
  example a follower FE when partitioning by account). The executor design
  allows adding alternating projections as extra map/reduce rounds.
- **Two local FEs plus crossing FEs** build dense dummy columns (a warning is
  logged above 300 columns).
- **Prep reads whole files per task.** A dataset stored as a few very large
  Parquet files should be split into many smaller files (say 1–5 GB each)
  first, so prep can spread over the workers.
- **The ETA cannot foresee workers that have not started yet.** The "if all
  workers start" figure is a lower bound.
- **Schedulers:** Slurm is tested on a real cluster; the PBS, SGE and LSF
  templates are not yet.
- **Upstream APIs:** dfeols uses polars' `PartitionBy` sink (marked unstable
  by polars) and pyfixest's compiled demeaner (a private API, with a NumPy
  fallback). Supported version ranges are pinned in `pyproject.toml` and a
  nightly CI job tests the newest releases.

## Citing

If you use dfeols in published work, please cite it; see
[CITATION.cff](https://github.com/jemather/dfeols/blob/main/CITATION.cff) (GitHub's "Cite this repository" button gives
BibTeX and APA).

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](https://github.com/jemather/dfeols/blob/main/CONTRIBUTING.md).
Tested site profiles for new clusters are especially useful.

## License

dfeols is free software, licensed under the
[GNU General Public License v3.0 or later](https://github.com/jemather/dfeols/blob/main/LICENSE).
