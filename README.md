# dfeols

[![CI](https://github.com/jemather/dfeols/actions/workflows/ci.yml/badge.svg)](https://github.com/jemather/dfeols/actions/workflows/ci.yml)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](https://github.com/jemather/dfeols/blob/main/LICENSE)
![Python 3.10–3.14](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue.svg)

**Sharded, elastic high-dimensional fixed-effects OLS and IV for HPC clusters.**

dfeols estimates `y ~ x | fe1 + fe2^fe3` models on datasets far larger than
one machine's memory (benchmarked at 1e10 rows) by splitting the data into
shards and running two map/reduce rounds over a pool of cluster workers that
grows as the scheduler starts them.

- **General HDFE.** fixest-style formulas; OLS and IV (2SLS, LIML, Fuller);
  iid, heteroskedasticity-robust (HC1), and one- or two-way cluster-robust
  (CRV1) SEs; analytic weights; several outcomes on a common sample. FEs that
  cross the shards are absorbed directly or, when they have too many levels,
  by conjugate gradient across shards. Unit × time panels are the typical
  case; nothing is tied to a particular dataset.
- **Interactive sessions.** Absorb a set of FEs once, then fit many models
  against it from Python, the command line or R: coefficients and iid SEs with
  no pass over the data, robust SEs with one.
- **Plans before it runs.** `dfeols plan` probes a specification before prep:
  which shard key to use, the solver path, worker and coordinator memory, and
  a wall-time range from earlier runs, with advice on one job vs a session.
- **Any scheduler.** Site details (scheduler, queue, modules, scratch, worker
  sizes, network interface) live in one profile file. Slurm, PBS/Torque,
  SGE/UGE, LSF and local are supported through dask-jobqueue.
- **Elastic.** Workers join as the scheduler starts them; the fit starts with
  the first one and the ETA shortens as more arrive. Lost workers' shards are
  retried.
- **Exact.** Matches pyfixest/fixest to about 1e-8 or better (coefficients,
  SEs, small-sample corrections), and recovers known parameters on synthetic
  data.
- **Records.** Every fit writes an [estrec](https://github.com/estrec/estrec)
  record: estimates, sample, small-sample corrections and provenance, in a
  form other tools and journals can check.

See **[HOWTO.md](https://github.com/jemather/dfeols/blob/main/HOWTO.md)** for the step-by-step workflow with a real
dataset, and [docs/ARCHITECTURE.md](https://github.com/jemather/dfeols/blob/main/docs/ARCHITECTURE.md) for how it works
inside.

## Install

```bash
pip install "dfeols[dask] @ git+https://github.com/jemather/dfeols@v0.8.0"
# development version:
pip install "dfeols[dask] @ git+https://github.com/jemather/dfeols"
```

`[dask]` adds dask, distributed and dask-jobqueue. dfeols is not on PyPI
until its dependency [estrec](https://github.com/estrec/estrec) is (see
[Limitations](#limitations)). The R session client installs from `r/` (see
[its README](https://github.com/jemather/dfeols/blob/main/r/README.md)).

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

iv = dfeols.fit(ds, "y ~ x1 | unit + time | d ~ z", cluster="unit+state", executor=ex)
```

`cluster="unit+state"` clusters two ways (Cameron, Gelbach and Miller), as
pyfixest's `vcov={"CRV1": "unit+state"}`.

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
| crossing (e.g. `time`; `carrier^quarter`) | dummy regressors; with one local FE (or several, when a crossing FE exceeds `max_dummy_levels`) their cross-products come from per-level sums, so no dummy columns are built, and the coordinator solves the dummy block (practical to about 30,000 levels; 15,000 take under a minute and 11 GB). Otherwise dense dummy columns, up to `max_dummy_levels`=2000 |
| crossing, many levels (above 30,000 dummies, or above `max_dummy_levels` with no local FE) | conjugate gradient across shards (`crossing_fe="iterate"`, "plan C"): A = D'WD is built in round 1 and solved on the coordinator when it fits `level_space_gb`, else one map/reduce round per step |

A fit is two map/reduce rounds over shards: (1) demean and return `X'WX`,
`X'Wy`; solve for β on the coordinator; (2) residuals and the sandwich meat.
Worker memory scales with shard size, not total N.

When the data compress (many rows per combination of every FE and the
cluster), `fit(..., cells=True)` instead fits exactly from per-cell sums in
one pass (10.3M DB1B rows in 329,000 cells: 9.8 s against 27 s): OLS and IV, with iid and
one- or two-way CRV1 SEs.

**Choosing the partition key.** Pick the key the most FEs are nested in.
Unit × time: partition by unit (unit FE local, time as dummies). With
`unit + group^time` and units nested in groups: partition by group (both FEs
local). The same model under different keys gives identical results
(tested). `dfeols info <shard_dir>` shows how each FE was classified.

### Formula syntax

`y1 + y2 ~ x + a:b + i(c) + x:i(c) + x:i(c, ref=0) | fe1 + fe2^fe3 | d1 + d2 ~ z1 + z2`

| Term | Meaning | Stata | fixest/pyfixest |
|---|---|---|---|
| `x` | continuous | `x` | `x` |
| `a:b` | product | `c.a#c.b` | `a:b` |
| `i(c)` | dummies, first level dropped | `i.c` | `i(c)` |
| `x:i(c)` | x times every level of c | `c.x#ibn.c` | `i(c, x)` |
| `x:i(c, ref=0)` | same, level 0 dropped | `c.x#i.c` | `i(c, x, ref=0)` |
| `\| a + b^d` | absorbed FEs | `absorb(a b#d)` | `\| a + b^d` |
| `\| a[x]` | FE a plus a slope on x per level of a | `absorb(a##c.x)` | `\| a[x]` (fixest only) |
| `\| a[[x]]`, `a[x, z]` | slopes only (no FE intercept); several slopes | `absorb(a#c.x)`, `absorb(a##c.(x z))` | the same |
| `\| d ~ z` | IV: d endogenous, z excluded instruments | `ivreghdfe y (d = z), absorb(...)` | `\| d ~ z` |

**Varying slopes** (`unit[t]`: unit-specific trends) follow fixest: the FE `unit`
must be encoded by `prep()` (its slopes need no new prep), only FEs with an intercept drop
singletons, every slope coefficient counts in K, and with only slope-only FEs the constant
is reported. On an FE nested in the partition key, slopes are projected out exactly per
group (one local FE) or by alternating projections (several, in NumPy, as pyfixest's
compiled demeaner has no slopes). With slopes in the model, crossing FEs are dense dummies,
up to `max_dummy_levels`; the sparse path, plan C, `cells=True` and sessions don't take them yet.

Multiple outcomes share one pass and a common sample (rows missing any
outcome are dropped). IV takes `estimator="2sls"` (default) or `"liml"`, and
`fuller=` for Fuller's modified LIML; `Result.iv["first_stage"]` has each
endogenous regressor's first stage (partial R², F tests under each vcov, the
effective F with one endogenous regressor), and each outcome the
overidentification tests.

## Planning a run

```bash
dfeols plan /data/panel/*.parquet --formula "y ~ x | unit + time" --cluster state \
    --profile mycluster
```

`dfeols plan` (or `dfeols.plan()`) reads the source with streaming aggregates
before anything is written: rows and missingness, levels and singletons of
each FE, which FEs and the cluster each candidate shard key nests, the largest
groups, and for each key the solver path `fit` would take, shard count and
worker and coordinator memory, checked against the profile. It recommends a
key (`prep(partition="auto", formula=...)` takes it), and `fit(cells=True)`
where the data compress.

It also estimates **wall time**. Every `fit` appends its path, size, passes,
seconds and workers to `<scratch>/timings/fits.jsonl`, and the report gives
each key a range from earlier runs: with a dozen or more (one on the key's path), a regression
of log time on the fit's size, parallelism and path, its range the 5-95% of earlier misses
(on OSCER's record it held 80% of fits, median width 10x); with fewer, the spread of earlier
runs' rates. It then advises one job,
or demeaning first in a session: for plan C, when the estimate is over half
the profile's `max_walltime`, or when `--specs N` says several specifications
will share the FEs.

## Interactive sessions

A session demeans a set of declared variables on its FEs once, keeps the
residualised columns on scratch, and fits models from them while it runs:
coefficients and iid SEs from their Gram matrix with no pass over the data,
hetero and CRV1 (one- or two-way) with one pass. Results equal `dfeols.fit`
on the same sample.

```bash
dfeols session start flags --data /scratch/me/shards/db1b --fe "route + carrier^quarter" \
    --vars lfare,hhi,dist,entrants --cluster route,carrier --hours 12 --profile mycluster --wait
dfeols session fit flags "lfare ~ hhi + dist" --cluster route+carrier
```

```python
s = dfeols.Session.attach("flags", profile="mycluster")
s.add_variable("hhi_x_dist", "hhi * dist")           # from raw columns, demeaned like the rest
m1 = s.fit("lfare ~ hhi + hhi_x_dist + i(qtr) + dist:i(region)", cluster="route")
m2 = s.refit(m1, vcov="hetero")                      # b again; one pass for the SEs
m3 = s.fit("lfare ~ dist | hhi ~ entrants", estimator="liml")
s.export([m1, m2, m3], "table2.tex")                 # through estrec's table tool
```

The **R client** in [`r/`](https://github.com/jemather/dfeols/tree/main/r)
attaches to the same session through its request files (no network path to the
compute node, no Python) and returns each model as an estrec record, so
`tidy()`, `glance()` and modelsummary (with broom installed) work. On OSCER, from a login node
against a 40-worker session, its models equal Python's to 1e-10 and each request waits about
0.6 s beyond the session's own work. Each model's record carries the
session id, and the session's collection logs every model, created variable
and setting change, so `dfeols session replay` can run them again. Demeaned
data are kept under `<scratch>/demeaned/` and reused by a later session with
the same inputs; `dfeols demeaned list` and `dfeols demeaned rm` manage them.
Design: [docs/design/sessions.md](https://github.com/jemather/dfeols/blob/main/docs/design/sessions.md).

## Stacked equations

To test coefficients across equations (an outcome against another, mediation paths),
stack them into one regression with each equation's own FEs; the one clustered VCE then
covers every contrast:

```python
eqs = [dfeols.Equation(1, y="eng", x={"x": "feat", "p": "posts"}, fe=["spell", "week"]),
       dfeols.Equation(2, y="unf", x={"x": "lfeat", "p": "lposts"},
                       fe=["id_f", "id_a", "week"], rows=pl.col("at_risk"))]
lf = pl.scan_parquet(src).filter(dfeols.sample_groups("id_f", 0.15, seed=1))  # 15% of followers
st = dfeols.stack(lf, eqs, partition="id_f", keep=["tbin", "id_a"])
ds = dfeols.prep(st.frame.collect(), shard_dir=..., partition="id_f",
                 fixed_effects=st.fixed_effects, clusters=["id_a"])
r = dfeols.fit(ds, st.formula("x:i(tbin) + p:i(tbin)"), cluster="id_a")
r.coef()[st.coef_index(r.coefnames, "x", 2)]          # equation 2's x-by-bin coefficients
```

For data too large to collect, `st.frame.sink_parquet(...)` and prep the files.
Each FE is absorbed as `k^fe`. Where one equation has fewer FEs than another, its
slot is filled with a level its own FEs already span, so the fit doesn't change. Such
fillers still count in K, as fixest would count them; see `dfeols/stack.py`.

## Poisson and cloglog

```python
r = dfeols.fit_glm(ds, "unf ~ lfeat:i(tbin) + lposts | id_f + id_a + week",
                   family="cloglog", cluster="id_a")     # or family="poisson"
r.options["glm"]   # iterations, deviance, rows dropped as separated
```

Fitted by IRLS, each step a weighted dfeols fit, so it scales and shards like `fit`; the
working response lives in per-shard files between steps. Separated FE groups are dropped
first. Equal to pyfixest's `fepois` and R fixest's `feglm`; see `dfeols/glm.py`.

## Discrete-time hazard models

`dfeols.lpm_dthazard(ds, "exit ~ x1 + i(age) | fe", unit="id", time="t")`
fits a discrete-time hazard linear probability model (an absorbing outcome) by
the Farbmacher and Tauchmann (2021) IV: levels instrumented by their
differences, with the first-difference and pooled OLS estimates for
comparison. It builds the sample per shard (gaps and missing lags dropped and
counted), absorbs extra FEs with a warning where that is not equivalent, and
reproduces the published estimates of Farbmacher and Tauchmann's Table 3.
Design: [docs/design/hazard.md](https://github.com/jemather/dfeols/blob/main/docs/design/hazard.md).

## Site profiles (any HPC system)

```bash
dfeols config init          # writes ~/.config/dfeols/config.toml; edit [profiles.cluster]
dfeols config init --from-slurm --profile mysite --exclude 'interactive*'   # draft node classes
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

On Slurm, `dfeols config init --from-slurm` drafts a profile's node classes
from `sinfo` and `scontrol show partition` (run on a login node, or from saved
output with `--sinfo`/`--scontrol`): partitions over the same nodes, limits
and policy collapse to one class, with cores (capped at `MaxCPUsPerNode`),
memory, walltime and nodes in service. GPU, down, hidden or redundant
partitions, and any matching `--exclude PATTERN`, are drafted with
`use = false` and the reason. It prints the draft; `--write` appends it to the
config and never replaces a profile.

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
| `test_vs_pyfixest.py` | equals pyfixest on hard FE structures, weights, multiple outcomes, clusters crossing shards, only-dummy FEs, every crossing-FE path including `cells=True`, Dask executor |
| `test_panel.py` | generic panel: each spec finds what it should, unit- vs group-partitioned fits identical, group clustering matters, weights, pyfixest agreement |
| `test_influencer.py` | influencer DGP: recovery, negative controls, shard and executor invariance, Monte Carlo coverage (`-m slow`) |
| `test_prep_distributed.py` | distributed prep equals single-node prep under every strategy (merge, two-level shuffle, hash ids, range shards, multi-column keys) |
| `test_demean.py` | compiled and NumPy demeaning agree; exact two-way equals iteration |
| `test_iv.py`, `test_iv_iterate.py` | 2SLS, LIML and Fuller equal pyfixest, fixest, linearmodels and ivreghdfe on every FE path |
| `test_crossing.py` | plan C's pieces: block pseudo-inverses, the bounded sum of the shards' parts of A, the preconditioner choice, CG |
| `test_fit_twoway.py` | two-way clustering equals pyfixest for each kind of cluster pair on every crossing-FE path, and for IV |
| `test_session*.py` | session models equal `fit` (OLS, IV, created variables, `i()`, two-way clustering); the R client equals Python |
| `test_dthazard.py` | `lpm_dthazard` equals hand-built IV samples fitted with pyfixest and `fit` |
| `test_record.py` | every record and collection validates against estrec's schemas |

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
  coefficients. With several local FEs the dummy block's projection is solved
  per connected component of the local FEs (a Schur complement, as for exact
  two-way demeaning); `crossing_fe="auto"` uses it above `max_dummy_levels`. At T=1000 the dense path ran out of memory on a 2M-row panel;
  the sparse path took 5 s.
- Keep `fe_tol` tight (default 1e-10). At pyfixest's default 1e-6 a two-FE
  within-account model stopped early and was off by up to 2.2 in testing.

## Limitations

- **Not yet supported:** clustering on three or more variables.
- **Varying slopes (`fe[x]`)** with crossing FEs take dense dummies only, so every crossing
  FE must fit `max_dummy_levels`; not on the sparse path, plan C, `cells=True` or in
  sessions yet.
- **The cell path** (`cells=True`) has no HC1 SEs (they need each row's residual) and no
  varying slopes.
- **Plan C's speed depends on the FE graph.** Crossing FEs too large for a direct solve
  are absorbed by conjugate gradient; when A does not fit `level_space_gb` each step is
  a pass over the shards, and steps grow as the FEs connect less well. `iter_tol` (1e-8)
  and `iter_maxiter` (5,000) control it, the step count is in
  `Result.options["crossing_cg"]`, and a fit stopped by walltime resumes from its
  checkpoint. `dfeols plan` shows which path a key gets and warns before a slow one.
- **Two local FEs plus crossing FEs** build dense dummy columns up to
  `max_dummy_levels` (a warning is logged above 300 columns; `crossing_fe="sparse"`
  avoids them). Above it they take the sparse path, which needs each connected
  component of the local FEs within a shard to have at most 20,000 levels outside
  its largest FE.
- **Prep reads whole files per task.** A dataset stored as a few very large
  Parquet files should be split into many smaller files (say 1–5 GB each)
  first, so prep can spread over the workers.
- **The ETA cannot foresee workers that have not started yet.** The "if all
  workers start" figure is a lower bound.
- **Schedulers:** Slurm is tested on a real cluster; the PBS, SGE and LSF
  templates are not yet. A fit's workers are single-node jobs; there is no
  multi-node launch.
- **Not on PyPI yet:** dfeols depends on estrec by a pinned git commit until estrec
  is on PyPI; install from GitHub.
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
Its `estimation` result records are written and validated with
[estrec](https://github.com/estrec/estrec) (MIT; the format's
specification and schemas are CC0 1.0).
