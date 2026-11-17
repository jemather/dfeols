# <img src="https://raw.githubusercontent.com/jemather/dfeols/main/docs/dfeols.svg" alt="" width="36" height="36"> dfeols

[![CI](https://github.com/jemather/dfeols/actions/workflows/ci.yml/badge.svg)](https://github.com/jemather/dfeols/actions/workflows/ci.yml)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](https://github.com/jemather/dfeols/blob/main/LICENSE)
![Python 3.10–3.14](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue.svg)
[![Docs](https://img.shields.io/badge/docs-jemather.github.io%2Fdfeols-0b2545.svg)](https://jemather.github.io/dfeols/)

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

**Documentation: [jemather.github.io/dfeols](https://jemather.github.io/dfeols/)**

## Install

```bash
pip install "dfeols[dask]"
```

`[dask]` adds dask, distributed and dask-jobqueue; without it, fits run serially in one
process. Python 3.10–3.14, on Linux HPC clusters (macOS for local use). Estimation records come
from [estrec](https://estrec.org), installed with dfeols. See [Install](https://jemather.github.io/dfeols/install/) for the
other extras and for setting up your cluster.

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

## Learn more

- [Workflow on a cluster](https://jemather.github.io/dfeols/guide/workflow/): the full workflow on a real dataset
- [How it works](https://jemather.github.io/dfeols/guide/how-it-works/): shards, partition keys, and how each FE is absorbed
- [Formulas](https://jemather.github.io/dfeols/guide/formulas/): the formula syntax, next to Stata's and fixest's
- [Planning a run](https://jemather.github.io/dfeols/guide/planning/) and [site profiles](https://jemather.github.io/dfeols/guide/site-profiles/)
- [Interactive sessions](https://jemather.github.io/dfeols/guide/sessions/), with an [R client](https://jemather.github.io/dfeols/guide/r-client/)
- [Stacked equations](https://jemather.github.io/dfeols/guide/stacked/), [Poisson and cloglog](https://jemather.github.io/dfeols/guide/glm/),
  [discrete-time hazard models](https://jemather.github.io/dfeols/guide/hazard/)
- [Benchmarks](https://jemather.github.io/dfeols/performance/), [validation](https://jemather.github.io/dfeols/validation/) and
  [limitations](https://jemather.github.io/dfeols/limitations/)
- [API reference](https://jemather.github.io/dfeols/reference/)

## Citing, contributing, license

If you use dfeols in published work, please cite it; see
[CITATION.cff](https://github.com/jemather/dfeols/blob/main/CITATION.cff) (GitHub's "Cite this
repository" button gives BibTeX and APA). Issues and pull requests are welcome; see
[CONTRIBUTING.md](https://github.com/jemather/dfeols/blob/main/CONTRIBUTING.md). Tested site
profiles for new clusters are especially useful.

dfeols is free software, licensed under the
[GNU General Public License v3.0 or later](https://github.com/jemather/dfeols/blob/main/LICENSE).
Its result records are written and validated with [estrec](https://estrec.org) (MIT; the
format's specification and schemas are CC0 1.0).
