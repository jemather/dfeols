---
hide:
  - navigation
---

# dfeols

**Sharded, elastic high-dimensional fixed-effects OLS and IV for HPC clusters.**

dfeols estimates `y ~ x | fe1 + fe2^fe3` models on datasets far larger than one machine's
memory (benchmarked at 1e10 rows) by splitting the data into shards and running two
map/reduce rounds over a pool of cluster workers that grows as the scheduler starts them.

```bash
pip install "dfeols[dask]"
```

```python
import dfeols

ex = dfeols.executor("mycluster")          # a site profile; "local"; or omit for serial
ds = dfeols.prep("/data/panel/*.parquet", shard_dir="/scratch/me/shards/panel",
                 partition="unit", fixed_effects=["unit", "time"],
                 clusters=["unit", "state"], executor=ex)
res = dfeols.fit(ds, "y ~ x1 + x2:i(size) | unit + time", cluster="state", executor=ex)
print(res.summary())
```

## What it does

- **General HDFE.** fixest-style formulas; OLS and IV (2SLS, LIML, Fuller); iid,
  heteroskedasticity-robust (HC1), and one- or two-way cluster-robust (CRV1) SEs; analytic
  weights; several outcomes on a common sample. FEs that cross the shards are absorbed
  directly or, when they have too many levels, by conjugate gradient across shards. Unit ×
  time panels are the typical case; nothing is tied to a particular dataset.
- **Interactive sessions.** Absorb a set of FEs once, then fit many models against it from
  Python, the command line or R: coefficients and iid SEs with no pass over the data, robust
  SEs with one.
- **Plans before it runs.** `dfeols plan` probes a specification before prep: which shard key
  to use, the solver path, worker and coordinator memory, and a wall-time range from earlier
  runs, with advice on one job vs a session.
- **Any scheduler.** Site details (scheduler, queue, modules, scratch, worker sizes, network
  interface) live in one profile file. Slurm, PBS/Torque, SGE/UGE, LSF and local are supported
  through dask-jobqueue.
- **Elastic.** Workers join as the scheduler starts them; the fit starts with the first one
  and the ETA shortens as more arrive. Lost workers' shards are retried.
- **Exact.** Matches pyfixest/fixest to about 1e-8 or better (coefficients, SEs, small-sample
  corrections), and recovers known parameters on synthetic data.
- **Records.** Every fit writes an [estrec](https://estrec.org) record: estimates, sample,
  small-sample corrections and provenance, in a form other tools and journals can check.

## Where to go next

<div class="grid cards" markdown>

- **[Install](install.md)**: the package, the Dask extra, and a site profile for your
  cluster.
- **[Quick start](quickstart.md)**: prepare a dataset and fit a model, locally or on a
  cluster.
- **[Workflow on a cluster](guide/workflow.md)**: the full workflow on a real dataset, from
  setup to results.
- **[How it works](guide/how-it-works.md)**: shards, partition keys and how each FE is
  absorbed.
- **[Reference](reference/index.md)**: every public function and class.

</div>
