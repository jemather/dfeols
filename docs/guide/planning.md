# Planning a run

```bash
dfeols plan /data/panel/*.parquet --formula "y ~ x | unit + time" --cluster state \
    --profile mycluster
```

`dfeols plan` (or [`dfeols.plan()`](../reference/data.md#dfeols.planner.plan)) reads the source with streaming aggregates
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
