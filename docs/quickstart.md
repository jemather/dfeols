# Quick start

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

- [`prep`](reference/data.md#dfeols.prep.prep) encodes the FEs and clusters and writes the
  data as shards, split by the **partition key** (`unit` here). Every FE a later fit uses
  must be listed in `fixed_effects`. See [How it works](guide/how-it-works.md) for choosing
  the key, or let [`dfeols plan`](guide/planning.md) choose it.
- [`fit`](reference/fitting.md#dfeols.estimate.fit) runs the model and returns a
  [`Result`](reference/fitting.md#dfeols.results.Result). The formula syntax follows fixest;
  see [Formulas](guide/formulas.md).
- `cluster="unit+state"` clusters two ways (Cameron, Gelbach and Miller), as pyfixest's
  `vcov={"CRV1": "unit+state"}`.

## On a cluster

Run the script as a batch job, so it outlives your login session:

```bash
dfeols config init          # writes ~/.config/dfeols/config.toml; edit it for your site
dfeols check                # start one worker through the scheduler and run a task
dfeols submit run.py        # run.py becomes the coordinator job
dfeols status /scratch/me/shards/panel --watch 60
```

[Workflow on a cluster](guide/workflow.md) walks through the whole workflow on a real
dataset.
