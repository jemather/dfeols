# Install

```bash
pip install "dfeols[dask]"
# development version:
pip install "dfeols[dask] @ git+https://github.com/jemather/dfeols"
```

`[dask]` adds dask, distributed and dask-jobqueue. Without it, fits run serially in one
process, which is useful for testing a model on a subsample. Estimation records come from
[estrec](https://estrec.org), installed with dfeols.

Python 3.10–3.14. Linux HPC clusters are the target; CI also runs the test suite on macOS
for local use.

Other extras:

| Extra | Adds |
|---|---|
| `dask` | dask, distributed, dask-jobqueue: fits and prep on cluster workers |
| `estrec` | jsonschema, so records are validated against estrec's JSON Schemas (otherwise built-in checks) |
| `amg` | pyamg, for plan C's preconditioner on large sparse crossing-FE components |

The R session client installs from `r/`; see [R client](guide/r-client.md).

## Set up your cluster

dfeols reaches your cluster's scheduler through a **site profile**: one TOML file with the
scheduler, queue, modules, scratch folder and worker sizes. Create one, then check that a
worker starts:

```bash
dfeols config init          # writes ~/.config/dfeols/config.toml; edit it for your site
dfeols check                # start one worker through the scheduler and run a task
```

On Slurm, `dfeols config init --from-slurm` drafts the profile from `sinfo`. See
[Site profiles](guide/site-profiles.md) for the details and tested examples.
