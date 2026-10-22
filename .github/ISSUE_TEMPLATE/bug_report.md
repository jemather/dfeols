---
name: Bug report
about: Wrong results, crashes, or cluster/scheduler problems
labels: bug
---

**What happened**

**What you expected**

**Reproduction**
Formula, `prep()` arguments, and ideally a small synthetic example
(`python -m dfeols.synth ...`).

**Environment**
- `dfeols --version`:
- Python:
- polars / pyfixest / dask / dask-jobqueue versions:
- Scheduler and site (Slurm, PBS, local, …):
- `dfeols config show` (remove account names):

**Logs**
Coordinator log and, for cluster problems, one worker log.
