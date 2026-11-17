# Limitations

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
- **Upstream APIs:** dfeols uses polars' `PartitionBy` sink (marked unstable
  by polars) and pyfixest's compiled demeaner (a private API, with a NumPy
  fallback). Supported version ranges are pinned in `pyproject.toml` and a
  nightly CI job tests the newest releases.
