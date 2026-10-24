# Design: the dfeols result file (`dfeols-result/1`)

Status: proposal for discussion (2026-10-24). Replaces the broom-only sketch in `sessions.md`.

## Purpose

One file per estimated model that holds everything needed to (1) read and report the results,
(2) explain a discrepancy with another run or tool, and (3) reproduce the run with confidence.
Rule of thumb: if two results differ, comparing their files should show why (different rows,
FE handling, tolerance, small-sample correction, code version, data version) without rerunning.

Today `Result.save()` writes coefficients, vcov matrices, N, dropped rows, clusters, FE names and
levels, collinear terms, R2s and a few timings. It lacks the data identity, the resolved options,
the environment, convergence information and most diagnostics.

## Principles

- **Self-contained**: one JSON file, readable without dfeols (`jq`, R jsonlite, Python json).
- **Resolved, not as typed**: every option is written with its effective value, defaults
  included, so a later change of default can't change what the file means.
- **Lossless numbers**: float64 written with round-trip precision; non-finite values as the
  strings `"NaN"`, `"Inf"`, `"-Inf"` (JSON has no NaN), with `null` meaning "not computed".
- **Versioned**: `schema` names the format version; a JSON Schema in the repo defines it and the
  tests validate every file the code writes against it.
- **Broom-compatible subset**: `tidy` and `glance` blocks use broom names, so R's modelsummary
  and similar tools read the file directly. They are a view; the full detail is elsewhere.
- **Small**: no per-row data. FE coefficient estimates, residuals and fitted values, if wanted,
  go in a separate Parquet file the result points to (by path and sha256).

## Layout

```text
schema, id, status, warnings, notes, tags
model         what was estimated
sample        which rows, and why rows were dropped
estimates     every parameter
vcov          every vcov computed, with its corrections
statistics    fit and diagnostic statistics
fixed_effects structure and levels of each FE
numerics      convergence, solver, rank, accuracy checks
data          identity of the input data
run           when, where, by whom, how long, with what software
replay        the exact call that reproduces this result
tidy, glance  broom-compatible views
validation    (optional) reference values and differences
```

### Header

| Field | Content |
|---|---|
| `schema` | `"dfeols-result/1"` |
| `id` | UUID of this result; `parent` if it is a refit (e.g. new vcov) of another |
| `status` | `ok`, `warning` or `failed` (a failed run still writes what it knows) |
| `warnings` | list of `{code, message}`: collinear terms dropped, slow convergence, few clusters, singletons, ... |
| `notes`, `tags` | user free text and labels (e.g. `"Table 3 col. 9"`) |

### `model`

- `formula` as written; `depvar`; `terms` (expanded regressor names in order);
  `factor_terms` (each `i()` variable with its levels and the reference level);
  `interactions`; `created_variables` (name, expression, source columns) when from a session.
- `method`: `ols` or `iv`; for IV: `endogenous`, `instruments`, `exogenous`.
- `weights`: column name and type (analytic/frequency) or `null`.
- `fixed_effects`: the FE terms as written (detail under `fixed_effects`).
- `options`: every fit option with its effective value (`fe_tol`, `fe_method`, `drop_singletons`,
  `max_dummy_levels`, `cache`, `crossing_fe`, collinearity tolerance, ...).

### `sample`

- `rows_source`: rows in the dataset.
- `restriction`: the sample definition (expression or declared-variable rule) and rows after it.
- `dropped`: counts by reason: missing per variable, singletons (with passes), zero/negative
  weights, other.
- `nobs`, `sum_weights`.
- `row_key`: a hash of the indices of the rows used (computed shard by shard and combined), so
  two runs can be checked for using the identical sample without storing the rows.
- `partition`: the shard partition key and shard count (results must not depend on it, but it is
  the first thing to check when they do).

### `estimates`

One entry per parameter, in model order, including dropped ones:
`term`, `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`, `conf_high`, `dropped`
(`null`, `"collinear"`, `"reference"`), for the default vcov. Other vcov types appear under
`vcov` with their own standard errors. `conf_level` is recorded once.

### `vcov`

One block per vcov type computed:

- `type` (`iid`, `hetero`/HC1, `CRV1`, later two-way), `cluster` variable(s), `n_clusters` per
  variable, whether each FE is nested in a cluster (affects dof).
- `ssc`: the small-sample correction written out: the formula and every factor
  (`(N-1)/(N-K)`, `G/(G-1)`), what `K` counted (regressors, FE levels absorbed, FE levels nested
  in clusters excluded), and the fixest/reghdfe/Stata convention it matches.
- `df_t`: degrees of freedom for t tests and confidence intervals.
- `matrix`: the full k x k matrix with `terms` labels; `std_errors`.

### `statistics`

- Fit: `rss`, `tss`, `tss_within`, `r2`, `adj_r2`, `r2_within`, `adj_r2_within`, `sigma` (RMSE),
  `df_residual`, `df_model`, `n_parameters` (regressors and absorbed FE levels separately).
- Tests: overall `F` (iid) and Wald test (for the default vcov) with p-values.
- Outcome: `depvar_mean`, `depvar_sd` (raw), `depvar_sd_within`.
- IV (when added): first-stage F per endogenous variable, Kleibergen-Paap rk Wald F,
  Cragg-Donald, Sargan/Hansen J, first-stage partial R2.

### `fixed_effects`

One entry per FE: `name`, `components`, `levels_dataset`, `levels_sample` (after singletons),
`redundant_levels` and how they were determined (exact for nested FEs, approximate otherwise,
as in fixest), `absorbed` (`local` within shards, `crossing_dense`, `crossing_sparse`),
`nested_in_partition`, `nested_in` (clusters and other FEs it nests in).

### `numerics`

- Demeaning: algorithm, `fe_tol`, iterations (max, median over shards), converged (bool),
  worst shard's final change.
- Solver: method (Cholesky/QR), `rank`, condition number of the (within) X'X, collinearity
  tolerance and which terms were dropped.
- Checks: max |X'e| relative to scale (orthogonality of residuals), float dtype (float64).

### `data`

- `dataset`: name and version (e.g. `parsons2011_v1`), path as used, and the sha256 of its
  `MANIFEST.sha256` (identifies every file of the pushed dataset).
- `prep`: shard folder, dfeols prep version, partition, target rows, shard count, FE/cluster
  specs encoded, and a hash of `dfeols_meta.json`.
- `columns`: name and dtype of every column the model read.
- `demeaned`: the key of persisted demeaned data, if the model came from a session.

### `run`

- Times (UTC, ISO 8601, plus the local zone): `submitted`, `started`, `finished`; queue wait if
  known.
- Timing by stage (singletons, round 1, solve, round 2), worker-seconds, peak worker memory,
  rows/s, and the estimate shown at submission (to calibrate future estimates).
- Software: dfeols version, git commit and whether the tree was dirty; Python; versions of
  numpy, scipy, polars, pyarrow, pandas, dask, distributed, dask-jobqueue; BLAS library and thread
  settings (`OMP_NUM_THREADS`, `POLARS_MAX_THREADS`, ...).
- Machine: scheduler (slurm/local), job id, partition, account, coordinator host, workers used
  (count, cores, memory), the resolved profile.
- Invocation: user, working directory, command line, script path and its sha256, session id.

### `replay`

The call that reproduces the result, with every argument explicit:

```json
{"function": "dfeols.fit",
 "dataset": {"open": "/scratch/ted/dfeols/shards/parsons2011_v1_by_batter",
             "meta_sha256": "..."},
 "args": {"formula": "...", "cluster": "pitcher", "vcov": ["CRV1", "hetero"],
          "fe_tol": 1e-10, "drop_singletons": true, "...": "..."},
 "requires": {"dfeols": "0.8.0", "commit": "..."}}
```

`dfeols replay result.json` reruns it and reports differences against the stored values.

## Questions for Ted

1. **Identifying details** (user, host, paths, job ids) are useful for reconstruction but are
   private. Always include them, or include them by default with a `redact=True` option for files
   meant to be shared?
2. **FE coefficients, residuals and fitted values**: never stored, stored on request in the
   side Parquet file, or always for small models?
3. **Several outcomes in one fit** (dfeols supports `y1 + y2 ~ x`): one file per outcome (simpler
   to compare) or one file with a block per outcome?
4. **Validation block**: write pyfixest/reghdfe reference values and differences into the
   result when a run is a validation run, or keep those in the replication harness only?
5. **Units**: anything else you routinely look for when chasing a discrepancy that isn't above?
