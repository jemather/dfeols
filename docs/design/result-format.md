# Design: estimation records (working name `estimation-record/0.1`)

Status: draft 2 (2026-10-24), for discussion. Replaces draft 1 (dfeols-only `dfeols-result/1`,
commit b903bc1) and the broom-only sketch in `sessions.md`. Draft 2 follows Ted's answers and
his `ModelEstimationResult` sketch (meta / metrics / parameters / structures).

## Purpose

A file format for recording one estimation run (what was estimated, what came out, and
everything about how and where it ran) that works for any software and any kind of model:
dfeols, fixest, pyfixest, reghdfe, base R `lm`, statsmodels, a random forest. dfeols is the
first producer and the most detailed one, but nothing in the core is dfeols-specific.

It has to serve three uses:

1. Reading and reporting the results (tables, plots, modelsummary).
2. Explaining a discrepancy between two runs or two tools *without rerunning*: comparing two
   records should show whether the rows, options, small-sample correction, convergence,
   software, environment or data differ.
3. Reproducing a run with confidence.

No existing format covers this. The nearest pieces:

| Existing | Covers | Missing |
|---|---|---|
| broom `tidy`/`glance` (R) | parameter table and fit statistics names; read by modelsummary | not a file format; no vcov, provenance, environment |
| PMML (XML, DMG) | scoring a fitted model in another system | inference (SEs, vcov, tests), provenance; dated, XML-heavy |
| R `sessionInfo()`, `pip freeze`, conda env | software environment | nothing about the model |
| W3C PROV, RO-Crate, CodeMeta | generic provenance and software metadata | nothing statistical |
| Stata `e()`/`r()`, fixest objects | full detail, one tool | tool-specific, binary or in-memory |

So the format borrows names where they exist (broom for parameters and metrics, CodeMeta-like
fields for software, PROV's idea of entity/activity/agent for provenance) and defines the rest.

## Decisions so far (Ted, 2026-10-24)

- General, not dfeols-only: a small core every tool can fill, with room for model-specific and
  tool-specific detail.
- **One record per run.** A run is one estimation call at one time; a call that fits several
  outcomes (`y1 + y2 ~ x`) gives one record whose parameter rows carry `response`. An "all
  results" collection is an export built from records, not the source of truth.
- **Provenance is open-ended and user-defined**: as much detail as the producer can give. For
  base R that is OS, R version and loaded packages; for dfeols it is also launch parameters,
  the SLURM command, workers and the Dask environment.
- **Identifying details included by default**, with a redaction option for shared files.
- **FE coefficients, residuals, fitted values: export only**, never by default.
- **A validation block**: reference values from other software or a paper, and the differences.

## Format choice: JSON, with JSON Schema

Recommended over XML:

- Read and written natively by R (jsonlite), Python (json), Julia, JavaScript and `jq`; Stata
  through its Python integration or a small ado-file.
- Maps directly onto the dict/list structure of the sketch, and broom's tables map onto it.
- JSON Schema (2020-12) gives validation, and lets profiles (below) add rules on top of the core.
- XML's advantage is namespaces; namespaced keys (`extensions.dfeols`) give the same thing.

Encoding rules:

- UTF-8, one JSON object per file, extension `.json` (proposed: `<id>.er.json`).
- Floats with round-trip precision (17 significant digits); non-finite values as the strings
  `"NaN"`, `"Inf"`, `"-Inf"`.
- **Absent** key = this producer doesn't record it. **`null`** = applies but wasn't computed.
- Times in UTC, ISO 8601, with the local zone recorded separately.
- Values are **resolved, not as typed**: every option with its effective value, defaults
  included, so a later change of default can't change what a record means.

## Structure

```text
schema         "estimation-record/0.1"
meta           identity and type of the run          (sketch: meta)
specification  what was asked for: formula, options, sample rules
metrics        flat dict of scalar results           (sketch: metrics)
parameters     list of rows, one per parameter        (sketch: parameters)
structures     named complex objects: vcov, FEs, ... (sketch: structures)
provenance     software, environment, execution, data
validation     reference values and differences      (optional)
attachments    exported side files: FE coefs, ...    (optional)
redaction      what was removed and how              (only when redacted)
extensions     tool-specific blocks, namespaced      (optional)
```

Only `schema`, `meta`, `parameters` and `provenance.software` are required. Everything else is
filled as far as the producer can. A base R `lm` record and a dfeols record have the same
shape; the dfeols one is just much fuller.

### `meta`

| Field | Content |
|---|---|
| `id` | UUID of this record |
| `parent` | id of the record this one refits (e.g. same model, new vcov), or absent |
| `created` | when the record was written |
| `status` | `ok`, `warning` or `failed` (a failed run still writes what it knows) |
| `messages` | list of `{level, code, message, source}`: collinear terms dropped, slow convergence, few clusters, singletons |
| `estimator` | `{family, method, ...}`, e.g. `{family: "linear", method: "ols"}`, `{family: "glm", method: "mle", distribution: "binomial", link: "logit"}`, `{family: "ensemble", method: "random_forest"}` |
| `responses` | the dependent variable(s) |
| `profiles` | profiles the record claims to follow, e.g. `["linear-fe/0.1", "dfeols/0.1"]` |
| `title`, `tags`, `notes` | user labels and free text (`"Table 3 col. 9"`) |
| `producer` | the code that wrote the record (`{name, version}`), when not the estimating software (e.g. a converter that reads Stata `e()`) |

`meta.estimator.family` and `method` use a recommended vocabulary (listed in the schema), but
any string is valid: a new model type doesn't need a format change.

### `specification`

What was asked for, in the estimating software's own terms, fully resolved:

- `call`: the command as written (`feols(y ~ x | fe, data)`, `reghdfe y x, absorb(fe)`).
- `formula`: formula string and syntax (`fixest`, `R`, `patsy`, `stata`).
- `terms`: expanded regressors in model order; `factor_terms` with levels and reference level.
- `weights`: `{variable, type}` (analytic, frequency, probability) or `null`.
- `fixed_effects`, `instruments`, `clusters` as specified.
- `sample`: the restriction as written, plus `rows_input` and the drop counts by reason
  (missing per variable, singletons, zero weights, other). `nobs` lives in `metrics`.
- `options`: every option with its effective value (tolerances, algorithms, seeds,
  hyperparameters for ML).

### `metrics`

A flat dict of named scalars, as in the sketch: `nobs` is universal, everything else depends on
the estimator (`r2`, `r2_within`, `aic`, `log_lik`, `auc`, `mse`). Names follow broom `glance`
where one exists; the schema keeps a registry of recommended names and their definitions, so
two records' `r2_within` mean the same thing. Unregistered names are allowed.

A metric can be a plain number, or `{value, definition, ...}` when the producer wants to say
exactly what it computed (e.g. which `df` an F test used).

### `parameters`

A list of rows, as in the sketch. `term` is required; other columns depend on the estimator:

- Inference: broom `tidy` names: `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`,
  `conf_high` (from the default vcov; `conf_level` in `metrics`).
- ML: `importance`, `rank`, and so on.
- Multi-part models: `response` (which outcome), `component` (e.g. `first_stage`,
  `zero_inflation`, `random`), `group` for random-effect terms.
- `dropped`: `null`, `"collinear"`, `"reference"`, kept as rows so the term list is complete.

Because `term` names differ by tool (fixest `x::2`, Stata `2.x`), the record keeps the producer's
own name. Validation records how names were matched (below).

### `structures`

Named complex objects, each with a `kind` so a reader knows how to decode it without knowing
the estimator:

```json
"vcov": [{"kind": "matrix", "labels": ["x1", "x2"], "data": [[...], [...]],
          "type": "CRV1", "cluster": ["pitcher"], "n_clusters": {"pitcher": 1172},
          "ssc": {"formula": "G/(G-1) * (N-1)/(N-K)", "factors": {...}, "K_counts": "...",
                  "convention": "fixest"},
          "df_t": 1171, "default": true}]
```

Defined kinds: `matrix`, `table` (columns + rows), `tree`, `series`, `object` (free nested dict).
Registered names, with their expected shapes defined by profiles:

- `vcov`: a list, one per vcov computed (iid, HC1, CRV1, ...), each with its correction written
  out in full. The default one also feeds `parameters`.
- `fixed_effects`: per FE: levels in the data and in the sample, redundant levels and how they
  were found, nesting (in clusters, in other FEs).
- `random_effects`: variance components.
- `numerics`: convergence (algorithm, tolerance, iterations, converged), solver, rank,
  condition number, residual orthogonality check.
- `hyperparameters`: for ML estimators; tuning results if any.
- `first_stage`: IV diagnostics (first-stage F, Kleibergen-Paap, partial R2).

Large structures (a vcov with 10,000 factor terms) can move to an attachment and leave a
pointer.

### `provenance`

Everything about how and where the run happened. Each sub-block is a list or dict that the
producer fills as far as it can. **This is where a discrepancy is traced**, so it is
deliberately open-ended.

- `software`: list of components, each `{name, version, role, source, attributes}`.
  `role` is `estimator` (dfeols, fixest), `language` (R, Python), `dependency` (numpy, polars),
  `system` (BLAS). `source` is CRAN, PyPI, conda, or a git commit with a dirty flag. Base R: R plus
  attached packages from `sessionInfo()`. dfeols: dfeols at commit X, Python, numpy, scipy,
  polars, pyarrow, dask, distributed, dask-jobqueue, BLAS.
- `environment`: OS, kernel, CPU model and cores, memory, locale, timezone, thread settings,
  and **an allowlist of environment variables** (`OMP_NUM_THREADS`, `POLARS_MAX_THREADS`, ...).
  Never a full dump: environments carry tokens.
- `execution`: `started`, `finished`, duration and timing by stage; `invocation` (command
  line, script path and sha256, working directory); `scheduler` (`{type: "slurm", job_id,
  submit_command, partition, account, resources, nodes}`); `workers` (count, cores, memory, peak
  use); launch parameters (dfeols profile, fully resolved); `session` id if from a session.
- `agent`: user and host.
- `data`: list of inputs, each `{role, name, version, location, checksum, rows, columns,
  source, license}`. `checksum` is `{algorithm, value, of}`; `of` says what was hashed (a file,
  a manifest such as `MANIFEST.sha256`, a data frame's content).

### `validation` (optional)

For validation runs: what this result was checked against, and how close it came. References
can be other records (any software) or published values:

```json
"validation": {
  "references": [
    {"id": "pyfixest", "record": {"path": "ref/abc.er.json", "sha256": "...", "id": "..."}},
    {"id": "aer_table3", "published": {"citation": "Parsons et al. (2011) AER", "table": "3",
                                       "column": "9"}, "values": {"x1": {"estimate": 0.0047}}}
  ],
  "comparisons": [
    {"reference": "pyfixest", "quantities": ["estimate", "std_error", "nobs"],
     "term_map": {"x::2": "x[T.2]"},
     "tolerance": {"rel": 1e-6}, "max_abs_diff": {...}, "max_rel_diff": {...},
     "pass": true, "details": [{"term": "x1", "quantity": "estimate", "value": ...,
                                "reference": ..., "diff": ...}]}
  ],
  "verdict": "pass"
}
```

A reference record can also be embedded whole instead of pointed to, so a validation file
travels on its own. It carries its own provenance, so a pyfixest reference says which pyfixest,
which Python and which machine.

### `attachments` (optional, never by default)

Side files with per-row or per-level output: `{role, path, media_type, sha256, rows, columns}`.
Roles: `fe_coefficients`, `residuals`, `fitted`, `large_structure`, or anything else. Parquet
is recommended for tables. Paths are relative to the record. Producers write attachments only
when asked (`export=["fe_coefficients"]`).

### `redaction` (only when redacted)

The schema marks identifying fields: `provenance.agent`, hosts, paths, job ids, account,
command lines and data locations. A redacted record:

- replaces each such value with `null`, or with a salted hash, so two records from the same
  host still compare equal without revealing it;
- keeps content hashes (data checksums, script sha256), which identify content, not people;
- lists what it changed: `{applied: true, method: "remove" | "hash", fields: [...]}`.

Identifying details stay in by default; `redact=True` (dfeols) or a standalone tool produces a
shareable copy.

### `extensions`

Tool-specific blocks, namespaced and versioned. The core ignores their content; each tool's
own schema validates them:

```json
"extensions": {"dfeols": {"schema": "dfeols/0.1", ...}}
```

## Profiles

A profile is a named, versioned set of rules on top of the core: which `metrics`, `parameters`
columns and `structures` a record of that kind must have, and their shapes. A record lists the
profiles it follows in `meta.profiles`; a validator checks the core schema plus each profile.
This gives model-specific requirements without a new format per model.

Proposed first profiles:

| Profile | Requires |
|---|---|
| `inference/0.1` | `parameters` with estimate, std_error, p_value; a default `vcov` with `type` and `ssc`; `metrics.nobs`, `df_residual` |
| `linear-fe/0.1` | `inference` + `structures.fixed_effects`; `r2_within`; singleton handling in `specification.sample` |
| `iv/0.1` | `inference` + endogenous/instruments in `specification`; `structures.first_stage` |
| `dfeols/0.1` | `linear-fe` + `extensions.dfeols` |

Later, as needed: `glm/0.1` (deviance, log-likelihood, family/link), `mixed/0.1` (random
effects), `ml/0.1` (hyperparameters, importance, CV metrics).

## The dfeols extension (`dfeols/0.1`)

What only dfeols has. Most of it came from draft 1:

- `shards`: shard folder, prep version, partition key, target rows, shard count, a hash of
  `dfeols_meta.json`. Results must not depend on the partition, but it is the first thing to
  check when they do.
- `row_key`: a hash of the indices of the rows used, combined shard by shard, so two runs can
  be checked for an identical sample without storing rows.
- `absorption`: per FE, `local` / `crossing_dense` / `crossing_sparse`, `nested_in_partition`.
- `demeaning`: per-shard iterations (max, median), the worst shard's final change, round 1/2
  timing; the persisted demeaned-data key if from a session.
- `dask`: scheduler address, worker specs, resolved profile.
- `estimate`: the time estimate shown at submission (to calibrate future estimates).
- `replay`: the `dfeols.fit` call with every argument explicit, and the required dfeols commit.
  `dfeols replay record.json` reruns it and compares against the record, using the
  `validation` block.

## Collections ("all results")

`estimation-collection/0.1`: a header (`id`, `created`, `title`, members) and the member records
embedded whole, or as JSON Lines with one record per line. It is an export (e.g. at the end of
a session, or for a paper's tables), built from records and never edited by itself. A flat
`parameters` table across records (Parquet/CSV, with record id, response, term) is a convenience
export.

## Example: base R `lm`, minimal

```json
{"schema": "estimation-record/0.1",
 "meta": {"id": "6f1c...", "created": "2026-10-24T15:02:11Z", "status": "ok",
          "estimator": {"family": "linear", "method": "ols"}, "responses": ["mpg"],
          "profiles": ["inference/0.1"]},
 "specification": {"call": "lm(mpg ~ wt, data = mtcars)",
                   "formula": {"text": "mpg ~ wt", "syntax": "R"}},
 "metrics": {"nobs": 32, "df_residual": 30, "r2": 0.7528, "sigma": 3.046, "conf_level": 0.95},
 "parameters": [
   {"term": "(Intercept)", "estimate": 37.285, "std_error": 1.8776, "statistic": 19.858,
    "p_value": 8.24e-19},
   {"term": "wt", "estimate": -5.3445, "std_error": 0.5591, "statistic": -9.559,
    "p_value": 1.29e-10}],
 "structures": {"vcov": [{"kind": "matrix", "labels": ["(Intercept)", "wt"],
                          "data": [[3.525, -1.0], [-1.0, 0.3126]], "type": "iid",
                          "default": true}]},
 "provenance": {"software": [{"name": "R", "version": "4.5.1", "role": "language"},
                             {"name": "stats", "version": "4.5.1", "role": "estimator"}],
                "environment": {"os": "Ubuntu 24.04", "blas": "OpenBLAS 0.3.26"}}}
```

(Numbers illustrative.) A dfeols record has the same top level with every block filled and
`extensions.dfeols`.

## Repository layout (proposed)

- `docs/spec/estimation-record.md`: the specification (this document, once settled).
- `schema/estimation-record-0.1.json` and `schema/profiles/*.json`: JSON Schemas.
- `src/dfeols/record.py`: writer, reader, redaction, validation against the schema.
- Tests validate every record dfeols writes, and round-trip a hand-written base R example.
- Later, if the format proves useful beyond dfeols: its own repository, plus small writers for
  R (from `lm`/fixest objects) and Stata (from `e()`).

## Questions for Ted

1. **Name.** `estimation-record` is a placeholder. A neutral name suits a format meant to outlive
   dfeols.
2. **Section names.** Draft 2 keeps the sketch's `meta`/`metrics`/`parameters`/`structures` and
   adds `specification`, `provenance`, `validation`, `attachments`, `redaction`, `extensions`.
   Should specification and provenance sit under `meta` instead?
3. **Profiles.** Is the profile mechanism the right way to get model-specific rules, and are
   the first four the right set?
4. **Term names across tools.** Keep each tool's own names and match them in validation (as
   drafted), or also store a canonical name per term?
5. **Data checksums for in-memory data** (an R data frame): hash the content, or leave it to
   the producer?
