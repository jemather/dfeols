# Design: the `estimation` format (`estimation/0.1`)

Status: draft 3 (2026-10-24), for discussion. Draft 1 was the dfeols-only `dfeols-result/1`
(b903bc1); draft 2 made it general (faddceb, as `result-format.md`). Draft 3 follows Ted's
answers on naming, validation, attachments, identifying details and chained extensions.

## Purpose

A file format for recording one estimation run: what was estimated, what came out, and how
and where it ran. It works for any software and any kind of model: dfeols, fixest, pyfixest,
reghdfe, base R `lm`, statsmodels, a random forest. dfeols is the first producer and the most
detailed one, but nothing in the core is dfeols-specific.

It has to serve three uses:

1. Reading and reporting the results (tables, plots, modelsummary).
2. Explaining a discrepancy between two runs or two tools *without rerunning*: comparing two
   records should show whether the rows, options, small-sample correction, convergence,
   software, environment or data differ.
3. Reproducing a run with confidence.

The file stays **small**: results and metadata, never per-row data. Large outputs are linked,
not embedded.

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

## Decisions (Ted, 2026-10-24)

- Name: **`estimation`**. Schema id `estimation/0.1` until settled, then `estimation/1`.
- General, not dfeols-only: a small core every tool can fill, with room for model-specific and
  tool-specific detail. The core sections are Ted's sketch: `meta`, `metrics`, `parameters`,
  `structures`.
- `specification` and `provenance` are top-level sections that writers include by default.
- **One record per run.** A run is one estimation call at one time. An "all results"
  collection is an export built from records, not the source of truth.
- **Provenance is open-ended**: as much detail as the producer can give. For base R that is OS,
  R version and loaded packages. For dfeols it is also launch parameters, the SLURM command,
  workers and the Dask environment.
- **Identifying details**: writers include them by default, but the format never requires
  them, and they must be easy to remove.
- **Attachments are links**, never embedded: path, timestamp, hash. FE coefficients, residuals
  and fitted values are written only on request.
- **Validation is optional**: a convenience for checking an estimator against other software
  (expected for a custom or untested estimator, unusual otherwise), in open-ended styles.
- **Extensions can be chained**: each tool that handles a record can add its own diagnostic
  block.

## Format: JSON, with JSON Schema

Recommended over XML:

- R (jsonlite), Python (json), Julia, JavaScript and `jq` read and write it natively. Stata can
  use its Python integration or a small ado-file.
- It maps directly onto the dict/list structure of the sketch, and broom's tables map onto it.
- JSON Schema (2020-12) gives validation, and lets profiles (below) add rules on top of the core.
- XML's advantage is namespaces. Namespaced keys (`extensions.dfeols`) give the same thing.

Encoding rules:

- UTF-8, one JSON object per file. File name `<name>.estimation.json`.
- Floats with round-trip precision (17 significant digits); non-finite values as the strings
  `"NaN"`, `"Inf"`, `"-Inf"`.
- **Absent** key = this producer doesn't record it. **`null`** = applies but wasn't computed.
- Times in UTC, ISO 8601, with the local zone recorded separately.
- Values are **resolved, not as typed**: every option with its effective value, defaults
  included, so a later change of default can't change what a record means.

## Structure

Each section is required, expected, or optional. A record missing a **required** section is
invalid. A record missing an **expected** section is valid, but validators warn and writers
include it by default.

```text
schema         required   "estimation/0.1"
meta           required   identity and type of the run          (sketch: meta)
parameters     required   list of rows, one per parameter        (sketch: parameters)
metrics        expected   flat dict of scalar results           (sketch: metrics)
specification  expected   what was asked for: call, formula, options, sample
provenance     expected   software, environment, execution, data
structures     optional   named complex objects: vcov, FEs, ... (sketch: structures)
validation     optional   other software's version of this model; comparisons
attachments    optional   links to large side files
extensions     optional   blocks added by tools, one per namespace
history        optional   every tool that created, extended or redacted the record
```

A base R `lm` record and a dfeols record have the same shape. The dfeols one is just fuller.

### `meta`

| Field | Content |
|---|---|
| `id` | UUID of this record |
| `parent` | id of the record this one derives from (same model, new vcov), or absent |
| `created` | when the record was written |
| `status` | `ok`, `warning` or `failed` (a failed run still writes what it knows) |
| `messages` | list of `{level, code, message, source}`: collinear terms dropped, slow convergence, few clusters, singletons |
| `estimator` | `{family, method, ...}`, e.g. `{family: "linear", method: "ols"}`, `{family: "glm", method: "mle", distribution: "binomial", link: "logit"}`, `{family: "ensemble", method: "random_forest"}` |
| `responses` | the dependent variable(s) |
| `profiles` | profiles the record claims to follow, e.g. `["linear-fe/0.1", "dfeols/0.1"]` |
| `title`, `tags`, `notes` | user labels and free text (`"Table 3 col. 9"`) |
| `producer` | the code that wrote the record (`{name, version}`), when not the estimating software (e.g. a converter that reads Stata `e()`) |

`estimator.family` and `method` use a recommended vocabulary (listed in the schema), but any
string is valid, so a new model type doesn't need a format change.

### `specification` (expected)

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

### `metrics` (expected)

A flat dict of named scalars, as in the sketch. `nobs` is universal; everything else depends
on the estimator (`r2`, `r2_within`, `aic`, `log_lik`, `auc`, `mse`). Names follow broom
`glance` where one exists. The schema keeps a registry of recommended names and their
definitions, so `r2_within` means the same thing in any two records. Unregistered names are
allowed.

A metric can be a plain number, or `{value, definition, ...}` when the producer wants to say
exactly what it computed (e.g. which `df` an F test used).

### `parameters` (required)

A list of rows, as in the sketch. `term` is required; other columns depend on the estimator:

- Inference: broom `tidy` names: `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`,
  `conf_high` (from the default vcov; `conf_level` in `metrics`).
- ML: `importance`, `rank`, and so on.
- Multi-part models: `response` (which outcome), `component` (e.g. `first_stage`,
  `zero_inflation`, `random`), `group` for random-effect terms.
- `dropped`: `null`, `"collinear"`, `"reference"`, kept as rows so the term list is complete.

Rows are the model's reported parameters. Per-level estimates (FE coefficients, BLUPs) are not
parameters for this purpose; they go in an attachment.

Because `term` names differ by tool (fixest `x::2`, Stata `2.x`), the record keeps the producer's
own name. Validation records how names were matched (below).

### `structures` (optional)

Named complex objects, each with a `kind` so a reader can decode it without knowing the
estimator:

```json
"vcov": [{"kind": "matrix", "labels": ["x1", "x2"], "data": [[...], [...]],
          "type": "CRV1", "cluster": ["pitcher"], "n_clusters": {"pitcher": 1172},
          "ssc": {"formula": "G/(G-1) * (N-1)/(N-K)", "factors": {...}, "K_counts": "...",
                  "convention": "fixest"},
          "df_t": 1171, "default": true}]
```

Defined kinds: `matrix`, `table` (columns + rows), `tree`, `series`, `object` (free nested dict),
and `link` (the structure lives in an attachment; see below). Registered names, with their
expected shapes defined by profiles:

- `vcov`: a list, one per vcov computed (iid, HC1, CRV1, ...), each with its correction written
  out in full. The default one also feeds `parameters`.
- `fixed_effects`: per FE: levels in the data and in the sample, redundant levels and how they
  were found, nesting (in clusters, in other FEs). Counts only, never the level values.
- `random_effects`: variance components.
- `numerics`: convergence (algorithm, tolerance, iterations, converged), solver, rank,
  condition number, residual orthogonality check.
- `hyperparameters`: for ML estimators; tuning results if any.
- `first_stage`: IV diagnostics (first-stage F, Kleibergen-Paap, partial R2).

### `provenance` (expected)

Everything about how and where the run happened. Each sub-block is a list or dict that the
producer fills as far as it can. **This is where a discrepancy is traced**, so it is
deliberately open-ended.

- `software`: list of components, each `{name, version, role, source, attributes}`.
  `role` is `estimator` (dfeols, fixest), `language` (R, Python), `dependency` (numpy, polars) or
  `system` (BLAS). `source` is CRAN, PyPI, conda, or a git commit with a dirty flag. For base R:
  R plus attached packages from `sessionInfo()`. For dfeols: dfeols at commit X, Python,
  numpy, scipy, polars, pyarrow, dask, distributed, dask-jobqueue and BLAS.
- `environment`: OS, kernel, CPU model and cores, memory, locale, timezone, thread settings,
  and **an allowlist of environment variables** (`OMP_NUM_THREADS`, `POLARS_MAX_THREADS`, ...).
  Never a full dump: environments carry tokens.
- `execution`: `started`, `finished`, duration and timing by stage; `invocation` (command
  line, script path and sha256, working directory); `scheduler` (`{type: "slurm", job_id,
  submit_command, partition, account, resources, nodes}`); `workers` (count, cores, memory, peak
  use); launch parameters (the dfeols profile, fully resolved); `session` id if from a session.
- `agent`: user and host.
- `data`: list of inputs, each `{role, name, version, location, checksum, rows, columns,
  source, license}`. `checksum` is `{algorithm, value, of}`, where `of` says what was hashed (a
  file, a manifest such as `MANIFEST.sha256`, a data frame's content).

### Identifying details

The format never requires identifying details; every such field is optional. Writers include
them by default, because they matter for tracing a discrepancy, and make them easy to remove:

- **The schema marks them.** Each identifying field carries the annotation
  `"x-identifying": true` in the JSON Schema: `provenance.agent`, hosts, paths, job ids,
  account, command lines, working directory and data locations. An extension's schema marks its
  own identifying fields the same way. A redactor therefore needs only the schemas, not
  knowledge of each tool.
- **A standalone redactor**: `estimation redact in.json -o out.json [--hash]`, a small
  package-independent tool (also `dfeols` `redact=True` when writing). It either removes each
  marked value or, with `--hash`, replaces it with a salted hash, so two records from the same
  host still compare equal. Content hashes (data checksums, script sha256) are kept, since they
  identify content, not people.
- The redactor appends a `history` entry (`action: "redacted"`, method, fields changed), so a
  reader knows what is missing and why.

### `validation` (optional)

Most records won't have one: re-running an `lm` in Stata is not the norm. It is for when the
estimator itself is in question: a custom or untested estimator, a new release, a replication.
It has two parts.

**Equivalents**: the same model in other software, each in a *style*:

| Style | Content |
|---|---|
| `code` | `{software, language, code}`: e.g. `reghdfe y x, absorb(fe) vce(cluster c)` |
| `description` | prose: how the reference was produced |
| `record` | another `estimation` record (by path + sha256 + id, or embedded whole) |
| `published` | `{citation, table, column, values}`: numbers from a paper |

The style list is open: a writer can use a new style (e.g. `notebook`, `container`), and
readers skip styles they don't know. A `record` reference carries its own provenance, so a
pyfixest reference says which pyfixest, which Python and which machine.

**Comparisons**: what was compared against which equivalent, and how close it came:

```json
"comparisons": [
  {"reference": "pyfixest", "quantities": ["estimate", "std_error", "nobs"],
   "term_map": {"x::2": "x[T.2]"}, "tolerance": {"rel": 1e-6},
   "max_abs_diff": {...}, "max_rel_diff": {...}, "pass": true,
   "details": [{"term": "x1", "quantity": "estimate", "value": ..., "reference": ...,
                "diff": ...}]}],
"verdict": "pass"
```

An equivalent can be listed without a comparison (the code to run, not yet run).

### `attachments` (optional): links, not content

Large outputs (FE coefficients, residuals, fitted values, a vcov with thousands of terms,
predictions) live in side files. The record holds only a link:

```json
{"id": "fe_coefs", "role": "fe_coefficients", "path": "fe_coefs.parquet",
 "media_type": "application/vnd.apache.parquet", "created": "2026-10-24T15:02:11Z",
 "sha256": "...", "bytes": 81234567, "rows": 1838676, "columns": ["fe", "level", "estimate"]}
```

- `path` is relative to the record (or a URI). The record stays valid if the file is missing;
  the hash says whether a file found later is the right one.
- A structure can point to an attachment: `{"kind": "link", "attachment": "vcov_full"}`.
- Parquet is recommended for tables. Producers write attachments only when asked.

To keep records small, the format **encourages links in three ways**:

1. **The spec says so**: per-observation and per-level output MUST be an attachment, and the
   `parameters` and `structures` definitions exclude it.
2. **Validators warn**: above 1 MB for a whole record, or 10,000 elements in a single
   array, the validator warns and names the section to move to an attachment. Records above
   16 MB fail validation.
3. **Reference writers spill automatically**: the dfeols writer, and any helper library, move
   a structure above the threshold into an attachment and leave a `link`. The easy path is the
   small one.

### `extensions` (optional): chained diagnostic blocks

Any tool that handles a record can add a block under its own namespace. The estimator can
(dfeols's shard and demeaning detail), and so can tools later in a chain: a table exporter, a
replication harness, a cluster accounting tool, a redactor. Each block:

```json
"extensions": {
  "dfeols":   {"schema": "dfeols/0.1", "by": {"name": "dfeols", "version": "0.8.0"},
               "added": "2026-10-24T15:02:11Z", ...},
  "harness":  {"schema": "dfeols-harness/0.1", "by": {...}, "added": "...",
               "reference_file": "reference.json", ...}
}
```

Rules:

- A namespace belongs to the tool that writes it. A tool **adds or replaces only its own
  block**, never another tool's block or the core sections.
- A tool that changes results (new vcov, new sample) writes a **new record** with `parent`, not
  a block.
- Each tool appends to `history`: `{when, by: {name, version}, action: "created" | "extended" |
  "redacted" | "validated", sections: [...]}`. The chain of tools that touched a record is
  therefore explicit and in order.
- Optional integrity check: `meta.core_digest` is the sha256 of the canonical JSON (RFC 8785)
  of the core sections, written at creation. Any later tool can verify that the results it is
  annotating haven't changed.
- The core validator ignores block content; each block's `schema` names the schema that
  validates it, and readers skip blocks they don't know.

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

What only dfeols has (most of it from draft 1):

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
  `dfeols replay record.estimation.json` reruns it and compares against the record, writing the
  result as a `validation` comparison.

## Collections ("all results")

`estimation-collection/0.1`: a header (`id`, `created`, `title`, members) and the member records,
embedded whole or as JSON Lines with one record per line. It is an export (e.g. at the end of a
session, or for a paper's tables), built from records and never edited by itself. A flat
`parameters` table across records (Parquet/CSV, with record id, response and term) is a
convenience export.

## Example: base R `lm`, minimal

```json
{"schema": "estimation/0.1",
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

(Numbers illustrative.) A dfeols record has the same top level with every expected block
filled, plus `extensions.dfeols` and `history`.

## Repository layout (proposed)

- `docs/spec/estimation.md`: the specification (this document, once settled).
- `schema/estimation-0.1.json` and `schema/profiles/*.json`: JSON Schemas, with
  `x-identifying` annotations.
- `src/dfeols/estimation.py`: writer, reader, redactor, validator, attachment spilling.
- Tests validate every record dfeols writes, round-trip a hand-written base R example, and
  check that redaction removes every marked field.
- Later, if the format proves useful beyond dfeols: its own repository and package (the
  `estimation` CLI), plus small writers for R (from `lm`/fixest objects) and Stata (from `e()`).

## Questions for Ted

1. **Profiles.** Is the profile mechanism the right way to get model-specific rules, and are
   the first four the right set?
2. **Term names across tools.** Keep each tool's own names and match them in validation (as
   drafted), or also store a canonical name per term?
3. **Data checksums for in-memory data** (an R data frame): hash the content, or leave it to
   the producer?
4. **Size thresholds**: 1 MB warning, 10,000-element arrays, 16 MB hard limit. Right order of
   magnitude?
5. **Core digest**: worth having from the start, or later?
