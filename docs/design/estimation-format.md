# Design: the `estimation` format (`estimation/0.1`)

Status: draft 6 (2026-10-24), all questions settled (`ssc` last); ready for the schema. Draft 6
settles term names (verbatim, plus an optional R-style canonical form), the `linear-fe`
required set and the core digest (on by default). Draft 5 added canonical names, flagged disagreements and collection kinds. Draft 1 was the dfeols-only `dfeols-result/1`
(b903bc1); draft 2 made it general (faddceb, as `result-format.md`); draft 3 (463110d) settled
naming, validation, attachments, identifying details and chained extensions. Draft 4 adds what
other tools record (SPSS, Stata, R, Python, lock files, SBOMs, SLURM), the `publication` block
(citations, table/column locators, AsCollected/ResearchBox links), profiles built from the union
of what popular tools save, and session metadata for collections.

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
| Stata `e()`/`r()`/`c()`, `estimates save` (`.ster`), `collect` | full detail of one run; `c()` holds the environment; `collect` groups results with tags | Stata-only, binary `.ster`; nothing outside Stata reads it |
| SPSS OMS (OXML), the "Notes" table, `OUTFILE=COVB` | OMS routes any output table to XML/SAV; every procedure's Notes table records time created, input file, active filter/weight/split, rows, missing-value handling, the syntax and processor/elapsed time | OXML describes *display tables* (pivot dimensions and cells), not a model; the vcov goes to a separate `.sav`; no software or hardware detail |
| fixest / pyfixest / linearmodels result objects | full detail, one tool | in-memory, tool-specific |
| MLflow run records | params, metrics, tags, artifacts by link, source commit | ML-oriented; no inference, vcov or data identity |
| renv.lock, pylock.toml (PEP 751), conda lock, `pip freeze` | exact package sets with sources and hashes | nothing about the model or run |
| CycloneDX / SPDX SBOMs, Package URLs (purl) | software components with a universal identifier (`pkg:cran/fixest@0.12.1`, `pkg:pypi/numpy@2.1.0`) and hashes | supply-chain focus; heavy |
| RO-Crate Process Run Crate | one execution of a tool: inputs, outputs, start/end, agent, instrument | generic; nothing statistical |
| SLURM `sacct`/`scontrol` | job id, partition, account, nodes, allocated TRES, submit line, elapsed, MaxRSS, exit state | scheduler only |
| AsCollected, ResearchBox, AsPredicted (Wharton CredLab) | provenance of a paper's data; data/code boxes with file ids; pre-registrations; each with a stable short URL | the estimation results themselves |

So the format borrows names and identifiers where they exist and defines the rest:

- broom for parameters and metrics;
- purl to identify software;
- lock-file hashes for package sets;
- CSL-JSON (the citation format used by Zotero, pandoc and CrossRef) for citations;
- SLURM's own field names for scheduler data;
- SPSS's Notes table as a checklist of run state worth recording (active filter, weight, split,
  missing-value handling, processor vs elapsed time);
- PROV and RO-Crate for the run as an activity with inputs, outputs and an agent;
- Stata `collect` and MLflow for collections and tags.

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
- **Size thresholds**: warn above 1 MB per record or 10,000 elements in one array; fail above
  16 MB.
- **Publication links**: a record can say where its result appears ("Study 1, Table 2, Column
  1" plus a citation) and link to AsCollected, ResearchBox and similar records. No author-role
  detail: that belongs in AsCollected itself.
- **Profiles** are the way to get model-specific rules. Each is built from the union of what the
  most popular Stata, R and Python tools for that task save.
- **Collections record the session**: duration, environment and system details, like a record
  does for a run.

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
publication    optional   where the result appears: citation, table/column, links
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
- `state`: settings in force from *outside* the call, which change results without appearing
  in it. SPSS's Notes table and Stata's settings are the model here: active filter, weight and
  split file (SPSS); `xtset`/`tsset`, `if`/`in`, `set seed` (Stata); `options()` that change
  estimates (R); user-defined missing-value codes and how they were treated.
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

#### Term names: as the tool reports them, plus an optional canonical form (Ted, 2026-10-24)

`term` is **verbatim**: the name exactly as the producing software reports it, transformations
and all (`log(price)`, `i(year, ref = 2000)::2005`, `2005.year`). That is enough to replicate
the result in that software, so it is the required name, and nothing is reinterpreted.

`canonical` is **optional but encouraged**: a standard form that lets records from different
tools be matched and tabulated together. Writers are encouraged to fill it, and reference
writers do.

```json
{"term": "i(year, ref = 2000)::2005",
 "canonical": {"name": "year2005",
               "factors": [{"variable": "year", "level": "2005", "reference": "2000"}]}}
{"term": "c.price#c.weight",
 "canonical": {"name": "price:weight", "factors": [{"variable": "price"}, {"variable": "weight"}]}}
{"term": "2005.year#c.x",
 "canonical": {"name": "2005.year#c.x",
               "factors": [{"variable": "year", "level": "2005"}, {"variable": "x"}]}}
```

- `factors` is the definition, and matching across tools uses it. `name` is a readable
  rendering.
- **Rendering follows R** (`model.matrix` naming): `(Intercept)`, variable name followed by
  level (`year2005`), `:` between interacted terms. R's concatenation can be ambiguous
  (`year2005` could be variable `year2`, level `005`); `factors` resolves that.
- **Exception: factor × continuous interactions use Stata's form** (`2005.year#c.x`), which is
  clearer than R's (`factor(year)2005:x`).
- Levels and transformations stay as the tool reports them (codes or labels, `log(x)` as
  written). `canonical` may add `level_label` or `transform` when the producer knows them.
- Interaction order is kept as specified in both `term` and `name`; matching on `factors` is
  order-insensitive.
- FE models: no constant row is required. A tool that reports one (reghdfe `_cons`) includes
  it, and `structures.fixed_effects.constant_reported` flags the convention.

How tools spell the same term:

| Term | R `lm` | fixest | Stata | patsy (statsmodels, pyfixest) | canonical `name` |
|---|---|---|---|---|---|
| constant | `(Intercept)` | `(Intercept)` | `_cons` | `Intercept` (statsmodels `const`) | `(Intercept)` |
| factor level | `factor(year)2005` | `year::2005` | `2005.year` | `C(year)[T.2005]` | `year2005` |
| base level | (not shown) | (not shown) | `2000b.year` | (not shown) | row with `dropped: "reference"` |
| interaction | `price:weight` | `price:weight` | `c.price#c.weight` | `price:weight` | `price:weight` |
| factor × continuous | `factor(year)2005:x` | `year::2005:x` | `2005.year#c.x` | `C(year)[T.2005]:x` | `2005.year#c.x` |
| omitted (collinear) | `NA` estimate | dropped, in `collin.var` | `o.x` | dropped | row with `dropped: "collinear"` |

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

- `software`: list of components, each `{name, version, purl, role, source, hash, attributes}`.
  - `purl` (Package URL) identifies a component the same way across ecosystems:
    `pkg:cran/fixest@0.12.1`, `pkg:pypi/numpy@2.1.0`, `pkg:github/jemather/dfeols@ce05eec`,
    `pkg:conda/...`. Stata and SPSS have no purl type, so they use `pkg:generic/stata@19.5`
    plus `attributes`.
  - `role`: `estimator` (dfeols, fixest), `language` (R, Python, Stata), `dependency` (numpy,
    polars), `system` (BLAS, OpenMP, MPI), `container`.
  - `source`: CRAN, PyPI, conda, SSC, or a git commit with a dirty flag.
  - `lock`: optional pointer to a full lock file (`renv.lock`, `pylock.toml`, `conda-lock.yml`,
    CycloneDX SBOM), linked like an attachment by path + sha256. The record lists only the
    components that matter; the lock file has the rest.
  - `container`: image reference and digest (Docker/OCI, Apptainer `.sif` sha256) when the run
    was containerised.
- `environment`: OS, kernel, locale, timezone, and the details below.
  - **Hardware**: CPU model, sockets, cores, threads, instruction-set flags that change
    floating-point results (AVX2, AVX-512, FMA), memory, GPU model and driver.
  - **Numerical libraries and threads**: the BLAS/LAPACK in use, its version, threading layer
    and thread count (Python `threadpoolctl.threadpool_info()`; R `sessionInfo()` BLAS/LAPACK
    and `La_version()`; `numpy.show_config()`).
  - **Random numbers**: generator type, seed, state (R `RNGkind()`, Stata `c(rngtype)`,
    `c(rngstate)`, numpy's bit generator).
  - **Environment variables**: an allowlist only (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
    `POLARS_MAX_THREADS`, ...). Never a full dump: environments carry tokens.
- `execution`:
  - Times: `started`, `finished`, elapsed and **processor time** (SPSS reports both; a large gap
    between them means waiting or contention), timing by stage, peak memory.
  - `invocation`: command line, script path and sha256, working directory, interactive or
    batch (Stata `c(mode)`).
  - `scheduler`: `{type: "slurm", ...}`, using SLURM's own field names: `JobID`, `JobName`,
    `Partition`, `Account`, `QOS`, `NodeList`, `AllocCPUS`, `AllocTRES`, `ReqMem`, `Submit`,
    `Start`, `End`, `Elapsed`, `MaxRSS`, `State`, `ExitCode`, `SubmitLine`, `WorkDir`. Other
    schedulers (PBS, LSF, SGE) use their own names under their own `type`.
  - `workers`: count, cores, memory, peak use.
  - `launch`: launch parameters, fully resolved (the dfeols profile, `dfeols submit`
    arguments).
  - `session`: session id if the run came from a session (see Collections).
- `agent`: user and host.
- `data`: list of inputs, each `{role, name, version, location, checksum, rows, columns,
  source, license, publication}`.
  - `checksum` is `{algorithm, value, of, method}`. `of` says what was hashed (a file, a
    manifest such as `MANIFEST.sha256`, in-memory content). For in-memory content, the format
    defines no single hash (Ted, 2026-10-24); `method` is required and names the procedure (`stata-datasignature`, `R digest::digest(df, "xxhash64")`,
    `pandas.util.hash_pandas_object`). Two checksums are comparable only if `method` matches.
  - `publication`: where the data is published (same block as below: DOI, AsCollected or
    ResearchBox link, openICPSR project).

What each kind of producer can usually fill, as a guide for writers:

| Producer | Software | Environment | Execution |
|---|---|---|---|
| base R | `R.version`, attached and loaded packages (`sessionInfo()`), optionally `renv.lock` | `platform`, `running`, BLAS/LAPACK, `La_version()`, locale, `tzone`, `RNGkind()` | `Sys.time()` around the call, `proc.time()`, `commandArgs()` |
| Stata | `c(stata_version)`, `c(version)`, `c(born_date)`, edition (BE/SE/MP), ado versions (`which`), SSC packages | `c(os)`, `c(osdtl)`, `c(machine_type)`, `c(byteorder)`, `c(processors)`, `c(rngtype)`, `c(rngstate)` | `c(current_date)`, `c(current_time)`, `timer`, `c(mode)`, `c(pwd)`; `e(cmdline)` |
| SPSS | version, edition, fix pack, extension bundles | OS, `SHOW` settings (e.g. `MXCELLS`, `RNG`, `SEED`) | Notes table: Output Created, Syntax, Input data, Filter/Weight/Split, Processor/Elapsed Time |
| Python | interpreter, packages (`importlib.metadata`), optionally `pylock.toml`/conda lock | `platform`, `threadpoolctl`, `os.cpu_count()`, allowlisted variables | `time`, `resource.getrusage`, `sys.argv` |
| dfeols | all of the Python row, plus dfeols commit and dirty flag, Dask stack | per worker as well as the coordinator | SLURM fields, launch profile, workers, per-stage timing |

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

### `publication` (optional)

Where this result appears, so a number in a paper can be traced to the record that produced it,
and the record back to the paper and its materials. The same block describes published
reference values (validation style `published`) and published data (`provenance.data`).

```json
"publication": {
  "citation": {"type": "article-journal", "title": "Strike Three: Discrimination, Incentives, and Evaluation",
               "author": [{"family": "Parsons", "given": "Christopher A."}, "..."],
               "issued": {"date-parts": [[2011]]}, "container-title": "American Economic Review",
               "volume": "101", "issue": "4", "page": "1410-1435",
               "DOI": "10.1257/aer.101.4.1410"},
  "locator": {"study": "1", "table": "3", "panel": "A", "column": "9", "row": null,
              "label": "Table 3, column 9"},
  "status": "published",
  "ascollected_url": "https://ascollected.org/ZJ6_Z47",
  "researchbox_url": "https://researchbox.org/15",
  "researchbox_files": [7, 12],
  "aspredicted_url": null,
  "replication_package": {"doi": "10.3886/E112434V1", "url": "https://www.openicpsr.org/openicpsr/project/112434"},
  "links": [{"rel": "preregistration", "url": "..."}, {"rel": "osf", "url": "..."}]
}
```

(The citation is Parsons et al.'s; the AsCollected and ResearchBox URLs are the public examples
and belong to other papers. They only illustrate the fields.)

- `citation`: **CSL-JSON**, the citation format Zotero, pandoc, Mendeley and CrossRef use, so
  a citation can be pasted in from a reference manager or fetched from a DOI. Common fields:
  `type`, `title`, `author`, `issued`, `container-title`, `volume`, `issue`, `page`, `DOI`,
  `URL`, `version`. Working papers and preprints use `type: "report"` or `"article"` with
  `number` and `publisher`.
- `locator`: where in the work, as structured fields **plus** a `label` as printed. Fields:
  `study`, `table`, `panel`, `column`, `row`, `figure`, `equation`, `page`, `appendix`,
  `section`. Any can be absent. Structured fields make "every record for Table 3" a simple
  query; `label` keeps the paper's own wording ("Table A4, col. (3)"). One record has one
  locator. A result that appears twice (main text and appendix) lists `locators` instead.
- `status`: `published`, `accepted`, `submitted`, `working-paper`, `preprint`, `unpublished`.
  `version` in `citation` says which version of the paper the locator refers to, since table
  numbers move between drafts.
- **Named links** for the services the format wants to encourage, so writers see them and
  readers can rely on them: `ascollected_url`, `researchbox_url` (with `researchbox_files`:
  file ids in the box's bingo table), `aspredicted_url`, `replication_package` (DOI and URL:
  openICPSR, Dataverse, Zenodo). Use the URL of the specific version cited: AsCollected and
  ResearchBox give each public version its own URL.
- `links`: anything else, as `{rel, url}` (OSF, GitHub release, Code Ocean capsule).

Writers can fill `publication` from options (`dfeols.fit(..., publication={...})`, or a
project-level default plus a per-model locator), so a paper's records all carry the citation
and each its table and column.

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
| `published` | `{publication, values}`: numbers from a paper, with a `publication` block (citation + locator) as above |

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
   16 MB fail validation. (Thresholds agreed by Ted, 2026-10-24.)
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
- Integrity check (`meta.core_digest`), **on by default in writers** (Ted, 2026-10-24): when
  the record is created, the writer computes one sha256 fingerprint over the result sections
  (`meta` except the digest itself, `specification`, `metrics`, `parameters`, `structures`),
  serialised in a fixed canonical way (RFC 8785), so the same content always gives the same hash.
  Later tools only add blocks, so the fingerprint should never change. Any tool, or a reader,
  can recompute it to check whether someone edited the results after the run (by hand or by a
  buggy tool), as opposed to only annotating them.
  The field is optional in the format. Writers include it by default and may let users turn it
  off, and a producer may leave it out entirely (e.g. a Stata converter). A validator reports
  "no digest" as information, not a warning. A digest that doesn't match is a warning.
- The core validator ignores block content; each block's `schema` names the schema that
  validates it, and readers skip blocks they don't know.

## Profiles

A profile is a named, versioned set of rules on top of the core: which `metrics`, `parameters`
columns and `structures` a record of that kind must have, and their shapes. A record lists the
profiles it follows in `meta.profiles`; a validator checks the core schema plus each profile.
This gives model-specific requirements without a new format per model.

### How a profile is built (Ted, 2026-10-24)

For each task, take the most popular tools in Stata, R and Python and list everything each one
saves and every option it takes. Then:

- **Vocabulary = the union.** Every quantity any of them saves gets one profile name, a
  definition, and a mapping to each tool's own name. A writer that has the quantity uses the
  profile name. One that doesn't omits it.
- **Requirements = what (nearly) all of them save.** Only those are required for a record to
  claim the profile. Where tools disagree on whether a quantity exists or what it means, the
  profile flags it for discussion instead of deciding silently (see `linear-fe` below).
- **Options = the union of options**, with each tool's default. A record lists the effective
  value of every option the producer has. The mapping shows which options in another tool
  correspond, which is most of what explains a discrepancy between tools.
- The mapping table ships with the profile (`schema/profiles/linear-fe.md`), so it also serves
  as a cross-tool reference, and converters (Stata `e()` → record, fixest → record) are
  written from it.

Proposed first profiles:

| Profile | Tools surveyed | Requires |
|---|---|---|
| `inference/0.1` | R `lm`/estimatr, statsmodels, Stata `regress` | `parameters` with estimate, std_error, p_value; a default `vcov` with `type` and `ssc`; `metrics.nobs`, `df_residual` |
| `linear-fe/0.1` | fixest, lfe (R); pyfixest, linearmodels (Python); reghdfe, areg, xtreg (Stata); FixedEffectModels.jl | `inference` + below |
| `iv/0.1` | fixest, ivreg (R); pyfixest, linearmodels IV (Python); ivreghdfe, ivreg2, ivregress (Stata) | `inference` + endogenous/instruments in `specification`; `structures.first_stage` |
| `dfeols/0.1` | dfeols | `linear-fe` + `extensions.dfeols` |

Later, as needed: `glm/0.1` (deviance, log-likelihood, family/link; fixest `feglm`/`fepois`,
`glm`, statsmodels, ppmlhdfe, Stata `glm`), `mixed/0.1` (lme4, Stata `mixed`, statsmodels
MixedLM), `ml/0.1` (scikit-learn, tidymodels/parsnip, xgboost).

### Worked example: `linear-fe/0.1`

Sources: fixest `feols` value and arguments, reghdfe stored results and options, linearmodels
`PanelEffectsResults` (all checked against current documentation, 2026-10-24). pyfixest, lfe,
Stata `areg`/`xtreg` and FixedEffectModels.jl are from memory and must be checked before the
schema is written.

**Metrics** (✓ = required):

| Profile name | Meaning | fixest | reghdfe | linearmodels | others |
|---|---|---|---|---|---|
| ✓ `nobs` | rows used | `nobs` | `e(N)` | `nobs` | pyfixest `_N`; FEM.jl `nobs` |
| `nobs_full` | rows before singleton removal | `nobs_origin`† | `e(N_full)` | — | |
| ✓ `n_singletons` | singleton rows dropped | (`obs_selection`) | `e(num_singletons)` | — | |
| ✓ `df_residual` | residual df | `df.residual`* | `e(df_r)` | `df_resid` | FEM.jl `dof_residual` |
| `df_model` | model df | | `e(df_m)` | `df_model` | |
| `df_absorbed` | df lost to the FEs | | `e(df_a)` | | FEM.jl `dof_fes` |
| `df_absorbed_nested` | FE df redundant because nested in a cluster | | `e(df_a_nested)` | | |
| `rss`, `tss`, `tss_within` | sums of squares | `ssr`*, `ssr_null`, `ssr_fe_only` | `e(rss)`, `e(tss)`, within tss | `resid_ss`, `total_ss`, `model_ss` | |
| `r2`, `adj_r2` | R² | `r2()` | `e(r2)`, `e(r2_a)` | `rsquared` | |
| `r2_within`, `adj_r2_within` | within R² | `r2(, "wr2")` | `e(r2_within)`, `e(r2_a_within)` | `rsquared_within` | xtreg `e(r2_w)` |
| `r2_between`, `r2_overall` | panel R²s | | | `rsquared_between`, `rsquared_overall` | xtreg `e(r2_b)`, `e(r2_o)` |
| `r2_inclusive` | R² including the fit of the FEs | | | `rsquared_inclusive` | |
| `sq_corr_fitted` | squared corr(y, ŷ) | `sq.cor` | | `corr_squared_*` | |
| `sigma` | RMSE | `sigma()` | `e(rmse)` | `s2` (variance) | xtreg `e(sigma_e)` |
| `sigma_fe`, `rho` | FE standard deviation; share of variance due to the FEs | | | `variance_decomposition` | xtreg `e(sigma_u)`, `e(rho)` |
| `log_lik`, `log_lik_fe_only`, `log_lik_null` | log-likelihoods | `loglik`, `ll_fe_only`, `ll_null` | `e(ll)`, `e(ll_0)` | `loglik` | |
| `f_stat`, `f_stat_robust` | joint test of regressors | `fitstat` | `e(F)` | `f_statistic`, `f_statistic_robust` | |
| `f_fe` | test that the FEs are jointly zero | | | `f_pooled` | xtreg `e(F_f)` |
| `sum_weights` | | | `e(sumweights)` | | |
| `rank` | rank of the vcov | | `e(rank)` | | |

\* method or derived (`r2()`, `fitstat()`, `sigma()`), not a stored element. † not in the
documented Value list; from memory, to check.

**Required set: decided (Ted, 2026-10-24):**

1. **`nobs` means different samples** across tools: reghdfe drops singletons by default, while
   fixest (`fixef.rm`) and pyfixest keep them unless asked. `nobs`, `n_singletons` and
   `specification.options.drop_singletons` are all required, so the difference is visible.
2. **`df_residual`** = N − K − absorbed levels (the model's residual df), always. The df used
   for t tests and CIs goes in `vcov.df_t`, with the tool's convention (reghdfe and fixest
   use G − 1 under clustering).
3. **`r2_within`** is not required (areg has none, lfe defines it differently). It is
   vocabulary, with `definition` when it isn't the standard within R².
4. **FE level counts** (`fixed_effects[].levels`) are required for FE models.
5. **`ssc`** is required for inference records, with at least `convention` (e.g. `"fixest"`,
   `"reghdfe"`, `"linearmodels debiased=True"`) and the individual factors (what K counted,
   G/(G−1), (N−1)/(N−K)) when the producer has them.
6. **Convergence** (`iterations`, `converged`) is required when the algorithm is iterative
   (reghdfe, fixest, pyfixest, dfeols, FEM.jl), and absent for exact methods (xtreg, areg,
   one-FE linearmodels).

**`structures.fixed_effects`**, one entry per FE: `name`, `components` (for interactions and
slopes, as reghdfe `e(extended_absvars)`), `levels` (fixest `fixef_sizes`), `redundant`
and `redundant_method` (reghdfe `e(dof_table)` and `dofadjustments()`: exact for nested FEs,
`pairwise`, `firstpair`, `clusters`, `continuous`), `nested_in` (clusters, other FEs),
`group_size` `{min, mean, max}` (linearmodels `entity_info`/`time_info`, xtreg `e(g_min)`,
`e(g_avg)`, `e(g_max)`), `constant_reported` (reghdfe `e(report_constant)`).

**`structures.numerics`**: `algorithm` (reghdfe `technique()`: map, lsmr, lsqr;
fixest `fixef.algo`), `acceleration` (reghdfe: cg, sd, aitken, none), `transform` (reghdfe:
symmetric_kaczmarz, kaczmarz, cimmino), `preconditioner`, `tolerance` (fixest `fixef.tol`,
reghdfe `tolerance()`), `max_iterations` (fixest `fixef.iter`, reghdfe `iterations()`),
`iterations` (reghdfe `e(ic)`, FEM.jl `iterations`), `converged` (reghdfe `e(converged)`,
FEM.jl `converged`), `collinearity_tolerance` (fixest `collin.tol`), `collinear_terms`
(fixest `collin.var`), `collinearity_min_norm` (fixest `collin.min_norm`).

**`structures.vcov`**: `type` (vocabulary: `iid`, `HC0`-`HC3`, `CRV1`, `CRV3`,
`twoway`/`multiway`, `HAC`, `DK`; mapped to fixest `vcov=`, reghdfe `vce()`, linearmodels
`cov_type`), `cluster`, `n_clusters` per variable (reghdfe `e(clust#)`) and the minimum
(`e(N_clust)`), and `ssc` written out with every factor: fixest `ssc()` (`K.adj`, `K.fixef`,
`G.adj`, `G.df`, `t.df`, `K.exact`), pyfixest `ssc()`, reghdfe `e(dofmethod)`. These factors
are the most common source of SE differences between tools, so `ssc` must name the convention
it reproduces (`fixest`, `reghdfe`, `stata`, `linearmodels`).

**`specification.options`**, the union with each tool's default (reghdfe tolerance `1e-8` and
16,000 iterations; fixest `fixef.tol` `1e-6`, `fixef.iter` 10,000, `collin.tol` `1e-9`,
`fixef.rm` `"perfect_fit"`): `drop_singletons` (reghdfe `keepsingletons` off), `fixef_rm`
(fixest), `threads` (fixest `nthreads`, reghdfe `parallel()`), weights type (frequency,
analytic, probability).

**Sample**: reghdfe `e(sample)` and fixest `obs_selection` mark the rows used. The profile
records the counts (above) and, as an attachment or a hash (dfeols `row_key`), the rows
themselves.

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

`estimation-collection/0.1`: a set of records plus what is true of the set as a whole. There are
two main uses, and each leans on a different part (Ted, 2026-10-24):

- **A paper's results** (`kind: "paper"`) is organised by `members`. They are ordered by
  locator (Table 1 col. 1, col. 2, ...), with the paper's citation in `publication`. It answers
  "which record is behind each number?".
- **Replicating results** (`kind: "session"` or `"replication"`) is organised by `events`: the
  timeline of what was done, in order, including the fits (each fit event points to its member).
  It answers "what exactly happened, so that I can do it again?".

Both parts are always allowed, and a collection can have both. The `kind` sets which one a
validator expects.

The details: It is an export, built from records and
never edited by itself. Each member record stays self-contained: it can be pulled out and
read alone.

```text
schema         "estimation-collection/0.1"
meta           id, created, title, tags, notes, kind ("paper" | "session" | "replication" | "batch" | "other")
session        what the session was and how it ran (kind "session")
provenance     the session's own software, environment, execution, agent, data
publication    optional: the paper a "paper" collection belongs to (citation; members carry locators)
members        the records, in order
events         what happened in the session besides fits
extensions     chained blocks, as for records
history        as for records
```

- `session`: `id` (the session id that member records carry in
  `provenance.execution.session`), `started`, `ended`, `duration`, `status` (`closed`,
  `crashed`, `timed_out`), clients that connected (`{language, version, host, connected,
  disconnected}`, e.g. a Python and an R client), and what was fixed for the session (FE
  structure, weights, sample definition, declared variables).
- `provenance`: the same structure as a record's, describing the **session process**, not one
  fit. Software, environment and hardware of the manager and workers. Execution: the
  scheduler job that held the session (SLURM fields), launch parameters, total wall time and
  worker-seconds, peak memory, idle time. Data: the dataset and its checksum, and persisted
  demeaned data (key, size, sha256). Members repeat what applies to them, so a member read alone
  loses nothing, while the session-level block shows whether the environment changed *during*
  the session (e.g. workers replaced after a failure).
- `members`: each `{id, sha256, path or embedded record, created, order, label}`, in the order
  run. Embedded whole, or as a JSON Lines file (one record per line) linked like an attachment.
  A member's `sha256` lets the collection prove which version of each record it holds.
- `events`: `{when, type, detail, member}` in order: fits (`type: "fit"`, with `member` set to
  the record id), variables created, data loaded, demeaned data persisted or reloaded, settings
  changed, workers lost and replaced, client connects and disconnects, errors. Replaying the
  events in order should reproduce the members.
- Redaction and size rules are the same as for records. Since a collection can embed many
  records, the 16 MB limit applies to the collection file; a larger one keeps its members as a
  linked JSON Lines file instead of embedding them.

A flat `parameters` table across members (Parquet/CSV, with record id, response, term and
locator) is a convenience export for tables and plots.

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

## Sources consulted (2026-10-24)

- fixest `feols` reference: https://lrberge.github.io/fixest/reference/feols.html
- reghdfe help (stored results, options): https://scorreia.com/help/reghdfe.html
- linearmodels `PanelEffectsResults`:
  https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.results.PanelEffectsResults.html
- SPSS OMS overview and syntax: https://ibm.com/support/knowledgecenter/SSLVMB_24.0.0/spss/base/syn_oms_overview.html,
  https://www.ibm.com/support/knowledgecenter/SSLVMB_24.0.0/spss/base/syn_oms.html;
  REGRESSION `OUTFILE=COVB`: https://stats.oarc.ucla.edu/spss/faq/how-can-i-output-the-results-of-my-regression-to-an-spss-data-file/;
  CSLOGISTIC `OUTFILE` (XML model): https://www.ibm.com/docs/en/SSLVMB_sub/statistics_reference_project_ddita/spss/complex_samples/syn_cslogistic_outfile.html
- Stata `creturn`: https://www.stata.com/manuals/pcreturn.pdf
- renv lockfile: https://rstudio.github.io/renv/articles/lockfile.html
- AsCollected: https://ascollected.org/, https://ascollected.org/help, examples
  https://ascollected.org/ZJ6_Z47 and https://ascollected.org/G2Q_W7V; ResearchBox example
  https://researchbox.org/15
- From memory, to verify before the schema: pyfixest, lfe, Stata `areg`/`xtreg` stored
  results, FixedEffectModels.jl, `threadpoolctl`, R `sessionInfo()` fields, SLURM `sacct`
  fields, CSL-JSON, purl, PEP 751.

## Questions for Ted

None open. Still to revisit: interaction order ("as specified" for now; revisit if matching
across tools needs a normalised order in `name` as well).

## Next

1. Verify the tool fields marked "from memory" (pyfixest, lfe, Stata `areg`/`xtreg`,
   FixedEffectModels.jl, `threadpoolctl`, `sessionInfo()`, SLURM `sacct`, CSL-JSON, purl,
   PEP 751).
2. JSON Schema: `schema/estimation-0.1.json`, `schema/profiles/{inference,linear-fe,iv,dfeols}-0.1.json`,
   with `x-identifying` marks.
3. `src/dfeols/estimation.py`: writer, reader, validator, redactor, core digest, attachment
   spilling; tests.
