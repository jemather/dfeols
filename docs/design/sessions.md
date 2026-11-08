# Design: interactive sessions (fixed FEs, many models)

Status: build order step 2 (session core) built 2026-11-02 in `src/dfeols/session.py`, with
step 3 (persisted data found by key and reused, `persist()` after `--no-persist`, `dfeols
demeaned list/rm`, the scratch checks; 2026-11-08) and step 4 (created variables, refit;
2026-11-08); see "As built" below. Steps 5-6 are not built.

## Goal

Absorb a set of fixed effects once on a large dataset, then estimate many models against it from
Python or R without re-queueing or re-demeaning: `Y1 ~ X`, `Y1 ~ X + A + B`, `Y2 ~ X1`, a new
variable, a different vcov. At 1e10 rows a model should take tens of seconds to a few minutes
(bounded by reading the demeaned columns), where no in-memory tool can fit it at all.

## Workflow

```text
$ dfeols session start db1b --profile oscer --data /scratch/.../db1b_by_route \
      --fe "route + carrier^quarter" --vars lfare,hhi,lpax,dist --cluster route \
      --hours 3
  dataset: 412,000,000 rows, 13,200 shards; FEs: route (local), carrier^quarter (crossing, 6,840 levels)
  demeaning 4 variables: estimated 25-60 min on 20 workers (from 3 earlier runs on this profile)
  persisted demeaned data: ~13 GB in /scratch/ted/dfeols/demeaned/... (quota: 4.1 TB free)
  requesting 3h + 90 min; waiting for SLURM ...            [job 35450012, est. start unknown]
  [demean] 41% | 2 of 4 variables | ETA 18 min
  ready 03:12; ends 06:42 or after 30 min idle
```

```python
s = dfeols.Session.attach("db1b")            # finds the session through scratch
m1 = s.fit("lfare ~ hhi", vcov={"CRV1": "route"})          # "estimated 40 s" -> progress -> result
m2 = s.fit("lfare ~ hhi + lpax + dist")
s.add_variable("lfare_x_hhi", "lfare_raw * hhi")           # built from raw columns, then demeaned
m3 = s.refit(m2, vcov="hetero")                            # residual pass only
s.persist()                                                 # if started with --no-persist
s.export([m1, m2, m3], "table2.tex")                       # also .docx, .md; JSON always written
```

```r
s <- dfeols::attach_session("db1b")
m1 <- fit(s, lfare ~ hhi, vcov = list(CRV1 = "route"))
modelsummary::modelsummary(list(m1, m2))   # results carry broom tidy/glance
```

## Fixed per session, free per model

| Fixed at start | Free per model |
|---|---|
| dataset (a prepped shard folder), FE structure, weights | dependent variable, regressors (any declared or created variable) |
| estimation sample: rows complete on all declared variables, minus singletons over all FEs | vcov type and cluster variable (among those encoded at prep) |
| demeaning tolerance | which models to export |

Declared variables cannot be changed. New variables can be **created**: an expression on raw
columns of the dataset (`log(x)`, `a * b`, `x > 0`), computed per shard and then demeaned (one
pass, one column). Interactions are always formed from raw columns before demeaning (the product
of two demeaned columns is not the demeaned product). A created variable that is missing on any
row of the session sample is an error, with the count; that needs a new session (Ted, 2026-10-24:
fixed sample).

## How it works

- **Manager job.** `session start` submits one coordinator job (walltime = requested hours +
  1.5 × the demeaning estimate) that holds a Dask worker pool, as `dfeols submit` does today.
  It takes a lock on the session (DirLock, after the race fix).
- **Demean once.** Per shard, absorb the local FE for every declared variable (as round 1 does).
  Crossing FEs enter as dummies; their block of the normal equations depends only on the FEs,
  sample and weights, so it is factorised once and reused to partial them out of every variable.
  The result is each variable fully residualised on all FEs, kept per shard (worker memory when
  it fits, else node-local disk or scratch), plus the cluster ids and the absorbed-level count
  needed for degrees of freedom.
- **Per model.** OLS on residualised columns is exact (Frisch-Waugh-Lovell): cross-products of
  the model's columns over shards, a k x k solve, then one more pass for hetero/CRV1 meat. IV
  (2SLS) is exact on residualised columns too and needs only more cross-products, so it fits
  the same machinery.
- **Requests.** The client writes a JSON request into `<scratch>/sessions/<name>/requests/`; the
  manager polls (about 1 s), runs it, and writes the result file and progress beside it. This
  needs no network path from login node to compute node, and makes the R client independent of
  Python (jsonlite only). A direct Dask client connection can come later.

## Estimates and progress

- **Before submitting:** rows x variables x throughput for this profile and FE type, from a local
  record of earlier runs (`<scratch>/dfeols/timings.jsonl`), shown as a range, plus the size of
  persisted output. With no history: a short calibration on a few shards at startup, and a wide
  range until then.
- **Per model, at submission:** passes x columns x the throughput measured in this session;
  accurate once the session has run one model.
- **Progress:** the existing per-stage progress (shards, rows/s, ETA) streamed to the client.

## Persisted demeaned data

```text
<scratch>/dfeols/demeaned/<dataset-name>/<key>/
  manifest.json       # what this is and how to reuse it (below)
  shards/part-000123.parquet   # residualised float64 columns + cluster ids, one file per shard
```

`<key>` is a hash of: source dataset manifest, FE structure, weights, sample definition (declared
variables), singleton rule, tolerance. `session start` with the same inputs finds it and skips
demeaning; a different key never reuses it. `manifest.json` records those inputs in plain form,
the per-variable list (with creation expressions), rows, absorbed-level counts per FE, dfeols
version and commit, and creation time. Persisting is the default (`--no-persist` to skip;
Ted, 2026-10-28, see `planner.md`): the session prints the size (from earlier builds'
compressed size, at most rows x columns x 8 bytes) and the scratch quota, and asks for confirmation only where the site profile says
scratch is scarce (`demeaned_confirm_gb`). Removal is explicit (`dfeols demeaned rm`). Columns
are float64 throughout (Ted, 2026-10-24).

The files are plain Parquet, so any tool can read them; correct SEs need the dof information in
`manifest.json`.

## Results file

Superseded by `estimation-format.md` (the full format); the sketch below is the earlier broom-only version.

There is no standard JSON format for regression results. The closest shared convention is R's
broom (`tidy`: term, estimate, std.error, statistic, p.value, conf.low, conf.high; `glance`: one
row of model statistics), which modelsummary and others consume. Each model is one
self-contained file, `dfeols-result/1`:

```json
{
  "schema": "dfeols-result/1",
  "model": {"formula": "lfare ~ hhi + lpax", "fe": ["route", "carrier^quarter"], "weights": null,
            "method": "ols"},
  "tidy": [{"term": "hhi", "estimate": 0.0, "std.error": 0.0, "statistic": 0.0, "p.value": 0.0,
            "conf.low": 0.0, "conf.high": 0.0}],
  "glance": {"nobs": 0, "r.squared": 0.0, "adj.r.squared": 0.0, "r.squared.within": 0.0,
             "sigma": 0.0, "df.residual": 0, "vcov.type": "CRV1", "cluster": "route",
             "n.clusters": 0},
  "vcov": {"terms": ["hhi", "lpax"], "matrix": [[0.0, 0.0], [0.0, 0.0]], "ssc": {"adj": true,
           "cluster.adj": true, "fe.dof": 0}},
  "fixed_effects": [{"name": "route", "levels": 0, "absorbed": "local"},
                    {"name": "carrier^quarter", "levels": 0, "absorbed": "crossing"}],
  "sample": {"rows": 0, "singletons.dropped": 0, "key": "<persisted-data key>"},
  "provenance": {"dfeols": "0.8.0", "commit": "...", "dataset": "<manifest sha256>",
                 "session": "db1b", "created": "2026-10-24T03:20:11Z",
                 "timing": {"seconds": 0.0, "passes": 2}}
}
```

Numbers are written with full float64 precision (`repr` round-trip). A JSON Schema for it lives
in the repo and is versioned. Exports: Python through `maketables` (.tex, .docx, .md; needs a
small extractor for dfeols results); R through modelsummary, since the file already carries
`tidy` and `glance`. A session index (`results/index.jsonl`) lists every model run.

## Session collection

A session also leaves one `estrec-collection/0.1` file (`kind: "session"`; estimation spec
section 11), so the whole run can be read, checked and replayed later. The session manager
writes it to `<scratch>/sessions/<name>/session.estrec-collection.json` on `export`
(`status: "open"`) and again when it ends (`closed`, `timed_out`, or `crashed` when the
manager is restarted after a crash).

- **Members:** the `estimation` record of every model the session ran, in the order run.
  Each member record carries the session id in `provenance.execution.session` and stays
  complete on its own. Above 16 MB they move to `session.members.jsonl`, as the spec does.
- **`session`:** `id` (a UUID per session run; the name, e.g. `db1b`, can be reused, so it
  goes in `fixed`), `started`, `ended`, `elapsed_seconds`, `status`, `clients` (each Python or R
  client that attached, with connect and disconnect times), and `fixed`: the FE structure,
  weights, sample definition, declared variables, singleton rule, tolerance and the
  persisted-data key.
- **`provenance`:** the manager job (SLURM fields, walltime requested and used, worker-seconds,
  peak memory, idle time), the software and hardware of the manager and workers, and the
  dataset's manifest sha256 and the persisted demeaned data (key, size, sha256) in `data`.
- **`events`:** built from the request log and the session index: `data_loaded`,
  `demeaned_persisted` or `demeaned_loaded`, `variable_created`, `setting_changed` (a vcov
  change on an existing model), a `fit` per model with `member` set to its record id,
  `worker_lost` and `worker_added`, `client_connected` and `client_disconnected`, and `error`
  with the message. Replaying the events in order reproduces the members.

`results/index.jsonl` stays the quick list of models; the collection is what gets shared, after
`estrec redact`.

## Interfaces

- **Python:** `dfeols.Session` (start, attach, fit, refit, add_variable, persist, export, close;
  `export` also writes the session collection).
- **R:** a small package (jsonlite only) speaking the same request files: `attach_session`,
  `fit`, `add_variable`, `persist`; results as objects with `tidy()`/`glance()` methods.
- **CLI:** `dfeols session start|status|fit|stop`.

## Lifecycle and safety

- The session ends at walltime, after `idle_minutes` without a request, or on `session stop`.
  Demeaned columns are checkpointed as they finish, so an interrupted start can resume.
- Starting a second session on the same name or persisted key fails cleanly (DirLock).
- Pool size between `min_workers` (held while idle) and `max_workers` (during a request).

## Validation

- Every model vs pyfixest on Parsons (same N, b to 1e-7, SE to 1e-7), including created
  variables and vcov changes; T3c9 across partitions.
- FWL check: session results equal `dfeols.fit` on the same formula.
- Persisted data reloaded in a new session gives identical results.
- IV vs pyfixest IV once added.

## Build order

1. DirLock race fix (prerequisite).
2. Session core: start/attach/stop, demean once, OLS with iid/hetero/CRV1, request files,
   estimates, `estimation` records, the session collection. Python client and CLI.
3. Persisted demeaned data: write, find by key, reuse; size warning.
4. Created variables; vcov changes on an existing model.
5. Exports (maketables extractor); R client.
6. IV in sessions.

## As built (step 2, 2026-11-02)

`dfeols.session`: `Spec` (what is fixed), `demean_once`, `fit_model`, `Manager` (the request
loop), `Session` (the Python client), and `dfeols session start|serve|status|fit|export|stop`.
`tests/test_session.py` checks every model against `dfeols.fit` on the same sample (FWL; local
and crossing FEs, weights, iid/hetero/CRV1, a crossing cluster), the records and the collection
against estrec, reuse of persisted data, the lock, crash recovery and the CLI with
the manager as its own process. Also run on a 2-worker Dask cluster: b and V equal to `fit`'s
within 5e-15.

- **Demean once.** The declared variables are the outcome block of a worker `Plan` with no
  regressors, so the session reuses fit's singleton passes (`_drop_singletons`) and round-1
  demeaning (`_prepare`, `_round1_dense`/`_round1_sparse`). Crossing FEs are absorbed one of
  two ways (`crossing_fe`, default `auto`):
  - **dense** (all crossing FEs within `max_dummy_levels`): dummies demeaned by the local FEs,
    `D~'WD~` and `D~'WV~` summed, `Gamma` solved once on the coordinator;
  - **iterate** (plan C, when any crossing FE is larger; added 2026-11-02): round 1 returns
    the per-level sums `D'WV~` and each shard's part of `A = D~'WD~` (summed on the workers,
    within `level_space_gb`), and `fit`'s CG (`_crossing_cg`, same preconditioners, `iter_tol`
    default 1e-8) solves `A Gamma = D~'WV~` once for **all** declared variables together:
    CG's cost is paid at start-up, never per model. Above the level-space budget each CG
    step is a pass over the shards, with fit's slow-convergence warning.

  A second pass writes `V~ - D~ Gamma` (plan C: `D Gamma`, each row's levels summed, demeaned
  by the local FEs). Each shard's residualised columns go to one Parquet file with the
  weights and each declared cluster's codes. Models are then exact OLS on these columns; on
  plan C they carry CG's error in `Gamma` (tests: within 1e-7 of the dense solve at
  `iter_tol=1e-10`).
- **Gram matrix.** The demeaning passes also sum the weighted Gram matrix of all the
  residualised columns (kept in `manifest.json`). A model's `X'WX`, `X'Wy` and `y'Wy` are
  blocks of it, so coefficients, iid SEs, RSS and within R2 need **no pass**; hetero and CRV1
  take one pass (residuals and meat). Several responses (`y1 + y2 ~ x`) share the pass.
- **Clusters are declared at start** (`--cluster a,b`), like the variables: the nesting of
  each FE in each cluster and the cluster counts (for the dof rules) are found while demeaning.
- **Models** are `y ~ x1 + x2` on declared (or, step 4, created) variables; the FE part may
  be left out or must be the session's. Interactions of declared variables and `i()` terms
  are refused (an interaction of demeaned columns is not the demeaned interaction; the error
  names the `add_variable` call that makes it from raw columns). IV is step 6.
- **Records.** Each model writes `result.json` and `result.estrec.json` (one per response)
  in `models/<nnn>-<label>/`. The record is a normal dfeols record plus:
  `provenance.execution.session` = the session id, the demeaned data as a `derived` data
  entry, `specification.sample.restriction` naming the declared variables, and the call
  `dfeols.Session.attach(name).fit(...)` (in `specification.call` and the dfeols replay).
- **Requests.** `requests/<id>.json` (written atomically) -> `responses/<id>.json`; types
  `attach`, `detach`, `fit`, `status`, `export`, `stop`. Errors come back in the response and
  are logged as `error` events; the session keeps running.
- **Folders.** `<scratch>/sessions/<name>/` (`session.json`, `requests/`, `responses/`,
  `models/`, `results/index.jsonl`, `events.jsonl`, the collection), locked by the sibling
  `<name>.lock`. A new run of a name moves the last run's files to `previous/<id>/`; if that
  run's manager died, its collection is first rewritten from what it left, as `crashed`.
  Demeaned data in `<scratch>/demeaned/<dataset>/<key>/` (`shards/part-*.parquet`, then
  `manifest.json` last, so a manifest means complete), built under a DirLock on the key; a
  later session with the same key loads it (`demeaned_loaded`). `--no-persist` keeps them in
  the session folder and deletes them at the end, unless `s.persist()` (or `dfeols session
  persist <name>`) moves them to the demeaned folder first (`demeaned_persisted` event, with
  `from`); if that key was kept by another session meanwhile, the session switches to that
  copy and deletes its own. Records written before the move keep the old location. Each build appends to
  `<scratch>/demeaned/timings.jsonl`, which `session start` uses for its estimate.
- **Collection.** `session.estrec-collection.json` (`kind: "session"`) on `export`
  (`open`) and at the end (`closed`, also after `idle_minutes` idle (Ted 2026-11-02);
  `timed_out` at walltime; `crashed` as above),
  through `estrec.new_collection`/`write_collection`: members in run order with
  labels, `session` (id, times, status, clients, `fixed`), `provenance` (software,
  environment, execution, the dataset and the demeaned data), `events`.
- **Jobs.** `session start` with a SLURM profile submits the coordinator script running
  `dfeols session serve` with walltime `--hours` (default 2) + 1.5 x the high end of the
  demeaning estimate (1 h without history); on a local profile it starts the manager as a
  detached process logging to `<scratch>/sessions/<name>.log`. SIGTERM closes cleanly.
  `--hours` count from ready, not from submission. The manager asks for `max_workers` workers
  and holds them for the whole session, idle or not (Ted 2026-11-02: once allocated for a
  slot they are kept; `cluster.executor(hold=True)`), with the manager job's walltime.

- **Estimates and progress** (2026-11-02). Before demeaning, `session start` and the manager
  estimate it from `timings.jsonl` (rows x variables over earlier runs' throughput, as a
  range) and put it in `session.json` with the path of the live progress file
  (`progress/demean/status.json`, the usual `Progress` snapshot, refreshed every 2 s);
  `Session.attach` and `session start --wait` print both while waiting. Per model, before it
  runs, the manager writes `responses/<id>.accepted.json` with passes, columns and an
  estimate: none needed when the Gram matrix answers it (b, iid), else rows x the columns the
  pass reads (regressors, responses, weight, cluster codes) over a measured throughput. The
  first pass reads files no node has cached (OSCER 35828091, 2026-11-07: m1 17 s with 5
  columns, m2 6 s with 8), so it is kept apart (2026-11-08): after a later pass, that pass's
  rate; after only the first, a range up to the first pass's rate; before any, a range from
  the first passes of earlier sessions (`<scratch>/demeaned/passes.jsonl`, one line per pass
  with `first`), or with none from the read rate of the three largest shard files on one
  worker (high end) and on every connected worker at once (low end). `estimate_seconds` is
  the high end and `estimate_range` the range. The client prints the estimate, the model's
  progress lines (`models/<dir>/status.json`) and the time it took; the index and the fit's
  result keep the estimate beside the time.
- **Size** (2026-11-08). `timings.jsonl` also records each build's columns and bytes on
  disk; `session start` gives the demeaned size as rows x columns x the range of earlier
  builds' bytes per value, capped at 8 bytes a value (the flags data took 1.3 bytes a value,
  6.9 GB against the 44 GB uncompressed figure it printed before), or the 8-byte bound with
  no history.
- **Storage** (step 3, 2026-11-08; policy from planner.md, Ted 2026-10-28). `dfeols demeaned
  list` shows each `<dataset>/<key>` folder: size, when written, rows, variables and FEs, the
  purge date (written + the profile's `scratch_purge_days`), the running sessions using it (a
  session counts while it holds its folder's lock and its `session.json` names the folder),
  and folders that are incomplete (an interrupted build), being built (the key's lock is
  held) or in an old format (no longer loaded: a session rebuilds them under a new key).
  `--json` gives the same as data. `dfeols demeaned rm <key>...` (a unique prefix, or
  `<dataset>/<key>`) asks first (`--yes` to skip; refused without a terminal), refuses
  while a running session uses the folder or a build holds its lock, and otherwise takes
  the key's lock, checks the users again, renames the folder aside (so a session starting
  meanwhile builds afresh rather than loading half a folder) and deletes it, then the
  dataset's folder if it was the last key. `timings.jsonl` and `passes.jsonl` stay. Space:
  `session start` prints the free space of the file system that will hold the data (a
  site's per-user quota is not visible to dfeols), stops when even the low end of the size
  estimate does not fit and warns when the high end does not; the manager makes the same
  stop before it demeans (so `Session.start` in Python gets it too). A profile with
  `demeaned_storage = "confirm"` makes `session start` ask before writing more than
  `demeaned_confirm_gb` (default 500), or exit asking for `--yes` without a terminal;
  `"plentiful"`, the default (OSCER), never asks.
- **Empty samples** (2026-11-08). When no rows are left (all missing, zero weight or
  singletons), the demeaning and `fit` raise a ValueError that says where the rows went;
  `fit` used to return an empty result, and a session went on to fit models on nothing.

- **Created variables** (step 4, 2026-11-08). `s.add_variable(name, expression)` (CLI
  `dfeols session add-variable <session> <name> <expression>`; it returns the session, so
  calls chain). The expression is on raw columns of the prepared data: numbers, `+ - * /`,
  `^` (or `**`), comparisons (1.0 or 0.0), `&`, `|`, `!` (or `and`, `or`, `not`), and
  `log`, `log2`, `log10`, `log1p`, `exp`, `sqrt`, `abs`, `ifelse(cond, a, b)`, with R's
  precedence. `dfeols.expr` turns R's spelling into Python's, parses it with `ast` (which
  runs nothing), refuses any node outside an allowlist and compiles the tree to a Polars
  expression; it is null wherever a column it reads is null or not finite, so `&` or
  `ifelse` cannot hide a missing input. The manager computes it per shard on the session's
  rows (worker `Plan.filter_vars`: listwise deletion on the declared variables, not on the
  new column; `Plan.created`: the expressions, computed after it) and the singleton masks,
  which format 3 keeps in `<key>/sample/` (one bit a row; format 2 deleted them). Then it is
  demeaned exactly as a declared variable (`demean_once(create=, base=)`: round 1, the dense
  solve or plan C's CG, round 2), and the same passes give its Gram entry, its raw sums
  (collinearity scale) and its cross-products with every column of the session, so a model
  with it needs no extra pass. A row where it is missing or not finite (`log(0)`) is an
  error with the count, and nothing is kept. It goes to `<key>/created/<digest>/`
  (`shards/part-*.parquet` with one column, `variable.json` last, under a DirLock), where
  `<digest>` names the expression's tree with the two operands of each `+` and `*` sorted
  (exact in float64). A later session on the same key reuses it under any name; the
  cross-products with that session's other created columns come from either column's
  `variable.json`, or else from one pass (`cross_pass`). Asking again for the same name and
  expression does nothing; a name in use, or an expression already created under another
  name, is an error. `persist()` takes the session's created columns with it (into a key
  another session kept, too). Models: `Demeaned.names`/`gram`/`raw_ss` cover the declared
  and created columns, and the meat pass reads each column's file (`Demeaned.files`). Records
  list the created variables used (`extensions.dfeols.created_variables`, one `derived` data
  entry per column) and the replay call adds them (`attach(...).add_variable(...).fit(...)`);
  the collection logs `variable_created` (name, expression, digest, reused, seconds).
  Creating a variable costs about a declared variable's demeaning; on plan C the crossing
  system `A` is rebuilt (it is not persisted), so it costs most of the original build.
- **Refit** (step 4). `s.refit(model, vcov=, cluster=)` (a Result or a label; CLI `dfeols
  session refit <session> <model> --vcov ... --cluster ...`) fits the model's formula again
  with the new vcov: b from the Gram matrix, one pass for hetero or CRV1, none for iid. A
  CRV1 refit keeps the model's cluster unless given another; the same vcov and cluster is
  an error. The label is `<model>.<vcov>[.<cluster>]` by default. The manager logs
  `setting_changed` (model, its record as `member`, vcov and cluster from and to) and then
  the new model's `fit`; the index entry has `refit_of`.

Not built: the replay of events.

## Open questions

- Multi-way clustering (CRV1 on two variables) in sessions: wanted?
- Where the R package lives (this repo under `r/`, or its own).
- Upper bound on declared variables (memory and persisted size scale with it).
