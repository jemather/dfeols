# Design: interactive sessions (fixed FEs, many models)

Status: build order step 2 (session core) built 2026-11-02 in `src/dfeols/session.py`, with
parts of step 3 (persisted data found by key and reused); see "As built" below. Steps 4-6 are
not built.

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
Ted, 2026-10-28, see `planner.md`): the session prints the size (rows x columns x 8 bytes,
plus ids) and the scratch quota, and asks for confirmation only where the site profile says
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

A session also leaves one `estimation-collection/0.1` file (`kind: "session"`; estimation spec
section 11), so the whole run can be read, checked and replayed later. The session manager
writes it to `<scratch>/sessions/<name>/session.estimation-collection.json` on `export`
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
`estimation redact`.

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
against estimation-format, reuse of persisted data, the lock, crash recovery and the CLI with
the manager as its own process. Also run on a 2-worker Dask cluster: b and V equal to `fit`'s
within 5e-15.

- **Demean once.** The declared variables are the outcome block of a worker `Plan` with no
  regressors, so the session reuses fit's singleton passes (`_drop_singletons`) and round-1
  demeaning (`_prepare`, `_round1_dense`). Crossing FEs enter as dense dummies; their block
  `D~'WD~` and `D~'WV~` are summed, `Gamma` solved once on the coordinator, and a second pass
  writes `V~ - D~ Gamma`. Each shard's residualised columns go to one Parquet file with the
  weights and each declared cluster's codes.
- **Gram matrix.** The demeaning passes also sum the weighted Gram matrix of all the
  residualised columns (kept in `manifest.json`). A model's `X'WX`, `X'Wy` and `y'Wy` are
  blocks of it, so coefficients, iid SEs, RSS and within R2 need **no pass**; hetero and CRV1
  take one pass (residuals and meat). Several responses (`y1 + y2 ~ x`) share the pass.
- **Clusters are declared at start** (`--cluster a,b`), like the variables: the nesting of
  each FE in each cluster and the cluster counts (for the dof rules) are found while demeaning.
- **Models** are `y ~ x1 + x2` on declared variables; the FE part may be left out or must be
  the session's. Interactions of declared variables and `i()` terms are refused (an
  interaction of demeaned columns is not the demeaned interaction; created variables are step
  4). IV is step 6.
- **Records.** Each model writes `result.json` and `result.estimation.json` (one per response)
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
  the session folder and deletes them at the end. Each build appends to
  `<scratch>/demeaned/timings.jsonl`, which `session start` uses for its estimate.
- **Collection.** `session.estimation-collection.json` (`kind: "session"`) on `export`
  (`open`) and at the end (`closed`, also after `idle_minutes` idle (Ted 2026-11-02);
  `timed_out` at walltime; `crashed` as above),
  through `estimation_format.new_collection`/`write_collection`: members in run order with
  labels, `session` (id, times, status, clients, `fixed`), `provenance` (software,
  environment, execution, the dataset and the demeaned data), `events`.
- **Jobs.** `session start` with a SLURM profile submits the coordinator script running
  `dfeols session serve` with walltime `--hours` (default 2) + 1.5 x the high end of the
  demeaning estimate (1 h without history); on a local profile it starts the manager as a
  detached process logging to `<scratch>/sessions/<name>.log`. SIGTERM closes cleanly.
  `--hours` count from ready, not from submission. The manager asks for `max_workers` workers
  and holds them for the whole session, idle or not (Ted 2026-11-02: once allocated for a
  slot they are kept; `cluster.executor(hold=True)`), with the manager job's walltime.

Not built: crossing FEs above `max_dummy_levels` (the sparse and plan C paths), progress
streamed to the client and the per-model time estimate, `persist()` after `--no-persist`,
`dfeols demeaned rm` and the scratch-quota check, and the replay of events.

## Open questions

- Multi-way clustering (CRV1 on two variables) in sessions: wanted?
- Where the R package lives (this repo under `r/`, or its own).
- Upper bound on declared variables (memory and persisted size scale with it).
