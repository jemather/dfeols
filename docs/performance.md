# Benchmarks and performance

These benchmarks answer two questions:

1. **How does dfeols compare with the tools people already use?** That means pyfixest
   (Python), fixest (R) and reghdfe (Stata), in speed and memory, on the same data and models.
2. **When does running across many nodes pay?** Most researchers' alternative is the biggest
   single machine they can get. So the main comparison point is one large-memory node, with
   dfeols run both on that node and across a cluster.

The models are ten groups of specifications on four public datasets, from 9.5 million to
1.6 billion rows.

## Summary

- **On one big node, dfeols is the fastest of the four tools on 7 of the 10 groups.** On
  the 200M-row and flood-insurance specifications it is 2.5-5x faster than fixest and 13-22x
  faster than pyfixest. At 1.6 billion rows it is 1.9x faster than fixest and 3.5x faster
  than pyfixest.
- **dfeols needs far less memory.** At 1.6 billion rows, pyfixest peaked at 867 GB, reghdfe
  at 629 GB and fixest at 476 GB. None of them would fit on a typical 256-512 GB node. dfeols's
  largest worker used 30 GB on the big node, and its whole process 79.6 GB. Across the
  cluster no worker used more than 12 GB.
- **Many nodes buy memory before speed.** Below about 200M rows, one big node was faster
  than 40 cluster workers on all but one specification. At 1.6 billion rows, 40 ordinary
  24 GB workers took as long as the biggest node: the same fit, without a large-memory node.
- **reghdfe is far slower.** It was 6-36x slower than fixest on the same node. At 1.6 billion
  rows it took 11.4 hours, against 21 minutes for dfeols.
- **The tools agree.** Every tool's coefficients equal dfeols's to within 1e-6 relative,
  and most to within 1e-9. The small differences in N and SEs come from known conventions
  (see [Agreement between tools](#agreement-between-tools)).

## Setup

| | One large-memory node | Cluster |
|---|---|---|
| Hardware | One OSCER `largemem` node: 128 cores (4 x Xeon Gold 6448H), 4 TB memory | Up to 40 OSCER workers through Slurm, each 4 cores and 24 GB |
| pyfixest 0.70.0, fixest 0.12.1 (R 4.4.2) | all 128 threads | (not run) |
| reghdfe 6.14.1, ftools 2.50.0 (Stata/MP 19) | 4 cores (the license's limit) | (not run) |
| dfeols 0.8.0 | the local executor: 32 workers x 4 cores | the Dask executor: 40 workers x 4 cores |

How the runs were measured:

- **Fit time only.** The data were already Parquet files on scratch. Loading them into memory
  (for pyfixest, fixest and reghdfe) and dfeols's shard preparation are timed separately
  (see [Load and preparation time](#load-and-preparation-time)).
- **Peak memory** is the peak resident memory of the process (`/usr/bin/time -v`). For
  dfeols it is the largest worker's, since that is what decides the node size a worker needs.
- **The same estimator everywhere:** singletons dropped, standard errors clustered (CRV1), and
  each tool's default convergence tolerance.
- **Runs:** each in-memory fit and each cluster count ran once. dfeols on the node ran twice
  (the ranges below), and the two runs agree to a few seconds. On a shared cluster, timings
  move from day to day (see [Choosing where to run](#choosing-where-to-run)).

## Datasets and models

| Data | Source | Rows used |
|---|---|---|
| **EVE orders:** EVE Online market orders, one row per order per day, Jun-Nov 2023 | [EVE Ref](https://data.everef.net) market-order archive | 9.5M (one week) and 202.6M (all 147 days) |
| **Rideshare:** NYC high-volume for-hire trips (Uber, Lyft, ...), Feb 2019 - Sep 2026 | [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | 20.4M (Mar 2024) and 1.60B (all 92 months) |
| **Flood policies:** FEMA flood-insurance policies, 2009-2026 | [OpenFEMA NFIP redacted policies](https://www.fema.gov/openfema-data-page/nfip-redacted-policies-v3) | 13.6-30.9M per specification |
| **Taxi 2009:** NYC yellow-cab trips in 2009, with driver and car IDs | Haggag and Paci (2014), [*Default Tips*](https://doi.org/10.1257/app.6.3.1), replication package | 42.6M (card trips) and 160.9M (all) |

The code that builds each analysis file is in
[`replication/`](https://github.com/jemather/dfeols/tree/main/replication) in the repository
(`eve/`, `nyc_tlc/`, `nfip/` and `haggag2014/`).

| Specification | Model | N |
|---|---|---|
| EVE, one week | Order exit (3 outcomes) on price, volume, age, price changes and side. FE item x region x day + location; cluster item | 9.5M |
| Rideshare, one month | Log fare on log miles, log time, shared request and company. FE pickup zone x hour + dropoff zone x hour + zone pair; cluster zone pair | 20.4M |
| Flood A, all policies (4 specifications) | Event study of log premium around the Risk Rating 2.0 pricing change. FE block group or tract x group + block group or tract x quarter + flood zone x quarter; cluster community | 30.1-30.9M |
| Flood A, single-family (4) | As Flood A, single-family premiums only | 25.2-26.0M |
| Flood B (4) | Log premium on the community's rating class. FE county x month + flood zone x month; cluster community | 13.6-30.4M |
| Taxi 2009, tips | Tip share on log fare. FE driver + car + pickup x dropoff neighbourhood + pickup date-hour; cluster driver | 42.6M |
| Taxi 2009, duration | Log ride duration on log distance. Same FEs and cluster as tips | 160.9M |
| EVE, all days | As EVE, one week, on every day | 202.6M |
| EVE, all days, crossing FE | As EVE, all days, with location FE replaced by location x day, which crosses item x region x day | 202.6M |
| Rideshare, all months | As Rideshare, one month, on every month | 1.60B |

`a x b` is an interacted FE (`a^b` in a formula). Every model has a few controls besides the
regressors named.

## Speed and memory against other tools

Fit time and peak memory. **Bold** marks the fastest time in each row.

| Specification | N | pyfixest | fixest | reghdfe (4 cores) | dfeols, one node | dfeols, 40 workers |
|---|---|---|---|---|---|---|
| EVE, one week | 9.5M | 42 s, 9.2 GB | 11 s, 5.4 GB | 269 s, 3.4 GB | **9 s** | (not timed separately) |
| Rideshare, one month | 20.4M | 63 s, 11.4 GB | **20 s**, 8.3 GB | 267 s, 6.5 GB | 35-38 s, 16.6 GB | 274 s, 3.5 GB |
| Flood A, all | 30.1-30.9M | 416-466 s, 89-93 GB | 100-109 s, 65-66 GB | 3,366-3,585 s, 41-42 GB | **22-28 s**, up to 12.4 GB | 39-303 s, up to 8.1 GB |
| Flood A, single-family | 25.2-26.0M | 356-425 s, 79-81 GB | 89-98 s, 66 GB | 2,565-2,771 s, 35-36 GB | **18-21 s**, up to 12.4 GB | 40-107 s, up to 8.1 GB |
| Flood B | 13.6-30.4M | 152-202 s, 27-41 GB | **33-41 s**, 25-27 GB | 195-416 s, 11-19 GB | 208-213 s, up to 12.4 GB | 283-304 s, up to 8.1 GB |
| Taxi 2009, tips | 42.6M | **80 s**, 27.6 GB | 285 s, 9.0 GB | 2,723 s, 11.6 GB | 155-157 s | 106 s |
| Taxi 2009, duration | 160.9M | 278 s, 76.4 GB | 1,065 s, 28.7 GB | 18,072 s, 37.8 GB | **229-234 s** | 726 s |
| EVE, all days | 202.6M | 857 s, 148 GB | 182 s, 106 GB | 6,518 s, 69.4 GB | **39 s**, 25 GB | 70-80 s, 8.6 GB |
| EVE, all days, crossing FE | 202.6M | 1,198 s, 169 GB | 228 s, 106 GB | 8,011 s, 69.4 GB | **90-95 s**, 25 GB | 325 s, 7.3 GB |
| Rideshare, all months | 1.60B | 4,424 s, 867 GB | 2,413 s, 476 GB | 41,114 s, 629 GB | 1,251 s, 30 GB | **1,116-1,422 s**, 12 GB |

dfeols's memory is its largest worker's. Its whole process (32 workers and the coordinator)
peaked at 79.6 GB on Rideshare, all months. That was the only run where the whole process
was measured. Memory for dfeols on the Taxi 2009 specifications was not recorded.

What the table shows:

- **dfeols on one node is fastest on 7 of the 10 rows.** fixest or pyfixest wins in three cases:
    - **Small fits** (Rideshare, one month, 20M rows): fixest takes 20 s against 35-38 s.
      dfeols's fixed costs matter more when the fit itself is short.
    - **A large crossing FE on few rows** (Flood B): fixest takes 33-41 s against about 210 s.
      About 3 min 10 s of dfeols's time is spent building the crossing-FE system for the
      conjugate-gradient solve, a step that runs on one core.
    - **Four high-dimensional FEs at moderate N** (Taxi 2009, tips): pyfixest takes 80 s
      against 155 s.
- **fixest is usually 4-5x faster than pyfixest and uses less memory.** The exception is Taxi
  2009, whose four high-dimensional FEs make fixest 3.6-3.8x slower than pyfixest.
- **reghdfe uses 4 cores and is the slowest:** 6-36x slower than fixest (35-36x on EVE, all
  days; 17x on Taxi 2009, duration; 6-10x on Flood B). It used the least memory of the
  in-memory tools up to 203M rows, but peaked at 629 GB at 1.6 billion rows.
- **Only dfeols keeps memory flat as N grows.** From 200M to 1.6 billion rows, pyfixest and
  fixest went from 106-169 GB to 476-867 GB. dfeols's largest worker went from 25 GB to
  30 GB, since a worker's memory depends on its shard's size, not on the total N.

## One node or many

### dfeols on one node against dfeols across the cluster

The same dfeols fits, on one large-memory node and on 40 four-core cluster workers:

| Specification | N | One node (128 cores) | 40 workers (160 cores) | Largest worker, node / cluster |
|---|---|---|---|---|
| Rideshare, one month | 20.4M | 35-38 s | 274 s | 16.6 / 3.5 GB |
| Flood A | 25.2-30.9M | 18-28 s | 39-303 s | up to 12.4 / 8.1 GB |
| Taxi 2009, tips | 42.6M | 155-157 s | 106 s | |
| Taxi 2009, duration | 160.9M | 229-234 s | 726 s | |
| EVE, all days | 202.6M | 39 s | 70-80 s | 25 / 8.6 GB |
| EVE, all days, crossing FE | 202.6M | 90-95 s | 325 s | 25 / 7.3 GB |
| Rideshare, all months | 1.60B | 1,251 s | 1,116-1,422 s | 30 / 12 GB |

**Up to about 200M rows, one node is faster.** It was 1.4-11x faster on every specification
but Taxi 2009, tips. A cluster run pays for Slurm starting the workers, for moving data
between nodes, and for the coordinator's serial steps. At this size the work isn't large
enough to cover those costs.

**At 1.6 billion rows the cluster matches the node.** 40 workers took 19-24 minutes against
the node's 21, and no worker used more than 12 GB. The same fit runs on ordinary 24 GB nodes,
without a large-memory node.

### Worker count

Rideshare, all months, and EVE, all days, at 10, 20, 40 and 80 workers (fit seconds from when
the first worker arrived):

| Workers requested | EVE, all days | Most workers used | Rideshare, all months | Its CG solve (22 steps) | Rideshare worker-seconds |
|---|---|---|---|---|---|
| 10 | 71 s | 10 | 4,506 s (75 min) | 3,761 s | 158,832 |
| 20 | 50 s | 20 | 3,921 s (65 min) | 2,467 s | 264,958 |
| 40 | 70 s | 8 | 1,422 s (24 min) | 956 s | 131,808 |
| 80 | 50 s | 22 | 1,200 s (20 min) | 725 s | 137,756 |

- **Rideshare, all months, scales.** Its conjugate-gradient phase ran 3.9x faster on 40
  workers than on 10, close to linear, and the whole fit 3.2x faster. The 80-worker run only
  ever got 56 workers from the queue, and was 1.2x faster than 40.
- **EVE, all days, is too small to scale.** At about 1,900 worker-seconds in all, it never kept
  more than 22 workers busy, so every worker count finished in 50-71 s.
- **The 20-worker run is an outlier.** Its singleton passes took 1,158 s, against 143-275 s in
  the other runs, and it used twice the worker-seconds, probably because of slow or late
  nodes. Each count ran once, so treat single points with care.

## Agreement between tools

- **Coefficients:** every tool's b equals dfeols's to within 1e-6 relative on every
  specification, and most to within 1e-9. On Rideshare, all months, pyfixest and fixest are
  within 6e-11 and reghdfe within 1e-11.
- **N:** pyfixest and reghdfe have exactly dfeols's N everywhere. fixest keeps a few more rows
  on 12 of the 19 specifications (at most 11,792, or 0.04%), because it runs fewer singleton
  passes. dfeols and pyfixest repeat them until none are left.
- **Standard errors:** pyfixest's equal dfeols's (on Rideshare, all months, the ratio is
  1.00000000). reghdfe's are identical where no FE level is redundant (Taxi 2009, Flood B,
  EVE). Elsewhere they are smaller by a constant factor (0.9973-0.9988x on Flood A, 0.99998x on
  Rideshare), because reghdfe takes redundant FE levels off the degrees-of-freedom count and
  dfeols counts every level. fixest's SEs differ by at most 3e-4 relative where its N differs,
  and 1e-7-1e-6 elsewhere.

None of these differences is an error. When you check dfeols against your own fixest,
pyfixest or reghdfe results, expect them.

## Load and preparation time

These are not in the fit times above.

- **pyfixest:** 1-4 s to load the columns needed with Polars (27 s for Rideshare, all months).
- **fixest:** 27-41 s on unfiltered specifications. Filtered ones took longer, up to 913 s
  (Rideshare, one month), because the R Parquet reader available on the cluster
  (nanoparquet) reads the filter column file by file. Rideshare, all months, took 334 s.
- **reghdfe:** Stata has no Parquet reader in this version, so each specification's data was
  first written as a `.dta` file (25 minutes for Rideshare, all months) and then loaded (37 s).
- **dfeols:** shard preparation took 645 s for Rideshare, all months. It is done once per
  partition key, and every later fit on that key reuses the shards.

## Choosing where to run

Each configuration here ran once, on one cluster, so use these as rules of thumb, not as
thresholds.

- **If a node that holds the data is free, use it with the local executor.** It runs the same
  code as a cluster run, so its results are the same. It was fastest on 7 of 10 groups, and at
  1.6 billion rows its whole process used 79.6 GB. That suggests a 128-256 GB node would do,
  though this was not tested.
- **Use many workers when the data won't fit in memory on a node you can get**, or when the
  large-memory nodes are queued. OSCER has three, and one snapshot showed 12 jobs waiting for
  them.
- **Below about 50M rows, a model that fits in memory runs just as well in fixest or
  pyfixest.** dfeols's fixed costs are seconds on one node but minutes on a cluster.
- **Size the worker count to the total work.** A fit of a few thousand worker-seconds gains
  nothing past 10-20 workers. Beyond 40, the gain depends on how many workers the queue gives
  you. `dfeols plan` counts the work. Treat its wall-time range as wide.
- **Leave slack in the wall time.** The same 40-worker Rideshare fit took 1,116 s one day and
  1,422 s another, 27% apart. Ask for about 1.5x the expected time.

## Billion-row synthetic panels

Synthetic panels of 1e9 rows with known parameters, on 40 four-core OSCER workers (about 36
on average):

| | 10M units × T=100 | 1M units × T=1000 |
|---|---|---|
| fit `unit + time` | 178 s | 91 s |
| fit `unit + group^time` | 158 s | 157 s |
| peak worker memory | 3.4 GB | 2.8 GB |
| prep by unit / by group (distributed) | 700 s / 543 s | 524 s / 1300 s |
| recovers the truth | yes | yes |

At **1e10 rows** (100M units × T=100, 5,000 shards per set, up to 40
workers), both specs recovered the truth (`unit + time`: max |z| 0.35;
`unit + group^time`: max |z| 1.79). The fits took 47 and 59 minutes, peak
worker memory was 4.3 GB, and the shard sets took 39 minutes (by unit) and
74 minutes (by group, through the two-level shuffle) to prepare: about 3 hours
40 minutes end to end. At this size no in-memory tool fits on any single node.

Reproduce with `dfeols bench --profile <name> --preset units10m_t100` (or
`units1m_t1000`); see [Workflow on a cluster](guide/workflow.md#scaling-to-a-billion-rows-and-beyond)
for resource budgets and the 1e10-row stress presets.

## Performance notes

- Python is not the bottleneck: cross-products run in BLAS, demeaning in
  pyfixest's compiled core, I/O in Arrow/Polars.
- FEs local to the shard are demeaned **exactly, without iteration** when
  there are one or two of them. Two FEs use a Schur-complement solve: the FE
  with more levels per connected component is eliminated in closed form, and
  the other is solved with a small dense Cholesky (LAPACK) per component.
  On the influencer `shape` spec (spell + account × week) this cut demeaning
  from 168 s to 6 s per 600k-row shard and agrees with iteration to 1e-14.
  `fe_method="iterate"` forces iteration; three or more local FEs always
  iterate.
- Crossing FEs (e.g. time in unit × time, partitioned by unit) never build
  dummy columns when there is one local FE: their blocks of X'WX come from
  per-level sums, and the vcov uses Frisch–Waugh–Lovell on the reported
  coefficients. With several local FEs the dummy block's projection is solved
  per connected component of the local FEs (a Schur complement, as for exact
  two-way demeaning); `crossing_fe="auto"` uses it above `max_dummy_levels`. At T=1000 the dense path ran out of memory on a 2M-row panel;
  the sparse path took 5 s.
- Keep `fe_tol` tight (default 1e-10). At pyfixest's default 1e-6 a two-FE
  within-account model stopped early and was off by up to 2.2 in testing.

## Acknowledgement

The computing for this project was performed at the
[OU Supercomputing Center for Education & Research (OSCER)](https://www.ou.edu/oscer) at the
University of Oklahoma (OU).

We also thank the following OSCER personnel for their support:

- Henry Neeman, Director
- David Akin, Senior System Administrator
- Thang Ha
- Tim Huber, Research Computing Facilitator
- Horst Severini, Associate Director for Remote & Heterogeneous Computing, Research Computing
  Facilitator
