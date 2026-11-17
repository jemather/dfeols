# Benchmarks and performance

Synthetic panels of 1e9 rows with known parameters, on a Slurm cluster (OU
OSCER) with up to 40 four-core workers (about 36 on average):

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
40 minutes end to end.

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
