# Progress and ETA

Each fit writes `status.json` and `history.jsonl` to its run directory. Before the first
stage starts, the log lists the stages:

```
09:00:02 fit H:L3: plan
  - drop singletons: passes over the shards until one drops nothing (the last fit of these FEs took 7 passes)
  - demean + X'X: one pass over the 27,011 shards, demeaning within the local FEs and summing the cross-products
  - FE solve: conjugate gradient for the crossing FEs (cell, donor; 1,234,567 levels) to a residual of 1e-08; on the coordinator if their matrix fits, else a pass over the shards per step (the last fit of these FEs took 14 steps)
  - residuals + SEs: one pass, each shard's residuals and its share of the standard errors
  - the last fit of these FEs (2026-11-20T04:12:00Z) took 9:12:40 at 40 workers
```

The "last fit" notes come from the timings record (`timings.py`), when an earlier fit
absorbed the same FEs on the same data. Then a row per milestone:

```
run             stage                   shards    rows  workers    rows/s  stage left  all left  done at
cellxgene:H:L3  14/15 FE solve step 4  8911/27011  33.0%    23/40    29.33M     3:25:15  11:02:21    20:04  (1:58:01 at 40 workers)  solve step 3: residual 1.2e-04, stops at 1e-08, about 5 more steps
```

| Stage | What it does |
|---|---|
| `singletons pass N` | drops rows whose FE level appears once, until a pass drops nothing (only when an FE crosses the partition) |
| `demean + X'X` | each shard demeans within its local FEs and sums the cross-products |
| `FE solve step N` | a conjugate-gradient step for the crossing FEs, when each step is a pass over the shards |
| `residuals + SEs` | each shard's residuals and its share of the standard errors |

While stages are still to come whose count is unknown (the singleton passes, and the CG
steps until three steps give an estimate of the steps left), the row shows no stage count,
and `all left` and `done at` show `-`. `status.json` names the stages `singletons1`,
`round1`, `cgN` and `round2`.

`dfeols status <run dir or shard dir> --watch 60` shows it from any login node.
The ETA divides remaining work (from measured seconds per row) by the workers
connected now, never less than the largest shard still waiting, with a
bootstrap band; round 2 is projected from round 1's timing until measured.
