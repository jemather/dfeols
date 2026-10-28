# Design: `dfeols plan` (probe the FE structure, pick the path, size the job)

Status: draft for review (2026-10-27). Nothing here is implemented yet. Plan A (crossing FEs
above `max_dummy_levels` on the sparse path, 04ff7a7) is the only piece of the solver table
below that exists beyond the paths dfeols already had.

## Goal

Before anything is queued, tell the user how hard a specification will be and how to run it:
which partition key, which solver path, how much memory on workers and coordinator, how many
passes over the data, a wall-time range, and whether to run it as one job or as a demean-first
job whose output later fits reuse (the persisted demeaned data of `sessions.md`). Today the
user settles all of this implicitly by choosing `partition=` at prep time, and finds out it was
wrong from a NotImplementedError, a worker killed for memory, or a job that hits its walltime.

The probe is cheap: the level and pair counts below took 14 s on NFIP (75M rows) and 34 s on
DB1B's market table (683M rows) as Polars streaming queries on one node (2026-10-27).

## Workflow

Illustrative output: the level counts are D4's real ones (2026-10-27); group sizes, memory and
times are made up.

```text
$ dfeols plan 'raw/db1b/market/*.parquet' \
      --fe "carrier^route + carrier^quarter" --cluster route --vars lfare,hhi,lpax \
      --partition route,carrier --profile oscer
  683,414,514 rows, 3 variables + 1 outcome
  FE                 levels    nested in: route   carrier
  carrier^route     760,576               yes      yes
  carrier^quarter     7,474               no       yes
  route             193,758 (cluster)     yes      no
  partition    largest group   crossing FEs            path                     coordinator
  route        2.1M rows       carrier^quarter 7,474   sparse, dense solve (A)  ~3 GB, ~10 s
  carrier      61M rows (!)    none                    local                    -
  -> partition by route. One job: 1 fit pass + 1 pass for SEs; ~350 shards of 2M rows;
     workers ~4 GB each (64 GB nodes fine); est. 25-50 min on 20 workers.
```

The same command with `--fe "carrier^route + carrier^quarter + origin^quarter"` would report
two crossing FEs (72k levels together), a crossing block that is sparse and block-diagonal by
quarter, and the path the solver table picks for that (sparse direct solve, or iterative).

## What the probe measures

All from streaming aggregates on the raw (or prepped) data; nothing is sharded.

1. **Rows and missingness** of every variable in the spec (the sample after listwise deletion).
2. **Levels** of every FE, interaction and cluster variable.
3. **Nesting** between every FE or cluster and every candidate partition key: A is nested in K
   when `n_unique(A) == n_unique(A, K)`. Also FE-in-cluster nesting (degrees of freedom).
4. **Partition skew** per candidate key: rows in the largest groups, against the shard target.
   A key whose largest group exceeds what a worker can hold is flagged (state partitions put
   ~25M NFIP rows in Florida's shard).
5. **Crossing structure** for each candidate key: which crossing levels share a nested group
   (the nonzero pattern of the reduced dummy block `D'WD - C'diag(1/W_a)C`), its block
   structure and fill, and connected components of the FE graph (identified levels; a hint of
   how many iterations an iterative path needs).
6. **Singletons**, one pass's worth (enough to say whether they matter).

Candidate keys: the ones the user lists, else every FE component and cluster variable.

## The solver table

For each candidate key the planner picks the first path that applies:

| Structure (given the key) | Path | Passes over the data | Status |
|---|---|---|---|
| every FE nested | local demeaning in each shard | 2 | exists |
| crossing FEs ≤ 2,000 levels, any nested FEs | dense dummy columns | 2 | exists |
| one nested FE, crossing block ≤ ~30k levels | sparse path, dense solve on the coordinator | 2 | plan A, done |
| several nested FEs, crossing block ≤ ~30k | sparse path, nested FEs projected in the shard | 2 | plan B |
| crossing block large but sparse with little fill | sparse direct solve on the coordinator | 2 | to build |
| anything else | alternating projections across shards | 2 + iterations | plan C |

Every row except the last is a fixed number of passes. Iterations of the last depend on the
structure; the probe's connectivity statistics give a range, not a number.

## One job or demean-first

The planner recommends one of two shapes and says why.

**One job** (`dfeols submit` running `fit`): the right choice when the path has a fixed number of
passes and the estimate fits well inside the walltime, or there are only a few specifications.

**Demean-first** (the sessions design's persisted demeaned data): a job that residualises every
declared variable on all FEs and writes the result to scratch with its manifest; later fits
are one or two cheap passes over those columns. Recommended when any of these hold:

- the path is iterative (plan C), so the expensive part is the demeaning, and it should be done
  once for all specifications that share the FEs, sample and weights;
- the estimate approaches the walltime (48 h on OSCER), so the work has to checkpoint and resume
  across jobs;
- many specifications share one FE set (the sessions use case, at any path).

## Plan C on OSCER

Alternating projections distribute as map/reduce rounds: nested FEs are demeaned inside each
shard with no communication; for a crossing FE each shard returns per-level sums and counts,
the coordinator combines them into means, and broadcasts them back. One cycle is one round;
all variables go through it together.

- **Data in memory.** A cycle that re-reads Parquet from Lustre costs about a round-1 time
  (minutes at 683M rows); with the shards held in worker memory it is a memory-speed pass
  (seconds). The planner sizes the pool so the data fits: D4's market table at ~10 float64
  columns is ~55 GB, which is a few 250 GB nodes or a dozen 64 GB nodes. dfeols re-reads
  shards each round today, so keeping them resident is part of building C.
- **Reductions.** Per cycle each shard sends levels × variables × 8 bytes: at 64k levels and 10
  variables, 5 MB per shard, 1.5 GB per cycle over 300 shards into one coordinator. That needs
  a tree reduction across workers (Dask can do this), and is where the FDR10 InfiniBand nodes
  pay off.
- **Convergence and walltime.** A tolerance on the change in the demeaned columns, checked at the
  coordinator each cycle, a cap on cycles with a clear message, an acceleration step (Irons-Tuck
  as fixest, or conjugate gradient; each step is still one pass), and a checkpoint of the
  demeaned columns every N cycles so a run can resume in a new job.

## Node classes come from the profile

The planner never hard-codes a cluster. The site profile describes node classes; for OSCER (Ted,
2026-10-27): 278 normal nodes with up to 250 GB RAM, 267 dual-Haswell nodes limited to 64 GB on
an FDR10 InfiniBand switch, and a 48-hour walltime limit. The recommendation names a class for
the coordinator (a big node when the dense solve needs tens of GB, e.g. ~43 GB at 30k levels)
and for workers (many 64 GB nodes when shards are small; 250 GB nodes when one partition group is
large), plus the walltime to request.

## Wall-time estimates

Rows × passes × columns over a throughput for this profile and path, taken from the local record
of earlier runs that `sessions.md` proposes (`<scratch>/dfeols/timings.jsonl`), shown as a
range; with no history, a wide range and a suggestion to calibrate on a few shards. Iterative
paths multiply by the iteration range.

## Interfaces

- `dfeols plan SOURCE --fe ... [--cluster ...] [--vars ...] [--partition K1,K2] [--profile P]`:
  prints the report; `--json` writes it for scripts. Runs on a login or interactive node
  (streaming, bounded memory), or as a short job for very large sources.
- `dfeols.plan(source, formula, ...)` returns the same as a dict.
- `prep` and `fit` log the planner's verdict for the spec they were given, and `fit` refuses
  early, with the planner's explanation, where it now raises deep inside a run.

## Build order

1. Probe and report: levels, nesting, skew, the solver table and memory (no wall time yet).
   Delivers the pre-SLURM scale estimator Ted asked for (STATE.md, 2026-10-27).
2. Crossing structure (fill, block structure, components) and node-class recommendations from
   the profile.
3. Wall-time ranges from recorded runs.
4. One job vs demean-first, once sessions' persisted demeaned data exists.
5. New paths as they are built (B, sparse direct, C), each added to the table with its own
   estimate.

## Open questions

- Should `prep` take `partition="auto"` and use the planner's pick, or only report it?
- How far should the probe go on its own for candidate keys when none are listed (every FE
  component can be many queries on a large source)?
- Thresholds for recommending demean-first: number of specifications, fraction of walltime.
