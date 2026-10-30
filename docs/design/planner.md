# Design: `dfeols plan` (probe the FE structure, pick the path, size the job)

Status: draft for review (2026-10-27; Ted's answers folded in 2026-10-28). Build step 1 (probe
and report) is implemented in `src/dfeols/planner.py`, and `config init --from-slurm` in
`src/dfeols/slurm_site.py` (2026-10-28); the rest is not. Plan A (crossing FEs
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

Candidate keys: the ones the user lists, else every FE, every component of an interaction, and
every cluster variable. Specifications rarely have more than 3 or 4 FEs, some with interactions
or nesting (Ted, 2026-10-28), so that is a handful of keys and the probe tries them all: the
nesting counts for every FE × key pair share one pass over the data.

## The solver table

For each candidate key the planner picks the first path that applies:

| Structure (given the key) | Path | Passes over the data | Status |
|---|---|---|---|
| every FE nested | local demeaning in each shard | 2 | exists |
| crossing FEs ≤ 2,000 levels, any nested FEs | dense dummy columns | 2 | exists |
| one nested FE, crossing block ≤ ~30k levels | sparse path, dense solve on the coordinator | 2 | plan A, done |
| several nested FEs, crossing block ≤ ~30k | sparse path, nested FEs projected in the shard | 2 | plan B, done |
| crossing block too large to solve directly, A fits the coordinator | level space: A built in round 1, preconditioned CG on the coordinator | 2 | proposed (below) |
| anything else | conjugate gradient across shards, with a slow-convergence warning | 2 + iterations | plan C, done; warning to build |

Every row except the last is a fixed number of passes. Iterations of the last depend on the
structure; the probe's connectivity statistics give a range, not a number. Level space
replaces the earlier "sparse direct solve on the coordinator" row: preconditioned CG on A took
seconds wherever A fit, without a factorisation's fill (see Crossing-FE solver).

## One job or demean-first

The planner recommends one of two shapes and says why.

**One job** (`dfeols submit` running `fit`): the right choice when the path has a fixed number of
passes and the estimate fits well inside the walltime, or there are only a few specifications.

**Demean-first** (the sessions design's persisted demeaned data): a job that residualises every
declared variable on all FEs and writes the result to scratch with its manifest; later fits
are one or two cheap passes over those columns. Recommended when any of these hold:

- the path is iterative (plan C), so the expensive part is the demeaning, and it should be done
  once for all specifications that share the FEs, sample and weights;
- the estimate exceeds 50% of the site's walltime limit (Ted, 2026-10-28), so the work should
  checkpoint and be able to resume in a new job. The limit is the profile's `max_walltime`
  (48 h on OSCER), never a constant in the code;
- several specifications share one FE set, sample and weights (the sessions use case, at any
  path). The planner suggests demean-first when `plan` is given more than one specification
  with the same FEs, and `fit` warns when it sees a second specification on FEs it has just
  absorbed for the same sample (it compares the persisted-data key of `sessions.md`), naming
  the demean-first command that would reuse the work. A suggestion, not a switch (Ted,
  2026-10-28).

**Demeaned data is kept by default** (Ted, 2026-10-28). Absorbing the FEs is the expensive part,
so every run that demeans, a one-job `fit` included, writes the demeaned columns and manifest to
the profile's scratch (`<scratch>/dfeols/demeaned/`, as in `sessions.md`) unless told not to
(`persist=False`), and a later run with the same key reuses them. Removal is explicit
(`dfeols demeaned list` / `dfeols demeaned rm <key>`). Whether storage is plentiful is a site
fact: the profile can say so (`demeaned_storage = "plentiful"`, the default, as on OSCER where
temporary storage is ample and free) or ask for confirmation above a size (`demeaned_confirm_gb
= 500`) where scratch is scarce or charged. A site that purges scratch says so in the profile
(`scratch_purge_days`; OSCER deletes scratch files after 14 days, Ted 2026-10-28). The manifest
records when persisted data was written, `plan` and `dfeols demeaned list` show when each key
will be purged, and `plan` budgets a re-demean when the key it would reuse expires before the
job can start. dfeols never touches files to dodge a purge.

## Plan C on OSCER

**As built (2026-10-28, `crossing_fe="iterate"`).** Not alternating projections on the data
but conjugate gradient on the crossing-FE coefficients, which needs less resident state and
fewer rounds. Round 1 returns D'W[X~ Y~] (per-level sums of the locally demeaned columns) and a
Jacobi diagonal; the coordinator solves (D~'W D~) Gamma = D~'W[X~ Y~], one column per regressor
and outcome, where each step broadcasts a levels x columns block p and every shard returns
D'W Q D p (Q its exact local demeaning: one FE closed form, two Schur, more iterate). Shards
keep their codes and weights, not the data, in worker memory between steps (`DFEOLS_ITER_CACHE_GB`,
default 8; a worker without them re-reads the shard), and the Dask executor scatters p once
per step instead of once per task. beta follows by Frisch-Waugh-Lovell on the coordinator, and
round 2 is the sparse path's, so every vcov works. Equal to pyfixest and to the dense-solve
path; on a 1M-row worker panel with 8,000 firms it took 13 steps (5 s, against 11 s for plan
A's dense solve). Not built yet: IV on this path, a tree reduction for the per-step sums, and
checkpoints (a step's state is the coordinator's few P x c blocks, so resuming is cheap to
add). The sketch below was the starting point.

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

## Crossing-FE solver (proposal, 2026-10-30)

From a bench of plan C's solve against alternating projections and other preconditioners on
DB1B's market fit, NFIP (two specs) and synthetic worker-firm panels (0.5/2/10% mobility).
The bench solves each spec exactly on its cells and reproduces OSCER job 35564818 to 2e-12 in
beta; results and code are in the project notes (`/srv/dfeols/docs/cgbench.md`,
`scripts/cgbench.py`), outside the repo. Parts marked *decided* are Ted's; the rest is for review.

**Two modes.** Plan C as built makes one pass over the shards per CG step: 293 steps, 90 of the
job's 94 minutes on 20 workers.

- *Level space* (to build): build A = D~'W D~ once and run CG on the coordinator with no further
  passes. Each shard returns its part, a sum over its local groups g of
  D_g'W_g D_g - (D_g'w_g)(w_g'D_g)/N_g (one nested FE; Q's other cases as plan C's matvec), and
  can do so in round 1, next to D'W[X~ Y~], so the path stays at two passes. DB1B: A is 66,821
  levels with 57M nonzeros (652 MB), and the solve takes 12 s instead of 90 min.
- *Fallback* (plan C as built): one pass per step, used when A would exceed the coordinator's
  memory budget. *Decided:* it warns, before the run and in the log, that convergence may be
  slow and names what made A too large.

**Size of A.** The probe's bound, the sum over local groups of (crossing levels the group
touches)^2, overstates A by about 110x on DB1B (6.4e9 against 57M), so it can only rule level
space in, not out. Above the budget by the bound, the build counts A's nonzeros exactly in row
chunks and stops at the budget, falling back with the warning.

**Partition.** *Decided:* the planner picks the partition key that makes the crossing system
smallest, subject to shard balance. NFIP B (`| tract + zone^month + community^month`) has 1.72M
crossing levels and A of ~3e9 nonzeros when sharded by tract; no fallback method reached 1e-8 in
2,000 passes. Sharded by community (largest group 3% of rows), community^month is nested, the
crossing system is tract + zone^month (235k levels, A 226M nonzeros), and it solves in level
space: the same beta, SEs and RSS.

**Preconditioner.** Chosen from the structure of each crossing FE's own block A_kk, which the
build has in hand: the sizes of its connected components (levels linked through local groups).

| A_kk components | Level space | Fallback (passes per step) |
|---|---|---|
| all small (<= ~400 levels), or few and dense | AP-sweep: symmetric block Gauss-Seidel | block-Jacobi (1; the AP-sweep costs 3) |
| some large and sparse (worker-firm graphs, NFIP's tract block) | block-Jacobi or AP-sweep, AMG on the large components | block-Jacobi with AMG blocks |
| neither affordable | Jacobi (as built) | Jacobi |

Steps to 1e-10: DB1B Jacobi 294, block-Jacobi 40, AP-sweep 19; NFIP A 537, 17, 8; synthetic
(0.5% mobility) 91, 22 and 18 with AMG blocks. Two rules from the bench:

- *Block solves are pseudo-inverses, never regularised.* Each A_kk is singular, one constant per
  component, absorbed by the local FE. Pin one level per component and remove the component mean
  before and after the solve. A + 1e-10 I stalled CG near 1e-6 on NFIP A.
- *AMG blocks use plain aggregation* (pyamg `smooth=None`, one V-cycle). The default smoothed
  aggregation filled in 293x on the 0.5%-mobility firm block (4 s a cycle, against 13 ms).
  Exact sparse LU of such blocks took over 185 s to set up; AMG setup is under a second.

**Tolerance.** *Decided:* default 1e-8 on the relative residual (was 1e-10, next to the
floating-point floor of ~1e-11 on DB1B). beta settles well before: within 1e-4 SE after 3 to 9
steps of the AP-sweep or block-Jacobi on DB1B and NFIP A.

**Alternating projections are not adopted.** In passes, AP and Irons-Tuck reach 1e-2 SE as fast
as anything (about 8 on DB1B and NFIP A) but crawl after: never within 1e-4 SE in 6,000 passes
on NFIP A and B, and 1,724 passes for 1e-6 SE on DB1B with Irons-Tuck against 24 for
block-Jacobi CG. At 0.5% mobility AP needed 3,114 passes to 1e-4 SE; Jacobi CG 37.

**Separate fixes the bench found.** Workers use one Dask thread of the cores they are given
(4 on OSCER: 60 of 80 idle); the record labels plan C's numerics `one-pass group demeaning`,
`Cholesky`, `iterative: false`.

**Fitting from cells.** Every FE and the cluster are constant within a cell (a distinct
combination of their keys), so the fit, CRV1 included, can run on per-cell sums and
cross-products. DB1B's 669M rows are 21.3M cells; the whole fit took 2.5 min on one machine
(one streaming pass 58 s, system 81 s, solve 14 s; peak 15 GB). The gain is data-dependent:
NFIP compresses 3-14x, the worker-firm panels not at all. The probe could report the cell count
(one group-by) and recommend it when the ratio is large. *Decided:* pursue it, with CRV1 at a
cluster in the cell key; HC1 is not needed (users of large FE fits cluster).

## Site facts come from the profile

The planner never hard-codes a cluster: the walltime limit, the node classes and what each costs
are site facts, so they live in the site profile (`site.py`), next to today's single `worker`
and `coordinator` specs, which stay the defaults when a profile declares no classes. Proposed
additions:

```toml
[profiles.oscer]
max_walltime = "48:00:00"          # longest general job; drives checkpointing and the
                                   # demean-first threshold (50% of it). A class may override it
billing = "none"                   # what the site charges: none | cores | nodes | tres
                                   # (OSCER: none, Ted 2026-10-28)

[profiles.oscer.nodes.haswell32]   # one table per node class
partition = "32gb_20core"          # what selects it (partition, or constraint = "...")
count = 299
cores = 20
memory = "31916MB"                 # the most a job can request (MaxMemPerNode); a mixed class
                                   # gives the smallest it can promise
interconnect = "FDR10"             # "" if unknown
availability = "high"              # high | medium | low; optional, the planner prefers measured
                                   # queue waits from earlier runs when it has them

[profiles.oscer.nodes.hdr256]
partition = "256gb_64core"
count = 26
cores = 64
memory = "257442MB"
interconnect = "HDR100"

[profiles.oscer.nodes.long]
partition = "longjobs"
count = 60
cores = 20
memory = "31894MB"
max_walltime = "7-00:00:00"        # per-class override
```

OSCER's draft profile, from `sinfo`/`scontrol` output Ted pasted (2026-10-28), is in
`/srv/dfeols/docs/oscer.md` (project notes, outside the repo): haswell32 (299 nodes, 20 cores,
32 GB, FDR10), haswell64 (35, 24 cores, 64 GB, FDR10), hdr128 (40, 64 cores, 128 GB, Ice Lake /
Rome, HDR100), hdr256 (26, 64 cores, 256 GB), largemem (3, 2-4 TB), and `longjobs` (60 mixed
nodes, 7-day limit). Three things in that output shape the design:

- **Walltime is per class.** 2 days on general partitions, 7 days on `longjobs`; the planner
  weighs a 7-day class against checkpointing across 2-day jobs.
- **No charge.** OSCER bills nothing per use (Ted, 2026-10-28); Slurm's `billing` TRES (the CPU
  count, no weights) is bookkeeping. With `billing = "none"` the ranking drops the cost column
  and orders options by fit, estimated wall time and availability; at a site that does charge,
  `cores`, `nodes` or `tres` gives the cost of each option.
- **Slurm can draft the profile.** Partitions, cores, memory caps, CPU and interconnect
  features, walltime and billing weights all come from `sinfo` and `scontrol`, so `dfeols
  config init --from-slurm` writes the node classes for any Slurm site (Ted, 2026-10-28) and
  leaves only policy (eligibility, preemption, purge) to the user. Per class it records the
  total nodes and the nodes in service (state alloc, mix or idle): on 2026-10-28 `normal` had
  278 of 367 in service, `normal_hdr100` 34 of 54. Totals describe the class; nodes in service
  bound what a job can get, so `plan` re-reads them (a cheap `sinfo`) rather than trusting the
  profile. Groups of partitions over the same nodes (OSCER's `normal`, `normal_avx2`, `all`,
  `sooner_test` ...) collapse to classes by node set and memory; policy partitions (`debug`,
  `ood_*`, GPU, `largejobs` with its 13-node minimum) are listed but left out of worker choices
  unless the user keeps them.

The planner matches the job's needs to the classes: the coordinator on a class with enough
memory for the dense solve (~43 GB at 30k levels); workers on the class that holds a shard (and,
for plan C, the resident data) at the lowest cost, preferring fast interconnect when the path
moves data between workers every cycle (plan C's reductions; HDR100 nodes may suit those best).
It ranks the feasible options rather than picking one silently, since a class can be cheap but
contended, or fast but expensive in the site's accounting:

```text
  workers   option                          est. wall    core-hours   availability
  1 (rec.)  24 x haswell32 (8 cores, 28 GB) 3.5-6 h      670-1150     high
  2         6 x hdr256 (16 cores, 120 GB)   2.5-4 h      240-380      low
  3         12 x haswell64 (12 cores, 60 GB) 3-5 h       430-720      medium
```

(Illustrative numbers.) The chosen option becomes the overrides for `dfeols.executor(profile,
...)`, so a recommendation can be run as printed.

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
- `prep(..., partition="auto")` runs the probe and uses the recommended key (Ted, 2026-10-28);
  it logs the alternatives with their trade-offs, and an explicit key still wins.
- `prep` and `fit` log the planner's verdict for the spec they were given, and `fit` refuses
  early, with the planner's explanation, where it now raises deep inside a run.

## Build order

1. Probe and report: levels, nesting, skew, the solver table and memory (no wall time yet).
   Delivers the pre-SLURM scale estimator Ted asked for (STATE.md, 2026-10-27). **Done**
   (`planner.py`): the group-bys run one at a time, since run together their hash tables
   peaked at 31 GB on DB1B's market table (7.5 GB one at a time, 32 s against 19 s). Plan C
   keys are reported as not built (plan B is now a path, b819291); "sparse direct or plan C" stays one row until
   step 2 measures fill.
2. ... `config init --from-slurm` **done** (`slurm_site.py`): it runs `sinfo -h -e -o
   "%R|%D|%c|%m|%l|%f|%G|%T|%N"` and `scontrol show partition -o`, or reads saved output
   (`--sinfo`, `--scontrol`). Classes are named after their partition (rename freely); the
   mixed `normal` and `normal_hdr100` come out as `use = false`, spanning the uniform
   classes, while `longjobs` stays usable for its 7-day limit. Tested on a fixture shaped
   like OSCER, not yet on OSCER's own output.
2. Crossing structure (fill, block structure, components); `partition="auto"` in prep; site
   profile additions (`max_walltime`, node classes, `scratch_purge_days`), `dfeols config init
   --from-slurm`, and ranked node-class options.
3. Wall-time ranges from recorded runs.
4. One job vs demean-first, once sessions' persisted demeaned data exists.
5. New paths as they are built (B, sparse direct, C), each added to the table with its own
   estimate.

## Decided (Ted, 2026-10-28)

- Walltime limit and node classes (memory, interconnect, cost, availability) are site profile
  settings, not code.
- `prep` takes `partition="auto"`; the planner recommends and lists the other options.
- Demean-first above 50% of the site's walltime limit.
- The probe tries every FE, interaction component and cluster variable as a key (specs have at
  most 3-4 FEs).
- Several specifications on one FE set: suggest demean-first in `plan`, warn in `fit`; no hard
  threshold.
- Demeaned data persists in scratch by default; deletion is explicit; storage policy (plentiful,
  confirm above a size, purge age) is a profile setting.
- `dfeols config init --from-slurm` drafts the node classes from Slurm.

Added 2026-10-28/29: crossing-FE tolerance 1e-8 by default; the partition key that makes the
crossing system smallest, subject to shard balance; the per-pass fallback warns that it may
converge slowly.

Added 2026-10-30: level space (A built in round 1, CG on the coordinator) with the preconditioner
chosen by block structure, as proposed above; AP not added; a cell path for data that compresses,
CRV1 only.

## Open questions

- None for OSCER. `hep_killable` is unknown and left out; the 11 unlabelled 256 GB nodes are
  taken as 64 cores / 256 GB, interconnect unknown (`docs/oscer.md`).
