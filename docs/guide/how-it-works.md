# How it works

The data are split into shards by a **partition key**: all rows with the same value of
the key go to the same shard. Each fixed effect (FE) is then handled according to how it
relates to that key:

| FE relative to the partition key | handled as |
|---|---|
| nested in it (e.g. `unit` when partitioning by unit) | demeaned inside each shard: exactly for one or two such FEs, iterated for three or more |
| crossing (e.g. `time`; `carrier^quarter`) | dummy regressors; with one local FE (or several, when a crossing FE exceeds `max_dummy_levels`) their cross-products come from per-level sums, so no dummy columns are built, and the coordinator solves the dummy block (up to 5,000 dummies under `crossing_fe="auto"`; `"sparse"` takes it to about 30,000, though CG is usually far faster there). Otherwise dense dummy columns, up to `max_dummy_levels`=2000 |
| crossing, many levels (above 5,000 dummies, or above `max_dummy_levels` with no local FE) | conjugate gradient across shards (the CG solve, `crossing_fe="iterate"`): A = D'WD is built in round 1 and solved on the coordinator when it fits `level_space_gb`, else one map/reduce round per step. In level space the product with A runs on several threads, and a 10-step trial picks the block preconditioner or plain Jacobi by which reaches the tolerance sooner |

A fit is two map/reduce rounds over shards: (1) demean and return `X'WX`,
`X'Wy`; solve for β on the coordinator; (2) residuals and the sandwich meat.
Worker memory scales with shard size, not total N.

When the data compress (many rows per combination of every FE and the
cluster), `fit(..., cells=True)` instead fits exactly from per-cell sums in
one pass (10.3M DB1B rows in 329,000 cells: 9.8 s against 27 s): OLS and IV, with iid and
one- or two-way CRV1 SEs.

**Choosing the partition key.** Pick the key the most FEs are nested in.
Unit × time: partition by unit (unit FE local, time as dummies). With
`unit + group^time` and units nested in groups: partition by group (both FEs
local). The same model under different keys gives identical results
(tested). `dfeols info <shard_dir>` shows how each FE was classified.
