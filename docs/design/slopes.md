# Varying slopes: `fe[x]`

Status: built 2026-11-10 (`tests/test_slopes.py`), with Ted's answers under Questions.

## Syntax (fixest's)

| Term | Means | Columns absorbed per level g of `fe` |
|---|---|---|
| `fe[x]` | FE plus a slope on x per level | 1, x |
| `fe[[x]]` | slope only, no FE intercept | x |
| `fe[x, z]` | FE plus slopes on x and z | 1, x, z |
| `a^b[x]` | slopes on the interacted FE | 1, x |

`fe` must be an FE that `prep()` encoded, and x a numeric column in the shards. Neither
needs a new prep, because the slope variable is just another column that shard tasks read.

## What fixest does (0.14.2, probes on simulated data, 2026-11-10)

Checked against `lm` with `factor(id) + factor(id):z`:

- **Coefficients:** equal to lm for `id[z]`, `id[z] + t`, weights, and `id[[z]]`.
- **`id[[z]]` alone keeps an intercept**, reported as the coefficient `(Intercept)`. b equals
  lm *with* an intercept, and K counts it.
- **Singletons:** dropped for FEs with an intercept (`id[z]` drops 2 singleton rows),
  not for slope-only FEs (`id[[z]]` drops none). A two-row group with a slope fits
  perfectly, and fixest keeps it.
- **K (small-sample df):** every slope coefficient counts, one per level and slope variable,
  even when it is not identified. A group with z constant or z = 0 still adds 1, so the
  lm rank is 100 and fixest's K is 101, and fixest's SE differs from lm's in the 4th digit.
- **Nesting in the cluster:** only the FE's intercept levels are discounted. With
  `id[z]` and cluster `id`, K = 1 (x) + 1 (id nested) + 50 (slopes) = 52.
  `id[[z]]` is never discounted.
- Collinear regressors (`y ~ x + z | id[z]`) are dropped, as for any absorbed column. So is
  x in `y ~ x | t[x]`: the slopes of every level sum to x.
- **An older difference, not about slopes:** with two FEs nested in the cluster (`id_a` and
  `spell` inside it, clustered by `id_a`), fixest counts the second one's levels (K 2736)
  and pyfixest does not (K 3). dfeols has always followed pyfixest there, and the slope part
  of K is the same in both (`id_a[week] + spell`: 2796 = 2736 + 60). The tests avoid
  clustering on such a pair.

fixest's convention is what dfeols already follows, so the proposal matches it, including
the count of unidentified slopes.

## How it fits dfeols's FE paths

**Local `fe[x]`** (fe nested in the partition key, e.g. `unit[t]` partitioned by unit) is
the common case: unit-specific trends. It is demeaned inside the shard, where the work is:

- **One local FE with slopes:** exact. Per group, a weighted projection of every column on
  [1, x...] (a (1+s)×(1+s) solve per group, vectorized by group size). Groups where the
  matrix is singular (x constant) use a pseudo-inverse, which leaves the same residuals
  as dropping the slope.
- **Several local FEs, any with slopes:** alternating projections, with the slope FE's
  projection as one of the sweeps (dfeols's NumPy demeaner, accelerated as now).
  pyfixest's compiled core has no slopes, so this path is slower than plain FEs.
  The exact two-way Schur solve can be extended later if it matters.

**Crossing `fe[x]`** (fe crosses partitions, e.g. `year[x]` or `region[t]` partitioned by
firm) adds columns D_g·x next to the FE's dummies:

- **Dense dummies** (crossing FEs up to `max_dummy_levels`): the slope columns are more
  dummy columns. Small change.
- **Sparse path and plan C** (per-level sums, CG): would need the per-level sums weighted by
  x, i.e. new blocks in A and new matvecs. Proposed later, if a dataset needs it. Until then
  the fit refuses with a clear NotImplementedError naming the FE.

**Other parts:**

- Parser: `fe[x]`, `fe[[x]]`, `fe[x, z]`, `a^b[x]`. The FE's key stays `fe`, so prep's
  encoding, nesting flags and clusters are found as now.
- Singletons: only FEs with an intercept count, as in fixest.
- df: slopes are added to K in full, without nesting discounts (`vcov.df_k`).
- With only slope-only FEs the constant is a reported regressor `Intercept` (fixest's
  `(Intercept)`). A slope-only local FE next to crossing dummies keeps the hidden constant
  that stands in for the dummies' dropped levels.
- The collinearity scale (`reference_ss`) is each column's uncentered sum of squares when no
  local FE has an intercept, since then the FEs do not remove the mean.
- Records: each slope FE goes in `fixed_effects` with `components` (e.g. `["unit", "t"]`)
  and its own levels, so estrec's schema needs no change.
- `cells=True` refuses slopes (cells are OLS on discrete cells; a slope makes them continuous).
- IV takes slopes unchanged, since the demeaning is the same for every block.
- Sessions refuse slopes with a message for now. Adding them means the session's demeaning
  (`session.py`) and its cache key take the slope variables.
- Planner: counts the slope columns in the dummy block, takes the dense path with slopes,
  and marks keys where a crossing FE is over `max_dummy_levels` as not built.

## Tests

- fixest via R (`Rscript`, as the R client tests already do) is the reference, since pyfixest
  has none: `id[z]`, `id[[z]]`, `id[z, t]`, `id[z] + t`, local and crossing,
  weights, iid / hetero / CRV1 (nested and not), two-way clustering, IV, a constant-x group,
  and partition invariance (a unit-partitioned fit equals a group-partitioned one).
- `demean` unit tests: exact one-way slopes equal the lm residuals; iteration equals exact.

## Questions for Ted (answered 2026-11-10)

1. **Scope of the first step:** local slopes (exact or iterated) and crossing slopes as
   dense dummies only; sparse and plan C refuse for now? **Yes.**
2. **Unidentified slopes in K:** match fixest (count every slope, even a constant-x group's),
   or count only identified ones (matches lm, differs from fixest)? **Match fixest.**
3. **Reference:** fixest in R in the test suite (skip when R or fixest is missing, as the
   R-client tests do)? **Yes, plus Python checks that always run.**

## Result

19 fit specs equal fixest 0.14.2 (coefficients to 1e-9 relative, SEs 1e-8, N, t df and
within R²): `spell[tenure]`, `spell[[tenure]]`, `id_a[week] + spell` (iterated),
`spell + week[feat]`, `spell + week[tenure]`, `spell + week^tbin[[posts]]`,
`id_a[feat, week] + week`, `week[[posts]]`, `id_a[[week]] + spell`, and IV with
`spell[tenure]` (+ `week`), under iid, HC1, CRV1 nested and crossing, two-way CRV1 and
weights. The demeaner equals dummy regressions to 1e-11 on six FE patterns. A slope FE gives
the same fit whether local (partition id_a) or crossing (partition week), and with or without
the demeaned cache.

Later, if a dataset needs it: the sparse path and plan C (per-level sums weighted by the
slope variables), slopes in sessions, an exact two-way solve with slopes.
