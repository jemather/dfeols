# Formulas

`y1 + y2 ~ x + a:b + i(c) + x:i(c) + x:i(c, ref=0) | fe1 + fe2^fe3 | d1 + d2 ~ z1 + z2`

| Term | Meaning | Stata | fixest/pyfixest |
|---|---|---|---|
| `x` | continuous | `x` | `x` |
| `a:b` | product | `c.a#c.b` | `a:b` |
| `i(c)` | dummies, first level dropped | `i.c` | `i(c)` |
| `x:i(c)` | x times every level of c | `c.x#ibn.c` | `i(c, x)` |
| `x:i(c, ref=0)` | same, level 0 dropped | `c.x#i.c` | `i(c, x, ref=0)` |
| `\| a + b^d` | absorbed FEs | `absorb(a b#d)` | `\| a + b^d` |
| `\| a[x]` | FE a plus a slope on x per level of a | `absorb(a##c.x)` | `\| a[x]` (fixest only) |
| `\| a[[x]]`, `a[x, z]` | slopes only (no FE intercept); several slopes | `absorb(a#c.x)`, `absorb(a##c.(x z))` | the same |
| `\| d ~ z` | IV: d endogenous, z excluded instruments | `ivreghdfe y (d = z), absorb(...)` | `\| d ~ z` |

**Varying slopes** (`unit[t]`: unit-specific trends) follow fixest: the FE `unit`
must be encoded by `prep()` (its slopes need no new prep), only FEs with an intercept drop
singletons, every slope coefficient counts in K, and with only slope-only FEs the constant
is reported. On an FE nested in the partition key, slopes are projected out exactly per
group (one local FE) or by alternating projections (several, in NumPy, as pyfixest's
compiled demeaner has no slopes). With slopes in the model, crossing FEs are dense dummies,
up to `max_dummy_levels`; the sparse path, plan C, `cells=True` and sessions don't take them yet.

Multiple outcomes share one pass and a common sample (rows missing any
outcome are dropped). IV takes `estimator="2sls"` (default) or `"liml"`, and
`fuller=` for Fuller's modified LIML; `Result.iv["first_stage"]` has each
endogenous regressor's first stage (partial R², F tests under each vcov, the
effective F with one endogenous regressor), and each outcome the
overidentification tests.
