# Discrete-time hazard LPM: the Farbmacher-Tauchmann IV estimator

Status: built (`src/dfeols/dthazard.py`, `tests/test_dthazard.py`), 2026-11-08.

Source: Farbmacher, H. and Tauchmann, H. (2021), "Linear fixed-effects estimation with
non-repeated outcomes", FAU Discussion Papers in Economics 03/2021 (`raw/03-2021-1.pdf`).
Equation and table numbers below are the paper's.

## The problem

Unit i is observed in periods t = 1..T_i of a single spell at risk; y_it = 1 marks the
absorbing event (the last row of an uncensored unit), 0 otherwise. The DGP is the linear
hazard y_it = a_i + x_it b + e_it (1). Because y_{i,t-1} = 0 on every row in the sample,
Δy_it = y_it:

- **First differences** (5) regress y_it on Δx_it. The error's conditional mean is
  a_i + x_{i,t-1} b (6), so b_FD converges to G b + (bias from a_i) (7), with
  G = (ΣΔx'Δx)⁻¹ ΣΔx'x. b_FD is the *reduced form*; for stationary x, G → I/2 and b_FD ≈ b/2.
- **Within** (A.3): the error's mean involves E(x̄_i | survival to t), i.e. future x and
  T_i itself; biased in any direction (Table 1: 0.90, 0.94, 3.98 for a true 1) and no IV
  fix exists.
- **Pooled OLS**: biased if Cov(a_i, x_it) ≠ 0, plus survival bias.

## The estimator (8)

Just-identified IV of y_it on the levels x_it, with Δx_it as instruments, constant in both
stages (fn. 6), on rows t ≥ 2 of each spell:

    b_IV = (Σ_i Σ_{t=2}^{T_i} Δx_it' x_it)⁻¹ Σ_i Σ_{t=2}^{T_i} Δx_it' y_it  =  G⁻¹ b_FD

- The unit FE is **not** absorbed; Δx being uncorrelated with a_i in the population is the
  identifying assumption. What remains is survival bias only (9, 10), which shrinks with
  Var(a_i) and with the persistence of x (zero for a random walk).
- **Higher order** (3.2): b_IV,j uses Δ^j x_it as instruments (Δ² x_it = Δx_it − Δx_{i,t-1}),
  on rows with j predecessors. j = 1 is b_IV, j = 0 is pooled OLS.
- **Identification**: every regressor needs informative Δ^j. A regressor with Δx ≡ 0 in the
  sample (time-invariant, or the youngest-age dummy, which is 0 on all t ≥ 2 rows) makes G
  singular: an error, naming it.
- **Inference** (11-13): errors are heteroskedastic (LPM variance, unit-specific) but not
  serially correlated, so White SEs on the IV scores; the paper finds them accurate and
  probably conservative (Table 2). Clustering is still allowed (the application clusters by
  school).

## Extra fixed effects (checked 2026-11-08)

Time effects, age effects, region × time and so on can be absorbed as ordinary exogenous FEs
(levels instrument themselves) **exactly when a unit's cell at t−1 is determined by its cell
at t**. Then ΔD = D(I − P), with P mapping each cell to its predecessor; I − P is unit
triangular, so the instrument spans agree and the just-identified b is identical. Simulation
(`/srv/dfeols/scripts/hazard_fecheck.py`, outside the repo): time FE and group × time with fixed groups agree
to 1e-13; group × time with units moving between groups disagree (0.901 vs 0.953).

So the paper's specification (wave and age dummies, Table 3) is the same as absorbing `year`
and `age` as FEs. A unit-level FE (or anything nested in the unit) must **not** be in the
model: that is the within estimator again.

**Decided (Ted, 2026-11-08).** FEs that fail the predecessor condition (movers) are allowed,
absorbed as exogenous in levels, with a warning giving the share of rows whose cell's
predecessor is ambiguous. Differencing high-dimensional dummies is a different, much harder
estimator (no FWL shortcut), so it is not offered. A unit-level FE is refused.

## In dfeols: its own function

Ted 2026-11-08: its own estimator, not a recipe on the IV path. Internally it reuses the IV
algebra (iv.md): x is the endogenous block X₂, Δ^j x the excluded instruments Z₂, the extra
FEs partialled out as usual. What is new:

1. **Sample builder (per shard).** Shards must be partitioned by the unit (or by a key the
   unit nests in), so each spell's rows are in one shard; checked from `dfeols_meta.json`,
   error otherwise. Per shard: sort by (unit, time); check y is 0/1 with at most one 1, on
   the unit's last row; form Δ^j x from the unit's own rows at t−1..t−j; drop the first j
   rows of each spell and any row whose window has a gap (time not consecutive); count
   every drop by reason, as singletons are counted.
2. **Fit.** One IV fit on the built sample; the first-stage, reduced-form and OLS blocks are
   in the same cross-products, so the result also carries b_FD (reduced form), G (the
   first stage) and the first-stage diagnostics at no extra pass.
3. **Record.** An `iv/0.1` record (endogenous = all x, instruments = their Δ^j), plus the
   method, j, the unit and time keys and the drop counts in the dfeols extension.

Built samples are cached like the demeaned data (keyed by dataset, unit, time, j and the
columns), so refits and sessions don't rebuild.

**API (decided 2026-11-08).**

    dfeols.lpm_dthazard(ds, "exit ~ x1 + x2 | year + region^year",
                  unit="order_id", time="day", diff=1,
                  vcov="hetero", cluster=None, after_event="error")

- formula: the usual `y ~ x | fe` (no IV part; every x is instrumented by its own Δ^j).
  `i(c)`, `i(c, ref=)` and `x:i(c)` terms (Ted 2026-11-08: needed): estimated on the levels
  present in the sample, each dummy instrumented by Δ^j of itself; levels that never occur in
  the sample (the youngest age, seen only in first periods) get no coefficient and are listed
  as `excluded_levels`, as the paper drops age 53
- `diff=j` (default 1)
- `levels=[...]`: regressors assumed uncorrelated with a_i in levels, so instrumented by
  themselves (default none). Optional; useful for regressors with no within variation.
- default vcov `hetero` (the paper's), clusters allowed
- `after_event`: rows after a unit's first y = 1: `"error"` (default) or `"drop"`
- name `lpm_dthazard` (Ted: says what it does: linear probability model, discrete-time
  hazard)

**Time gaps (decided 2026-11-08).** A row whose lag window is incomplete is dropped and
counted, never silently: no row at t−1..t−j (unbalanced panel), a missing value in any
regressor at t−1..t−j, and the first j rows of every spell. The counts by reason go in the
printed summary, the `Result` and the estrec record (sample section), and the docstring and
user docs describe each rule in full. The paper does the same (2 rows, Table 3 notes).

**Comparison estimators (decided 2026-11-08).** b_FD (the reduced form), the first-stage
matrix G and pooled OLS on the same sample are reported alongside b_IV by default: they
come from the same pass.

## Validation plan

1. **Algebra**: on small data, dfeols equals pyfixest/linearmodels IV on hand-built Δx
   columns (b and SE to 1e-10), including absorbed time FEs and j = 2.
2. **Paper's Monte Carlo**, section 4 DGPs (stationary, random walk, trend; a_i ~
   U(−.05, .05), α = 0.1, b = 1, T = 5): Table 1 at N = 4·10⁷ (~72M rows each) should
   reproduce b_OLS, b_WI, b_FD, b_IV to about 2 SE (Table 1 SEs 0.001-0.0025). Also a good
   OSCER scale test, built as an outbox dataset. Tables A3/A4 (wave dummies, true time
   effects) check the absorbed-FE path.
3. **Brown & Laschever (2012) application** (Table 3): b_WI, b_FD, b_IV columns, school
   clusters, 12,968 obs for FD/IV. Needs openICPSR 113831 (Ted downloads).
4. **D6 EVE order-days** (~200M rows, torrent 09) as the real-data scale case.
