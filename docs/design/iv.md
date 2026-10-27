# Instrumental variables in dfeols

Status: implemented, LIML's robust SEs still to check (2026-10-27). Decided by Ted: 2SLS and LIML (k-class) in the first version,
any number of endogenous regressors, and a basic set of first-stage diagnostics computed in the
existing two passes. Comparisons: pyfixest (one endogenous regressor), fixest in R (several),
ivreghdfe in Stata (LIML, diagnostics), linearmodels (LIML on small data with FE dummies).

## Syntax

fixest and pyfixest's: `y ~ exog | fe1 + fe2 | endog1 + endog2 ~ inst1 + inst2`, and without
FEs `y ~ exog | endog ~ inst`. Endogenous and instrument terms are parsed like the other
regressors (interactions, `i()`). `fit(..., estimator="2sls" | "liml", fuller=None)`;
`fuller=a` gives Fuller's modified LIML (κ = κ_LIML − a / (N − L)).

## Algebra

After the FEs are partialled out (local FEs demeaned in the shard, crossing FEs as dummy
columns, dense or by per-level sums on the sparse path):

| | columns | where in the shard |
|---|---|---|
| y | outcomes (m) | outcome block `Y` |
| X₂ | endogenous (p) | outcome block `Y`, after the outcomes |
| X₁ | exogenous regressors, crossing-FE dummies | regressor block `Z` |
| Z₂ | excluded instruments (q ≥ p) | regressor block `Z` |

Putting X₂ in the outcome block means it is demeaned, and on the sparse path partialled
against the dummies, exactly like an outcome. Round 1 then needs the full `Y'WY` (it summed
only its diagonal). With Z = [X₁, Z₂] and X = [X₁, X₂], every quantity the coordinator needs
is a block of `Z'WZ`, `Z'WY` and `Y'WY`:

- First stage: Π = (Z'Z)⁻¹ Z'X₂; X̂ = P_Z X = [X₁, ZΠ].
- k-class: β(κ) = [X'(I − κM_Z)X]⁻¹ X'(I − κM_Z)y, with X'(I − κM_Z)X = (1 − κ)X'X + κ X̂'X̂.
  OLS is κ = 0, 2SLS κ = 1.
- LIML: κ is the smallest root of det(W'M_{X₁}W − κ W'M_Z W) = 0, W = [y, X₂], per outcome.
  Fuller: κ − a / (N − L).

Round 2 builds, per outcome j, the residual e = y − Xβ (with the actual X₂) and the score
regressors X̃ = (1 − κ)X + κX̂ = [X₁, (1 − κ)X₂ + κZΠ]. Both are linear maps of the shard's
`[X₂, Z]` columns, so round 2 takes the maps (`c` for the residual, `H` for X̃) instead of β
alone; OLS is the special case `c = β`, `H = I`. The bread is [X'(I − κM_Z)X]⁻¹, the meat
Σ X̃ᵢ'eᵢ²X̃ᵢ (hetero) or by cluster, and iid uses σ² = RSS / (N − K) with RSS from the actual X.
On the sparse path, the dummy part of X̃ and e is partialled out by FWL as for OLS: the
coordinator also passes the dummy coefficients of X₂ and of the non-dummy Z columns.

K, the small-sample correction and the t-test df are as for OLS, with K counting X₁ and X₂
(fixest's convention).

## First-stage diagnostics (`structures.first_stage`)

Per endogenous regressor, from the first-stage regression of X₂ₗ on Z:

- coefficients on the excluded instruments, with standard errors under each requested vcov;
- R² and partial R² (of Z₂ given X₁; within R² when there are FEs);
- `f_stat`: the iid partial F of Z₂ (fixest `ivf`), and the Wald F under each requested vcov
  (fixest `ivwald`, pyfixest `_f_stat_1st_stage`). The robust versions need the first-stage
  residuals v = X₂ − ZΠ in round 2, with Z̃₂ = M_{X₁}Z₂ as the score regressors (FWL gives the
  Z₂ block of the full sandwich);
- `effective_f` (Montiel Olea-Pflueger), with one endogenous regressor only.

Cheap from the same cross-products, proposed: Sargan's overidentification test for 2SLS
(iid) and the Anderson-Rubin overidentification test for LIML, (N − L)(κ − 1).

Not in the first version: Sanderson-Windmeijer and Angrist-Pischke conditional F,
Cragg-Donald and Kleibergen-Paap rk statistics, Hansen J, endogeneity tests, GMM.

## Collinearity and identification

Collinear columns are dropped from Z as for OLS (`collin_tol`). After that the fit needs at
least as many kept excluded instruments as endogenous regressors, else it stops with an
error naming them. An endogenous regressor whose fitted values are collinear with X₁ is
dropped and reported.

## Plan

1. Formula: the IV part; `Formula.endog`, `Formula.instruments`.
2. Worker: endogenous columns in the outcome block, instruments in the regressor block, full
   `Y'WY`; round 2 from linear maps (OLS unchanged in output).
3. Coordinator: `dfeols/iv.py` with the k-class solve, LIML κ, first-stage statistics.
4. Result and record: `estimator`, κ, Fuller's a, `first_stage`; records claim `iv/0.1`.
5. Tests: pyfixest (2SLS, one endogenous, iid/hetero/CRV1, FEs local and crossing, dense
   and sparse), fixest (several endogenous), linearmodels and ivreghdfe (LIML, Fuller,
   diagnostics); sharded result equals the one-shard result.

## Status (2026-10-27)

Implemented as above (`src/dfeols/iv.py`; round 2 in `worker.shard_round2` now takes linear
maps for OLS too). `tests/test_iv.py`:

- 2SLS equals pyfixest (coefficients, SEs, p-values, N) with a local FE, crossing FE dense
  and sparse, no FE, a just-identified model; iid, hetero, CRV1 by a cluster nested in the
  partition and by one crossing it; weights. pyfixest's first-stage F and effective F match.
- Two endogenous regressors equal fixest (coefficients, clustered SEs, `ivwald` per
  endogenous regressor); sparse equals dense.
- LIML κ and coefficients equal a direct computation; just-identified LIML equals 2SLS.
- Records validate against `iv/0.1` with no warnings.
- Waiting for packages (the tests skip): LIML and Fuller robust/clustered SEs against
  linearmodels and ivreghdfe, Fuller's L, and ivreghdfe's first-stage F and partial R².

Open for Ted: fixest's iid `ivf` takes its denominator df from the clustered fit's K (an FE
nested in the cluster counts once); dfeols's iid F counts every absorbed level, which is the
iid convention and what pyfixest does for an iid fit. The record gives df2, and the test
converts. The Wald F p-values use the vcov's t-test df (G − 1 under CRV1); fixest prints
N − K.
