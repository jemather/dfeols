# Validation

`dfeols.synth` generates data with known parameters at any size, chunk by
chunk (in parallel locally or on cluster workers):

- **panel** (default): generic unit × time panel, units nested in groups,
  unbalanced, regressors confounded with unit, time and (optionally)
  group-time effects; AR(1), heteroskedastic, group-correlated errors.
- **influencer** (example): follower × account × week panel with tenure bins
  and a piecewise-linear true response; clusters cross shards.

Each dataset's `truth.json` lists the true coefficients and a set of specs
with what each should find: `unbiased`, `biased` (omits a confounded FE) or
`se_wrong` (clustering ignores the error structure). Tests and benchmarks are
driven by these, so a new DGP brings its own checks.

```bash
python -m dfeols.synth /scratch/me/raw --rows 1e8 --jobs 16 --set T=104
pytest tests/ --rows 5e6 --data-dir /scratch/me/dfeols-tests
python -m dfeols.bench --rows 1e7 1e8 --profile cluster
```

| Test file | Checks |
|---|---|
| `test_vs_pyfixest.py` | equals pyfixest on hard FE structures, weights, multiple outcomes, clusters crossing shards, only-dummy FEs, every crossing-FE path including `cells=True`, Dask executor |
| `test_panel.py` | generic panel: each spec finds what it should, unit- vs group-partitioned fits identical, group clustering matters, weights, pyfixest agreement |
| `test_influencer.py` | influencer DGP: recovery, negative controls, shard and executor invariance, Monte Carlo coverage (`-m slow`) |
| `test_prep_distributed.py` | distributed prep equals single-node prep under every strategy (merge, two-level shuffle, hash ids, range shards, multi-column keys) |
| `test_demean.py` | compiled and NumPy demeaning agree; exact two-way equals iteration |
| `test_iv.py`, `test_iv_iterate.py` | 2SLS, LIML and Fuller equal pyfixest, fixest, linearmodels and ivreghdfe on every FE path |
| `test_crossing.py` | plan C's pieces: block pseudo-inverses, the bounded sum of the shards' parts of A, the preconditioner choice, CG |
| `test_fit_twoway.py` | two-way clustering equals pyfixest for each kind of cluster pair on every crossing-FE path, and for IV |
| `test_session*.py` | session models equal `fit` (OLS, IV, created variables, `i()`, two-way clustering); the R client equals Python |
| `test_dthazard.py` | `lpm_dthazard` equals hand-built IV samples fitted with pyfixest and `fit` |
| `test_record.py` | every record and collection validates against estrec's schemas |

In a Monte Carlo of 40 replications, CRV1 intervals covered the truth 95.8%
of the time (iid: 85.8%, as expected with clustered errors).
