# Poisson and cloglog

```python
r = dfeols.fit_glm(ds, "unf ~ lfeat:i(tbin) + lposts | id_f + id_a + week",
                   family="cloglog", cluster="id_a")     # or family="poisson"
r.options["glm"]   # iterations, deviance, rows dropped as separated
```

Fitted by IRLS, each step a weighted dfeols fit, so it scales and shards like `fit`; the
working response lives in per-shard files between steps. Separated FE groups are dropped
first. Equal to pyfixest's `fepois` and R fixest's `feglm`. API:
[`fit_glm`](../reference/fitting.md#dfeols.glm.fit_glm).
