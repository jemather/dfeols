# Stacked equations

To test coefficients across equations (an outcome against another, mediation paths),
stack them into one regression with each equation's own FEs; the one clustered VCE then
covers every contrast:

```python
eqs = [dfeols.Equation(1, y="eng", x={"x": "feat", "p": "posts"}, fe=["spell", "week"]),
       dfeols.Equation(2, y="unf", x={"x": "lfeat", "p": "lposts"},
                       fe=["id_f", "id_a", "week"], rows=pl.col("at_risk"))]
lf = pl.scan_parquet(src).filter(dfeols.sample_groups("id_f", 0.15, seed=1))  # 15% of followers
st = dfeols.stack(lf, eqs, partition="id_f", keep=["tbin", "id_a"])
ds = dfeols.prep(st.frame.collect(), shard_dir=..., partition="id_f",
                 fixed_effects=st.fixed_effects, clusters=["id_a"])
r = dfeols.fit(ds, st.formula("x:i(tbin) + p:i(tbin)"), cluster="id_a")
r.coef()[st.coef_index(r.coefnames, "x", 2)]          # equation 2's x-by-bin coefficients
```

For data too large to collect, `st.frame.sink_parquet(...)` and prep the files.
Each FE is absorbed as `k^fe`. Where one equation has fewer FEs than another, its
slot is filled with a level its own FEs already span, so the fit doesn't change. Such
fillers still count in K, as fixest would count them. API:
[`stack`, `Equation`, `Stacked`, `sample_groups`](../reference/data.md#stacked-equations).
