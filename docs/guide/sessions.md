# Interactive sessions

A session demeans a set of declared variables on its FEs once, keeps the
residualised columns on scratch, and fits models from them while it runs:
coefficients and iid SEs from their Gram matrix with no pass over the data,
hetero and CRV1 (one- or two-way) with one pass. Results equal `dfeols.fit`
on the same sample.

```bash
dfeols session start flags --data /scratch/me/shards/db1b --fe "route + carrier^quarter" \
    --vars lfare,hhi,dist,entrants --cluster route,carrier --hours 12 --profile mycluster --wait
dfeols session fit flags "lfare ~ hhi + dist" --cluster route+carrier
```

```python
s = dfeols.Session.attach("flags", profile="mycluster")
s.add_variable("hhi_x_dist", "hhi * dist")           # from raw columns, demeaned like the rest
m1 = s.fit("lfare ~ hhi + hhi_x_dist + i(qtr) + dist:i(region)", cluster="route")
m2 = s.refit(m1, vcov="hetero")                      # b again; one pass for the SEs
m3 = s.fit("lfare ~ dist | hhi ~ entrants", estimator="liml")
s.export([m1, m2, m3], "table2.tex")                 # through estrec's table tool
```

The **[R client](r-client.md)** in [`r/`](https://github.com/jemather/dfeols/tree/main/r)
attaches to the same session through its request files (no network path to the
compute node, no Python) and returns each model as an estrec record, so
`tidy()`, `glance()` and modelsummary (with broom installed) work. On OSCER, from a login node
against a 40-worker session, its models equal Python's to 1e-10 and each request waits about
0.6 s beyond the session's own work. Each model's record carries the
session id, and the session's collection logs every model, created variable
and setting change, so `dfeols session replay` can run them again. Demeaned
data are kept under `<scratch>/demeaned/` and reused by a later session with
the same inputs; `dfeols demeaned list` and `dfeols demeaned rm` manage them.
Design: [Sessions design note](../design/sessions.md). API: [`Session`](../reference/sessions.md).
