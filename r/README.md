# dfeols (R): the session client

An R client for dfeols interactive sessions (`docs/design/sessions.md`). A session's manager
(started with `dfeols session start`, on a cluster or a workstation) has absorbed a set of
fixed effects once on a large dataset; this package attaches to it and fits models against
it. It speaks the session's request files, so it needs no network path to the compute node
and no Python, only jsonlite. Each model comes back as an [estrec](https://github.com/estrec/estrec)
record, which estrec gives `tidy()` and `glance()` methods for, so modelsummary and other
table packages take it as they take a fitted model.

```r
library(dfeols)
s <- attach_session("db1b", profile = "oscer")          # waits until the session is ready
add_variable(s, "lfare_x_hhi", "lfare * hhi")           # from raw columns, then demeaned
m1 <- fit(s, lfare ~ hhi + lfare_x_hhi, cluster = "route")
m2 <- refit(s, m1, vcov = "hetero")                     # b again, one pass for the SEs
m3 <- fit(s, "lfare ~ dist | hhi ~ entrants", estimator = "liml")   # IV
modelsummary::modelsummary(list(m1, m2, m3))
export_table(s, list(m1, m2), "table2.tex", title = "Fares")        # estrec's table tool
detach_session(s)
```

Finding the session: `root =` (the sessions folder), else the `DFEOLS_SESSIONS` environment
variable, else `dfeols session where <name> --profile <profile>` (the site profile is
Python's to read; `DFEOLS_CLI` names the command if it is not on `PATH`).

Functions: `attach_session`, `fit`, `refit`, `add_variable`, `variables`, `persist`,
`export_session` (the session collection), `export_table`, `session_status`,
`detach_session`, `stop_session`.

Installing: `R CMD INSTALL r` from the dfeols repository, after estrec's R package
(`R CMD INSTALL r` in the estrec repository; it is not on CRAN yet). The tests run from
Python (`tests/test_session_r.py`), which starts a session and drives
`inst/tests/session_client.R` against it.
