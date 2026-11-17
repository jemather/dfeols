# Discrete-time hazard models

`dfeols.lpm_dthazard(ds, "exit ~ x1 + i(age) | fe", unit="id", time="t")`
fits a discrete-time hazard linear probability model (an absorbing outcome) by
the Farbmacher and Tauchmann (2021) IV: levels instrumented by their
differences, with the first-difference and pooled OLS estimates for
comparison. It builds the sample per shard (gaps and missing lags dropped and
counted), absorbs extra FEs with a warning where that is not equivalent, and
reproduces the published estimates of Farbmacher and Tauchmann's Table 3.
Design: [Hazard models design note](../design/hazard.md). API:
[`lpm_dthazard`](../reference/fitting.md#dfeols.dthazard.lpm_dthazard).
