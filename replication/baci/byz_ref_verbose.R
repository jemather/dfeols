# SPDX-License-Identifier: GPL-3.0-or-later
# D13 diagnostic: fixest fepois p1 on R0 with verbose = 3, so the log shows the deviance of
# every IRLS iteration and every step halving (fixest reports halvings only at verbose >= 3).
# For comparing with dfeols fit_glm's halvings (DFEOLS_GLM_DIAG).
#
#   Rscript byz_ref_verbose.R <data.parquet>
suppressPackageStartupMessages(library(fixest))
a <- commandArgs(trailingOnly = TRUE)
cols <- c("t", "i", "j", "k", "pair", "v", "rta")
d <- if (requireNamespace("nanoparquet", quietly = TRUE)) {
  nanoparquet::read_parquet(a[1], col_select = cols)
} else {
  as.data.frame(arrow::read_parquet(a[1], col_select = dplyr::all_of(cols)))
}
setFixest_notes(FALSE)
t0 <- proc.time()[["elapsed"]]
m <- fepois(v ~ rta | i^k^t + j^k^t + i^j^k, d, cluster = ~pair, verbose = 3)
cat(sprintf("RESULT N %d b %.15g se_pair %.15g iterations %d deviance %.12g sec %.1f\n",
            nobs(m), coef(m)[["rta"]], se(m)[["rta"]], m$iterations, deviance(m),
            proc.time()[["elapsed"]] - t0))
