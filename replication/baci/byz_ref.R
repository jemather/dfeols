# SPDX-License-Identifier: GPL-3.0-or-later
# D13: fixest reference fits of the product-level BYZ model on one Parquet file (the R0 sample).
#
#   Rscript byz_ref.R <data.parquet> <out.json>
#
# p1: fepois(v ~ rta | i^k^t + j^k^t + i^j^k), all cells (zeros included)
# o1: feols(lv ~ rta | i^k^t + j^k^t + i^j^k), positive flows only
# Both CRV1 by pair (i * 1000 + j). Writes N, b, se (CRV1 pair and hetero), seconds, iterations
# and VmHWM to <out.json>. fixest's defaults otherwise (fepois drops separated observations
# and fixed-effect-only groups; feols drops singletons with fixef.rm = "singleton" so N follows
# the dfeols / pyfixest convention).
suppressPackageStartupMessages({ library(fixest); library(jsonlite) })
a <- commandArgs(trailingOnly = TRUE)
cols <- c("t", "i", "j", "k", "pair", "v", "lv", "pos", "rta")
# nanoparquet where installed (OSCER), else arrow (valiant), as fixest_fit.R
d <- if (requireNamespace("nanoparquet", quietly = TRUE)) {
  nanoparquet::read_parquet(a[1], col_select = cols)
} else {
  as.data.frame(arrow::read_parquet(a[1], col_select = dplyr::all_of(cols)))
}
setFixest_notes(FALSE)
fe <- "i^k^t + j^k^t + i^j^k"
res <- list(rows = nrow(d), fixest = as.character(packageVersion("fixest")), R = R.version.string,
            nthreads = getFixest_nthreads())
one <- function(m, t0) {
  list(N = nobs(m), b = unname(coef(m)["rta"]),
       se_pair = unname(se(m, cluster = ~pair)["rta"]), se_hetero = unname(se(m, vcov = "hetero")["rta"]),
       iterations = if (!is.null(m$iterations)) m$iterations else NA,
       fe_sizes = as.list(m$fixef_sizes), sec = round(proc.time()[["elapsed"]] - t0, 1))
}
t0 <- proc.time()[["elapsed"]]
m <- fepois(as.formula(paste("v ~ rta |", fe)), d, cluster = ~pair)
res$p1 <- one(m, t0); rm(m); gc()
t0 <- proc.time()[["elapsed"]]
m <- feols(as.formula(paste("lv ~ rta |", fe)), d[d$pos == 1, ], cluster = ~pair, fixef.rm = "singleton")
res$o1 <- one(m, t0)
s <- readLines("/proc/self/status"); res$vmhwm_gb <- round(as.numeric(gsub("[^0-9]", "", grep("^VmHWM:", s, value = TRUE))) / 2^20, 2)
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA, pretty = TRUE), a[2])
