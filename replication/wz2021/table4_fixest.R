# SPDX-License-Identifier: GPL-3.0-or-later
# D13: Weidner & Zylkin (2021) Table 4, PPML column, with fixest::fepois as the reference.
# The do-file's model: ppmlhdfe2 trade FTA, a(exp_id#year imp_id#year exp_id#imp_id)
# cluster(exp_id#imp_id) tol(1e-6), one fit per isic_id (they skip 5, 6, 31, 34; we run all).
# se: fixest's default small-sample factor; se_feK_none: FEs left out of K (ppmlhdfe's count).
#   cd /srv/dfeols && Rscript repo/replication/wz2021/table4_fixest.R tmp/wz2021/baci_example.csv build/wz2021/fixest.csv
suppressMessages({library(data.table); library(fixest)})
a <- commandArgs(TRUE)
setFixest_nthreads(4)
d <- fread(a[1])
out <- rbindlist(lapply(sort(unique(d$isic_id)), function(k) {
  s <- d[isic_id == k]
  t0 <- Sys.time()
  m <- tryCatch(fepois(trade ~ FTA | expyear + impyear + pair, data = s, cluster = ~pair,
                       fixef.tol = 1e-10, glm.tol = 1e-10, glm.iter = 200),
                error = function(e) NULL)
  if (is.null(m) || !("FTA" %in% names(coef(m))))
    return(data.table(isic_id = k, isiccode = s$isiccode[1], b = NA_real_, se = NA_real_, se_feK_none = NA_real_, nobs = NA_integer_,
                      iter = NA_integer_, sec = as.numeric(Sys.time() - t0, units = "secs")))
  data.table(isic_id = k, isiccode = s$isiccode[1], b = unname(coef(m)["FTA"]), se = unname(se(m)["FTA"]),
             se_feK_none = unname(se(summary(m, ssc = ssc(fixef.K = "none")))["FTA"]),
             nobs = nobs(m), iter = m$iterations, sec = as.numeric(Sys.time() - t0, units = "secs"))
}))
fwrite(out, a[2])
print(out)
