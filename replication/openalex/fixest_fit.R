# SPDX-License-Identifier: GPL-3.0-or-later
# D14: fixest fits of one Parquet file, for the comparisons with dfeols and pyfixest.
#
#   Rscript fixest_fit.R <data.parquet> <out.json> <mode> <cluster> <formula> [<formula> ...]
#
# mode "default": feols(fml, d, cluster = ~<cluster>, fixef.rm = "singleton"), fixest's defaults
# otherwise (singletons dropped as dfeols and pyfixest do, so N and the small-sample factors match);
# mode "lean": also lean = TRUE, mem.clean = TRUE. Reads the file with nanoparquet (arrow as a
# fallback). Writes load_sec, fit_sec, status, N/coef/se per formula, and R's peak RSS
# (VmHWM) to <out.json>; a failure is recorded, not raised.
#
# With $FIXEST_RECORD set (pyfixest_probe.py sets it), each fit is also written as an estrec record
# by estrec R's write_estimation() at disclosure level 5, to that path (numbered _<i> when there are
# several formulas). That happens after every fit is done and <out.json> is written with its times
# and VmHWM, and after an empty file fit_done is created next to <out.json>: the probe's watchdog
# counts the peak memory before and after it apart, so writing records never changes what is
# measured. A missing estrec package or a writer error is recorded as estrec_error; the fit stands.

suppressPackageStartupMessages({
  library(fixest)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]; out_path <- args[2]; mode <- args[3]; clus <- args[4]; fmls <- args[-(1:4)]
setFixest_notes(FALSE)

vmhwm_gb <- function() {
  s <- tryCatch(readLines("/proc/self/status"), error = function(e) character())
  l <- grep("^VmHWM:", s, value = TRUE)
  if (length(l)) round(as.numeric(gsub("[^0-9]", "", l)) / 2^20, 2) else NA
}

out <- list(engine = paste0("fixest", if (mode == "lean") "_lean" else ""),
            fixest = as.character(packageVersion("fixest")), R = R.version.string,
            nthreads = getFixest_nthreads(), formulas = fmls, status = "error")
t0 <- proc.time()[["elapsed"]]
tryCatch({
  words <- unique(c(unlist(regmatches(fmls, gregexpr("[A-Za-z_][A-Za-z0-9_]*", fmls))), clus))
  if (requireNamespace("nanoparquet", quietly = TRUE)) {
    cols <- intersect(nanoparquet::read_parquet_schema(data_path)$name, words)
    d <- nanoparquet::read_parquet(data_path, col_select = cols)
    out$reader <- paste("nanoparquet", as.character(packageVersion("nanoparquet")))
  } else if (requireNamespace("arrow", quietly = TRUE)) {
    d <- as.data.frame(arrow::read_parquet(data_path, col_select = dplyr::any_of(words)))
    out$reader <- paste("arrow", as.character(packageVersion("arrow")))
  } else {
    stop("no Parquet reader in R: install nanoparquet")
  }
  out$rows <- nrow(d)
  out$load_sec <- round(proc.time()[["elapsed"]] - t0, 1)
  t1 <- proc.time()[["elapsed"]]
  out$fits <- list()
  models <- list()
  for (f in fmls) {
    m <- if (mode == "lean") {
      feols(as.formula(f), d, cluster = as.formula(paste0("~", clus)), fixef.rm = "singleton",
            lean = TRUE, mem.clean = TRUE)
    } else {
      feols(as.formula(f), d, cluster = as.formula(paste0("~", clus)), fixef.rm = "singleton")
    }
    out$fits[[f]] <- list(N = nobs(m), coef = as.list(coef(m)), se = as.list(se(m)))
    if (nzchar(Sys.getenv("FIXEST_RECORD"))) models[[f]] <- m
    rm(m)
  }
  out$fit_sec <- round(proc.time()[["elapsed"]] - t1, 1)
  out$status <- "ok"
}, error = function(e) {
  out$error <<- substr(conditionMessage(e), 1, 2000)
})
out$sec <- round(proc.time()[["elapsed"]] - t0, 1)
out$self_maxrss_gb <- vmhwm_gb()
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), out_path)
rpath <- Sys.getenv("FIXEST_RECORD")
if (nzchar(rpath) && out$status == "ok") {
  file.create(file.path(dirname(out_path), "fit_done"))
  t2 <- proc.time()[["elapsed"]]
  errs <- character()
  if (!requireNamespace("estrec", quietly = TRUE)) {
    errs <- "estrec R package not installed"
  } else {
    for (i in seq_along(models)) {
      rp <- if (length(models) > 1) sub("\\.estrec\\.json$", paste0("_", i, ".estrec.json"), rpath) else rpath
      e <- tryCatch({
        estrec::write_estimation(models[[i]], rp, data = d, level = 5L)
        NULL
      }, error = function(e) substr(conditionMessage(e), 1, 500))
      if (!is.null(e)) {
        errs <- c(errs, e)
        if (file.exists(rp)) unlink(rp)
      }
    }
  }
  if (length(errs)) out$estrec_error <- paste(errs, collapse = " | ")
  out$record_sec <- round(proc.time()[["elapsed"]] - t2, 1)
  writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), out_path)
}
