# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
#
# The R client of a dfeols session (docs/design/sessions.md). A session's manager watches
# <sessions>/<name>/requests/ for JSON requests and writes responses/<id>.json; this client
# writes the same files as the Python client (dfeols.Session), with jsonlite only. Models
# come back as estrec records (estrec::read_estimation), whose tidy()/glance() methods
# modelsummary uses.

`%||%` <- function(a, b) if (is.null(a)) b else a

.now <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%OS6Z", tz = "UTC")

.read_json <- function(path) {
  tryCatch(jsonlite::fromJSON(path, simplifyVector = FALSE), error = function(e) NULL)
}

# Atomic write: the manager sees the old file or the new one, never half of one.
.write_json <- function(obj, path) {
  tmp <- file.path(dirname(path), sprintf(".%s.%s.tmp", basename(path),
                                          paste(sample(c(0:9, letters[1:6]), 8, TRUE), collapse = "")))
  writeLines(jsonlite::toJSON(obj, auto_unbox = TRUE, null = "null", digits = NA), tmp,
             useBytes = TRUE)
  if (!file.rename(tmp, path)) stop("could not write ", path, call. = FALSE)
}

.say <- function(s, ...) {
  if (isTRUE(s$progress)) message(sprintf("[session %s] ", s$name), ...)
}

.secs <- function(x) {
  if (x < 1) return(sprintf("%.1f s", x))
  x <- round(x)
  if (x < 60) return(sprintf("%d s", x))
  sprintf("%d:%02d:%02d", x %/% 3600, x %% 3600 %/% 60, x %% 60)
}

# The sessions folder: `root`, else DFEOLS_SESSIONS, else what `dfeols session where` says
# for the profile (the site profile is Python's to read).
.session_dir <- function(name, root, profile) {
  if (!is.null(root)) return(file.path(root, name))
  env <- Sys.getenv("DFEOLS_SESSIONS")
  if (nzchar(env)) return(file.path(env, name))
  exe <- Sys.getenv("DFEOLS_CLI", "dfeols")
  args <- c("session", "where", name, if (!is.null(profile)) c("--profile", profile))
  out <- tryCatch(suppressWarnings(system2(exe, args, stdout = TRUE, stderr = FALSE)),
                  error = function(e) character())
  if (!length(out) || !nzchar(out[length(out)]))
    stop("cannot find session '", name, "': give root=, set DFEOLS_SESSIONS, or put the ",
         "dfeols command on PATH (DFEOLS_CLI)", call. = FALSE)
  out[length(out)]
}

.state <- function(s) .read_json(file.path(s$dir, "session.json"))

#' Attach to a running session; waits until it is ready (showing the demeaning's progress).
attach_session <- function(name, root = NULL, profile = NULL, timeout = NULL, progress = TRUE) {
  s <- structure(new.env(parent = emptyenv()), class = "dfeols_session")
  s$name <- name
  s$dir <- .session_dir(name, root, profile)
  s$progress <- progress
  s$shown <- character()
  s$client <- NULL
  t0 <- Sys.time()
  repeat {
    st <- .state(s)
    if (!is.null(st) && identical(st$status, "ready")) break
    if (!is.null(st) && st$status %in% c("failed", "closed", "timed_out", "crashed"))
      stop("session ", name, " is ", st$status,
           if (!is.null(st$error)) paste0(": ", st$error), call. = FALSE)
    if (!is.null(timeout) && as.numeric(Sys.time() - t0, units = "secs") > timeout)
      stop("session ", name, " not ready after ", timeout, " s", call. = FALSE)
    if (!is.null(st)) {
      est <- st$demean_estimate_seconds
      if (!is.null(est) && !"demean_estimate" %in% s$shown) {
        s$shown <- c(s$shown, "demean_estimate")
        .say(s, "demeaning: estimated ", .secs(est[[1]]), " to ", .secs(est[[2]]))
      }
      if (!is.null(st$progress)) .show_status(s, file.path(s$dir, st$progress))
    }
    Sys.sleep(0.5)
  }
  r <- .request(s, "attach", language = "R", version = paste(R.version$major, R.version$minor, sep = "."),
                host = Sys.info()[["nodename"]])
  s$client <- r$client
  s
}

.show_status <- function(s, path) {
  snap <- .read_json(path)
  if (is.null(snap) || !length(snap$stages)) return(invisible())
  key <- paste(path, snap$updated)
  if (key %in% s$shown) return(invisible())
  s$shown <- c(s$shown, key)
  st <- snap$stages[[length(snap$stages)]]
  done <- st$done %||% st$tasks_done %||% NA
  total <- st$total %||% st$tasks %||% NA
  .say(s, sprintf("[%s] %s/%s shards", st$name %||% "stage", done, total))
}

.watch <- function(s, rid) {
  acc <- file.path(s$dir, "responses", paste0(rid, ".accepted.json"))
  if (!file.exists(acc)) return(invisible())
  a <- .read_json(acc)
  if (is.null(a)) return(invisible())
  key <- paste0("est:", rid)
  if (!key %in% s$shown) {
    s$shown <- c(s$shown, key)
    rng <- a$estimate_range
    e <- if (!is.null(rng) && .secs(rng[[1]]) != .secs(rng[[2]]))
      paste(.secs(rng[[1]]), "to", .secs(rng[[2]])) else .secs(a$estimate_seconds %||% 0)
    if (!is.null(a$what)) .say(s, a$what, ": estimated ", e, " (", a$basis, ")")
    else if (!isTRUE(a$passes > 0)) .say(s, "no pass over the data (from the Gram matrix)")
    else .say(s, "estimated ", e, ": ", a$passes, " pass over ", a$columns, " columns (", a$basis, ")")
  }
  .show_status(s, file.path(s$dir, a$model_dir, "status.json"))
}

.request <- function(s, type, ..., .timeout = NULL, .watch = FALSE) {
  rid <- sprintf("%s-%s", format(Sys.time(), "%Y%m%dT%H%M%S"),
                 paste(sample(c(0:9, letters[1:6]), 8, TRUE), collapse = ""))
  req <- c(list(id = rid, type = type, client = s$client, sent = .now()), list(...))
  .write_json(req, file.path(s$dir, "requests", paste0(rid, ".json")))
  resp <- file.path(s$dir, "responses", paste0(rid, ".json"))
  t0 <- Sys.time()
  while (!file.exists(resp)) {
    if (!is.null(.timeout) && as.numeric(Sys.time() - t0, units = "secs") > .timeout)
      stop("no response to ", type, " after ", .timeout, " s", call. = FALSE)
    st <- .state(s)
    if (!is.null(st) && !st$status %in% c("ready", "starting") && !file.exists(resp))
      stop("session ", s$name, " is ", st$status, call. = FALSE)
    if (.watch) .watch(s, rid)
    Sys.sleep(0.1)
  }
  if (.watch) .watch(s, rid)
  r <- NULL
  for (i in 1:50) {  # the manager renames the file into place; read until it parses
    r <- .read_json(resp)
    if (!is.null(r)) break
    Sys.sleep(0.05)
  }
  if (!isTRUE(r$ok)) stop(r$error %||% "request failed", call. = FALSE)
  r
}

.formula_text <- function(formula) {
  if (inherits(formula, "formula")) paste(deparse(formula, width.cutoff = 500L), collapse = " ")
  else as.character(formula)
}

# A model's records as estrec records: one, or a named list for several responses.
.records <- function(s, m) {
  recs <- lapply(m$records, function(r) {
    rec <- estrec::read_estimation(file.path(s$dir, r$path))
    attr(rec, "dfeols_label") <- m$label
    rec
  })
  if (length(recs) == 1L) recs[[1]] else recs
}

.ref <- function(model) {
  if (is.character(model) || is.numeric(model)) return(as.character(model))
  lab <- attr(model, "dfeols_label")
  if (is.null(lab)) stop("`model` is a label or a record this client returned", call. = FALSE)
  lab
}

fit <- function(object, ...) UseMethod("fit")

#' Fit `formula` (y ~ x1 + x2 on declared or created variables; the FEs are the session's).
#' Returns the model's estrec record (a named list of them for several responses).
#' IV: "y ~ x | d ~ z1 + z2", with estimator = "liml" and fuller = for LIML and Fuller.
#' `cluster` is one variable, or two ("a+b") for two-way clustering.
fit.dfeols_session <- function(object, formula, vcov = NULL, cluster = NULL, label = NULL,
                               estimator = NULL, fuller = NULL, timeout = NULL, ...) {
  s <- object
  r <- .request(s, "fit", formula = .formula_text(formula), vcov = vcov, cluster = cluster,
                label = label, estimator = estimator, fuller = fuller, .timeout = timeout,
                .watch = TRUE)
  m <- r$model
  if (m$passes > 0 || m$seconds >= 1) .say(s, m$label, ": done in ", .secs(m$seconds))
  .records(s, m)
}

#' An earlier model (a returned record or its label) with another vcov or cluster.
refit <- function(s, model, vcov = NULL, cluster = NULL, label = NULL, timeout = NULL) {
  r <- .request(s, "refit", model = .ref(model), vcov = vcov, cluster = cluster, label = label,
                .timeout = timeout, .watch = TRUE)
  m <- r$model
  if (m$passes > 0 || m$seconds >= 1) .say(s, m$label, ": done in ", .secs(m$seconds))
  .records(s, m)
}

#' Create `name` from raw columns of the dataset ("log(fare)", "hhi * dist", "pax > 0"),
#' demeaned on the session's FEs. Returns the session (invisibly), so calls chain with |>.
add_variable <- function(s, name, expression, timeout = NULL) {
  r <- .request(s, "add_variable", name = name, expression = expression, .timeout = timeout,
                .watch = TRUE)
  v <- r$variable
  if (!isTRUE(v$already))
    .say(s, "created ", v$name, " = ", v$expression,
         if (isTRUE(v$reused)) " (reused from an earlier session)" else paste0(" (in ", .secs(v$seconds), ")"))
  invisible(s)
}

#' Several created variables (a named character vector or list: name = expression); the
#' new ones are demeaned together, in one demeaning however many there are.
add_variables <- function(s, variables, timeout = NULL) {
  r <- .request(s, "add_variables", variables = as.list(variables), .timeout = timeout, .watch = TRUE)
  made <- Filter(function(v) !isTRUE(v$already), r$variables)
  if (length(made)) .say(s, "created ", paste(vapply(made, `[[`, "", "name"), collapse = ", "))
  invisible(s)
}

#' The declared variables and the created ones (name -> expression).
variables <- function(s) {
  st <- .state(s)
  created <- st$created %||% list()
  list(declared = unlist(st$spec$variables),
       created = stats::setNames(vapply(created, `[[`, "", "expression"),
                                 vapply(created, `[[`, "", "name")))
}

#' Keep a --no-persist session's demeaned data after it ends.
persist <- function(s) {
  r <- .request(s, "persist")
  if (!isTRUE(r$already)) .say(s, "demeaned data kept in ", r$location)
  invisible(r$location)
}

#' Write the session collection now (status open); returns its path.
export_session <- function(s, path = NULL) .request(s, "export", path = path)$path

#' A table of `models` (records or labels) through the estrec table tool, in the format of
#' `path`'s suffix (.tex, .md, .html, .csv, .txt, .doc); the table spec and the session
#' collection go beside it. `...` goes to estrec's new_table (stats, rows, uncertainty, ...).
export_table <- function(s, models, path, title = NULL, headers = NULL, ...) {
  refs <- vapply(if (is.list(models) && !inherits(models, "estimation_record")) models else list(models),
                 .ref, "")
  table <- list(...)
  r <- .request(s, "export", models = as.list(refs), path = normalizePath(path, mustWork = FALSE),
                title = title, headers = if (!is.null(headers)) as.list(headers),
                table = if (length(table)) table else NULL)
  r$path
}

session_status <- function(s) .request(s, "status")$session

#' Detach this client; the session keeps running.
detach_session <- function(s) {
  if (!is.null(s$client)) {
    .request(s, "detach")
    s$client <- NULL
  }
  invisible(s)
}

#' End the session (writes the collection, releases the lock).
stop_session <- function(s, wait = 60) {
  .request(s, "stop")
  s$client <- NULL
  t0 <- Sys.time()
  while (identical(.state(s)$status, "ready") && as.numeric(Sys.time() - t0, units = "secs") < wait)
    Sys.sleep(0.2)
  invisible(.state(s)$status)
}

print.dfeols_session <- function(x, ...) {
  st <- .state(x)
  cat("<dfeols session ", x$name, ": ", st$status %||% "unknown", ", ", st$models %||% 0,
      " models>\n", sep = "")
  invisible(x)
}
