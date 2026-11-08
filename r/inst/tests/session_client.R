# Driven by tests/test_session_r.py: a Python manager runs session `name` under
# DFEOLS_SESSIONS; this script attaches from R, creates a variable, fits and refits, exports
# a table, and writes what it got to `out` (JSON) for the Python side to compare.
args <- commandArgs(trailingOnly = TRUE)
name <- args[[1]]; out <- args[[2]]; table_path <- args[[3]]
library(dfeols)
s <- attach_session(name, progress = FALSE, timeout = 60)
add_variable(s, "fp", "feat * posts") |> add_variable("f2", "feat^2")
m1 <- fit(s, eng ~ posts + fp, cluster = "id_a", label = "r1")
m2 <- fit(s, "eng ~ posts + fp + f2", vcov = c("hetero", "iid"))
m3 <- refit(s, m1, vcov = "hetero")
two <- fit(s, eng + f2 ~ posts, label = "two")
add_variables(s, c(p2 = "posts^2", p3 = "posts^3"))
iv <- fit(s, "eng ~ f2 | posts ~ feat + p2", estimator = "liml", cluster = "id_a")
twoway <- fit(s, "eng ~ posts + i(tbin)", cluster = "id_a+week")
td <- generics::tidy(m1)
gl <- generics::glance(m1)
ms <- modelsummary::modelsummary(list(r1 = m1, r2 = m2), output = "markdown")
p <- export_table(s, list(m1, "m2"), table_path, title = "From R", stats = "nobs")
err <- tryCatch({fit(s, "eng ~ nope"); NA_character_}, error = function(e) conditionMessage(e))
v <- variables(s)
detach_session(s)
jsonlite::write_json(list(
  tidy = td, nobs = gl$nobs, labels = c(attr(m1, "dfeols_label"), attr(m3, "dfeols_label")),
  two = names(two), modelsummary = paste(ms, collapse = "\n"), table = p, error = err,
  created = as.list(v$created), declared = v$declared,
  iv = generics::tidy(iv), twoway = generics::tidy(twoway)
), out, auto_unbox = TRUE, digits = NA)
