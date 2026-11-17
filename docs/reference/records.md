# Records

Every fit writes [estrec](https://estrec.org) records by default (`fit(record=True)`). To
write them yourself, use `Result.write_estimation` (one record per outcome) or
`write_collection` (several fits in one file).

::: dfeols.record.write_collection

::: dfeols.results.Result.write_estimation
    options:
      show_root_full_path: false

::: dfeols.results.Result.to_estimation
    options:
      show_root_full_path: false
