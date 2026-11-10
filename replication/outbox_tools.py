# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers for packaging a finished dataset into ``outbox/<name>/`` (CLAUDE.md layout).

``write_parts`` splits a lazy frame into ``data/part-NNNN.parquet`` files of roughly equal row
counts by contiguous ranges of a partition key, so a key value never straddles two parts. Each
part is streamed (``sink_parquet``) and sorted by the key within the part. The manifest is
written by the shell command in CLAUDE.md, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def key_ranges(lf: pl.LazyFrame, key: str, n_parts: int) -> list[tuple[object, object, int]]:
    """Split the key's sorted values into n_parts contiguous ranges of about equal rows."""
    counts = lf.group_by(key).len().sort(key).collect(engine="streaming")
    return split_counts(counts[key], counts["len"], n_parts)


def split_counts(keys: pl.Series, lens: pl.Series, n_parts: int) -> list[tuple[object, object, int]]:
    """key_ranges on (sorted key, rows per key). Counts are cast to Int64 first: polars-runtime-32
    counts in UInt32, so a sum or cum_sum over 2^32 rows wraps (nyc_tlc_trips_v1, 4.4B rows, got
    80 parts of the first ~131M rows and one of the rest)."""
    counts = pl.DataFrame({"key": keys, "len": lens.cast(pl.Int64)}).with_columns(cum=pl.col("len").cum_sum())
    key = "key"
    total = int(counts["len"].sum())
    ranges, start, done = [], 0, 0
    for i in range(1, n_parts + 1):
        target = total * i / n_parts
        end = start
        while end < counts.height and (counts["cum"][end] <= target or end == start):
            end += 1
        if i == n_parts:
            end = counts.height
        if end > start:
            rows = int(counts["cum"][end - 1]) - done
            ranges.append((counts[key][start], counts[key][end - 1], rows))
            done += rows
        start = end
    if sum(r[2] for r in ranges) != total:
        raise ValueError(f"ranges hold {sum(r[2] for r in ranges)} rows, keys {total}")
    # a range of several keys holds about total / n_parts rows; far more means the split failed
    big = [r for r in ranges if r[0] != r[1] and r[2] > 3 * total / n_parts]
    if big:
        raise ValueError(f"lopsided split: {big[0]} of {total} rows in {n_parts} parts")
    return ranges


def write_parts(
    lf: pl.LazyFrame, key: str, out: Path, n_parts: int, transform=None, scan_for=None
) -> list[dict]:
    """transform(lf) is applied after the key-range filter (for window functions within a key).
    scan_for(lo, hi), if given, returns a narrower frame holding every row of that key range
    (e.g. only the source files that can contain it), used in place of lf for the part."""
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (lo, hi, rows) in enumerate(key_ranges(lf, key, n_parts)):
        p = data / f"part-{i:04d}.parquet"
        part = (lf if scan_for is None else scan_for(lo, hi)).filter(
            pl.col(key).is_between(pl.lit(lo), pl.lit(hi))
        )  # lit: a str bound is a column name
        if transform is not None:
            part = transform(part)
        part.sort(key).sink_parquet(p, compression="zstd", statistics=True)
        got = pl.scan_parquet(p).select(pl.len()).collect().item()
        if got != rows:
            raise ValueError(f"{p}: {got} rows written, {rows} expected")
        parts.append(
            {
                "file": f"data/{p.name}",
                "rows": rows,
                f"{key}_min": lo,
                f"{key}_max": hi,
                "bytes": p.stat().st_size,
            }
        )
    return parts


def level_counts(lf: pl.LazyFrame, fes: dict[str, list[str]]) -> dict[str, int]:
    """Distinct non-null levels of each FE (a name -> the columns it interacts)."""
    exprs = []
    for name, cols in fes.items():
        e = pl.struct(cols) if len(cols) > 1 else pl.col(cols[0])
        exprs.append(
            e.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in cols])).n_unique().alias(name)
        )
    return {k: int(v) for k, v in lf.select(exprs).collect(engine="streaming").row(0, named=True).items()}


def level_counts_parts(files: list[Path], fes: dict[str, list[str]], within: set[str]) -> dict[str, int]:
    """level_counts one part file at a time, for data too big for one pass: a struct n_unique
    does not stream, so level_counts loads every column (nyc_tlc_trips_v1, 4.4B rows, was
    killed). FEs in `within` never span two parts (they nest in the partition key), so their
    counts add up over parts; the others keep each part's distinct levels and take the union."""
    sums = dict.fromkeys(within & fes.keys(), 0)
    seen: dict[str, list[pl.DataFrame]] = {k: [] for k in fes if k not in within}
    for f in files:
        lf = pl.scan_parquet(f)
        names = list(fes)
        got = pl.collect_all([lf.select(fes[k]).drop_nulls().unique() for k in names], engine="streaming")
        for k, df in zip(names, got):
            if k in sums:
                sums[k] += df.height
            else:
                seen[k] = [pl.concat(seen[k] + [df]).unique()]
    return {k: sums[k] if k in sums else seen[k][0].height for k in fes}


def fit_reference(pdf, formula: str, cluster: str, **kw) -> dict:
    import pyfixest as pf

    m = pf.feols(formula, data=pdf, vcov={"CRV1": cluster}, **kw)
    coef, se = m.coef(), m.se()
    return {
        "formula": formula,
        "vcov": {"CRV1": cluster},
        "nobs": int(m._N),
        **({"fixef_rm": kw["fixef_rm"]} if "fixef_rm" in kw else {}),
        "coef": {k: [float(coef[k]), float(se[k])] for k in coef.index},
    }


def schema_table(lf: pl.LazyFrame, notes: dict[str, str]) -> str:
    rows = ["| Column | Type | Notes |", "|---|---|---|"]
    for c, t in lf.collect_schema().items():
        rows.append(f"| {c} | {t} | {notes.get(c, '')} |")
    return "\n".join(rows)


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
