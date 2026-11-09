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
    counts = (
        lf.group_by(key).len().sort(key).collect(engine="streaming").with_columns(cum=pl.col("len").cum_sum())
    )
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
        part = (lf if scan_for is None else scan_for(lo, hi)).filter(pl.col(key).is_between(lo, hi))
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
