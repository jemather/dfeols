# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Stable, order-independent hashes of sets of rows (`dfeols-rowset-v1`).

Each row is hashed from its values in the named columns: per column, the value's bits
(float64 with -0.0 and NaN normalised, int64, or the first 8 bytes of the sha256 of a
string) are mixed with a salt from the column name by the splitmix64 finaliser, and the
columns are chained in name order. The row hashes are summed mod 2^64 in two lanes with
different seeds, so the result is the same however the rows are ordered or split into
shards, and shard results combine by addition.

Polars' own hash is not used: it is documented as unstable across Polars versions, and
records are compared across machines. This is an identity check for samples and data,
not a cryptographic hash.
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

ALGORITHM = "dfeols-rowset-v1"
_SEEDS = (0x9E3779B97F4A7C15, 0xD1B54A32D192ED03)
_NULL = np.uint64(0x6A09E667F3BCC908)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser, elementwise on uint64 (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint64(30))
        x = x * np.uint64(0xBF58476D1CE4E5B9)
        x = x ^ (x >> np.uint64(27))
        x = x * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def _h8(b: bytes) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.sha256(b).digest()[:8], "little"))


def column_bits(s: pl.Series) -> np.ndarray:
    """One uint64 per value; equal values give equal bits on any machine."""
    dt = s.dtype
    nulls = s.is_null().to_numpy() if s.null_count() else None
    if dt.is_float():
        a = s.cast(pl.Float64).fill_null(0.0).to_numpy().astype(np.float64)
        a = np.where(np.isnan(a), np.nan, a) + 0.0  # one NaN, and -0.0 -> 0.0
        out = a.view(np.uint64).copy()
    elif dt == pl.Boolean or dt.is_integer():
        out = (
            s.fill_null(0).to_numpy().astype(np.uint64)
            if dt == pl.UInt64
            else s.cast(pl.Int64).fill_null(0).to_numpy().view(np.uint64).copy()
        )
    elif dt.is_temporal():
        out = s.to_physical().cast(pl.Int64).fill_null(0).to_numpy().view(np.uint64).copy()
    elif dt in (pl.String, pl.Categorical) or isinstance(dt, pl.Enum):
        st = s.cast(pl.String).fill_null("")
        table = {v: int(_h8(v.encode("utf-8"))) for v in st.unique().to_list()}
        out = st.replace_strict(table, return_dtype=pl.UInt64).to_numpy().astype(np.uint64)
    else:
        raise TypeError(f"column {s.name!r}: no stable row hash for dtype {dt}")
    if nulls is not None:
        out[nulls] = _NULL
    return out


def row_hashes(df: pl.DataFrame, columns: list[str], seed: int) -> np.ndarray:
    h = np.full(df.height, np.uint64(seed), dtype=np.uint64)
    for c in sorted(columns):
        salt = _h8(c.encode("utf-8"))
        h = _mix(h ^ _mix(column_bits(df[c]) ^ salt))
    return h


def rowset(df: pl.DataFrame, columns: list[str]) -> dict:
    """{lanes: [sum mod 2^64 per seed], rows}: add these over shards with `combine`."""
    lanes = [int(row_hashes(df, columns, seed).sum(dtype=np.uint64)) for seed in _SEEDS]
    return {"lanes": lanes, "rows": df.height}


def combine(parts) -> dict:
    lanes, rows = [0, 0], 0
    for p in parts:
        lanes = [(a + b) & 0xFFFFFFFFFFFFFFFF for a, b in zip(lanes, p["lanes"])]
        rows += p["rows"]
    return {"lanes": lanes, "rows": rows}


def hexdigest(acc: dict) -> str:
    return "".join(f"{v:016x}" for v in acc["lanes"])
