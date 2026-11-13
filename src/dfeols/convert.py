# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Stream a Stata .dta file into Parquet parts with bounded memory.

    dfeols convert-dta panel.dta OUT/ [--columns a,b,c] [--chunk-rows 2000000] [--part-gb 1]

The file is read in chunks of rows (pandas' StataReader iterator), so memory scales
with the chunk, not the file. Every chunk is cast to one Arrow schema fixed from the
.dta header, so a column keeps its type across parts even when pandas would upcast a
chunk with missing values to float:

    Stata byte/int/long  ->  int8/int16/int32 (nullable)
    float/double         ->  float32/float64 (Stata missing -> null, not NaN)
    str#/strL            ->  string

All Stata missing codes (., .a-.z) become null. Value labels are not applied (the
codes are kept); they, the variable labels and display formats go in
OUT/_dta_meta.json, written last with the row count checked against the header.
Dates are kept as Stata's integer or float codes (%tw weeks stay week numbers);
the formats in _dta_meta.json say which columns are dates.

Parts are written to part-NNNN.parquet.tmp and renamed when complete, and a new
part starts once the current one reaches --part-gb on disk.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

# StataReader's type codes: 'b','h','l','f','d' or an int (str width, 32768 = strL)
_ARROW = {"b": pa.int8(), "h": pa.int16(), "l": pa.int32(), "f": pa.float32(), "d": pa.float64()}


def _schema(reader, columns) -> pa.Schema:
    names = reader._varlist
    typ = dict(zip(names, reader._typlist))
    fields = []
    for c in columns:
        t = typ[c]
        fields.append(pa.field(c, _ARROW[t] if isinstance(t, str) else pa.string()))
    return pa.schema(fields)


def _to_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    arrays = []
    for f in schema:
        s = df[f.name]
        if pa.types.is_string(f.type):
            # str# pads nothing in pandas; empty string is Stata's missing string
            arr = pa.array(s.to_numpy(dtype=object), type=pa.string(), from_pandas=True)
        else:
            v = s.to_numpy()
            mask = np.isnan(v) if v.dtype.kind == "f" else None
            if pa.types.is_integer(f.type) and v.dtype.kind == "f":
                vi = np.where(mask, 0, v)
                if not np.array_equal(vi, np.round(vi)):
                    raise ValueError(f"column {f.name}: non-integer values in an integer Stata type")
                v = vi.astype(f.type.to_pandas_dtype())
            elif v.dtype != f.type.to_pandas_dtype():
                v = v.astype(f.type.to_pandas_dtype())
            arr = pa.array(v, type=f.type, mask=mask)
        arrays.append(arr)
    return pa.Table.from_arrays(arrays, schema=schema)


def convert_dta(
    src,
    out,
    columns: list[str] | None = None,
    chunk_rows: int = 2_000_000,
    part_gb: float = 1.0,
    compression: str = "zstd",
    overwrite: bool = False,
) -> dict:
    """Convert `src` (.dta) to `out`/part-NNNN.parquet; returns the metadata written."""
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("part-*.parquet*")) + list(out.glob("_dta_meta.json"))
    if existing and not overwrite:
        raise FileExistsError(f"{out} already has {len(existing)} output files; pass overwrite=True")
    for p in existing:
        p.unlink()
    t0 = time.time()
    part_bytes = int(part_gb * 2**30)
    reader = pd.read_stata(
        src,
        iterator=True,
        chunksize=chunk_rows,
        convert_categoricals=False,
        convert_dates=False,
        convert_missing=False,
        preserve_dtypes=True,
        columns=columns,
    )
    with reader:
        var_labels = reader.variable_labels()  # opens the file and reads the header
        # pandas has no public accessor for these; present since pandas 1.x
        names = reader._varlist
        cols = list(columns) if columns else list(names)
        missing = set(cols) - set(names)
        if missing:
            raise KeyError(f"not in {src.name}: {sorted(missing)}")
        schema = _schema(reader, cols)
        nobs = reader._nobs
        meta = {
            "source": str(src),
            "source_bytes": src.stat().st_size,
            "source_mtime": src.stat().st_mtime,
            "data_label": reader.data_label,
            "time_stamp": reader.time_stamp,
            "nobs": nobs,
            "columns": [
                {
                    "name": f.name,
                    "arrow": str(f.type),
                    "stata_type": str(dict(zip(names, reader._typlist))[f.name]),
                    "format": dict(zip(names, reader._fmtlist))[f.name],
                    "label": var_labels.get(f.name, ""),
                    "value_label": dict(zip(names, reader._lbllist)).get(f.name, ""),
                }
                for f in schema
            ],
        }
        parts, rows, nulls = [], 0, dict.fromkeys(cols, 0)
        writer, tmp, idx = None, None, 0

        def close():
            nonlocal writer, tmp
            if writer is not None:
                writer.close()
                final = tmp.with_suffix("")
                os.replace(tmp, final)
                parts.append({"file": final.name, "rows": writer_rows, "bytes": final.stat().st_size})
                writer, tmp = None, None

        writer_rows = 0
        for chunk in reader:
            tab = _to_table(chunk, schema)
            for c in cols:
                nulls[c] += tab.column(c).null_count
            if writer is None:
                tmp = out / f"part-{idx:04d}.parquet.tmp"
                writer = pq.ParquetWriter(tmp, schema, compression=compression)
                writer_rows, idx = 0, idx + 1
            writer.write_table(tab)
            writer_rows += tab.num_rows
            rows += tab.num_rows
            log.info("convert-dta: %s / %s rows (%.0f s)", f"{rows:,}", f"{nobs:,}", time.time() - t0)
            if tmp.stat().st_size >= part_bytes:
                close()
        close()
        value_labels = reader.value_labels()
    if rows != nobs:
        raise RuntimeError(f"read {rows:,} rows but the header says {nobs:,}")
    meta.update(
        rows=rows,
        nulls=nulls,
        value_labels={k: {str(c): v for c, v in d.items()} for k, d in value_labels.items()},
        parts=parts,
        seconds=round(time.time() - t0, 1),
    )
    (out / "_dta_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def add_parser(sub) -> None:
    p = sub.add_parser("convert-dta", help="stream a Stata .dta file into Parquet parts")
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--columns", help="comma-separated subset (default: all)")
    p.add_argument("--chunk-rows", type=int, default=2_000_000)
    p.add_argument("--part-gb", type=float, default=1.0)
    p.add_argument("--compression", default="zstd")
    p.add_argument("--overwrite", action="store_true")


def main(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    m = convert_dta(
        args.src,
        args.out,
        columns=args.columns.split(",") if args.columns else None,
        chunk_rows=args.chunk_rows,
        part_gb=args.part_gb,
        compression=args.compression,
        overwrite=args.overwrite,
    )
    print(f"{m['rows']:,} rows, {len(m['parts'])} parts, {m['seconds']} s -> {args.out}")
