# SPDX-License-Identifier: GPL-3.0-or-later
"""D10 A: ARCHS4 v2.5 human gene counts to long sample x gene Parquet (bulk, filtered, zeros kept).

Reads `raw/archs4/human_gene_v2.5.h5` (docs/sources/gtex.md, ARCHS4 section). Rules, columns and
the model: docs/gtex.md (Ted 2026-11-17):

  samples: singlecellprobability < 0.5, library_source "transcriptomic", molecule_ch1 total or
           polyA RNA, library (sum over all 67,186 genes) >= 1M reads; series = first
           series_id; series with >= 5 such samples
  genes:   protein-coding; per series, genes with CPM >= 1 in >= 20% of the series' samples
  rows:    every kept sample x every kept gene of its series, zeros included

    cd /srv/dfeols && .venv/bin/python -I repo/replication/archs4/build.py --out build/archs4 --procs 6

Two passes over the H5 in chunks of CHUNK samples (contiguous column blocks are fast, single
columns slow). Pass 1 writes `pass1/c<NNNN>.npz` (library sizes; per-series counts of samples
with CPM >= 1 per gene); pass 2 writes `data/part-<NNNN>.parquet`, one per chunk, rows sorted
by (sample, gene). Both skip chunks already done, so a rerun after a kill resumes.

Output in --out: data/, samples.parquet (all 888,821 with rule flags and codes), genes.parquet
(protein-coding genes with codes and flags), series.parquet, instruments.parquet,
R0/data/part-0000.parquet (reference tier), build.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

H5 = Path("raw/archs4/human_gene_v2.5.h5")
CHUNK = 16_000  # samples per chunk (one part file each)
BLOCK = 2_000  # samples per H5 read inside a chunk (~0.5 GB of uint32)
MIN_LIB = 1_000_000
MIN_SERIES = 5
CPM_MIN, SHARE_MIN = 1.0, 0.2
HISTONE = r"^(H1-[0-6]|H2AC\d|H2BC\d|H3C\d|H4C\d)"  # replication-dependent histone genes
NC_MARKERS = ("RN7SK", "RN7SL1", "RN7SL2", "RMRP", "RPPH1")  # non-polyadenylated ncRNAs
REF_ROWS = 7_500_000  # per half of the reference tier (mixed-label series, then the rest)
c = pl.col

SCHEMA = pa.schema(
    [
        ("sample", pa.int32()),
        ("gene", pa.int32()),
        ("series", pa.int32()),
        ("instrument", pa.int16()),
        ("year", pa.int16()),
        ("log_cpm", pa.float32()),
        ("count", pa.int64()),
        ("log_lib", pa.float32()),
        ("total_rna", pa.int8()),
        ("log_nc_share", pa.float32()),
        ("histone_gene", pa.int8()),
        ("mt_gene", pa.int8()),
    ]
)


def bucket(s: pl.Expr, mod: int) -> pl.Expr:
    """Stable hash bucket of a GSE accession's number."""
    n = s.str.extract(r"(\d+)").cast(pl.Int64)
    return ((n * 2654435761) % (1 << 32)) % mod


def read_meta(out: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    f = h5py.File(H5, "r")

    def col(k):
        v = f["meta/samples/" + k][:]
        return v if v.dtype.kind == "f" else [x.decode("utf-8", "replace") for x in v]

    s = pl.DataFrame(
        {
            k: col(k)
            for k in (
                "geo_accession",
                "series_id",
                "singlecellprobability",
                "library_source",
                "molecule_ch1",
                "instrument_model",
                "submission_date",
            )
        }
    ).with_row_index("sample")
    s = s.with_columns(
        c("sample").cast(pl.Int32),
        c("series_id").str.split(",").list.first().str.strip_chars().alias("gse"),
        c("submission_date").str.slice(-4).cast(pl.Int16, strict=False).alias("year"),
        (c("molecule_ch1") == "total RNA").cast(pl.Int8).alias("total_rna"),
        (
            (c("singlecellprobability") < 0.5)
            & (c("library_source") == "transcriptomic")
            & c("molecule_ch1").is_in(["total RNA", "polyA RNA"])
        ).alias("meta_ok"),
    )
    g = pl.DataFrame(
        {k: [x.decode() for x in f["meta/genes/" + k][:]] for k in ("ensembl_gene", "symbol", "biotype")}
    )
    nc = np.flatnonzero(g["symbol"].is_in(list(NC_MARKERS)).to_numpy())
    np.save(out / "nc_rows.npy", nc)  # all-gene rows of the marker ncRNAs (not protein-coding)
    g = g.with_row_index("row").filter(c("biotype") == "protein_coding").with_row_index("gene")
    g = g.with_columns(
        c("gene").cast(pl.Int32),
        c("symbol").str.contains(HISTONE).cast(pl.Int8).alias("histone_gene"),
        c("symbol").str.starts_with("MT-").cast(pl.Int8).alias("mt_gene"),
    )
    if g["ensembl_gene"].n_unique() != g.height:
        raise SystemExit("duplicate protein-coding Ensembl IDs")
    inst = s.select("instrument_model").unique().sort("instrument_model").with_row_index("instrument")
    s = (
        s.join(inst, on="instrument_model", how="left")
        .with_columns(c("instrument").cast(pl.Int16))
        .sort("sample")
    )
    inst.write_parquet(out / "instruments.parquet")
    g.write_parquet(out / "genes.parquet")
    return s, g


_W: dict = {}


def _init(out: str):
    o = Path(out)
    s = pl.read_parquet(o / "samples.parquet")
    g = pl.read_parquet(o / "genes.parquet")
    _W.update(out=o, rows=g["row"].to_numpy(), meta_ok=s["meta_ok"].to_numpy(), gse=s["gse"].to_numpy())
    for k in ("series", "instrument", "year", "total_rna"):
        if k in s.columns:
            _W[k] = s[k].fill_null(-1).to_numpy()
    _W["histone"], _W["mt"] = g["histone_gene"].to_numpy(), g["mt_gene"].to_numpy()
    _W["nc_rows"] = np.load(o / "nc_rows.npy")
    kp = o / "keep.npz"
    if kp.exists():
        z = np.load(kp)
        _W["keep"] = np.unpackbits(z["keep"], axis=1, count=len(_W["rows"])).astype(bool)
        _W["lib"] = s["lib"].to_numpy()
        _W["kept"] = s["kept"].to_numpy()
        _W["log_nc_share"] = s["log_nc_share"].to_numpy()


def _read(d, b: int, e: int) -> tuple[np.ndarray, list[int]]:
    """Columns b..e of the H5 matrix. The published v2.5 file has a few gzip chunks that fail to
    decompress (samples 776627, 776630, 776639, 776645, gene rows 56000-58000, found 2026-11-17).
    On a failure, read sample by sample and leave unreadable samples all zero, so their library
    size is 0 and the MIN_LIB rule drops them; they are returned for the log."""
    try:
        return d[:, b:e], []
    except OSError:
        X = np.zeros((d.shape[0], e - b), np.uint32)
        bad = []
        for i in range(b, e):
            try:
                X[:, i - b] = d[:, i]
            except OSError:
                bad.append(i)
        return X, bad


def _blocks(lo: int, hi: int, bad: list | None = None):
    f = h5py.File(H5, "r")
    d = f["data/expression"]
    for b in range(lo, hi, BLOCK):
        e = min(b + BLOCK, hi)
        X, bb = _read(d, b, e)  # genes x samples, uint32
        if bad is not None:
            bad += bb
        yield b, e, X


def pass1(k: int) -> int:
    dst = _W["out"] / "pass1" / f"c{k:04d}.npz"
    if dst.exists():
        return k
    lo, hi = k * CHUNK, min((k + 1) * CHUNK, len(_W["meta_ok"]))
    lib = np.zeros(hi - lo, np.int64)
    nc = np.zeros(hi - lo, np.int64)
    acc: dict[str, np.ndarray] = {}
    bad: list[int] = []
    for b, e, X in _blocks(lo, hi, bad):
        L = X.sum(0, dtype=np.int64)
        lib[b - lo : e - lo] = L
        nc[b - lo : e - lo] = X[_W["nc_rows"]].sum(0, dtype=np.int64)
        ok = _W["meta_ok"][b:e] & (L >= MIN_LIB)
        if not ok.any():
            continue
        P = X[_W["rows"]][:, ok].astype(np.float64)
        hit = P / L[ok] * 1e6 >= CPM_MIN  # pc genes x kept samples
        gs = _W["gse"][b:e][ok]
        for s in np.unique(gs):
            v = hit[:, gs == s].sum(1, dtype=np.int32)
            acc[s] = acc[s] + v if s in acc else v
    names = np.array(sorted(acc), dtype=object)
    cnt = np.stack([acc[n] for n in names]) if len(names) else np.zeros((0, len(_W["rows"])), np.int32)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(
        tmp, lib=lib, nc=nc, bad=np.array(bad, np.int64), names=names.astype(str), cnt=cnt.astype(np.uint16)
    )
    tmp.rename(dst)
    return k


def pass2(k: int) -> dict:
    dst = _W["out"] / "data" / f"part-{k:04d}.parquet"
    if dst.exists():
        return {"chunk": k, "skipped": True, "rows": pq.ParquetFile(dst).metadata.num_rows}
    t0 = time.time()
    lo, hi = k * CHUNK, min((k + 1) * CHUNK, len(_W["meta_ok"]))
    tmp = dst.with_suffix(".tmp")
    w = pq.ParquetWriter(tmp, SCHEMA, compression="zstd")
    rows = 0
    for b, e, X in _blocks(lo, hi):
        idx = np.flatnonzero(_W["kept"][b:e]) + b
        if not len(idx):
            continue
        P = X[_W["rows"]][:, idx - b]  # pc genes x kept samples
        parts = []
        for j, i in enumerate(idx):
            gk = np.flatnonzero(_W["keep"][_W["series"][i]])
            cnt = P[gk, j].astype(np.int64)
            lib = _W["lib"][i]
            n = len(gk)
            parts.append(
                {
                    "sample": np.full(n, i, np.int32),
                    "gene": gk.astype(np.int32),
                    "series": np.full(n, _W["series"][i], np.int32),
                    "instrument": np.full(n, _W["instrument"][i], np.int16),
                    "year": np.full(n, _W["year"][i], np.int16),
                    "log_cpm": np.log1p(cnt / lib * 1e6).astype(np.float32),
                    "count": cnt,
                    "log_lib": np.full(n, np.log(lib), np.float32),
                    "total_rna": np.full(n, _W["total_rna"][i], np.int8),
                    "log_nc_share": np.full(n, _W["log_nc_share"][i], np.float32),
                    "histone_gene": _W["histone"][gk],
                    "mt_gene": _W["mt"][gk],
                }
            )
        tab = pa.table({k2: np.concatenate([p[k2] for p in parts]) for k2 in SCHEMA.names}, schema=SCHEMA)
        w.write_table(tab, row_group_size=4_000_000)
        rows += tab.num_rows
    w.close()
    tmp.rename(dst)
    return {"chunk": k, "rows": rows, "bytes": dst.stat().st_size, "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("build/archs4"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--chunks", default="", help="comma list of chunk numbers (testing); default all")
    a = ap.parse_args(argv)
    for d in ("pass1", "data"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s, g = read_meta(a.out)
    s.write_parquet(a.out / "samples.parquet")
    n_chunks = -(-s.height // CHUNK)
    chunks = [int(x) for x in a.chunks.split(",")] if a.chunks else list(range(n_chunks))
    print(
        f"{s.height:,} samples ({s['meta_ok'].sum():,} pass the metadata rules), {g.height:,} protein-coding "
        f"genes, {g['histone_gene'].sum()} histone, {n_chunks} chunks",
        flush=True,
    )
    ctx = mp.get_context("spawn")  # Polars ran in this process: fork can hang (memory note)

    with ctx.Pool(a.procs, initializer=_init, initargs=(str(a.out),)) as pool:
        for i, k in enumerate(pool.imap_unordered(pass1, chunks)):
            print(f"pass1 chunk {k} ({i + 1}/{len(chunks)}, {time.time() - t0:.0f} s)", flush=True)
    # merge pass 1: library sizes, kept samples and series, per-series gene filter
    lib = np.zeros(s.height, np.int64)
    nc = np.zeros(s.height, np.int64)
    acc: dict[str, np.ndarray] = {}
    unreadable: list[int] = []
    for k in chunks:  # all chunks, or a test subset treated as the whole file
        z = np.load(a.out / "pass1" / f"c{k:04d}.npz", allow_pickle=False)
        lib[k * CHUNK : k * CHUNK + len(z["lib"])] = z["lib"]
        nc[k * CHUNK : k * CHUNK + len(z["nc"])] = z["nc"]
        unreadable += z["bad"].tolist() if "bad" in z else []
        for n, v in zip(z["names"], z["cnt"].astype(np.int32)):
            acc[n] = acc[n] + v if n in acc else v
    s = s.with_columns(
        pl.Series("lib", lib),
        pl.Series("nc", nc),
        (c("meta_ok") & (pl.Series(lib) >= MIN_LIB)).alias("lib_ok"),
        # log10 share of reads in the marker ncRNAs: a data-based ribo-depletion measure
        (pl.Series(nc) / pl.Series(lib).clip(1) + 1e-6).log10().cast(pl.Float32).alias("log_nc_share"),
    )
    ns = s.filter(c("lib_ok")).group_by("gse").len().filter(c("len") >= MIN_SERIES).sort("gse")
    ser = ns.with_row_index("series").with_columns(c("series").cast(pl.Int32)).rename({"len": "samples"})
    cnt = np.stack([acc[n] for n in ser["gse"]])
    keep = cnt >= np.ceil(SHARE_MIN * ser["samples"].to_numpy())[:, None]  # CPM >= 1 in >= 20%
    ser = ser.with_columns(pl.Series("genes", keep.sum(1).astype(np.int32)))
    s = s.join(ser.select("gse", "series"), on="gse", how="left").with_columns(
        (c("lib_ok") & c("series").is_not_null()).alias("kept")
    )
    s = s.sort("sample")
    mix = s.filter(c("kept")).group_by("series").agg(c("total_rna").mean().alias("total_share"))
    ser = ser.join(mix, on="series", how="left").with_columns(
        ((c("total_share") > 0) & (c("total_share") < 1)).alias("mixed"),
        (c("samples") * c("genes")).alias("rows"),
    )
    ser.write_parquet(a.out / "series.parquet")
    s.write_parquet(a.out / "samples.parquet")
    np.savez(a.out / "keep.npz", keep=np.packbits(keep, axis=1))
    expect = int(ser["rows"].sum())
    print(
        f"kept {s['kept'].sum():,} samples in {ser.height:,} series; {int(keep.sum()):,} gene x series; "
        f"{expect:,} rows expected ({time.time() - t0:.0f} s)",
        flush=True,
    )

    info = {
        "rules": {"min_lib": MIN_LIB, "min_series": MIN_SERIES, "cpm_min": CPM_MIN, "share_min": SHARE_MIN}
    }
    info["parts"] = {}
    with ctx.Pool(a.procs, initializer=_init, initargs=(str(a.out),)) as pool:
        for r in pool.imap_unordered(pass2, chunks):
            info["parts"][str(r["chunk"])] = r
            print(f"pass2 chunk {r['chunk']}: {r['rows']:,} rows ({time.time() - t0:.0f} s)", flush=True)
    files = [a.out / "data" / f"part-{k:04d}.parquet" for k in sorted(chunks)]
    # check one part at a time: n_unique over all ~7B rows exceeds Polars' spill budget.
    # A sample never spans parts (parts are sample ranges); series can, so union their sets.
    chk: dict = {}
    series_seen: set[int] = set()
    for f in files:
        r = (
            pl.scan_parquet(f)
            .select(
                pl.len().alias("rows"),
                *[c(k).null_count().alias(f"null_{k}") for k in SCHEMA.names],
                c("sample").n_unique().alias("samples"),
                (c("count") > 0).sum().alias("nonzero"),
                c("count").max().alias("count_max"),
            )
            .collect()
            .row(0, named=True)
        )
        for k2, v in r.items():
            chk[k2] = max(chk.get(k2, 0), v or 0) if k2 == "count_max" else chk.get(k2, 0) + (v or 0)
        series_seen |= set(pl.scan_parquet(f).select(c("series").unique()).collect()["series"].to_list())
    chk["series"] = len(series_seen)
    info["check"] = chk
    if (
        chk["rows"] != expect
        or chk["samples"] != s["kept"].sum()
        or any(v for k, v in chk.items() if k.startswith("null_"))
    ):
        raise SystemExit(f"check failed: {chk}, expected {expect} rows")

    # reference tier: series in hash order, mixed-label series up to REF_ROWS, then others up to REF_ROWS
    order = ser.with_columns(bucket(c("gse"), 1 << 30).alias("h")).sort("h")
    pick = []
    for flag in (True, False):
        sub = order.filter(c("mixed") == flag)
        pick += sub.filter(c("rows").cum_sum() <= REF_ROWS)["series"].to_list()
    r0 = a.out / "R0" / "data"
    r0.mkdir(parents=True, exist_ok=True)
    pl.scan_parquet(files).filter(c("series").is_in(pick)).sink_parquet(
        r0 / "part-0000.parquet", compression="zstd"
    )
    info["R0"] = {
        "rule": f"series in hash order: mixed total/polyA series up to {REF_ROWS:,} rows, then others up to "
        f"{REF_ROWS:,}",
        "series": len(pick),
        "rows": pl.scan_parquet(r0 / "part-0000.parquet").select(pl.len()).collect().item(),
    }
    info["unreadable_samples"] = sorted(unreadable)
    info["counts"] = {
        "samples_all": s.height,
        "meta_ok": int(s["meta_ok"].sum()),
        "lib_ok": int(s["lib_ok"].sum()),
        "kept": int(s["kept"].sum()),
        "series": ser.height,
        "series_mixed": int(ser["mixed"].sum()),
        "rows_in_mixed_series": int(ser.filter(c("mixed"))["rows"].sum()),
        "gene_x_series": int(keep.sum()),
        "genes": g.height,
        "instruments": int(s.filter(c("kept"))["instrument"].n_unique()),
        "gene_x_instrument_max": g.height * int(s.filter(c("kept"))["instrument"].n_unique()),
        "series_rows_max": int(ser["rows"].max()),
    }
    info["seconds"] = round(time.time() - t0, 1)
    (a.out / "build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("check", "counts", "R0")}, indent=1))


if __name__ == "__main__":
    main()
