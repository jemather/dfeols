# SPDX-License-Identifier: GPL-3.0-or-later
"""D10 G: GTEx v11 bulk RNA-seq to long sample x gene Parquet, all genes, zeros kept.

Reads `raw/gtex/` (docs/sources/gtex.md): the 2026-05-19 v11 gene TPM and read-count Parquet
(genes x samples), the sample and subject annotations, and GTEx's collapsed gene model
`references/v11/gencode.v47.genes.gtf`. Rules, columns and the model: docs/gtex.md
(Ted 2026-11-17: all 19,616 samples x all 74,628 genes, no expression filter).

    cd /srv/dfeols && .venv/bin/python -I repo/replication/gtex/build.py --out build/gtex

Output in --out:
  tissues/t<NN>.parquet   one file per tissue, rows sorted by (sample, gene)
  samples.parquet         sample codes and traits (tissue, donor, center, batch, sex, RIN, ...)
  genes.parquet           gene codes and traits (chromosome, length, flags)
  tissues.parquet, donors.parquet, centers.parquet   code tables
  R0/data/part-0000.parquet   reference tier (ref_gene == 1), all samples
  build.json              per-tissue rows and checks

Sample codes run in (tissue, SAMPID) order, so each tissue's samples are one contiguous range.
Gene codes follow the expression file's row order.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

RAW = Path("raw/gtex")
EXPR = RAW / "bulk-gex/v11/rna-seq/GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_"
ATTR = RAW / "annotations/v11/metadata-files/GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt"
SUBJ = RAW / "annotations/v11/metadata-files/GTEx_Analysis_v11_Annotations_SubjectPhenotypesDS.txt"
GTF = RAW / "references/v11/gencode.v47.genes.gtf"
c = pl.col

# reference tier: genes with ref_bucket(gene) == 0, plus protein-coding chrY genes, XIST, MT- genes
REF_MOD = 256

SCHEMA = pa.schema(
    [
        ("sample", pa.int32()),
        ("gene", pa.int32()),
        ("tissue", pa.int16()),
        ("donor", pa.int16()),
        ("center", pa.int8()),
        ("log_tpm", pa.float32()),
        ("log_count", pa.float32()),
        ("male", pa.int8()),
        ("y_gene", pa.int8()),
        ("xist", pa.int8()),
        ("rin", pa.float32()),
        ("log_gene_length", pa.float32()),
        ("mt_gene", pa.int8()),
        ("ref_gene", pa.int8()),
    ]
)


def ref_bucket(ensg: pl.Expr) -> pl.Expr:
    """Stable hash bucket of the Ensembl gene number (version stripped) in [0, REF_MOD)."""
    n = ensg.str.extract(r"ENSG0*(\d+)").cast(pl.Int64)
    return ((n * 2654435761) % (1 << 32)) % REF_MOD


def read_gtf() -> pl.DataFrame:
    """Gene chromosome, type and collapsed exon length (one transcript per gene in GTEx's model)."""
    gtf = pl.read_csv(
        GTF,
        separator="\t",
        comment_prefix="#",
        has_header=False,
        quote_char=None,
        new_columns=["chrom", "src", "feature", "start", "end", "score", "strand", "frame", "attr"],
        schema_overrides={"start": pl.Int64, "end": pl.Int64},
    ).with_columns(c("attr").str.extract(r'gene_id "([^"]+)"').alias("Name"))
    genes = gtf.filter(c("feature") == "gene").select(
        "Name", "chrom", c("attr").str.extract(r'gene_type "([^"]+)"').alias("gene_type")
    )
    ex = (
        gtf.filter(c("feature") == "exon")
        .group_by("Name")
        .agg((c("end") - c("start") + 1).sum().alias("length"))
    )
    return genes.join(ex, on="Name", how="left")


def tables(out: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    cols = [
        n for n in pq.ParquetFile(f"{EXPR}tpm.parquet").schema_arrow.names if n not in ("Name", "Description")
    ]
    attr = pl.read_csv(ATTR, separator="\t", infer_schema_length=0)
    subj = pl.read_csv(SUBJ, separator="\t", infer_schema_length=0)
    s = (
        pl.DataFrame({"SAMPID": cols, "col": range(len(cols))})
        .join(
            attr.select("SAMPID", "SMTSD", "SMTS", "SMCENTER", "SMGEBTCH", "SMRIN", "SMTSISCH", "SMAFRZE"),
            on="SAMPID",
            how="left",
        )
        .with_columns(c("SAMPID").str.split("-").list.slice(0, 2).list.join("-").alias("SUBJID"))
        .join(subj, on="SUBJID", how="left")
    )
    bad = s.filter(
        c("SMTSD").is_null() | c("SEX").is_null() | c("SMRIN").is_null() | (c("SMAFRZE") != "RNASEQ")
    )
    if bad.height:
        raise SystemExit(
            f"{bad.height} samples lack tissue, sex or RIN, or aren't RNASEQ: {bad['SAMPID'].head(5)}"
        )
    code = {}
    for k, col in (("tissue", "SMTSD"), ("donor", "SUBJID"), ("center", "SMCENTER")):
        code[k] = s.select(col).unique().sort(col).with_row_index(k)
        code[k].write_parquet(out / f"{k}s.parquet")
    for k, col in (("tissue", "SMTSD"), ("donor", "SUBJID"), ("center", "SMCENTER")):
        s = s.join(code[k], on=col, how="left")
    s = s.sort("SMTSD", "SAMPID").with_row_index("sample")
    rin = s["SMRIN"].cast(pl.Float64)
    s = s.with_columns(
        c("sample").cast(pl.Int32),
        c("tissue").cast(pl.Int16),
        c("donor").cast(pl.Int16),
        c("center").cast(pl.Int8),
        (c("SEX") == "1").cast(pl.Int8).alias("male"),
        (rin - rin.mean()).cast(pl.Float32).alias("rin"),
        c("SMTSISCH").cast(pl.Float64).alias("ischemic_min"),
    )
    s.write_parquet(out / "samples.parquet")

    g = pl.from_arrow(pq.read_table(f"{EXPR}tpm.parquet", columns=["Name", "Description"])).with_row_index(
        "gene"
    )
    gtf = read_gtf()
    g = g.join(gtf, on="Name", how="left", maintain_order="left")
    miss = g.filter(c("chrom").is_null() | c("length").is_null())
    if miss.height:
        raise SystemExit(
            f"{miss.height} GTEx genes not in {GTF} (versioned ID): {miss['Name'].head(5).to_list()}"
        )
    ll = c("length").log()
    g = g.with_columns(
        c("gene").cast(pl.Int32),
        (c("chrom") == "chrY").cast(pl.Int8).alias("y_gene"),
        (c("Description") == "XIST").cast(pl.Int8).alias("xist"),
        c("Description").str.starts_with("MT-").cast(pl.Int8).alias("mt_gene"),
        (c("gene_type") == "protein_coding").cast(pl.Int8).alias("protein_coding"),
        (ll - ll.mean()).cast(pl.Float32).alias("log_gene_length"),
    ).with_columns(
        (
            (ref_bucket(c("Name")) == 0)
            | ((c("y_gene") == 1) & (c("protein_coding") == 1))
            | (c("xist") == 1)
            | (c("mt_gene") == 1)
        )
        .cast(pl.Int8)
        .alias("ref_gene")
    )
    g.write_parquet(out / "genes.parquet")
    return s, g


_G: dict = {}


def _init(genes_path: str):
    g = pl.read_parquet(genes_path)
    _G.update({k: g[k].to_numpy() for k in ("y_gene", "xist", "log_gene_length", "mt_gene", "ref_gene")})


def build_tissue(args) -> dict:
    t, samp, out = args
    sid = samp["SAMPID"]
    n, t0 = len(sid), time.time()

    def mat(kind):
        tab = pq.read_table(f"{EXPR}{kind}.parquet", columns=sid)
        return np.column_stack([x.to_numpy() for x in tab.columns])  # genes x samples

    tpm, cnt = mat("tpm"), mat("reads")
    G = tpm.shape[0]
    if (tpm < 0).any() or (cnt < 0).any() or np.isnan(tpm).any():
        raise SystemExit(f"tissue {t}: negative or NaN values")
    rep = lambda a: np.repeat(np.asarray(a), G)
    til = lambda a: np.tile(a, n)
    cols_ = {
        "sample": rep(samp["sample"]).astype(np.int32),
        "gene": til(np.arange(G, dtype=np.int32)),
        "tissue": np.full(n * G, t, np.int16),
        "donor": rep(samp["donor"]).astype(np.int16),
        "center": rep(samp["center"]).astype(np.int8),
        "log_tpm": np.log1p(tpm.T.astype(np.float64)).ravel().astype(np.float32),
        "log_count": np.log1p(cnt.T.astype(np.float64)).ravel().astype(np.float32),
        "male": rep(samp["male"]).astype(np.int8),
        "y_gene": til(_G["y_gene"]),
        "xist": til(_G["xist"]),
        "rin": rep(samp["rin"]).astype(np.float32),
        "log_gene_length": til(_G["log_gene_length"]),
        "mt_gene": til(_G["mt_gene"]),
        "ref_gene": til(_G["ref_gene"]),
    }
    tab = pa.table(cols_, schema=SCHEMA)
    dst = out / "tissues" / f"t{t:02d}.parquet"
    tmp = dst.with_suffix(".tmp")
    pq.write_table(tab, tmp, compression="zstd", row_group_size=4_000_000)
    tmp.rename(dst)
    return {
        "tissue": t,
        "samples": n,
        "rows": tab.num_rows,
        "nonzero_reads": int((cnt > 0).sum()),
        "ref_rows": int(n * _G["ref_gene"].sum()),
        "bytes": dst.stat().st_size,
        "seconds": round(time.time() - t0, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("build/gtex"))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--tissues", default="", help="comma list of tissue codes (testing); default all")
    a = ap.parse_args(argv)
    (a.out / "tissues").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s, g = tables(a.out)
    print(
        f"{s.height:,} samples, {g.height:,} genes, {s['tissue'].n_unique()} tissues, "
        f"{s['donor'].n_unique()} donors; "
        f"y genes {g['y_gene'].sum()}, ref genes {g['ref_gene'].sum()}",
        flush=True,
    )
    todo = sorted(s["tissue"].unique().to_list())
    if a.tissues:
        todo = [int(x) for x in a.tissues.split(",")]
    jobs = []
    for t in todo:
        if (a.out / "tissues" / f"t{t:02d}.parquet").exists():
            continue
        sub = s.filter(c("tissue") == t)
        jobs.append(
            (
                t,
                {k: sub[k].to_list() for k in ("SAMPID", "col", "sample", "donor", "center", "male", "rin")},
                a.out,
            )
        )
    bj = a.out / "build.json"
    info = json.loads(bj.read_text()) if bj.exists() else {"tissues": {}}
    ctx = mp.get_context("spawn")  # Polars ran in this process: fork can hang (memory note)
    with ctx.Pool(a.procs, initializer=_init, initargs=(str(a.out / "genes.parquet"),)) as pool:
        for r in pool.imap_unordered(build_tissue, jobs):
            info["tissues"][str(r["tissue"])] = r
            bj.write_text(json.dumps(info, indent=1))
            print(
                f"tissue {r['tissue']}: {r['samples']} samples, {r['rows']:,} rows, "
                f"{r['bytes'] / 1e9:.2f} GB, "
                f"{r['seconds']:.0f} s",
                flush=True,
            )
    files = sorted((a.out / "tissues").glob("t*.parquet"))
    chk = (
        pl.scan_parquet(files)
        .select(
            pl.len().alias("rows"),
            *[c(k).null_count().alias(f"null_{k}") for k in SCHEMA.names],
            c("sample").n_unique().alias("samples"),
            c("log_tpm").min().alias("min_log_tpm"),
            (c("log_count") > 0).sum().alias("nonzero"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    info["check"] = chk
    if len(files) == s["tissue"].n_unique():
        expect = s.height * g.height
        if (
            chk["rows"] != expect
            or chk["samples"] != s.height
            or any(v for k, v in chk.items() if k.startswith("null_"))
        ):
            raise SystemExit(f"check failed: {chk}, expected {expect} rows")
        r0 = a.out / "R0" / "data"
        r0.mkdir(parents=True, exist_ok=True)
        pl.scan_parquet(files).filter(c("ref_gene") == 1).sink_parquet(
            r0 / "part-0000.parquet", compression="zstd"
        )
        info["R0"] = {
            "rule": f"genes with ((ENSG number * 2654435761) mod 2^32) mod {REF_MOD} == 0, "
            "plus protein-coding chrY genes, XIST, MT- genes",
            "genes": int(g["ref_gene"].sum()),
            "rows": pl.scan_parquet(r0 / "part-0000.parquet").select(pl.len()).collect().item(),
        }
    info["levels"] = {
        "sample": s.height,
        "gene": g.height,
        "tissue": s["tissue"].n_unique(),
        "donor": s["donor"].n_unique(),
        "center": s["center"].n_unique(),
        "gene^tissue": g.height * s["tissue"].n_unique(),
        "gene^center": g.height * s["center"].n_unique(),
    }
    info["genes"] = {k: int(g[k].sum()) for k in ("y_gene", "xist", "mt_gene", "protein_coding", "ref_gene")}
    info["samples"] = {"male": int(s["male"].sum()), "rin_mean": float(s["SMRIN"].cast(pl.Float64).mean())}
    info["seconds"] = round(time.time() - t0, 1)
    bj.write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("check", "levels", "genes", "samples") if k in info}, indent=1))
    if "R0" in info:
        print("R0", info["R0"])


if __name__ == "__main__":
    main()
