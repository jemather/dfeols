# SPDX-License-Identifier: GPL-3.0-or-later
"""Package the D10 G build (build.py + reference.py output) as outbox/gtex_v1 (CLAUDE.md layout).

    cd /srv/dfeols && .venv/bin/python -I repo/replication/gtex/package.py \
        --build build/gtex --out outbox/gtex_v1 --commit <sha>

data/part-NNNN.parquet: the tissue files concatenated in order (so contiguous sample ranges,
sorted by sample then gene), about 1 GB each. ref/data/ (R0), ref/reference.json, lookups/,
README.md. The manifest is written afterwards by the shell command in CLAUDE.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import outbox_tools as ot

c = pl.col
PART_BYTES = 1.0e9
FES = {
    "sample": ["sample"],
    "gene": ["gene"],
    "tissue": ["tissue"],
    "donor": ["donor"],
    "center": ["center"],
    "gene^tissue": ["gene", "tissue"],
    "gene^center": ["gene", "center"],
}
NOTES = {
    "sample": "code of SAMPID, in (tissue, SAMPID) order; FE and partition key; lookups/samples.parquet",
    "gene": "code of the versioned Ensembl ID, in the expression file's order; lookups/genes.parquet",
    "tissue": "code of SMTSD (54 detailed tissues); lookups/tissues.parquet",
    "donor": "code of SUBJID (first two parts of SAMPID); cluster; lookups/donors.parquet",
    "center": "code of SMCENTER (7); lookups/centers.parquet",
    "log_tpm": "ln(1 + TPM): the outcome",
    "log_count": "ln(1 + read count): alternative outcome",
    "male": "donor SEX == 1",
    "y_gene": "gene on chrY (gencode.v47.genes.gtf)",
    "xist": "gene is XIST",
    "rin": "SMRIN (RNA integrity) minus its mean over samples",
    "log_gene_length": "ln(collapsed exon length) minus its mean over genes",
    "mt_gene": "symbol starts with MT-",
    "ref_gene": "gene is in the reference tier",
}
LADDER = {
    "g1": "sample + gene",
    "g2": "sample + gene^tissue",
    "g3": "sample + gene^tissue + gene^center",
}
X = "male:y_gene + male:xist + rin:log_gene_length"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=Path("build/gtex"))
    ap.add_argument("--out", type=Path, default=Path("outbox/gtex_v1"))
    ap.add_argument("--commit", required=True, help="dfeols repo commit of the build code")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; packages are never changed after creation (build _v2)")
    ref_json = a.build / "R0" / "reference.json"
    if not ref_json.exists():
        raise SystemExit(f"{ref_json} missing: run reference.py first (jobs/gtex_ref.sh)")
    info = json.loads((a.build / "build.json").read_text())
    tmp = a.out.with_name(a.out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "data").mkdir(parents=True)

    # group consecutive tissue files into ~1 GB parts (sample codes run in tissue order)
    tissues = sorted((a.build / "tissues").glob("t*.parquet"))
    groups, cur, size = [], [], 0
    for f in tissues:
        cur.append(f)
        size += f.stat().st_size
        if size >= PART_BYTES:
            groups.append(cur)
            cur, size = [], 0
    if cur:
        groups.append(cur)
    parts = []
    for i, g in enumerate(groups):
        p = tmp / "data" / f"part-{i:04d}.parquet"
        pl.scan_parquet(g).sink_parquet(p, compression="zstd", statistics=True, row_group_size=4_000_000)
        r = pl.scan_parquet(p).select(
            pl.len().alias("rows"), c("sample").min().alias("lo"), c("sample").max().alias("hi")
        )
        r = r.collect().row(0, named=True)
        want = sum(info["tissues"][str(int(f.stem[1:]))]["rows"] for f in g)
        if r["rows"] != want:
            raise SystemExit(f"{p}: {r['rows']} rows, expected {want}")
        parts.append(
            {
                "file": f"data/{p.name}",
                "rows": r["rows"],
                "lo": r["lo"],
                "hi": r["hi"],
                "tissues": len(g),
                "bytes": p.stat().st_size,
            }
        )
        print(parts[-1], flush=True)
    total = sum(p["rows"] for p in parts)
    if total != info["check"]["rows"]:
        raise SystemExit(f"parts hold {total} rows, build {info['check']['rows']}")

    (tmp / "ref" / "data").mkdir(parents=True)
    shutil.copy2(a.build / "R0" / "data" / "part-0000.parquet", tmp / "ref" / "data" / "part-0000.parquet")
    ref = json.loads(ref_json.read_text())
    ot.dump_json(
        tmp / "ref" / "reference.json",
        {"pyfixest": ref["pyfixest"], "dfeols_local": ref["dfeols"], "tier": info["R0"]},
    )
    ref_levels = ot.level_counts(pl.scan_parquet(tmp / "ref" / "data" / "*.parquet"), FES)
    (tmp / "lookups").mkdir()
    for n in ("samples", "genes", "tissues", "donors", "centers"):
        shutil.copy2(a.build / f"{n}.parquet", tmp / "lookups" / f"{n}.parquet")

    # every sample has every gene, so the full levels are products of the counts (build.json)
    lv = info["levels"]
    full = {k: lv[k] for k in FES}
    lvt = "\n".join(f"| {k} | {full[k]:,} | {ref_levels[k]:,} |" for k in FES)
    pt = "\n".join(
        f"| {p['file']} | {p['rows']:,} | {p['lo']}..{p['hi']} | {p['tissues']} | {p['bytes'] / 1e9:.2f} GB |"
        for p in parts
    )
    py = ref["pyfixest"]
    rf = "\n".join(
        f"| {k} | `{fe}` | {py[k]['N']:,} | "
        + " | ".join(
            f"{py[k]['coefs']['CRV1'][t][0]:.6f} ({py[k]['coefs']['CRV1'][t][1]:.6f})" for t in X.split(" + ")
        )
        + " |"
        for k, fe in LADDER.items()
        if k in py
    )
    g = info["genes"]
    readme = f"""# gtex_v1

GTEx v11 bulk RNA-seq, long format: one row per sample x gene, **all 19,616 RNA-seq samples x
all 74,628 genes, zeros included** ({total:,} rows in {len(parts)} parts). A dfeols test dataset (a
1.46B-row panel with a 4.03M-level crossing FE), not a research project.

- **Source:** GTEx Portal open-access files, adult-gtex bucket:
  `GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_{{tpm,reads}}.parquet`, the v11 sample and
  subject annotations, and GTEx's collapsed gene model `gencode.v47.genes.gtf` (sha256 in
  `raw/gtex/SHA256SUMS` on valiant). Public open-access data; cite the GTEx Consortium and
  release v11. Notes: `docs/gtex.md`, `docs/sources/gtex.md` on valiant
- **Build:** `repo/replication/gtex/build.py` and `package.py` (jemather/dfeols `{a.commit}`)
- **Sample** (Decisions 2026-11-17, Ted: "make it big"): no expression filter. Gene x tissue
  pairs that are zero in every sample of the tissue are fit exactly by their FE
- {info["samples"]["male"]:,} male samples; genes: {g["y_gene"]} on chrY, XIST, {g["mt_gene"]} MT-,
  {g["protein_coding"]:,} protein-coding

## Model (Decisions 2026-11-17)

`log_tpm ~ {X} | FEs`, CRV1 clustered by `donor` (946; also hetero), partition `sample`.

| Spec | FEs | What it tests |
|---|---|---|
| g1 | `sample + gene` | two-way, 74.6k crossing levels |
| g2 | `sample + gene^tissue` | 4.03M crossing levels (AKM-like) |
| g3 | `sample + gene^tissue + gene^center` | two crossing FEs |

Expected signs: `male:y_gene` strongly positive (chrY genes are expressed only in males),
`male:xist` strongly negative (XIST is expressed in females). `rin:log_gene_length` is a weak
continuous x continuous term with no firm expected sign.

## Parts

Tissue files concatenated in order: contiguous sample ranges, sorted by sample then gene.

| File | Rows | Sample range | Tissues | Size |
|---|---:|---|---:|---:|
{pt}

## Reference tier (`ref/`)

`ref/data/part-0000.parquet`: {info["R0"]["rule"]}; {info["R0"]["genes"]} genes x all samples,
{info["R0"]["rows"]:,} rows, same columns. `ref/reference.json`: pyfixest (demeaning tolerance
1e-13) for g1-g3, CRV1 by donor and hetero, plus the local dfeols fits (partitioned by sample
and by tissue, CG iter_tol 1e-10, level space off). CRV1 coefficients (SE):

| Spec | FEs | N | {" | ".join(f"`{t}`" for t in X.split(" + "))} |
|---|---|---:|---:|---:|---:|
{rf}

## Schema

{ot.schema_table(pl.scan_parquet(tmp / "data" / "part-0000.parquet"), NOTES)}

`lookups/`: `samples.parquet` (sample code, SAMPID, tissue, donor, center, SMGEBTCH, SMRIN,
SMTSISCH, SEX, AGE, DTHHRDY, ...), `genes.parquet` (gene code, Name, Description, chrom,
gene_type, length, flags), `tissues.parquet`, `donors.parquet`, `centers.parquet`.

## FE / cluster levels

| FE | Full | ref |
|---|---:|---:|
{lvt}
"""
    (tmp / "README.md").write_text(readme)
    tmp.rename(a.out)
    print(readme)


if __name__ == "__main__":
    main()
