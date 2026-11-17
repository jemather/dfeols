# SPDX-License-Identifier: GPL-3.0-or-later
"""Package the D10 A build (build.py + reference.py output) as outbox/archs4_v1 (CLAUDE.md layout).

    cd /srv/dfeols && .venv/bin/python -I repo/replication/archs4/package.py \
        --build build/archs4 --out outbox/archs4_v1 --commit <sha>

data/part-NNNN.parquet: the build's chunk files (contiguous ARCHS4 sample ranges, sorted by
sample then gene) concatenated in order into parts of about 1 GB. ref/data/ (R0),
ref/reference.json, lookups/, README.md. The manifest is written afterwards (CLAUDE.md).
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
    "series": ["series"],
    "instrument": ["instrument"],
    "gene^series": ["gene", "series"],
    "gene^instrument": ["gene", "instrument"],
}
NOTES = {
    "sample": "ARCHS4 sample index (column of data/expression); FE; lookups/samples.parquet",
    "gene": "code of the protein-coding gene; lookups/genes.parquet",
    "series": "code of the first GEO series (GSE); cluster and partition key; lookups/series.parquet",
    "instrument": "code of instrument_model; lookups/instruments.parquet",
    "year": "GEO submission year",
    "log_cpm": "ln(1 + CPM), CPM = count / library x 1e6, library = reads over all 67,186 genes: the outcome",
    "count": "read count",
    "log_lib": "ln(library size)",
    "total_rna": "GEO molecule_ch1 == 'total RNA' (else 'polyA RNA'); an unreliable label",
    "log_nc_share": "log10(share of reads in RN7SK, RN7SL1/2, RMRP, RPPH1 + 1e-6): ribo-depletion measure",
    "histone_gene": "replication-dependent histone gene (symbol H1-[0-6], H2AC*, H2BC*, H3C*, H4C*)",
    "mt_gene": "symbol starts with MT-",
}
LADDER = {
    "a1": "sample + gene",
    "a2": "sample + gene^series",
    "a3": "sample + gene^series + gene^instrument",
}
X = "log_nc_share:histone_gene + total_rna:histone_gene"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=Path("build/archs4"))
    ap.add_argument("--out", type=Path, default=Path("outbox/archs4_v1"))
    ap.add_argument("--commit", required=True, help="dfeols repo commit of the build code")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; packages are never changed after creation (build _v2)")
    ref_json = a.build / "R0" / "reference.json"
    if not ref_json.exists():
        raise SystemExit(f"{ref_json} missing: run reference.py first")
    info = json.loads((a.build / "build.json").read_text())
    tmp = a.out.with_name(a.out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "data").mkdir(parents=True)

    src = sorted((a.build / "data").glob("part-*.parquet"))
    groups, cur, size = [], [], 0
    for f in src:
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
        want = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in g)
        if r["rows"] != want:
            raise SystemExit(f"{p}: {r['rows']} rows, expected {want}")
        parts.append(
            {
                "file": f"data/{p.name}",
                "rows": r["rows"],
                "lo": r["lo"],
                "hi": r["hi"],
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
    for n in ("samples", "genes", "series", "instruments"):
        shutil.copy2(a.build / f"{n}.parquet", tmp / "lookups" / f"{n}.parquet")

    # full levels: counts from the build, plus the few that need a pass over the data
    k = info["counts"]
    data = pl.scan_parquet(tmp / "data" / "*.parquet")
    gi = data.select("gene", "instrument").unique().collect(engine="streaming")
    full = {
        "sample": k["kept"],
        "gene": gi["gene"].n_unique(),
        "series": k["series"],
        "instrument": gi["instrument"].n_unique(),
        "gene^series": k["gene_x_series"],
        "gene^instrument": gi.height,
    }
    lvt = "\n".join(f"| {n} | {full[n]:,} | {ref_levels[n]:,} |" for n in FES)
    pt = "\n".join(
        f"| {p['file']} | {p['rows']:,} | {p['lo']}..{p['hi']} | {p['bytes'] / 1e9:.2f} GB |" for p in parts
    )
    py = ref["pyfixest"]
    rf = "\n".join(
        f"| {s} | `{fe}` | {py[s]['N']:,} | "
        + " | ".join(
            f"{py[s]['coefs']['CRV1'][t][0]:.6f} ({py[s]['coefs']['CRV1'][t][1]:.6f})" for t in X.split(" + ")
        )
        + " |"
        for s, fe in LADDER.items()
        if s in py
    )
    rules = info["rules"]
    readme = f"""# archs4_v1

ARCHS4 v2.5 human gene counts (uniformly re-aligned public GEO/SRA RNA-seq), long format: one
row per sample x gene for bulk samples and, per GEO series, the protein-coding genes expressed
in the series, zeros included. {total:,} rows in {len(parts)} parts. A dfeols test dataset (a
multi-billion-row panel whose big FEs nest in the series partition), not a research project.

- **Source:** `human_gene_v2.5.h5` (Ensembl 107; https://maayanlab.cloud/archs4/download.html;
  SHA1 `ae96de0519b9f008b0dc3a9f944ee9007daf2f6a` matches). Derived from public GEO/SRA data;
  cite Lachmann et al. (2018), Nat Commun 9:1366. Notes: `docs/gtex.md` and
  `docs/sources/gtex.md` on valiant
- **Build:** `repo/replication/archs4/build.py` and `package.py` (jemather/dfeols `{a.commit}`)
- **Sample** (Decisions 2026-11-17): of {k["samples_all"]:,} samples, {k["meta_ok"]:,} are bulk
  (single-cell probability < 0.5, library source transcriptomic, molecule total or polyA RNA),
  {k["lib_ok"]:,} of those have >= {rules["min_lib"]:,} reads, and {k["kept"]:,} are in the
  {k["series"]:,} series (first GSE) with >= {rules["min_series"]} such samples. Genes:
  protein-coding, kept in a series when CPM >= {rules["cpm_min"]:g} in >= {rules["share_min"]:.0%} of its
  samples ({k["gene_x_series"]:,} gene x series pairs)
- **File defect:** {len(info.get("unreadable_samples", []))} samples have gzip chunks that fail to decompress
  in the published H5 ({", ".join(map(str, info.get("unreadable_samples", [])))}); they are dropped

## Model (Decisions 2026-11-17)

`log_cpm ~ {X} | FEs`, CRV1 clustered by `series` (also hetero), partition `series`.

| Spec | FEs | What it tests |
|---|---|---|
| a1 | `sample + gene` | two-way, 22.9k crossing levels, billions of rows |
| a2 | `sample + gene^series` | a huge FE nested in the partition: no crossing system |
| a3 | `sample + gene^series + gene^instrument` | a crossing FE on top of a2 |

Expected sign: `log_nc_share:histone_gene` positive (replication-dependent histone mRNAs have no
poly-A tail, so they show up in ribo-depleted libraries, which have a high share of
non-polyadenylated ncRNA reads). `total_rna:histone_gene` uses GEO's label, which is unreliable
(on a test it had the wrong sign), so it has no expected sign. In a2 and a3 it is identified only
in the {k["series_mixed"]:,} series that mix the two labels.

## Parts

The build's chunk files concatenated in order: contiguous sample ranges, sorted by sample then
gene. Series are mostly contiguous in the H5, but dfeols prep re-shards by series anyway.

| File | Rows | Sample range | Size |
|---|---:|---|---:|
{pt}

## Reference tier (`ref/`)

`ref/data/part-0000.parquet`: {info["R0"]["rule"]}; {info["R0"]["series"]} series,
{info["R0"]["rows"]:,} rows, same columns. `ref/reference.json`: pyfixest (demeaning tolerance
1e-13) for a1-a3, CRV1 by series and hetero, plus the local dfeols fits (partitioned by series and
by sample, CG iter_tol 1e-10, level space off). CRV1 coefficients (SE):

| Spec | FEs | N | {" | ".join(f"`{t}`" for t in X.split(" + "))} |
|---|---|---:|---:|---:|
{rf}

## Schema

{ot.schema_table(pl.scan_parquet(tmp / "data" / "part-0000.parquet"), NOTES)}

`lookups/`: `samples.parquet` (all {k["samples_all"]:,} samples: GSM, GSE, metadata, rule flags,
library size, codes), `genes.parquet` (protein-coding genes: code, H5 row, Ensembl ID, symbol,
flags), `series.parquet` (code, GSE, samples, genes kept, rows, mixed labels),
`instruments.parquet`.

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
