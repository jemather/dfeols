"""D11: CZ CELLxGENE Census (2025-01-30) to long cell x gene Parquet, nonzero counts only.

Reads the local SOMA copy (`raw/cellxgene/soma_2025-01-30/`, docs/sources/cellxgene.md) and
writes one row per (cell, gene) with count > 0, for primary cells with a known cell type.
Rules and columns: docs/cellxgene.md (Ted 2026-11-11/12).

    cd /srv/dfeols && .venv/bin/python -I repo/replication/cellxgene/build.py --tier R0 --out build/cellxgene/R0

Output in --out:
  data/part-00000.parquet ...   rows sorted by (cell, gene); a part never splits a cell
  cells.parquet                 every obs field of the kept cells, plus the codes
  genes.parquet                 var plus codes and gene flags
  build.json                    tier, row and level counts, per-part rows

Codes (cell_type, dataset, donor, assay) are built over the organism's whole primary sample,
so they are the same in every tier. Each part's row count is checked against obs `nnz`.
Parts are written to a temp name and renamed, and a rerun skips parts already done.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SOMA = Path("raw/cellxgene/soma_2025-01-30")

ONEK1K = "3faad104-2ab8-4434-816d-474d8d2641db"
R1_DATASETS = [
    "53d208b0-2cfd-4366-9866-c3c6114081bc",  # Tabula Sapiens - All Cells
    "1a38e762-2465-418f-b81c-6a4bce261c34",  # prenatal and postnatal cortex (nuclei)
    "d4e69e01-3ba2-4d6b-a15d-e7048f78f22e",  # adult heart (cells and nuclei)
    "a5d5c529-8a1f-40b5-bda3-35208970070d",  # multiple cortical areas, Smart-seq v4 (nuclei)
    "ca46ffe6-0a26-4d8d-82fa-81722daf1102",  # white adipose (cells and nuclei)
    "66d15835-5dc8-4e96-b0eb-f48971cb65e8",  # pancreas, Smart-seq2
    ONEK1K,  # OneK1K PBMC, 981 donors (Ted 2026-11-12)
]
R0_PER_1000 = 3  # R0: cells of R1 with hash(soma_joinid) % 1000 < 3
TIERS = {"R0": "homo_sapiens", "R1": "homo_sapiens", "H": "homo_sapiens", "M": "mus_musculus"}
FULL_LENGTH_ASSAYS = ("Smart-seq2", "Smart-seq v4")

OBS_COLS = [
    "soma_joinid", "dataset_id", "assay", "assay_ontology_term_id", "cell_type",
    "cell_type_ontology_term_id", "development_stage", "development_stage_ontology_term_id",
    "disease", "disease_ontology_term_id", "donor_id", "is_primary_data", "observation_joinid",
    "self_reported_ethnicity", "self_reported_ethnicity_ontology_term_id", "sex",
    "sex_ontology_term_id", "suspension_type", "tissue", "tissue_ontology_term_id", "tissue_type",
    "tissue_general", "tissue_general_ontology_term_id", "raw_sum", "nnz", "raw_mean_nnz",
    "raw_variance_nnz", "n_measured_vars",
]
SCHEMA = pa.schema([
    ("cell", pa.int32()), ("gene", pa.int32()), ("count", pa.int32()), ("log_count", pa.float32()),
    ("cell_type", pa.int16()), ("donor", pa.int32()), ("dataset", pa.int16()), ("assay", pa.int8()),
    ("nucleus", pa.int8()), ("full_length", pa.int8()), ("organoid", pa.int8()),
    ("log_gene_length", pa.float32()), ("mt_gene", pa.int8()), ("lnc_gene", pa.int8()),
])


def r0_mask(joinid: np.ndarray) -> np.ndarray:
    h = (joinid.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)
    return (h % np.uint64(1000)) < R0_PER_1000


def codes(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    levels = sorted(values.astype(str).unique())
    m = {v: i for i, v in enumerate(levels)}
    return values.astype(str).map(m).to_numpy(), levels


def load_tables(soma: Path, organism: str, tier: str):
    import tiledbsoma as ts

    with ts.open(str(soma / "census_data" / organism)) as e:
        obs = e.obs.read(column_names=OBS_COLS).concat().to_pandas()
        var = e.ms["RNA"].var.read().concat().to_pandas()
    obs = obs[obs.is_primary_data & (obs.cell_type.astype(str) != "unknown")].copy()
    # codes over the organism's whole kept sample, so tiers share them
    obs["cell_type_code"], ct_levels = codes(obs.cell_type_ontology_term_id)
    obs["dataset_code"], ds_levels = codes(obs.dataset_id)
    obs["assay_code"], as_levels = codes(obs.assay_ontology_term_id)
    dd = obs.dataset_id.astype(str) + "|" + obs.donor_id.astype(str)
    obs["donor_code"], dn_levels = codes(dd)
    obs["nucleus"] = (obs.suspension_type.astype(str) == "nucleus").astype(np.int8)
    obs["full_length"] = obs.assay.astype(str).isin(FULL_LENGTH_ASSAYS).astype(np.int8)
    obs["organoid"] = (obs.tissue_type.astype(str) == "organoid").astype(np.int8)
    if tier in ("R0", "R1"):
        obs = obs[obs.dataset_id.astype(str).isin(R1_DATASETS)]
    if tier == "R0":
        obs = obs[r0_mask(obs.soma_joinid.to_numpy())]
    obs = obs.sort_values("soma_joinid").reset_index(drop=True)
    var = var.sort_values("soma_joinid").reset_index(drop=True)
    assert (var.soma_joinid.to_numpy() == np.arange(len(var))).all()
    var["log_gene_length"] = np.log(var.feature_length.astype(np.float64)).astype(np.float32)
    var["mt_gene"] = var.feature_name.astype(str).str.startswith("MT-").astype(np.int8)
    var["lnc_gene"] = (var.feature_type.astype(str) == "lncRNA").astype(np.int8)
    levels = {"cell_type": ct_levels, "dataset": ds_levels, "assay": as_levels, "donor": dn_levels}
    return obs, var, levels


def plan_parts(obs: pd.DataFrame, rows_per_part: int) -> list[dict]:
    """Consecutive kept cells, cut when a part would pass rows_per_part."""
    parts, start, acc = [], 0, 0
    nnz = obs.nnz.to_numpy()
    for i, n in enumerate(nnz):
        if acc and acc + n > rows_per_part:
            parts.append((start, i, acc))
            start, acc = i, 0
        acc += int(n)
    parts.append((start, len(nnz), acc))
    return [{"part": k, "lo": a, "hi": b, "rows": r} for k, (a, b, r) in enumerate(parts)]


_W: dict = {}


def _init(soma: str, organism: str, cells_path: str, genes_path: str):
    _W["X"] = f"{soma}/census_data/{organism}/ms/RNA/X/raw"
    _W["cells"] = pd.read_parquet(cells_path, columns=[
        "soma_joinid", "cell_type_code", "donor_code", "dataset_code", "assay_code",
        "nucleus", "full_length", "organoid"])
    g = pd.read_parquet(genes_path, columns=["log_gene_length", "mt_gene", "lnc_gene"])
    _W["genes"] = {c: g[c].to_numpy() for c in g.columns}


def build_part(task: dict, out_dir: str) -> dict:
    import tiledbsoma as ts

    out = Path(out_dir) / f"part-{task['part']:05d}.parquet"
    if out.exists() and pq.ParquetFile(out).metadata.num_rows == task["rows"]:
        return {**task, "skipped": True}
    t0 = time.time()
    cells = _W["cells"].iloc[task["lo"]:task["hi"]]
    ids = cells.soma_joinid.to_numpy()
    span = int(ids[-1] - ids[0] + 1)
    # dense parts (H, R1: a few dropped cells) read the joinid range and filter;
    # sparse ones (R0's sampled cells) read the listed cells only
    dense = len(ids) >= 0.2 * span
    with ts.open(_W["X"]) as X:
        coords = (slice(int(ids[0]), int(ids[-1])),) if dense else (ids,)
        tb = pa.concat_tables(list(X.read(coords=coords).tables()))
    c = tb["soma_dim_0"].to_numpy()
    g = tb["soma_dim_1"].to_numpy()
    v = tb["soma_data"].to_numpy()
    del tb
    if len(ids) < span:
        keep = np.isin(c, ids)
        if not keep.all():
            c, g, v = c[keep], g[keep], v[keep]
    order = np.argsort(c * np.int64(1 << 20) + g, kind="stable")
    c, g, v = c[order], g[order], v[order]
    del order
    if len(c) != task["rows"]:
        raise RuntimeError(f"part {task['part']}: {len(c)} rows read, obs nnz says {task['rows']}")
    cnt = v.astype(np.int32)
    if not (cnt == v).all() or cnt.min() < 1:
        raise RuntimeError(f"part {task['part']}: non-integer or < 1 count")
    pos = np.searchsorted(ids, c)  # row -> cell position in this part
    cols = {
        "cell": c.astype(np.int32), "gene": g.astype(np.int32), "count": cnt,
        "log_count": np.log(v, dtype=np.float32),
        "cell_type": cells.cell_type_code.to_numpy().astype(np.int16)[pos],
        "donor": cells.donor_code.to_numpy().astype(np.int32)[pos],
        "dataset": cells.dataset_code.to_numpy().astype(np.int16)[pos],
        "assay": cells.assay_code.to_numpy().astype(np.int8)[pos],
        "nucleus": cells.nucleus.to_numpy()[pos], "full_length": cells.full_length.to_numpy()[pos],
        "organoid": cells.organoid.to_numpy()[pos],
        "log_gene_length": _W["genes"]["log_gene_length"][g],
        "mt_gene": _W["genes"]["mt_gene"][g], "lnc_gene": _W["genes"]["lnc_gene"][g],
    }
    table = pa.table({k: pa.array(cols[k], type=SCHEMA.field(k).type) for k in SCHEMA.names},
                     schema=SCHEMA)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    pq.write_table(table, tmp, compression="zstd", row_group_size=8_000_000)
    os.replace(tmp, out)
    return {**task, "skipped": False, "seconds": round(time.time() - t0, 1),
            "bytes": out.stat().st_size, "first_cell": int(ids[0]), "last_cell": int(ids[-1])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=sorted(TIERS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--soma", type=Path, default=SOMA)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rows-per-part", type=int, default=150_000_000)
    a = ap.parse_args(argv)
    organism = TIERS[a.tier]
    t0 = time.time()
    (a.out / "data").mkdir(parents=True, exist_ok=True)
    obs, var, levels = load_tables(a.soma, organism, a.tier)
    cells_path, genes_path = a.out / "cells.parquet", a.out / "genes.parquet"
    obs.to_parquet(cells_path, index=False)
    var.to_parquet(genes_path, index=False)
    with open(a.out / "levels.json", "w") as f:
        json.dump(levels, f)
    tasks = plan_parts(obs, a.rows_per_part)
    print(f"{a.tier}: {len(obs):,} cells, {int(obs.nnz.sum()):,} rows, {len(tasks)} parts", flush=True)
    stale = {p.name for p in (a.out / "data").glob("part-*.parquet")} - {
        f"part-{t['part']:05d}.parquet" for t in tasks}
    if stale:
        raise SystemExit(f"parts not in this plan (different --rows-per-part?): {sorted(stale)[:5]}")
    done = []
    ctx = mp.get_context("spawn")  # Polars/Arrow threads in the parent: never fork
    with ctx.Pool(a.workers, initializer=_init,
                  initargs=(str(a.soma), organism, str(cells_path), str(genes_path))) as pool:
        for r in pool.imap_unordered(_call, [(t, str(a.out / "data")) for t in tasks]):
            done.append(r)
            print(f"part {r['part']:5d} {r['rows']:>12,} rows "
                  f"{'skipped' if r['skipped'] else str(r['seconds']) + ' s'} "
                  f"({len(done)}/{len(tasks)})", flush=True)
    done.sort(key=lambda r: r["part"])
    summary = {
        "tier": a.tier, "organism": organism, "source": str(a.soma),
        "census_release": "2025-01-30", "rows": int(obs.nnz.sum()), "cells": len(obs),
        "genes": len(var), "parts": len(tasks),
        "levels": {"cell_type": int(obs.cell_type_code.nunique()),
                   "dataset": int(obs.dataset_code.nunique()),
                   "donor": int(obs.donor_code.nunique()), "assay": int(obs.assay_code.nunique())},
        "cells_nucleus": int(obs.nucleus.sum()), "cells_full_length": int(obs.full_length.sum()),
        "cells_organoid": int(obs.organoid.sum()),
        "bytes": int(sum((a.out / "data" / f"part-{t['part']:05d}.parquet").stat().st_size for t in tasks)),
        "seconds": round(time.time() - t0, 1), "part_list": done,
    }
    with open(a.out / "build.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "part_list"}, indent=1))


def _call(args):
    return build_part(*args)


if __name__ == "__main__":
    main()
