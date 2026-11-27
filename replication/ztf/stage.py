# SPDX-License-Identifier: GPL-3.0-or-later
"""ZTF stage 1: download ZTF DR24 light-curve partitions and flatten them to one row per detection.

    python -I repo/replication/ztf/stage.py --out build/ztf/stage --npix 3/385 [3/386 ...]
    python -I repo/replication/ztf/stage.py --out build/ztf/stage --first 20     # first 20 partitions
    python -I repo/replication/ztf/stage.py --out build/ztf/stage --all --workers 8

Source: IRSA's HATS copy of ZTF DR24 light curves on AWS open data (bucket ``ipac-irsa-ztf``,
us-east-1, no account needed): 5,401,134,101 light curves (one per object x filter), 8.4 TB in
12,485 HEALPix partitions (Norder/Dir/Npix). Each light curve holds lists of hmjd, mag, magerr,
clrcoeff and catflags; this script explodes them to one row per detection and keeps:

  objectid, filterid, fieldid, rcid, objra, objdec, hmjd, mag, magerr, clrcoeff, catflags

One output file per partition, ``<out>/Norder=N_Npix=P.parquet``; partitions already written are
skipped. Raw downloads go to ``<out>/_raw`` and are deleted after flattening unless --keep-raw.
Exposure IDs are built across all partitions afterwards by ``build.py`` (a quadrant's exposure
spans partitions).

Cite the dataset DOI 10.26131/IRSA598 and the IRSA/ZTF acknowledgements (see README.md).
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

BASE = "https://ipac-irsa-ztf.s3.amazonaws.com/ztf/enhanced/dr24/lc/hats/ztf_dr24_lc-hats"
LISTS = ["hmjd", "mag", "magerr", "clrcoeff", "catflags"]
KEEP = ["objectid", "filterid", "fieldid", "rcid", "objra", "objdec", "lightcurve"]


def partitions() -> list[tuple[int, int, int]]:
    """(Norder, Npix, num_rows) for every partition, from the catalog's partition_info.csv."""
    with urllib.request.urlopen(f"{BASE}/partition_info.csv", timeout=120) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode())))
    return [(int(x["Norder"]), int(x["Npix"]), int(x["num_rows"])) for x in rows]


def url(norder: int, npix: int) -> str:
    return f"{BASE}/dataset/Norder={norder}/Dir={10_000 * (npix // 10_000)}/Npix={npix}/part0.snappy.parquet"


def stage_one(norder: int, npix: int, out: str, keep_raw: bool, threads: int) -> tuple[str, int, float]:
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    import polars as pl

    t0 = time.time()
    dst = Path(out) / f"Norder={norder}_Npix={npix}.parquet"
    raw = Path(out) / "_raw" / f"Norder={norder}_Npix={npix}.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        tmp = raw.with_suffix(".part")
        urllib.request.urlretrieve(url(norder, npix), tmp)
        tmp.rename(raw)
    d = (
        pl.scan_parquet(raw)
        .select(KEEP)
        .unnest("lightcurve")
        .explode(LISTS)
        .filter(pl.col("hmjd").is_not_null())
        .collect()
    )
    tmp = dst.with_suffix(".part")
    d.write_parquet(tmp)
    tmp.rename(dst)
    if not keep_raw:
        raw.unlink()
    return dst.name, d.height, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--npix", nargs="+", help="partitions as Norder/Npix, e.g. 3/385")
    g.add_argument("--first", type=int, help="the first N partitions in partition_info.csv")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4, help="POLARS_MAX_THREADS per worker")
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()

    if a.npix:
        todo = [tuple(int(v) for v in s.split("/")) for s in a.npix]
    else:
        info = partitions()
        todo = [(n, p) for n, p, _ in (info if a.all else info[: a.first])]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    todo = [(n, p) for n, p in todo if not (Path(a.out) / f"Norder={n}_Npix={p}.parquet").exists()]
    print(f"{len(todo)} partition(s) to stage", flush=True)
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as pool:
        futs = [pool.submit(stage_one, n, p, a.out, a.keep_raw, a.threads) for n, p in todo]
        for f in as_completed(futs):
            name, rows, secs = f.result()
            print(f"{name}: {rows:,} detections in {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
