# SPDX-License-Identifier: GPL-3.0-or-later
"""ZTF stage 1: flatten downloaded ZTF DR24 light-curve partitions to one row per detection.

    python -I repo/replication/ztf/stage.py --raw raw/ztf/dr24 --out build/ztf/stage --npix 3/385 [3/386 ...]
    python -I repo/replication/ztf/stage.py --raw raw/ztf/dr24 --out build/ztf/stage --all --workers 4

Source: IRSA's HATS copy of ZTF DR24 light curves on AWS open data (bucket ``ipac-irsa-ztf``,
us-east-1, public): 5,401,134,101 light curves (one per object x filter), 8.4 TB in 12,485
HEALPix partitions. This script downloads nothing. The partitions must already be under
``--raw`` at the bucket's own layout, ``Norder=N/Dir=D/Npix=P/part0.snappy.parquet``; the URLs
and copy commands are in ``docs/sources/ztf.md``. ``--all`` takes every partition found there.

Each light curve holds lists of hmjd, mag, magerr, clrcoeff and catflags. This script explodes
them to one row per detection and keeps:

  objectid, filterid, fieldid, rcid, objra, objdec, hmjd, mag, magerr, clrcoeff, catflags

One output file per partition, ``<out>/Norder=N_Npix=P.parquet``, written by a streaming sink;
partitions already written are skipped. Exposure IDs are built across all partitions afterwards
by ``build.py`` (a quadrant's exposure spans partitions).

Cite the dataset DOI 10.26131/IRSA598 and the IRSA/ZTF acknowledgements (see README.md).
"""

from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

LISTS = ["hmjd", "mag", "magerr", "clrcoeff", "catflags"]
KEEP = ["objectid", "filterid", "fieldid", "rcid", "objra", "objdec", "lightcurve"]
PART = re.compile(r"Norder=(\d+)/Dir=\d+/Npix=(\d+)/part0\.snappy\.parquet$")


def raw_path(raw: str, norder: int, npix: int) -> Path:
    return Path(raw) / f"Norder={norder}/Dir={10_000 * (npix // 10_000)}/Npix={npix}/part0.snappy.parquet"


def found(raw: str) -> list[tuple[int, int]]:
    """(Norder, Npix) of every partition file under ``raw``."""
    out = []
    for p in sorted(Path(raw).glob("Norder=*/Dir=*/Npix=*/part0.snappy.parquet")):
        m = PART.search(p.as_posix())
        if m:
            out.append((int(m[1]), int(m[2])))
    return out


def stage_one(raw: str, norder: int, npix: int, out: str, threads: int) -> tuple[str, int, float]:
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    import polars as pl

    t0 = time.time()
    dst = Path(out) / f"Norder={norder}_Npix={npix}.parquet"
    tmp = dst.with_suffix(".part")
    (
        pl.scan_parquet(raw_path(raw, norder, npix))
        .select(KEEP)
        .unnest("lightcurve")
        .explode(LISTS)
        .filter(pl.col("hmjd").is_not_null())
        .sink_parquet(tmp)
    )
    tmp.rename(dst)
    rows = pl.scan_parquet(dst).select(pl.len()).collect().item()
    return dst.name, rows, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="downloaded partitions, e.g. raw/ztf/dr24")
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--npix", nargs="+", help="partitions as Norder/Npix, e.g. 3/385")
    g.add_argument("--all", action="store_true", help="every partition found under --raw")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4, help="POLARS_MAX_THREADS per worker")
    a = ap.parse_args()

    todo = found(a.raw) if a.all else [tuple(int(v) for v in s.split("/")) for s in a.npix]
    missing = [f"{n}/{p}" for n, p in todo if not raw_path(a.raw, n, p).exists()]
    if missing:
        raise SystemExit(f"not downloaded (see docs/sources/ztf.md): {' '.join(missing)}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    todo = [(n, p) for n, p in todo if not (Path(a.out) / f"Norder={n}_Npix={p}.parquet").exists()]
    print(f"{len(todo)} partition(s) to stage", flush=True)
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as pool:
        futs = [pool.submit(stage_one, a.raw, n, p, a.out, a.threads) for n, p in todo]
        for f in as_completed(futs):
            name, rows, secs = f.result()
            print(f"{name}: {rows:,} detections in {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
