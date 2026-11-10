# SPDX-License-Identifier: GPL-3.0-or-later
"""D5: put the yellow 2009-10 lat/lon pickups and dropoffs in TLC taxi zones.

    python repo/replication/nyc_tlc/zones.py [--stage build/nyc_tlc/stage] [--out build/nyc_tlc/zoned]
        [--zones raw/nyc_tlc/docs/taxi_zones.zip] [--months 2009-01,2010-12] [--workers 4] [--force]

Run from the project root. Yellow files before 2011 carry coordinates, not zone IDs
(``docs/sources/nyc_tlc.md``). For each such stage file this writes a copy to
``<out>/yellow_YYYY-MM.parquet`` with ``pu_location_id`` and ``do_location_id`` filled by
point-in-polygon on the TLC taxi zone shapefile (EPSG:2263, NY State Plane feet; coordinates
are WGS84 lon/lat, projected with pyproj), and a ``.json`` sidecar with the counts. Every
other column is unchanged and rows out equal rows in.

Rules (Decisions 2026-11-09): a point in no zone (outside every polygon, including the 0/0 and
other junk coordinates, or a missing coordinate) gets a null zone, so the package drops the
trip. A point on a shared boundary (in two or more zones) gets the lowest LocationID; the
sidecar counts them. A file whose sidecar matches its stage file's size and mtime is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import numpy as np
import polars as pl

# generous WGS84 box around the five boroughs and Newark airport (zone 1); points outside it
# cannot be in a zone, so they skip the projection and the tree query
BOX = (-74.35, 40.45, -73.65, 41.0)


def load_zones(path: Path):
    """(LocationID array, shapely polygons) from the TLC shapefile zip."""
    import pyogrio
    import shapely

    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(path) as z:
            z.extractall(d)
        shp = next(Path(d).rglob("*.shp"))
        meta, _, wkb, fields = pyogrio.raw.read(shp)
    if "2263" not in str(meta["crs"]):
        raise SystemExit(f"{path}: CRS {meta['crs']}, expected EPSG:2263")
    ids = np.asarray(fields[list(meta["fields"]).index("LocationID")], dtype=np.int16)
    if sorted(ids.tolist()) != list(range(1, 264)):
        raise SystemExit(f"{path}: LocationIDs are not exactly 1..263")
    geoms = shapely.from_wkb(wkb)
    shapely.prepare(geoms)
    return ids, geoms


def assign(lon: np.ndarray, lat: np.ndarray, ids, tree, transformer) -> tuple[np.ndarray, dict]:
    """Zone of each point (0 where none), and counts."""
    import shapely

    n = len(lon)
    out = np.zeros(n, dtype=np.int16)
    ok = np.isfinite(lon) & np.isfinite(lat)
    ok &= (lon >= BOX[0]) & (lon <= BOX[2]) & (lat >= BOX[1]) & (lat <= BOX[3])
    idx = np.flatnonzero(ok)
    x, y = transformer.transform(lon[idx], lat[idx])
    pts = shapely.points(x, y)
    pi, gi = tree.query(pts, predicate="intersects")  # pairs (point index, polygon index)
    z = ids[gi]
    # lowest LocationID per point: sort by (point, zone) and keep each point's first
    order = np.lexsort((z, pi))
    pi, z = pi[order], z[order]
    first = np.ones(len(pi), dtype=bool)
    first[1:] = pi[1:] != pi[:-1]
    out[idx[pi[first]]] = z[first]
    return out, {
        "in_box": int(len(idx)),
        "matched": int(first.sum()),
        "boundary": int(len(pi) - first.sum()),  # extra matches beyond the first
    }


def one(src: Path, dst: Path, zones_path: Path) -> dict:
    import pyproj
    import shapely

    t = time.time()
    ids, geoms = load_zones(zones_path)
    tree = shapely.STRtree(geoms)
    tf = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
    df = pl.read_parquet(src)
    if df["pu_location_id"].is_not_null().any() or df["do_location_id"].is_not_null().any():
        raise SystemExit(f"{src}: already has zone IDs; not a lat/lon file")
    rec = {"source": str(src), "source_bytes": src.stat().st_size, "source_mtime": src.stat().st_mtime,
           "rows": df.height}  # fmt: skip
    cols = {}
    for end, lon, lat in [("pu", "pickup_longitude", "pickup_latitude"), ("do", "dropoff_longitude", "dropoff_latitude")]:
        z, c = assign(df[lon].to_numpy().astype(float), df[lat].to_numpy().astype(float), ids, tree, tf)
        cols[f"{end}_location_id"] = pl.Series(z).replace(0, None).cast(pl.Int16)
        rec |= {f"{end}_{k}": v for k, v in c.items()}
    df = df.with_columns(**cols)
    tmp = dst.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd", statistics=True)
    if pl.scan_parquet(tmp).select(pl.len()).collect().item() != rec["rows"]:
        raise SystemExit(f"{tmp}: row count changed")
    tmp.rename(dst)
    rec["seconds"] = round(time.time() - t, 1)
    rec["build"] = subprocess.run(["git", "-C", "repo", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()  # fmt: skip
    dst.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", type=Path, default=Path("build/nyc_tlc/stage"))
    ap.add_argument("--out", type=Path, default=Path("build/nyc_tlc/zoned"))
    ap.add_argument("--zones", type=Path, default=Path("raw/nyc_tlc/docs/taxi_zones.zip"))
    ap.add_argument("--months", default="2009-01,2010-12")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    lo, hi = a.months.split(",")
    a.out.mkdir(parents=True, exist_ok=True)
    todo = []
    for src in sorted((a.stage / "yellow").glob("yellow_*.parquet")):
        m = src.stem.split("_")[1]
        if not (lo <= m <= hi):
            continue
        dst = a.out / src.name
        side = dst.with_suffix(".json")
        if not a.force and dst.exists() and side.exists():
            s = json.loads(side.read_text())
            if s["source_bytes"] == src.stat().st_size and s["source_mtime"] == src.stat().st_mtime:
                print(f"skip {src.name} (done)", flush=True)
                continue
        todo.append((src, dst))
    failed = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, s, d, a.zones): s for s, d in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
                print(f"{futs[f].name}: {r['rows']:,} rows, pu {r['pu_matched']:,} do {r['do_matched']:,} "
                      f"zoned, boundary {r['pu_boundary']}/{r['do_boundary']}, {r['seconds']} s", flush=True)  # fmt: skip
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAILED {futs[f].name}: {e!r}", flush=True)
    print(f"{len(todo) - failed} written, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
