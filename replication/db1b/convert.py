# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert the raw DB1B and T-100 downloads to harmonised Parquet, one file per table-quarter
(T-100: per month), with a JSON sidecar per file.

    python repo/replication/db1b/convert.py [--tables Coupon,Market,Ticket,T100]
        [--from 1993Q1] [--to 2025Q2] [--raw raw/db1b] [--out build/db1b/stage]
        [--scratch tmp/db1b_convert] [--force]

Run from the project root. For each file:

  1. check every source zip against the SHA256SUMS in its folder;
  2. extract the CSV to --scratch, read it all as text, rename to snake_case (sources.snake);
  3. cast to sources.SCHEMA strictly (blank -> null; anything else that doesn't parse fails);
     split quarters join their column parts 1:1 on the keys; missing columns are typed nulls;
  4. check year/quarter (T-100: year/month) against the file name, write
     <out>/<table>/<table>_<Y>Q<Q>.parquet atomically, then the .json sidecar
     (sources and hashes, rows, null counts, build commit).

A file whose sidecar matches its sources' hashes is skipped, so a rerun resumes. Failures are
logged and the run carries on; the exit code is 1 if any file failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

log = logging.getLogger("db1b.convert")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def expected_sha(z: Path) -> str:
    sums = z.parent / "SHA256SUMS"
    for line in sums.read_text().splitlines():
        digest, _, name = line.partition("  ")
        if name == z.name:
            return digest
    raise ValueError(f"{z.name} not listed in {sums}")


def checked(z: Path) -> dict:
    digest = sha256(z)
    if digest != expected_sha(z):
        raise ValueError(f"{z}: sha256 {digest} differs from SHA256SUMS")
    return {"path": str(z), "sha256": digest, "bytes": z.stat().st_size}


def extract(z: Path, dest: Path, member: str | None = None) -> Path:
    """Extract the zip's CSV (or `member`) to dest; reading a raw zip is data only."""
    with zipfile.ZipFile(z) as zf:
        if member is None:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if len(csvs) != 1:
                raise ValueError(f"{z}: expected one CSV, found {csvs}")
            member = csvs[0]
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / Path(member).name
        with zf.open(member) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
    return out


def scan(csv: Path, schema: dict[str, str], keep: set[str] | None = None) -> pl.LazyFrame:
    """Read a CSV as text, rename to snake_case, cast to `schema`. `keep` limits the columns."""
    lf = pl.scan_csv(csv, infer_schema=False)
    raw = [c for c in lf.collect_schema().names() if c.strip()]  # trailing commas give unnamed columns
    exprs = []
    for c in raw:
        name = S.snake(c)
        if keep is not None and name not in keep:
            continue
        if name not in schema:
            raise ValueError(f"{csv}: column {c} ({name}) not in the schema")
        e = pl.col(c)
        if schema[name] != "String":
            e = e.str.strip_chars()
            e = pl.when(e == "").then(None).otherwise(e).cast(getattr(pl, schema[name]), strict=True)
        exprs.append(e.alias(name))
    return lf.select(exprs)


def harmonise(lf: pl.LazyFrame, schema: dict[str, str]) -> pl.LazyFrame:
    have = set(lf.collect_schema().names())
    return lf.select(
        [pl.col(c) if c in have else pl.lit(None, getattr(pl, t)).alias(c) for c, t in schema.items()]
    )


def build_commit() -> str:
    repo = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--", "replication/db1b"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def up_to_date(sidecar: Path, zips: tuple[Path, ...]) -> bool:
    if not sidecar.exists() or not sidecar.with_suffix(".parquet").exists():
        return False
    meta = json.loads(sidecar.read_text())
    return [s["sha256"] for s in meta["sources"]] == [expected_sha(z) for z in zips]


def write(lf: pl.LazyFrame, out: Path, checks: dict[str, int], meta: dict) -> dict:
    """Sink to out atomically, check the period columns, write the sidecar."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    lf.sink_parquet(tmp, compression="zstd", statistics=True)
    stats = (
        pl.scan_parquet(tmp)
        .select(
            pl.len().alias("rows"),
            *[(pl.col(c) != v).sum().alias(f"bad_{c}") for c, v in checks.items()],
            pl.all().null_count().name.prefix("null_"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    bad = {k: v for k, v in stats.items() if k.startswith("bad_") and v}
    if bad:
        tmp.unlink()
        raise ValueError(f"{out.name}: rows outside the file's period {bad}")
    os.replace(tmp, out)
    meta |= {
        "rows": stats["rows"],
        "nulls": {k[5:]: v for k, v in stats.items() if k.startswith("null_") and v},
        "bytes": out.stat().st_size,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def convert_quarter(src: S.Source, out_root: Path, scratch: Path, commit: str) -> dict:
    schema = S.SCHEMA[src.table]
    work = scratch / src.label
    shutil.rmtree(work, ignore_errors=True)
    try:
        sources = [checked(z) for z in src.zips]
        if src.layout == "split":
            keys = [S.snake(k) for k in S.SPLIT_KEYS[src.table]]
            parts, rows = [], []
            for i, z in enumerate(src.zips):
                csv = extract(z, work / f"p{i + 1}")
                lf = scan(csv, schema)
                rows.append(lf.select(pl.len()).collect(engine="streaming").item())
                parts.append(lf)
            if len(set(rows)) != 1:
                raise ValueError(f"{src.label}: column parts have different row counts {rows}")
            lf = parts[0]
            for p in parts[1:]:
                lf = lf.join(p, on=keys, how="inner", validate="1:1")
            # an inner join that loses rows is caught by the row check below
            expect_rows = rows[0]
        else:
            lf = scan(extract(src.zips[0], work), schema)
            expect_rows = None
        meta = {
            "table": src.table,
            "year": src.year,
            "quarter": src.quarter,
            "layout": src.layout,
            "sources": sources,
            "missing_columns": [c for c in schema if c not in lf.collect_schema().names()],
            "build_commit": commit,
            "polars": pl.__version__,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        out = out_root / src.table.lower() / f"{src.table.lower()}_{src.year}Q{src.quarter}.parquet"
        meta = write(harmonise(lf, schema), out, {"year": src.year, "quarter": src.quarter}, meta)
        if expect_rows is not None and meta["rows"] != expect_rows:
            raise ValueError(f"{src.label}: joined {meta['rows']:,} rows from parts of {expect_rows:,}")
        return meta
    finally:
        shutil.rmtree(work, ignore_errors=True)


def convert_t100(year: int, month: int, z: Path, out_root: Path, scratch: Path, commit: str) -> dict:
    schema = S.SCHEMA["T100"]
    work = scratch / f"t100_{year}_{month:02d}"
    shutil.rmtree(work, ignore_errors=True)
    try:
        sources = [checked(z)]
        lf = scan(extract(z, work, S.T100_MEMBER), schema)
        meta = {
            "table": "T100",
            "year": year,
            "month": month,
            "layout": "form",
            "sources": sources,
            "missing_columns": [c for c in schema if c not in lf.collect_schema().names()],
            "build_commit": commit,
            "polars": pl.__version__,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        out = out_root / "t100" / f"t100_{year}_{month:02d}.parquet"
        return write(harmonise(lf, schema), out, {"year": year, "month": month}, meta)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def parse_quarter(s: str) -> tuple[int, int]:
    y, q = s.upper().split("Q")
    return int(y), int(q)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default="Coupon,Market,Ticket,T100")
    ap.add_argument("--from", dest="first", default="1993Q1", type=parse_quarter)
    ap.add_argument("--to", dest="last", default="2025Q2", type=parse_quarter)
    ap.add_argument("--raw", default="raw/db1b", type=Path)
    ap.add_argument("--out", default="build/db1b/stage", type=Path)
    ap.add_argument("--scratch", default="tmp/db1b_convert", type=Path)
    ap.add_argument("--force", action="store_true", help="rebuild files whose sidecar is current")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    tables = a.tables.split(",")
    commit = build_commit()
    log.info("build commit %s, polars %s, out %s", commit, pl.__version__, a.out)
    jobs = []
    for src in S.catalog(a.raw, [t for t in tables if t in S.TABLES], a.first, a.last):
        out = a.out / src.table.lower() / f"{src.table.lower()}_{src.year}Q{src.quarter}.json"
        jobs.append(
            (src.label, out, src.zips, lambda src=src: convert_quarter(src, a.out, a.scratch, commit))
        )
    if "T100" in tables:
        for y, m, z in S.t100_months(a.raw):
            if a.first <= (y, (m - 1) // 3 + 1) <= a.last:
                out = a.out / "t100" / f"t100_{y}_{m:02d}.json"
                jobs.append(
                    (
                        f"T100_{y}-{m:02d}",
                        out,
                        (z,),
                        lambda y=y, m=m, z=z: convert_t100(y, m, z, a.out, a.scratch, commit),
                    )
                )

    failed, done, skipped = [], 0, 0
    for i, (label, sidecar, zips, run) in enumerate(jobs, 1):
        if not a.force and up_to_date(sidecar, zips):
            skipped += 1
            continue
        t0 = time.time()
        try:
            meta = run()
        except Exception as e:  # keep going; report at the end
            log.error("[%d/%d] %s FAILED: %s", i, len(jobs), label, e)
            failed.append((label, str(e)))
            continue
        done += 1
        log.info(
            "[%d/%d] %s %s: %s rows, %.0f MB, %.0f s",
            i,
            len(jobs),
            label,
            meta["layout"],
            f"{meta['rows']:,}",
            meta["bytes"] / 1e6,
            time.time() - t0,
        )
    log.info("done %d, skipped %d (current), failed %d", done, skipped, len(failed))
    for label, err in failed:
        log.info("FAILED %s: %s", label, err)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
