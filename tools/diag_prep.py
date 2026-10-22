# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Read-only diagnosis of a distributed prep that failed with
"prep: encoded N rows, expected M". Run on a login node in the dfeols venv:

    python diag_prep.py /scratch/ted/dfeols/bench/shards_panel_1e10_T100_bal1_G5000_by_<part>_2000000 \
        /scratch/ted/dfeols/bench/raw_panel_1e10_T100_bal1_G5000

It changes nothing. It reports:
  1. duplicate keys in the code tables (_codes/), which would multiply rows in the encode join;
  2. per encode task and attempt, rows written vs rows in that task's input files;
  3. for any task that wrote too many rows, whether the extra rows are exact duplicates and
     which input file / shard they belong to.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

root, raw = Path(sys.argv[1]), Path(sys.argv[2])
rows_per_task = float(sys.argv[3]) if len(sys.argv) > 3 else 1e7


def nrows(p):
    return pq.read_metadata(str(p)).num_rows


def nrows_many(paths, what):
    """Footer reads in parallel: one at a time, 50,000 files take over an hour on Lustre."""
    from concurrent.futures import ThreadPoolExecutor

    paths = list(paths)
    out = []
    with ThreadPoolExecutor(64) as pool:
        for i, n in enumerate(pool.map(nrows, paths), 1):
            out.append(n)
            if i % 5000 == 0 or i == len(paths):
                print(f"  read {i:,}/{len(paths):,} {what}", flush=True)
    return out


print(f"shard folder: {root}")
# 1. code tables ------------------------------------------------------------
codes = root / "_codes"
for f in sorted(codes.glob("*.parquet")) if codes.exists() else []:
    df = pl.read_parquet(f)
    keys = [c for c in df.columns if not c.startswith("__fe__") and c not in ("b", "__shard")]
    if f.name == "__shard_map.parquet":
        keys = [c for c in df.columns if c != "__shard"]
    if not keys:
        continue
    dup = df.height - df.select(keys).n_unique()
    print(f"  code table {f.name}: {df.height:,} rows, {dup:,} duplicate keys on {keys}")

# 2. pieces per task and attempt ---------------------------------------------
files = sorted(str(p) for p in raw.glob("*.parquet"))
rows_in = nrows_many(files, "input files")
batches, cur, n = [], [], 0
for i, r in enumerate(rows_in):
    cur.append(i)
    n += r
    if n >= rows_per_task:
        batches.append((cur, n))
        cur, n = [], 0
if cur:
    batches.append((cur, n))
print(f"raw: {len(files)} files, {sum(rows_in):,} rows; {len(batches)} encode tasks")

pat = re.compile(r"^([sb])(\d+)_t(\d+)_([0-9a-f]+)_(\d+)\.parquet$")
attempts = defaultdict(lambda: defaultdict(list))  # tid -> token -> [(path, rows)]
found = []
for d in (root / "shards", root / "_buckets"):
    if d.exists():
        found += [(p, m) for p in d.iterdir() if (m := pat.match(p.name))]
for (p, m), n in zip(found, nrows_many([p for p, _ in found], "piece files")):
    attempts[int(m.group(3))][m.group(4)].append((p, n))
print(f"pieces found for {len(attempts)} tasks")

multi = {t: v for t, v in attempts.items() if len(v) > 1}
print(f"tasks with more than one attempt on disk: {len(multi)}")
bad = []
for tid, toks in sorted(attempts.items()):
    expect = batches[tid][1] if tid < len(batches) else None
    for tok, ps in toks.items():
        got = sum(r for _, r in ps)
        if got != expect:
            bad.append((tid, tok, got, expect, ps))
print(f"attempts whose rows differ from their input: {len(bad)}")
for tid, tok, got, expect, ps in bad[:10]:
    print(
        f"  task {tid} attempt {tok}: wrote {got:,} rows, input {expect:,} "
        f"(files {[Path(files[i]).name for i in batches[tid][0]]}); {len(ps)} pieces"
    )

# 3. look inside the first over-full attempt -------------------------------
for tid, _tok, got, expect, ps in bad[:2]:
    if expect is None or got <= expect:
        continue
    df = pl.concat([pl.read_parquet(p) for p, _ in ps], how="diagonal_relaxed")
    keys = [c for c in ("unit", "time") if c in df.columns]
    exact = df.height - df.unique().height
    by_key = df.height - df.select(keys).unique().height if keys else None
    print(f"  task {tid}: {df.height:,} rows, {exact:,} exact duplicate rows, {by_key:,} duplicate {keys}")
    if keys and by_key:
        d = df.filter(pl.struct(keys).is_duplicated())
        print("   duplicated units:", d["unit"].n_unique(), "min/max", d["unit"].min(), d["unit"].max())
        per_piece = [
            (
                p.name,
                pl.read_parquet(p)
                .filter(pl.struct(keys).is_in(d.select(pl.struct(keys)).to_series().implode()))
                .height,
            )
            for p, _ in ps
        ]
        print("   pieces holding them:", [x for x in per_piece if x[1]])
