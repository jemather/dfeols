# SPDX-License-Identifier: GPL-3.0-or-later
"""D13: size of the product-level PPML sample, before any fit.

A zero cell (i, j, k, t) can only enter a PPML fit with exporter^product^year,
importer^product^year and pair^product FEs if pair-product (i, j, k) trades in some year,
exporter i ships k in t and importer j buys k in t; every other zero is separated (its FE
would go to -inf). Counts those cells per year from stage 1 output (an upper bound: the
iterated separation and singleton drops can remove more). Writes <dir>/ppml_sample_count.json.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/baci/ppml_count.py build/baci/hs92
"""
import os, sys, time, json
os.environ.setdefault("POLARS_MAX_THREADS", "8")
import polars as pl
d = sys.argv[1]
f = pl.scan_parquet(f"{d}/flows/*.parquet").select("t", "i", "j", "k")
t0 = time.time()
ijk = f.select("i", "j", "k").unique().collect(engine="streaming")
print("pair-products ever trading", f"{ijk.height:,}", round(time.time()-t0), "s", flush=True)
out = {}
for y in range(1995, 2025):
    fy = pl.scan_parquet(f"{d}/flows/y{y}.parquet").select("i", "j", "k").collect()
    ik = fy.select("i", "k").unique(); jk = fy.select("j", "k").unique()
    c = ijk.join(ik, on=["i", "k"], how="semi").join(jk, on=["j", "k"], how="semi").height
    out[y] = {"positive": fy.height, "ppml_cells": c}
    print(y, f"{fy.height:,}", f"{c:,}", f"{c/fy.height:.2f}x", flush=True)
tp = sum(v["positive"] for v in out.values()); tc = sum(v["ppml_cells"] for v in out.values())
print(f"total positive {tp:,}; ppml cells {tc:,} ({tc/tp:.2f}x); ijk {ijk.height:,}; secs {time.time()-t0:.0f}")
json.dump({"ijk": ijk.height, "years": out, "positive": tp, "ppml_cells": tc}, open(f"{d}/ppml_sample_count.json", "w"), indent=1)
