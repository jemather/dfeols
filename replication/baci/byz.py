# SPDX-License-Identifier: GPL-3.0-or-later
"""D13: product-level structural gravity panel (Baier, Yotov & Zylkin 2019 at HS6), from stage 1.

Rows are exporter i x importer j x HS92 product k x year t, 1995-2024: every positive BACI flow,
plus the zero cells a PPML fit with exporter^product^year, importer^product^year and
pair^product FEs can use (the pair-product trades in some year, i exports k in t, j imports k in
t; any other zero is separated by construction; see ppml_count.py). Agreement dummies from Mario
Larch's RTA database (release 2026-01-11, `raw/rta_larch/rta_20260111_csv.zip`) by
(exporter, importer, year).

BACI codes with no RTA data are dropped: 697 "Europe EFTA, nes" and 711 "Southern African
Customs Union (...1999)" (SACU's members report jointly before 2000). Their counts are in
summary.json.

Columns: t Int16, i Int16, j Int16 (BACI country codes), k Int32 (HS92 code), pair Int32
(i * 1000 + j), v Float64 (thousand USD, 0 for a zero cell), lv Float64 (ln v, null when v = 0),
q Float64 (tons, null when zero or missing), pos Int8 (v > 0), rta, cu, fta, psa, eia, cueia,
ftaeia, psaeia Int8.

Every FE contains k, so the model is block-diagonal in products: the FE problem splits into one
independent problem per product, coupled only through the common rta coefficient. Parts are
contiguous k ranges, sorted by k, i, j, t. The reference sample R0 is the products with
(k * 2654435761) mod 2^32 mod 50 == 0, a 2% subset that is an exact sub-model.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/baci/byz.py \
        --stage build/baci/hs92 --rta raw/rta_larch/rta_20260111_csv.zip \
        --work build/baci/byz --out outbox/baci_byz_v1
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from pathlib import Path

import polars as pl

DROP = [697, 711]
RTA_COLS = ["rta", "cu", "fta", "psa", "eia", "cueia", "ftaeia", "psaeia"]
YEARS = range(1995, 2025)
TARGET_PART_BYTES = 0.9e9


def r0_expr() -> pl.Expr:
    return ((pl.col("k").cast(pl.UInt64) * 2654435761) % (1 << 32)) % 50 == 0


def rta_table(path: Path, countries: pl.DataFrame) -> pl.DataFrame:
    with zipfile.ZipFile(path) as z:
        r = pl.read_csv(io.BytesIO(z.read("rta_20260111.csv")))
    cc = countries.select(code=pl.col("country_code"), iso3=pl.col("iso3_grav"))
    r = (r.filter(pl.col("year").is_between(min(YEARS), max(YEARS)))
          .join(cc.rename({"code": "i", "iso3": "exporter"}), on="exporter")
          .join(cc.rename({"code": "j", "iso3": "importer"}), on="importer")
          .select(t=pl.col("year").cast(pl.Int16), i="i", j="j", *[pl.col(c).cast(pl.Int8) for c in RTA_COLS]))
    assert r.select(pl.struct("i", "j", "t").n_unique()).item() == r.height, "duplicate RTA keys"
    return r


def build_rect(stage: Path, rta: pl.DataFrame, work: Path) -> dict:
    flows = lambda y: pl.scan_parquet(stage / "flows" / f"y{y}.parquet")
    keep = ~pl.col("i").is_in(DROP) & ~pl.col("j").is_in(DROP)
    allf = pl.scan_parquet(stage / "flows" / "*.parquet")
    dropped = allf.filter(~keep).select(rows=pl.len(), v_bn=pl.col("v").sum() / 1e6).collect(engine="streaming").row(0, named=True)
    domestic = allf.filter(pl.col("i") == pl.col("j")).select(pl.len()).collect(engine="streaming").item()
    ijk = allf.filter(keep).select("i", "j", "k").unique().collect(engine="streaming")
    (work / "rect").mkdir(parents=True, exist_ok=True)
    years = {}
    for y in YEARS:
        p = work / "rect" / f"y{y}.parquet"
        fy = flows(y).filter(keep).collect()
        if p.exists():
            years[y] = json.loads((work / "rect" / f"y{y}.json").read_text())
            continue
        t0 = time.time()
        cells = (ijk.join(fy.select("i", "k").unique(), on=["i", "k"], how="semi")
                    .join(fy.select("j", "k").unique(), on=["j", "k"], how="semi"))
        d = (cells.join(fy.select("i", "j", "k", "v", "q"), on=["i", "j", "k"], how="left")
                  .with_columns(t=pl.lit(y, pl.Int16))
                  .join(rta.filter(pl.col("t") == y), on=["t", "i", "j"], how="left"))
        miss = d.filter(pl.col("rta").is_null()).select("i", "j").unique()
        assert miss.height == 0, f"{y}: pairs without RTA data {miss.head(10).to_dicts()}"
        d = d.with_columns(
            v=pl.col("v").fill_null(0.0),
            pos=pl.col("v").is_not_null().cast(pl.Int8),
            pair=(pl.col("i").cast(pl.Int32) * 1000 + pl.col("j").cast(pl.Int32)),
        ).with_columns(lv=pl.when(pl.col("v") > 0).then(pl.col("v").log()))
        d = d.select("t", "i", "j", "k", "pair", "v", "lv", "q", "pos", *RTA_COLS).sort("k", "i", "j")
        assert d.height == cells.height and d.filter(pl.col("pos") == 1).height == fy.height
        s = {"rows": d.height, "positive": fy.height, "rta_share": round(d["rta"].mean(), 4),
             "rta_share_positive": round(d.filter(pl.col("pos") == 1)["rta"].mean(), 4), "sec": round(time.time() - t0, 1)}
        d.write_parquet(p, compression="zstd", statistics=True)
        (work / "rect" / f"y{y}.json").write_text(json.dumps(s))
        years[y] = s
        print(f"{time.strftime('%H:%M:%S')} {y}: {s['rows']:,} cells, {s['positive']:,} positive, rta share {s['rta_share']}, {s['sec']} s", flush=True)
    return {"years": years, "dropped_rows": dropped["rows"], "dropped_v_bn": round(dropped["v_bn"], 1),
            "domestic_rows": domestic, "pair_products": ijk.height}


def package(work: Path, out: Path) -> list[dict]:
    rect = pl.scan_parquet(work / "rect" / "*.parquet")
    by_k = rect.group_by("k").agg(rows=pl.len()).sort("k").collect(engine="streaming")
    total = by_k["rows"].sum()
    probe = pl.read_parquet(work / "rect" / "y2010.parquet")
    bpr = (work / "rect" / "y2010.parquet").stat().st_size / probe.height
    nparts = max(1, round(total * bpr / TARGET_PART_BYTES))
    target = total / nparts
    cum = by_k["rows"].cum_sum()
    bounds, lo = [], by_k["k"][0]
    for n in range(1, nparts):
        idx = int((cum >= n * target).arg_max())
        bounds.append((lo, by_k["k"][idx]))
        lo = by_k["k"][idx + 1]
    bounds.append((lo, by_k["k"][-1]))
    (out / "data").mkdir(parents=True, exist_ok=True)
    parts = []
    for n, (a, b) in enumerate(bounds):
        f = out / "data" / f"part-{n:04d}.parquet"
        t0 = time.time()
        d = rect.filter(pl.col("k").is_between(a, b)).collect(engine="streaming").sort("k", "i", "j", "t")
        d.write_parquet(f, compression="zstd", statistics=True, row_group_size=1 << 20)
        parts.append({"file": f"data/{f.name}", "rows": d.height, "positive": int(d["pos"].sum()),
                      "k": [int(a), int(b)], "products": d["k"].n_unique(), "bytes": f.stat().st_size})
        print(f"{time.strftime('%H:%M:%S')} {f.name}: {d.height:,} rows, k {a}-{b}, {f.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)
        del d
    ref = out / "ref" / "data"
    ref.mkdir(parents=True, exist_ok=True)
    r0 = rect.filter(r0_expr()).collect(engine="streaming").sort("k", "i", "j", "t")
    r0.write_parquet(ref / "part-0000.parquet", compression="zstd", statistics=True)
    print(f"R0: {r0.height:,} rows, {r0['k'].n_unique()} products, {int(r0['pos'].sum()):,} positive", flush=True)
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--rta", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rect-only", action="store_true")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    countries = pl.read_parquet(a.stage / "countries.parquet")
    t0 = time.time()
    rta = rta_table(a.rta, countries)
    s = build_rect(a.stage, rta, a.work)
    s["rows"] = sum(v["rows"] for v in s["years"].values())
    s["positive"] = sum(v["positive"] for v in s["years"].values())
    print(f"rectangle: {s['rows']:,} cells, {s['positive']:,} positive; dropped {s['dropped_rows']:,} flows "
          f"(codes {DROP}, {s['dropped_v_bn']} bn USD); domestic rows {s['domestic_rows']}; {time.time() - t0:.0f} s", flush=True)
    if not a.rect_only:
        s["parts"] = package(a.work, a.out)
    (a.work / "summary.json").write_text(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    main()
