# SPDX-License-Identifier: GPL-3.0-or-later
"""D13 stage 1: BACI HS92 (CEPII, release V202601) and CEPII Gravity -> Parquet.

Writes to --out (build/baci/hs92):

  flows/y<year>.parquet   t (Int16), i, j (Int16, BACI country codes), k (Int32, HS92 6-digit
                          code as a number; the CSV's leading zero is restored by zero-padding
                          to 6), v (Float64, thousand USD), q (Float64, metric tons; null when
                          BACI leaves it empty), sorted by i, j, k. One file a year, 1995-2024.
  countries.parquet       BACI's country list plus `iso3_grav`, the ISO3 code used to join
                          Gravity: the BACI iso3, except 490 "Other Asia, nes" -> TWN (BACI's
                          Taiwan, as in CEPII's own Gravity tradeflow_baci).
  products.parquet        k (Int32), hs6 (6-character code), hs2, hs4, description.
  gravity_pairs.parquet   Gravity V202211 pair x year rows for 1995 on, keyed by BACI codes
                          (i, j, t), for every exporter-importer pair of BACI countries Gravity
                          has: dist, distw_harmonic, contig, comlang_off, comcol, col45,
                          wto_o/d, eu_o/d, fta_wto, rta_type, gdp_o/d, pop_o/d. Gravity ends
                          in 2021; agreements for the models come from Larch's RTA database
                          (docs/sources/baci.md item 8), so only the time-invariant pair
                          variables are meant to be used past 2021.
  summary.json            per year: rows, distinct exporters / importers / products / pairs,
                          duplicate (i, j, k) keys (must be 0), v <= 0, q null; and the BACI
                          codes that have no Gravity match.

Each year is read from the zip into memory (the largest CSV is 0.34 GB) by a spawn pool
worker; years already written are skipped unless --force.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/baci/stage.py \
        --zip raw/baci/BACI_HS92_V202601.zip --gravity raw/cepii_gravity/Gravity_csv_V202211.zip \
        --out build/baci/hs92 --workers 6
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

RELEASE = "V202601"
GRAVITY_COLS = ["dist", "distw_harmonic", "contig", "comlang_off", "comcol", "col45", "wto_o", "wto_d",
                "eu_o", "eu_d", "fta_wto", "rta_type", "gdp_o", "gdp_d", "pop_o", "pop_d"]
ISO3_FIX = {490: "TWN"}


def year_files(zpath: Path) -> dict[int, str]:
    with zipfile.ZipFile(zpath) as z:
        out = {}
        for n in z.namelist():
            m = re.fullmatch(rf"BACI_HS92_Y(\d{{4}})_{RELEASE}\.csv", n)
            if m:
                out[int(m.group(1))] = n
    return dict(sorted(out.items()))


def stage_year(zpath: str, name: str, year: int, out: str, threads: int) -> dict:
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    import polars as pl

    t0 = time.time()
    with zipfile.ZipFile(zpath) as z:
        raw = z.read(name)
    df = pl.read_csv(io.BytesIO(raw), schema={"t": pl.Int16, "i": pl.Int16, "j": pl.Int16, "k": pl.Int32,
                                              "v": pl.Float64, "q": pl.Float64},
                     null_values=["", "NA"])
    del raw
    df = df.sort(["i", "j", "k"])
    bad_t = df.filter(pl.col("t") != year).height
    s = df.select(
        rows=pl.len(),
        exporters=pl.col("i").n_unique(),
        importers=pl.col("j").n_unique(),
        products=pl.col("k").n_unique(),
        pairs=pl.struct("i", "j").n_unique(),
        dup_keys=pl.len() - pl.struct("i", "j", "k").n_unique(),
        v_nonpos=(pl.col("v") <= 0).sum(),
        v_null=pl.col("v").null_count(),
        q_null=pl.col("q").null_count(),
        k_min=pl.col("k").min(),
        k_max=pl.col("k").max(),
        v_total_bn=(pl.col("v").sum() / 1e6),
    ).row(0, named=True)
    s["t_mismatch"] = bad_t
    path = Path(out) / "flows" / f"y{year}.parquet"
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp, compression="zstd", statistics=True)
    tmp.rename(path)
    s["sec"] = round(time.time() - t0, 1)
    s["year"] = year
    return s


def lookups(zpath: Path, out: Path) -> "pl.DataFrame":
    import polars as pl

    with zipfile.ZipFile(zpath) as z:
        c = pl.read_csv(io.BytesIO(z.read(f"country_codes_{RELEASE}.csv")), infer_schema_length=0)
        p = pl.read_csv(io.BytesIO(z.read(f"product_codes_HS92_{RELEASE}.csv")), infer_schema_length=0)
    c = c.with_columns(pl.col("country_code").cast(pl.Int16))
    c = c.with_columns(pl.col("country_code").replace_strict(ISO3_FIX, default=pl.col("country_iso3"),
                                                             return_dtype=pl.Utf8).alias("iso3_grav"))
    c.write_parquet(out / "countries.parquet")
    p = p.select(k=pl.col("code").cast(pl.Int32, strict=False), hs6=pl.col("code").str.zfill(6), description=pl.col("description"))
    p = p.with_columns(hs2=pl.col("hs6").str.slice(0, 2), hs4=pl.col("hs6").str.slice(0, 4))
    p.write_parquet(out / "products.parquet")
    return c


def gravity(gzip_path: Path, countries, out: Path, tmpdir: Path) -> dict:
    import polars as pl

    csv = tmpdir / "Gravity_V202211.csv"
    if not csv.exists():
        with zipfile.ZipFile(gzip_path) as z, open(csv, "wb") as f:
            with z.open("Gravity_V202211.csv") as src:
                while b := src.read(1 << 24):
                    f.write(b)
    g = pl.scan_csv(csv, infer_schema_length=0).select(["year", "iso3_o", "iso3_d", *GRAVITY_COLS])
    g = g.with_columns(pl.col("year").cast(pl.Int16), *[pl.col(c).cast(pl.Float64) for c in GRAVITY_COLS])
    g = g.filter(pl.col("year") >= 1995)
    cc = countries.lazy().select(code=pl.col("country_code"), iso3=pl.col("iso3_grav"))
    g = (g.join(cc.rename({"code": "i", "iso3": "iso3_o"}), on="iso3_o", how="inner")
          .join(cc.rename({"code": "j", "iso3": "iso3_d"}), on="iso3_d", how="inner")
          .rename({"year": "t"})
          .select(["i", "j", "t", "iso3_o", "iso3_d", *GRAVITY_COLS])
          .sort(["i", "j", "t"]))
    g.sink_parquet(out / "gravity_pairs.parquet")
    gp = pl.scan_parquet(out / "gravity_pairs.parquet")
    have = set(gp.select(pl.col("i").unique()).collect(engine="streaming")["i"].to_list())
    years = gp.select(pl.col("t").min().alias("lo"), pl.col("t").max().alias("hi"), pl.len().alias("rows")).collect().row(0, named=True)
    missing = countries.filter(~pl.col("country_code").is_in(list(have))).select("country_code", "country_name", "iso3_grav")
    csv.unlink()
    return {"rows": years["rows"], "years": [years["lo"], years["hi"]],
            "baci_codes_without_gravity": missing.to_dicts()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--gravity", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-gravity", action="store_true")
    a = ap.parse_args()
    (a.out / "flows").mkdir(parents=True, exist_ok=True)
    summf = a.out / "summary.json"
    summ = json.loads(summf.read_text()) if summf.exists() else {"release": RELEASE, "years": {}}
    files = year_files(a.zip)
    todo = {y: n for y, n in files.items()
            if (not a.years or y in a.years) and (a.force or not (a.out / "flows" / f"y{y}.parquet").exists())}
    print(f"{len(files)} years in {a.zip.name} ({min(files)}-{max(files)}); {len(todo)} to stage", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
        futs = {ex.submit(stage_year, str(a.zip), n, y, str(a.out), a.threads): y
                for y, n in sorted(todo.items(), key=lambda kv: -kv[0])}
        for f in as_completed(futs):
            s = f.result()
            summ["years"][str(s["year"])] = s
            summf.write_text(json.dumps(summ, indent=1))
            print(f"{time.strftime('%H:%M:%S')} {s['year']}: {s['rows']:,} rows, {s['products']} products, "
                  f"{s['pairs']:,} pairs, dup {s['dup_keys']}, v<=0 {s['v_nonpos']}, q null {s['q_null']:,}, {s['sec']} s",
                  flush=True)
    countries = lookups(a.zip, a.out)
    if not a.skip_gravity:
        tmpdir = Path(os.environ.get("POLARS_TEMP_DIR", a.out))
        summ["gravity"] = gravity(a.gravity, countries, a.out, tmpdir)
        print(f"gravity: {summ['gravity']['rows']:,} pair-years {summ['gravity']['years']}; BACI codes without "
              f"Gravity: {[(d['country_code'], d['country_name']) for d in summ['gravity']['baci_codes_without_gravity']]}",
              flush=True)
    ys = summ["years"]
    summ["total_rows"] = sum(v["rows"] for v in ys.values())
    summ["problems"] = {y: {k: v[k] for k in ("dup_keys", "v_nonpos", "v_null", "t_mismatch") if v[k]}
                        for y, v in ys.items() if any(v[k] for k in ("dup_keys", "v_nonpos", "v_null", "t_mismatch"))}
    summf.write_text(json.dumps(summ, indent=1))
    print(f"done in {time.time() - t0:.0f} s: {summ['total_rows']:,} rows over {len(ys)} years; problems: {summ['problems'] or 'none'}",
          flush=True)


if __name__ == "__main__":
    main()
