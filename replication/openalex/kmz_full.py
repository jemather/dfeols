# SPDX-License-Identifier: GPL-3.0-or-later
"""D14: KMZ (2009) Table 5 Panel D specification on the full panel (`full.py` variants affil, edu, lag2).

y_it = a_i + decade + sum_d b_d [top 25 x decade d] + career cubic, CRV1 by author, dfeols and pyfixest.
Samples (economics = modal field 20; "ever top 25" = 2+ active years at a KMZ top-25 school in the window;
core = career span 5+ years and 3+ 3-page articles in the 41 journals):
  kmz_core        ever top 25 in 1970-2001, core, years 1970-2001, KMZ decades (1990s = 1990-2001)
  kmz_all         ever top 25 in 1970-2001, years 1970-2001
  ext_core        ever top 25 in 1970-2025, core, years 1970-2025, calendar decades 1970s-2020s
  ext_all_econ    every economics author, 1970-2025 (KMZ's 'others' then include never-top-25 authors)
Outcomes impact_pages, articles3 (41 journals) and frac_works (all journals).

    cd /srv/dfeols && .venv/bin/python -I repo/replication/openalex/kmz_full.py --out build/openalex/kmz_full
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

c = pl.col
KMZ = {"econ": [0.839, 0.490, -0.482], "finance": [1.030, 0.444, -0.276]}


def load(full: Path, variant: str, fields: list[int]) -> pl.DataFrame:
    return (pl.scan_parquet(full / variant / "*.parquet").filter(c("field_id").is_in(fields) & c("year").is_between(1970, 2025))
            .select("author_id", "year", "inst", "move", "filled", "first_year", "last_year", "top25",
                    "impact_pages", "articles3", "frac_works").collect(engine="streaming"))


def sample(s: pl.DataFrame, y1: int, ever: bool, core: bool, kmz_dec: bool) -> pl.DataFrame:
    w = s.filter(c("year") <= y1)
    if ever:
        ev = w.filter(c("top25") & ~c("filled")).group_by("author_id").len().filter(c("len") >= 2).select("author_id")
        w = w.join(ev, on="author_id", how="semi")
    if core:
        k = (w.filter(c("year") <= 2001).group_by("author_id").agg(c("articles3").sum().alias("k"),
                                                                    (c("last_year") - c("first_year")).first().alias("span"))
             .filter((c("k") >= 3) & (c("span") >= 5)).select("author_id"))
        w = w.join(k, on="author_id", how="semi")
    dec = (pl.when(c("year") >= 1990).then(1990).otherwise(c("year") // 10 * 10) if kmz_dec else c("year") // 10 * 10)
    w = w.with_columns(dec.cast(pl.Int16).alias("decade"))
    decs = sorted(w["decade"].unique().to_list())
    cy = (c("year") - c("first_year")).cast(pl.Float64)
    return w.with_columns(cy.alias("cy"), (cy**2 / 100).alias("cy2"), (cy**3 / 1000).alias("cy3"),
                          *[(c("top25") & (c("decade") == d)).cast(pl.Float64).alias(f"t{d}") for d in decs])


def fit(w: pl.DataFrame, work: Path, label: str, pyfixest: bool) -> dict:
    import pyfixest as pf

    import dfeols

    ts = [x for x in w.columns if x.startswith("t") and x[1:].isdigit()]
    ys = ("impact_pages", "articles3", "frac_works")
    cols = ["author_id", "decade", *ys, *ts, "cy", "cy2", "cy3"]
    work.mkdir(parents=True, exist_ok=True)
    w.select(cols).write_parquet(work / "data.parquet")
    ds = dfeols.prep(str(work / "data.parquet"), shard_dir=work / "shards", partition="author_id",
                     fixed_effects=["author_id", "decade"], clusters=["author_id"], target_rows=500_000, overwrite=True)
    out = {"label": label, "N": w.height, "authors": w["author_id"].n_unique(), "moves": int(w["move"].sum()),
           "filled_share": float(w["filled"].mean()), "means": {y: float(w[y].mean()) for y in ys},
           "means_by_decade": (w.group_by("decade", "top25").agg(c("impact_pages").mean(), c("articles3").mean(), pl.len())
                               .sort("decade", "top25").to_dicts())}
    pdf = w.select(cols).to_pandas() if pyfixest else None
    for y in ys:
        f = f"{y} ~ {' + '.join(ts)} + cy + cy2 + cy3 | author_id + decade"
        t0 = time.time()
        r = dfeols.fit(ds, f, cluster="author_id", workdir=str(work / f"fit_{y}"), log_every=1e9, record=False)
        td = r.tidy(None, "CRV1")
        rec = {"N": int(r.N), "sec": round(time.time() - t0, 1), "coef": {}}
        if pyfixest:
            tp = pf.feols(f, data=pdf, vcov={"CRV1": "author_id"}, fixef_tol=1e-12).tidy()
        for t in ts:
            e = {"b": float(td.loc[t, "Estimate"]), "se": float(td.loc[t, "Std. Error"])}
            if pyfixest:
                e["rel_b_pyfixest"] = abs(e["b"] - float(tp.loc[t, "Estimate"])) / max(abs(float(tp.loc[t, "Estimate"])), 1e-12)
                e["rel_se_pyfixest"] = abs(e["se"] - float(tp.loc[t, "Std. Error"])) / float(tp.loc[t, "Std. Error"])
            rec["coef"][t] = e
        out[y] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=Path, default=Path("build/openalex/full"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", nargs="+", default=["affil", "edu", "lag2"])
    ap.add_argument("--pyfixest-max-rows", type=int, default=20_000_000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = {"kmz": KMZ, "results": []}
    specs = {"kmz_core": (2001, True, True, True), "kmz_all": (2001, True, False, True),
             "ext_core": (2025, True, True, False), "ext_all_econ": (2025, False, False, False)}
    for v in a.variant:
        s = load(a.full, v, [20])
        for k, sp in specs.items():
            w = sample(s, *sp)
            r = fit(w, a.out / "work" / v / k, k, w.height <= a.pyfixest_max_rows)
            r["variant"] = v
            res["results"].append(r)
            print(json.dumps({"variant": v, "sample": k, "N": r["N"], "authors": r["authors"],
                              "impact_pages": {t: round(e["b"], 3) for t, e in r["impact_pages"]["coef"].items()}}), flush=True)
            (a.out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
