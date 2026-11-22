# SPDX-License-Identifier: GPL-3.0-or-later
"""D14: Kim, Morse & Zingales (2009, JFE) Table 5 Panel D on OpenAlex: the pooled top-25 effect by decade.

KMZ: productivity_it = a_i + sum_d b_d [top 25 x decade d] + decade + career cubic (+ rank, editor),
economics and finance faculty ever at a top-25 school, 1970-2001. Economics: b = 0.839 / 0.490 / -0.482
(1970s / 80s / 90s), finance 1.030 / 0.444 / -0.276, impact productivity mean 3.1 (Table 3).

Here (choices agreed with Ted 2026-11-22, docs/openalex.md):
  institution  main institution per author-year, rolled up to the top of the OpenAlex lineage; it changes
               only at a counted move: R+P+S (new main institution held the next calendar year, main
               share >= 0.5 both sides); other changes keep the previous institution
  years        the author's active years (any article/review), 1970-2001; decades 1970s, 1980s, 1990s+2000-01
  career       years since the author's first article/review (KMZ: since PhD)
  sample       ever at a top-25 school (KMZ Table 1) in >= 2 active years of 1970-2001, and
                 econ:     modal field 20
                 econbus:  >= 1 article in KMZ's 41 journals 1970-2001 and modal field 20 or 14
  outcome      impact: sum over the year's articles in the 41 journals of IF(journal, decade) / coauthors
               (KMZ multiply by AER-equivalent pages too; OpenAlex lacks last pages for most articles in
               Econometrica, REStat, JASA, RAND, IER, ...), and articles: plain count (KMZ "article counts")
Not available: rank, chaired, visiting, editor; zero-output CV years (only publishing years are seen).

    cd /srv/dfeols && .venv/bin/python -I repo/replication/openalex/kmz.py --out build/openalex/kmz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

c = pl.col
TOP25 = {40347166: "Chicago", 79576946: "Penn", 136199984: "Harvard", 57206974: "NYU", 63966007: "MIT",
         97018004: "Stanford", 111979921: "Northwestern", 161318765: "UCLA", 27837315: "Michigan",
         78577930: "Columbia", 5388228: "Rochester", 95457486: "Berkeley", 32971472: "Yale",
         20089843: "Princeton", 52357470: "Ohio State", 205783295: "Cornell", 135310074: "Wisconsin",
         170897317: "Duke", 141945490: "UBC", 219193219: "Purdue", 201448701: "Washington",
         592451: "Indiana", 86519309: "UT Austin", 74973139: "Carnegie Mellon", 1174212: "USC"}
# KMZ Table 2: impact factor by decade (1970s, 1980s, 1990s); journals founded in the 1980s get the 1980s value
IF = {"AER": (1.00, 1.00, 1.00), "JPE": (.93, .75, .72), "JFE": (.85, 1.04, .88), "REStat": (.74, .43, .51),
      "Econometrica": (.71, .89, .49), "REStud": (.69, .59, .67), "RAND": (.61, .66, .41), "Bell": (.61, .66, .41),
      "JF": (.60, .60, .96), "JME": (.58, .75, .37), "IER": (.49, .27, .33), "QJE": (.43, .62, .80),
      "JASA": (.43, .37, .34), "JET": (.43, .37, .17), "JPubE": (.42, .28, .27), "JMCB": (.40, .39, .32),
      "NTJ": (.40, .16, .28), "JEconometrics": (.35, .29, .24), "JIE": (.33, .43, .35), "EI": (.32, .27, .15),
      "JBus": (.31, .37, .30), "ILRR": (.30, .23, .31), "JHR": (.29, .15, .24), "JUE": (.28, .17, .13),
      "Economica": (.27, .22, .17), "JFQA": (.26, .28, .35), "JLawEcon": (.21, .26, .16), "SEJ": (.20, .16, .08),
      "EJ": (.18, .23, .20), "JLS": (.16, .24, .12), "JEDC": (.14, .15, .15), "OEP": (.13, .12, .15),
      "JEH": (.13, .13, .07), "JRS": (.13, .12, .07), "EER": (.12, .17, .26), "JDE": (.10, .13, .19),
      "EDCC": (.09, .05, .06), "JBES": (.51, .51, .39), "RFS": (.32, .32, .36), "JOLE": (.29, .29, .26),
      "JLEO": (.24, .24, .26), "JIMF": (.26, .26, .17)}
KMZ = {"econ": [0.839, 0.490, -0.482], "finance": [1.030, 0.444, -0.276], "impact_mean": 3.1, "articles_mean": 0.62}
Y0, Y1 = 1970, 2001
DEC = pl.when(c("year") >= 1990).then(1990).otherwise(c("year") // 10 * 10)


def main_inst(df: pl.DataFrame) -> pl.DataFrame:
    d = df.group_by("author_id", "year", "root").agg(c("weight").sum())
    return (d.sort(["author_id", "year", "weight", "root"], descending=[False, False, True, False])
            .group_by("author_id", "year", maintain_order=True)
            .agg(c("root").first().alias("inst"), (c("weight").first() / c("weight").sum()).alias("share")))


def build(a) -> pl.DataFrame:
    stage, pan = a.openalex / "stage", a.openalex / "panel"
    inst = pl.read_parquet(stage / "institutions" / "*.parquet", columns=["institution_id", "lineage"])
    tops = inst.filter(c("lineage").list.len() == 1).select(c("institution_id").alias("lineage"))
    root = (inst.explode("lineage").join(tops, on="lineage", how="semi").sort("institution_id", "lineage")
            .group_by("institution_id").agg(c("lineage").first().alias("root")))
    jb = pl.read_parquet(a.openalex / "kmz" / "journals.parquet")
    jauth = pl.read_parquet(a.openalex / "kmz" / "authors.parquet").select("author_id")

    rows = []
    for b in range(64):
        p = pl.read_parquet(pan / "panel" / f"b={b:02d}.parquet", columns=["author_id", "year", "field_id"])
        fld = p.unique("author_id").select("author_id", "field_id")
        econ = fld.filter(c("field_id") == 20).select("author_id")
        eb = fld.filter(c("field_id").is_in([20, 14])).join(jauth, on="author_id", how="semi").select("author_id")
        u = pl.concat([econ, eb]).unique()
        start = p.join(u, on="author_id", how="semi").group_by("author_id").agg(c("year").min().alias("first_year"))
        yrs = p.join(u, on="author_id", how="semi").filter(c("year").is_between(Y0 - 15, Y1 + 5)).select("author_id", "year")
        ayi = (pl.scan_parquet(pan / "a" / "ayi" / f"b={b:02d}" / "*.parquet")
               .filter(c("year").is_between(Y0 - 15, Y1 + 5)).join(u.lazy(), on="author_id", how="semi")
               .collect().join(root, on="institution_id", how="left").with_columns(c("root").fill_null(c("institution_id"))))
        m = main_inst(ayi).sort("author_id", "year")
        o = "author_id"
        m = m.with_columns(c("inst").shift(1).over(o).alias("p_inst"), c("share").shift(1).over(o).alias("p_share"),
                           c("inst").shift(-1).over(o).alias("n_inst"), c("year").shift(-1).over(o).alias("n_year"),
                           pl.int_range(pl.len()).over(o).alias("k"))
        move = ((c("inst") != c("p_inst")) & (c("n_year") == c("year") + 1) & (c("n_inst") == c("inst"))
                & (c("share") >= 0.5) & (c("p_share") >= 0.5)).fill_null(False)
        m = m.with_columns(move.alias("move"), pl.when((c("k") == 0) | move).then(c("inst")).alias("anchor"))
        s = (yrs.join(m.select("author_id", "year", "anchor", "move"), on=["author_id", "year"], how="left")
             .sort("author_id", "year").with_columns(c("anchor").forward_fill().over(o).alias("inst"))
             .filter(c("year").is_between(Y0, Y1) & c("inst").is_not_null())
             .join(start, on="author_id").join(fld, on="author_id")
             .with_columns(c("move").fill_null(False),
                           c("author_id").is_in(econ["author_id"].implode()).alias("in_econ"),
                           c("author_id").is_in(eb["author_id"].implode()).alias("in_econbus")))
        rows.append(s)
    s = pl.concat(rows)
    s = s.with_columns(c("inst").is_in(list(TOP25)).alias("top25"), DEC.alias("decade"))
    s = s.join(s.group_by("author_id").agg(c("top25").sum().alias("top_years")), on="author_id").filter(c("top_years") >= 2)

    # outcomes from the 41 journals
    ifd = pl.DataFrame([{"abbr": k, "decade": d, "if": v[i]} for k, v in IF.items() for i, d in enumerate((1970, 1980, 1990))])
    w = (pl.scan_parquet(stage / "works" / "*.parquet").join(jb.lazy(), on="source_id", how="inner")
         .filter(c("type").is_in(["article", "review"]) & c("year").is_between(Y0, Y1)
                 & ~c("is_paratext").fill_null(False) & ~c("is_retracted").fill_null(False))
         .select("work_id", "abbr", "authors_count").collect(engine="streaming"))
    au = s.select("author_id").unique()
    aw = (pl.scan_parquet(stage / "authorships" / "*.parquet").select("work_id", "author_id", "year")
          .join(w.lazy(), on="work_id", how="inner").join(au.lazy(), on="author_id", how="semi")
          .unique(["work_id", "author_id"]).collect(engine="streaming")
          .with_columns(DEC.alias("decade")).join(ifd, on=["abbr", "decade"], how="left"))
    y = aw.group_by("author_id", "year").agg(
        (c("if") / c("authors_count").clip(lower_bound=1)).sum().alias("impact"), pl.len().cast(pl.Float64).alias("articles"))
    s = s.join(y, on=["author_id", "year"], how="left").with_columns(c("impact").fill_null(0.0), c("articles").fill_null(0.0))
    cy = (c("year") - c("first_year")).cast(pl.Float64)
    s = s.with_columns(cy.alias("cy"), (cy**2 / 100).alias("cy2"), (cy**3 / 1000).alias("cy3"),
                       *[(c("top25") & (c("decade") == d)).cast(pl.Float64).alias(f"t{d % 100:02d}") for d in (1970, 1980, 1990)])
    return s.drop("anchor")


def estimate(s: pl.DataFrame, work: Path, label: str) -> dict:
    import pyfixest as pf

    import dfeols

    xs = "t70 + t80 + t90 + cy + cy2 + cy3"
    out = {"label": label, "N": s.height, "authors": s["author_id"].n_unique(), "moves": int(s["move"].sum()),
           "means": {k: float(s[k].mean()) for k in ("impact", "articles")}}
    work.mkdir(parents=True, exist_ok=True)
    s.select("author_id", "decade", "impact", "articles", "t70", "t80", "t90", "cy", "cy2", "cy3").write_parquet(work / "data.parquet")
    ds = dfeols.prep(str(work / "data.parquet"), shard_dir=work / "shards", partition="author_id",
                     fixed_effects=["author_id", "decade"], clusters=["author_id"], target_rows=50_000, overwrite=True)
    pdf = s.select("author_id", "decade", "impact", "articles", "t70", "t80", "t90", "cy", "cy2", "cy3").to_pandas()
    for yv in ("impact", "articles"):
        f = f"{yv} ~ {xs} | author_id + decade"
        t0 = time.time()
        r = dfeols.fit(ds, f, cluster="author_id", workdir=str(work / f"fit_{yv}"), log_every=1e9, record=False)
        td = r.tidy(None, "CRV1")
        sec = round(time.time() - t0, 1)
        mp = pf.feols(f, data=pdf, vcov={"CRV1": "author_id"}, fixef_tol=1e-12)
        tp = mp.tidy()
        rec = {}
        for v in ("t70", "t80", "t90"):
            rec[v] = {"b": float(td.loc[v, "Estimate"]), "se": float(td.loc[v, "Std. Error"]),
                      "b_pyfixest": float(tp.loc[v, "Estimate"]), "se_pyfixest": float(tp.loc[v, "Std. Error"]),
                      "b_over_mean": float(td.loc[v, "Estimate"]) / out["means"][yv]}
        out[yv] = {"coef": rec, "N_dfeols": int(r.N), "N_pyfixest": int(mp._N), "sec_dfeols": sec}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openalex", type=Path, default=Path("build/openalex"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s = build(a)
    s.write_parquet(a.out / "kmz_panel.parquet")
    print(f"panel {s.height:,} rows, {s['author_id'].n_unique():,} authors, {time.time() - t0:.0f}s", flush=True)
    samples = {
        "econ": s.filter(c("in_econ")),
        "econbus": s.filter(c("in_econbus")),
        "econbus_f20": s.filter(c("in_econbus") & (c("field_id") == 20)),
        "econbus_f14": s.filter(c("in_econbus") & (c("field_id") == 14)),
    }
    res = {"kmz": KMZ, "kmz_b_over_mean": {k: [b / KMZ["impact_mean"] for b in v] for k, v in KMZ.items() if k in ("econ", "finance")},
           "results": []}
    for k, d in samples.items():
        r = estimate(d, a.out / "work" / k, k)
        res["results"].append(r)
        print(json.dumps({"sample": k, "N": r["N"], "authors": r["authors"], "moves": r["moves"],
                          **{yv: {v: round(r[yv]["coef"][v]["b"], 4) for v in ("t70", "t80", "t90")} for yv in ("impact", "articles")}}), flush=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
