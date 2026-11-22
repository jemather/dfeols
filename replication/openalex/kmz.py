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
  --fill       years with no papers between two active years, as a stand-in for KMZ's zero-output CV
               years (Ted 2026-11-22): none (default), or filled with impact = articles = 0 and the
               institution carried forward; when the institution changes across the gap, the move is put
               at the gap's midpoint (midpoint) or at a uniformly drawn year (random, --seed). Years before
               the first or after the last active year are not filled. The sample (>= 2 top-25 years) is
               always judged on active years, so the authors are the same under every --fill.
  --affil-fallback  author-years with no institution on any paper take OpenAlex's author-level
               affiliation years (author_affil), weight 1 per listed institution
  --lag L      output is credited L years before publication (KMZ: Ellison's submission-acceptance lag
               + 1 year, about 2-3 years); default 0
  pages        an article's length is last - first + 1 when OpenAlex has a last page, else the next
               distinct first page in the same journal and year minus its own first page (pages run on
               within a volume); the year's last article gets the journal-decade median. Articles under
               3 pages are dropped (KMZ do too). Outcomes impact_pages (pages x IF / coauthors, KMZ's unit
               without the AER-equivalent font adjustment) and articles3 (count of 3+ page articles)
  core samples econ_core / econbus_core: also a career span (first to last active year) of 5+ years and
               3+ articles in the 41 journals 1970-2001, a stand-in for KMZ's faculty-only sample
  --prefer-edu the main institution is chosen among education-type institutions when the author-year has
               one (a CV's primary appointment, not the Fed, NBER or World Bank listed alongside)
  aliases      ALIAS maps system-office roots to the KMZ school (always on, from 2026-11-22)
Not available: rank, chaired, visiting, editor.

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
# OpenAlex roots that stand for a KMZ school (system offices): Wisconsin System -> Madison, UT System -> Austin
ALIAS = {1304256225: 135310074, 16452829: 86519309}
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


def main_inst(df: pl.DataFrame, prefer_edu: bool = False) -> pl.DataFrame:
    d = df.group_by("author_id", "year", "root").agg(c("weight").sum(), c("edu").max())
    if prefer_edu:
        d = d.filter(c("edu") | ~c("edu").any().over("author_id", "year"))
    return (d.sort(["author_id", "year", "weight", "root"], descending=[False, False, True, False])
            .group_by("author_id", "year", maintain_order=True)
            .agg(c("root").first().alias("inst"), (c("weight").first() / c("weight").sum()).alias("share")))


def fill_gaps(s: pl.DataFrame, how: str, seed: int) -> pl.DataFrame:
    """Add the years between consecutive active years (s: author_id, year, inst, move, filled)."""
    import numpy as np

    o = "author_id"
    g = (s.sort(o, "year").with_columns(c("year").shift(-1).over(o).alias("ny"), c("inst").shift(-1).over(o).alias("ni"))
         .filter(c("ny") > c("year") + 1))
    if how == "midpoint":  # the move year T runs over year+1 .. ny; T = ny leaves the move where it was
        g = g.with_columns((c("year") + 1 + (c("ny") - c("year") - 1) // 2).alias("T"))
    else:
        u = np.random.default_rng(seed).random(g.height)
        g = g.with_columns((c("year") + 1 + (pl.Series(u) * (c("ny") - c("year")).cast(pl.Float64)).floor().cast(pl.Int16)).alias("T"))
    chg = c("ni") != c("inst")
    early = g.select(o, c("ny").alias("year"), (chg & (c("T") < c("ny"))).alias("early"))
    gy = (g.with_columns(pl.int_ranges(c("year") + 1, c("ny"), dtype=pl.Int16).alias("gy")).explode("gy")
          .select(o, c("gy").alias("year"), pl.when(chg & (c("gy") >= c("T"))).then(c("ni")).otherwise(c("inst")).alias("inst"),
                  (chg & (c("gy") == c("T"))).alias("move"), pl.lit(True).alias("filled")))
    s = (s.join(early, on=[o, "year"], how="left")
         .with_columns((c("move") & ~c("early").fill_null(False)).alias("move")).drop("early"))
    return pl.concat([s, gy.select(s.columns)]).sort(o, "year")


def build(a) -> pl.DataFrame:
    stage, pan = a.openalex / "stage", a.openalex / "panel"
    inst = pl.read_parquet(stage / "institutions" / "*.parquet", columns=["institution_id", "lineage", "type"])
    edu = inst.select(c("institution_id").alias("root"), (c("type") == "education").alias("edu"))
    tops = inst.filter(c("lineage").list.len() == 1).select(c("institution_id").alias("lineage"))
    root = (inst.select("institution_id", "lineage").explode("lineage").join(tops, on="lineage", how="semi").sort("institution_id", "lineage")
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
        start = p.join(u, on="author_id", how="semi").group_by("author_id").agg(c("year").min().alias("first_year"),
                                                                                  c("year").max().alias("last_year"))
        yrs = p.join(u, on="author_id", how="semi").filter(c("year").is_between(Y0 - 15, Y1 + 5)).select("author_id", "year")
        ayi = (pl.scan_parquet(pan / "a" / "ayi" / f"b={b:02d}" / "*.parquet")
               .filter(c("year").is_between(Y0 - 15, Y1 + 5)).join(u.lazy(), on="author_id", how="semi")
               .collect())
        if a.affil_fallback:
            fb = (pl.scan_parquet(stage / "author_affil" / "*.parquet")
                  .filter(c("year").is_between(Y0 - 15, Y1 + 5) & c("institution_id").is_not_null())
                  .join(u.lazy(), on="author_id", how="semi").unique().collect()
                  .join(ayi.select("author_id", "year").unique(), on=["author_id", "year"], how="anti")
                  .with_columns(pl.lit(1.0).alias("weight")))
            ayi = pl.concat([ayi, fb.select(ayi.columns)], how="vertical_relaxed")
        ayi = (ayi.join(root, on="institution_id", how="left").with_columns(c("root").fill_null(c("institution_id")))
               .with_columns(c("root").replace(ALIAS)).join(edu, on="root", how="left").with_columns(c("edu").fill_null(False)))
        m = main_inst(ayi, a.prefer_edu).sort("author_id", "year")
        o = "author_id"
        m = m.with_columns(c("inst").shift(1).over(o).alias("p_inst"), c("share").shift(1).over(o).alias("p_share"),
                           c("inst").shift(-1).over(o).alias("n_inst"), c("year").shift(-1).over(o).alias("n_year"),
                           pl.int_range(pl.len()).over(o).alias("k"))
        move = ((c("inst") != c("p_inst")) & (c("n_year") == c("year") + 1) & (c("n_inst") == c("inst"))
                & (c("share") >= 0.5) & (c("p_share") >= 0.5)).fill_null(False)
        m = m.with_columns(move.alias("move"), pl.when((c("k") == 0) | move).then(c("inst")).alias("anchor"))
        s = (yrs.join(m.select("author_id", "year", "anchor", "move"), on=["author_id", "year"], how="left")
             .sort("author_id", "year").with_columns(c("anchor").forward_fill().over(o).alias("inst"))
             .filter(c("inst").is_not_null())
             .select("author_id", "year", "inst", c("move").fill_null(False), pl.lit(False).alias("filled")))
        if a.fill != "none":
            s = fill_gaps(s, a.fill, a.seed + b)
        s = (s.filter(c("year").is_between(Y0, Y1))
             .join(start, on="author_id").join(fld, on="author_id")
             .with_columns(
                           c("author_id").is_in(econ["author_id"].implode()).alias("in_econ"),
                           c("author_id").is_in(eb["author_id"].implode()).alias("in_econbus")))
        rows.append(s)
    s = pl.concat(rows)
    s = s.with_columns(c("inst").is_in(list(TOP25)).alias("top25"), DEC.alias("decade"))
    s = s.join(s.group_by("author_id").agg((c("top25") & ~c("filled")).sum().alias("top_years")), on="author_id").filter(c("top_years") >= 2)

    # outcomes from the 41 journals
    ifd = pl.DataFrame([{"abbr": k, "decade": d, "if": v[i]} for k, v in IF.items() for i, d in enumerate((1970, 1980, 1990))])
    w = (pl.scan_parquet(stage / "works" / "*.parquet").join(jb.lazy(), on="source_id", how="inner")
         .filter(c("type").is_in(["article", "review"]) & c("year").is_between(Y0, Y1 + a.lag)
                 & ~c("is_paratext").fill_null(False) & ~c("is_retracted").fill_null(False))
         .select("work_id", "source_id", "abbr", "year", "authors_count",
                 c("first_page").cast(pl.Int32, strict=False).alias("fp"), c("last_page").cast(pl.Int32, strict=False).alias("lp"))
         .collect(engine="streaming"))
    nxt = (w.filter(c("fp").is_not_null()).select("source_id", "year", "fp").unique().sort("source_id", "year", "fp")
           .with_columns(c("fp").shift(-1).over("source_id", "year").alias("nfp")))
    w = w.join(nxt, on=["source_id", "year", "fp"], how="left").with_columns(
        pl.when(c("lp") > c("fp")).then(c("lp") - c("fp") + 1)
        .when(c("nfp") > c("fp")).then(c("nfp") - c("fp")).alias("pages"))
    w = w.with_columns(pl.when(c("pages").is_between(1, 100)).then(c("pages")).alias("pages"))
    w = w.with_columns(DEC.alias("decade")).with_columns(
        c("pages").fill_null(c("pages").median().over("source_id", "decade")).fill_null(10).alias("pages"))
    w = w.select("work_id", "abbr", "authors_count", "pages")
    au = s.select("author_id").unique()
    aw = (pl.scan_parquet(stage / "authorships" / "*.parquet").select("work_id", "author_id", "year")
          .join(w.lazy(), on="work_id", how="inner").join(au.lazy(), on="author_id", how="semi")
          .unique(["work_id", "author_id"]).collect(engine="streaming")
          .with_columns(DEC.alias("decade")).join(ifd, on=["abbr", "decade"], how="left")
          .with_columns((c("year") - a.lag).cast(pl.Int16).alias("year")))
    fr = c("if") / c("authors_count").clip(lower_bound=1)
    ok = c("pages") >= 3
    y = aw.group_by("author_id", "year").agg(
        fr.sum().alias("impact"), pl.len().cast(pl.Float64).alias("articles"),
        (fr * c("pages")).filter(ok).sum().alias("impact_pages"), ok.sum().cast(pl.Float64).alias("articles3"))
    ys = ("impact", "articles", "impact_pages", "articles3")
    kart = aw.filter(c("year").is_between(Y0, Y1)).group_by("author_id").agg(pl.len().alias("kmz_articles"))
    s = (s.join(y, on=["author_id", "year"], how="left").with_columns(*[c(v).fill_null(0.0) for v in ys])
         .join(kart, on="author_id", how="left").with_columns(c("kmz_articles").fill_null(0)))
    cy = (c("year") - c("first_year")).cast(pl.Float64)
    s = s.with_columns(cy.alias("cy"), (cy**2 / 100).alias("cy2"), (cy**3 / 1000).alias("cy3"),
                       *[(c("top25") & (c("decade") == d)).cast(pl.Float64).alias(f"t{d % 100:02d}") for d in (1970, 1980, 1990)])
    return s


def estimate(s: pl.DataFrame, work: Path, label: str) -> dict:
    import pyfixest as pf

    import dfeols

    xs = "t70 + t80 + t90 + cy + cy2 + cy3"
    ys = ("impact", "articles", "impact_pages", "articles3")
    desc = (s.group_by("decade", "top25").agg(*[c(v).mean() for v in ("impact_pages", "articles3")], pl.len())
            .sort("decade", "top25").to_dicts())
    out = {"label": label, "N": s.height, "authors": s["author_id"].n_unique(), "moves": int(s["move"].sum()),
           "filled_share": float(s["filled"].mean()), "years_per_author": s.height / s["author_id"].n_unique(),
           "means": {k: float(s[k].mean()) for k in ys}, "by_decade_top25": desc}
    work.mkdir(parents=True, exist_ok=True)
    cols = ["author_id", "decade", *ys, "t70", "t80", "t90", "cy", "cy2", "cy3"]
    s.select(cols).write_parquet(work / "data.parquet")
    ds = dfeols.prep(str(work / "data.parquet"), shard_dir=work / "shards", partition="author_id",
                     fixed_effects=["author_id", "decade"], clusters=["author_id"], target_rows=50_000, overwrite=True)
    pdf = s.select(cols).to_pandas()
    for yv in ys:
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
    ap.add_argument("--fill", choices=["none", "midpoint", "random"], default="none")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--affil-fallback", action="store_true")
    ap.add_argument("--prefer-edu", action="store_true")
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
    core = (c("last_year") - c("first_year") >= 5) & (c("kmz_articles") >= 3)
    samples["econ_core"] = s.filter(c("in_econ") & core)
    samples["econbus_core"] = s.filter(c("in_econbus") & core)
    res = {"fill": a.fill, "seed": a.seed if a.fill == "random" else None, "lag": a.lag, "affil_fallback": a.affil_fallback, "prefer_edu": a.prefer_edu, "kmz": KMZ, "kmz_b_over_mean": {k: [b / KMZ["impact_mean"] for b in v] for k, v in KMZ.items() if k in ("econ", "finance")},
           "results": []}
    for k, d in samples.items():
        r = estimate(d, a.out / "work" / k, k)
        res["results"].append(r)
        print(json.dumps({"sample": k, "N": r["N"], "authors": r["authors"], "moves": r["moves"],
                          **{yv: {v: round(r[yv]["coef"][v]["b"], 4) for v in ("t70", "t80", "t90")} for yv in ("impact_pages", "articles3")}}), flush=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
