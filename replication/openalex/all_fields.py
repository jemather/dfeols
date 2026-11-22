# SPDX-License-Identifier: GPL-3.0-or-later
"""D14 stage 4: the KMZ (2009) Table 5 Panel D specification in every OpenAlex field.

Uses a `full.py` variant (default edu). Field = the author's modal OpenAlex field (26 fields).
Top institutions, two definitions:
  kmz     KMZ's 25 economics schools, the same 25 in every field
  field   per field, the 25 education institutions with the most frac_cites by authors whose main
          institution it was in 1990-1999 (fixed ranking, as KMZ's)
Sample per field and definition: authors with 2+ unfilled years at a top institution in 1970-2025,
career span 5+ years and 3+ works (the all-journal analogue of KMZ's core); years 1970-2025.
Outcomes frac_works and frac_cites (all journals; impact_pages exists only for economics).
y = author FE + decade FE + sum_d b_d [top x decade d] + career cubic, CRV1 by author, per field.
Pooled: all fields together, y divided by its field x year mean in the sample, FE author +
field x decade, top x decade terms common to all fields.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/openalex/all_fields.py --out build/openalex/all_fields
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kmz import TOP25  # noqa: E402

c = pl.col
FIELDS = {11: "Agricultural and Biological Sciences", 12: "Arts and Humanities", 13: "Biochemistry, Genetics and Molecular Biology",
          14: "Business, Management and Accounting", 15: "Chemical Engineering", 16: "Chemistry", 17: "Computer Science",
          18: "Decision Sciences", 19: "Earth and Planetary Sciences", 20: "Economics, Econometrics and Finance", 21: "Energy",
          22: "Engineering", 23: "Environmental Science", 24: "Immunology and Microbiology", 25: "Materials Science",
          26: "Mathematics", 27: "Medicine", 28: "Neuroscience", 29: "Nursing", 30: "Pharmacology, Toxicology and Pharmaceutics",
          31: "Physics and Astronomy", 32: "Psychology", 33: "Social Sciences", 34: "Veterinary", 35: "Dentistry",
          36: "Health Professions"}
YS = ("frac_works", "frac_cites")


def load(full: Path, variant: str) -> pl.DataFrame:
    return (pl.scan_parquet(full / variant / "*.parquet").filter(c("year").is_between(1970, 2025) & c("field_id").is_in(list(FIELDS)))
            .select("author_id", "year", "inst", "filled", "first_year", "last_year", "field_id", "works", "frac_works", "frac_cites",
                    c("top25").alias("top_kmz"))
            .collect(engine="streaming"))


def field_tops(s: pl.DataFrame, oa: Path) -> pl.DataFrame:
    inst = pl.read_parquet(oa / "stage" / "institutions" / "*.parquet", columns=["institution_id", "display_name", "type"])
    edu = inst.filter(c("type") == "education").select(c("institution_id").alias("inst"), "display_name")
    return (s.filter(c("year").is_between(1990, 1999) & ~c("filled")).join(edu, on="inst", how="semi")
            .group_by("field_id", "inst").agg(c("frac_cites").sum())
            .sort(["field_id", "frac_cites", "inst"], descending=[False, True, False])
            .group_by("field_id", maintain_order=True).head(25)
            .join(edu, on="inst", how="left").with_columns(pl.int_range(pl.len()).over("field_id").alias("rank")))


def sample(s: pl.DataFrame, top: str) -> pl.DataFrame:
    ev = s.filter(c(top) & ~c("filled")).group_by("author_id").len().filter(c("len") >= 2).select("author_id")
    core = (s.group_by("author_id").agg(c("works").sum().alias("k"), (c("last_year") - c("first_year")).first().alias("span"))
            .filter((c("k") >= 3) & (c("span") >= 5)).select("author_id"))
    w = s.join(ev, on="author_id", how="semi").join(core, on="author_id", how="semi")
    cy = (c("year") - c("first_year")).cast(pl.Float64)
    w = w.with_columns((c("year") // 10 * 10).cast(pl.Int16).alias("decade"), cy.alias("cy"), (cy**2 / 100).alias("cy2"),
                       (cy**3 / 1000).alias("cy3"))
    decs = sorted(w["decade"].unique().to_list())
    w = w.with_columns(*[(c(top) & (c("decade") == d)).cast(pl.Float64).alias(f"t{d}") for d in decs],
                       c(top).alias("top"))
    m = [(c(y) / c(y).mean().over("field_id", "year")).alias(f"{y}_n") for y in YS]
    return w.with_columns(*m, (c("field_id").cast(pl.Int32) * 10000 + c("decade")).alias("field_decade"))


def fit(w: pl.DataFrame, work: Path, ys: list[str], fe: list[str], pyfixest: bool) -> dict:
    import pyfixest as pf

    import dfeols

    ts = [x for x in w.columns if x.startswith("t") and x[1:].isdigit() and w[x].sum() > 0]
    cols = ["author_id", *fe[1:], *ys, *ts, "cy", "cy2", "cy3"]
    work.mkdir(parents=True, exist_ok=True)
    w.select(cols).write_parquet(work / "data.parquet")
    ds = dfeols.prep(str(work / "data.parquet"), shard_dir=work / "shards", partition="author_id",
                     fixed_effects=fe, clusters=["author_id"], target_rows=1_000_000, overwrite=True)
    out = {"N": w.height, "authors": w["author_id"].n_unique(), "top_share": float(w["top"].mean()),
           "means": {y: float(w[y].mean()) for y in ys}}
    pdf = w.select(cols).to_pandas() if pyfixest else None
    for y in ys:
        f = f"{y} ~ {' + '.join(ts)} + cy + cy2 + cy3 | {' + '.join(fe)}"
        t0 = time.time()
        r = dfeols.fit(ds, f, cluster="author_id", workdir=str(work / f"fit_{y}"), log_every=1e9, record=False)
        td = r.tidy(None, "CRV1")
        rec = {"N": int(r.N), "sec": round(time.time() - t0, 1), "coef": {}}
        tp = pf.feols(f, data=pdf, vcov={"CRV1": "author_id"}, fixef_tol=1e-12).tidy() if pyfixest else None
        for t in ts:
            e = {"b": float(td.loc[t, "Estimate"]), "se": float(td.loc[t, "Std. Error"])}
            if pyfixest:
                pb, ps = float(tp.loc[t, "Estimate"]), float(tp.loc[t, "Std. Error"])
                e["rel_b_pyfixest"] = abs(e["b"] - pb) / max(abs(pb), 1e-12)
                e["rel_se_pyfixest"] = abs(e["se"] - ps) / ps
            rec["coef"][t] = e
        out[y] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openalex", type=Path, default=Path("build/openalex"))
    ap.add_argument("--full", type=Path, default=Path("build/openalex/full"))
    ap.add_argument("--variant", default="edu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tops", nargs="+", default=["kmz", "field"])
    ap.add_argument("--fields", type=int, nargs="*", help="only these fields (testing)")
    ap.add_argument("--no-pooled", action="store_true")
    ap.add_argument("--pyfixest-max-rows", type=int, default=20_000_000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s = load(a.full, a.variant)
    if a.fields:
        s = s.filter(c("field_id").is_in(a.fields))
    ft = field_tops(s, a.openalex)
    ft.write_parquet(a.out / "field_tops.parquet")
    s = s.join(ft.select("field_id", "inst", pl.lit(True).alias("top_field")), on=["field_id", "inst"], how="left") \
         .with_columns(c("top_field").fill_null(False))
    print(f"{time.time() - t0:.0f}s loaded {s.height:,} rows", flush=True)
    res = {"variant": a.variant, "fields": FIELDS, "results": []}
    for top in a.tops:
        w = sample(s, f"top_{top}")
        for fid in sorted(w["field_id"].unique().to_list()):
            wf = w.filter(c("field_id") == fid)
            r = {"top": top, "field_id": fid, "field": FIELDS[fid],
                 **fit(wf, a.out / "work" / top / str(fid), list(YS), ["author_id", "decade"], wf.height <= a.pyfixest_max_rows)}
            res["results"].append(r)
            print(json.dumps({"top": top, "field": fid, "N": r["N"], "sec": r["frac_works"]["sec"],
                              "frac_works": {t: round(e["b"], 3) for t, e in r["frac_works"]["coef"].items()}}), flush=True)
            (a.out / "results.json").write_text(json.dumps(res, indent=1))
        if not a.no_pooled:
            r = {"top": top, "field_id": None, "field": "pooled (normalized, field x decade FE)",
                 **fit(w, a.out / "work" / top / "pooled", [f"{y}_n" for y in YS], ["author_id", "field_decade"],
                       w.height <= a.pyfixest_max_rows)}
            res["results"].append(r)
            print(json.dumps({"top": top, "pooled": True, "N": r["N"],
                              "frac_works_n": {t: round(e["b"], 3) for t, e in r["frac_works_n"]["coef"].items()}}), flush=True)
            (a.out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"{time.time() - t0:.0f}s done", flush=True)


if __name__ == "__main__":
    main()
