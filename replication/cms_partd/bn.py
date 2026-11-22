# SPDX-License-Identifier: GPL-3.0-or-later
"""D9 extension: Beheshti & Neller (2026) mover event study, eq. (1) and (2), on the movers panel.

Reads build/cms_partd/movers/panel.parquet (movers.py) and writes, per censoring variant, the
mover regression file with delta and the event-time regressors, then fits eq. (1) and (2) with
pyfixest (in memory; ~0.7M rows). Paper: raw/papers/ (RAND J. Econ. 57(3):601-626); notes in
docs/cms_partd.md.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/cms_partd/bn.py --out build/cms_partd/bn
    (--grouping tab1_4|raw_spec|none for the specialty-group sensitivity, into its own --out;
     --window all for 2013-2024, movers and delta over that window)

Spec (paper section 2, fn 10, 34; rules Ted 2026-11-21):
  y_its = a_i + g_{t,o} + sum_{tau != -1} b_tau 1(t = tau) delta_ido_s + e      (1)
  y_its = a_i + g_{t,o} + b_tr 1(0 <= t <= 1) delta + b_post 1(t > 1) delta + e   (2)
  t = year - move year (first year in the new HRR); FE: NPI, relative year x origin HRR;
  SE clustered by NPI. Movers: contiguous years in 2013-2019 and exactly one HRR change.
  delta = ybar_{d,s} - ybar_{o,s}: for each HRR x calendar year x specialty group, the mean of y
  over all prescribers excluding the mover; then averaged over the years 2013-2019 (fn 10).
  Specialty group: GROUPS below (proposal, not yet agreed with Ted), one per NPI, the most
  frequent over 2013-2019 (ties: alphabetical).
  Censoring variants (fn 34; Ted: both): "py" drops prescriber-years with censored
  beneficiaries, "pr" drops every prescriber with one; applied both to the cell means and to the
  regression sample, so both outcomes use the same rows.
Outcomes: log cost (Tot_Drug_Cst) and log cost per beneficiary; B&N: b_post 0.42 and 0.29.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import polars as pl

c = pl.col
WINDOWS = {"bn": (2013, 2019), "all": (2013, 2024)}  # as in movers.py
LO, HI = WINDOWS["bn"]
W = "bn"
OUTCOMES = {"log_cost": "log_cost", "log_cost_pb": "log_cost_pb"}
# Section 4.3: each part regressed on the delta of its total (fn 33: the parts' b_post sum to the total's)
PARTS = {
    "log_cost": ["log_cost_per_claim", "log_claims_pb", "log_benes"],
    "log_cost_pb": ["log_cost_per_claim", "log_claims_pb"],
}
VARIANTS = ("py", "pr")

# Specialty groups (B&N section 3: primary care, medical, surgical, nurse practitioners, dental;
# Table 1 reports primary / medical / surgical / other). First match wins; the rest is "other".
GROUPS = [
    ("np", r"^(nurse practitioner|certified clinical nurse specialist|certified nurse midwife)$"),
    ("dental", r"dentist|oral surgery|maxillofacial|^oral & maxillofacial|oral medicinist|dental therap"),
    (
        "primary",
        r"^(internal medicine|family practice|family medicine|general practice|pediatric medicine|"
        r"pediatrics|geriatric medicine|preventive medicine|osteopathic manipulative medicine)$",
    ),
    (
        "surgical",
        r"surg|^urology$|^otolaryngology$|^ophthalmology$|^obstetrics|gynecolog|^colon & rectal",
    ),
    (
        "medical",
        r"cardi|oncolog|hematolog|nephrolog|gastroenterolog|pulmonary|endocrinolog|rheumatolog|"
        r"infectious|neurolog|psychiatr|dermatolog|allergy|emergency medicine|hospitalist|critical care|"
        r"physical medicine|anesthesiolog$|^anesthesiology$|pain|sleep medicine|addiction medicine|"
        r"hospice and palliative|sports medicine|radiology|nuclear medicine|pathology$|medical genetics|"
        r"medical toxicology|clinical pharmacology|peripheral vascular|undersea|electrodiagnostic|"
        r"phlebology|legal medicine|neuromusculoskeletal|hematopoietic|neuropsychiatry|radiation oncology",
    ),
]


def spec_group(s: str | None) -> str:
    t = (s or "").strip().lower()
    for g, pat in GROUPS:
        if re.search(pat, t):
            return g
    return "other"


GROUPINGS = ("groups6", "tab1_4", "raw_spec", "none")


def load(panel: Path, grouping: str = "groups6", keep_npis: Path | None = None) -> pl.DataFrame:
    """grouping: groups6 = GROUPS; tab1_4 = Table 1's four (np, dental into other); raw_spec =
    the CMS Prscrbr_Type label; none = one group (HRR means over all prescribers).
    keep_npis: Parquet with an `npi` column; only those prescribers are kept, for the cell means
    (delta) and the movers alike (e.g. NPIs found in Physician Compare, as B&N's sample)."""
    lf = pl.scan_parquet(panel)
    if keep_npis is not None:
        lf = lf.join(pl.scan_parquet(keep_npis).select("npi").unique(), on="npi", how="semi")
    p = (
        lf
        .filter(c("year").is_between(LO, HI))
        .select("npi", "year", "hrr", "spec", "cost", "claims", "benes", "benes_censored",
                f"contig_{W}", f"moves_{W}", f"move_year_{W}", f"npi_any_censored_{W}")
        .collect(engine="streaming")
    )
    specs = p["spec"].unique().to_list()
    gmap = pl.DataFrame({"spec": specs, "grp": [spec_group(s) for s in specs]})
    p = p.join(gmap, on="spec", how="left").with_columns(c("grp").fill_null("other"))
    if grouping == "tab1_4":
        p = p.with_columns(pl.when(c("grp").is_in(["np", "dental"])).then(pl.lit("other")).otherwise(c("grp")).alias("grp"))
    elif grouping == "raw_spec":
        p = p.with_columns(c("spec").fill_null("?").alias("grp"))
    elif grouping == "none":
        p = p.with_columns(pl.lit("all").alias("grp"))
    mode = (
        p.group_by("npi", "grp").len()
        .sort(["npi", "len", "grp"], descending=[False, True, False])
        .unique("npi", keep="first", maintain_order=True)
        .select("npi", c("grp").alias("group"))
    )
    return p.join(mode, on="npi").with_columns(
        c("cost").log().alias("log_cost"),
        (c("cost") / c("benes")).log().alias("log_cost_pb"),
        (c("cost") / c("claims")).log().alias("log_cost_per_claim"),
        (c("claims") / c("benes")).log().alias("log_claims_pb"),
        c("benes").log().alias("log_benes"),
    )


def deltas(s: pl.DataFrame, movers: pl.DataFrame, y: str) -> pl.DataFrame:
    """Leave-out delta per mover for outcome y; s is the variant's prescriber-year sample."""
    d = s.filter(c(y).is_finite())
    cells = d.group_by("hrr", "year", "group").agg(c(y).sum().alias("S"), pl.len().alias("n"))
    own = d.select("npi", "hrr", "year", c(y).alias("y_own"))
    out = {}
    for role in ("origin", "dest"):
        m = (
            movers.select("npi", "group", c(role).alias("hrr"))
            .join(cells, on=["hrr", "group"], how="inner")
            .join(own, on=["npi", "hrr", "year"], how="left")
            .with_columns(
                pl.when(c("y_own").is_null()).then(c("S") / c("n"))
                .when(c("n") > 1).then((c("S") - c("y_own")) / (c("n") - 1))
                .otherwise(None).alias("m")
            )
            .group_by("npi").agg(c("m").mean().alias(role))
        )
        out[role] = m
    return out["origin"].join(out["dest"], on="npi", how="inner", suffix="_d").select(
        "npi", (c("dest") - c("origin")).alias(f"delta_{y}")
    )


def build(p: pl.DataFrame, variant: str) -> tuple[pl.DataFrame, dict]:
    if variant == "py":
        s = p.filter(~c("benes_censored"))
    else:
        s = p.filter(~c(f"npi_any_censored_{W}"))
    mv = p.filter(c(f"contig_{W}") & (c(f"moves_{W}") == 1))
    my = mv.select("npi", c(f"move_year_{W}").alias("move_year")).unique("npi")
    od = (
        mv.join(my, on="npi")
        .group_by("npi")
        .agg(
            c("hrr").filter(c("year") < c("move_year")).first().alias("origin"),
            c("hrr").filter(c("year") >= c("move_year")).first().alias("dest"),
            c("group").first(),
            c("move_year").first(),
        )
    )
    reg = s.join(od.select("npi", "origin", "dest", "move_year"), on="npi", how="inner")
    for y in OUTCOMES:
        reg = reg.join(deltas(s, od, y), on="npi", how="left")
    reg = reg.with_columns((c("year") - c("move_year")).cast(pl.Int16).alias("rel"))
    reg = reg.with_columns((c("rel").cast(pl.Int32) * 1000 + c("origin")).alias("rel_origin"))
    info = {
        "prescribers_in_sample": s["npi"].n_unique(),
        "prescriber_years_in_sample": s.height,
        "movers_defined": od.height,
        "movers_in_regression": reg["npi"].n_unique(),
        "mover_years": reg.height,
        "mover_years_pre": int((reg["rel"] < 0).sum()),
        "mover_years_post": int((reg["rel"] >= 0).sum()),
        "rel_range": [int(reg["rel"].min()), int(reg["rel"].max())],
        "groups_npis": dict(s.unique("npi")["group"].value_counts().iter_rows()),
        "groups_prescriber_years": dict(s["group"].value_counts().iter_rows()),
    }
    for y in OUTCOMES:
        dl = reg.unique("npi")[f"delta_{y}"]
        info[f"delta_{y}"] = {"mean": dl.mean(), "sd": dl.std(), "null": int(dl.is_null().sum())}
    return reg, info


def regressors(reg: pl.DataFrame, y: str) -> tuple[pl.DataFrame, list[str], list[str]]:
    d = c(f"delta_{y}")
    taus = [t for t in range(int(reg["rel"].min()), int(reg["rel"].max()) + 1) if t != -1]
    name = lambda t: f"es_m{-t}" if t < 0 else f"es_p{t}"  # noqa: E731
    reg = reg.with_columns(
        *[(pl.when(c("rel") == t).then(d).otherwise(0.0)).alias(name(t)) for t in taus],
        pl.when(c("rel").is_between(0, 1)).then(d).otherwise(0.0).alias("b_transition"),
        pl.when(c("rel") > 1).then(d).otherwise(0.0).alias("b_post"),
    )
    return reg, [name(t) for t in taus], ["b_transition", "b_post"]


def cohort_regressors(reg: pl.DataFrame, y: str) -> tuple[pl.DataFrame, list[str]]:
    """Eq. (2) with transition and post slopes by move year (cohorts with a post period)."""
    d = c(f"delta_{y}")
    years = sorted(int(v) for v in reg.filter(c("rel") > 1)["move_year"].unique())
    xs = []
    for my in years:
        k = c("move_year") == my
        reg = reg.with_columns(
            pl.when(k & c("rel").is_between(0, 1)).then(d).otherwise(0.0).alias(f"tr_{my}"),
            pl.when(k & (c("rel") > 1)).then(d).otherwise(0.0).alias(f"post_{my}"),
        )
        xs += [f"tr_{my}", f"post_{my}"]
    # cohorts with no post period (moved in the last two years) keep one pooled transition slope
    late = ~c("move_year").is_in(years)
    reg = reg.with_columns(pl.when(late & c("rel").is_between(0, 1)).then(d).otherwise(0.0).alias("tr_late"))
    if reg["tr_late"].abs().sum() > 0:
        xs.append("tr_late")
    return reg, xs


def fit(df, y: str, xs: list[str]) -> dict:
    import pyfixest as pf

    t0 = time.time()
    m = pf.feols(f"{y} ~ {' + '.join(xs)} | npi + rel_origin", data=df, vcov={"CRV1": "npi"},
                 fixef_tol=1e-10)
    t = m.tidy()
    return {
        "N": int(m._N),
        "seconds": round(time.time() - t0, 1),
        "coef": {k: [float(t.loc[k, "Estimate"]), float(t.loc[k, "Std. Error"])] for k in xs},
    }


def main() -> None:
    global LO, HI, W
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=Path("build/cms_partd/movers/panel.parquet"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--grouping", choices=GROUPINGS, default="groups6")
    ap.add_argument("--window", choices=list(WINDOWS), default="bn", help="bn 2013-2019 (paper), all 2013-2024")
    ap.add_argument("--keep-npis", type=Path, help="Parquet of NPIs to keep (e.g. build/cms_partd/physcomp/npis_2014_2021.parquet)")
    a = ap.parse_args()
    W = a.window
    LO, HI = WINDOWS[W]
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = load(a.panel, a.grouping, a.keep_npis)
    res: dict = {"grouping": a.grouping, "window": [W, LO, HI], "groups_regex": GROUPS,
                 "keep_npis": str(a.keep_npis) if a.keep_npis else None, "variants": {}}
    for v in VARIANTS:
        reg, info = build(p, v)
        reg.sort("npi", "year").write_parquet(a.out / f"movers_{v}.parquet")
        res["variants"][v] = {"info": info, "fits": {}}
        print(v, json.dumps(info, default=str), flush=True)
        if a.no_fit:
            continue
        for y in OUTCOMES:
            ok = c(f"delta_{y}").is_not_null() & pl.all_horizontal([c(z).is_finite() for z in [y, *PARTS[y]]])
            r, es, summ = regressors(reg.filter(ok), y)
            r, coh = cohort_regressors(r, y)
            df = r.select("npi", "rel_origin", y, *PARTS[y], *es, *summ, *coh).to_pandas()
            fy = res["variants"][v]["fits"][y] = {"eq1": fit(df, y, es), "eq2": fit(df, y, summ),
                                                  "eq2_cohort": fit(df, y, coh)}
            # decomposition: each part on the total's delta, same rows
            fy["parts"] = {z: fit(df, z, summ) for z in PARTS[y]}
            tot = sum(fy["parts"][z]["coef"]["b_post"][0] for z in PARTS[y])
            print(v, y, "eq2", fy["eq2"]["N"], fy["eq2"]["coef"]["b_post"], "parts sum", round(tot, 6), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    (a.out / "results.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
