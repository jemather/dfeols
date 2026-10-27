# SPDX-License-Identifier: GPL-3.0-or-later
"""Gerardi & Shapiro (2009) Tables 2 and 3 on build/db1b/gs/panel.parquet.

    python repo/replication/db1b/table2.py [--panel build/db1b/gs/panel.parquet] [--dfeols]

Samples: all routes; big-city routes (both endpoints big-city airports, App. C Table C3); leisure
routes (an endpoint is a leisure airport, Table C2). Table 2: lodd(Gini) on all three samples.
Table 3: ln P90 and ln P10 on big-city and leisure routes. Each with col. 1 (-lnHERF, IV), col. 2
(lnN, IV), col. 3 (LEGN, LCCN, OLS); n. 32 gives OLS versions of cols. 1-2 for Table 2, all routes.

--dfeols also fits the all-routes Table 2 models with dfeols, on the same sample sharded by route
(the cluster; carrier-route FEs nest in it), and prints the differences from pyfixest.

Model (eq. 1, p. 14): lodd(Gini) on competition, a bankruptcy dummy, time dummies and carrier-
route fixed effects; SEs clustered by route. pyfixest throughout (dfeols has no IV yet).

Signs: the PDF text loses minus signs; the paper reports -lnHERF and finds competition lowers
dispersion, so all are negative.

Instruments (p. 15, App. A): lnPASSRTE, GENSP and IRUTHERF. IRUTHERF (Borenstein and Rose 1994,
App. A, XFLTHERF) is s_hat^2 + (HERF - s^2) / (1 - s)^2 * (1 - s_hat)^2, s the carrier's market share
and s_hat its first-stage fitted value. The first stage is not given in either paper; here
MKTSHARE ~ lnPASSRTE + GENSP + bankrupt | carrier_route + time. This is an assumption (Ted,
2026-10-27): of the readings tried it gives the estimates closest to the published ones
(docs/db1b.md, "IRUTHERF first stage").
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bankruptcy


# {(sample, outcome): {model: {term: (b, se)}}}; Table 2 (p. 17) and Table 3 (p. 18)
def _cols(herf, n, legn, lccn):
    return {"col1": {"neg_ln_herf": herf}, "col2": {"ln_n": n}, "col3": {"legn": legn, "lccn": lccn}}


PAPER = {
    ("all", "lodd"): _cols((-0.121, 0.022), (-0.177, 0.029), (-0.005, 0.004), (-0.056, 0.007))
    | {"n32_herf": {"neg_ln_herf": (-0.033, 0.016)}, "n32_n": {"ln_n": (-0.021, 0.006)}},
    ("big", "lodd"): _cols((-0.229, 0.038), (-0.357, 0.060), (-0.013, 0.006), (-0.089, 0.010)),
    ("leisure", "lodd"): _cols((-0.076, 0.044), (-0.132, 0.058), (-0.006, 0.009), (-0.034, 0.012)),
    ("big", "ln_p90"): _cols((-0.406, 0.041), (-0.652, 0.067), (-0.041, 0.005), (-0.156, 0.014)),
    ("big", "ln_p10"): _cols((-0.166, 0.027), (-0.273, 0.039), (-0.025, 0.004), (-0.054, 0.007)),
    ("leisure", "ln_p90"): _cols((-0.168, 0.028), (-0.259, 0.041), (-0.011, 0.008), (-0.090, 0.011)),
    ("leisure", "ln_p10"): _cols((-0.119, 0.024), (-0.176, 0.036), (-0.008, 0.007), (-0.065, 0.009)),
}
PAPER_N = {"all": 112_499, "big": 32_188, "leisure": 24_555}
HERE = Path(__file__).resolve().parent


def samples(d):
    """Boolean masks (pandas) for the all, big-city and leisure samples (App. C, Tables C2-C3)."""
    c2 = pl.read_csv(HERE / "table_c2.csv", comment_prefix="#")
    c3 = pl.read_csv(HERE / "table_c3.csv", comment_prefix="#")
    leisure = c2.filter(leisure="Yes")["airport"].to_list()
    big = c3.filter(big_city="Yes")["airport"].unique().to_list()
    return {
        "all": d["origin"].notna(),
        "big": d["origin"].isin(big) & d["dest"].isin(big),
        "leisure": d["origin"].isin(leisure) | d["dest"].isin(leisure),
    }


def formula(model: str, y: str) -> str:
    return {
        "col1": f"{y} ~ bankrupt | {FE} | neg_ln_herf ~ {INSTRUMENTS}",
        "col2": f"{y} ~ bankrupt | {FE} | ln_n ~ {INSTRUMENTS}",
        "col3": f"{y} ~ legn + lccn + bankrupt | {FE}",
        "n32_herf": f"{y} ~ neg_ln_herf + bankrupt | {FE}",
        "n32_n": f"{y} ~ ln_n + bankrupt | {FE}",
    }[model]


FE = "carrier_route + time"
INSTRUMENTS = "ln_passrte + gensp + irutherf"


def bankrupt_quarters() -> pl.DataFrame:
    """carrier, year, quarter for quarters with any day in Chapter 11 (STATE.md 2026-10-27)."""
    rows = set()
    for p in bankruptcy.periods():
        for y in range(1993, 2026):  # 1993Q1-2025Q2 and beyond; quarters outside the panel never join
            for q in range(1, 5):
                start, end = date(y, 3 * q - 2, 1), date(y + (q == 4), (3 * q) % 12 + 1, 1)
                if p["start"] < end and p["end"] >= start:
                    rows.add((p["carrier"], y, q))
    return pl.DataFrame(sorted(rows), schema=["carrier", "year", "quarter"], orient="row").with_columns(
        bankrupt=pl.lit(1, pl.Int8)
    )


def estimation_sample(panel: pl.DataFrame) -> pl.DataFrame:
    p = panel.join(bankrupt_quarters(), on=["carrier", "year", "quarter"], how="left")
    p = p.with_columns(pl.col("bankrupt").fill_null(0), neg_ln_herf=-pl.col("ln_herf"))
    p = p.filter(pl.col("lodd").is_finite(), pl.col("ln_passrte").is_finite())
    # carrier-routes seen once are singletons for the carrier-route FE; dropping them keeps every
    # model (and the first-stage fitted values) on one sample. Time FE have no singletons here.
    return p.filter(pl.len().over("carrier_route") > 1)


def add_irutherf(d, pf):
    """First-stage fitted market share and the IRUTHERF instrument (pandas frame, in place)."""
    first = pf.feols(f"mktshare ~ ln_passrte + gensp + bankrupt | {FE}", data=d)
    s_hat = first.predict().clip(0, 1)
    s = d["mktshare"]
    others = (d["herf"] - s**2) / (1 - s) ** 2
    others = others.where(s < 1, 0.0)  # monopoly: no other carriers to rescale
    d["irutherf"] = s_hat**2 + others * (1 - s_hat) ** 2
    return first


COLUMNS = [
    "lodd",
    "ln_p90",
    "ln_p10",
    "neg_ln_herf",
    "ln_n",
    "legn",
    "lccn",
    "bankrupt",
    "ln_passrte",
    "gensp",
    "irutherf",
    "carrier_route",
    "route",
    "time",
]


def dfeols_dataset(d, work: Path, target_rows: int):
    """The estimation sample as Parquet, sharded by route for dfeols."""
    import dfeols

    work.mkdir(parents=True, exist_ok=True)
    src = work / "sample.parquet"
    pl.from_pandas(d[COLUMNS]).write_parquet(src)
    return dfeols.prep(
        str(src),
        shard_dir=str(work / "shards"),
        partition="route",
        fixed_effects=["carrier_route", "time"],
        clusters=["route"],
        target_rows=target_rows,
        overwrite=True,
    )


def dfeols_fit(ds, fml: str, label: str):
    import dfeols

    r = dfeols.fit(ds, fml, cluster="route", vcov=["CRV1"], label=label, log_every=1e9, record=False)
    return r.tidy(None, "CRV1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default="build/db1b/gs/panel.parquet", type=Path)
    ap.add_argument("--dfeols", action="store_true", help="also fit with dfeols and compare")
    ap.add_argument("--work", default="build/db1b/gs/table2", type=Path, help="dfeols sample and shards")
    ap.add_argument("--target-rows", default=20_000, type=int, help="dfeols rows per shard")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    import pyfixest as pf

    d = estimation_sample(pl.read_parquet(a.panel)).to_pandas()
    first = add_irutherf(d, pf)
    print(f"N = {len(d):,}; bankrupt rows {int(d['bankrupt'].sum()):,}")
    print("first stage for MKTSHARE:", first.tidy()[["Estimate", "Std. Error"]].round(4).to_dict("index"))

    vc = {"CRV1": "route"}
    masks = samples(d)
    print("samples:", {k: f"{int(m.sum()):,} (paper {PAPER_N[k]:,})" for k, m in masks.items()})
    ds = dfeols_dataset(d, a.work, a.target_rows) if a.dfeols else None
    head = f"\n{'sample':8s} {'y':7s} {'model':9s} {'term':12s} {'pyfixest':>18s} {'paper':>16s}"
    print(head + (f" {'dfeols':>18s} {'rel b':>8s} {'rel se':>8s}" if ds else ""))
    for (sample, y), models in PAPER.items():
        sub = d[masks[sample]]
        for name, terms in models.items():
            fml = formula(name, y)
            t = pf.feols(fml, data=sub, vcov=vc).tidy()
            td = dfeols_fit(ds, fml, name) if ds is not None and sample == "all" else None
            for term, (pb, pse) in terms.items():
                b, se = t.loc[term, "Estimate"], t.loc[term, "Std. Error"]
                line = f"{sample:8s} {y:7s} {name:9s} {term:12s} {b:+9.4f} ({se:.4f}) {pb:+8.3f} ({pse:.3f})"
                if td is not None:
                    db, dse = td.loc[term, "Estimate"], td.loc[term, "Std. Error"]
                    line += f" {db:+9.4f} ({dse:.4f}) {abs(db - b) / abs(b):8.1e} {abs(dse - se) / se:8.1e}"
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
