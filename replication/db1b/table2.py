# SPDX-License-Identifier: GPL-3.0-or-later
"""Gerardi & Shapiro (2009) Table 2, panel A (all routes), on build/db1b/gs/panel.parquet.

    python repo/replication/db1b/table2.py [--panel build/db1b/gs/panel.parquet] [--dfeols]

--dfeols also fits every model with dfeols, on the same sample sharded by route (the cluster;
carrier-route FEs nest in it), and prints the differences from pyfixest.

Model (eq. 1, p. 14): lodd(Gini) on competition, a bankruptcy dummy, time dummies and carrier-
route fixed effects; SEs clustered by route. pyfixest throughout (dfeols has no IV yet).

  col. 1   -lnHERF, instrumented          paper  -.121 (.022)
  col. 2   lnN, instrumented              paper  -.177 (.029)
  col. 3   LEGN, LCCN, not instrumented   paper  -.005 (.004), -.056 (.007)
  n. 32    cols. 1-2 by OLS               paper  -.033 (.016), -.021 (.006)

Signs: the PDF text loses minus signs; the paper reports -lnHERF and finds competition lowers
dispersion, so all are negative.

Instruments (p. 15, App. A): lnPASSRTE, GENSP and IRUTHERF. IRUTHERF (Borenstein and Rose 1994,
App. A, XFLTHERF) is s_hat^2 + (HERF - s^2) / (1 - s)^2 * (1 - s_hat)^2, s the carrier's market share
and s_hat its first-stage fitted value. The first stage is not given in either paper; here
MKTSHARE ~ lnPASSRTE + GENSP + bankrupt | carrier_route + time (Claude's reading, for Ted).
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

PAPER = {
    "col1": {"neg_ln_herf": (-0.121, 0.022)},
    "col2": {"ln_n": (-0.177, 0.029)},
    "col3": {"legn": (-0.005, 0.004), "lccn": (-0.056, 0.007)},
    "n32_herf": {"neg_ln_herf": (-0.033, 0.016)},
    "n32_n": {"ln_n": (-0.021, 0.006)},
}
FE = "carrier_route + time"
INSTRUMENTS = "ln_passrte + gensp + irutherf"


def bankrupt_quarters() -> pl.DataFrame:
    """carrier, year, quarter for quarters with any day in Chapter 11 (STATE.md 2026-10-27)."""
    rows = set()
    for p in bankruptcy.periods():
        for y in range(1993, 2007):
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
    print(f"N = {len(d):,} (paper 112,499); bankrupt rows {int(d['bankrupt'].sum()):,}")
    print("first stage for MKTSHARE:", first.tidy()[["Estimate", "Std. Error"]].round(4).to_dict("index"))

    vc = {"CRV1": "route"}
    models = {
        "col1": f"lodd ~ bankrupt | {FE} | neg_ln_herf ~ {INSTRUMENTS}",
        "col2": f"lodd ~ bankrupt | {FE} | ln_n ~ {INSTRUMENTS}",
        "col3": f"lodd ~ legn + lccn + bankrupt | {FE}",
        "n32_herf": f"lodd ~ neg_ln_herf + bankrupt | {FE}",
        "n32_n": f"lodd ~ ln_n + bankrupt | {FE}",
    }
    ds = dfeols_dataset(d, a.work, a.target_rows) if a.dfeols else None
    head = f"\n{'model':9s} {'term':12s} {'pyfixest':>18s} {'paper':>16s}"
    print(head + (f" {'dfeols':>18s} {'rel b':>8s} {'rel se':>8s}" if ds else ""))
    for name, fml in models.items():
        m = pf.feols(fml, data=d, vcov=vc)
        t = m.tidy()
        td = dfeols_fit(ds, fml, name) if ds else None
        for term, (pb, pse) in PAPER[name].items():
            b, se = t.loc[term, "Estimate"], t.loc[term, "Std. Error"]
            line = f"{name:9s} {term:12s} {b:+9.4f} ({se:.4f}) {pb:+8.3f} ({pse:.3f})"
            if td is not None:
                db, dse = td.loc[term, "Estimate"], td.loc[term, "Std. Error"]
                line += f" {db:+9.4f} ({dse:.4f}) {abs(db - b) / abs(b):8.1e} {abs(dse - se) / se:8.1e}"
            print(line)
        iv = fml.split("|")[-1]
        if "~" in iv:
            endog = iv.split("~")[0].strip()
            ft = pf.feols(f"{endog} ~ {INSTRUMENTS} + bankrupt | {FE}", data=d, vcov=vc).tidy()
            print(
                f"{'':9s} first stage {endog}: "
                + ", ".join(f"{k} {r['Estimate']:+.4f} (t {r['t value']:.1f})" for k, r in ft.iterrows())
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
