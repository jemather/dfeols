# SPDX-License-Identifier: GPL-3.0-or-later
"""db1b_gs_tickets_v2: the v1 ticket rows plus the columns for the post-2006 specs (STATE.md 2026-10-28).

    python repo/replication/db1b/tickets_v2.py [--tickets build/db1b/tickets] [--out build/db1b/tickets_v2]

Adds, per row of build/db1b/tickets (one Parquet file per quarter, rows and order unchanged):

neg_ln_herf      -ln_herf
post, period     post = from 2006Q4; period 1 1993-99, 2 2000-2006Q3, 3 2006Q4-2012, 4 2013-19, 5 2020-25
est              the carrier-route-quarter cell is in the Table 2 estimation sample
                 (`table2.estimation_sample`)
ln_passrte, gensp, irutherf, irutherf_pp
                 Table 2 instruments for est cells (null otherwise); irutherf from the first stage on
                 the whole sample, irutherf_pp from first stages fit separately before and after 2006Q4
tail             0 price <= the cell's p10, 2 price >= its p90, 1 between (panel cells; null otherwise).
                 p10/p90 are the panel's passenger-weighted cutoffs, so ties put >= 10% in each tail
hubsh, hub       the carrier's largest share of T-100 passengers enplaned at either endpoint;
                 hub = hubsh >= .5
ln_lf            ln T-100 load factor on the airport pair, all carriers
big_city, leisure
                 both endpoints big-city airports (App. C Table C3); an endpoint leisure (Table C2)
w_*              passenger weights, zero outside a subsample (dfeols drops zero weights): w_est (est),
                 w_pre / w_post, w_top / w_bot (tail 2 / 0), w_top_pre, w_top_post, w_bot_pre, w_bot_post
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gs
import table2 as T

HERE = Path(__file__).resolve().parent
KEY = gs.KEY
POST = 20064  # 2006Q4, the first extension quarter
PERIODS = [(19931, 1), (20001, 2), (20064, 3), (20131, 4), (20201, 5)]  # first quarter of each


def cells(panel: pl.DataFrame) -> pl.DataFrame:
    """Instruments for the estimation sample, pooled and per period (pre/post 2006Q4)."""
    import pyfixest as pf

    warnings.simplefilter("ignore")
    d = T.estimation_sample(panel)
    pooled = d.to_pandas()
    T.add_irutherf(pooled, pf)
    out = pl.from_pandas(pooled[[*KEY, "ln_passrte", "gensp", "irutherf"]])
    parts = []
    for f in [pl.col("time") < POST, pl.col("time") >= POST]:
        s = d.filter(f).filter(pl.len().over("carrier_route") > 1).to_pandas()
        T.add_irutherf(s, pf)
        parts.append(pl.from_pandas(s[[*KEY, "irutherf"]]).rename({"irutherf": "irutherf_pp"}))
    out = out.join(pl.concat(parts), on=KEY, how="left").with_columns(pl.lit(1, pl.Int8).alias("est"))
    return out.with_columns(pl.col("year", "quarter").cast(pl.Int32))


def hubs(t100: pl.DataFrame) -> pl.DataFrame:
    """Each carrier's share of T-100 passengers enplaned at each airport and quarter."""
    t = ["year", "quarter"]
    e = t100.group_by(*t, "carrier", pl.col("origin_airport_id").alias("airport")).agg(
        pl.col("t100_passengers").sum().alias("e")
    )
    return e.select(
        *t, "carrier", "airport", (pl.col("e") / pl.col("e").sum().over(*t, "airport")).alias("sh")
    )


def airport_flags(panel: pl.DataFrame) -> tuple[list[int], list[int]]:
    """Big-city and leisure airport IDs, from App. C codes via the panel's ID-code pairs."""
    c2 = pl.read_csv(HERE / "table_c2.csv", comment_prefix="#")
    c3 = pl.read_csv(HERE / "table_c3.csv", comment_prefix="#")
    ids = pl.concat(
        [
            panel.select(pl.col("origin_airport_id").alias("id"), pl.col("origin").alias("code")),
            panel.select(pl.col("dest_airport_id").alias("id"), pl.col("dest").alias("code")),
        ]
    ).unique()
    leisure = ids.filter(pl.col("code").is_in(c2.filter(leisure="Yes")["airport"].to_list()))["id"].to_list()
    big = ids.filter(pl.col("code").is_in(c3.filter(big_city="Yes")["airport"].unique().to_list()))[
        "id"
    ].to_list()
    return big, leisure


def period() -> pl.Expr:
    e = pl.lit(PERIODS[0][1], pl.Int8)
    for first, k in PERIODS[1:]:
        e = pl.when(pl.col("time") >= first).then(pl.lit(k, pl.Int8)).otherwise(e)
    return e.alias("period")


def weights() -> list[pl.Expr]:
    w = pl.col("passengers") * pl.col("est").fill_null(0)
    pre, post = pl.col("post") == 0, pl.col("post") == 1
    top, bot = pl.col("tail") == 2, pl.col("tail") == 0
    subs = {
        "w_est": pl.lit(True),
        "w_pre": pre,
        "w_post": post,
        "w_top": top,
        "w_bot": bot,
        "w_top_pre": top & pre,
        "w_top_post": top & post,
        "w_bot_pre": bot & pre,
        "w_bot_post": bot & post,
    }
    return [(w * m.fill_null(False).cast(pl.Float64)).alias(k) for k, m in subs.items()]


def quarter(
    path: Path, inst: pl.DataFrame, panel: pl.DataFrame, sh: pl.DataFrame, lf: pl.DataFrame, big, leisure
):
    t = pl.scan_parquet(path)
    y, q = t.select(pl.col("year", "quarter").first()).collect().row(0)
    yq = [pl.col("year") == y, pl.col("quarter") == q]
    cut = panel.filter(*yq).select(*KEY, "p10", "p90").lazy()
    shq = sh.filter(*yq).drop("year", "quarter").lazy()
    lfq = lf.filter(*yq).drop("year", "quarter").lazy()
    instq = inst.filter(*yq).drop("year", "quarter").lazy()
    cr = ["carrier", *gs.ROUTE]
    tail = (
        pl.when(pl.col("price") <= pl.col("p10"))
        .then(0)
        .when(pl.col("price") >= pl.col("p90"))
        .then(2)
        .otherwise(1)
    )
    return (
        t.join(instq, on=cr, how="left")
        .join(cut.drop("year", "quarter"), on=cr, how="left")
        .join(
            shq.rename({"airport": "origin_airport_id", "sh": "sh_o"}),
            on=["carrier", "origin_airport_id"],
            how="left",
        )
        .join(
            shq.rename({"airport": "dest_airport_id", "sh": "sh_d"}),
            on=["carrier", "dest_airport_id"],
            how="left",
        )
        .join(lfq, on=gs.ROUTE, how="left")
        .with_columns(
            (-pl.col("ln_herf")).alias("neg_ln_herf"),
            (pl.col("time") >= POST).cast(pl.Int8).alias("post"),
            period(),
            pl.col("est").fill_null(0).cast(pl.Int8),
            pl.when(pl.col("p10").is_not_null()).then(tail).cast(pl.Int8).alias("tail"),
            pl.max_horizontal(pl.col("sh_o").fill_null(0), pl.col("sh_d").fill_null(0)).alias("hubsh"),
            # T-100 reports zero seats on some pairs (2001 most of all): null, not inf/NaN
            pl.when(pl.col("lf").is_finite() & (pl.col("lf") > 0)).then(pl.col("lf").log()).alias("ln_lf"),
            (pl.col("origin_airport_id").is_in(big) & pl.col("dest_airport_id").is_in(big))
            .cast(pl.Int8)
            .alias("big_city"),
            (pl.col("origin_airport_id").is_in(leisure) | pl.col("dest_airport_id").is_in(leisure))
            .cast(pl.Int8)
            .alias("leisure"),
        )
        .with_columns((pl.col("hubsh") >= 0.5).cast(pl.Int8).alias("hub"), *weights())
        .drop("p10", "p90", "sh_o", "sh_d", "lf")
    ), (y, q)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickets", default="build/db1b/tickets", type=Path)
    ap.add_argument("--gs", default="build/db1b/gs_2025", type=Path)
    ap.add_argument("--stage", default="build/db1b/stage", type=Path)
    ap.add_argument("--out", default="build/db1b/tickets_v2", type=Path)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    panel = pl.read_parquet(a.gs / "panel.parquet")
    inst = cells(panel)
    print(f"estimation cells {inst.height:,}", flush=True)
    t100 = gs.t100_quarters(a.stage).collect().with_columns(pl.col("year", "quarter").cast(pl.Int32))
    sh = hubs(t100)
    lf = t100.group_by("year", "quarter", *gs.ROUTE).agg(
        (pl.col("t100_passengers").sum() / pl.col("t100_seats").sum()).alias("lf")
    )
    big, leisure = airport_flags(panel)
    panel = panel.with_columns(pl.col("year", "quarter").cast(pl.Int32))
    for f in sorted(a.tickets.glob("tickets_*.parquet")):
        t0 = time.time()
        lf_q, (y, q) = quarter(f, inst, panel, sh, lf, big, leisure)
        path = a.out / f"tickets_v2_{y}Q{q}.parquet"
        lf_q.sink_parquet(path)
        n = pl.scan_parquet(path).select(pl.len()).collect().item()
        print(f"{y}Q{q}: {n:,} rows, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
