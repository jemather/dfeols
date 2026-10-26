# SPDX-License-Identifier: GPL-3.0-or-later
"""Gerardi & Shapiro (2009) ticket rules and carrier-route-quarter panel from build/db1b/stage.

    python repo/replication/db1b/gs.py collapse [--from 1993Q1] [--to 2006Q3]
    python repo/replication/db1b/gs.py steps [options]   # panel counts vs App. B

Ticket rules (App. B, pp. 33-34; docs/db1b.md):

  direct      every market has one coupon, at most two markets (one-way or round trip)
  credible    dollar_cred == 1; itin_fare > 0; fare > $20 round trip, > $10 one-way
  coach       the first coupon's fare class is X or Y (STATE.md 2026-10-26)
  code-share  ticketing != operating carrier on any coupon drops the ticket, except from
              1998Q3 when the operating carrier is in CS_EXEMPT (fitted to Table B1, not unique)

Each kept ticket gives one directional observation per coupon (round trips two), priced at the
fare (one-way) or half the fare (round trip), with the coupon's ticketing carrier. `collapse`
writes one Parquet per quarter, carrier x origin x destination airport: passengers, records,
passenger-weighted Gini, P10, P90 and mean price. `steps` runs the panel steps and prints each
count next to the paper's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

COACH = ["X", "Y"]
CS_EXEMPT_FROM = (1998, 3)
CS_EXEMPT = ["MQ", "RU", "EV", "QX", "XJ", "12", "DH", "9E", "AL", "9N", "S5", "UA", "XE"]
KEY = ["year", "quarter", "carrier", "origin_airport_id", "dest_airport_id"]
ROUTE = ["origin_airport_id", "dest_airport_id"]

# App. B counts, 1993Q1-2006Q3
PAPER = {
    "collapsed": 606_015,
    "merged": 274_821,
    "merged_routes": 8_850,
    "filtered": 222_261,
    "one_direction": 112_499,
    "one_direction_carrier_routes": 5_444,
    "one_direction_routes": 2_902,
}


def observations(
    stage: Path, year: int, quarter: int, cs_exempt=CS_EXEMPT, exempt_from=CS_EXEMPT_FROM
) -> pl.LazyFrame:
    """Directional observations from one quarter's kept tickets."""
    T = pl.scan_parquet(stage / f"ticket/ticket_{year}Q{quarter}.parquet")
    M = pl.scan_parquet(stage / f"market/market_{year}Q{quarter}.parquet")
    C = pl.scan_parquet(stage / f"coupon/coupon_{year}Q{quarter}.parquet")
    direct = (
        M.group_by("itin_id")
        .agg(pl.len().alias("n_markets"), (pl.col("market_coupons") == 1).all().alias("direct"))
        .filter(pl.col("direct") & (pl.col("n_markets") <= 2))
        .select("itin_id")
    )
    floor = (
        pl.when(pl.col("roundtrip") == 1).then(pl.col("itin_fare") > 20).otherwise(pl.col("itin_fare") > 10)
    )
    tickets = T.join(direct, on="itin_id").filter(pl.col("dollar_cred") == 1, pl.col("itin_fare") > 0, floor)
    exempt = cs_exempt if (year, quarter) >= exempt_from else []
    codeshare = (pl.col("ticket_carrier") != pl.col("operating_carrier")) & ~pl.col(
        "operating_carrier"
    ).is_in(exempt)
    coupons = C.join(tickets.select("itin_id"), on="itin_id").sort("itin_id", "seq_num")
    keep = (
        coupons.group_by("itin_id")
        .agg(pl.col("fare_class").first().is_in(COACH).alias("coach"), codeshare.any().alias("codeshare"))
        .filter(pl.col("coach") & ~pl.col("codeshare"))
        .select("itin_id")
    )
    return (
        coupons.join(keep, on="itin_id")
        .join(tickets.select("itin_id", "roundtrip", "itin_fare"), on="itin_id")
        .select(
            pl.lit(year).alias("year"),
            pl.lit(quarter).alias("quarter"),
            pl.col("ticket_carrier").alias("carrier"),
            pl.col("operating_carrier").alias("op"),
            "origin_airport_id",
            "dest_airport_id",
            "origin",
            "dest",
            "itin_id",
            "roundtrip",
            pl.when(pl.col("roundtrip") == 1)
            .then(pl.col("itin_fare") / 2)
            .otherwise(pl.col("itin_fare"))
            .alias("price"),
            "passengers",
        )
    )


def collapse(obs: pl.LazyFrame) -> pl.LazyFrame:
    """Carrier-route-quarter statistics, weighting each record by its passengers."""
    w, x = pl.col("passengers"), pl.col("price")
    cw, W = w.cum_sum(), w.sum()
    wx = w * x
    return (
        obs.sort([*KEY, "price"])
        .group_by(KEY, maintain_order=True)
        .agg(
            pl.col("origin").first(),
            pl.col("dest").first(),
            pl.col("op").unique().sort().alias("operators"),
            pl.len().alias("records"),
            W.alias("passengers"),
            (wx.sum() / W).alias("price_mean"),
            x.filter(cw >= 0.1 * W).first().alias("p10"),
            x.filter(cw >= 0.5 * W).first().alias("p50"),
            x.filter(cw >= 0.9 * W).first().alias("p90"),
            (1 - (w * (2 * (wx.cum_sum() - wx) + wx)).sum() / (W * wx.sum())).alias("gini"),
        )
    )


def t100_quarters(stage: Path, carrier_field: str = "carrier", classes=None) -> pl.LazyFrame:
    """T-100 Domestic Segment summed to carrier x origin x destination airport x quarter."""
    lf = pl.scan_parquet(stage / "t100/t100_*.parquet")
    if classes:
        lf = lf.filter(pl.col("class").is_in(classes))
    return lf.group_by(
        pl.col(carrier_field).alias("carrier"), "origin_airport_id", "dest_airport_id", "year", "quarter"
    ).agg(
        pl.col("passengers").sum().alias("t100_passengers"),
        pl.col("departures_performed").sum().alias("t100_departures"),
        pl.col("seats").sum().alias("t100_seats"),
    )


def merge_t100(panel: pl.DataFrame, t100: pl.DataFrame, how: str = "ticketing") -> pl.DataFrame:
    """Inner merge with T-100 quarters.

    ticketing  match the observation's (ticketing) carrier
    operators  match every carrier that operated its tickets (the ticketing carrier, or an exempt
               regional for code-shares kept from 1998Q3) and sum their segments
    """
    if how == "ticketing":
        return panel.join(t100, on=KEY, how="inner")
    t = (
        panel.select(*KEY, "operators")
        .explode("operators")
        .join(t100.rename({"carrier": "operators"}), on=["year", "quarter", "operators", *ROUTE], how="inner")
    )
    t = t.group_by(KEY).agg(pl.col("t100_passengers", "t100_departures", "t100_seats").sum())
    return panel.join(t, on=KEY, how="inner")


def steps(
    panel: pl.DataFrame,
    t100: pl.DataFrame,
    min_passengers: int = 100,
    coverage: float = 0.25,
    mean_over: str = "merged",
    merge: str = "ticketing",
    direction: str = "per_quarter",
) -> dict:
    """App. B panel steps; returns the frame after each and the counts.

    mean_over  'merged': carrier-route means over the merged sample; 'after_min': over the sample
               left after the minimum-passenger rule (the paper doesn't say which).
    """
    out = {"collapsed": panel.height}
    m = merge_t100(panel, t100, merge)
    out |= {"merged": m.height, "merged_routes": m.select(ROUTE).n_unique()}
    cr = ["carrier", *ROUTE]
    base = m.filter(pl.col("passengers") >= min_passengers) if mean_over == "after_min" else m
    means = base.group_by(cr).agg(
        pl.col("passengers").mean().alias("mean_db1b"), pl.col("t100_passengers").mean().alias("mean_t100")
    )
    f = (
        m.filter(pl.col("passengers") >= min_passengers)
        .join(means, on=cr)
        .with_columns(
            low_db1b=pl.col("passengers") < coverage * pl.col("mean_db1b"),
            low_t100=pl.col("t100_passengers") < coverage * pl.col("mean_t100"),
        )
        .filter(pl.col("low_db1b") == pl.col("low_t100"))
    )
    out["filtered"] = f.height
    if direction == "fixed":
        # one direction per airport pair throughout: origin id below destination id
        o = f.filter(pl.col("origin_airport_id") < pl.col("dest_airport_id"))
    else:
        # one direction per carrier, airport pair and quarter: the lower origin id when both exist
        pair = [pl.min_horizontal(*ROUTE).alias("a"), pl.max_horizontal(*ROUTE).alias("b")]
        o = (
            f.with_columns(*pair)
            .sort("origin_airport_id")
            .group_by("year", "quarter", "carrier", "a", "b", maintain_order=True)
            .first()
            .drop("a", "b")
        )
    out |= {
        "one_direction": o.height,
        "one_direction_carrier_routes": o.select(cr).n_unique(),
        "one_direction_routes": o.select(ROUTE).n_unique(),
    }
    return {"counts": out, "merged": m, "filtered": f, "panel": o}


def print_counts(counts: dict) -> None:
    for k, v in counts.items():
        p = PAPER.get(k)
        rel = f"{(v - p) / p:+.2%}" if p else ""
        print(f"  {k:30s} {v:>10,}  paper {p:>10,}  {rel}" if p else f"  {k:30s} {v:>10,}")


def parse_quarter(s: str) -> tuple[int, int]:
    y, q = s.upper().split("Q")
    return int(y), int(q)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["collapse", "steps"])
    ap.add_argument("--from", dest="first", default="1993Q1", type=parse_quarter)
    ap.add_argument("--to", dest="last", default="2006Q3", type=parse_quarter)
    ap.add_argument("--stage", default="build/db1b/stage", type=Path)
    ap.add_argument("--out", default="build/db1b/gs", type=Path)
    ap.add_argument("--t100-carrier", default="carrier", choices=["carrier", "unique_carrier"])
    ap.add_argument("--t100-class", default=None, help="comma-separated T-100 service classes, e.g. F")
    ap.add_argument("--min-passengers", default=100, type=int)
    ap.add_argument("--mean-over", default="merged", choices=["merged", "after_min"])
    ap.add_argument(
        "--cs-exempt",
        default=",".join(CS_EXEMPT),
        help="operating carriers exempt from the code-share rule from 1998Q3; 'none' for none",
    )
    ap.add_argument("--cs-exempt-from", default="1998Q3", type=parse_quarter)
    ap.add_argument("--merge", default="ticketing", choices=["ticketing", "operators"])
    ap.add_argument("--direction", default="per_quarter", choices=["per_quarter", "fixed"])
    a = ap.parse_args(argv)

    if a.command == "collapse":
        (a.out / "collapsed").mkdir(parents=True, exist_ok=True)
        for y, q in S.quarters(a.first, a.last):
            exempt = [] if a.cs_exempt == "none" else a.cs_exempt.split(",")
            df = collapse(observations(a.stage, y, q, exempt, a.cs_exempt_from)).collect(engine="streaming")
            df.write_parquet(a.out / "collapsed" / f"gs_{y}Q{q}.parquet")
            print(
                f"{y}Q{q}: {df.height:,} carrier-routes, {df['passengers'].sum():,.0f} passengers", flush=True
            )
        return 0

    panel = pl.read_parquet(a.out / "collapsed" / "gs_*.parquet")
    yq = pl.col("year") * 10 + pl.col("quarter")
    panel = panel.filter(yq.is_between(a.first[0] * 10 + a.first[1], a.last[0] * 10 + a.last[1]))
    classes = a.t100_class.split(",") if a.t100_class else None
    t100 = t100_quarters(a.stage, a.t100_carrier, classes).collect(engine="streaming")
    r = steps(panel, t100, a.min_passengers, mean_over=a.mean_over, merge=a.merge, direction=a.direction)
    print(f"T-100 carrier {a.t100_carrier}, classes {classes or 'all'}, means over {a.mean_over},", end=" ")
    print(f"merge {a.merge}, direction {a.direction}")
    print_counts(r["counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
