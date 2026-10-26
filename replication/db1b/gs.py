# SPDX-License-Identifier: GPL-3.0-or-later
"""Gerardi & Shapiro (2009) ticket rules and carrier-route-quarter panel from build/db1b/stage.

    python repo/replication/db1b/gs.py regionals         # derive the exempt operating carriers
    python repo/replication/db1b/gs.py collapse [--from 1993Q1] [--to 2006Q3]
    python repo/replication/db1b/gs.py steps [options]   # panel counts vs App. B
    python repo/replication/db1b/gs.py panel [options]   # final panel with Table 2 variables

Ticket rules (App. B, pp. 33-34; docs/db1b.md):

  direct      every market has one coupon, at most two markets (one-way or round trip)
  credible    dollar_cred == 1; itin_fare > 0; fare > $20 round trip, > $10 one-way
  coach       the first coupon's fare class is X or Y, or any class for single-cabin carriers
              (SINGLE_CABIN: Southwest and JetBlue code ~all fares F/G but have no first class)
  code-share  ticketing != operating carrier on any coupon drops the ticket (App. B). The
              `regionals` exemption (operating carriers selling < 0.5% of direct tickets) is kept
              as an option, `--cs-exempt regionals`; it only fit while Southwest was missing

Each kept ticket gives one directional observation per coupon (round trips two), priced at the
fare (one-way) or half the fare (round trip), with the coupon's ticketing carrier. `collapse`
writes one Parquet per quarter, carrier x origin x destination airport: passengers, records,
passenger-weighted Gini, P10, P90 and mean price. `steps` runs the panel steps and prints each
count next to the paper's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

COACH = ["X", "Y"]
# Carriers coding >= 90% of direct tickets F/G with no premium cabin, 1993-2006 (docs/db1b.md)
SINGLE_CABIN = ["WN", "B6"]
CS_EXEMPT_FROM = (1998, 1)
REGIONAL_WINDOW = ((1998, 3), (2006, 3))
REGIONAL_SHARE = 0.005
LEGACY = ["UA", "US", "DL", "AA", "AS", "TW", "CO", "NW", "HP"]  # G&S n. 4
# G&S n. 5 (Ito and Lee 2003), codes from T-100 carrier names: Air South, AirTran, ATA, Eastwind,
# Frontier, JetBlue, Kiwi, Morris, National, Pro Air, Reno, Southwest, Spirit, Sun Country (two
# codes), ValuJet, Vanguard, Western Pacific. Access Air: see ACCESS_AIR.
LCC = [
    "WV",
    "FL",
    "TZ",
    "W9",
    "F9",
    "B6",
    "KP",
    "KN",
    "N7",
    "P9",
    "QQ",
    "WN",
    "NK",
    "SY",
    "SYQ",
    "J7",
    "NJ",
    "W7",
]
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


def candidates(stage: Path, year: int, quarter: int) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """(tickets, coupons) for one quarter's direct, credible tickets above the fare floors."""
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
    coupons = C.join(tickets.select("itin_id"), on="itin_id").sort("itin_id", "seq_num")
    return tickets, coupons


def regionals(stage: Path, window=REGIONAL_WINDOW, share=REGIONAL_SHARE) -> list[str]:
    """Operating carriers of coach code-share coupons whose own share of direct-ticket sales
    (first coupon's ticketing carrier, pooled over `window`) is below `share`."""
    sales, ops = [], []
    for y, q in S.quarters(*window):
        _, coupons = candidates(stage, y, q)
        first = coupons.group_by("itin_id").agg(
            pl.col("ticket_carrier").first().alias("tk"), pl.col("fare_class").first().alias("fc")
        )
        sales.append(first.group_by("tk").len())
        cs = coupons.join(first.filter(pl.col("fc").is_in(COACH)).select("itin_id"), on="itin_id").filter(
            pl.col("ticket_carrier") != pl.col("operating_carrier")
        )
        ops.append(cs.select(pl.col("operating_carrier").alias("op")).unique())
    sold = pl.concat(pl.collect_all(sales)).group_by("tk").agg(pl.col("len").sum())
    sold = sold.with_columns(s=pl.col("len") / pl.col("len").sum())
    op = pl.concat(pl.collect_all(ops)).unique()
    out = op.join(sold, left_on="op", right_on="tk", how="left").filter(pl.col("s").fill_null(0) < share)
    return sorted(out["op"].to_list())


def observations(
    stage: Path, year: int, quarter: int, cs_exempt, exempt_from=CS_EXEMPT_FROM, single_cabin=SINGLE_CABIN
) -> pl.LazyFrame:
    """Directional observations from one quarter's kept tickets."""
    tickets, coupons = candidates(stage, year, quarter)
    exempt = cs_exempt if (year, quarter) >= exempt_from else []
    codeshare = (pl.col("ticket_carrier") != pl.col("operating_carrier")) & ~pl.col(
        "operating_carrier"
    ).is_in(exempt)
    keep = (
        coupons.group_by("itin_id")
        .agg(
            (
                pl.col("fare_class").first().is_in(COACH)
                | pl.col("ticket_carrier").first().is_in(single_cabin)
            ).alias("coach"),
            codeshare.any().alias("codeshare"),
        )
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
            "origin_city_market_id",
            "dest_city_market_id",
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
            pl.col("origin_city_market_id").first(),
            pl.col("dest_city_market_id").first(),
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


def variables(collapsed: pl.DataFrame, panel: pl.DataFrame, t100: pl.DataFrame) -> pl.DataFrame:
    """Table 2 variables (App. A) on the final panel.

    Route competition comes from every carrier on the directional airport pair in the collapsed
    DB1B data (before the T-100 merge and filters): HERF from passenger shares, N carriers.
    LEGN/LCCN count legacy and low-cost carriers on the directional city-market pair (n. 24).
    PASSRTE is T-100 passengers on the airport pair, all carriers.
    """
    t = ["year", "quarter"]
    route = collapsed.group_by(*t, *ROUTE).agg(
        ((pl.col("passengers") / pl.col("passengers").sum()) ** 2).sum().alias("herf"),
        pl.col("carrier").n_unique().alias("n_carriers"),
    )
    city = ["origin_city_market_id", "dest_city_market_id"]
    cities = collapsed.group_by(*t, *city).agg(
        pl.col("carrier").filter(pl.col("carrier").is_in(LEGACY)).n_unique().alias("legn"),
        pl.col("carrier").filter(pl.col("carrier").is_in(LCC)).n_unique().alias("lccn"),
    )
    passrte = t100.group_by(*t, *ROUTE).agg(pl.col("t100_passengers").sum().alias("passrte"))
    return (
        panel.join(route, on=[*t, *ROUTE], how="left")
        .join(cities, on=[*t, *city], how="left")
        .join(passrte, on=[*t, *ROUTE], how="left")
        .with_columns(
            lodd=(pl.col("gini") / (1 - pl.col("gini"))).log(),
            ln_p10=pl.col("p10").log(),
            ln_p90=pl.col("p90").log(),
            ln_herf=pl.col("herf").log(),
            ln_n=pl.col("n_carriers").log(),
            ln_passrte=pl.col("passrte").log(),
            time=pl.col("year") * 10 + pl.col("quarter"),
            route=pl.format("{}-{}", "origin_airport_id", "dest_airport_id"),
            carrier_route=pl.format("{}:{}-{}", "carrier", "origin_airport_id", "dest_airport_id"),
        )
    )


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
    ap.add_argument("command", choices=["regionals", "collapse", "steps", "panel"])
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
        default="none",
        help="operating carriers exempt from the code-share rule: 'regionals' (derived), 'none', or a list",
    )
    ap.add_argument("--cs-exempt-from", default="1998Q1", type=parse_quarter)
    ap.add_argument(
        "--single-cabin", default=",".join(SINGLE_CABIN), help="carriers whose every fare is coach; 'none'"
    )
    ap.add_argument("--merge", default="operators", choices=["ticketing", "operators"])
    ap.add_argument("--direction", default="fixed", choices=["per_quarter", "fixed"])
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    if a.command == "regionals":
        r = regionals(a.stage)
        (a.out / "regionals.json").write_text(json.dumps(r) + "\n")
        print(f"{len(r)} regional operating carriers: {','.join(r)}")
        return 0

    if a.command == "collapse":
        if a.cs_exempt == "regionals":
            exempt = json.loads((a.out / "regionals.json").read_text())
        else:
            exempt = [] if a.cs_exempt == "none" else a.cs_exempt.split(",")
        (a.out / "collapsed").mkdir(exist_ok=True)
        for y, q in S.quarters(a.first, a.last):
            single = [] if a.single_cabin == "none" else a.single_cabin.split(",")
            obs = observations(a.stage, y, q, exempt, a.cs_exempt_from, single)
            df = collapse(obs).collect(engine="streaming")
            df.write_parquet(a.out / "collapsed" / f"gs_{y}Q{q}.parquet")
            print(
                f"{y}Q{q}: {df.height:,} carrier-routes, {df['passengers'].sum():,.0f} passengers", flush=True
            )
        return 0

    collapsed = pl.read_parquet(a.out / "collapsed" / "gs_*.parquet")
    yq = pl.col("year") * 10 + pl.col("quarter")
    collapsed = collapsed.filter(yq.is_between(a.first[0] * 10 + a.first[1], a.last[0] * 10 + a.last[1]))
    classes = a.t100_class.split(",") if a.t100_class else None
    t100 = t100_quarters(a.stage, a.t100_carrier, classes).collect(engine="streaming")
    r = steps(collapsed, t100, a.min_passengers, mean_over=a.mean_over, merge=a.merge, direction=a.direction)
    print(f"T-100 carrier {a.t100_carrier}, classes {classes or 'all'}, means over {a.mean_over},", end=" ")
    print(f"merge {a.merge}, direction {a.direction}")
    print_counts(r["counts"])
    if a.command == "panel":
        out = variables(collapsed, r["panel"], t100)
        out.write_parquet(a.out / "panel.parquet")
        print(f"wrote {a.out / 'panel.parquet'}: {out.height:,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
