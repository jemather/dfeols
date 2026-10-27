# SPDX-License-Identifier: GPL-3.0-or-later
"""The two ticket-level D4 datasets (STATE.md 2026-10-26), one Parquet file per quarter.

    python repo/replication/db1b/tickets.py tickets [--from 1993Q1] [--to 2025Q2]
    python repo/replication/db1b/tickets.py flags   [--from 1993Q1] [--to 2025Q2]

tickets  G&S-filtered direct coach tickets: one row per directional observation (`gs.observations`,
         the rows `collapse` aggregates into the panel), priced at the fare (one-way) or half the
         fare (round trip). Route competition (HERF, N, LEGN, LCCN) as in the panel, from the
         collapsed data of the same quarter; the bankruptcy dummy; `in_panel` marks observations
         whose carrier-route-quarter cell is in the final panel.
flags    every DB1B Market row with each G&S ticket rule as a 0/1 column (`direct`, `credible`,
         `fare_floor`, `coach`, `codeshare`) and `gs_keep`, their conjunction. A direct ticket's
         markets are one coupon each, so the `gs_keep` rows are the `tickets` rows one for one.

Both read build/db1b/stage and the collapsed files and panel from `gs.py collapse` / `panel`
(--gs, default build/db1b/gs_2025). `route_id` = origin_airport_id * 100000 + dest_airport_id;
FEs such as carrier^route_id are left to the estimator.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gs
import sources as S
from table2 import bankrupt_quarters

T = ["year", "quarter"]
CITY = ["origin_city_market_id", "dest_city_market_id"]


def route_id(o: str = "origin_airport_id", d: str = "dest_airport_id") -> pl.Expr:
    return (pl.col(o).cast(pl.Int64) * 100_000 + pl.col(d)).alias("route_id")


def quarter_cols(year: int, quarter: int) -> list[pl.Expr]:
    return [
        pl.lit(year, pl.Int16).alias("year"),
        pl.lit(quarter, pl.Int8).alias("quarter"),
        pl.lit(year * 10 + quarter, pl.Int32).alias("time"),
    ]


def bankrupt(year: int, quarter: int, carrier: str) -> pl.LazyFrame:
    b = bankrupt_quarters().filter(pl.col("year") == year, pl.col("quarter") == quarter)
    return b.select(pl.col("carrier").alias(carrier), "bankrupt").lazy()


def tickets(stage: Path, gsdir: Path, panel: pl.DataFrame, year: int, quarter: int) -> pl.LazyFrame:
    collapsed = pl.read_parquet(gsdir / "collapsed" / f"gs_{year}Q{quarter}.parquet").with_columns(
        pl.lit(year).alias("year"), pl.lit(quarter).alias("quarter")
    )
    route, cities = gs.competition(collapsed)
    cells = (
        panel.filter(pl.col("year") == year, pl.col("quarter") == quarter)
        .select("carrier", *gs.ROUTE, pl.lit(1, pl.Int8).alias("in_panel"))
        .lazy()
    )
    obs = gs.observations(stage, year, quarter, [], single_cabin=gs.SINGLE_CABIN)
    return (
        obs.join(route.drop(T).lazy(), on=gs.ROUTE, how="left")
        .join(cities.drop(T).lazy(), on=CITY, how="left")
        .join(cells, on=["carrier", *gs.ROUTE], how="left")
        .join(bankrupt(year, quarter, "carrier"), on="carrier", how="left")
        .select(
            *quarter_cols(year, quarter),
            "itin_id",
            "carrier",
            pl.col("op").alias("operating_carrier"),
            "origin_airport_id",
            "dest_airport_id",
            *CITY,
            route_id(),
            pl.col("roundtrip").cast(pl.Int8),
            "price",
            pl.col("price").log().alias("ln_price"),
            "passengers",
            "herf",
            pl.col("herf").log().alias("ln_herf"),
            pl.col("n_carriers").cast(pl.Int16),
            pl.col("n_carriers").log().alias("ln_n"),
            pl.col("legn").cast(pl.Int16),
            pl.col("lccn").cast(pl.Int16),
            pl.col("bankrupt").fill_null(0).cast(pl.Int8),
            pl.col("in_panel").fill_null(0),
        )
    )


def flags(stage: Path, year: int, quarter: int) -> pl.LazyFrame:
    Tk = pl.scan_parquet(stage / f"ticket/ticket_{year}Q{quarter}.parquet")
    M = pl.scan_parquet(stage / f"market/market_{year}Q{quarter}.parquet")
    C = pl.scan_parquet(stage / f"coupon/coupon_{year}Q{quarter}.parquet")
    single = [c for c, last in gs.SINGLE_CABIN.items() if (year, quarter) <= last]
    by_itin = M.group_by("itin_id").agg(
        ((pl.len() <= 2) & (pl.col("market_coupons") == 1).all()).alias("direct")
    )
    by_coupons = (
        C.sort("itin_id", "seq_num")
        .group_by("itin_id")
        .agg(
            (
                pl.col("fare_class").first().is_in(gs.COACH) | pl.col("ticket_carrier").first().is_in(single)
            ).alias("coach"),
            (pl.col("ticket_carrier") != pl.col("operating_carrier")).any().alias("codeshare"),
        )
    )
    floor = (
        pl.when(pl.col("roundtrip") == 1).then(pl.col("itin_fare") > 20).otherwise(pl.col("itin_fare") > 10)
    )
    tk = Tk.select(
        "itin_id",
        "roundtrip",
        ((pl.col("dollar_cred") == 1) & (pl.col("itin_fare") > 0)).alias("credible"),
        floor.alias("fare_floor"),
    )
    rules = ["direct", "credible", "fare_floor", "coach", "codeshare"]
    yq = year * 10 + quarter
    lcc = pl.col("ticket_carrier").is_in(gs.LCC)
    for c, (y, q) in gs.LCC_FROM.items():
        if yq >= y * 10 + q:
            lcc = lcc | (pl.col("ticket_carrier") == c)
    return (
        M.join(by_itin, on="itin_id", how="left")
        .join(by_coupons, on="itin_id", how="left")
        .join(tk, on="itin_id", how="left")
        .join(bankrupt(year, quarter, "ticket_carrier"), on="ticket_carrier", how="left")
        .select(
            *quarter_cols(year, quarter),
            "itin_id",
            "mkt_id",
            "ticket_carrier",
            "operating_carrier",
            "origin_airport_id",
            "dest_airport_id",
            *CITY,
            route_id(),
            pl.col("market_coupons").cast(pl.Int8),
            pl.col("roundtrip").cast(pl.Int8),
            "passengers",
            "market_fare",
            pl.when(pl.col("market_fare") > 0).then(pl.col("market_fare").log()).alias("ln_market_fare"),
            "market_distance",
            "nonstop_miles",
            pl.col("ticket_carrier").is_in(gs.LEGACY).cast(pl.Int8).alias("legacy"),
            lcc.cast(pl.Int8).alias("lcc"),
            pl.col("bankrupt").fill_null(0).cast(pl.Int8),
            *[pl.col(r).fill_null(False).cast(pl.Int8) for r in rules],
            (pl.all_horizontal(rules[:4]) & ~pl.col("codeshare"))
            .fill_null(False)
            .cast(pl.Int8)
            .alias("gs_keep"),
        )
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["tickets", "flags"])
    ap.add_argument("--from", dest="first", default="1993Q1", type=gs.parse_quarter)
    ap.add_argument("--to", dest="last", default="2025Q2", type=gs.parse_quarter)
    ap.add_argument("--stage", default="build/db1b/stage", type=Path)
    ap.add_argument("--gs", default="build/db1b/gs_2025", type=Path, help="collapsed files and panel")
    ap.add_argument("--out", default=None, type=Path, help="default build/db1b/<command>")
    a = ap.parse_args(argv)
    out = a.out or Path("build/db1b") / a.command
    out.mkdir(parents=True, exist_ok=True)
    panel = pl.read_parquet(a.gs / "panel.parquet") if a.command == "tickets" else None
    for y, q in S.quarters(a.first, a.last):
        t0 = time.time()
        lf = tickets(a.stage, a.gs, panel, y, q) if a.command == "tickets" else flags(a.stage, y, q)
        path = out / f"{a.command}_{y}Q{q}.parquet"
        lf.sink_parquet(path)
        n = pl.scan_parquet(path).select(pl.len()).collect().item()
        print(f"{y}Q{q}: {n:,} rows, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
