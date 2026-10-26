# SPDX-License-Identifier: GPL-3.0-or-later
"""Check the converted DB1B stage (convert.py output) quarter by quarter.

    python repo/replication/db1b/check_stage.py [--stage build/db1b/stage] [--csv summary.csv]

Every quarter 1993Q1-2025Q2 must have all three tables, each with a sidecar, and the tables
must describe the same itineraries:

  ticket rows == distinct market itin_id == distinct coupon itin_id
  coupon rows == sum(ticket.coupons)
  ticket itin_id unique

T-100 must have every month from 1993-01 to the last DB1B quarter's end. Writes one row per
quarter to --csv (default <stage>/summary.csv) and exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S


def quarter_row(stage: Path, y: int, q: int) -> dict:
    row: dict = {"year": y, "quarter": q}
    paths = {t: stage / t.lower() / f"{t.lower()}_{y}Q{q}.parquet" for t in S.TABLES}
    missing = [t for t, p in paths.items() if not p.exists() or not p.with_suffix(".json").exists()]
    if missing:
        return row | {"ok": False, "problems": f"missing {missing}"}
    c, m, t = (pl.scan_parquet(paths[x]) for x in S.TABLES)
    ct = c.select(pl.len().alias("coupon_rows"), pl.col("itin_id").n_unique().alias("coupon_itins"))
    mt = m.select(pl.len().alias("market_rows"), pl.col("itin_id").n_unique().alias("market_itins"))
    tt = t.select(
        pl.len().alias("ticket_rows"),
        pl.col("itin_id").n_unique().alias("ticket_itins"),
        pl.col("coupons").sum().alias("ticket_coupons"),
        pl.col("passengers").sum().alias("ticket_passengers"),
    )
    for lf in (ct, mt, tt):
        row |= lf.collect(engine="streaming").row(0, named=True)
    problems = []
    if not row["ticket_rows"] == row["ticket_itins"]:
        problems.append("ticket itin_id not unique")
    if not row["ticket_rows"] == row["market_itins"] == row["coupon_itins"]:
        problems.append("itineraries differ across tables")
    if row["coupon_rows"] != row["ticket_coupons"]:
        problems.append("coupon rows != sum(ticket.coupons)")
    return row | {"ok": not problems, "problems": "; ".join(problems)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="build/db1b/stage", type=Path)
    ap.add_argument("--csv", default=None, type=Path)
    a = ap.parse_args(argv)

    rows = []
    for y, q in S.quarters():
        r = quarter_row(a.stage, y, q)
        rows.append(r)
        status = "ok" if r["ok"] else "FAIL " + r["problems"]
        print(f"{y}Q{q}: {status}  ticket rows {r.get('ticket_rows', 0):,}")
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(a.csv or a.stage / "summary.csv")

    want = {(y, m) for y, q in S.quarters() for m in range(3 * q - 2, 3 * q + 1)}
    have = {(int(p.stem[5:9]), int(p.stem[10:12])) for p in (a.stage / "t100").glob("t100_*.parquet")}
    t100_missing = sorted(want - have)
    bad = df.filter(~pl.col("ok"))
    shown = ", ".join(f"{y}-{m:02d}" for y, m in t100_missing[:12]) + (
        " ..." if len(t100_missing) > 12 else ""
    )
    print(f"\n{len(df)} quarters, {len(bad)} failed; T-100 months missing: {len(t100_missing)} {shown}")
    if len(df):
        print(
            f"total rows: ticket {df['ticket_rows'].sum():,}, market {df['market_rows'].sum():,}, "
            f"coupon {df['coupon_rows'].sum():,}"
        )
    return 1 if len(bad) or t100_missing else 0


if __name__ == "__main__":
    sys.exit(main())
