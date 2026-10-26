# SPDX-License-Identifier: GPL-3.0-or-later
"""Where each DB1B table-quarter and T-100 month lives in raw/db1b, and how its columns map
to the harmonised names (lower snake_case, decision 2026-10-26).

Three source layouts (docs/sources/db1b.md):

  form      raw/db1b/DB1B{Table}_{Y}_{Q}.zip, one TranStats form CSV, UPPER_SNAKE names
  split     raw/db1b/DB1B{Table}_p{1,2,3}/DB1B{Table}_{Y}_{Q}.zip, column parts sharing the keys
  prezip    raw/db1b/prezip/Origin_and_Destination_Survey_DB1B{Table}_{Y}_{Q}.zip, CamelCase

Coverage (every quarter 1993Q1-2025Q2 from exactly one layout; Coupon 1997Q1 is in both form
and split, identical, and form wins):

  Coupon  form 1993Q1-1997Q1   split 1997Q2-2000Q4   prezip 2001Q1+
  Market  form 1993Q1-1995Q4   split 1996Q1-1999Q4   prezip 2000Q1+
  Ticket  form 1993Q1-1999Q4                         prezip 2000Q1+

The Coupon form files have 24 of PREZIP's 36 columns; the other 12 are null before 2001
(decision 2026-10-26).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TABLES = ("Coupon", "Market", "Ticket")
FIRST, LAST = (1993, 1), (2025, 2)
SPLIT_KEYS = {"Coupon": ("ITIN_ID", "SEQ_NUM"), "Market": ("ITIN_ID", "MKT_ID")}
T100_PREFIX = "T-100 Domestic Segment  (All Carriers)"  # two spaces, as TranStats names it
T100_MEMBER = "T_T100D_SEGMENT_ALL_CARRIER.csv"

# PREZIP names whose snake_case form isn't the form file's name
PREZIP_RENAMES = {
    "OriginState": "origin_state_abr",
    "OriginStateName": "origin_state_nm",
    "DestState": "dest_state_abr",
    "DestStateName": "dest_state_nm",
    "Break": "trip_break",
    "TkCarrier": "ticket_carrier",
    "OpCarrier": "operating_carrier",
    "RPCarrier": "reporting_carrier",
    "MktCoupons": "market_coupons",
    "MktFare": "market_fare",
    "MktDistance": "market_distance",
    "MktDistanceGroup": "distance_group",
    "MktMilesFlown": "market_miles_flown",
    "NonStopMiles": "nonstop_miles",
    "TkCarrierChange": "tk_carrier_change",
    "TkCarrierGroup": "tk_carrier_group",
    "OpCarrierChange": "op_carrier_change",
    "OpCarrierGroup": "op_carrier_group",
    "FarePerMile": "itin_yield",
}


# Harmonised schema per table, in PREZIP column order (T-100 in form order). Types are what
# Polars infers on 1993 form and 2025 PREZIP files alike; the converter casts strictly, so a
# value that doesn't fit fails the quarter instead of turning into a null.
I, F, S = "Int64", "Float64", "String"
SCHEMA: dict[str, dict[str, str]] = {
    "Coupon": {
        "itin_id": I,
        "mkt_id": I,
        "seq_num": I,
        "coupons": I,
        "year": I,
        "origin_airport_id": I,
        "origin_airport_seq_id": I,
        "origin_city_market_id": I,
        "quarter": I,
        "origin": S,
        "origin_country": S,
        "origin_state_fips": I,
        "origin_state_abr": S,
        "origin_state_nm": S,
        "origin_wac": I,
        "dest_airport_id": I,
        "dest_airport_seq_id": I,
        "dest_city_market_id": I,
        "dest": S,
        "dest_country": S,
        "dest_state_fips": I,
        "dest_state_abr": S,
        "dest_state_nm": S,
        "dest_wac": I,
        "trip_break": S,
        "coupon_type": S,
        "ticket_carrier": S,
        "operating_carrier": S,
        "reporting_carrier": S,
        "passengers": F,
        "fare_class": S,
        "distance": F,
        "distance_group": I,
        "gateway": F,
        "itin_geo_type": I,
        "coupon_geo_type": I,
    },
    "Market": {
        "itin_id": I,
        "mkt_id": I,
        "market_coupons": I,
        "year": I,
        "quarter": I,
        "origin_airport_id": I,
        "origin_airport_seq_id": I,
        "origin_city_market_id": I,
        "origin": S,
        "origin_country": S,
        "origin_state_fips": I,
        "origin_state_abr": S,
        "origin_state_nm": S,
        "origin_wac": I,
        "dest_airport_id": I,
        "dest_airport_seq_id": I,
        "dest_city_market_id": I,
        "dest": S,
        "dest_country": S,
        "dest_state_fips": I,
        "dest_state_abr": S,
        "dest_state_nm": S,
        "dest_wac": I,
        "airport_group": S,
        "wac_group": S,
        "tk_carrier_change": F,
        "tk_carrier_group": S,
        "op_carrier_change": F,
        "op_carrier_group": S,
        "reporting_carrier": S,
        "ticket_carrier": S,
        "operating_carrier": S,
        "bulk_fare": F,
        "passengers": F,
        "market_fare": F,
        "market_distance": F,
        "distance_group": I,
        "market_miles_flown": F,
        "nonstop_miles": F,
        "itin_geo_type": I,
        "mkt_geo_type": I,
    },
    "Ticket": {
        "itin_id": I,
        "coupons": I,
        "year": I,
        "quarter": I,
        "origin": S,
        "origin_airport_id": I,
        "origin_airport_seq_id": I,
        "origin_city_market_id": I,
        "origin_country": S,
        "origin_state_fips": I,
        "origin_state_abr": S,
        "origin_state_nm": S,
        "origin_wac": I,
        "roundtrip": F,
        "online": F,
        "dollar_cred": I,
        "itin_yield": F,
        "reporting_carrier": S,
        "passengers": F,
        "itin_fare": F,
        "bulk_fare": F,
        "distance": F,
        "distance_group": I,
        "miles_flown": F,
        "itin_geo_type": I,
    },
    "T100": {
        "departures_scheduled": F,
        "departures_performed": F,
        "payload": F,
        "seats": F,
        "passengers": F,
        "freight": F,
        "mail": F,
        "distance": F,
        "ramp_to_ramp": F,
        "air_time": F,
        "unique_carrier": S,
        "airline_id": I,
        "unique_carrier_name": S,
        "unique_carrier_entity": S,
        "region": S,
        "carrier": S,
        "carrier_name": S,
        "carrier_group": I,
        "carrier_group_new": I,
        "origin_airport_id": I,
        "origin_airport_seq_id": I,
        "origin_city_market_id": I,
        "origin": S,
        "origin_city_name": S,
        "origin_state_abr": S,
        "origin_state_fips": I,
        "origin_state_nm": S,
        "origin_wac": I,
        "dest_airport_id": I,
        "dest_airport_seq_id": I,
        "dest_city_market_id": I,
        "dest": S,
        "dest_city_name": S,
        "dest_state_abr": S,
        "dest_state_fips": I,
        "dest_state_nm": S,
        "dest_wac": I,
        "aircraft_group": I,
        "aircraft_type": I,
        "aircraft_config": I,
        "year": I,
        "quarter": I,
        "month": I,
        "distance_group": I,
        "class": S,
    },
}


def snake(name: str) -> str:
    """Harmonised name for a raw column: form UPPER_SNAKE or PREZIP CamelCase."""
    if name in PREZIP_RENAMES:
        return PREZIP_RENAMES[name]
    if "_" in name or name.isupper():
        return name.lower()
    # ItinID -> itin_id, OriginAirportSeqID -> origin_airport_seq_id, OnLine -> on_line
    s = re.sub(r"ID$", "Id", name)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s).lower()
    return {"on_line": "online", "round_trip": "roundtrip"}.get(s, s)


@dataclass(frozen=True)
class Source:
    """One table-quarter: the zip(s) it comes from and how to read them."""

    table: str
    year: int
    quarter: int
    layout: str  # form | split | prezip
    zips: tuple[Path, ...]  # one, or one per column part (in part order)

    @property
    def label(self) -> str:
        return f"{self.table}_{self.year}Q{self.quarter}"


def quarters(first=FIRST, last=LAST):
    y, q = first
    while (y, q) <= last:
        yield y, q
        y, q = (y, q + 1) if q < 4 else (y + 1, 1)


def find(root: Path, table: str, year: int, quarter: int) -> Source:
    """The source for one table-quarter; raises if it is missing."""
    root = Path(root)
    stem = f"DB1B{table}_{year}_{quarter}.zip"
    if (root / stem).exists():
        return Source(table, year, quarter, "form", (root / stem,))
    parts = sorted(root.glob(f"DB1B{table}_p*/{stem}"), key=lambda p: int(p.parent.name.rsplit("p", 1)[1]))
    if parts:
        if len(parts) != 3:
            raise FileNotFoundError(f"{table} {year}Q{quarter}: {len(parts)} column parts, expected 3")
        return Source(table, year, quarter, "split", tuple(parts))
    pz = root / "prezip" / f"Origin_and_Destination_Survey_DB1B{table}_{year}_{quarter}.zip"
    if pz.exists():
        return Source(table, year, quarter, "prezip", (pz,))
    raise FileNotFoundError(f"{table} {year}Q{quarter}: no form, split or prezip zip under {root}")


def catalog(root: Path, tables=TABLES, first=FIRST, last=LAST) -> list[Source]:
    return [find(root, t, y, q) for t in tables for y, q in quarters(first, last)]


def t100_months(root: Path) -> list[tuple[int, int, Path]]:
    """(year, month, zip) for every T-100 Domestic Segment download, in date order."""
    pat = re.compile(re.escape(T100_PREFIX) + r"_(\d{4})_(\d{1,2})\.zip$")
    out = []
    for p in Path(root).glob(f"{T100_PREFIX}_*.zip"):
        m = pat.search(p.name)
        if m:
            out.append((int(m[1]), int(m[2]), p))
    return sorted(out)
