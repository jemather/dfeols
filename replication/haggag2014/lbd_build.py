# SPDX-License-Identifier: GPL-3.0-or-later
"""D2: port of Learning by Driving's ``lbd_clean.do`` (Haggag, McManus & Paci 2017), core
sections, from the 2009 trip sheets.

    python repo/replication/haggag2014/lbd_build.py [--trips build/haggag2014/tripsheets]
        [--out build/haggag2014] [--workers 4] [--buckets 0123456789ABCDEF]

Run from the project root, after ``convert_tripsheets.py`` and ``link_drivers.py``. Cited
lines are lbd_clean.do's (``docs/haggag2014.md``). Decisions 2026-11-08:

- Short shifts: lines 185 and 250 drop shifts under 120 minutes (the comment says 2 minutes).
  This build drops shifts under 2 minutes and flags those under 120 (``short_shift_120``), so
  both readings can be compared. Problem shifts (line 233) keep the do-file's definition.
- Experienced-driver samples (lines 453-460, 539-547, Stata RNG): taken from the published
  shift file (``exper_driver_group``, ``new_driver``) through ``driver_link.parquet``. Our own
  new and incumbent rules (lines 418-437) are computed alongside, for comparison.

Ported from section 9 so far: total experience at ride 3 (``end_exp_all``, Table 3 col 4).
Not ported yet: the difficulty and location-specific experience measures (sections 7 and 9,
and section 12's variables built from them).

Ties: Stata's ``sort`` orders ties arbitrarily; here ties in pickup time are broken by
dropoff time, then file order.

Pass 1, per bucket (first hex digit of the hashed licence; every step up to section 6 is
within a driver): sections 1-5 and the section 11 hour-after-ride-3 measures. Writes
``<out>/lbd_trips/bucket=X.parquet`` (cleaned trips) and ``<out>/lbd_shifts/bucket=X.parquet``.
Pass 2: section 4's crosswalk merge is per trip and done in pass 1; pass 2 renumbers shifts
(line 354, a global ``egen group``), applies the driver samples and section 12, and writes
``<out>/lbd_shift_analysis.parquet``.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import logging
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")
import numpy as np
import polars as pl

log = logging.getLogger("haggag2014.lbd_build")
D2_ZIP = "raw/haggag2017/113636-V1.zip"
TIGER = "build/haggag2014/extract/tl_2011_*/*.shp"
MIN = 60_000_000  # microseconds per minute
TIE_DESC = os.environ.get("LBD_TIE", "desc") == "desc"
DEDUP = [
    "car_id",
    "driver_id",
    "distance",
    "pkp_long",
    "pkp_lat",
    "drf_long",
    "drf_lat",
    "fare_original",
    "surcharge",
    "mta_tax",
    "tip",
    "toll",
    "total_amount",
    "payment_type",
    "pkp_time_tmp",
    "drf_time_tmp",
    "vendor_name",
    "passenger_count",
    "trip_time",
]


# ---------------------------------------------------------------- crosswalks (section 4)


def _zip_csv(member_suffix: str, **kw) -> pl.DataFrame:
    with zipfile.ZipFile(D2_ZIP) as z:
        name = next(n for n in z.namelist() if n.endswith(member_suffix))
        return pl.read_csv(io.BytesIO(z.read(name)), infer_schema=False, **kw)


def crosswalk() -> pl.DataFrame:
    """fips -> puma, nta, boro, as lines 264-325 build pkp_fips.dta."""
    g = _zip_csv("geocorr12.csv").slice(1)  # row 2 holds the labels
    g = (
        g.with_columns(
            fips=pl.col("county") + pl.col("tract").str.replace(".", "", literal=True),
            afact=pl.col("afact").str.strip_chars().cast(pl.Float64),
        )
        .sort(["fips", "afact"], descending=[False, True])
        .unique("fips", keep="first", maintain_order=True)  # gsort fips -afact; duplicates drop
        .select("fips", puma=pl.col("puma2k"))
    )
    n = _zip_csv("nyc2010census_tabulation_equiv.csv")
    n = n.rename(
        {
            "2010 Census Bureau FIPS County Code": "county",
            "2010 NYC Borough Code": "boro_code",
            "2010 Census Tract": "tract",
            "PUMA": "puma_n",
            "NTA Code": "ntacode",
            "Borough": "boro",
        }
    )
    n = n.with_columns(
        pumanta="0" + pl.col("puma_n"),
        fips="36" + pl.col("county").str.zfill(3) + pl.col("tract").str.zfill(6),
        nta=pl.col("boro_code") + pl.col("ntacode").str.slice(2, 3),
    )
    spans = n.group_by("nta").agg(pl.col("pumanta").n_unique().alias("k"))
    n = (
        n.join(spans, on="nta")
        .with_columns(
            nta=pl.when(pl.col("k") > 1)
            .then(pl.col("nta") + "." + pl.col("pumanta"))
            .otherwise(pl.col("nta"))
        )
        .select("fips", "pumanta", "nta", "boro")
    )
    x = n.join(g, on="fips", how="full", coalesce=True)  # old-style merge keeps both sides
    x = x.with_columns(
        puma=pl.when(pl.col("puma").is_null() | (pl.col("fips") == "36085022800"))
        .then(pl.col("pumanta"))
        .otherwise(pl.col("puma"))
    ).drop("pumanta")
    if x["fips"].n_unique() != x.height:
        raise ValueError("crosswalk fips not unique")
    return x


class Tracts:
    def __init__(self):
        import pyogrio
        import shapely

        geo, ids = [], []
        for f in sorted(glob.glob(TIGER)):
            r = pyogrio.raw.read(f, columns=["GEOID"])
            geo.append(shapely.from_wkb(r[2]))
            ids.append(r[3][0])
        self.ids = np.concatenate(ids).astype(str)
        self.tree = shapely.STRtree(np.concatenate(geo))
        self.shapely = shapely

    def __call__(self, lon: np.ndarray, lat: np.ndarray) -> pl.Series:
        ok = np.isfinite(lon) & np.isfinite(lat)
        pts = self.shapely.points(np.where(ok, lon, 0.0), np.where(ok, lat, 0.0))
        pi, gi = self.tree.query(pts, predicate="intersects")
        o = np.lexsort((gi, pi))
        pi, gi = pi[o], gi[o]
        first = np.r_[True, pi[1:] != pi[:-1]] if len(pi) else np.zeros(0, bool)
        out = np.full(len(lon), None, object)
        out[pi[first]] = self.ids[gi[first]]
        out[~ok] = None
        return pl.Series(out, dtype=pl.String)


def add_geo(df: pl.DataFrame, tracts: Tracts, xw: pl.DataFrame) -> pl.DataFrame:
    for side, lo, la in (("pkp", "pkp_long", "pkp_lat"), ("drf", "drf_long", "drf_lat")):
        u = df.select(lo, la).unique()
        u = u.with_columns(tracts(u[lo].to_numpy(), u[la].to_numpy()).alias(f"{side}_fips"))
        df = df.join(u, on=[lo, la], how="left", nulls_equal=True)
        df = df.join(xw.rename({c: f"{side}_{c}" for c in xw.columns}), on=f"{side}_fips", how="left")
    return df


# ---------------------------------------------------------------- pass 1


def bucket_trips(trips: str, b: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(f"{trips}/*.parquet")
        .filter(pl.col("hack_license").str.starts_with(b))
        .with_row_index("row")
        .select(
            "row",
            car_id="medallion",
            driver_id="hack_license",
            vendor_name="vendor_name",
            passenger_count="passenger_count",
            trip_time="trip_time",
            distance="trip_distance",
            pkp_time_tmp="trip_pickup_datetime",
            drf_time_tmp="trip_dropoff_datetime",
            pkp_long="start_lon",
            pkp_lat="start_lat",
            drf_long="end_lon",
            drf_lat="end_lat",
            payment_type=pl.col("payment_type").replace({"CREDIT": "Credit", "CASH": "Cash"}),
            fare_original="fare_amt",
            surcharge="surcharge",
            mta_tax="mta_tax",
            tip="tip_amt",
            toll="tolls_amt",
            total_amount="total_amt",
        )
        .collect()
    )


def build_bucket(trips: str, b: str, out: str, commit: str) -> dict:
    t0 = time.time()
    df = bucket_trips(trips, b)
    n_raw = df.height
    # section 1: duplicates (87), flips (96-103), sequencing (105-112)
    df = df.unique(DEDUP, keep="first", maintain_order=True)
    n_dedup = df.height
    p, d = pl.col("pkp_time_tmp"), pl.col("drf_time_tmp")
    df = df.with_columns(pkp_time=pl.min_horizontal(p, d), drf_time=pl.max_horizontal(p, d), flipped=p > d)
    # line 107 (sort driver_id pkp_time) leaves ties in pickup time unordered; TIE_DESC puts the
    # later drop-off first, so the sequencing fix (line 111) truncates that trip
    df = df.sort("driver_id", "pkp_time", "drf_time", "row", descending=[False, False, TIE_DESC, False])
    nxt = pl.col("pkp_time").shift(-1).over("driver_id")
    df = df.with_columns(seq_fix=(pl.col("drf_time") > nxt).fill_null(False))
    df = df.with_columns(drf_time=pl.when("seq_fix").then(nxt).otherwise("drf_time"))
    df = df.with_columns(
        ride_duration=(pl.col("drf_time") - pl.col("pkp_time")).dt.total_microseconds() / MIN,
        mta_tax=pl.col("mta_tax").fill_null(0.0),
    )
    df = df.with_columns(
        surcharge_error=pl.col("surcharge") > 1,
        mta_tax_error=pl.col("mta_tax") > 0.5,
        below_minfare=pl.col("fare_original") < 2.5,
        fare=pl.when(pl.col("payment_type") == "No Charge")
        .then(None)
        .otherwise(pl.col("fare_original") + pl.col("surcharge")),
        # 133-138: the date comes from the raw time string, the hour from the adjusted time
        pkp_date=p.dt.date(),
        drf_date=d.dt.date(),
        pkp_hour=pl.col("pkp_time").dt.hour().cast(pl.Int8),
        drf_hour=pl.col("drf_time").dt.hour().cast(pl.Int8),
    )
    df = df.with_columns(
        pkp_dow=(pl.col("pkp_date").dt.weekday() % 7).cast(pl.Int8),
        drf_dow=(pl.col("drf_date").dt.weekday() % 7).cast(pl.Int8),
    )
    # section 2: shifts (145-187)
    df = df.sort("driver_id", "pkp_time", "drf_time", "row")  # line 145
    df = df.with_columns(
        duration_outlier=(pl.col("ride_duration") > 180) | (pl.col("ride_duration") <= 0),
        distance_outlier=(pl.col("distance") > 100) | (pl.col("distance") <= 0),
        gap=(pl.col("pkp_time") - pl.col("drf_time").shift(1).over("driver_id")).dt.total_microseconds()
        / MIN,
    )
    df = df.with_columns(
        shift=(1 + (pl.col("gap") > 300).fill_null(False).cast(pl.Int32).cum_sum().over("driver_id")).cast(
            pl.Int32
        )
    ).drop("gap")
    s = ["driver_id", "shift"]
    df = df.with_columns(
        shift_total_mins=(pl.col("drf_time").last() - pl.col("pkp_time").first())
        .over(s)
        .dt.total_microseconds()
        / MIN,
        trip_number=pl.int_range(1, pl.len() + 1).over(s).cast(pl.Int32),
        num_trips=pl.len().over(s).cast(pl.Int32),
        shift_numcars=pl.col("car_id").n_unique().over(s),
    )
    df = df.with_columns(
        final_trip=pl.col("trip_number") == pl.col("num_trips"),
        multiple_cars_per_shift=pl.col("shift_numcars") > 1,
        long_shifts_outlier=pl.col("shift_total_mins") > 1200,
        short_shift_120=pl.col("shift_total_mins") < 120,
        short_shift_2=pl.col("shift_total_mins") < 2,
        driver_ridecount=pl.len().over("driver_id"),
    )
    # section 3: drops (227-253)
    df = df.filter(pl.col("driver_ridecount") >= 100)
    tag = pl.col("trip_number") == 1
    df = df.with_columns(
        driver_problemshifts=(
            tag
            & (pl.col("long_shifts_outlier") | pl.col("short_shift_120") | pl.col("multiple_cars_per_shift"))
        )
        .sum()
        .over("driver_id")
        .cast(pl.Int32)
    )
    counts = {"rows_raw": n_raw, "rows_dedup": n_dedup, "rows_100rides": df.height}
    df = df.filter(
        ~pl.col("multiple_cars_per_shift"), ~pl.col("long_shifts_outlier"), ~pl.col("short_shift_2")
    )
    counts["rows_kept"] = df.height
    counts["rows_short_120"] = int(df["short_shift_120"].sum())
    # section 4: tracts and crosswalk
    df = add_geo(df, Tracts(), crosswalk())
    # section 5 (357-395), ordered by trip_number within shift
    df = df.sort("driver_id", "shift", "trip_number")
    sttime = pl.col("pkp_time").first().over(s)
    df = df.with_columns(sttime=sttime, entime=pl.col("drf_time").last().over(s))
    df = df.with_columns(
        st=(pl.col("pkp_time") - pl.col("sttime")).dt.total_microseconds() / MIN,
        en=(pl.col("drf_time") - pl.col("sttime")).dt.total_microseconds() / MIN,
    )
    df = df.with_columns(slacktime=(pl.col("st").shift(-1).over(s) - pl.col("en")).fill_null(0.0))
    df = df.with_columns(
        brk=pl.col("slacktime") >= 60,
        shift_hours=pl.col("en").last().over(s) / 60,
        shift_fare=pl.col("fare").sum().over(s),
        shift_miles=pl.col("distance").sum().over(s),
        shift_ride_duration=pl.col("ride_duration").sum().over(s),
        shift_slacktime=pl.col("slacktime").sum().over(s),
        shift_rides=pl.len().over(s).cast(pl.Int32),
    )
    df = df.with_columns(
        shift_miles_perride=pl.col("shift_miles") / pl.col("shift_rides"),
        shift_speed_perride=pl.col("shift_miles") / (pl.col("shift_ride_duration") / 60),
    )
    # section 11 (811-901): the 60 minutes after ride 3's drop-off
    e3 = pl.col("en").filter(pl.col("trip_number") == 3).first().over(s)
    df = df.with_columns(en_trip=e3, miles=pl.col("distance"), rides=pl.lit(1.0))
    st_, en_, et = pl.col("st"), pl.col("en"), pl.col("en_trip")
    inside = (st_ < et + 60) & (st_ >= et)
    for x in ("fare", "miles", "rides", "ride_duration"):
        a = (
            pl.when(inside & (en_ <= et + 60))
            .then(pl.col(x))
            .when(inside & (en_ > et + 60))
            .then(pl.col(x) * ((et + 60) - st_) / (en_ - st_))
            .otherwise(0.0)
        )
        df = df.with_columns(a.sum().over(s).fill_null(0.0).alias(f"{x}_hour"))
    st_next = pl.col("st").shift(-1).over(s)
    has_next = st_next.is_not_null()
    sl = pl.col("slacktime")
    a = (
        pl.when((en_ >= et) & (st_next < et + 60) & has_next)
        .then(sl)
        .when((en_ >= et) & (en_ < et + 60) & (st_next >= et + 60) & has_next & (sl != 0))
        .then(sl * (((et + 60) - en_) / sl))
        .otherwise(0.0)
    )
    df = df.with_columns(_tmp=a.fill_null(0.0).sum().over(s))
    df = df.with_columns(
        _tmp=pl.when((pl.col("_tmp") == 0) & (en_ < et) & (st_next >= et + 60) & has_next & (sl >= 60))
        .then(60.0)
        .otherwise("_tmp")
    )
    df = df.with_columns(slacktime_hour=pl.col("_tmp").max().over(s)).drop("_tmp")
    df = df.with_columns(
        miles_perride_hour=pl.col("miles_hour") / pl.col("rides_hour"),
        speed_perride_hour=pl.col("miles_hour") / (pl.col("ride_duration_hour") / 60),
    )
    r3 = {
        "start_time": "pkp_time",
        "start_puma": "pkp_puma",
        "start_nta": "pkp_nta",
        "start_tract": "pkp_fips",
        "end_time": "drf_time",
        "end_puma": "drf_puma",
        "end_nta": "drf_nta",
        "end_tract": "drf_fips",
        "end_fare": "fare",
        "end_slacktime": "slacktime",
        "end_en": "en",
    }
    r4 = {"start_nta4": "pkp_nta", "start_puma4": "pkp_puma", "start_time4": "pkp_time"}
    first_of = lambda c, k: pl.col(c).filter(pl.col("trip_number") == k).first().over(s)
    df = df.with_columns(
        *[first_of(c, 3).alias(n) for n, c in r3.items()], *[first_of(c, 4).alias(n) for n, c in r4.items()]
    )
    o = Path(out)
    (o / "lbd_trips").mkdir(parents=True, exist_ok=True)
    (o / "lbd_shifts").mkdir(parents=True, exist_ok=True)
    trip_cols = [
        c
        for c in df.columns
        if c not in ("row", "miles", "rides", "shift_numcars", "driver_ridecount", "num_trips", *r3, *r4)
    ]
    df.select(trip_cols).write_parquet(o / "lbd_trips" / f"bucket={b}.parquet", compression="zstd")
    shift_cols = [
        "driver_id",
        "car_id",
        "shift",
        "sttime",
        "entime",
        "shift_hours",
        "shift_fare",
        "shift_miles",
        "shift_ride_duration",
        "shift_slacktime",
        "shift_rides",
        "shift_miles_perride",
        "shift_speed_perride",
        "short_shift_120",
        "driver_problemshifts",
        "fare_hour",
        "miles_hour",
        "rides_hour",
        "ride_duration_hour",
        "slacktime_hour",
        "miles_perride_hour",
        "speed_perride_hour",
        *r3,
        *r4,
    ]
    sh = df.filter(pl.col("trip_number") == 1).select(shift_cols)
    sh.write_parquet(o / "lbd_shifts" / f"bucket={b}.parquet", compression="zstd")
    return {
        "bucket": b,
        **counts,
        "shifts": sh.height,
        "drivers": sh["driver_id"].n_unique(),
        "seconds": round(time.time() - t0, 1),
        "build_commit": commit,
    }


# ---------------------------------------------------------------- pass 2


def finish(out: Path, link_path: str, pub_path: str, commit: str) -> dict:
    sh = pl.read_parquet(out / "lbd_shifts" / "*.parquet")
    # line 354: egen shift=group(shift1) is a global dense rank of the shift numbers
    sh = sh.with_columns(shift=pl.col("shift").rank("dense").cast(pl.Int32))
    sh = sh.with_columns(
        start_date=pl.col("sttime").dt.date(), start_hour=pl.col("sttime").dt.hour().cast(pl.Int8)
    )
    # section 6, our own rules (402-437), for comparison with the published samples
    apr1, jan15 = date(2009, 4, 1), date(2009, 1, 15)
    drv = sh.group_by("driver_id").agg(
        entry_date=pl.col("start_date").min(),
        driver_shifts=pl.len(),
        pre=(pl.col("start_date") < apr1).sum(),
        post=(pl.col("start_date") >= apr1).sum(),
    )
    drv = drv.with_columns(
        new_driver_ours=(pl.col("entry_date") >= apr1) & (pl.col("post") >= 50),
        incumb_ours=(pl.col("entry_date") <= jan15)
        & (pl.col("driver_shifts") >= 100)
        & (pl.col("pre") >= 30),
    )
    sh = sh.join(drv.select("driver_id", "entry_date", "new_driver_ours", "incumb_ours"), on="driver_id")
    sh = sh.with_columns(
        start_dow=(pl.col("start_date").dt.weekday() % 7).cast(pl.Int8),
        start_month=pl.col("start_date").dt.month().cast(pl.Int8),
    )
    sh = sh.with_columns(entry_month=pl.col("start_month").min().over("driver_id"))
    # line 595: experience at ride 3's drop-off = the driver's earlier trips (earlier shifts' rides
    # plus rides 1-2). end_exp_all counts only shifts the code keeps (120-minute rule);
    # end_exp_all_keep2 also counts the 2-120 minute shifts. Missing without a ride 3 (line 893).
    sh = sh.sort("driver_id", "shift")
    kept = (~pl.col("short_shift_120")).cast(pl.Int64)
    sh = sh.with_columns(
        end_exp_all=pl.when(pl.col("end_time").is_not_null() & ~pl.col("short_shift_120")).then(
            (pl.col("shift_rides") * kept).cum_sum().shift(1).fill_null(0).over("driver_id") + 2
        ),
        end_exp_all_keep2=pl.when(pl.col("end_time").is_not_null()).then(
            pl.col("shift_rides").cast(pl.Int64).cum_sum().shift(1).fill_null(0).over("driver_id") + 2
        ),
    )
    # published samples via the link
    link = pl.read_parquet(link_path).select(pub_driver_id="driver_id", driver_id="hack_license")
    pubd = (
        pl.scan_parquet(pub_path)
        .group_by("driver_id")
        .agg(pl.col("new_driver").first(), pl.col("exper_driver_group").first())
        .rename({"driver_id": "pub_driver_id"})
        .collect()
    )
    sh = sh.join(link, on="driver_id", how="left").join(pubd, on="pub_driver_id", how="left")
    comp = {
        "drivers_all": drv.height,
        "ours_new": int(drv["new_driver_ours"].sum()),
        "ours_incumbent": int(drv["incumb_ours"].sum()),
        "published_drivers_found": sh["pub_driver_id"].drop_nulls().n_unique(),
    }
    pd_ = sh.filter(pl.col("pub_driver_id").is_not_null()).unique("driver_id")
    comp["published_new_but_ours_not"] = pd_.filter(
        (pl.col("new_driver") == 1) & ~pl.col("new_driver_ours")
    ).height
    comp["published_exp_but_ours_not_incumbent"] = pd_.filter(
        (pl.col("new_driver") == 0) & ~pl.col("incumb_ours")
    ).height
    comp["published_new_with_problemshifts_over5"] = pd_.filter(
        (pl.col("new_driver") == 1) & (pl.col("driver_problemshifts") > 5)
    ).height
    # section 12 on the published sample (1024-1145, core variables)
    a = sh.filter(pl.col("pub_driver_id").is_not_null(), pl.col("start_month") >= 4)
    a = a.with_columns(
        log_shift=pl.col("shift").log(),
        log_shift_hours=pl.col("shift_hours").log(),
        avg_earn=pl.col("shift_fare") / pl.col("shift_hours"),
        share_idle=100 * pl.col("shift_slacktime") / (pl.col("shift_hours") * 60),
    )
    ok_hr = pl.col("shift_hours") >= (pl.col("end_en") + 60) / 60
    a = a.with_columns(
        log_new_shift=pl.col("log_shift") * pl.col("new_driver"),
        log_all_exper=pl.col("end_exp_all").log(),
        log_new_all_exper=pl.col("end_exp_all").log() * pl.col("new_driver"),
        log_new_all_exper_keep2=pl.col("end_exp_all_keep2").log() * pl.col("new_driver"),
        log_avg_earn=pl.col("avg_earn").log(),
        log_hr_fare=pl.when(ok_hr & (pl.col("fare_hour") > 0)).then(pl.col("fare_hour").log()),
        log_hr_fare_ib=pl.when(ok_hr).then((1 + pl.col("fare_hour")).log()),
        share_hr_idle=pl.when(ok_hr).then(100 * pl.col("slacktime_hour") / 60),
        date_r3=pl.col("end_time").dt.date(),
        hour_r3=pl.col("end_time").dt.hour().cast(pl.Int8),
    )

    def grp(*c):  # egen group(): missing if any part is missing; labels are arbitrary
        key = pl.concat_str([pl.col(x).cast(pl.String) for x in c], separator="|")
        return (
            pl.when(pl.all_horizontal([pl.col(x).is_not_null() for x in c]))
            .then(key)
            .rank("dense")
            .cast(pl.Int32)
        )

    a = a.with_columns(
        date_hour=grp("date_r3", "hour_r3"),
        date_hour_startnta=grp("date_r3", "hour_r3", "start_nta"),
        date_hour_startpuma=grp("date_r3", "hour_r3", "start_puma"),
        date_hour_endnta=grp("date_r3", "hour_r3", "end_nta"),
    )
    a = a.sort("pub_driver_id", "shift")
    a.write_parquet(out / "lbd_shift_analysis.parquet", compression="zstd")
    comp.update(
        analysis_rows=a.height,
        analysis_rows_without_short120=a.filter(~pl.col("short_shift_120")).height,
        build_commit=commit,
    )
    return comp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trips", default="build/haggag2014/tripsheets")
    ap.add_argument("--out", default="build/haggag2014")
    ap.add_argument("--link", default="build/haggag2014/driver_link.parquet")
    ap.add_argument("--published", default="build/haggag2017/shift_analysis_stata.parquet")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--buckets", default="0123456789ABCDEF")
    ap.add_argument("--skip-pass1", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    commit = (
        subprocess.run(
            ["git", "-C", "repo", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )
    out = Path(a.out)
    failed = 0
    stats = []
    if not a.skip_pass1:
        log.info("pass 1: %d buckets, %d workers, commit %s", len(a.buckets), a.workers, commit)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(build_bucket, a.trips, b, str(out), commit): b for b in a.buckets}
            for f in as_completed(futs):
                try:
                    s = f.result()
                    stats.append(s)
                    log.info("bucket %s: %s", futs[f], json.dumps(s))
                except Exception as e:
                    failed += 1
                    log.exception("FAILED bucket %s: %s", futs[f], e)
        if failed:
            return 1
        (out / "lbd_shifts" / "_pass1.json").write_text(
            json.dumps(sorted(stats, key=lambda s: s["bucket"]), indent=1)
        )
    comp = finish(out, a.link, a.published, commit)
    (out / "lbd_shift_analysis.json").write_text(json.dumps(comp, indent=1))
    log.info("pass 2: %s", json.dumps(comp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
