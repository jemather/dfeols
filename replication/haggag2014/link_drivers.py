# SPDX-License-Identifier: GPL-3.0-or-later
"""D2: link Learning by Driving's random ``driver_id`` to the trip sheets' hashed ``hack_license``.

    python repo/replication/haggag2014/link_drivers.py
        [--shifts build/haggag2017/shift_analysis_stata.parquet]
        [--trips build/haggag2014/tripsheets] [--out build/haggag2014/driver_link.parquet]

Why: lbd_clean.do draws the experienced-driver samples with Stata's RNG (lines 453-460,
539-547), which Python can't reproduce. The published shift file carries the draw
(``exper_driver_group``) under its own ``driver_id``; this link carries it to the trips.

How: each published shift's ``sttime`` is the first pickup of the shift (after lbd_clean's
pickup/dropoff swap, lines 96-103, hence the min of the two times). Trips at those exact times
vote for a licence, one vote per (driver, shift). A driver links to the licence matching the
most shifts. The run fails unless every driver has a match, the winner matches at least
``--min-share`` of the driver's shifts and beats the runner-up, and no licence links twice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")
import polars as pl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shifts", default="build/haggag2017/shift_analysis_stata.parquet")
    ap.add_argument("--trips", default="build/haggag2014/tripsheets")
    ap.add_argument("--out", default="build/haggag2014/driver_link.parquet")
    ap.add_argument("--min-share", type=float, default=0.99)
    a = ap.parse_args(argv)
    pub = (pl.scan_parquet(a.shifts).select("driver_id", "shift", "sttime")
           .with_columns(t=pl.col("sttime").cast(pl.Datetime("us"))).collect())
    trips = (pl.scan_parquet(f"{a.trips}/*.parquet")
             .select("hack_license",
                     t=pl.min_horizontal("trip_pickup_datetime", "trip_dropoff_datetime"))
             .join(pub.lazy().select("t").unique(), on="t", how="semi").unique()
             .collect(engine="streaming"))
    votes = (pub.join(trips, on="t").select("driver_id", "shift", "hack_license").unique()
             .group_by("driver_id", "hack_license").len().sort("len", "hack_license",
                                                              descending=[True, False]))
    n = pub.group_by("driver_id").len().rename({"len": "shifts"})
    best = (votes.group_by("driver_id")
            .agg(pl.col("hack_license").first(), matched=pl.col("len").first(),
                 runner_up=pl.col("len").slice(1, 1).first().fill_null(0))
            .join(n, on="driver_id", how="right")
            .with_columns(share=pl.col("matched") / pl.col("shifts"))
            .sort("driver_id"))
    problems = {
        "unlinked": best.filter(pl.col("hack_license").is_null()).height,
        "below_min_share": best.filter(pl.col("share") < a.min_share).height,
        "tied": best.filter(pl.col("matched") <= pl.col("runner_up")).height,
        "licence_twice": best.height - best["hack_license"].n_unique(),
    }
    summary = {"drivers": best.height, "min_share": best["share"].min(),
               "max_runner_up_share": (best["runner_up"] / best["shifts"]).max(), **problems}
    print(json.dumps(summary))
    if any(problems.values()):
        print("link not unique; nothing written", file=sys.stderr)
        return 1
    out = Path(a.out)
    best.write_parquet(out)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
