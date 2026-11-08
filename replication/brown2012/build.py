# SPDX-License-Identifier: GPL-3.0-or-later
"""Brown & Laschever (2012) analysis file for the Farbmacher & Tauchmann (2021) Table 3 check.

Reads Data_Sixty_Four.dta from the openICPSR 113831 zip (16 MB; read in memory, as the
Parsons file was, Decisions 2026-10-23) and writes build/brown2012/sample.parquet: the
`non_mover_3yr == 1` sample of Tables_2_to_5.do column 7 (21,296 teacher-years), the
variables of that model, `agecat` (55 = reference; 66 = age_over_65) from the age dummies, and
`bad_age`: the two teachers F&T exclude because their age steps by -1 and +4 in one year
(empno 34920, 51920; 6 rows). See docs/brown2012.md.

    python build.py raw/brown2012/113831-V1.zip build/brown2012/sample.parquet
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import polars as pl

DTA = "AEJApp_2011_0132_data_and_programs/Data_Sixty_Four.dta"
KEEP = [
    "empno",
    "schoolcode",
    "ac_year",
    "retire",
    "PW_0_100k",
    "PK_is_positive",
    "salary_10k",
    "serv_sqr",
    "age_aug_31_o55_minusi",
    "service_o55_minusi",
    "pupilteachratio",
    "pct_maplus",
    "pct_feml",
    "qmathmeanss",
    "retire_o55_sizeminusi",
    "intage_aug_31",
]
AGE_DUMMIES = {**{f"age{a}": a for a in (52, 53, 54, *range(56, 66))}, "age_over_65": 66}


def build(zip_path: str) -> pl.DataFrame:
    raw = zipfile.ZipFile(zip_path).read(DTA)
    df = pd.read_stata(io.BytesIO(raw), convert_categoricals=False)
    s = df.loc[df["non_mover_3yr"] == 1, KEEP + list(AGE_DUMMIES)].copy()
    if (s[list(AGE_DUMMIES)].sum(axis=1) > 1).any():
        raise ValueError("more than one age dummy set on a row")
    s["agecat"] = 55
    for c, a in AGE_DUMMIES.items():
        s.loc[s[c] == 1, "agecat"] = a
    s = s.sort_values(["empno", "ac_year"])
    step = s.groupby("empno")[["agecat", "ac_year"]].diff()
    odd = (step["ac_year"] == 1) & (step["agecat"] != 1) & ~((s["agecat"] == 66) & (step["agecat"] == 0))
    bad = set(s.loc[odd, "empno"])
    s["bad_age"] = s["empno"].isin(bad).astype("int8")
    out = pl.from_pandas(s.drop(columns=list(AGE_DUMMIES))).with_columns(
        pl.col("empno").cast(pl.Int64),
        pl.col("schoolcode").cast(pl.Int64),
        pl.col("ac_year").cast(pl.Int32),
        pl.col("retire").cast(pl.Int8),
        pl.col("PK_is_positive").cast(pl.Float64),
        pl.col("agecat").cast(pl.Int32),
    )
    return out


if __name__ == "__main__":
    df = build(sys.argv[1])
    Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(sys.argv[2])
    print(df.height, "rows;", df["empno"].n_unique(), "teachers;", int(df["bad_age"].sum()), "bad_age rows")
