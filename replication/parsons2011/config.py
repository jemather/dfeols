# SPDX-License-Identifier: GPL-3.0-or-later
"""Parsons, Sulaeman, Yates & Hamermesh (2011), "Strike Three: Discrimination,
Incentives, and Evaluation," AER 101(4): 1410-35.

Replication data: openICPSR 112434 (doi:10.3886/E112434V1), file calledpitch.dta.

Design: linear probability model of a called strike on an umpire-pitcher
race/ethnicity match (UPM), count/inning/score controls, and absorbed pitcher,
umpire and batter effects (with QuesTec interactions in some tables).
Heteroskedasticity-robust SEs.

Variables and specs follow 20081366_statacode.do (openICPSR 112434); notes and line cites in
docs/parsons2011.md. PUBLISHED is empty until the AER tables are entered.

Most published estimates do not use exact fixed effects. Table 3 cols. 8-9 demean once by
umpire (then batter) and absorb the pitcher; Tables 4-5 demean once each by the QuesTec-specific
batter, umpire and pitcher ids, in that order, then run OLS. Both demean on the full file,
Asian pitchers included, and then regress on non-Asian pitchers. AUTHORS reproduces that
procedure (`harness.py authors`), so the data can be checked against the published tables;
dfeols, pyfixest and reghdfe all fit the exact FE model, which differs in the 4th decimal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DatasetSpec, Spec

TITLE = "Parsons et al. (2011), Strike Three: dfeols replication"

# --------------------------------------------------------------------------- 1. variables
# Standardized name -> column in calledpitch.dta. None means "not in the file":
# the harness derives it where it can (see DERIVED) or leaves it out.
VARMAP: dict[str, str | None] = {
    # outcome
    "strike": "callstrike",  # 1 = called strike, 0 = called ball (called pitches only)
    # identifiers (any type; strings are factorized)
    "umpire": "uid",         # home-plate umpire
    "pitcher": "pid",
    "batter": "bid",
    "catcher": None,         # not in the file (only ucm, the umpire-catcher match)
    "game": None,            # not in the file
    "year": None,            # not in the file
    # treatment and moderators
    "upm": "upm",            # == (u0race == prace) on every row
    "ump_race": "u0race",
    "pitcher_race": "prace", # 1 White, 2 Black, 3 Hispanic, 4 Asian
    "questec": "questec",
    "well_att": "att_dummy", # attendance > 69% of capacity (ReadMe); built by the authors
    "attendance": None,
    "capacity": None,
    "terminal": "term",      # == (3 balls or 2 strikes) on every row
    # controls
    "balls": None,
    "strikes": None,
    "count": None,           # from the cd* dummies, see DUMMY_SETS
    "inning": None,          # from the id* dummies, see DUMMY_SETS
    "top": "top",            # 1 = top of the inning
    "score_adv": "score",    # pitcher's score advantage
}

# Categorical controls rebuilt from the authors' dummy sets: {name: (base level, {column: level})}.
# count = 10*balls + strikes, base 0-0; inning base 1, 9 = ninth or later.
DUMMY_SETS = {
    "count": (0, {f"cd{b}{s}": 10 * b + s for b in range(4) for s in range(3) if (b, s) != (0, 0)}),
    "inning": (1, {f"id{i}": i for i in range(2, 10)}),
}

# Every pooled regression in the do-file is on non-Asian pitchers (prace<4 / prace~=4).
SAMPLE = "pitcher_race < 4"

# Only used when well_att is built from attendance / capacity, which this file doesn't need.
WELL_ATTENDED_SHARE = 0.70

REQUIRED = ["strike", "umpire", "pitcher", "batter", "upm", "questec", "count", "inning", "top",
            "score_adv"]
IDS = ["umpire", "pitcher", "batter", "catcher", "game"]

# --------------------------------------------------------------------------- 2. models
CONTROLS = "i(count) + i(inning) + top + score_adv"
QFE = "pitcher^questec + umpire^questec + batter^questec"  # the authors' qpid, quid, qbid

# Partition keys. dfeols handles an FE locally when it is nested in the key and
# as (sparse) dummies when it crosses shards. Sharding the same data two ways and
# getting the same answer is itself one of the checks.
DATASETS = [
    DatasetSpec("by_batter", "batter",
                ("batter", "pitcher", "umpire", "pitcher^questec", "umpire^questec", "batter^questec"),
                ("pitcher",)),
    DatasetSpec("by_pitcher", "pitcher", ("pitcher", "umpire", "batter")),
]

# Pooled columns only; the subsample columns (by race, QuesTec, attendance, count) are left out.
SPECS = [
    # `score- top` (line 37) is a range in the file's column order, which in the deposited file is
    # score term term_upm top: no count or inning dummies. Published R2 0.031 confirms it.
    Spec("T3c7", "strike ~ upm + score_adv + terminal + upm:terminal + top | pitcher", ("by_pitcher",),
         paper="Table 3, col. 7", note="line 37; areg, so exact: compare with the paper directly"),
    Spec("T3c8", f"strike ~ upm + {CONTROLS} | pitcher + umpire", ("by_pitcher",),
         paper="Table 3, col. 8", note="lines 41-48; paper demeans once by umpire"),
    Spec("T3c9", f"strike ~ upm + {CONTROLS} | pitcher + umpire + batter", ("by_batter", "by_pitcher"),
         paper="Table 3, col. 9", note="lines 51-59; run on two partitions: invariance check"),
    Spec("T4c3", f"strike ~ upm + upm:questec + {CONTROLS} | {QFE}", ("by_batter",),
         paper="Table 4, col. 3", note="lines 67-93"),
    Spec("T5c3", f"strike ~ upm + well_att + upm:well_att + {CONTROLS} | {QFE}", ("by_batter",),
         paper="Table 5, col. 3 (panel A)", note="line 108"),
    Spec("T5c6", f"strike ~ upm + upm:terminal + {CONTROLS} | {QFE}", ("by_batter",),
         paper="Table 5, col. 6 (panel B)", note="line 120; terminal itself is absorbed by i(count)"),
    Spec("T5c9", f"strike ~ upm + upm:questec + well_att + upm:well_att + upm:terminal + {CONTROLS}"
                 f" | {QFE}", ("by_batter",), paper="Table 5, col. 9 (panel C)", note="line 132"),
    # Not in the paper: CRV1 with clusters (pitchers) that cross the batter shards.
    Spec("T3c9_crv", f"strike ~ upm + {CONTROLS} | pitcher + umpire + batter", ("by_batter",),
         vcov=("CRV1", "hetero"), cluster="pitcher", paper="(extra check)"),
]

# The authors' procedure, on calledpitch.dta's own columns (see the module docstring).
# demean: one pass each, in order, on the full file; absorb: areg FE on the sample; sample: Stata if.
# terms maps the authors' regressor names to ours.
_T45 = dict(demean=("qbid", "quid", "qpid"), absorb=None, sample="prace != 4")
AUTHORS = {
    "T3c7": dict(x=["upm"], demean=(), absorb="pid", sample="prace < 4",
                 controls=["score", "term", "term_upm", "top"]),
    "T3c8": dict(x=["upm"], demean=("uid",), absorb="pid", sample="prace < 4"),
    "T3c9": dict(x=["upm"], demean=("uid", "bid"), absorb="pid", sample="prace < 4"),
    "T4c3": dict(x=["upm", "q_upm"], **_T45),
    "T5c3": dict(x=["upm", "att_dummy", "att_dummy_upm"], **_T45),
    "T5c6": dict(x=["upm", "term_upm"], **_T45),
    "T5c9": dict(x=["upm", "q_upm", "att_dummy", "att_dummy_upm", "term_upm"], **_T45),
}
AUTHORS_CONTROLS = ["score", *DUMMY_SETS["count"][1], *DUMMY_SETS["inning"][1], "top"]
AUTHORS_TERMS = {"q_upm": "upm:questec", "att_dummy": "well_att", "att_dummy_upm": "upm:well_att",
                 "term_upm": "upm:terminal"}

# --------------------------------------------------------------------------- 3. published
# {spec: {term: (coef, se, decimals)}}, AER 101(4) Tables 3-5. The target is the authors' code
# (AUTHORS); every published number is within 0.0001 of it, but the tables mix rounding and
# truncation, so expect last-digit INFO flags. Only T3c7 is an exact FE model; for the rest,
# dfeols (exact) and the paper (one-pass demeaning) differ by design.
PUBLISHED: dict[str, dict[str, tuple[float, float | None, int]]] = {
    "T3c7": {"upm": (0.0024, 0.0013, 4)},
    "T3c8": {"upm": (0.0021, 0.0017, 4)},
    "T3c9": {"upm": (0.0016, 0.0017, 4)},
    "T4c3": {"upm": (0.0059, 0.0022, 4), "upm:questec": (-0.0107, 0.0035, 4)},
    "T5c3": {"upm": (0.0036, 0.0019, 4), "well_att": (0.0059, 0.0012, 4),
             "upm:well_att": (-0.0037, 0.0015, 4)},
    "T5c6": {"upm": (0.0031, 0.0018, 4), "upm:terminal": (-0.0058, 0.0014, 4)},
    "T5c9": {"upm": (0.0089, 0.0024, 4), "upm:questec": (-0.0102, 0.0036, 4),
             "upm:well_att": (-0.0035, 0.0015, 4), "upm:terminal": (-0.0058, 0.0014, 4)},
}
PUBLISHED_SOURCE = "AER 101(4): 1410-35, Tables 3-5"
PUBLISHED_N: dict[str, int] = {k: 1_838_676 for k in PUBLISHED}
