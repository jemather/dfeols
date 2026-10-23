# SPDX-License-Identifier: GPL-3.0-or-later
"""Parsons, Sulaeman, Yates & Hamermesh (2011), "Strike Three: Discrimination,
Incentives, and Evaluation," AER 101(4): 1410-35.

Replication data: openICPSR 112434 (doi:10.3886/E112434V1), file calledpitch.dta.

Design: linear probability model of a called strike on an umpire-pitcher
race/ethnicity match (UPM), count/inning/score controls, and absorbed pitcher,
umpire and batter effects (with QuesTec interactions in some tables).
Heteroskedasticity-robust SEs.

What to edit here:
  1. VARMAP: the authors' column names (run `harness.py inspect` first).
  2. SPECS: check each regressor list and FE set against 20081366_statacode.do.
  3. PUBLISHED: replace the working-paper values with the AER tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DatasetSpec, Spec  # noqa: E402

TITLE = "Parsons et al. (2011), Strike Three: dfeols replication"

# --------------------------------------------------------------------------- 1. variables
# Standardized name -> column in calledpitch.dta. None means "not in the file":
# the harness derives it where it can (see DERIVED) or leaves it out.
VARMAP: dict[str, str | None] = {
    # outcome
    "strike": None,        # 1 = called strike, 0 = called ball (called pitches only)
    # identifiers (any type; strings are factorized)
    "umpire": None,        # home-plate umpire
    "pitcher": None,
    "batter": None,
    "catcher": None,       # optional
    "game": None,          # optional: enables the CRV1-by-game check
    "year": None,          # optional: enables `build --years`
    # treatment and moderators
    "upm": None,           # umpire-pitcher match; or set ump_race + pitcher_race
    "ump_race": None,
    "pitcher_race": None,
    "questec": None,       # 1 = QuesTec park
    "well_att": None,      # 1 = well-attended game; or set attendance + capacity
    "attendance": None,
    "capacity": None,
    "terminal": None,      # 1 = two strikes or three balls; derived from balls/strikes if None
    # controls
    "balls": None,
    "strikes": None,
    "count": None,         # categorical count; derived as 10*balls + strikes if None
    "inning": None,
    "top": None,           # 1 = top of the inning
    "score_adv": None,     # pitching team's score advantage
}

# Threshold for the well-attended indicator when it is built from attendance /
# capacity. The working paper splits at roughly 69-70% of capacity; use the
# authors' definition from the do-file.
WELL_ATTENDED_SHARE = 0.70

REQUIRED = ["strike", "umpire", "pitcher", "batter", "upm", "questec", "count", "inning", "top",
            "score_adv"]
IDS = ["umpire", "pitcher", "batter", "catcher", "game"]

# --------------------------------------------------------------------------- 2. models
CONTROLS = "i(count) + i(inning) + top + score_adv"

# Partition keys. dfeols handles an FE locally when it is nested in the key and
# as (sparse) dummies when it crosses shards; crossing FEs need <= max_dummy_levels
# levels. Batters have the most levels, so they are the default key. Sharding the
# same data two ways and getting the same answer is itself one of the checks.
DATASETS = [
    DatasetSpec("by_batter", "batter",
                ("batter", "pitcher", "umpire", "pitcher^questec", "umpire^questec"), ("game",)),
    DatasetSpec("by_pitcher", "pitcher", ("pitcher", "umpire", "batter"), ("game",)),
    DatasetSpec("by_batter_qt", ("batter", "questec"),
                ("batter^questec", "pitcher^questec", "umpire^questec"), ("game",)),
]

SPECS = [
    Spec("T3c7", f"strike ~ upm + {CONTROLS} | pitcher", ("by_pitcher",),
         paper="Table 3, col. 7"),
    Spec("T3c8", f"strike ~ upm + {CONTROLS} | pitcher + umpire", ("by_pitcher",),
         paper="Table 3, col. 8"),
    Spec("T3c9", f"strike ~ upm + {CONTROLS} | pitcher + umpire + batter", ("by_batter", "by_pitcher"),
         paper="Table 3, col. 9", note="run on two partitions: invariance check"),
    Spec("T4", f"strike ~ upm + upm:questec + {CONTROLS} | pitcher^questec + umpire^questec + batter",
         ("by_batter",), paper="Table 4, pooled"),
    Spec("T5A", f"strike ~ upm + well_att + upm:well_att + {CONTROLS} | pitcher + umpire + batter",
         ("by_batter",), paper="Table 5A, pooled"),
    Spec("T5B", f"strike ~ upm + upm:terminal + {CONTROLS} | pitcher + umpire + batter",
         ("by_batter",), paper="Table 5B, pooled",
         note="terminal itself is a function of the count dummies, so it is absorbed by i(count)"),
    Spec("T7", f"strike ~ upm + upm:questec + upm:well_att + upm:terminal + well_att + {CONTROLS}"
               " | pitcher^questec + umpire^questec + batter^questec",
         ("by_batter_qt",), paper="Table 7", note="check the exact regressor list in the do-file"),
    # Not in the paper: exercises CRV1 with clusters (games) that cross shards.
    Spec("T3c9_game", f"strike ~ upm + {CONTROLS} | pitcher + umpire + batter", ("by_batter",),
         vcov=("CRV1", "hetero"), cluster="game", paper="(extra check)"),
]

# --------------------------------------------------------------------------- 3. published
# {spec: {term: (coef, se, decimals)}}. The values below are from the NBER
# working paper (w13665: seasons 2004-06, N = 1,132,145 called pitches), read
# from the PDF; the AER version uses a different sample, so these are for
# orientation only. Replace them with the AER tables and set PUBLISHED_N.
PUBLISHED: dict[str, dict[str, tuple[float, float | None, int]]] = {
    "T3c7": {"upm": (0.0034, 0.0017, 4)},
    "T3c8": {"upm": (0.0028, 0.0023, 4)},
    "T3c9": {"upm": (0.0027, 0.0023, 4)},
    "T4": {"upm": (0.0064, 0.0029, 4), "upm:questec": (-0.0098, 0.0047, 4)},
    "T5A": {"upm": (0.0052, 0.0025, 4), "well_att": (0.0062, 0.0016, 4),
            "upm:well_att": (-0.0050, 0.0019, 4)},
    "T5B": {"upm": (0.0042, 0.0024, 4), "upm:terminal": (-0.0073, 0.0018, 4)},
}
PUBLISHED_SOURCE = "NBER WP w13665 (orientation only; replace with the AER tables)"
PUBLISHED_N: dict[str, int] = {k: 1_132_145 for k in PUBLISHED}
