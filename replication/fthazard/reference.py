# SPDX-License-Identifier: GPL-3.0-or-later
"""Published large-sample estimates (N = 4e7 units, T = 5) from Farbmacher & Tauchmann (2021),
FAU DP 03/2021, Tables 1, A3 and A4, as (coefficient, SE), with the specs that reproduce them.

Names map the paper's columns to dfeols terms: b -> x, alpha -> Intercept, tau_t -> t::t
(levels) or D1_t_t (first differences); b_FD's slope is on D1_x. b_WI's constant is not
reported by dfeols (absorbed). Observation counts are the FD/IV samples (rows - units).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---- Table 1 (no wave indicators); data: section 4 DGPs
T1 = {
    "st": {
        "ols": {"x": (1.4866, 0.0010), "Intercept": (0.0010, 0.0002)},
        "wi": {"x": (0.9023, 0.0017)},
        "fd": {"D1_x": (0.5043, 0.0013), "Intercept": (0.2896, 0.0001)},
        "iv": {"x": (1.0045, 0.0025), "Intercept": (0.0942, 0.0005)},
        "n_iv": 71_732_683,
    },
    "rw": {
        "ols": {"x": (1.2574, 0.0007), "Intercept": (0.0468, 0.0001)},
        "wi": {"x": (0.9447, 0.0013)},
        "fd": {"D1_x": (0.9991, 0.0013), "Intercept": (0.2859, 0.0001)},
        "iv": {"x": (0.9992, 0.0012), "Intercept": (0.0952, 0.0002)},
        "n_iv": 71_929_363,
    },
    "tr": {
        "ols": {"x": (1.4350, 0.0010), "Intercept": (0.0095, 0.0002)},
        "wi": {"x": (3.9783, 0.0014)},
        "fd": {"D1_x": (0.6685, 0.0013), "Intercept": (0.2947, 0.0001)},
        "iv": {"x": (1.0015, 0.0019), "Intercept": (0.0949, 0.0004)},
        "n_iv": 72_211_807,
    },
}


def _waves(ols, wi, fd, iv, n_iv):
    """Rows of Tables A3/A4: (b, tau2..tau5, alpha) per estimator; FD and IV have no tau2."""

    def lv(vals, names):
        return {k: v for k, v in zip(names, vals) if v is not None}

    lvl = ["x", "t::2", "t::3", "t::4", "t::5", "Intercept"]
    return {
        "ols": lv(ols, lvl),
        "wi": lv(wi[:5], lvl[:5]),  # WI's constant is absorbed in dfeols
        "fd": lv(fd, ["D1_x", None, "D1_t_3", "D1_t_4", "D1_t_5", "Intercept"]),
        "iv": lv(iv, ["x", None, "t::3", "t::4", "t::5", "Intercept"]),
        "n_iv": n_iv,
    }


N = None
# ---- Table A3 (wave indicators; same data as Table 1)
A3 = {
    "st": _waves(
        [
            (1.4837, 0.0010),
            (-0.0011, 0.0001),
            (-0.0025, 0.0001),
            (-0.0037, 0.0001),
            (-0.0049, 0.0002),
            (0.0032, 0.0002),
        ],
        [(0.6411, 0.0015), (0.2945, 0.0001), (0.4369, 0.0001), (0.5289, 0.0001), (0.5956, 0.0002)],
        [(0.5043, 0.0013), N, (-0.0049, 0.0001), (-0.0143, 0.0002), (-0.0283, 0.0003), (0.2947, 0.0001)],
        [(1.0045, 0.0025), N, (-0.0025, 0.0001), (-0.0048, 0.0001), (-0.0071, 0.0002), (0.0968, 0.0005)],
        71_732_683,
    ),
    "rw": _waves(
        [
            (1.2550, 0.0007),
            (-0.0017, 0.0001),
            (-0.0028, 0.0001),
            (-0.0033, 0.0001),
            (-0.0031, 0.0002),
            (0.0489, 0.0002),
        ],
        [(1.2017, 0.0011), (0.2953, 0.0001), (0.4371, 0.0001), (0.5265, 0.0001), (0.5888, 0.0002)],
        [(0.9990, 0.0013), N, (-0.0073, 0.0001), (-0.0242, 0.0002), (-0.0529, 0.0003), (0.2953, 0.0001)],
        [(0.9992, 0.0012), N, (-0.0024, 0.0001), (-0.0048, 0.0001), (-0.0070, 0.0002), (0.0978, 0.0003)],
        71_929_363,
    ),
    "tr": _waves(
        [
            (1.5074, 0.0011),
            (-0.0076, 0.0001),
            (-0.0151, 0.0001),
            (-0.0225, 0.0001),
            (-0.0300, 0.0002),
            (0.0050, 0.0002),
        ],
        [(0.8991, 0.0016), (0.2840, 0.0001), (0.4279, 0.0001), (0.5251, 0.0002), (0.5989, 0.0002)],
        [(0.6650, 0.0013), N, (0.0075, 0.0001), (0.0223, 0.0002), (0.0442, 0.0003), (0.2869, 0.0001)],
        [(1.0032, 0.0019), N, (-0.0024, 0.0001), (-0.0047, 0.0002), (-0.0071, 0.0002), (0.0970, 0.0004)],
        72_211_807,
    ),
}
# ---- Table A4 (true time effects tau3 = tau5 = 0.2, wave indicators; its own data)
A4 = {
    "st": _waves(
        [
            (1.4838, 0.0011),
            (-0.0011, 0.0001),
            (0.1975, 0.0001),
            (-0.0041, 0.0002),
            (0.1946, 0.0002),
            (0.0032, 0.0002),
        ],
        [(0.6275, 0.0017), (0.2945, 0.0001), (0.6369, 0.0001), (0.5937, 0.0002), (0.8600, 0.0002)],
        [(0.5045, 0.0014), N, (0.1951, 0.0001), (0.1836, 0.0002), (0.3679, 0.0004), (0.2947, 0.0001)],
        [(1.0041, 0.0028), N, (0.1975, 0.0001), (-0.0057, 0.0002), (0.1920, 0.0002), (0.0969, 0.0006)],
        64_986_815,
    ),
    "rw": _waves(
        [
            (1.2770, 0.0008),
            (-0.0016, 0.0001),
            (0.1975, 0.0001),
            (-0.0032, 0.0002),
            (0.1976, 0.0002),
            (0.0445, 0.0002),
        ],
        [(1.1867, 0.0013), (0.2953, 0.0001), (0.6373, 0.0001), (0.5905, 0.0002), (0.8523, 0.0002)],
        [(1.0012, 0.0014), N, (0.1929, 0.0001), (0.1721, 0.0002), (0.3402, 0.0004), (0.2953, 0.0001)],
        [(1.0013, 0.0014), N, (0.1978, 0.0001), (-0.0057, 0.0002), (0.1925, 0.0002), (0.0974, 0.0003)],
        65_167_537,
    ),
    "tr": _waves(
        [
            (1.5380, 0.0012),
            (-0.0079, 0.0001),
            (0.1843, 0.0001),
            (-0.0238, 0.0002),
            (0.1684, 0.0002),
            (-0.0008, 0.0002),
        ],
        [(0.8951, 0.0018), (0.2840, 0.0001), (0.6280, 0.0001), (0.5900, 0.0002), (0.8634, 0.0002)],
        [(0.6745, 0.0015), N, (0.2075, 0.0001), (0.2201, 0.0003), (0.4403, 0.0004), (0.2868, 0.0001)],
        [(1.0039, 0.0022), N, (0.1976, 0.0001), (-0.0055, 0.0002), (0.1920, 0.0002), (0.0969, 0.0004)],
        65_445_856,
    ),
}

SPECS = {
    "t1": {"rhs": "x", "table": "Table 1"},
    "a3": {"rhs": "x + i(t)", "table": "Table A3"},
    "a4": {"rhs": "x + i(t)", "table": "Table A4"},
}


def reference(dgp: str, time_effects: bool) -> dict:
    """reference.json for one dataset: the specs to run and the published values."""
    tables = {"a4": A4[dgp]} if time_effects else {"t1": T1[dgp], "a3": A3[dgp]}
    specs = {}
    for name, pub in tables.items():
        rhs = SPECS[name]["rhs"]
        specs[name] = {
            "table": SPECS[name]["table"],
            "ols": {"formula": f"y ~ {rhs}", "vcov": "hetero"},
            "wi": {"formula": f"y ~ {rhs} | uid", "vcov": "hetero"},
            "hazard": {"formula": f"y ~ {rhs}", "unit": "uid", "time": "t", "diff": 1, "vcov": "hetero"},
            "published": {
                k: ({t: list(v) for t, v in d.items()} if isinstance(d, dict) else d) for k, d in pub.items()
            },
        }
    return {
        "source": "Farbmacher & Tauchmann (2021), FAU Discussion Papers in Economics 03/2021, "
        + ("Table A4" if time_effects else "Tables 1 and A3"),
        "units_published": 40_000_000,
        "prep": {"partition": "uid", "fixed_effects": ["uid"], "clusters": []},
        "rounding": 0.00005,
        "specs": specs,
    }


if __name__ == "__main__":
    dgp, te, out = sys.argv[1], sys.argv[2] == "te", Path(sys.argv[3])
    out.write_text(json.dumps(reference(dgp, te), indent=1))
