# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Generic unit x time panel with known parameters.

  y_it = sum_c theta_c * x1_it * 1[cat_it = c] + beta2 * x2_it + beta3 * x3_it
       + alpha_i + gamma_t + delta_{g(i),t} + e_it

Units i are nested in groups g (firms in industries, people in counties,
stores in regions). Panels are unbalanced (random entry and exit) unless
balanced=True. Time can be anything ordinal: days, weeks, years.

Confounding (so omitting an FE biases the estimates):
  * x's load on the unit's latent type, which also drives alpha_i;
  * x's load on a common time path, which also drives gamma_t;
  * with gt_confound > 0, x's load on group-time shocks that also enter y,
    so only a group x time FE removes the bias.
Errors: AR(1) within unit, heteroskedastic in x1, and (when the group-time
shock is not absorbed) correlated within group-period. With gt_confound = 0
the group-time shock is pure noise: unit + time FEs are enough for the
point estimates, but SEs must be clustered by group.

Specs (truth()["specs"]) say what each model should find:
  twfe         y ~ ... | unit + time          partition unit, cluster group
  twfe_by_grp  same model, partitioned by group (must equal twfe exactly)
  grouptime    y ~ ... | unit + group^time    partition group, cluster group
  no_unit_fe   y ~ ... | time                 biased (negative control)
  twfe (when gt_confound > 0) is biased too.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl


@dataclass
class PanelDGP:
    kind = "panel"

    n_units: int = 100_000
    T: int = 52
    n_groups: int = 500
    n_cat: int = 4
    balanced: bool = False
    min_periods: int = 2
    group_zipf: float = 0.0  # >0: group sizes ~ rank^-group_zipf (skewed)
    # coefficients
    theta: tuple = (0.5, 0.8, 1.1, 1.4)  # x1 effect by category (length n_cat)
    beta2: float = -0.3
    beta3: float = 0.2
    # confounding loadings
    unit_conf: float = 1.0
    time_conf: float = 1.0
    gt_confound: float = 0.0
    # errors
    rho: float = 0.5  # AR(1) within unit
    noise_sd: float = 1.0
    gt_sd: float = 1.0  # group-time shock in y
    hetero: float = 0.5
    seed: int = 20261021
    units_per_chunk: int = 20_000

    # ---- protocol ----------------------------------------------------------
    @property
    def n_chunks(self) -> int:
        return math.ceil(self.n_units / self.units_per_chunk)

    def scaled(self, n_units: int) -> PanelDGP:
        return PanelDGP(**{**asdict(self), "n_units": int(n_units)})

    def coefs(self) -> dict:
        th = list(self.theta)
        if len(th) != self.n_cat:
            raise ValueError("theta must have n_cat entries")
        out = {f"cat::{c}:x1": float(th[c]) for c in range(self.n_cat)}
        out.update({"x2": self.beta2, "x3": self.beta3})
        return out

    def truth(self) -> dict:
        rhs = "x1:i(cat) + x2 + x3"
        twfe_ok = "unbiased" if self.gt_confound == 0 else "biased"
        specs = {
            "twfe": {
                "formula": f"y ~ {rhs} | unit + time",
                "partition": "unit",
                "cluster": "group",
                "expect": twfe_ok,
            },
            "twfe_by_grp": {
                "formula": f"y ~ {rhs} | unit + time",
                "partition": "group",
                "cluster": "group",
                "expect": twfe_ok,
            },
            "grouptime": {
                "formula": f"y ~ {rhs} | unit + group^time",
                "partition": "group",
                "cluster": "group",
                "expect": "unbiased",
            },
            "no_unit_fe": {
                "formula": f"y ~ {rhs} | time",
                "partition": "unit",
                "cluster": "group",
                "expect": "biased",
            },
        }
        return {
            "coefs": self.coefs(),
            "outcome_scale": {"y": 1.0},
            "prep": {"fixed_effects": ["unit", "time", "group^time"], "clusters": ["unit", "group"]},
            "specs": specs,
        }

    # ---- data ----------------------------------------------------------------
    def _common(self):
        rng = np.random.default_rng([self.seed, 0])
        T, G = self.T, self.n_groups
        tau = np.cumsum(rng.normal(0, 0.3, T))  # common time path
        tau -= tau.mean()
        gamma = self.time_conf * tau + rng.normal(0, 0.5, T)
        c_gt = rng.normal(0, 1, (G, T))  # group-time factor in x
        delta = self.gt_confound * c_gt + rng.normal(0, self.gt_sd, (G, T))
        return tau, gamma, c_gt, delta

    def generate_chunk(self, chunk: int) -> pl.DataFrame:
        rng = np.random.default_rng([self.seed, 1, chunk])
        u0 = chunk * self.units_per_chunk
        U = min(self.units_per_chunk, self.n_units - u0)
        T = self.T
        tau, gamma, c_gt, delta = self._common()

        unit = np.arange(u0, u0 + U)
        if self.group_zipf > 0:
            # skewed group sizes: a few very large groups (stresses shards that
            # can't be split, since a partition group never spans shards)
            w = np.arange(1, self.n_groups + 1, dtype=float) ** -self.group_zipf
            group = np.random.default_rng([self.seed, 2, chunk]).choice(self.n_groups, size=U, p=w / w.sum())
        else:
            group = (unit * 2654435761) % self.n_groups  # spread units over groups
        utype = rng.normal(0, 1, U)
        alpha = self.unit_conf * utype + rng.normal(0, 1, U)

        if self.balanced:
            start = np.zeros(U, np.int64)
            length = np.full(U, T)
        else:
            mp = min(self.min_periods, T)
            length = rng.integers(mp, T + 1, U)
            start = rng.integers(0, T - length + 1)
        n = int(length.sum())
        ui = np.repeat(np.arange(U), length)
        t = start[ui] + (np.arange(n) - np.repeat(np.cumsum(length) - length, length))
        g = group[ui]

        # regressors: unit type + time path + group-time factor + noise
        x = {}
        for k, (a, b, c) in {"x1": (0.6, 0.6, 0.5), "x2": (-0.5, 0.4, 0.5), "x3": (0.4, -0.5, 0.5)}.items():
            x[k] = a * utype[ui] + b * tau[t] + c * c_gt[g, t] + rng.normal(0, 1, n)

        # categories: a unit's base class, which may switch once
        base = rng.integers(0, self.n_cat, U)
        switch = rng.integers(0, 2 * T, U)  # many never switch
        cat = np.where(t >= switch[ui], (base[ui] + 1) % self.n_cat, base[ui])

        # AR(1) errors within unit: e_t = rho e_{t-1} + sqrt(1-rho^2) z_t
        z = rng.normal(0, 1, n)
        e = _ar1(z, length, self.rho)
        scale = 1 + self.hetero * np.abs(x["x1"])
        th = np.asarray(self.theta, float)
        y = (
            th[cat] * x["x1"]
            + self.beta2 * x["x2"]
            + self.beta3 * x["x3"]
            + alpha[ui]
            + gamma[t]
            + delta[g, t]
            + self.noise_sd * scale * e
        )
        return pl.DataFrame(
            {
                "unit": unit[ui].astype(np.int64),
                "group": g.astype(np.int32),
                "time": t.astype(np.int32),
                "cat": cat.astype(np.int16),
                "x1": x["x1"],
                "x2": x["x2"],
                "x3": x["x3"],
                "y": y,
                "w": rng.uniform(0.5, 2.0, n),
            }
        )


def _ar1(z: np.ndarray, lengths: np.ndarray, rho: float) -> np.ndarray:
    """AR(1) within consecutive blocks of the given lengths (stationary start).
    Vectorised over blocks: iterate over the position within the block."""
    out = np.empty_like(z)
    starts = np.cumsum(lengths) - lengths
    s = math.sqrt(1 - rho**2)
    out[starts] = z[starts]
    maxlen = int(lengths.max()) if len(lengths) else 0
    for j in range(1, maxlen):
        live = lengths > j
        idx = starts[live] + j
        out[idx] = rho * out[idx - 1] + s * z[idx]
    return out
