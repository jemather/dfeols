# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Example DGP: synthetic follower x account x week panels with known parameters.

A worked example of a richer DGP than the generic ``panel`` one, modelled on
social-media engagement data: followers nested in follow spells, spells in
accounts, a tenure-binned treatment with a piecewise-linear true response, and
clusters (followers) that cross shards. It is used by the test suite and the
benchmarks; use ``panel`` as the starting point for your own DGP.

Model:

  eng_it = sum_b beta_b * feat_jt * 1[tbin_it = b]
         + sum_b gamma_b * posts_jt * 1[tbin_it = b]
         + alpha_spell + delta_week                      (fixed effects)
         + psi_{j,t} + e_it                               (error)

  beta_b  = l(G(a_b)) at bin midpoint a_b (months), with
            G(a) = beta0 - gamma_slope * (a - T1)^+,  l(G) = G or lambda*G,
            so the true curve has knots T1 and T0 = T1 + beta0/gamma_slope.
  gamma_b = constant posts effect.

Identification traps (so the tests have power):
  * alpha_spell is correlated with the account's feature propensity
    -> leaving out the spell FE biases the feature coefficients;
  * delta_week loads on a common trend in feature use
    -> leaving out the week FE biases them too.
Inference traps:
  * psi_{j,t} is an account-week shock shared by all followers of an account
    in that week, and e_it has an account random effect and heteroskedasticity
    -> iid SEs are too small; account-clustered SEs are right.

Accounts have heavy-tailed audience sizes, followers follow several accounts
(follower clusters cross shards), and many spells are already running when the
window opens (left-censored), so tenures span all bins.

Data are written chunk by chunk (one Parquet file per chunk of accounts), each
chunk seeded from (seed, chunk id), so any subset of chunks can be generated
independently: on one machine with a process pool, or on cluster workers.

    python -m dfeols.synth OUT --dgp influencer --rows 5e7
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl


@dataclass
class InfluencerDGP:
    kind = "influencer"
    n_accounts: int = 10_000
    weeks: int = 104
    n_bins: int = 37  # monthly tenure bins 0..35, 36 = 36+
    spells_logmean: float = 4.0  # log mean spells per account (heavy tail)
    spells_logsd: float = 1.0
    max_spells_per_account: int = 20_000
    followers_per_spell: float = 0.35  # follower pool = this x total spells
    prior_share: float = 0.5  # share of spells already running at week 0
    prior_max_weeks: int = 160  # tenure at week 0 for those spells
    mean_duration_weeks: float = 40.0  # geometric spell duration
    # feature-response curve (months)
    beta0: float = 0.30
    T1: float = 8.0
    T0: float = 14.0
    lam: float = 1.5
    gamma_posts: float = 0.05
    # confounding and noise
    alpha_on_propensity: float = 2.0  # spell FE loads on account feature propensity
    delta_on_trend: float = 3.0  # week FE loads on the common feature trend
    psi_sd: float = 0.5  # account-week shock
    acct_re_sd: float = 0.5  # account random effect in e
    noise_sd: float = 1.0
    hetero: float = 0.5  # sd multiplier: 1 + hetero * feat
    seed: int = 20261020
    accounts_per_chunk: int = 2_000

    # ---- truth --------------------------------------------------------------
    def beta(self) -> np.ndarray:
        a = np.arange(self.n_bins) + 0.5
        slope = self.beta0 / (self.T0 - self.T1)
        G = self.beta0 - slope * np.maximum(a - self.T1, 0)
        return np.where(G >= 0, G, self.lam * G)

    def truth(self) -> dict:
        b = self.beta()
        coefs = {f"tbin::{i}:feat": float(b[i]) for i in range(self.n_bins)}
        coefs.update({f"tbin::{i}:posts": self.gamma_posts for i in range(self.n_bins)})
        return {
            "coefs": coefs,
            "outcome_scale": {"eng": 1.0, "eng2": 0.5},
            "prep": {"fixed_effects": ["spell", "week", "id_a^week"], "clusters": ["id_a", "id_f"]},
            "specs": SPECS,
        }

    @property
    def n_chunks(self) -> int:
        return math.ceil(self.n_accounts / self.accounts_per_chunk)

    @property
    def units_per_chunk(self) -> int:
        return self.accounts_per_chunk

    def generate_chunk(self, chunk: int) -> pl.DataFrame:
        return generate_chunk(self, chunk)

    def scaled(self, n_units: int) -> InfluencerDGP:
        return InfluencerDGP(**{**asdict(self), "n_accounts": int(n_units)})

    @property
    def n_units(self) -> int:
        return self.n_accounts


# Specs: formula, partition key, cluster, and what to expect against the truth.
#   unbiased     estimates centred on the truth, valid clustered SEs
#   biased       omits a confounded FE: recovery should be rejected
#   se_wrong     unbiased point estimates, but clustering ignores the error
#                structure (account-week shocks), so SEs are too small
SPECS = {
    "level": {
        "formula": "eng ~ feat:i(tbin) + posts:i(tbin) | spell + week",
        "partition": "id_a",
        "cluster": "id_a",
        "expect": "unbiased",
    },
    "shape": {
        "formula": "eng ~ feat:i(tbin) + posts:i(tbin) | spell + id_a^week",
        "partition": "id_a",
        "cluster": "id_a",
        "expect": "unbiased",
    },
    "fcluster": {
        "formula": "eng ~ feat:i(tbin) + posts:i(tbin) | spell + week",
        "partition": "id_a",
        "cluster": "id_f",
        "expect": "se_wrong",
    },
    "no_week_fe": {
        "formula": "eng ~ feat:i(tbin) + posts:i(tbin) | spell",
        "partition": "id_a",
        "cluster": "id_a",
        "expect": "biased",
    },
    "no_spell_fe": {
        "formula": "eng ~ feat:i(tbin) + posts:i(tbin) | week",
        "partition": "id_a",
        "cluster": "id_a",
        "expect": "biased",
    },
}

DGP = InfluencerDGP  # backward compatibility


def _common_paths(d):
    """Week-level series shared by all chunks (from the master seed)."""
    rng = np.random.default_rng([d.seed, 0])
    t = np.arange(d.weeks)
    trend = 0.15 * np.sin(2 * np.pi * t / 52) + 0.1 * t / d.weeks  # common feature-use trend
    delta = d.delta_on_trend * trend + rng.normal(0, 0.3, d.weeks)
    return trend, delta


def generate_chunk(d, chunk: int) -> pl.DataFrame:
    rng = np.random.default_rng([d.seed, 1, chunk])
    a0 = chunk * d.accounts_per_chunk
    A = min(d.accounts_per_chunk, d.n_accounts - a0)
    W = d.weeks
    trend, delta = _common_paths(d)

    # accounts: propensity, posting rate, weekly feature use and posts
    prop = rng.uniform(0.1, 0.6, A)
    rate = rng.gamma(4.0, 1.0, A)  # posts / week
    ar = np.empty((A, W))
    ar[:, 0] = rng.normal(0, 0.15, A)
    eps = rng.normal(0, 0.15 * np.sqrt(1 - 0.6**2), (A, W))
    for t in range(1, W):
        ar[:, t] = 0.6 * ar[:, t - 1] + eps[:, t]
    feat_aw = np.clip(prop[:, None] + trend[None, :] + ar, 0, 1)  # proportion of text
    posts_aw = rng.poisson(rate[:, None], (A, W)).astype(np.float64)
    psi_aw = rng.normal(0, d.psi_sd, (A, W))
    acct_re = rng.normal(0, d.acct_re_sd, A)

    # spells
    ns = np.minimum(
        np.round(np.exp(rng.normal(d.spells_logmean, d.spells_logsd, A))).astype(np.int64) + 1,
        d.max_spells_per_account,
    )
    S = int(ns.sum())
    s_acct = np.repeat(np.arange(A), ns)
    s_local = np.arange(S) - np.repeat(np.cumsum(ns) - ns, ns)
    spell_id = (a0 + s_acct).astype(np.int64) * 100_000 + s_local
    n_spells_total = math.exp(d.spells_logmean + d.spells_logsd**2 / 2) * d.n_accounts
    pool = max(int(d.followers_per_spell * n_spells_total), 1000)
    follower = rng.integers(0, pool, S)
    prior = rng.uniform(size=S) < d.prior_share
    start = np.where(prior, 0, rng.integers(0, W, S))
    t0 = np.where(prior, rng.integers(1, d.prior_max_weeks, S), 0)  # tenure at start
    dur = rng.geometric(1 / d.mean_duration_weeks, S)
    end = np.minimum(start + dur, W)  # exclusive
    nobs = np.maximum(end - start, 0)
    keep = nobs > 0
    s_acct, spell_id, follower, start, t0, nobs = (
        x[keep] for x in (s_acct, spell_id, follower, start, t0, nobs)
    )
    alpha = d.alpha_on_propensity * prop[s_acct] + rng.normal(0, 1, len(s_acct))

    # expand to spell-weeks
    n = int(nobs.sum())
    sp = np.repeat(np.arange(len(nobs)), nobs)
    k = np.arange(n) - np.repeat(np.cumsum(nobs) - nobs, nobs)
    week = start[sp] + k
    tenure_w = t0[sp] + k
    tbin = np.minimum((tenure_w / 4.345).astype(np.int64), d.n_bins - 1)
    a = s_acct[sp]
    feat = feat_aw[a, week]
    posts = posts_aw[a, week]

    b = d.beta()
    # error: account-week shock + heteroskedastic noise with an account component
    # (all mean zero and independent of feat and posts -> estimates are unbiased)
    eng = (
        b[tbin] * feat
        + d.gamma_posts * posts
        + alpha[sp]
        + delta[week]
        + psi_aw[a, week]
        + (1 + d.hetero * feat) * (d.noise_sd * rng.normal(0, 1, n) + acct_re[a])
    )
    eng2 = 0.5 * eng + rng.normal(0, 1, n)
    return pl.DataFrame(
        {
            "id_a": (a0 + a).astype(np.int64),
            "id_f": follower[sp].astype(np.int64),
            "spell": spell_id[sp],
            "week": week.astype(np.int16),
            "tenure_w": tenure_w.astype(np.int32),
            "tbin": tbin.astype(np.int16),
            "feat": feat,
            "posts": posts,
            "eng": eng,
            "eng2": eng2,
            "wt": rng.uniform(0.5, 2.0, n),
        }
    )
