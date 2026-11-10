# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Small-sample conventions follow fixest/pyfixest defaults:

  df_k = k + k_fe_adj - k_fe_nested + n_fe_fully_nested      (k_fixef="nonnested")
  k_fe_adj = sum of FE levels - (number of FEs - 1)
  iid:    V = RSS/(N - df_k) * bread
  hetero: V = N/(N - df_k) * bread meat bread
  CRV1:   V = G/(G-1) * (N-1)/(N - df_k) * bread meat bread,  t-dist df = G-1

Here k counts the reported regressors kept after collinearity checks. FEs
entered as dummies (low-cardinality global FEs) are counted as FEs, not as k,
so results match a fit that absorbs them.

Varying slopes (fe[x]) as fixest counts them: every slope coefficient (levels x slope
variables) adds 1, identified or not and nested in the cluster or not; a slope-only FE
(fe[[x]]) has no levels of its own, and with no FE intercept at all the model's constant
adds 1.
"""

from __future__ import annotations


def df_k(
    k: int,
    fe_levels: list[int],
    fe_nested_in_cluster: list[bool] | None,
    slopes: list[int] | None = None,
    intercept: list[bool] | None = None,
) -> int:
    """`slopes`: per FE, its number of slope variables; `intercept`: per FE, False for a
    slope-only FE. Both default to plain FEs."""
    if slopes is not None and (any(slopes) or (intercept is not None and not all(intercept))):
        ic = list(intercept) if intercept is not None else [True] * len(fe_levels)
        nested = fe_nested_in_cluster or [False] * len(fe_levels)
        extra = sum(L * s for L, s in zip(fe_levels, slopes)) + int(bool(fe_levels) and not any(ic))
        keep = [i for i, ok in enumerate(ic) if ok]
        return extra + df_k(
            k,
            [fe_levels[i] for i in keep],
            [nested[i] for i in keep] if fe_nested_in_cluster else None,
        )
    n_fe = len(fe_levels)
    if n_fe == 0:
        return k
    k_fe = sum(fe_levels)
    k_fe_adj = k_fe - (n_fe - 1)
    if not fe_nested_in_cluster or not any(fe_nested_in_cluster):
        return k + k_fe_adj
    k_nested = sum(l for l, nst in zip(fe_levels, fe_nested_in_cluster) if nst)
    return k + k_fe_adj - k_nested + sum(fe_nested_in_cluster)


def vcov_iid(bread, rss, N, dfk):
    return bread * rss / (N - dfk)


def vcov_hetero(bread, meat, N, dfk):
    return (N / (N - dfk)) * bread @ meat @ bread


def vcov_crv1(bread, meat, N, dfk, G):
    return (G / (G - 1)) * ((N - 1) / (N - dfk)) * bread @ meat @ bread
