"""Small-sample conventions follow fixest/pyfixest defaults:

  df_k = k + k_fe_adj - k_fe_nested + n_fe_fully_nested      (k_fixef="nonnested")
  k_fe_adj = sum of FE levels - (number of FEs - 1)
  iid:    V = RSS/(N - df_k) * bread
  hetero: V = N/(N - df_k) * bread meat bread
  CRV1:   V = G/(G-1) * (N-1)/(N - df_k) * bread meat bread,  t-dist df = G-1

Here k counts the reported regressors kept after collinearity checks. FEs
entered as dummies (low-cardinality global FEs) are counted as FEs, not as k,
so results match a fit that absorbs them.
"""

from __future__ import annotations

import numpy as np


def df_k(k: int, fe_levels: list[int], fe_nested_in_cluster: list[bool] | None) -> int:
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
