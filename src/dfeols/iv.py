# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Instrumental variables on the coordinator: 2SLS and LIML (k-class) from the round-1
cross-products, the linear maps round 2 needs, and first-stage statistics
(docs/design/iv.md).

Layout. The outcome block is [y (m), X2 (p endogenous)]. The regressor block Z is
[reported exogenous (k_rep), excluded instruments (q), crossing-FE dummies]; X1 is the
exogenous columns and the dummies. With A = Z'WZ, B = Z'WY, C = Y'WY (FEs nested in the
partition already demeaned):

    Pi = A^-1 Z'X2                          first stage
    X'(I - k M_Z)X = (1 - k) X'X + k Xh'Xh  Xh = P_Z X = [X1, Z Pi]
    beta(k) = [X'(I - k M_Z)X]^-1 X'(I - k M_Z)y

k = 1 is 2SLS. LIML's k is the smallest root of det(W'M_X1 W - k W'M_Z W) = 0 with
W = [y, X2], per outcome; Fuller's modification subtracts a / (N - L).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, stats

from .linalg import collinear_keep, solve_spd
from .vcov import df_k, vcov_crv1, vcov_hetero, vcov_iid

ESTIMATORS = ("2sls", "liml")


@dataclass
class Solution:
    """What the coordinator needs from the solve, OLS or IV."""

    coefnames: list[str]
    coef: np.ndarray  # k_reported x m
    bread: list  # per outcome, in the meat's basis
    meat_idx: np.ndarray  # reported coefficients' positions in the meat's basis
    lin: dict  # round-2 maps (worker.shard_round2)
    dropped: list[str]  # collinear regressors (and instruments)
    kappa: np.ndarray | None = None  # per outcome (IV)
    fs: dict = field(default_factory=dict)  # first-stage pieces for first_stage()


def _levels(plan, zc_dummy: np.ndarray, P: int, coef: np.ndarray) -> np.ndarray:
    """Dummy coefficients (rows: kept dummy columns of Z) by crossing-FE level (P rows;
    dropped first levels stay 0), as the sparse round 2 wants them."""
    from .worker import _global_layout

    _, zpos, _ = _global_layout(plan)
    level_of_z = {int(z): i for i, z in enumerate(zpos) if z >= 0}
    out = np.zeros((P, coef.shape[1]))
    out[[level_of_z[int(z)] for z in zc_dummy]] = coef
    return out


def solve(plan, acc: dict, N: int, *, estimator: str = "2sls", fuller: float | None = None, collin_tol=1e-9):
    """The k-class fit of every outcome, and the maps for round 2."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    if fuller is not None and estimator != "liml":
        raise ValueError("fuller= modifies LIML: use estimator='liml'")
    m, p = len(plan.formula.depvars), len(plan.endog_names)
    k_rep, k_x = len(plan.colnames), plan.k_x
    ZtZ, ZtY, C = acc["ZtZ"], acc["ZtY"], acc["YtY"]
    k_tot = ZtZ.shape[0]

    # collinearity: X1 (exogenous, then dummies) before the instruments, so an instrument
    # collinear with the exogenous columns is the one dropped
    order = np.r_[np.arange(k_rep), np.arange(k_x, k_tot), np.arange(k_rep, k_x)]
    keep = np.zeros(k_tot, bool)
    keep[order[collinear_keep(ZtZ[np.ix_(order, order)], collin_tol)]] = True
    zc = np.flatnonzero(keep)  # kept Z columns, in Z order
    is_inst = (zc >= k_rep) & (zc < k_x)
    x1, z2 = np.flatnonzero(~is_inst), np.flatnonzero(is_inst)  # positions within zc
    dropped = [plan.colnames[i] for i in range(k_rep) if not keep[i]]
    dropped += [plan.inst_names[i - k_rep] for i in range(k_rep, k_x) if not keep[i]]
    q = len(z2)
    if q < p:
        raise ValueError(
            f"{p} endogenous regressor(s) but only {q} excluded instrument(s) left after "
            f"collinearity checks (dropped: {dropped or 'none'}); the model is not identified"
        )

    A, B = ZtZ[np.ix_(zc, zc)], ZtY[zc]
    AiB, Ainv = solve_spd(A, B)
    Pi = AiB[:, m:]  # |zc| x p
    A11 = A[np.ix_(x1, x1)]
    B1 = B[x1]
    XX = np.block([[A11, B1[:, m:]], [B1[:, m:].T, C[m:, m:]]])
    XhXh = np.block([[A11, B1[:, m:]], [B1[:, m:].T, B[:, m:].T @ Pi]])
    Xy = np.vstack([B1[:, :m], C[m:, :m]])
    Xhy = np.vstack([B1[:, :m], Pi.T @ B[:, :m]])

    # an endogenous regressor whose fitted values are collinear with X1 is dropped
    kx = collinear_keep(XhXh, collin_tol)
    ek = np.flatnonzero(kx[len(x1) :])  # kept endogenous
    dropped += [plan.endog_names[i] for i in range(p) if not kx[len(x1) + i]]
    if len(ek) == 0:
        raise ValueError("every endogenous regressor is collinear with the exogenous regressors")
    XX, XhXh, Xy, Xhy = XX[np.ix_(kx, kx)], XhXh[np.ix_(kx, kx)], Xy[kx], Xhy[kx]

    kappa = np.ones(m)
    if estimator == "liml":
        A11inv = np.linalg.inv(A11)
        for j in range(m):
            W = np.r_[j, m + ek]
            WW, WZ, WX1 = C[np.ix_(W, W)], B[:, W], B1[:, W]
            WMZW = WW - WZ.T @ Ainv @ WZ
            WMX1W = WW - WX1.T @ A11inv @ WX1
            kappa[j] = linalg.eigh(WMX1W, WMZW, eigvals_only=True)[0]
    lam = kappa.copy()  # LIML's lambda, before Fuller's adjustment
    if estimator == "liml":
        if fuller is not None:
            kappa -= fuller / (N - _n_instruments(plan, zc, acc))

    nx1 = len(x1)
    coef_x, bread = [], []
    for j in range(m):
        k_ = kappa[j]
        M = (1 - k_) * XX + k_ * XhXh
        r = (1 - k_) * Xy[:, j] + k_ * Xhy[:, j]
        b, Minv = solve_spd(M, r[:, None])
        coef_x.append(b[:, 0])
        bread.append(Minv)
    coef_x = np.column_stack(coef_x)  # (|x1| + |ek|) x m, basis [X1, X2 kept]
    # Z'e of each outcome's residuals (actual X), for Sargan's e'P_Z e
    Ze = B[:, :m] - A[:, x1] @ coef_x[:nx1] - B[:, m + ek] @ coef_x[nx1:]

    # reported: exogenous columns of X1 (not dummies), then the kept endogenous
    rep_x1 = np.flatnonzero(zc[x1] < k_rep)
    coefnames = [plan.colnames[zc[x1[i]]] for i in rep_x1] + [plan.endog_names[i] for i in ek]
    rep = np.r_[rep_x1, nx1 + np.arange(len(ek))]

    lin = _round2_maps(plan, m, p, zc, x1, z2, ek, Pi, coef_x, A, B, A11)
    if plan.sparse_global:  # meat already has the dummies partialled out: reported basis only
        bread = [b_[np.ix_(rep, rep)] for b_ in bread]
        meat_idx = np.arange(len(rep))
    else:
        meat_idx = rep
    fs = {
        "q": q,
        "k_fs": len(rep_x1) + q,  # first-stage K for the small-sample correction
        "Pi_z2": Pi[z2],  # q x p
        "bread_z2": Ainv[np.ix_(z2, z2)],  # (z2~'W z2~)^-1, z2~ = M_X1 Z2
        "rss_full": np.array([C[m + l, m + l] - B[:, m + l] @ AiB[:, m + l] for l in range(p)]),
        "rss_x1": np.array(
            [C[m + l, m + l] - B1[:, m + l] @ np.linalg.solve(A11, B1[:, m + l]) for l in range(p)]
        ),
        "inst_names": [plan.inst_names[zc[i] - k_rep] for i in z2],
        "ePZe": np.einsum("zj,zw,wj->j", Ze, Ainv, Ze),
        "overid_df": q - len(ek),
        "lambda": lam if estimator == "liml" else None,
    }
    return Solution(coefnames, coef_x[rep], bread, meat_idx, lin, dropped, kappa, fs)


def _n_instruments(plan, zc, acc) -> int:
    """L in Fuller's a / (N - L): every instrument, exogenous regressors included, plus the
    absorbed FE levels (as ivreghdfe counts them; checked in tests/test_iv.py)."""
    fe = int(acc["fe_levels"].sum()) - max(len(acc["fe_levels"]) - 1, 0) if len(acc["fe_levels"]) else 0
    return len(zc) + fe


def _round2_maps(plan, m, p, zc, x1, z2, ek, Pi, coef_x, A, B, A11):
    """Linear maps of T = [Y (m + p), Z[:, cols]] for worker.shard_round2: residuals
    y - X beta and score regressors Xh = [X1, Z Pi] of each outcome, and the first stage's
    residuals v = X2 - Z Pi with score regressors Z2~ = M_X1 Z2 (FWL: the Z2 block of the
    first stage's sandwich).

    The scores are Xh for every k-class estimator, with the k-class bread, as ivreg2 and
    linearmodels compute robust and clustered LIML vcovs. (X~ = (1 - k) X + k Xh, the
    k-class estimating equation's own regressors, differs from Xh by O(1/N).)"""
    k_x = plan.k_x
    nT_y = m + p
    nx1 = len(x1)
    sparse = plan.sparse_global
    nd = np.flatnonzero(zc < k_x) if sparse else np.arange(len(zc))  # zc positions in T
    pos = np.full(len(zc), -1)
    pos[nd] = nT_y + np.arange(len(nd))  # T row of each zc position (-1: a dummy, sparse only)
    nT = nT_y + len(nd)
    x1_t = [i for i in range(nx1) if pos[x1[i]] >= 0]  # X1 columns in T (all of them when dense)
    kXr = len(x1_t) + len(ek)

    def zrows(vec_zc: np.ndarray) -> np.ndarray:
        """A vector over zc positions as T rows (the dummy entries dropped)."""
        out = np.zeros((nT,) + vec_zc.shape[1:])
        out[pos[pos >= 0]] = vec_zc[pos >= 0]
        return out

    # residuals of the outcomes
    Ares = np.zeros((nT, m))
    Ares[np.arange(m), np.arange(m)] = 1.0
    beta_x1 = np.zeros((len(zc), m))
    beta_x1[x1] = coef_x[:nx1]
    Ares[m + ek] = -coef_x[nx1:]
    Ares -= zrows(beta_x1)

    # score regressors, in the meat's basis: [X1 in T (all of X1 when dense), kept X2 fitted]
    H = np.zeros((nT, kXr))
    for c, i in enumerate(x1_t):
        H[pos[x1[i]], c] = 1.0
    for c, l in enumerate(ek):
        H[:, len(x1_t) + c] = zrows(Pi[:, l])

    # first stage of each endogenous regressor
    Afs = np.zeros((nT, p))
    Afs[m + np.arange(p), np.arange(p)] = 1.0
    Afs -= zrows(Pi)
    Gam = np.linalg.solve(A11, A[np.ix_(x1, z2)])  # X1 coefficients of Z2: |x1| x q
    Gz = np.zeros((len(zc), len(z2)))
    Gz[z2] = np.eye(len(z2))
    Gz[x1] = -Gam
    Hfs = zrows(Gz)

    lin = {"cols": zc[nd], "A": Ares, "H": H, "G": None, "GH": None}
    fs = {"A": Afs, "H": Hfs, "G": None, "GH": None}
    if sparse:
        dum = np.flatnonzero(zc >= k_x)
        _, _, P = _layout(plan)
        # dummy coefficients by level, and the dummies' coefficients of each T column
        # (D~'D~)^-1 D~'T, from the kept-dummy blocks of A and B
        Add = A[np.ix_(dum, dum)]
        DT = np.zeros((len(dum), nT))
        DT[:, :nT_y] = B[dum]
        DT[:, nT_y:] = A[np.ix_(dum, nd)]
        piT = np.linalg.solve(Add, DT)
        lin["G"] = _levels(plan, zc[dum], P, beta_x1[dum])
        lin["GH"] = _levels(plan, zc[dum], P, piT @ H)
        fs["G"] = _levels(plan, zc[dum], P, Pi[dum])
        # M_D (Z2 - X1 Gam), which is M_X1 Z2 by FWL
        fs["GH"] = _levels(plan, zc[dum], P, piT @ Hfs)
    lin["fs"] = fs
    return lin


def _layout(plan):
    from .worker import _global_layout

    return _global_layout(plan)


def first_stage(plan, sol: Solution, acc: dict, acc2: dict, N: int, fe_levels, nested, G) -> dict:
    """`structures.first_stage` contents per endogenous regressor: coefficients on the
    excluded instruments with their vcovs, R², partial R², F statistics (iid partial F,
    Wald F under each vcov), and the effective F with one endogenous regressor."""
    m = len(plan.formula.depvars)
    fs = sol.fs
    q, k_fs = fs["q"], fs["k_fs"]
    bread = fs["bread_z2"]
    Q = np.linalg.inv(bread)
    dfk = df_k(k_fs, fe_levels, None)
    out = []
    for l, name in enumerate(plan.endog_names):
        pi = fs["Pi_z2"][:, l]
        rss, rss_x1 = fs["rss_full"][l], fs["rss_x1"][l]
        tss = (
            acc["YtY"][m + l, m + l]
            if plan.local_fes
            else acc["sum_wy2"][m + l] - acc["sum_wy"][m + l] ** 2 / acc["sumw"]
        )
        f_iid = ((rss_x1 - rss) / q) / (rss / (N - dfk))
        entry = {
            "endogenous": name,
            "instruments": fs["inst_names"],
            "coef": pi.tolist(),
            "r2": float(1 - rss / tss),
            "partial_r2": float(1 - rss / rss_x1),
            "rss": float(rss),
            "f_stat": [_test("f_iid", f_iid, q, N - dfk)],
            "vcov": {},
        }
        for t in plan.vcov_types:
            if t == "iid":
                V, df2 = vcov_iid(bread, rss, N, dfk), N - dfk
            elif t == "hetero":
                V, df2 = vcov_hetero(bread, acc2["fs_hetero"][l], N, dfk), N - dfk
            else:
                dfk_cl = df_k(k_fs, fe_levels, nested)
                V, df2 = vcov_crv1(bread, acc2["fs_crv"][l], N, dfk_cl, G), G - 1
            entry["vcov"][t] = V.tolist()
            if t != "iid":
                wald = float(pi @ np.linalg.solve(V, pi)) / q
                entry["f_stat"].append(_test(f"wald_{t}", wald, q, df2))
            if len(plan.endog_names) == 1:
                # Montiel Olea-Pflueger: pi'Q pi / tr(V Q), Q = Z2~'W Z2~
                eff = float(pi @ Q @ pi / np.trace(V @ Q))
                entry.setdefault("effective_f", []).append({"vcov": t, "statistic": eff})
        out.append(entry)
    return {"per_endogenous": out}


def overid(sol: Solution, rss: np.ndarray, N: int) -> list[list[dict]]:
    """Overidentification tests of each outcome, as ivreg2 defines them (none when the
    model is just identified): Sargan's N e'P_Z e / e'e with the fit's residuals, and for
    LIML the Anderson-Rubin statistics N ln(lambda) and N (lambda - 1); chi2(q - p)."""
    fs = sol.fs
    df = fs["overid_df"]
    out = []
    for j in range(len(rss)):
        tests = []
        if df > 0:
            tests.append(_chi2("sargan", N * fs["ePZe"][j] / rss[j], df))
            if fs["lambda"] is not None:
                lam = fs["lambda"][j]
                tests.append(_chi2("anderson_rubin", N * np.log(lam), df))
                tests.append(_chi2("anderson_rubin_linear", N * (lam - 1), df))
        out.append(tests)
    return out


def _chi2(name, stat, df) -> dict:
    return {
        "name": name,
        "statistic": float(stat),
        "df": int(df),
        "p_value": float(stats.chi2.sf(stat, df)),
        "distribution": "chi2",
    }


def _test(name, stat, df1, df2) -> dict:
    return {
        "name": name,
        "statistic": float(stat),
        "df1": int(df1),
        "df2": float(df2),
        "p_value": float(stats.f.sf(stat, df1, df2)),
        "distribution": "F",
    }
