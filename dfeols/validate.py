"""Compare a fit with the known truth of a synthetic dataset."""

from __future__ import annotations

import numpy as np
from scipy import stats


def true_values(res, truth: dict, y: str | None = None) -> tuple[list[int], np.ndarray]:
    """Indices of reported coefficients with a known truth, and their true values.

    When a coefficient is dropped as collinear (e.g. one tenure bin of an
    account-week variable under an account x week FE), the remaining
    coefficients of that variable are identified relative to it, so the
    truth is shifted by the dropped coefficient's true value."""
    tr = truth["coefs"]
    y = y or next(iter(res.outcomes))
    scale = truth.get("outcome_scale", {}).get(y, 1.0)
    ref = {c.split(":")[-1]: tr[c] for c in res.collinear if c in tr}
    idx = [i for i, c in enumerate(res.coefnames) if c in tr]
    vals = np.array([(tr[res.coefnames[i]] - ref.get(res.coefnames[i].split(":")[-1], 0.0)) * scale
                     for i in idx])
    return idx, vals


def recovery(res, truth: dict, y: str | None = None, vcov: str | None = None,
             alpha: float = 0.001) -> dict:
    """z-statistics of (estimate - truth) and a joint Wald test.

    With clustered SEs the Wald statistic uses the small-G correction
    F = W (G - k) / (k (G - 1)) ~ F(k, G - k)."""
    idx, beta = true_values(res, truth, y)
    if not idx:
        raise ValueError("no coefficients with a known truth in this fit")
    b = res.coef(y)[idx]
    V = res.vcov(y, vcov)[np.ix_(idx, idx)]
    z = (b - beta) / np.sqrt(np.diag(V))
    d = b - beta
    W = float(d @ np.linalg.solve(V, d))
    k, G = len(idx), res.G
    vt = vcov or res.vcov_default
    if vt == "CRV1" and G and G > k + 1:
        p = float(stats.f.sf(W * (G - k) / (k * (G - 1)), k, G - k))
    else:
        p = float(stats.chi2.sf(W, k))
    zcrit = float(stats.norm.isf(alpha / (2 * k)))
    worst = int(np.abs(z).argmax())
    return {"max_abs_z": float(np.abs(z).max()), "zcrit": zcrit, "wald": W, "p": p, "k": k,
            "G": G, "worst": res.coefnames[idx[worst]], "ok": bool(np.abs(z).max() < zcrit)}
