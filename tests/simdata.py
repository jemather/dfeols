"""Simulated follower x account x week panel for tests."""

import numpy as np
import polars as pl


def sim_panel(n_accounts=60, followers_per=40, weeks=30, seed=0, missing=0.01):
    rng = np.random.default_rng(seed)
    rows = []
    spell = 0
    for a in range(n_accounts):
        nf = rng.integers(followers_per // 2, followers_per * 2)
        acc_eff = rng.normal()
        for _ in range(nf):
            f = rng.integers(0, n_accounts * followers_per)
            start = rng.integers(0, weeks - 2)
            length = rng.integers(1, weeks - start + 1)   # some length-1 spells -> singletons
            wk = np.arange(start, start + length)
            rows.append((np.full(length, a), np.full(length, f), np.full(length, spell), wk,
                         wk - start, np.full(length, acc_eff)))
            spell += 1
    a, f, s, w, ten, ae = (np.concatenate(x) for x in zip(*rows))
    n = len(a)
    feat = rng.uniform(0, 1, n) + 0.2 * ae
    posts = rng.poisson(3, n).astype(float)
    tbin = np.minimum(ten // 4, 5)
    eff = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])[tbin]
    sp_eff = rng.normal(size=s.max() + 1)[s]
    wk_eff = rng.normal(size=weeks)[w]
    eng = eff * feat + 0.05 * posts + sp_eff + wk_eff + rng.normal(size=n) * (1 + 0.5 * feat)
    eng2 = 0.5 * eng + rng.normal(size=n)
    wt = rng.uniform(0.5, 2.0, n)
    eng[rng.uniform(size=n) < missing] = np.nan
    return pl.DataFrame({"id_a": a, "id_f": f, "spell": s, "week": w, "tenure": ten,
                         "tbin": tbin, "feat": feat, "posts": posts, "eng": eng,
                         "eng2": eng2, "wt": wt})
