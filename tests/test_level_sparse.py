"""Round 1's per-level outputs travel sparse (D6 r4, OSCER job 35953385: 15.5M location-days, so
each shard's dense present mask and cluster min/max came to ~265 MB, 1.15 TB through the
coordinator, and round 1 took ~4 h). Shards now send their own levels only; merged on the
coordinator they give the same masks and nesting as the dense arrays, which still merge."""

import numpy as np

from dfeols import worker as wk


def _dense(codes, cl, L):
    lo = np.full(L, np.iinfo(np.int64).max)
    hi = np.full(L, np.iinfo(np.int64).min)
    np.minimum.at(lo, codes, cl)
    np.maximum.at(hi, codes, cl)
    return np.bincount(codes, minlength=L) > 0, lo, hi


def test_sparse_merge_matches_dense():
    rng = np.random.default_rng(0)
    L = 1000
    acc_d = [np.zeros(L, bool), np.full(L, np.iinfo(np.int64).max), np.full(L, np.iinfo(np.int64).min)]
    acc_s = [a.copy() for a in acc_d]
    for s in range(7):
        n = int(rng.integers(0, 400))  # one shard with few rows, maybe none
        codes = rng.integers(0, L // 3 if s % 2 else L, n).astype(np.int64)
        cl = rng.integers(0, 50, n).astype(np.int64)
        pres, lo, hi = _dense(codes, cl, L)
        wk.merge_present(acc_d[0], pres)  # a dense mask, as older shard results
        wk.merge_minmax(acc_d[1], acc_d[2], (lo, hi))
        lev, slo, shi = wk.level_minmax(codes, cl)
        assert np.array_equal(lev, np.unique(codes)) and len(slo) == len(lev)
        wk.merge_present(acc_s[0], np.unique(codes))
        wk.merge_minmax(acc_s[1], acc_s[2], (lev, slo, shi))
    for a, b in zip(acc_d, acc_s):
        assert np.array_equal(a, b)
    p = acc_s[0]
    assert (acc_s[1][p] == acc_d[1][p]).all() and (acc_s[2][p] == acc_d[2][p]).all()


def test_level_minmax_payload_is_the_shards_levels():
    codes = np.array([5, 9_999_999, 5, 42], np.int64)
    lev, lo, hi = wk.level_minmax(codes, np.array([3, 1, 7, 2], np.int64))
    assert lev.tolist() == [5, 42, 9_999_999]
    assert lo.tolist() == [3, 2, 1] and hi.tolist() == [7, 2, 1]
