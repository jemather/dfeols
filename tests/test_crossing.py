"""Plan C's crossing-FE solve (dfeols.crossing): block pseudo-inverses, the bounded sum of the
shards' parts of A, the preconditioner choice and CG."""

import numpy as np
import pytest
import scipy.sparse as sp

from dfeols import crossing as cx


def _gram(levels: int, groups: int, rows: int, n_local: int, seed: int):
    """A_kk = D'W Q D for one crossing FE of `levels` levels, Q demeaning by `n_local` FEs."""
    rng = np.random.default_rng(seed)
    f = rng.integers(0, levels, rows)
    D = sp.csr_matrix((np.ones(rows), (np.arange(rows), f)), shape=(rows, levels)).toarray()
    L = np.hstack([np.eye(groups)[rng.integers(0, groups, rows)] for _ in range(n_local)])  # local FE dummies
    Q = np.eye(rows) - L @ np.linalg.pinv(L)
    return D.T @ Q @ D


@pytest.mark.parametrize("n_local", [1, 2])
def test_block_solve_is_the_pseudo_inverse(n_local):
    # with two local FEs the block has more null directions than one constant per component:
    # pinning alone leaves it singular, so the solve must not be a plain inverse
    B = _gram(levels=60, groups=25, rows=300, n_local=n_local, seed=n_local)
    s = cx.BlockSolve(sp.csr_matrix(B), singular=True, name="f")
    r = B @ np.random.default_rng(0).standard_normal((60, 3))  # in B's range, as CG's residuals are
    np.testing.assert_allclose(s.solve(r), np.linalg.pinv(B, rcond=1e-10, hermitian=True) @ r, atol=1e-8)
    assert s.kind == "exact"


def test_block_solve_large_components(monkeypatch):
    # components above SMALL: pinned Cholesky, or the pseudo-inverse when that is singular
    monkeypatch.setattr(cx.BlockSolve, "SMALL", 10)
    for n_local in (1, 2):
        B = _gram(levels=80, groups=10, rows=400, n_local=n_local, seed=7)
        s = cx.BlockSolve(sp.csr_matrix(B), singular=True, name="f")
        r = B @ np.random.default_rng(1).standard_normal((80, 2))
        np.testing.assert_allclose(s.solve(r), np.linalg.pinv(B, rcond=1e-10, hermitian=True) @ r, atol=1e-8)


def test_amg_blocks_on_a_sparse_component():
    # worker-firm graph: one large sparse firm component, solved by AMG V-cycles; block-Jacobi CG
    # reaches the same solution as Jacobi in fewer steps
    pytest.importorskip("pyamg")
    rng = np.random.default_rng(4)
    workers, firms = 4000, 1500
    w = np.repeat(np.arange(workers), 2)
    f = rng.integers(0, firms, 2 * workers)
    n = np.ones(len(w))
    S = sp.csr_matrix((n, (w, f)), shape=(workers, firms))
    B = (
        sp.diags(np.asarray(S.sum(0)).ravel()) - S.T @ sp.diags(1 / np.asarray(S.sum(1)).ravel()) @ S
    ).tocsr()
    s = cx.BlockSolve(B, singular=True, name="firm")
    assert s.kinds["amg"] > 1000 and cx.choose_method([s], level_space=True) == "bjacobi"
    R = B @ rng.standard_normal((firms, 2))
    d = B.diagonal()
    act = d > 1e-12
    offs = np.array([0, firms])
    Xj, itj, _ = cx.pcg(lambda p: B @ p, R, cx.make_precond("jacobi", d, offs), act, 1e-10, 2000)
    Xa, ita, rel = cx.pcg(lambda p: B @ p, R, cx.make_precond("bjacobi", d, offs, [s]), act, 1e-10, 2000)
    assert rel <= 1e-10 and ita < itj
    np.testing.assert_allclose(B @ Xa, B @ Xj, atol=1e-7)


def test_level_gram_gives_up_above_budget():
    offs = np.array([0, 3, 5])
    part = {
        "full": True,
        "blocks": True,
        "nnz": 6,
        "r": np.array([0, 1, 3, 0, 4, 2]),
        "c": np.array([0, 1, 3, 4, 0, 2]),
        "v": np.ones(6),
    }
    g = cx.LevelGram(offs, ["a", "b"], max_nnz=100)
    g.add(dict(part), 0)
    g.add(dict(part), 1)
    A, blocks = g.finish()
    assert A is not None and A[0, 4] == 2 and [b.nnz for b in blocks] == [3, 1]
    # A over the budget: only the diagonal blocks are kept, and the reason names the blocks
    g = cx.LevelGram(offs, ["a", "b"], max_nnz=4)
    g.add(dict(part), 0)
    A, blocks = g.finish()
    assert A is None and [b.nnz for b in blocks] == [3, 1] and "a x b" in g.reason
    # a shard whose own blocks are too large: nothing but Jacobi
    g = cx.LevelGram(offs, ["a", "b"], max_nnz=4)
    g.add({"full": False, "blocks": False, "nnz": 9, "block_nnz": 7}, 3)
    assert g.finish() == (None, None) and "shard 3" in g.reason


def test_choose_method():
    B = sp.csr_matrix(_gram(20, 8, 100, 1, 0))
    exact = cx.BlockSolve(B, True)
    assert cx.choose_method([exact], level_space=True) == "sgs"
    assert cx.choose_method([exact], level_space=False) == "bjacobi"
    assert cx.choose_method(None, level_space=False) == "jacobi"
    assert cx.choose_method([exact], level_space=False, want="sgs") == "bjacobi"
    with pytest.raises(ValueError):
        cx.choose_method([exact], True, want="amg")


def test_pcg_solves_a_consistent_singular_system():
    B = _gram(40, 12, 200, 1, 3)
    R = B @ np.random.default_rng(2).standard_normal((40, 2))
    act = B.diagonal() > 1e-12
    M = cx.make_precond("jacobi", B.diagonal(), np.array([0, 40]))
    X, it, rel = cx.pcg(lambda p: B @ p, R, M, act, 1e-10, 500)
    assert rel <= 1e-10 and it < 500
    np.testing.assert_allclose(B @ X, R, atol=1e-8)


@pytest.mark.parametrize("max_nnz", [10**6, 1000, 800, 400, 20])  # A; A; blocks only; none; none
def test_combine_parts_as_level_gram(max_nnz):
    # summing parts on the workers keeps LevelGram's rules: A within the budget, else its
    # diagonal blocks, else nothing; the coordinator then ends with the same A and blocks
    from dfeols.crossing import LevelGram, combine_parts

    rng = np.random.default_rng(0)
    offs = np.array([0, 30, 50])
    parts = []
    for _ in range(9):
        r, c = rng.integers(0, 50, 60), rng.integers(0, 50, 60)
        S = sp.csr_matrix((rng.random(60), (r, c)), shape=(50, 50))
        S = (S + S.T).tocoo()
        parts.append(
            {
                "full": True,
                "blocks": True,
                "nnz": S.nnz,
                "r": S.row.astype(np.int32),
                "c": S.col.astype(np.int32),
                "v": S.data,
            }
        )
    one = LevelGram(offs, ["a", "b"], max_nnz)
    for i, p in enumerate(parts):
        one.add(p, i)
    tree = LevelGram(offs, ["a", "b"], max_nnz)
    kw = dict(offs=offs, max_nnz=max_nnz)
    tree.add(combine_parts(combine_parts(*parts[:4], **kw), combine_parts(*parts[4:], **kw), **kw), None)
    (A1, B1), (A2, B2) = one.finish(), tree.finish()
    assert one.full == tree.full and one.blocks_ok == tree.blocks_ok
    for x, y in [(A1, A2)] + list(zip(B1 or [], B2 or [])):
        assert (x is None) == (y is None)
        if x is not None:
            assert abs(x - y).max() < 1e-12
    if not one.blocks_ok:
        assert B1 is None and B2 is None


def test_step_combine_sums_levels_like_the_coordinator():
    # CG steps' tree reduction: shards' (levels, sums) merged on workers equal the
    # coordinator adding every shard's sums into a dense P x c block
    rng = np.random.default_rng(3)
    P, c = 40, 3
    parts = []
    for _ in range(7):
        idx = np.sort(rng.choice(P, size=rng.integers(1, 15), replace=False))
        parts.append({"idx": idx, "S": rng.standard_normal((len(idx), c)), "timing": {"total": 0.1}})
    dense = np.zeros((P, c))
    for p in parts:
        dense[p["idx"]] += p["S"]
    a = cx.step_combine(*(cx.step_part(p) for p in parts[:3]))
    b = cx.step_combine(*(cx.step_part(p) for p in parts[3:]), None)
    lv, S = cx.step_combine(a, b)
    tree = np.zeros((P, c))
    tree[lv] = S
    np.testing.assert_allclose(tree, dense, rtol=1e-13)
    assert cx.step_strip(parts[0]) == {"timing": {"total": 0.1}}
    assert cx.step_combine(None) is None


def test_pcg_resumes_from_a_checkpoint(tmp_path):
    B = _gram(40, 12, 200, 1, 3)
    R = B @ np.random.default_rng(2).standard_normal((40, 2))
    act = B.diagonal() > 1e-12
    M = cx.make_precond("jacobi", B.diagonal(), np.array([0, 40]))
    X0, it0, _ = cx.pcg(lambda p: B @ p, R, M, act, 1e-10, 500)
    ck = cx.Checkpoint(tmp_path / "cg.npz", R, 1e-10, "jacobi")
    X, it, rel = cx.pcg(lambda p: B @ p, R, M, act, 1e-10, 5, on_state=ck.save)
    assert it == 5 and rel > 1e-10  # stopped early
    state = ck.load()
    assert int(state["it"]) == 5
    X1, it1, _ = cx.pcg(lambda p: B @ p, R, M, act, 1e-10, 500, state=state)
    assert it1 == it0
    np.testing.assert_array_equal(X1, X0)  # the same steps, bit for bit
    assert cx.Checkpoint(tmp_path / "cg.npz", R * (1 + 1e-14), 1e-10, "jacobi").load() is not None
    assert cx.Checkpoint(tmp_path / "cg.npz", R * 1.01, 1e-10, "jacobi").load() is None
    assert cx.Checkpoint(tmp_path / "cg.npz", R, 1e-8, "jacobi").load() is None


def test_iter_cache_budget_and_earlier_fits(monkeypatch):
    # a worker keeps one fit's shard state (D5 job 35888031: fare's 8 GB stayed through
    # pair x year's round 1), within its plan's budget, counting every array it holds
    from dfeols import worker as wk

    monkeypatch.delenv("DFEOLS_ITER_CACHE_GB", raising=False)
    monkeypatch.setattr(wk, "_ITER_CACHE", type(wk._ITER_CACHE)())
    n, P = 1000, 50
    Dm = sp.csr_matrix((np.ones(n), (np.arange(n), np.arange(n) % P)), shape=(n, P))
    codes, w = [np.zeros(n, np.int32)], np.ones(n)
    arrays = (codes[0], Dm.data, Dm.indices, Dm.indptr, w, np.arange(P, dtype=Dm.indices.dtype))
    one = sum(a.nbytes for a in arrays)
    assert cx.iter_cache_budget(24_000_000_000) == 6_000_000_000

    old = wk.Plan.__new__(wk.Plan)
    old.iter_token, old.iter_cache_bytes = "fare", 10 * one
    for sid in range(3):
        wk._iter_store(old, sid, codes, Dm, w)
    assert wk._iter_store(old, 9, codes, Dm, w)[-1] == one
    new = wk.Plan.__new__(wk.Plan)
    new.iter_token, new.iter_cache_bytes = "pairyear", int(2.5 * one)
    for sid in range(4):
        wk._iter_store(new, sid, codes, Dm, w)
    assert list(wk._ITER_CACHE) == [("pairyear", 2), ("pairyear", 3)]  # fare's gone, LRU kept 2
    monkeypatch.setenv("DFEOLS_ITER_CACHE_GB", str(3.5 * one / 1e9))
    for sid in range(4, 8):
        wk._iter_store(new, sid, codes, Dm, w)
    assert len(wk._ITER_CACHE) == 3


def test_iter_cache_threads(monkeypatch):
    # a worker's threads share the cache: job 35991256 lost 43 round-1 tasks to KeyError when
    # threads starting a new fit cleared the previous fit's entries together. Many threads
    # store and evict (budget of 3 entries) while a new fit's first stores clear the old one.
    import sys
    import threading

    from dfeols import worker as wk

    monkeypatch.delenv("DFEOLS_ITER_CACHE_GB", raising=False)
    monkeypatch.setattr(wk, "_ITER_CACHE", type(wk._ITER_CACHE)())
    monkeypatch.setattr(sys, "getswitchinterval", sys.getswitchinterval)
    sys.setswitchinterval(1e-6)  # switch threads as often as possible
    n, P = 200, 20
    Dm = sp.csr_matrix((np.ones(n), (np.arange(n), np.arange(n) % P)), shape=(n, P))
    codes, w = [np.zeros(n, np.int32)], np.ones(n)
    one = wk._iter_store(wk.Plan.__new__(wk.Plan), 0, codes, Dm, w)[-1]  # no token: not cached
    errors = []
    for rnd in range(30):
        old = wk.Plan.__new__(wk.Plan)
        old.iter_token, old.iter_cache_bytes = f"old{rnd}", 1000 * one
        for sid in range(200):
            wk._iter_store(old, sid, codes, Dm, w)
        new = wk.Plan.__new__(wk.Plan)
        new.iter_token, new.iter_cache_bytes = f"new{rnd}", 3 * one
        go = threading.Barrier(8)

        def work(t, go=go, new=new):
            try:
                go.wait()
                for sid in range(t, 64, 8):
                    wk._iter_store(new, sid, codes, Dm, w)
                    with wk._ITER_LOCK:
                        wk._ITER_CACHE.get((new.iter_token, sid))
            except Exception as e:
                errors.append(e)

        ts = [threading.Thread(target=work, args=(t,)) for t in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not errors, errors[:3]
        assert all(k[0] == new.iter_token for k in wk._ITER_CACHE) and len(wk._ITER_CACHE) <= 3
