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
