# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Crossing-FE solve (plan C): conjugate gradient on A Gamma = D~'W [X~ Y~], A = D~'W D~.

Two modes (docs/design/planner.md, "Crossing-FE solver"):

- level space: every shard returns its part of A in round 1 (`LevelGram` sums them), and CG runs
  on the coordinator with no further passes over the data;
- fallback: A would exceed the coordinator's budget, so every product with A is a pass over the
  shards (estimate._crossing_cg); the preconditioner uses each crossing FE's own block A_kk when
  those fit. dfeols warns that this may converge slowly (`SlowConvergenceWarning`).

The preconditioner is chosen from the connected components of each A_kk (levels linked through
local groups): exact pinned solves when every component is small or dense enough to factorise,
AMG V-cycles (plain aggregation; needs pyamg) on large sparse ones, else the diagonal. In level
space, exact blocks give the AP-sweep (symmetric block Gauss-Seidel over the crossing FEs), AMG
blocks block-Jacobi; the fallback always uses block-Jacobi, which costs no extra pass.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

log = logging.getLogger("dfeols")

# Coordinator bytes per nonzero of A while it is built and used: the CSR matrix (12), COO buffers
# summed between compactions (16) and the copies compaction and the block solvers make
NNZ_BYTES = 40
# A's share of the coordinator's memory when the profile does not say otherwise
LEVEL_SPACE_SHARE = 0.25
# A worker's memory a combine of parts of A may use (inputs and the sums it builds; the rest is
# the process itself, Dask's buffers and the nanny's 95% limit)
WORKER_SHARE = 0.6
# A worker's memory plan C keeps shard state in between CG steps (worker._ITER_CACHE); it
# sits beside every task's own peak, so it takes less than a combine
ITER_CACHE_SHARE = 0.25
# Bytes per nonzero of a combine's inputs while it sums them, on top of the inputs themselves
# (one at a time: the sum so far and its next copy; measured 21-24 on 4 x 5M random parts)
MERGE_BYTES = 24
FAN_IN = 4  # parts per combine (executors.Reduce.fan_in)


class SlowConvergenceWarning(UserWarning):
    """The crossing-FE system is too large for level space: CG makes a data pass per step."""


def level_space_budget(executor, level_space_gb: float | None) -> int:
    """Most nonzeros of A the coordinator holds: `level_space_gb` if given (0 turns level space
    off), else LEVEL_SPACE_SHARE of the profile's coordinator memory, or of this machine's."""
    return profile_budget(getattr(executor, "profile", None), level_space_gb)


def profile_budget(prof, level_space_gb: float | None) -> int:
    """level_space_budget from a site profile (or None: this machine), for the planner too."""
    if level_space_gb is None:
        mem = (getattr(prof, "coordinator", None) or {}).get("memory") if prof is not None else None
        if mem:
            from .cluster import _memory_bytes

            have = _memory_bytes(mem)
        else:
            import psutil

            have = psutil.virtual_memory().total
        level_space_gb = LEVEL_SPACE_SHARE * have / 1e9
    return int(level_space_gb * 1e9 / NNZ_BYTES)


def worker_bytes(executor=None, prof=None) -> int:
    """One worker's memory: the executor's (DaskExecutor.worker_memory), else the profile's
    worker memory, else this machine's (SerialExecutor runs the combines here)."""
    have = getattr(executor, "worker_memory", None)
    if have:
        return int(have)
    prof = prof if prof is not None else getattr(executor, "profile", None)
    mem = (getattr(prof, "worker", None) or {}).get("memory") if prof is not None else None
    if mem:
        from .cluster import _memory_bytes

        return _memory_bytes(mem)
    import psutil

    return int(psutil.virtual_memory().total)


def iter_cache_budget(wbytes: int) -> int:
    """Bytes of shard state a worker keeps between CG steps (DFEOLS_ITER_CACHE_GB overrides)."""
    return int(ITER_CACHE_SHARE * wbytes)


def part_budget(wbytes: int, fan_in: int = FAN_IN) -> int:
    """Most nonzeros of A one shard returns: fan_in such parts must fit in one combine on a
    worker of `wbytes` (taxi2009: 4 parts of 100M+ nonzeros killed 24 GB workers, because the
    shards' limit came from the coordinator's memory)."""
    return int(WORKER_SHARE * wbytes / (NNZ_BYTES * fan_in))


def worker_a_budget(wbytes: int) -> int:
    """Most nonzeros of A the workers can sum: the last combine holds A and its inputs (the planner)."""
    return int(WORKER_SHARE * wbytes / NNZ_BYTES)


def _part_bytes(p: dict) -> int:
    return sum(a.nbytes for a in p.values() if isinstance(a, np.ndarray))


def diagonal_blocks(rows: np.ndarray, cols: np.ndarray, fe_of: np.ndarray) -> np.ndarray:
    """Mask of entries (rows, cols) inside one crossing FE's own block."""
    return fe_of[rows] == fe_of[cols]


def part_of(res: dict):
    """A round-1 result's part of A (executors.Reduce.get)."""
    return (res.get("crossing") or {}).get("part")


def without_part(res: dict) -> dict:
    """A round-1 result without its part of A (executors.Reduce.strip)."""
    if "crossing" not in res:
        return res
    cr = {k: v for k, v in res["crossing"].items() if k != "part"}
    return {**res, "crossing": cr}


def combine_parts(*parts, offs: np.ndarray, max_nnz: int, worker_bytes: int | None = None) -> dict:
    """Sum shards' parts of A (worker.shard_round1's `part`, or earlier sums) on a worker, by
    LevelGram's rules: past `max_nnz` keep only the diagonal blocks, past it again nothing.
    Round 1 sums the parts in a tree on the workers (DaskExecutor's `reduce`), so the
    coordinator receives A once rather than every shard's part (DB1B, 683M rows: 3.4e9
    nonzeros shipped for an A of 5.7e7; round 1 took 755 s, most of it on the coordinator).

    With `worker_bytes` the same rules hold for the worker's memory: the parts are added one
    at a time, and once the inputs plus the sum (MERGE_BYTES a nonzero) would pass
    WORKER_SHARE of it, only the diagonal blocks are kept, then nothing, instead of the
    worker being killed."""
    fe_of = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
    P = int(offs[-1])
    out = {"full": True, "blocks": True, "parts_nnz": 0, "shards": 0, "reason": None}
    for p in parts:
        if p is None or p.get("none"):  # a shard could not form its part
            return {"none": True, "reason": (p or {}).get("reason") or "a shard could not form its part of A"}
        out["parts_nnz"] += p.get("parts_nnz", p["nnz"])
        out["shards"] += p.get("shards", 1)
        if not p["full"] and out["full"]:
            out["full"] = False
            out["reason"] = p.get("reason") or f"one shard's part alone has {p['nnz']:,} nonzeros"
        if not p["blocks"]:
            out["blocks"] = False
            out["reason"] = (
                p.get("reason") or f"one shard's diagonal blocks alone have {p['block_nnz']:,} nonzeros"
            )
    if not out["blocks"]:
        out.update(nnz=0, block_nnz=0)
        return out
    room = WORKER_SHARE * worker_bytes - sum(map(_part_bytes, parts)) if worker_bytes else np.inf
    S = None
    for p in parts:  # one at a time: the sum so far and its next copy, not every part as CSR
        M = part_matrix(p, P)
        if not out["full"]:
            M = _blocks_only(M, fe_of)
        n_next = (S.nnz if S is not None else 0) + M.nnz
        if out["full"] and MERGE_BYTES * n_next > room:
            out["full"] = False
            out["reason"] = (
                f"summing the parts on a worker needs more than its memory "
                f"({worker_bytes / 1e9:.0f} GB) allows"
            )
            M = _blocks_only(M, fe_of)
            S = _blocks_only(S, fe_of) if S is not None else None
            n_next = (S.nnz if S is not None else 0) + M.nnz
        if MERGE_BYTES * n_next > room:
            out.update(blocks=False, nnz=0, block_nnz=int(n_next))
            out["reason"] = (
                f"summing the diagonal blocks on a worker needs more than its memory "
                f"({worker_bytes / 1e9:.0f} GB) allows"
            )
            return out
        S = M if S is None else S + M  # CSR + CSR merges sorted rows (2.5x the COO route)
        del M
    if out["full"] and S.nnz > max_nnz:
        out["full"] = False
        out["reason"] = f"the shards' parts summed have more than {max_nnz:,} nonzeros"
        S = _blocks_only(S, fe_of)
    if S.nnz > max_nnz:
        out.update(blocks=False, nnz=0, block_nnz=int(S.nnz))
        out["reason"] = f"the diagonal blocks summed have more than {max_nnz:,} nonzeros"
        return out
    it = np.int32 if P < 2**31 else np.int64
    out.update(
        nnz=int(S.nnz),
        block_nnz=int(S.nnz),
        indptr=S.indptr.astype(np.int64),
        indices=S.indices.astype(it),
        data=S.data,
    )
    return out


def part_matrix(p: dict, P: int) -> sp.csr_matrix:
    """A part of A as CSR: a shard's (r, c, v), or a sum's (indptr, indices, data)."""
    if "indptr" in p:
        return sp.csr_matrix((p["data"], p["indices"], p["indptr"]), shape=(P, P))
    return sp.csr_matrix((p["v"], (p["r"], p["c"])), shape=(P, P))


def _blocks_only(S: sp.csr_matrix, fe_of: np.ndarray) -> sp.csr_matrix:
    A = S.tocoo()
    keep = diagonal_blocks(A.row, A.col, fe_of)
    return sp.csr_matrix((A.data[keep], (A.row[keep], A.col[keep])), shape=S.shape)


class LevelGram:
    """A = D~'W D~ summed from the shards' parts within a budget of `max_nnz` nonzeros.

    Parts are buffered and compacted (duplicates summed) whenever the buffer passes the budget.
    Once A itself passes it, or a shard reports that its own part did, only the diagonal blocks
    A_kk are kept (the fallback's preconditioner), and those too are dropped above the budget.
    `reason` says what made A too large, for the fallback's warning."""

    def __init__(self, offs: np.ndarray, names: list[str], max_nnz: int):
        self.P = int(offs[-1])
        self.names = names
        self.offs = offs
        self.fe_of = np.repeat(np.arange(len(names)), np.diff(offs))
        self.max_nnz = max_nnz
        self.full, self.blocks_ok = True, True
        self.reason: str | None = None
        self.blocks_reason: str | None = None
        self.buf: list[tuple] = []
        self.buffered = 0
        self.A = sp.csr_matrix((self.P, self.P))
        self.parts_nnz = 0  # sum over shards' parts, for the log
        self.shipped_nnz = 0  # what reached the coordinator (the parts, or the workers' sum)

    def add(self, part: dict | None, shard) -> None:
        """Add one shard's part, or (shard None) the workers' sum of all of them (combine_parts)."""
        who = f"shard {shard}" if shard is not None else "a shard"
        # a shard could not form its part (local FE components too large)
        if part is None or part.get("none"):
            why = (part or {}).get("reason") or f"{who} could not form its part of A"
            self._give_up(why, blocks=False)
            return
        if not part["full"] and self.full:
            why = part.get("reason") or f"{who}'s part alone has {part['nnz']:,} nonzeros"
            self._give_up(why, blocks=True)
        if not part["blocks"]:
            why = part.get("reason") or f"{who}'s diagonal blocks alone have {part['block_nnz']:,} nonzeros"
            self._give_up(why, blocks=False)
        if not self.blocks_ok:
            return
        if "indptr" in part:  # the workers' sum (combine_parts)
            r = np.repeat(np.arange(self.P, dtype=part["indices"].dtype), np.diff(part["indptr"]))
            c, v = part["indices"], part["data"]
        else:
            r, c, v = part["r"], part["c"], part["v"]
        if not self.full:
            keep = diagonal_blocks(r, c, self.fe_of)
            r, c, v = r[keep], c[keep], v[keep]
        self.parts_nnz += part.get("parts_nnz", len(v))
        self.shipped_nnz += len(v)
        self.buf.append((r, c, v))
        self.buffered += len(v)
        if self.buffered > self.max_nnz:
            self._compact()

    def _compact(self) -> None:
        if self.buf:
            r, c, v = (np.concatenate(x) for x in zip(*self.buf))
            self.buf, self.buffered = [], 0
            self.A = (self.A + sp.csr_matrix((v, (r, c)), shape=(self.P, self.P))).tocsr()
        if self.full and self.A.nnz > self.max_nnz:
            self._give_up(self._size_reason(), blocks=True)
        if not self.full and self.blocks_ok and self.A.nnz > self.max_nnz:
            self._give_up(f"the diagonal blocks alone have more than {self.max_nnz:,} nonzeros", blocks=False)

    def _size_reason(self) -> str:
        """Which pairs of crossing FEs hold A's nonzeros, largest first."""
        A = self.A.tocoo()
        K = len(self.names)
        cnt = np.bincount(self.fe_of[A.row] * K + self.fe_of[A.col], minlength=K * K).reshape(K, K)
        pairs = sorted(
            ((cnt[i, j] + (cnt[j, i] if i != j else 0), i, j) for i in range(K) for j in range(i, K)),
            reverse=True,
        )
        top = ", ".join(
            f"{self.names[i]}" + (f" x {self.names[j]}" if i != j else "") + f" {n:,}"
            for n, i, j in pairs[:3]
            if n
        )
        lv = ", ".join(f"{n} {int(L):,}" for n, L in zip(self.names, np.diff(self.offs)))
        return (
            f"A has more than {A.nnz:,} nonzeros over {self.P:,} crossing levels ({lv}); "
            f"largest blocks: {top}"
        )

    def _give_up(self, reason: str, blocks: bool) -> None:
        """Leave level space (keep only the diagonal blocks); with blocks=False drop those too."""
        if self.full:
            self.full = False
            self.reason = reason
            A = self.A.tocoo()
            keep = diagonal_blocks(A.row, A.col, self.fe_of)
            self.A = sp.csr_matrix((A.data[keep], (A.row[keep], A.col[keep])), shape=A.shape)
            self.buf = [tuple(x[diagonal_blocks(b[0], b[1], self.fe_of)] for x in b) for b in self.buf]
            self.buffered = sum(len(b[2]) for b in self.buf)
        if not blocks and self.blocks_ok:
            self.blocks_ok = False
            self.blocks_reason = reason
            self.A = sp.csr_matrix((self.P, self.P))
            self.buf, self.buffered = [], 0

    def finish(self):
        """(A or None, [A_kk] or None)."""
        self._compact()
        if not self.blocks_ok:
            return None, None
        A = ((self.A + self.A.T) * 0.5).tocsr()
        A.eliminate_zeros()
        blocks = [A[o0:o1, o0:o1].tocsr() for o0, o1 in zip(self.offs[:-1], self.offs[1:])]
        return (A if self.full else None), blocks


# ---------------------------------------------------------------- block solves


def _pinv(B: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Pseudo-inverses of symmetric PSD matrices (stacked on the first axis or one), by eigh."""
    lam, V = np.linalg.eigh(B)
    cut = rcond * np.max(np.abs(lam), axis=-1, keepdims=True)
    inv = np.where(lam > cut, 1.0 / np.where(lam > cut, lam, 1.0), 0.0)
    return (V * inv[..., None, :]) @ np.swapaxes(V, -1, -2)


class _Cholesky:
    """Cholesky of a pinned block; raises LinAlgError if the block is (numerically) singular, as
    when several local FEs absorb more of the crossing FE than one constant per component."""

    def __init__(self, B):
        import scipy.linalg as sla

        self.sla = sla
        self.c = sla.cho_factor(B, lower=True, check_finite=False)
        piv = np.diagonal(self.c[0]) ** 2
        if piv.min(initial=np.inf) < 1e-10 * max(float(np.diagonal(B).max(initial=0.0)), 1e-300):
            raise np.linalg.LinAlgError("pinned block is singular")

    def solve(self, r):
        return self.sla.cho_solve(self.c, r, check_finite=False)


class _Dense:
    def __init__(self, Binv):
        self.Binv = Binv

    def solve(self, r):
        return self.Binv @ r


class _AMG:
    """One V-cycle of plain aggregation AMG (pyamg, smooth=None): the default smoothed
    aggregation fills in hundreds of times on worker-firm graphs (planner.md)."""

    def __init__(self, B):
        import pyamg

        self.M = pyamg.smoothed_aggregation_solver(
            B.tocsr(), symmetry="hermitian", max_coarse=500, smooth=None
        ).aspreconditioner(cycle="V")

    def solve(self, r):
        return np.column_stack([self.M @ r[:, j] for j in range(r.shape[1])])


class BlockSolve:
    """Approximate or exact pseudo-inverse of one crossing FE's own block B = A_kk.

    With a local FE, B is singular: a constant on each connected component of its levels is
    absorbed by the local FE. A component's solve pins one level, solves for the rest, and
    removes the component mean before and after; regularising (B + eps I) amplifies rounding by
    1/eps and stalls CG near 1e-6 (NFIP A). Without a local FE, B is diagonal and solved as is.

    Per component: up to SMALL levels, batched pseudo-inverses (eigh; exact whatever the null
    space); up to DENSE_MAX levels and at least 10% full, Cholesky of the pinned block, or its
    pseudo-inverse up to PINV_MAX levels where the pinned block is still singular (several local
    FEs); larger and sparse, AMG (if pyamg is installed); otherwise the diagonal. Every null
    direction of B is one of A's (D_k v absorbed by the local FEs), so CG's residuals are
    orthogonal to it. `kind` is "exact", "amg", "diag" or "mixed"."""

    SMALL = 400
    DENSE_MAX = 20_000
    PINV_MAX = 8_000

    def __init__(self, B, singular: bool, name: str = ""):
        from scipy.sparse.csgraph import connected_components

        t = time.time()
        B = B.tocsr()
        self.P = B.shape[0]
        ncomp, lab = connected_components(B, directed=False)
        self.lab, self.ncomp, self.singular = lab, ncomp, singular
        cnt = np.bincount(lab, minlength=ncomp)
        self.cnt = cnt.astype(np.float64)
        order = np.argsort(lab, kind="stable")
        starts = np.r_[0, np.cumsum(cnt)]
        d = B.diagonal()
        self.diag_idx: list[np.ndarray] = []
        self.dinv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        by_size: dict[int, list] = {}
        self.large: list[tuple] = []
        kinds = {"exact": 0, "amg": 0, "diag": 0}
        try:
            import pyamg  # noqa: F401

            have_amg = True
        except ImportError:
            have_amg = False
        for c in range(ncomp):
            full = order[starts[c] : starts[c + 1]]
            if len(full) <= self.SMALL:
                by_size.setdefault(len(full), []).append(full)
                kinds["exact"] += len(full)
                continue
            idx = full[1:] if singular else full  # pinned: the first level
            Bc = B[idx][:, idx]
            if len(idx) <= self.DENSE_MAX and Bc.nnz > 0.1 * len(idx) ** 2:
                try:
                    self.large.append((idx, _Cholesky(Bc.toarray())))
                    kinds["exact"] += len(full)
                    continue
                except np.linalg.LinAlgError:
                    pass
                if len(full) <= self.PINV_MAX:
                    self.large.append((full, _Dense(_pinv(B[full][:, full].toarray()))))
                    kinds["exact"] += len(full)
                else:
                    self.diag_idx.append(full)
                    kinds["diag"] += len(full)
            elif have_amg and Bc.nnz <= 0.1 * len(idx) ** 2:
                self.large.append((full, _AMG(B[full][:, full])))
                kinds["amg"] += len(full)
            else:
                self.diag_idx.append(full)
                kinds["diag"] += len(full)
        self.small = []
        for idxs in by_size.values():
            I = np.array(idxs)
            self.small.append((I, _pinv(np.stack([B[i][:, i].toarray() for i in I]))))
        self.diag_idx = np.concatenate(self.diag_idx) if self.diag_idx else np.zeros(0, np.int64)
        used = [k for k, v in kinds.items() if v]
        self.kind = used[0] if len(used) == 1 else ("mixed" if used else "exact")
        self.kinds = kinds
        log.info(
            "crossing FE %s: %s levels, %s nonzeros, %s components (largest %s): %s (%.1fs)",
            name,
            f"{self.P:,}",
            f"{B.nnz:,}",
            f"{ncomp:,}",
            f"{int(cnt.max(initial=0)):,}",
            ", ".join(f"{v:,} levels {k}" for k, v in kinds.items() if v) or "empty",
            time.time() - t,
        )
        if self.diag_idx.size and not have_amg:
            log.info("crossing FE %s: install pyamg (dfeols[amg]) for AMG on its large components", name)

    def _demean(self, v):
        if not self.singular:
            return v
        mu = np.column_stack(
            [np.bincount(self.lab, weights=v[:, j], minlength=self.ncomp) for j in range(v.shape[1])]
        )
        return v - (mu / self.cnt[:, None])[self.lab]

    def solve(self, r):
        r = self._demean(r)
        z = np.zeros_like(r)
        for I, Binv in self.small:
            z[I] = Binv @ r[I]
        for idx, f in self.large:
            z[idx] = f.solve(np.ascontiguousarray(r[idx]))
        if self.diag_idx.size:
            z[self.diag_idx] = r[self.diag_idx] * self.dinv[self.diag_idx, None]
        return self._demean(z)


# ---------------------------------------------------------------- preconditioners


def choose_method(solvers: list[BlockSolve] | None, level_space: bool, want: str = "auto") -> str:
    """The preconditioner, from the block structure (planner.md's table) unless `want` names one."""
    if want != "auto":
        if want not in ("jacobi", "bjacobi", "sgs"):
            raise ValueError("crossing_precond must be auto, jacobi, bjacobi or sgs")
        if want != "jacobi" and solvers is None:
            log.warning("crossing_precond=%r needs the diagonal blocks, which are too large: Jacobi", want)
            return "jacobi"
        if want == "sgs" and not level_space:
            log.warning("crossing_precond='sgs' costs 3 passes a step outside level space: block-Jacobi")
            return "bjacobi"
        return want
    if solvers is None or all(s.kind == "diag" for s in solvers):
        return "jacobi"
    if level_space and all(s.kind == "exact" for s in solvers):
        return "sgs"
    return "bjacobi"


def make_precond(method: str, diag: np.ndarray, offs: np.ndarray, solvers=None, A=None):
    """z = M^-1 r for CG. `diag` is A's diagonal (levels with none are inactive), `offs` the
    crossing FEs' offsets in the level index, `A` the whole matrix (sgs only)."""
    act = diag > 1e-12 * max(float(diag.max(initial=0.0)), 1.0)
    dinv = np.where(act, 1.0 / np.where(act, diag, 1.0), 0.0)
    if method == "jacobi":
        return lambda r: r * dinv[:, None]
    sl = [slice(int(a), int(b)) for a, b in zip(offs[:-1], offs[1:])]
    K = len(sl)
    if method == "bjacobi":

        def bj(r):
            z = np.empty_like(r)
            for k in range(K):
                z[sl[k]] = solvers[k].solve(np.ascontiguousarray(r[sl[k]]))
            return z * act[:, None]

        return bj
    if method == "sgs":
        # symmetric block Gauss-Seidel over the crossing FEs (1, ..., K, ..., 1): one
        # alternating-projections sweep in level space. off[k]: A's rows of FE k without its block
        off = []
        for k in range(K):
            Ak = A[sl[k]].tocsr().tocoo()
            m = (Ak.col < sl[k].start) | (Ak.col >= sl[k].stop)
            off.append(sp.csr_matrix((Ak.data[m], (Ak.row[m], Ak.col[m])), shape=Ak.shape))

        def sgs(r):
            z = np.zeros_like(r)
            for k in range(K):  # forward: z of FEs not yet solved is still 0
                z[sl[k]] = solvers[k].solve(np.ascontiguousarray(r[sl[k]] - off[k] @ z))
            for k in range(K - 2, -1, -1):
                z[sl[k]] = solvers[k].solve(np.ascontiguousarray(r[sl[k]] - off[k] @ z))
            return z * act[:, None]

        return sgs
    raise ValueError(method)


def pcg(
    matvec,
    R: np.ndarray,
    precond,
    act: np.ndarray,
    tol: float,
    maxiter: int,
    on_step=None,
    state: dict | None = None,
    on_state=None,
):
    """Preconditioned CG for A X = R, one column per right-hand side, in lockstep: stops when
    every column's residual is below `tol` relative to its right-hand side. A is singular (one
    redundant level per connected component of the FEs) but the system is consistent, so CG
    converges to a solution whose D~ X, the only part used, is unique. `on_step(it, rel)` is
    called after each step and may raise. Returns X, steps and the largest relative residual.

    `state` (X, r, p, rz, it, as `on_state(state)` receives it after each step) resumes an
    interrupted solve exactly where it stopped: CG's whole state is these few P x c blocks."""
    P, c = R.shape
    r0 = R * act[:, None]
    bnorm = np.linalg.norm(r0, axis=0)
    bnorm[bnorm == 0] = 1.0
    if state is not None:
        X, r, p = state["X"].copy(), state["r"].copy(), state["p"].copy()
        rz, it = state["rz"].copy(), int(state["it"])
    else:
        X = np.zeros((P, c))
        r = r0
        z = precond(r)
        p = z.copy()
        rz = (r * z).sum(0)
        it = 0
    rel = np.linalg.norm(r, axis=0) / bnorm
    while rel.max() > tol:
        if it >= maxiter:
            return X, it, float(rel.max())
        it += 1
        Ap = matvec(p)
        pAp = (p * Ap).sum(0)
        alpha = np.divide(rz, pAp, out=np.zeros(c), where=pAp > 0)
        X += alpha * p
        r -= alpha * Ap
        z = precond(r)
        rz_new = (r * z).sum(0)
        beta = np.divide(rz_new, rz, out=np.zeros(c), where=rz > 0)
        p = z + beta * p
        rz = rz_new
        rel = np.linalg.norm(r, axis=0) / bnorm
        if on_state is not None:
            on_state({"X": X, "r": r, "p": p, "rz": rz, "it": it})
        if on_step is not None:
            on_step(it, rel)
    return X, it, float(rel.max())


class Checkpoint:
    """A CG solve's state on disk, written after each step (atomically) and read back by a
    rerun of the same solve: the same shape, tolerance and preconditioner, and a right-hand
    side equal to 1e-10 (shards' sums arrive in another order on a rerun, so not to the
    bit; the difference is far below any CG tolerance). Removed once the solve converges."""

    RTOL = 1e-10

    def __init__(self, path, R: np.ndarray, tol: float, method: str):
        self.path = Path(path)
        self.R = R
        self.digest = f"{R.shape}|{tol!r}|{method}"

    def load(self) -> dict | None:
        try:
            with np.load(self.path) as d:
                if str(d["digest"]) != self.digest:
                    return None
                if not np.allclose(
                    d["R"], self.R, rtol=self.RTOL, atol=self.RTOL * float(np.abs(self.R).max(initial=0.0))
                ):
                    return None
                return {k: d[k] for k in ("X", "r", "p", "rz", "it")}
        except (OSError, KeyError, ValueError):
            return None

    def save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, digest=self.digest, R=self.R, **state)
        os.replace(tmp, self.path)

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------- CG steps' tree reduction


def step_part(res: dict) -> tuple[np.ndarray, np.ndarray]:
    """A CG step's shard result as (levels, sums): what the tree adds up."""
    return np.asarray(res["idx"]), res["S"]


def step_strip(res: dict) -> dict:
    """The rest of a step's result, for the coordinator (timing only)."""
    return {k: v for k, v in res.items() if k not in ("idx", "S")}


def step_combine(*parts):
    """Sum shards' (levels, sums) on a worker: the union of the levels, sums added."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    idx = np.concatenate([p[0] for p in parts])
    uniq, inv = np.unique(idx, return_inverse=True)
    out = np.zeros((len(uniq), parts[0][1].shape[1]))
    np.add.at(out, inv, np.concatenate([p[1] for p in parts]))
    return uniq, out
