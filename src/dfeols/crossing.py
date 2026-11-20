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
# sits beside every task's own peak, so it takes less than a combine. 0.4 since CG steps hold
# only their shard's own levels (D11 L3: ~8 GB of shard state a 24 GB worker, peak RSS 8.2 GB)
ITER_CACHE_SHARE = 0.4
# Bytes per nonzero of a combine's inputs while it sums them, on top of the inputs themselves
# (one at a time: the sum so far and its next copy; measured 21-24 on 4 x 5M random parts)
MERGE_BYTES = 24
# Bytes per input entry while xtab_combine sums cross-tabs, on top of the inputs themselves (the
# keys, their sort order and sorted copy, then the values', and the sum; measured with
# tracemalloc on 4 x 5M random parts: 32 with the parts' keys all shared or half shared, 40 with
# none shared, where the sum is as large as its inputs)
XTAB_MERGE_BYTES = 40
# A worker's memory a round-1 combine with cross-tabs may use (the per-level sums' step_combine
# and xtab_combine) before it drops the cross-tab rather than be killed: nearer the nanny's 95%
# than WORKER_SHARE, because what it drops is the preconditioner (CG goes on with Jacobi, at
# ~40x the steps on D11 L3), not a level-space A the passes can stand in for
XTAB_WORKER_SHARE = 0.85
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


def level_sums_of(res: dict):
    """A plan C round-1 result's per-level sums, D'W [X Y] and D'W 1 side by side, as
    step_combine's (levels, sums) (executors.Reduce.get without level space), and its
    cross-tab (worker.shard_xtab) or None: ((levels, sums), xtab)."""
    cr = res.get("crossing")
    if not cr:
        return None
    return (np.asarray(cr["idx"]), np.hstack([cr["DR"], np.asarray(cr["ddiag"])[:, None]])), cr.get("xtab")


def without_level_sums(res: dict) -> dict:
    """A round-1 result without its per-level sums and cross-tab (executors.Reduce.strip)."""
    if "crossing" not in res:
        return res
    cr = {k: v for k, v in res["crossing"].items() if k not in ("idx", "DR", "ddiag", "xtab")}
    return {**res, "crossing": cr}


def level_sums_combine(*parts, P: int | None = None, worker_bytes: int | None = None, token: str | None = None):
    """Sum round-1 parts (level_sums_of): step_combine on the per-level sums, xtab_combine on
    the cross-tabs. With `worker_bytes` the cross-tabs' sum counts against the worker's memory
    (xtab_gb beside step_combine_gb): past XTAB_WORKER_SHARE of it the cross-tab is dropped
    (XTAB_DROPPED, and over_flag "xtab" for the shards still to run) instead of the worker being
    killed (D11 H: up to ~4 x 124M entries in round 1's last combine, ~28 GB)."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    sums = step_combine(*[p[0] for p in parts], P=P)
    xts = [p[1] for p in parts]
    if worker_bytes and not any(_xtab_dropped(x) for x in xts):
        c = parts[0][0][1].shape[1]
        n_lev = sum(len(p[0][0]) for p in parts)
        need = xtab_gb(sum(len(x[0]) for x in xts if x is not None))
        need += step_combine_gb(min(n_lev, P or n_lev), c, fan_in=len(parts)) - 0.2
        if need * 1e9 > XTAB_WORKER_SHARE * worker_bytes:
            over_flag(token, "xtab", set_it=True)
            xts = [
                xtab_dropped(
                    f"summing the shards' cross-tabs on a worker needs ~{need:.1f} GB, past "
                    f"{XTAB_WORKER_SHARE:.0%} of its {worker_bytes / 1e9:.1f} GB"
                )
            ]
    return sums, xtab_combine(*xts)


def xtab_gb(entries: int) -> float:
    """Memory xtab_combine needs for inputs of `entries` keys in all: the inputs (8-byte key,
    8-byte value) and XTAB_MERGE_BYTES of working copies an entry."""
    return entries * (16 + XTAB_MERGE_BYTES) / 1e9


def xtab_dropped(reason: str, skipped: bool = False) -> dict:
    """A cross-tab given up in round 1 (too large to sum on a worker): CG uses Jacobi.
    `skipped`: a shard's, which skipped its cross-tab after the verdict (worker.shard_round1)."""
    return {"none": True, "reason": reason, "skipped": skipped}


def _xtab_dropped(x) -> bool:
    return isinstance(x, dict) and bool(x.get("none"))


def xtab_combine(*parts):
    """Sum cross-tabs (sorted distinct keys, sums): the union of the keys, sums added. A dropped
    part (xtab_dropped) drops the sum: a cross-tab missing shards is not D'WD."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    dropped = [p for p in parts if _xtab_dropped(p)]
    if dropped:  # the verdict's own reason over a shard's "skipped"
        return min(dropped, key=lambda p: bool(p.get("skipped")))
    if len(parts) == 1:
        return parts[0]
    # one array at a time, each freed once gathered: the keys, their order and their sorted
    # copy, then the values' (40 bytes an entry at most, against 49 gathering both at once)
    k = np.concatenate([p[0] for p in parts])
    o = np.argsort(k, kind="stable")  # stable: timsort over the parts' sorted runs
    k = k[o]
    v = np.concatenate([p[1] for p in parts])[o]
    del o
    if not len(k):
        return k, v
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    return k[starts], np.add.reduceat(v, starts)


# Round 1's verdict on A, shared by its combines and the shards still to run (over_flag):
# once a combine finds the summed parts past the budget, the shards after it build only
# their diagonal blocks (or nothing), instead of every shard building a part the sum throws
# away (D6 r4 35948257: the parts came to ~4.5e10 nonzeros for a 4e8 budget, round 1 at 10%
# after 34 min). On Dask a named distributed.Event per verdict, else this process's set.
_OVER: set = set()


def _event(name: str):
    try:
        from distributed import Event, get_worker

        get_worker()
        return Event(name)
    except Exception:
        return None


def over_flag(token: str | None, what: str, set_it: bool = False) -> bool:
    """Round 1's verdict `what` ("full": A is past the budget, "blocks": its diagonal blocks
    too) for fit `token`: set it, or read it."""
    if not token:
        return False
    name = f"dfeols-over-{what}-{token}"
    ev = _event(name)
    try:
        if set_it:
            ev.set() if ev is not None else _OVER.add(name)
            return True
        return ev.is_set() if ev is not None else name in _OVER
    except Exception:  # advice only: a missed verdict costs time, not the result
        return False


def combine_parts(
    *parts, offs: np.ndarray, max_nnz: int, worker_bytes: int | None = None, token: str | None = None
) -> dict:
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
        over_flag(token, "full", set_it=True)
        S = _blocks_only(S, fe_of)
    if S.nnz > max_nnz:
        out.update(blocks=False, nnz=0, block_nnz=int(S.nnz))
        out["reason"] = f"the diagonal blocks summed have more than {max_nnz:,} nonzeros"
        over_flag(token, "blocks", set_it=True)
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


class XtabSolve:
    """Exact pseudo-inverse of the cross-tab preconditioner M = D'WD for two crossing FEs.

    With no local FE, D'WD = [[D1, B], [B', D2]]: D1 and D2 diagonal (a row has one level of
    each FE) and B the FEs' cross-tab (the sum of w over the rows with each pair of levels).
    A = D~'W D~ differs from it only by the local FEs' part. Where the crossing FEs overlap
    heavily, M carries what makes A hard: D11 L3 (cell + gene^cell_type + gene^dataset, cell
    local), where M is one small cell type x dataset block per gene, took 13 CG steps to 1e-10
    against ~505 with Jacobi on R0 and 11 on a 2% cell sample of R1 (Jacobi 1,503 on R1;
    docs/cgprecond.md in the data project). A pass-mode fit needs only B, summed over the
    shards in round 1, and a coordinator solve per step.

    M is singular: on each connected component of B's levels, +1 on FE 1 and -1 on FE 2 is
    in its null space (D v = 0 row by row), so also in A's. The solve removes that direction
    from r and from z, and per component either
    - eliminates the larger side (diagonal) and takes the dense pseudo-inverse of the Schur
      complement on the smaller side, when that side has at most SCHUR_MAX levels, or
    - factorises the component's signed Laplacian [[D1, -B], [-B', D2]] (M up to the signs of
      FE 2) with one level pinned, by sparse LU in chunks of components (LU_CHUNK_NNZ).
    The Schur pseudo-inverse cuts eigenvalues below CUT times the smaller side's largest
    diagonal entry, not its own largest eigenvalue: where a component's levels of the larger
    FE each pair with one level of the smaller (a gene's cell types each in one dataset), the
    Schur complement is all cancellation, zero up to rounding, and a relative cut inverts that
    noise (eigenvalues near 1e18 on D11 R1; CG diverged once past 1e-8)."""

    SCHUR_MAX = 64
    LU_CHUNK_NNZ = 5_000_000
    CUT = 1e-10

    def __init__(self, key: np.ndarray, val: np.ndarray, P1: int, P2: int, name: str = ""):
        from scipy.sparse.csgraph import connected_components

        t = time.time()
        key = np.asarray(key, np.int64)
        val = np.asarray(val, np.float64)
        a, b = key // P2, key % P2
        self.P1, self.P2, self.P = P1, P2, P1 + P2
        D1 = np.bincount(a, weights=val, minlength=P1)
        D2 = np.bincount(b, weights=val, minlength=P2)
        B = sp.csr_matrix((val, (a, b)), shape=(P1, P2))
        ncomp, lab = connected_components(sp.bmat([[None, B], [B.T, None]], format="csr"), directed=False)
        self.ncomp, self.lab = ncomp, lab
        n1 = np.bincount(lab[:P1], minlength=ncomp)
        n2 = np.bincount(lab[P1:], minlength=ncomp)
        live = (n1 > 0) & (n2 > 0)  # else a level with no rows: z = 0 there
        self.sign = np.r_[np.ones(P1), -np.ones(P2)]
        self.cnt = np.where(live, n1 + n2, 1).astype(np.float64)
        self.live_level = live[lab]
        schur = live & (np.minimum(n1, n2) <= self.SCHUR_MAX)
        small2 = schur & (n2 <= n1)  # eliminate FE 1, Schur on FE 2
        small1 = schur & ~small2
        lu = live & ~schur
        self.schur = []
        lab1, lab2 = lab[:P1], lab[P1:]
        for comps, flip in ((small2, False), (small1, True)):
            if not comps.any():
                continue
            if not flip:  # big side FE 1 (offset 0), small side FE 2 (offset P1)
                Bo, Db, Ds, labb, labs, ob, os_ = B, D1, D2, lab1, lab2, 0, P1
            else:
                Bo, Db, Ds, labb, labs, ob, os_ = B.T.tocsr(), D2, D1, lab2, lab1, P1, 0
            ib = np.flatnonzero(comps[labb])
            is_ = np.flatnonzero(comps[labs])
            Bc = Bo[ib][:, is_].tocsr()
            dbinv = 1.0 / Db[ib]
            S = (sp.diags(Ds[is_]) - Bc.T @ sp.diags(dbinv) @ Bc).tocoo()
            # dense blocks per component, batched by size
            cs = labs[is_]
            order = np.argsort(cs, kind="stable")
            cnt_s = np.bincount(cs, minlength=ncomp)
            start = np.r_[0, np.cumsum(cnt_s)]
            pos = np.empty(len(is_), np.int64)
            pos[order] = np.arange(len(is_)) - start[cs[order]]
            groups = []
            for m in np.unique(cnt_s[comps]):
                cg = np.flatnonzero(comps & (cnt_s == m))
                slot = np.full(ncomp, -1, np.int64)
                slot[cg] = np.arange(len(cg))
                blk = np.zeros((len(cg), m, m))
                sel = slot[cs[S.row]] >= 0
                np.add.at(blk, (slot[cs[S.row[sel]]], pos[S.row[sel]], pos[S.col[sel]]), S.data[sel])
                blk = 0.5 * (blk + np.swapaxes(blk, 1, 2))
                where = order[start[cg][:, None] + np.arange(m)[None, :]]  # (n_comp, m) into is_
                lam, V = np.linalg.eigh(blk)
                cut = self.CUT * np.maximum(Ds[is_][where].max(axis=1), 1e-300)[:, None]
                inv = np.where(lam > cut, 1.0 / np.where(lam > cut, lam, 1.0), 0.0)
                groups.append((where, (V * inv[:, None, :]) @ np.swapaxes(V, 1, 2)))
            self.schur.append((ib + ob, is_ + os_, Bc, dbinv, groups))
        # sparse LU of the pinned signed Laplacian, components in chunks
        self.lu = []
        fnnz = 0
        if lu.any():
            import scipy.sparse.linalg as spla

            L = sp.bmat([[sp.diags(D1), -B], [-B.T, sp.diags(D2)]], format="csr")
            comps = np.flatnonzero(lu)
            idx_all = np.flatnonzero(lu[lab])
            idx_all = idx_all[np.argsort(lab[idx_all], kind="stable")]
            cl = lab[idx_all]
            first = np.r_[True, cl[1:] != cl[:-1]]
            deg = np.diff(L.indptr)[idx_all]
            chunk, nz = [], 0
            bounds = np.r_[np.flatnonzero(first), len(idx_all)]
            for c0, c1 in zip(bounds[:-1], bounds[1:]):
                chunk.append((c0, c1))
                nz += int(deg[c0:c1].sum())
                if nz >= self.LU_CHUNK_NNZ or c1 == len(idx_all):
                    keep = np.concatenate([idx_all[s + 1 : e] for s, e in chunk])  # pin each first level
                    f = spla.splu(L[keep][:, keep].tocsc(), permc_spec="MMD_AT_PLUS_A")
                    fnnz += f.L.nnz + f.U.nnz
                    self.lu.append((keep, f))
                    chunk, nz = [], 0
            del comps
        self.kinds = {
            "schur": int(np.isin(lab, np.flatnonzero(schur)).sum()),
            "lu": int(np.isin(lab, np.flatnonzero(lu)).sum()),
        }
        log.info(
            "crossing FEs %s: cross-tab %s nonzeros over %s + %s levels, %s components (largest %s): "
            "%s levels by Schur blocks, %s by sparse LU (%s factor nonzeros) (%.1fs)",
            name,
            f"{len(val):,}",
            f"{P1:,}",
            f"{P2:,}",
            f"{int(live.sum()):,}",
            f"{int((n1 + n2)[live].max(initial=0)):,}",
            f"{self.kinds['schur']:,}",
            f"{self.kinds['lu']:,}",
            f"{fnnz:,}",
            time.time() - t,
        )

    def _project(self, v):
        """v minus its component along each component's null vector (+1 FE 1, -1 FE 2)."""
        s = self.sign[:, None]
        mu = np.column_stack(
            [
                np.bincount(self.lab, weights=self.sign * v[:, j], minlength=self.ncomp)
                for j in range(v.shape[1])
            ]
        )
        return (v - s * (mu / self.cnt[:, None])[self.lab]) * self.live_level[:, None]

    def solve(self, r):
        r = self._project(np.asarray(r, np.float64))
        z = np.zeros_like(r)
        for ib, is_, Bc, dbinv, groups in self.schur:
            rb = r[ib] * dbinv[:, None]
            t = r[is_] - Bc.T @ rb
            zs = np.zeros_like(t)
            for where, Sinv in groups:  # where: (components, m) positions in is_
                zs[where] = np.einsum("gab,gbc->gac", Sinv, t[where])
            z[is_] = zs
            z[ib] = rb - (Bc @ zs) * dbinv[:, None]
        for keep, f in self.lu:
            z[keep] = self.sign[keep, None] * f.solve(np.ascontiguousarray(self.sign[keep, None] * r[keep]))
        return self._project(z)


# ---------------------------------------------------------------- preconditioners


def choose_method(
    solvers: list[BlockSolve] | None, level_space: bool, want: str = "auto", xtab: XtabSolve | None = None
) -> str:
    """The preconditioner, from the block structure (planner.md's table) unless `want` names
    one. With a pass per step and the crossing FEs' cross-tab (two crossing FEs), "auto" takes
    the cross-tab preconditioner (XtabSolve): a step costs one pass whatever the
    preconditioner, and on D11 L3 it cut the steps ~40x against Jacobi."""
    if want == "xtab":
        if xtab is None:
            log.warning("crossing_precond='xtab' needs two crossing FEs and their cross-tab: Jacobi")
            return "jacobi"
        return "xtab"
    if want == "auto" and not level_space and xtab is not None:
        return "xtab"
    if want != "auto":
        if want not in ("jacobi", "bjacobi", "sgs"):
            raise ValueError("crossing_precond must be auto, jacobi, bjacobi, sgs or xtab")
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


def make_precond(method: str, diag: np.ndarray, offs: np.ndarray, solvers=None, A=None, xtab=None):
    """z = M^-1 r for CG. `diag` is A's diagonal (levels with none are inactive), `offs` the
    crossing FEs' offsets in the level index, `A` the whole matrix (sgs only), `xtab` the
    cross-tab solve (xtab only)."""
    act = diag > 1e-12 * max(float(diag.max(initial=0.0)), 1.0)
    dinv = np.where(act, 1.0 / np.where(act, diag, 1.0), 0.0)
    if method == "jacobi":
        return lambda r: r * dinv[:, None]
    if method == "xtab":
        return lambda r: xtab.solve(r) * act[:, None]
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


def matvec_threads() -> int:
    """Threads for A @ p in level space: DFEOLS_MATVEC_THREADS, else the CPUs this process may
    use, at most MATVEC_THREADS_MAX."""
    env = os.environ.get("DFEOLS_MATVEC_THREADS")
    if env:
        return max(1, int(env))
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        n = os.cpu_count() or 1
    return max(1, min(n, MATVEC_THREADS_MAX))


MATVEC_THREADS_MAX = 16
MATVEC_MIN_NNZ = 2_000_000  # below this one thread is as fast (a product takes a few ms)


def level_matvec(A: sp.csr_matrix, threads: int | None = None):
    """p -> A @ p, with A's rows in `threads` blocks of equal nonzeros multiplied in parallel
    (scipy's sparse product releases the GIL; one thread was 0.64 s of a 3 s CG step on EVE
    market r2's 182M nonzeros, idling 15 of the coordinator's 16 cores). Same result as A @ p
    to the bit: each row's sum is computed exactly as before, in one block."""
    threads = matvec_threads() if threads is None else threads
    if threads <= 1 or A.nnz < MATVEC_MIN_NNZ:
        return lambda p: A @ p
    from concurrent.futures import ThreadPoolExecutor

    P = A.shape[0]
    cuts = np.unique(np.r_[0, np.searchsorted(A.indptr, np.linspace(0, A.nnz, threads + 1)[1:-1]), P])
    blocks = [(int(a), int(b), A[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
    pool = ThreadPoolExecutor(len(blocks), thread_name_prefix="dfeols-matvec")

    def mv(p):
        out = np.empty((P,) + p.shape[1:], dtype=np.result_type(A.dtype, p.dtype))

        def one(blk):
            a, b, B = blk
            out[a:b] = B @ p

        for f in [pool.submit(one, blk) for blk in blocks]:
            f.result()
        return out

    mv.pool = pool  # the caller may shut it down; idle threads otherwise end with the process
    return mv


# pcg: a column's residual below this share of its no-cancellation scale is rounding (see pcg)
ROUND_REL = 1e-13
# estimate: stop a CG solve whose largest residual has had no new low in this many steps
STALL_STEPS = 300
PRECOND_TRIAL_STEPS = 10
# a candidate's trial stops once it has spent this many times the best estimate so far: it
# cannot win (the estimates run low, early CG falling faster, hence the margin)
PRECOND_TRIAL_CAP = 3.0
# below this many nonzeros of A a solve takes milliseconds whatever the preconditioner, and a
# trial's clock is noise (a serial and a Dask fit of the same data chose differently), so the
# structural choice stands; above it, it gives way only to a candidate this many times faster
PRECOND_TRIAL_MIN_NNZ = 5_000_000
PRECOND_TRIAL_MARGIN = 2.0


class _TrialStop(Exception):
    pass


def precond_trial(
    matvec, R, act, tol, maxiter, candidates: dict, prefer: str | None = None, steps: int = PRECOND_TRIAL_STEPS
):
    """Pick the preconditioner that reaches `tol` soonest by the clock, from up to `steps` CG
    steps with each candidate ({name: M}; put the cheapest first). A block preconditioner cuts
    the steps 7-30x on DB1B and NFIP but only ~12% on EVE contracts, where its block solves made
    a step 4-5x dearer than Jacobi's (planner.md), so the block structure alone does not decide
    it. Each estimate is the trial's seconds a step times the steps its rate of fall needs; a
    candidate whose trial outlasts PRECOND_TRIAL_CAP x the best estimate so far is stopped (it
    cannot win). `prefer` (the structural choice) is kept unless another is PRECOND_TRIAL_MARGIN
    times faster. The winner's state is returned, so its trial steps are kept. Returns (name,
    state or None, report)."""
    report, states = {}, {}
    for name, M in candidates.items():
        keep = {}
        best = min((v["est_seconds"] for v in report.values()), default=float("inf"))
        t = time.time()

        def on_step(it, rel, t=t, best=best):
            if it >= 2 and time.time() - t > PRECOND_TRIAL_CAP * best:
                raise _TrialStop

        try:
            _, it, rel = pcg(
                matvec, R, M, act, tol, min(steps, maxiter), on_step, on_state=lambda s, keep=keep: keep.update(s)
            )
            stopped = False
        except _TrialStop:
            it, stopped = int(keep["it"]), True
            rel = float((np.linalg.norm(keep["r"], axis=0) / _bnorm(R, act)).max())
        dt = time.time() - t
        per = dt / max(it, 1)
        if stopped:
            est = float("inf")
        elif rel <= tol:
            est = dt
        elif it and 0 < rel < 1:
            est = per * np.log(tol) / (np.log(rel) / it)
        else:
            est = float("inf")
        report[name] = {"steps": int(it), "seconds": dt, "residual": float(rel), "est_seconds": float(est),
                        "stopped": stopped}  # fmt: skip
        states[name] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in keep.items()} or None
        if rel <= tol and not stopped:  # converged within the trial: nothing to compare
            break
    best = min(report, key=lambda n: report[n]["est_seconds"])
    if prefer in report and best != prefer and np.isfinite(report[prefer]["est_seconds"]):
        if report[best]["est_seconds"] * PRECOND_TRIAL_MARGIN > report[prefer]["est_seconds"]:
            best = prefer
    return best, states[best], report


def _bnorm(R, act):
    b = np.linalg.norm(R * act[:, None], axis=0)
    b[b == 0] = 1.0
    return b


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
    scale: np.ndarray | None = None,
    stall_steps: int = 0,
):
    """Preconditioned CG for A X = R, one column per right-hand side, in lockstep: stops when
    every column's residual is below `tol` relative to its right-hand side. A is singular (one
    redundant level per connected component of the FEs) but the system is consistent, so CG
    converges to a solution whose D~ X, the only part used, is unique. `on_step(it, rel)` is
    called after each step and may raise. Returns X, steps and the largest relative residual.

    `scale` (one per column; estimate._cg_scale): the size R's column would have without
    cancellation, sqrt(max diag A) x the column's norm. A column whose residual is below
    ROUND_REL of it is at the rounding level of the sums that formed R and counts as converged:
    so a column whose R is nearly all cancellation (a regressor with no crossing-FE component,
    D10 G g1's rin:log_gene_length, |R| 5e-3 against 1e7) is not driven to `tol` relative to
    its own rounding noise. Rounding also leaves R a component along A's null space, which CG
    cannot remove: driven below it, the column's iterates wander and grow (36188818: 3.8e-9 at
    step 1, then 1e-7 to 0.29 for 700 steps). A converged column is frozen (its step and search
    direction set to 0), so the lockstep steps of the other columns do not disturb it.

    `stall_steps` > 0: stop early when the largest residual among the unconverged columns has
    not reached a new low in that many steps (the caller sees rel > tol before maxiter).

    `state` (X, r, p, rz, it, as `on_state(state)` receives it after each step) resumes an
    interrupted solve exactly where it stopped: CG's whole state is these few P x c blocks."""
    P, c = R.shape
    r0 = R * act[:, None]
    bnorm = np.linalg.norm(r0, axis=0)
    bnorm[bnorm == 0] = 1.0
    if scale is not None:
        # ||r|| <= tol ||R||  or  ||r|| <= ROUND_REL scale, as one relative measure
        bnorm = np.maximum(bnorm, ROUND_REL / tol * np.asarray(scale, float))
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
    done = rel <= tol
    p[:, done] = 0.0
    best, best_it = float(rel.max()), it
    while rel.max() > tol:
        if it >= maxiter:
            return X, it, float(rel.max())
        if stall_steps and it - best_it >= stall_steps:
            return X, it, float(rel.max())
        it += 1
        Ap = matvec(p)
        pAp = (p * Ap).sum(0)
        alpha = np.divide(rz, pAp, out=np.zeros(c), where=pAp > 0)
        alpha[done] = 0.0
        X += alpha * p
        r -= alpha * Ap
        z = precond(r)
        rz_new = (r * z).sum(0)
        beta = np.divide(rz_new, rz, out=np.zeros(c), where=rz > 0)
        p = z + beta * p
        rz = np.where(done, rz, rz_new)
        rel = np.linalg.norm(r, axis=0) / bnorm
        done |= rel <= tol
        p[:, done] = 0.0
        if rel.max() < best:
            best, best_it = float(rel.max()), it
        if on_state is not None:
            on_state({"X": X, "r": r, "p": p, "rz": rz, "it": it})
        if on_step is not None:
            on_step(it, rel)
    return X, it, float(rel.max())


# in Checkpoint's digest: a checkpoint resumes only under the pcg and preconditioners that wrote
# it (GTEx g1 resumed pre-be311ff state under be311ff's pcg and stalled). Bump it whenever what
# CG's state means changes
CG_STATE_VERSION = "cg2"


class Checkpoint:
    """A CG solve's state on disk, written after each step (atomically) and read back by a
    rerun of the same solve: the same shape, tolerance and preconditioner, and a right-hand
    side equal to 1e-10 (shards' sums arrive in another order on a rerun, so not to the
    bit; the difference is far below any CG tolerance). Removed once the solve converges."""

    RTOL = 1e-10

    def __init__(self, path, R: np.ndarray, tol: float, method: str, every_s: float = 0.0):
        """every_s: write at most this often (the latest state is kept for flush): D6 r4's
        state is 4 blocks of 12.1M x 7 doubles, 2.7 GB a step on the coordinator."""
        self.path = Path(path)
        self.R = R
        self.digest = f"{R.shape}|{tol!r}|{method}|{CG_STATE_VERSION}"
        self.every_s = every_s
        self._last = None  # (state, written)
        self._t = float("-inf")

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
        now = time.monotonic()
        if now - self._t < self.every_s:
            self._last = (state, False)
            return
        self._write(state)
        self._t, self._last = now, (state, True)

    def flush(self) -> None:
        """Write the latest state if save() skipped it (a solve stopped short of tol)."""
        if self._last is not None and not self._last[1]:
            self._write(self._last[0])
            self._last = (self._last[0], True)

    def _write(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, digest=self.digest, R=self.R, **state)
        os.replace(tmp, self.path)

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------- CG steps' tree reduction


def sorted_unique(a) -> np.ndarray:
    """np.unique(a) for integer codes, by sorting. NumPy 2.5's np.unique without return_*
    hashes, and on many distinct values that is 30-50x slower than a sort: 1M level codes
    0.69 s against 0.02 s, D6 r4's largest shard (6.6M) 4.2 s; with return_inverse it sorts
    and stays fast. Called on every shard in every round and CG step."""
    a = np.asarray(a).ravel()
    if a.size == 0:
        return a.copy()
    s = np.sort(a)
    keep = np.empty(s.size, bool)
    keep[0] = True
    np.not_equal(s[1:], s[:-1], out=keep[1:])
    return s[keep]


def step_part(res: dict) -> tuple[np.ndarray, np.ndarray]:
    """A CG step's shard result as (levels, sums): what the tree adds up."""
    return np.asarray(res["idx"]), res["S"]


def step_strip(res: dict) -> dict:
    """The rest of a step's result, for the coordinator (timing only)."""
    return {k: v for k, v in res.items() if k not in ("idx", "S")}


def step_combine(*parts, P: int | None = None):
    """Sum shards' (levels, sums) on a worker: the union of the levels, sums added. Each
    part's levels are sorted and distinct (sorted_unique in worker._iter_store, and here), so each
    adds into its rows of the union by position, without np.add.at (D6 r4: up to 4 x 12.1M
    levels x 7 near the root, where np.unique and np.add.at took most of the 30-100 s between
    a CG step's last shard and its sum). With `P` (all levels) a union that covers a large
    share of them (parts' levels adding to half of P or more; below that the sort is as fast,
    1-4 x 1.5M levels: 0.7 s against 0.9 s) is summed in a dense P-row block instead of
    sorting. The union is sorted_unique, not np.unique (0.73 of a 0.85 s combine of 4 x 300k
    levels at D6's P)."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    c = parts[0][1].shape[1]
    total = sum(len(p[0]) for p in parts)
    if P is not None and total >= P // 2:
        out = np.zeros((P, c))
        seen = np.zeros(P, bool)
        for idx, S in parts:
            out[idx] += S
            seen[idx] = True
        uniq = np.flatnonzero(seen)
        return uniq, out[uniq]
    uniq = sorted_unique(np.concatenate([p[0] for p in parts]))
    out = np.zeros((len(uniq), c))
    for idx, S in parts:
        out[np.searchsorted(uniq, idx)] += S
    return uniq, out


def step_combine_gb(P: int, c: int, fan_in: int = FAN_IN) -> float:
    """Most memory a step_combine needs: fan_in parts and the sum over at most P levels, the
    dense block and its gathered copy (8 bytes a value, 8 an index)."""
    return ((fan_in + 2) * P * (8 * c + 8) + P) / 1e9 + 0.2
