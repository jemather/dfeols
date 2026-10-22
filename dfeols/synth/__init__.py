"""Synthetic data with known parameters, for validation and benchmarking.

Two data-generating processes (DGPs), both written chunk by chunk with each
chunk seeded from (seed, chunk), so any subset can be generated anywhere:

  panel       generic unit x time panel, units nested in groups (default)
  influencer  follower x account x week panel with tenure bins

Every DGP writes truth.json next to the data: the true coefficients, the FE
and cluster specs to encode in prep(), and a set of model specs with what
each should find (unbiased / biased / se_wrong). Tests and benchmarks read
the specs from there, so they work for any DGP.

    python -m dfeols.synth OUT --rows 1e8 --jobs 16                 # panel
    python -m dfeols.synth OUT --dgp influencer --rows 1e8 --jobs 16
    python -m dfeols.synth OUT --rows 1e9 --profile mycluster       # on cluster workers

From Python:
    from dfeols.synth import PanelDGP, generate, plan_for_rows
    d = plan_for_rows(1e8, "panel", T=104, n_groups=1000)
    generate("/scratch/me/raw", d, jobs=16)               # or executor=dfeols.executor(...)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import polars as pl

from .influencer import DGP, InfluencerDGP, generate_chunk  # noqa: F401  (back-compat)
from .panel import PanelDGP

TRUTH = "truth.json"
DGPS = {"panel": PanelDGP, "influencer": InfluencerDGP}


def make_dgp(kind: str = "panel", **kw):
    if kind not in DGPS:
        raise KeyError(f"unknown DGP {kind!r}; have {sorted(DGPS)}")
    cls = DGPS[kind]
    fields = set(cls.__dataclass_fields__)
    bad = set(kw) - fields
    if bad:
        raise TypeError(f"{kind} DGP has no parameters {sorted(bad)}")
    return cls(**kw)


def rows_per_unit(d) -> float:
    """Expected rows per top-level unit, from a pilot chunk."""
    n_pilot = min(2000, d.units_per_chunk)
    pilot = d.scaled(n_pilot)
    pilot = type(d)(**{**asdict(pilot), "seed": pilot.seed + 7, **_chunk_kw(d, n_pilot)})
    return pilot.generate_chunk(0).height / n_pilot


def _chunk_kw(d, n):
    return {"accounts_per_chunk": n} if d.kind == "influencer" else {"units_per_chunk": n}


def plan_for_rows(rows: float, kind: str = "panel", **kw):
    """A DGP sized to about `rows` rows."""
    d = make_dgp(kind, **kw)
    return d.scaled(max(int(round(rows / rows_per_unit(d))), 10))


def _write_chunk(args):
    d, chunk, out = args
    p = Path(out) / f"chunk_{chunk:06d}.parquet"
    if p.exists():
        return chunk, pl.scan_parquet(p).select(pl.len()).collect().item(), 0.0
    t = time.perf_counter()
    df = d.generate_chunk(chunk)
    tmp = p.with_suffix(".tmp")
    df.write_parquet(tmp, row_group_size=500_000)
    os.replace(tmp, p)
    return chunk, df.height, time.perf_counter() - t


def _truth_doc(d) -> dict:
    return {"kind": d.kind, "dgp": asdict(d), **d.truth()}


def generate(out, d, chunks: list[int] | None = None, jobs: int = 1, executor=None,
             verbose: bool = True) -> Path:
    """Write the DGP's chunks to `out` (Parquet). Idempotent: existing chunks are
    kept, so an interrupted run resumes. Runs serially, on a local process
    pool (jobs > 1), or on a dfeols/Dask executor (executor=...)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tp = out / TRUTH
    doc = _truth_doc(d)
    if tp.exists():
        old = json.loads(tp.read_text())
        if old.get("dgp") != json.loads(json.dumps(doc["dgp"])):
            raise ValueError(f"{out} holds data from a different DGP; use a new directory")
    else:
        tp.write_text(json.dumps(doc, indent=1))
    chunks = list(range(d.n_chunks)) if chunks is None else list(chunks)
    chunks = [c for c in chunks if not (out / f"chunk_{c:06d}.parquet").exists()]
    if not chunks:
        return out
    args = [(d, c, str(out)) for c in chunks]
    t0, rows, done = time.time(), 0, 0

    def report(n):
        nonlocal rows, done
        rows += n
        done += 1
        if verbose and (done % max(1, len(args) // 20) == 0 or done == len(args)):
            el = time.time() - t0
            print(f"synth: {done}/{len(args)} chunks, {rows:,} rows, {el:.0f}s, "
                  f"ETA {el / done * (len(args) - done):.0f}s", flush=True)

    if executor is not None and hasattr(executor, "client"):
        from distributed import as_completed
        futs = [executor.client.submit(_write_chunk, a, pure=False) for a in args]
        for f in as_completed(futs):
            report(f.result()[1])
    elif jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        # spawn, not fork: Polars/numba thread pools in the parent make fork unsafe
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as ex:
            for _, n, _ in ex.map(_write_chunk, args):
                report(n)
    else:
        for a in args:
            report(_write_chunk(a)[1])
    return out


def load_truth(out) -> dict:
    return json.loads((Path(out) / TRUTH).read_text())


def dgp_from_truth(out):
    t = load_truth(out)
    d = dict(t["dgp"])
    if "theta" in d:
        d["theta"] = tuple(d["theta"])
    return DGPS[t.get("kind", "influencer")](**d)


def _parse_chunks(s: str, n: int) -> list[int]:
    """'3' | '0:10' | 'i::k' (every k-th chunk starting at i, for job arrays)."""
    if "::" in s:
        i, k = s.split("::")
        return list(range(int(i), n, int(k)))
    if ":" in s:
        a, b = s.split(":")
        return list(range(int(a or 0), min(int(b or n), n)))
    return [int(s)]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dfeols.synth", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--dgp", default="panel", choices=sorted(DGPS))
    ap.add_argument("--rows", type=float, required=True, help="target total rows")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="DGP parameter, e.g. --set T=104 --set n_groups=1000")
    ap.add_argument("--chunks", default=None, help="'i', 'a:b' or 'i::k' (job arrays)")
    ap.add_argument("--jobs", type=int, default=1, help="local processes")
    ap.add_argument("--profile", default=None, help="generate on cluster workers from this profile")
    a = ap.parse_args(argv)
    kw = {}
    for s in a.set:
        k, v = s.split("=", 1)
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
    if "theta" in kw:
        kw["theta"] = tuple(kw["theta"])
    d = plan_for_rows(a.rows, a.dgp, **kw)
    chunks = _parse_chunks(a.chunks, d.n_chunks) if a.chunks else None
    print(f"synth: {a.dgp} DGP, {d.n_units:,} units in {d.n_chunks} chunks -> {a.out}")
    ex = None
    if a.profile:
        import dfeols
        ex = dfeols.executor(a.profile)
    generate(a.out, d, chunks, a.jobs, executor=ex)


if __name__ == "__main__":
    main()
