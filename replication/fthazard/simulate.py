# SPDX-License-Identifier: GPL-3.0-or-later
"""Simulated discrete-time hazard panels: the Monte Carlo of Farbmacher & Tauchmann (2021),
"Linear fixed-effects estimation with non-repeated outcomes", FAU DP 03/2021, section 4.

DGP (section 4, Table 1): y_it = 1 with probability alpha + a_i + x_it b (+ tau_t, Table A4),
alpha = 0.1, b = 1, a_i ~ U(-0.05, 0.05), T = 5; each unit is observed until its first event
or T (rows after the event are never generated). x_it, with z, n, h iid Beta(0.2, 0.2):

  st  x_it = a_i + 0.165 + 0.07 z_it                          stationary
  rw  x_it = x_i,t-1 + (0.1 n_it - 0.05), x_i1 = a_i + 0.2    random walk without drift
  tr  x_it = a_i + 0.175 + 0.025 t h_it                       trend, rising variance

Table A4 (`--time-effects`): tau = (0, 0, 0.2, 0, 0.2) for t = 1..5.

Writes Parquet parts (uid Int64, t Int8, x Float64, y Int8, a Float64 = the true a_i) of about
`--part-rows` rows each, deterministic in (seed, dgp, chunk): the same arguments give the same
files. Usage:

    python simulate.py --dgp st --units 40000000 --out build/fthazard/st
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

DGPS = {"st": 1, "rw": 2, "tr": 3}
ALPHA, BETA, T = 0.1, 1.0, 5
TAU_A4 = (0.0, 0.0, 0.2, 0.0, 0.2)


def chunk(dgp: str, n: int, uid0: int, rng: np.random.Generator, time_effects: bool) -> pl.DataFrame:
    a = rng.uniform(-0.05, 0.05, n)
    b = rng.beta(0.2, 0.2, (n, T))
    if dgp == "st":
        x = a[:, None] + 0.165 + 0.07 * b
    elif dgp == "rw":
        steps = 0.1 * b[:, 1:] - 0.05
        x = (a + 0.2)[:, None] + np.concatenate([np.zeros((n, 1)), np.cumsum(steps, 1)], 1)
    else:
        x = a[:, None] + 0.175 + 0.025 * np.arange(1, T + 1)[None, :] * b
    p = ALPHA + a[:, None] + BETA * x + (np.array(TAU_A4)[None, :] if time_effects else 0.0)
    if p.min() < 0 or p.max() > 1:
        raise ValueError(f"hazard outside [0, 1]: {p.min()}, {p.max()}")
    ev = rng.random((n, T)) < p
    first = np.where(ev.any(1), ev.argmax(1), T - 1)
    keep = np.arange(T)[None, :] <= first[:, None]
    ii, tt = np.nonzero(keep)
    return pl.DataFrame(
        {
            "uid": (uid0 + ii).astype(np.int64),
            "t": (tt + 1).astype(np.int8),
            "x": x[ii, tt],
            "y": ev[ii, tt].astype(np.int8),
            "a": a[ii],
        }
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dgp", choices=list(DGPS), required=True)
    ap.add_argument("--units", type=int, default=40_000_000)
    ap.add_argument("--seed", type=int, default=20210503)
    ap.add_argument("--time-effects", action="store_true", help="Table A4: tau_3 = tau_5 = 0.2")
    ap.add_argument("--chunk-units", type=int, default=1_000_000)
    ap.add_argument("--part-rows", type=int, default=40_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    buf, nbuf, part, rows, events = [], 0, 0, 0, 0
    parts = []

    def flush():
        nonlocal buf, nbuf, part
        if not buf:
            return
        df = pl.concat(buf)
        name = f"data/part-{part:04d}.parquet"
        df.write_parquet(out / name, compression="zstd", statistics=True)
        parts.append({"file": name, "rows": df.height})
        print(f"{time.time() - t0:7.0f}s wrote {name} ({df.height:,} rows)", flush=True)
        buf, nbuf, part = [], 0, part + 1

    for k, start in enumerate(range(0, a.units, a.chunk_units)):
        n = min(a.chunk_units, a.units - start)
        rng = np.random.default_rng([a.seed, DGPS[a.dgp], int(a.time_effects), k])
        df = chunk(a.dgp, n, start, rng, a.time_effects)
        rows += df.height
        events += int(df["y"].sum())
        buf.append(df)
        nbuf += df.height
        if nbuf >= a.part_rows:
            flush()
    flush()
    info = {
        "paper": "Farbmacher & Tauchmann (2021), FAU DP 03/2021, section 4",
        "dgp": a.dgp,
        "time_effects": a.time_effects,
        "units": a.units,
        "T": T,
        "alpha": ALPHA,
        "beta": BETA,
        "seed": a.seed,
        "rows": rows,
        "events": events,
        "parts": parts,
        "seconds": round(time.time() - t0, 1),
    }
    (out / "simulate.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "parts"}), flush=True)


if __name__ == "__main__":
    main()
