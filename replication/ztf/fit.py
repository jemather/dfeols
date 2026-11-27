# SPDX-License-Identifier: GPL-3.0-or-later
"""ZTF stage 3: fit the photometric-calibration specs with dfeols.

    python -I repo/replication/ztf/fit.py --data build/ztf/analysis.parquet --root build/ztf \
        [--partition fieldid|objectid] [--profile local|<site profile>] [--only S1,S8] [--out results.json]

Model: a detection's catalogue magnitude is the star's brightness (object FE) plus the exposure's
zero-point (quadrant-exposure FE), minus the colour term the catalogue leaves out,
clrcoeff_t * (g - r)_i. If the calibration is right, the coefficient on cc_gr is -1.

OLS is attenuated because clrcoeff is fitted per quadrant-exposure and much of its variation is
fitting noise; the IV specs instrument it with the other quadrants' clrcoeff in the same visit
(see build.py). On the dial-in partition: OLS -0.57, weighted IV -1.04 (0.026); README.md.

--partition fieldid nests every FE inside shards; objectid puts the exposure FE across shards
(solved by CG), a check that the answer does not depend on sharding.
"""

from __future__ import annotations

import argparse
import json
import time

import dfeols

FE = "objectid + expid"
FE_SEASON = "objectid^season + expid"
AM = "airmass:gr"
IV = "cc_gr ~ z_gr"
OBJ, OBJ_VISIT = dict(cluster="objectid"), dict(cluster="objectid+visit")
SPECS = {  # id: (description, formula, fit options)
    "S1": ("colour term, OLS", f"mag ~ cc_gr | {FE}", OBJ),
    "S2": ("+ 2nd-order extinction", f"mag ~ cc_gr + {AM} | {FE}", OBJ),
    "S3": ("weighted, 2-way CRV", f"mag ~ cc_gr + {AM} | {FE}", dict(cluster="objectid+expid", weights="w")),
    "S4": ("object x season FE", f"mag ~ cc_gr + {AM} | {FE_SEASON}", OBJ),
    "S5": ("bright stars (mobj < 17)", f"mag ~ cc_gr + {AM} | {FE}", {**OBJ, "subset": "mobj < 17"}),
    "S6": ("first stage", f"cc_gr ~ z_gr | {FE}", OBJ_VISIT),
    "S7": ("IV", f"mag ~ {AM} | {FE} | {IV}", OBJ_VISIT),
    "S8": ("IV, weighted (headline)", f"mag ~ {AM} | {FE} | {IV}", {**OBJ_VISIT, "weights": "w"}),
    "S9": ("IV, object x season FE", f"mag ~ {AM} | {FE_SEASON} | {IV}", OBJ_VISIT),
}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--root", required=True, help="shards and work directories go here")
    ap.add_argument("--partition", default="fieldid")
    ap.add_argument("--profile", default=None, help="dfeols site profile; omit to run serially")
    ap.add_argument("--only", default=None, help="comma-separated spec ids")
    ap.add_argument("--target-rows", type=int, default=2_000_000)
    ap.add_argument("--out", default=None, help="results JSON (default <root>/results_<partition>.json)")
    a = ap.parse_args()

    ex = dfeols.executor(a.profile) if a.profile else None
    t = time.time()
    ds = dfeols.prep(
        a.data,
        shard_dir=f"{a.root}/shards_{a.partition}",
        partition=a.partition,
        fixed_effects=["objectid", "expid", "objectid^season"],
        clusters=["objectid", "expid", "visit"],
        keep=["mag", "cc_gr", "z_gr", "airmass", "gr", "w", "mobj"],
        target_rows=a.target_rows,
        overwrite=True,
        executor=ex,
    )
    print(f"prep {time.time() - t:.0f}s", flush=True)
    only = a.only.split(",") if a.only else list(SPECS)
    out = {}
    for sid in only:
        desc, f, kw = SPECS[sid]
        t = time.time()
        r = dfeols.fit(ds, f, executor=ex, workdir=f"{a.root}/work_{a.partition}", log_every=120, **kw)
        td = r.tidy()
        secs = time.time() - t
        print(f"\n== {sid} {desc}  [{f}]  {kw}  {secs:.0f}s\n{td.iloc[:, :3].to_string()}", flush=True)
        out[sid] = {
            "description": desc, "formula": f, "options": kw, "seconds": round(secs, 1),
            "coef": td["Estimate"].to_dict(), "se": td["Std. Error"].to_dict(),
        }  # fmt: skip
    with open(a.out or f"{a.root}/results_{a.partition}.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
