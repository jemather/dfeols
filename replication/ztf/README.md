# ZTF photometric calibration: an astrophysics scale test

Status: dialled in on one sky partition (24M detections). Next: more partitions, then the full
release on a cluster. Gaia DR4 (epoch photometry for all sources, expected around December 2026)
is the follow-on dataset with the same structure.

## The model

Each detection's catalogue magnitude is a star's brightness plus the zero-point of the exposure
(per CCD quadrant), minus the colour term the catalogue leaves out:

    mag_it = m_i + zp_t - clrcoeff_t * (g - r)_i + e_it

so `mag ~ cc_gr | objectid + expid`, with `cc_gr = clrcoeff * gr`, should give **-1** on `cc_gr`.
This is ubercalibration (Padmanabhan et al. 2008): two high-cardinality crossed FEs (objects x
exposures) linked by a sparse, uneven graph.

`clrcoeff` is fitted per quadrant-exposure, so OLS is attenuated by its fitting noise (its
within-quadrant sd is 0.049, but its correlation with the other quadrants of the same visit is only
0.11). The IV specs instrument `cc_gr` with `z_gr`: the mean clrcoeff of the *other* quadrants in
the same visit (same atmosphere, independent fitting errors), times g - r.

## Data

ZTF DR24 light curves, IRSA's HATS copy on AWS open data (`s3://ipac-irsa-ztf/ztf/enhanced/dr24/lc/hats/`,
us-east-1, public, also over HTTPS): 5.4e9 light curves (one per object x filter), 8.4 TB, 12,485
HEALPix partitions. Rough size of the full release: ~4e11 detections before cuts (the dial-in
partition holds 34M detections and is about average).

Cite DOI [10.26131/IRSA598](https://www.ipac.caltech.edu/doi/10.26131/IRSA598) and follow the
[ZTF](https://irsa.ipac.caltech.edu/Missions/ztf.html) and [IRSA](https://irsa.ipac.caltech.edu/ack.html)
acknowledgement guidelines and [registry.opendata.aws/ztf](https://registry.opendata.aws/ztf/).

## Running it

    python -I repo/replication/ztf/stage.py --raw raw/ztf/dr24 --out build/ztf/stage --npix 3/385   # flatten
    python -I repo/replication/ztf/build.py --stage build/ztf/stage --out build/ztf/analysis --workers 4
    python -I repo/replication/ztf/fit.py --data 'build/ztf/analysis/data/*.parquet' --root build/ztf
    python -I repo/replication/ztf/fit.py ... --partition objectid                   # sharding check
    python -I repo/replication/ztf/pyfixest_check.py --data 'build/ztf/analysis/data/*.parquet' --root build/ztf/check

`stage.py` downloads nothing: the partitions must already be in `raw/ztf/dr24/` at the bucket's
`Norder=/Dir=/Npix=` layout (URLs and copy commands in `docs/sources/ztf.md` of the data-build
project). `--all --workers K` stages every partition found there. `fit.py --profile <site>` runs on a
cluster; `--only S1,S8` picks specs. `jobs/ztf_dial_in.sh` runs the fits and checks below. See the
docstrings for the steps and the checks each prints.

## Dial-in results (partition Norder=3 Npix=385; RA ~186, Dec ~-28)

Build: 33.9M detections; 12,603 quadrant-exposures (no object twice in one; hmjd spread within an
exposure at most 10.4 s; 3 ambiguous time gaps); 27.3M good detections; 213k g/r pairs. Analysis
sample: 24,362,418 rows, 421,388 objects, 9,393 exposures, 754 visits.

| Spec | Formula (`FE = objectid + expid`) | `cc_gr` (SE) | `airmass:gr` (SE) | s |
|---|---|---|---|---|
| S1 OLS | `mag ~ cc_gr \| FE`, CRV objectid | -0.5677 (0.0074) | | 67 |
| S2 + 2nd-order extinction | `mag ~ cc_gr + airmass:gr \| FE` | -0.5688 (0.0074) | 0.0260 (0.0007) | 81 |
| S3 weighted, 2-way CRV | S2, `weights=w`, CRV objectid+expid | -0.6922 (0.0129) | -0.0021 (0.0009) | 95 |
| S4 object x season | S2 with `objectid^season + expid` | -0.6180 (0.0078) | 0.0277 (0.0007) | 87 |
| S5 bright stars | S2, `subset="mobj < 17"` | -0.5449 (0.0075) | 0.0009 (0.0004) | 13 |
| S6 first stage | `cc_gr ~ z_gr \| FE`, CRV objectid+visit | `z_gr` 0.841 (0.020), t = 43 | | 79 |
| S7 IV | `mag ~ airmass:gr \| FE \| cc_gr ~ z_gr` | -0.8900 (0.0842) | 0.0262 (0.0150) | 112 |
| **S8 IV, weighted** | S7, `weights=w` | **-1.0412 (0.0260)** | -0.0021 (0.0023) | 118 |
| S9 IV, object x season | S7 with `objectid^season + expid` | -1.0126 (0.0832) | 0.0295 (0.0131) | 119 |

S5 rules out noise in g - r as the cause of the OLS attenuation (bright stars are no closer to -1);
the instrument for clrcoeff's own noise moves it to -1.

All on one core, serial, 4 shards (`--partition fieldid`); 0.4-0.9M rows/s. First run in a cloud
session; reproduced on valiant 2026-11-27 (repo d25ef42, polars 2.0, pyfixest 0.60): every build count
and every estimate above to the printed digits (log `logs/ztf_dial_in_20261127_1529.log`).

Checks (valiant, 2026-11-27):

- **pyfixest**, field 222 (3,705,235 rows, 3,697,734 after singletons), S1-S3 and IV S7-S8:
  coefficients agree to <= 7e-15 relative, SEs to <= 1.1e-13. Two settings matter: pyfixest 0.60
  demeans to 1e-6 by default (tightened to 1e-13), and it keeps Float32 weights in lower precision
  (the weighted specs were ~1e-8 off until the columns were cast to float64).
- **Sharding**, `--partition objectid` (13 shards; the exposure FE's 9,393 levels cross shards and are
  solved exactly in level space, CG converging in one step): S1-S9 equal `--partition fieldid` to
  <= 1.9e-14 relative, coefficients and SEs. (The cloud session's objectid log was from an earlier
  build, 24,700,431 rows, so it was not this comparison.)

## Known issues / next steps

- This patch is far south (airmass >= 1.89), so `airmass:gr` is barely identified. Add a field
  near Dec +30 (e.g. a Norder=3 pixel there) to the dial-in set.
- Large between-field heterogeneity: field 222 alone gives OLS -0.27. 7 (visit, rcid) pairs hold
  two exposures (VISIT_GAP merges a few close visits; only affects the instrument). Both are left
  alone: this is a validation dataset for dfeols, not a calibration study (decision 2026-11-27).
- g/r matching is within one partition, so objects at partition edges lose their partner.
- `build.py` runs out of core: the distinct (quadrant, hmjd) table with counts across all
  partitions (88k rows per 34M detections), then one partition at a time. On the dial-in partition
  it reproduces the in-memory build exactly except `cc_loo`/`z_gr`, now summed in float64 (they
  moved by <= 2.6e-7 relative; S8 unchanged to the printed digits).
- Middle step (decision 2026-11-27): HEALPix base pixel 6, 660 partitions, ~1e10 detections
  (RA ~135-225, Dec -42..+42; includes the dial-in pixel and northern fields), fitted on OSCER.
