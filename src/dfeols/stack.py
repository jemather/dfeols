# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Stacked regressions with per-equation fixed effects, and group subsamples.

Several equations estimated as one regression share one coefficient vector and one
(clustered) VCE, so a test that compares coefficients across equations needs no
seemingly-unrelated-regression machinery: the joint VCE covers every cross-equation
contrast. `stack()` builds that regression's rows as a LazyFrame, ready for `prep`:

    eqs = [dfeols.Equation(1, y="eng", x={"x": "feat", "p": "posts"}, fe=["spell", "week"]),
           dfeols.Equation(2, y="unf", x={"x": "lfeat", "p": "lposts"},
                           fe=["id_f", "id_a", "week"], rows=pl.col("risk_set"))]
    st = dfeols.stack(pl.scan_parquet(src), eqs, partition="id_f", keep=["tbin", "id_a"])
    st.frame.sink_parquet(...)        # or pass st.frame to prep() directly
    ds = dfeols.prep(..., partition="id_f", fixed_effects=st.fixed_effects, clusters=["id_a"])
    r = dfeols.fit(ds, st.formula("x:i(tbin) + p:i(tbin)"), cluster="id_a")

What it does:

* Rows: each equation contributes its own rows (`rows`, a Polars expression, or all
  rows), tagged with the equation index `k`, and its outcome in the column `y`.
  `same_rows=True` first drops rows missing any outcome or regressor of any equation,
  so every equation uses the same observations (mediation systems need that).
* Regressors: each role in `x` (e.g. "x", "p") becomes one column per equation,
  `x_k1`, `x_k2`, ...: the equation's own variable on its rows and 0 on the others'.
  `formula()` expands every term that names a role into one term per equation that has
  it, so `x:i(tbin)` becomes `x_k1:i(tbin) + x_k2:i(tbin)` and each equation gets its
  own coefficients. A term naming no role is shared by every equation.
* Fixed effects: every FE is interacted with `k`, so each equation has its own levels.
  The equations' FEs are lined up in slots (`_fe_l0, _fe_l1, ...` for FEs nested in the
  partition key, `_fe_c0, ...` for crossing ones), and the model absorbs `k^<slot>` for
  every slot. An equation with fewer FEs of a kind than the slot count fills the slot
  with a level that its other FEs already span, so the fit is unchanged:
    - a crossing slot: one constant level per equation (spanned by any of its FEs);
    - a local slot: the partition key (spanned by any local FE, since local FEs nest in
      it). An equation that needs a local filler but has no local FE is an error.
  Fillers are redundant but still count in the small-sample K as fixest counts them
  (one level per crossing filler; one per partition group for a local filler, which
  can matter: about groups/N). The usual systems (the same FEs in every equation, or
  as many local FEs in each) need no local filler.

Coefficients equal fitting each equation on its own rows (tested). Standard errors
come from one joint fit: the CRV1 meat sums every equation's scores within a cluster,
which is what makes cross-equation tests valid; the small-sample factor uses the
stacked N and K.

`sample_groups(by, frac, seed)` is a filter expression that keeps a fixed share of
groups (e.g. followers) with all their rows, from a hash of the group id that is the
same on every machine and Polars version (`rowhash`), so a subsample is reproducible
and needs no coordination across files or shards.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from . import rowhash
from .formula import fe_components

log = logging.getLogger(__name__)

_SAMPLE_SEED = 0x5DEECE66D


def sample_groups(by: str | list[str], frac: float, seed: int = 0) -> pl.Expr:
    """Boolean expression: True for rows whose group (`by`) falls in a `frac` share.

    The same group is in or out on any machine and for any row order or file split:
    hash(group, seed) / 2^64 < frac, with the stable row hash of `dfeols.rowhash`.
    Nested fractions are nested samples (same seed): a 5% sample is inside the 15% one.
    """
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    cols = [by] if isinstance(by, str) else list(by)
    cut = np.uint64(min(int(frac * 2.0**64), 2**64 - 1))
    s = np.uint64((_SAMPLE_SEED ^ int(seed)) & (2**64 - 1))

    def keep(st: pl.Series) -> pl.Series:
        df = st.struct.unnest()
        h = rowhash.row_hashes(df, cols, int(s))
        return pl.Series(h < cut if frac < 1 else np.ones(len(h), bool))

    return pl.struct(cols).map_batches(keep, return_dtype=pl.Boolean, is_elementwise=True)


@dataclass
class Equation:
    """One equation of a stacked system.

    k     equation index (a positive integer, unique within the system)
    y     outcome column
    x     regressor roles -> this equation's column, e.g. {"x": "lfeat", "p": "lposts"}
    fe    absorbed FEs (column names or a^b interactions of integer columns)
    rows  Polars expression selecting this equation's rows (None: every row)
    """

    k: int
    y: str
    x: dict[str, str] = field(default_factory=dict)
    fe: list[str] = field(default_factory=list)
    rows: pl.Expr | None = None


@dataclass
class Stacked:
    frame: pl.LazyFrame
    equations: list[Equation]
    fixed_effects: list[str]  # for prep(fixed_effects=...): "k^_fe_l0", ...
    slots: dict[str, dict[int, str]]  # slot -> {k: the FE it holds, or a filler label}
    roles: dict[str, list[int]]  # role -> equations that have it

    @property
    def fe(self) -> str:
        return " + ".join(self.fixed_effects)

    def terms(self, spec: str) -> str:
        """Expand role names in `spec` ("x:i(tbin) + p") into per-equation terms."""
        out = []
        for term in (t.strip() for t in spec.split("+")):
            if not term:
                continue
            used = [r for r in self.roles if re.search(rf"(?<![\w.]){re.escape(r)}(?![\w(])", term)]
            if not used:
                out.append(term)
                continue
            for eq in self.equations:
                if all(r in eq.x for r in used):
                    t = term
                    for r in used:
                        t = re.sub(rf"(?<![\w.]){re.escape(r)}(?![\w(])", f"{r}_k{eq.k}", t)
                    out.append(t)
        return " + ".join(out)

    def formula(self, spec: str, y: str = "y") -> str:
        """Full dfeols formula: `y ~ <expanded terms> | <per-equation FEs>`."""
        return f"{y} ~ {self.terms(spec)} | {self.fe}"

    def coef_index(self, coefnames: list[str], role: str, k: int, pattern: str = "") -> np.ndarray:
        """Positions in `coefnames` of equation k's terms for `role` (e.g. role "x", k 1 ->
        every coefficient whose name contains the token x_k1), in their fitted order; an
        optional regex `pattern` narrows them further (e.g. "tbin")."""
        tok = re.compile(rf"(?<![\w.]){re.escape(f'{role}_k{k}')}(?![\w])")
        pat = re.compile(pattern) if pattern else None
        return np.array(
            [i for i, c in enumerate(coefnames) if tok.search(c) and (pat is None or pat.search(c))],
            dtype=np.int64,
        )


def _infer_local(lf: pl.LazyFrame, fes: list[str], partition: str) -> set[str]:
    """FEs nested in `partition` (each level within one partition group): one streaming
    aggregate per FE."""
    out = set()
    for fe in fes:
        comps = fe_components(fe)
        if comps == [partition] or partition in comps:
            out.add(fe)
            continue
        m = (
            lf.group_by(comps)
            .agg(pl.col(partition).n_unique().alias("_n"))
            .select(pl.col("_n").max())
            .collect(engine="streaming")
            .item()
        )
        if m is not None and m <= 1:
            out.add(fe)
    return out


def _codes(lf: pl.LazyFrame, fes: list[str]) -> dict[str, pl.Expr]:
    """An Int64 expression per FE: the column itself, or for a^b a mixed-radix code
    from the components' ranges (one streaming min/max pass)."""
    comps = sorted({c for fe in fes for c in fe_components(fe) if "^" in fe})
    rng = {}
    if comps:
        st = lf.select(
            [pl.col(c).min().alias(f"{c}__min") for c in comps]
            + [pl.col(c).max().alias(f"{c}__max") for c in comps]
        ).collect(engine="streaming")
        rng = {c: (int(st[f"{c}__min"][0]), int(st[f"{c}__max"][0])) for c in comps}
    schema = lf.collect_schema()
    out = {}
    for fe in fes:
        cs = fe_components(fe)
        for c in cs:
            if not schema[c].is_integer():
                raise TypeError(f"stack: FE column {c!r} is {schema[c]}; encode it as integers first")
        if len(cs) == 1:
            out[fe] = pl.col(cs[0]).cast(pl.Int64)
            continue
        code, span = pl.lit(0, pl.Int64), 1
        for c in cs:
            lo, hi = rng[c]
            width = hi - lo + 1
            code = code * width + (pl.col(c).cast(pl.Int64) - lo)
            span *= width
        if span >= 2**62:
            raise OverflowError(f"stack: {fe} has {span:,} possible levels; too many to code in int64")
        out[fe] = code
    return out


def stack(
    data,
    equations: list[Equation],
    partition: str,
    keep: list[str] | tuple[str, ...] = (),
    same_rows: bool = False,
    local: set[str] | list[str] | None = None,
) -> Stacked:
    """Build the stacked system's rows (see the module docstring).

    data       LazyFrame or DataFrame (or a Parquet path/glob)
    partition  the key the shards will be partitioned by (FEs nested in it are local)
    keep       columns to carry unchanged (bin factors, clusters, weights)
    same_rows  drop rows missing any outcome or regressor of any equation first
    local      FEs nested in `partition` (skips inferring them with a pass per FE)
    """
    lf = data.lazy() if isinstance(data, pl.DataFrame) else data
    if isinstance(lf, str):
        lf = pl.scan_parquet(lf)
    ks = [e.k for e in equations]
    if len(set(ks)) != len(ks) or min(ks) < 1:
        raise ValueError(f"equation indices must be distinct positive integers, got {ks}")
    for e in equations:
        if not e.fe:
            raise ValueError(f"equation {e.k} has no FE; give it at least one (its intercept)")
    all_fes = sorted({fe for e in equations for fe in e.fe})
    loc = set(local) if local is not None else _infer_local(lf, all_fes, partition)
    nl = max(sum(fe in loc for fe in e.fe) for e in equations)
    nc = max(sum(fe not in loc for fe in e.fe) for e in equations)
    codes = _codes(lf, all_fes)
    roles: dict[str, list[int]] = {}
    for e in equations:
        for r in e.x:
            roles.setdefault(r, []).append(e.k)
    slots: dict[str, dict[int, str]] = {f"_fe_l{i}": {} for i in range(nl)}
    slots.update({f"_fe_c{i}": {} for i in range(nc)})
    if same_rows:
        need = sorted({e.y for e in equations} | {c for e in equations for c in e.x.values()})
        lf = lf.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in need]))
    parts = []
    for e in equations:
        lo = [fe for fe in e.fe if fe in loc]
        cr = [fe for fe in e.fe if fe not in loc]
        if len(lo) < nl and not lo:
            raise ValueError(
                f"equation {e.k} has no FE nested in {partition!r}, but others have {nl}: "
                "no redundant filler exists for its local FE slot. Give it a local FE, or "
                "stack it with equations that have none."
            )
        cols = [pl.lit(e.k, pl.Int16).alias("k"), pl.col(e.y).cast(pl.Float64).alias("y")]
        for i in range(nl):
            if i < len(lo):
                cols.append(codes[lo[i]].alias(f"_fe_l{i}"))
                slots[f"_fe_l{i}"][e.k] = lo[i]
            else:
                cols.append(pl.col(partition).cast(pl.Int64).alias(f"_fe_l{i}"))
                slots[f"_fe_l{i}"][e.k] = f"(filler: {partition})"
                log.warning(
                    "stack: equation %d fills local FE slot %d with %s (redundant; its levels count in K)",
                    e.k, i, partition,
                )  # fmt: skip
        for i in range(nc):
            if i < len(cr):
                cols.append(codes[cr[i]].alias(f"_fe_c{i}"))
                slots[f"_fe_c{i}"][e.k] = cr[i]
            else:
                cols.append(pl.lit(0, pl.Int64).alias(f"_fe_c{i}"))
                slots[f"_fe_c{i}"][e.k] = "(filler: constant)"
        for r, ek in roles.items():
            for k in ek:
                name = f"{r}_k{k}"
                cols.append(
                    (pl.col(e.x[r]).cast(pl.Float64) if k == e.k else pl.lit(0.0, pl.Float64)).alias(name)
                )
        carry = [c for c in dict.fromkeys([partition, *keep]) if c not in {"k", "y"}]
        cols += [pl.col(c) for c in carry]
        sub = lf.filter(e.rows) if e.rows is not None else lf
        parts.append(sub.select(cols))
    frame = pl.concat(parts, how="vertical")
    fixed_effects = [f"k^{s}" for s in slots]
    return Stacked(frame, list(equations), fixed_effects, slots, roles)
