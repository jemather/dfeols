# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _t_p(t, df):
    from scipy import stats

    return 2 * stats.t.sf(np.abs(t), df)


@dataclass
class Result:
    """The result of `fit` (also `Session.fit`, `fit_glm`): estimates for each outcome.

    Printing a Result shows `summary()`. Results are saved with `save` and read back with
    `Result.load`; `write_estimation` writes them as estimation records.

    Attributes
    ----------
    formula : str
        The formula as given.
    label : str
        The fit's name (default its spec id).
    coefnames : list of str
        Names of the reported coefficients, in order.
    outcomes : dict
        Per outcome: `"coef"` (array), `"vcov"` ({type: (matrix, t degrees of freedom)}),
        `"rss"`, `"tss"`, and `"r2_within"` (with FEs) or `"r2"`; IV fits add `"kappa"`
        and `"overid"`.
    N : int
        Observations used.
    N_dropped : int
        Observations dropped (missing values, singletons, `subset`).
    G : int or None
        Number of clusters (the smaller count with two-way clustering).
    cluster : str or None
        The cluster variable.
    fe : list of str
        The absorbed FEs; `fe_levels` gives each one's number of levels.
    collinear : list of str
        Regressors dropped as collinear.
    vcov_default : str
        The vcov type used when none is asked for.
    seconds : float
        Wall time of the fit.
    workdir : str
        The fit's working folder.
    options, timing, run, iv : dict
        The resolved `fit` arguments; timings by stage; sample counts and provenance; and,
        for IV fits, the estimator, instruments and first-stage statistics.
    """

    formula: str
    label: str
    coefnames: list[str]
    outcomes: dict  # y -> {"coef", "vcov": {type: (V, df_t)}, "rss", "r2..."}
    N: int
    N_dropped: int
    G: int | None
    cluster: str | None
    fe: list[str]
    fe_levels: list[int]
    collinear: list[str]
    vcov_default: str
    seconds: float
    workdir: str = ""
    timing: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)  # fit() arguments, resolved
    terms: list = field(default_factory=list)  # every regressor, collinear ones too: {term, canonical}
    run: dict = field(default_factory=dict)  # sample counts, provenance, dfeols facts (record.run_info)
    iv: dict = field(default_factory=dict)  # IV: estimator, endogenous, instruments, first_stage

    def coef(self, y: str | None = None) -> np.ndarray:
        """Coefficients for outcome `y` (default the first), in `coefnames` order."""
        return self.outcomes[y or next(iter(self.outcomes))]["coef"]

    def vcov(self, y: str | None = None, type: str | None = None) -> np.ndarray:
        """Variance matrix for outcome `y` (default the first).

        `type` is one of the vcov types the fit computed (`"iid"`, `"hetero"`, `"CRV1"`);
        default `vcov_default`.
        """
        return self.outcomes[y or next(iter(self.outcomes))]["vcov"][type or self.vcov_default][0]

    def tidy(self, y: str | None = None, type: str | None = None, alpha: float = 0.05):
        """Coefficient table as a pandas DataFrame, like fixest's `coeftable`.

        Parameters
        ----------
        y : str, optional
            Outcome; default the first.
        type : str, optional
            Vcov type; default `vcov_default`.
        alpha : float, default 0.05
            For the confidence interval columns.

        Returns
        -------
        pandas.DataFrame
            Estimate, standard error, t value, p-value and confidence interval, indexed by
            coefficient.
        """
        import pandas as pd
        from scipy import stats

        y = y or next(iter(self.outcomes))
        type = type or self.vcov_default
        b = self.outcomes[y]["coef"]
        V, df = self.outcomes[y]["vcov"][type]
        se = np.sqrt(np.diag(V))
        t = b / se
        q = stats.t.ppf(1 - alpha / 2, df)
        return pd.DataFrame(
            {
                "Estimate": b,
                "Std. Error": se,
                "t value": t,
                "Pr(>|t|)": _t_p(t, df),
                "2.5%": b - q * se,
                "97.5%": b + q * se,
            },
            index=pd.Index(self.coefnames, name="Coefficient"),
        )

    def summary(self, type: str | None = None) -> str:
        """Printable summary: formula, FEs, sample, and the coefficient table per outcome."""
        lines = []
        ncl = (getattr(self, "run", None) or {}).get("n_clusters")
        if ncl and len(ncl) > 1:  # two-way: each count (G, the smaller, sets the adjustment)
            cl_text = "   Clusters: " + ", ".join(f"{c} {g:,}" for c, g in ncl.items())
        else:
            cl_text = f"   Clusters ({self.cluster}): {self.G:,}" if self.G else ""
        for y in self.outcomes:
            lines += [
                f"Formula: {self.formula}",
                f"Outcome: {y}",
                f"Fixed effects: {', '.join(self.fe) or 'none'}  levels: {self.fe_levels}",
                f"Obs: {self.N:,} (dropped {self.N_dropped:,})" + cl_text,
                f"Vcov: {type or self.vcov_default}   Time: {self.seconds:.1f}s",
                "",
            ]
            lines.append(self.tidy(y, type).to_string(float_format=lambda v: f"{v:.6g}"))
            r2 = {k: v for k, v in self.outcomes[y].items() if k.startswith("r2")}
            lines += ["", "  ".join(f"{k}: {v:.4f}" for k, v in r2.items()), ""]
        if self.collinear:
            lines.append(f"Dropped (collinear): {', '.join(self.collinear)}")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

    # ---- estimation records ---------------------------------------------------
    def to_estimation(self, y: str | None = None, **kwargs) -> dict:
        """Outcome `y` as an `estimation` record (a dict); see `dfeols.record.to_estimation`.

        `y` may be left out when the fit has one outcome. Keyword arguments (`redact`,
        `publication`, `level`, ...) are `to_estimation`'s.
        """
        from .record import to_estimation

        return to_estimation(self, y, **kwargs)

    def write_estimation(self, path, **kwargs) -> dict:
        """Write one estimation record per outcome.

        With one outcome the record goes to `path` (`<name>.estrec.json`); with several, to
        `<name>.<outcome>.estrec.json`. `redact=True` leaves out identifying fields and
        `publication={...}` adds a publication block.

        Returns
        -------
        dict
            The records as written, by outcome.
        """
        from .record import write_estimations

        return write_estimations(self, path, **kwargs)

    # ---- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        """The result as plain JSON-serialisable types (arrays as lists)."""
        outs = {}
        for y, o in self.outcomes.items():
            outs[y] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in o.items() if k != "vcov"
            }
            outs[y]["vcov"] = {t: {"V": V.tolist(), "df_t": float(df)} for t, (V, df) in o["vcov"].items()}
        d = {k: v for k, v in self.__dict__.items() if k != "outcomes"}
        d["outcomes"] = outs
        return d

    def save(self, path) -> None:
        """Save the result as JSON; read it back with `Result.load`."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, default=float))

    @classmethod
    def load(cls, path) -> Result:
        """Read a result written by `save`."""
        d = json.loads(Path(path).read_text())
        for o in d["outcomes"].values():
            o["coef"] = np.array(o["coef"])
            o["vcov"] = {t: (np.array(v["V"]), v["df_t"]) for t, v in o["vcov"].items()}
        return cls(**d)
