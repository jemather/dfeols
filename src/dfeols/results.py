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
        return self.outcomes[y or next(iter(self.outcomes))]["coef"]

    def vcov(self, y: str | None = None, type: str | None = None) -> np.ndarray:
        return self.outcomes[y or next(iter(self.outcomes))]["vcov"][type or self.vcov_default][0]

    def tidy(self, y: str | None = None, type: str | None = None, alpha: float = 0.05):
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
        lines = []
        for y in self.outcomes:
            lines += [
                f"Formula: {self.formula}",
                f"Outcome: {y}",
                f"Fixed effects: {', '.join(self.fe) or 'none'}  levels: {self.fe_levels}",
                f"Obs: {self.N:,} (dropped {self.N_dropped:,})"
                + (f"   Clusters ({self.cluster}): {self.G:,}" if self.G else ""),
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
        """Response `y` as an `estimation` record (dict); see dfeols.record.to_estimation.
        `y` may be left out when the fit has one response."""
        from .record import to_estimation

        return to_estimation(self, y, **kwargs)

    def write_estimation(self, path, **kwargs) -> dict:
        """Write one record per response: at `path` (`<name>.estimation.json`) for one
        response, else `<name>.<response>.estimation.json`. `redact=True` removes
        identifying fields, `publication={...}` adds a publication block. Returns the
        records as written, by response."""
        from .record import write_estimations

        return write_estimations(self, path, **kwargs)

    # ---- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
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
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, default=float))

    @classmethod
    def load(cls, path) -> Result:
        d = json.loads(Path(path).read_text())
        for o in d["outcomes"].values():
            o["coef"] = np.array(o["coef"])
            o["vcov"] = {t: (np.array(v["V"]), v["df_t"]) for t, v in o["vcov"].items()}
        return cls(**d)
