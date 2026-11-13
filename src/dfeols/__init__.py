# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""dfeols: sharded, elastic high-dimensional fixed-effects OLS and IV for HPC clusters."""

import logging as _logging

from .cluster import executor
from .dthazard import lpm_dthazard
from .estimate import build_plan, fit
from .executors import DaskExecutor, SerialExecutor
from .glm import fit_glm
from .lock import LockedError
from .planner import plan
from .prep import Dataset, prep
from .record import write_collection
from .results import Result
from .session import Session
from .site import profile
from .stack import Equation, Stacked, sample_groups, stack

__all__ = [
    "prep",
    "fit",
    "Dataset",
    "Result",
    "SerialExecutor",
    "DaskExecutor",
    "build_plan",
    "plan",
    "executor",
    "profile",
    "LockedError",
    "write_collection",
    "Session",
    "lpm_dthazard",
    "stack",
    "Equation",
    "Stacked",
    "sample_groups",
    "fit_glm",
]
__version__ = "0.8.0"

_log = _logging.getLogger("dfeols")
if not _log.handlers:
    _h = _logging.StreamHandler()
    _h.setFormatter(_logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    _log.addHandler(_h)
    _log.setLevel(_logging.INFO)
