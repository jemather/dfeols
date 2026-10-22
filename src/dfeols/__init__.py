# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""dfeols: sharded, elastic high-dimensional fixed-effects OLS for HPC clusters."""

import logging as _logging

from .cluster import executor
from .estimate import build_plan, fit
from .executors import DaskExecutor, SerialExecutor
from .lock import LockedError
from .prep import Dataset, prep
from .results import Result
from .site import profile

__all__ = [
    "prep",
    "fit",
    "Dataset",
    "Result",
    "SerialExecutor",
    "DaskExecutor",
    "build_plan",
    "executor",
    "profile",
    "LockedError",
]
__version__ = "0.7.0"

_log = _logging.getLogger("dfeols")
if not _log.handlers:
    _h = _logging.StreamHandler()
    _h.setFormatter(_logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    _log.addHandler(_h)
    _log.setLevel(_logging.INFO)
