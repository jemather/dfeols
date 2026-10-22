"""dfeols: sharded, elastic high-dimensional fixed-effects OLS for HPC clusters."""

import logging as _logging

from .cluster import executor
from .estimate import build_plan, fit
from .site import profile
from .executors import DaskExecutor, SerialExecutor
from .prep import Dataset, prep
from .results import Result

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
]
__version__ = "0.6.1"

_log = _logging.getLogger("dfeols")
if not _log.handlers:
    _h = _logging.StreamHandler()
    _h.setFormatter(_logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    _log.addHandler(_h)
    _log.setLevel(_logging.INFO)
