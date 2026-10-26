# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Deprecated: use `estimation_format` (https://github.com/estimation-format/estimation).

The reader, writer, validator and redactor for `estimation` records moved to the
`estimation-format` package, which dfeols now depends on, together with the schemas.
This module re-exports it so old code keeps working, with two dfeols additions:
DEFAULT_IDENTIFYING includes the dfeols block's fields, and `validate` checks
`dfeols/0.1` with dfeols's built-in check when jsonschema is missing. `dfeols estimation
...` runs the `estimation` command line.
"""

from __future__ import annotations

import warnings
from typing import Any

from estimation_format import *  # noqa: F403
from estimation_format import __all__ as _ef_all
from estimation_format import records as _records

from .record import IDENTIFYING, check_profile

warnings.warn("dfeols.estimation is deprecated; use estimation_format", DeprecationWarning, stacklevel=2)

DEFAULT_IDENTIFYING = sorted({*_records.DEFAULT_IDENTIFYING, *IDENTIFYING})


def validate(record: dict, **kwargs: Any) -> list:
    """`estimation_format.validate`, checking `dfeols/0.1` with dfeols's built-in check
    when jsonschema is missing."""
    if not _records._have_jsonschema():
        kwargs["profiles"] = {"dfeols/0.1": check_profile, **(kwargs.get("profiles") or {})}
    return _records.validate(record, **kwargs)


__all__ = list(_ef_all)
