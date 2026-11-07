# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Deprecated: use `estrec` (https://github.com/estrec/estrec).

The reader, writer, validator and redactor for `estimation` records moved to the
`estrec` package, which dfeols now depends on, together with the schemas.
This module re-exports it so old code keeps working, with two dfeols additions:
DEFAULT_IDENTIFYING includes the dfeols block's fields, and `validate` checks
`dfeols/0.1` with dfeols's built-in check when jsonschema is missing. `dfeols estrec
...` runs the `estrec` command line.
"""

from __future__ import annotations

import warnings
from typing import Any

from estrec import *  # noqa: F403
from estrec import __all__ as _ef_all
from estrec import records as _records

from .record import IDENTIFYING, check_profile

warnings.warn("dfeols.estimation is deprecated; use estrec", DeprecationWarning, stacklevel=2)

DEFAULT_IDENTIFYING = sorted({*_records.DEFAULT_IDENTIFYING, *IDENTIFYING})


def validate(record: dict, **kwargs: Any) -> list:
    """`estrec.validate`, checking `dfeols/0.1` with dfeols's built-in check
    when jsonschema is missing."""
    if not _records._have_jsonschema():
        kwargs["profiles"] = {"dfeols/0.1": check_profile, **(kwargs.get("profiles") or {})}
    return _records.validate(record, **kwargs)


__all__ = list(_ef_all)
