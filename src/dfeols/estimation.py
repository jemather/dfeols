# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Read, write, validate and redact `estimation` records (docs/design/estimation-format.md).

One record describes one estimation run as a JSON object. This module is
package-independent: it uses only the standard library, plus numpy (to accept
arrays) and pyarrow (for spilled attachments), both imported lazily.

    write(record, path)       spill large structures, add the core digest, check sizes
    read(path)                load, decoding "NaN"/"Inf"/"-Inf" in numeric fields
    validate(record)          core rules, sizes, digest, profiles, optional JSON Schema
    redact(record)            remove or HMAC the identifying fields
    core_digest(record)       sha256 over the RFC 8785 canonical form of the result sections

The JSON Schemas are in the repository's schema/ folder (or $ESTIMATION_SCHEMA_DIR):
`validate` checks the core schema plus each profile the record claims, and `redact`
takes the fields marked `x-identifying` in the core and each claimed profile. Where
the schemas or the `jsonschema` package are missing, both fall back to built-in
checks and DEFAULT_IDENTIFYING, and `validate` says so.

Command line: `python -m dfeols.estimation {validate,redact,digest} ...`
(also `dfeols estimation ...`).
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import functools
import hashlib
import hmac
import json
import math
import os
import re
import sys
import uuid
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

SCHEMA = "estimation/0.1"
WARN_BYTES = 1_000_000  # whole record
WARN_ELEMENTS = 10_000  # one array (nested lists count their leaves)
FAIL_BYTES = 16_000_000

REQUIRED = ("schema", "meta", "parameters")
EXPECTED = ("metrics", "specification", "provenance")
OPTIONAL = ("structures", "publication", "validation", "attachments", "extensions", "history")
# Sections covered by meta.core_digest (meta without the digest itself)
CORE = ("meta", "specification", "metrics", "parameters", "structures")
STATUSES = ("ok", "warning", "failed")
STRUCTURE_KINDS = ("matrix", "table", "tree", "series", "object", "link")
NONFINITE = {"NaN": math.nan, "Inf": math.inf, "-Inf": -math.inf}
# parameters columns that hold text; every other column is numeric for decoding
TEXT_COLUMNS = {"term", "canonical", "response", "component", "group", "dropped", "label"}

# Identifying fields when the schemas aren't available: what schema/ marks "x-identifying"
# in the core and the dfeols profile (tests/test_schema.py keeps the two equal). `*` matches
# any key of a dict, `[*]` any item of a list. Content hashes are not identifying and stay.
DEFAULT_IDENTIFYING = (
    "provenance.agent",
    "provenance.environment.host",
    "provenance.environment.hostname",
    "provenance.execution.invocation.command_line",
    "provenance.execution.invocation.argv",
    "provenance.execution.invocation.script_path",
    "provenance.execution.invocation.working_directory",
    "provenance.execution.scheduler.JobID",
    "provenance.execution.scheduler.JobName",
    "provenance.execution.scheduler.Account",
    "provenance.execution.scheduler.User",
    "provenance.execution.scheduler.NodeList",
    "provenance.execution.scheduler.SubmitLine",
    "provenance.execution.scheduler.WorkDir",
    "provenance.execution.workers[*].host",
    "provenance.data[*].location",
    "provenance.software[*].lock.path",
    "provenance.software[*].container.path",
    "extensions.dfeols.shards.folder",
    "extensions.dfeols.dask.scheduler_address",
)


class EstimationError(ValueError):
    """A record that can't be written or read as `estimation`."""


@dataclass(frozen=True)
class Issue:
    level: str  # "error", "warning" or "info"
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.path or '(record)'}: {self.message}"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _producer() -> dict:
    from . import __version__

    return {"name": "dfeols.estimation", "version": __version__}


# ---------------------------------------------------------------- encoding


def to_json(obj: Any) -> Any:
    """A copy of `obj` made of JSON types: numpy arrays and scalars converted, non-finite
    floats as the strings "NaN"/"Inf"/"-Inf", datetimes as ISO 8601 UTC, paths as strings."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, _dt.datetime):
        if obj.tzinfo is None:
            raise EstimationError(f"naive datetime {obj!r}: times must be timezone-aware (UTC)")
        return obj.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if hasattr(obj, "tolist"):  # numpy arrays and scalars, without importing numpy
        return to_json(obj.tolist())
    raise EstimationError(f"can't encode {type(obj).__name__} as JSON")


def _decode(v: Any) -> Any:
    if isinstance(v, str):
        return NONFINITE.get(v, v)
    if isinstance(v, list):
        return [_decode(x) for x in v]
    return v


def decode_nonfinite(record: dict) -> dict:
    """A copy with "NaN"/"Inf"/"-Inf" turned into floats where the format puts numbers:
    metric values, numeric parameters columns and matrix data. Text fields are left as is,
    so a term named "Inf" stays a string."""
    rec = copy.deepcopy(record)
    metrics = rec.get("metrics")
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if isinstance(v, dict) and "value" in v:
                v["value"] = _decode(v["value"])
            else:
                metrics[k] = _decode(v)
    for row in rec.get("parameters") or []:
        if isinstance(row, dict):
            for k, v in row.items():
                if k not in TEXT_COLUMNS:
                    row[k] = _decode(v)
    for s in _structures(rec):
        if s.get("kind") == "matrix" and "data" in s:
            s["data"] = _decode(s["data"])
    return rec


def _dumps(record: dict, indent: int | None = None) -> str:
    return json.dumps(record, ensure_ascii=False, allow_nan=False, indent=indent)


def _reject_constant(name: str):
    raise EstimationError(f"bare {name} in JSON; the format writes non-finite values as strings")


# ---------------------------------------------------------------- RFC 8785 canonical JSON


def _es_number(x: float) -> str:
    """ECMAScript Number.prototype.toString, which RFC 8785 prescribes for numbers."""
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    mant, _, exp = repr(abs(x)).partition("e")  # repr is the shortest round-trip form
    ip, _, fp = mant.partition(".")
    digits = ip + fp
    n = len(ip) + (int(exp) if exp else 0)  # decimal point after n digits
    stripped = digits.lstrip("0")
    n -= len(digits) - len(stripped)
    digits = stripped.rstrip("0")
    k = len(digits)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * -n + digits
    e = n - 1
    frac = "." + digits[1:] if k > 1 else ""
    return f"{sign}{digits[0]}{frac}e{'+' if e > 0 else '-'}{abs(e)}"


def _canon(o: Any, out: list[str]) -> None:
    if o is None:
        out.append("null")
    elif o is True:
        out.append("true")
    elif o is False:
        out.append("false")
    elif isinstance(o, int):
        if abs(o) > 2**53:
            raise EstimationError(f"integer {o} is outside the IEEE double range RFC 8785 allows")
        out.append(str(o))
    elif isinstance(o, float):
        if not math.isfinite(o):
            raise EstimationError("non-finite number in canonical JSON; encode it with to_json first")
        out.append(_es_number(o))
    elif isinstance(o, str):
        out.append(json.dumps(o, ensure_ascii=False))
    elif isinstance(o, list):
        out.append("[")
        for i, v in enumerate(o):
            if i:
                out.append(",")
            _canon(v, out)
        out.append("]")
    elif isinstance(o, dict):
        out.append("{")
        # keys sorted by their UTF-16 code units
        for i, k in enumerate(sorted(o, key=lambda s: s.encode("utf-16-be"))):
            if i:
                out.append(",")
            out.append(json.dumps(k, ensure_ascii=False))
            out.append(":")
            _canon(o[k], out)
        out.append("}")
    else:
        raise EstimationError(f"can't canonicalise {type(o).__name__}")


def canonical_json(obj: Any) -> bytes:
    """RFC 8785 (JSON Canonicalization Scheme) serialisation of a JSON value, as UTF-8."""
    out: list[str] = []
    _canon(obj, out)
    return "".join(out).encode("utf-8")


# ---------------------------------------------------------------- core digest


def core_digest(record: dict) -> str:
    """sha256 (hex) of the canonical form of the result sections: `meta` without
    `core_digest`, `specification`, `metrics`, `parameters`, `structures`. Absent sections
    are left out, so adding a section later changes the digest."""
    rec = to_json(record)
    view = {k: rec[k] for k in CORE if k in rec}
    if isinstance(view.get("meta"), dict):
        view["meta"] = {k: v for k, v in view["meta"].items() if k != "core_digest"}
    return hashlib.sha256(canonical_json(view)).hexdigest()


def add_digest(record: dict) -> dict:
    """Set `meta.core_digest` in place and return the record."""
    record.setdefault("meta", {})["core_digest"] = {
        "algorithm": "sha256",
        "canonicalization": "RFC8785",
        "value": core_digest(record),
    }
    return record


def check_digest(record: dict) -> bool | None:
    """True if `meta.core_digest` matches the record, False if not, None if absent."""
    d = (record.get("meta") or {}).get("core_digest")
    if d is None:
        return None
    if not isinstance(d, dict) or d.get("algorithm") != "sha256":
        return False
    return hmac.compare_digest(str(d.get("value", "")), core_digest(record))


# ---------------------------------------------------------------- sizes


def count_elements(v: Any) -> int:
    """Elements of an array as the size limits count them: items of a list, with nested
    lists counted by their leaves (a 100 x 100 matrix is 10,000). Dicts count as one."""
    if isinstance(v, list):
        return sum(count_elements(x) if isinstance(x, list) else 1 for x in v)
    return 1


def _large_arrays(v: Any, path: str, limit: int) -> Iterable[tuple[str, int]]:
    """Outermost arrays above `limit` elements, with their paths."""
    if isinstance(v, list):
        n = count_elements(v)
        if n > limit:
            yield path, n
            return
        for i, x in enumerate(v):
            yield from _large_arrays(x, f"{path}[{i}]", limit)
    elif isinstance(v, dict):
        for k, x in v.items():
            yield from _large_arrays(x, f"{path}.{k}" if path else k, limit)


# ---------------------------------------------------------------- attachments and spilling

# Fields of a structure that hold its data; everything else (type, ssc, cluster, ...) stays
# in the record when the structure is spilled, so profiles can still check it.
_DATA_FIELDS = {
    "matrix": ("labels", "row_labels", "col_labels", "data"),
    "table": ("columns", "rows"),
    "series": ("index", "values"),
}


def _structures(record: dict) -> Iterable[dict]:
    for v in (record.get("structures") or {}).values():
        for s in v if isinstance(v, list) else [v]:
            if isinstance(s, dict):
                yield s


def _structure_slots(record: dict) -> list[tuple[str, dict, Any, Any]]:
    """(id, structure, container, key) for every structure not yet spilled."""
    slots: list[tuple[str, dict, Any, Any]] = []
    structures = record.get("structures") or {}
    for name, v in structures.items():
        if isinstance(v, list):
            slots += [
                (f"{name}_{i}", s, v, i)
                for i, s in enumerate(v)
                if isinstance(s, dict) and s.get("kind") != "link"
            ]
        elif isinstance(v, dict) and v.get("kind") != "link":
            slots.append((name, v, structures, name))
    return slots


def _data_of(s: dict) -> dict:
    fields = _DATA_FIELDS.get(s.get("kind", ""))
    if fields is None:  # tree, object, unknown: the whole structure is data
        return {k: v for k, v in s.items() if k != "kind"}
    return {k: s[k] for k in fields if k in s}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_spill(s: dict, data: dict, path_stem: Path) -> tuple[Path, dict]:
    """Write a structure's data; Parquet for matrices and tables, JSON otherwise."""
    kind = s.get("kind")
    if kind in ("matrix", "table"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        try:
            if kind == "matrix":
                rows = data.get("data", [])
                labels = data.get("row_labels") or data.get("labels") or [None] * len(rows)
                table = pa.table(
                    {
                        "label": pa.array(labels, pa.string()),
                        "row": pa.array(_decode(rows), pa.list_(pa.float64())),
                    }
                )
                extra = {k: data[k] for k in ("labels", "col_labels") if k in data}
                table = table.replace_schema_metadata({"estimation": json.dumps(extra)})
            else:
                cols = data.get("columns", [])
                rows = data.get("rows", [])
                table = pa.table({c: [r[j] for r in rows] for j, c in enumerate(cols)})
            path = path_stem.with_suffix(".parquet")
            pq.write_table(table, path)
            return path, {
                "media_type": "application/vnd.apache.parquet",
                "rows": table.num_rows,
                "columns": table.column_names,
            }
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            pass  # mixed column types: fall back to JSON
    path = path_stem.with_suffix(".json")
    path.write_text(_dumps(data), encoding="utf-8")
    return path, {"media_type": "application/json"}


def spill(
    record: dict,
    directory: str | os.PathLike,
    stem: str,
    max_elements: int = WARN_ELEMENTS,
    max_bytes: int = WARN_BYTES,
) -> list[str]:
    """Move large structures into side files in `directory` and leave links, in place.

    A structure goes out if one of its arrays has more than `max_elements` elements; then,
    while the record is still above `max_bytes`, the largest remaining one goes. Only the
    data fields move (a vcov keeps its type, ssc and cluster in the record). Returns the
    ids of the new attachments. `record` must already be JSON-encoded (`to_json`)."""
    directory = Path(directory)
    attachments = record.setdefault("attachments", [])
    taken = {a.get("id") for a in attachments if isinstance(a, dict)}
    done: list[str] = []

    def out(sid: str, s: dict, container: Any, key: Any) -> None:
        aid, i = sid, 1
        while aid in taken:
            i += 1
            aid = f"{sid}_{i}"
        taken.add(aid)
        data = _data_of(s)
        path, info = _write_spill(s, data, directory / f"{stem}.{aid}")
        attachments.append(
            {
                "id": aid,
                "role": "structure",
                "path": path.name,
                "created": _now(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                **info,
            }
        )
        link = {k: v for k, v in s.items() if k not in data}
        link.update({"kind": "link", "attachment": aid, "spilled_kind": s.get("kind")})
        container[key] = link
        done.append(aid)

    for sid, s, container, key in _structure_slots(record):
        if any(True for _ in _large_arrays(_data_of(s), "", max_elements)):
            out(sid, s, container, key)
    while len(_dumps(record).encode()) > max_bytes:
        slots = _structure_slots(record)
        if not slots:
            break
        out(*max(slots, key=lambda t: len(_dumps(t[1]))))
    if not attachments:
        del record["attachments"]
    return done


def attachment(record: dict, aid: str) -> dict:
    for a in record.get("attachments") or []:
        if isinstance(a, dict) and a.get("id") == aid:
            return a
    raise KeyError(f"no attachment {aid!r}")


def attachment_path(record_path: str | os.PathLike, att: dict) -> Path:
    """Where an attachment's file is: `path` is relative to the record's folder."""
    p = Path(att["path"])
    return p if p.is_absolute() else Path(record_path).parent / p


def load_structure(record_path: str | os.PathLike, record: dict, link: dict, verify: bool = True) -> dict:
    """Rebuild a spilled structure from its link and attachment file."""
    att = attachment(record, link["attachment"])
    path = attachment_path(record_path, att)
    if verify and att.get("sha256") and _sha256_file(path) != att["sha256"]:
        raise EstimationError(f"{path}: sha256 does not match attachment {att['id']!r}")
    s = {k: v for k, v in link.items() if k not in ("kind", "attachment", "spilled_kind")}
    s["kind"] = link.get("spilled_kind", "object")
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        t = pq.read_table(path)
        if s["kind"] == "matrix":
            extra = json.loads((t.schema.metadata or {}).get(b"estimation", b"{}"))
            s.update(extra)
            s["data"] = t.column("row").to_pylist()
            if "labels" not in extra:
                s["row_labels"] = t.column("label").to_pylist()
        else:
            s["columns"] = t.column_names
            s["rows"] = [list(r.values()) for r in t.to_pylist()]
    else:
        s.update(json.loads(path.read_text(encoding="utf-8")))
    return s


# ---------------------------------------------------------------- writer and reader


def new_record(
    parameters: list[dict],
    *,
    estimator: dict | None = None,
    responses: list[str] | None = None,
    profiles: list[str] | None = None,
    **sections: Any,
) -> dict:
    """A record skeleton with `schema` and `meta` (id, created, status) filled."""
    meta: dict[str, Any] = {"id": str(uuid.uuid4()), "created": _now(), "status": "ok"}
    if estimator is not None:
        meta["estimator"] = estimator
    if responses is not None:
        meta["responses"] = responses
    if profiles is not None:
        meta["profiles"] = profiles
    meta.update(sections.pop("meta", {}))
    return {"schema": SCHEMA, "meta": meta, "parameters": parameters, **sections}


def prepare(
    record: dict,
    *,
    digest: bool = True,
    redact_identifying: bool = False,
    by: dict | None = None,
) -> dict:
    """The record as it would be written, without spilling: encoded, `schema`, `meta.id`
    and `meta.created` filled if missing, optionally redacted, history and digest added."""
    rec = to_json(record)
    rec.setdefault("schema", SCHEMA)
    meta = rec.setdefault("meta", {})
    meta.setdefault("id", str(uuid.uuid4()))
    meta.setdefault("created", _now())
    rec.setdefault("history", []).append(
        {
            "when": _now(),
            "by": by or _producer(),
            "action": "created",
            "sections": [k for k in (*REQUIRED, *EXPECTED, *OPTIONAL) if k in rec and k != "schema"],
        }
    )
    if redact_identifying:
        rec = redact(rec, by=by)
    if digest:
        add_digest(rec)
    else:
        meta.pop("core_digest", None)
    return rec


def write(
    record: dict,
    path: str | os.PathLike,
    *,
    digest: bool = True,
    spill_large: bool = True,
    redact_identifying: bool = False,
    by: dict | None = None,
    indent: int | None = 1,
) -> dict:
    """Write `record` to `path` (`<name>.estimation.json`) and return it as written.

    Large structures are spilled to side files next to it (`spill`), the core digest is
    computed after spilling (on by default), and the size limits are applied: a warning
    above 1 MB or 10,000 elements in one array, an error above 16 MB."""
    path = Path(path)
    rec = prepare(record, digest=False, redact_identifying=redact_identifying, by=by)
    if spill_large:
        stem = path.name.removesuffix(".json").removesuffix(".estimation")
        spill(rec, path.parent, stem)
    if digest:
        add_digest(rec)
    text = _dumps(rec, indent)
    size = len(text.encode())
    if size > FAIL_BYTES:
        raise EstimationError(f"{path}: record is {size:,} bytes, above the {FAIL_BYTES:,} byte limit")
    for issue in _size_issues(rec, size):
        warnings.warn(f"{path}: {issue}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return rec


def loads(text: str | bytes, *, nonfinite: str = "float") -> dict:
    rec = json.loads(text, parse_constant=_reject_constant)
    if not isinstance(rec, dict):
        raise EstimationError("an estimation record is a JSON object")
    if nonfinite == "float":
        return decode_nonfinite(rec)
    if nonfinite != "string":
        raise ValueError("nonfinite must be 'float' or 'string'")
    return rec


def read(path: str | os.PathLike, *, nonfinite: str = "float", check: bool = False) -> dict:
    """Load a record. Non-finite values come back as floats in numeric fields (`"string"`
    keeps them as written). With `check=True`, raise if `validate` finds an error."""
    rec = loads(Path(path).read_bytes(), nonfinite="string")
    if check:
        errors = [i for i in validate(rec, base_dir=Path(path).parent) if i.level == "error"]
        if errors:
            raise EstimationError(f"{path}: " + "; ".join(map(str, errors)))
    return decode_nonfinite(rec) if nonfinite == "float" else rec


# ---------------------------------------------------------------- validator


def _is_number(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)) or v in NONFINITE


def _size_issues(rec: dict, size: int) -> list[Issue]:
    issues = []
    if size > FAIL_BYTES:
        issues.append(Issue("error", "", f"{size:,} bytes, above the {FAIL_BYTES:,} byte limit"))
    elif size > WARN_BYTES:
        big = max((k for k in rec if k != "schema"), key=lambda k: len(_dumps(rec[k]).encode()), default="")
        issues.append(Issue("warning", "", f"{size:,} bytes, above {WARN_BYTES:,}; largest section {big!r}"))
    for p, n in _large_arrays(rec, "", WARN_ELEMENTS):
        issues.append(Issue("warning", p, f"{n:,} elements, above {WARN_ELEMENTS:,}; use an attachment"))
    return issues


def _check_core(rec: dict, structural: bool = True) -> list[Issue]:
    """Checks JSON Schema can't express (expected sections, cross-references, duplicate
    ids), plus, when `structural` (no schema in use), the core's shape by hand."""
    issues: list[Issue] = []

    def err(p: str, m: str) -> None:
        if structural:
            issues.append(Issue("error", p, m))

    xerr = lambda p, m: issues.append(Issue("error", p, m))  # always: the schema can't check these
    warn = lambda p, m: issues.append(Issue("warning", p, m))
    schema = rec.get("schema")
    if not isinstance(schema, str) or not schema.startswith("estimation/"):
        err("schema", f"expected 'estimation/<version>', got {schema!r}")
    elif schema != SCHEMA:
        warn("schema", f"{schema} is not {SCHEMA}; checked as {SCHEMA}")
    for k in REQUIRED:
        if k not in rec:
            err(k, "required section missing")
    for k in EXPECTED:
        if k not in rec:
            warn(k, "expected section missing")
    for k in rec:
        if k not in (*REQUIRED, *EXPECTED, *OPTIONAL):
            warn(k, "unknown section (tool-specific content belongs in extensions)")

    meta = rec.get("meta")
    if "meta" in rec and not isinstance(meta, dict):
        err("meta", "must be an object")
        meta = {}
    meta = meta or {}
    for k in ("id", "created", "status"):
        if "meta" in rec and k not in meta:
            err(f"meta.{k}", "required")
    if "status" in meta and meta["status"] not in STATUSES:
        err("meta.status", f"must be one of {', '.join(STATUSES)}")
    for k in ("profiles", "responses", "tags", "messages"):
        if k in meta and not isinstance(meta[k], list):
            err(f"meta.{k}", "must be a list")
    if "estimator" in meta and not isinstance(meta["estimator"], dict):
        err("meta.estimator", "must be an object ({family, method, ...})")

    params = rec.get("parameters")
    if "parameters" in rec:
        if not isinstance(params, list):
            err("parameters", "must be a list of rows")
        else:
            for i, row in enumerate(params):
                if not isinstance(row, dict):
                    err(f"parameters[{i}]", "must be an object")
                elif not isinstance(row.get("term"), str):
                    err(f"parameters[{i}].term", "required string")
                elif "canonical" in row and not isinstance(row["canonical"], dict):
                    err(f"parameters[{i}].canonical", "must be an object ({name, factors})")

    metrics = rec.get("metrics")
    if "metrics" in rec:
        if not isinstance(metrics, dict):
            err("metrics", "must be a flat object of scalars")
        else:
            for k, v in metrics.items():
                if isinstance(v, dict):
                    if "value" not in v:
                        err(f"metrics.{k}", "an object metric needs 'value'")
                elif v is not None and not _is_number(v) and not isinstance(v, (str, bool)):
                    err(f"metrics.{k}", "must be a scalar or {value, definition, ...}")

    for k in ("specification", "provenance", "publication", "validation", "structures", "extensions"):
        if k in rec and not isinstance(rec[k], dict):
            err(k, "must be an object")

    att_ids = set()
    if "attachments" in rec:
        if not isinstance(rec["attachments"], list):
            err("attachments", "must be a list")
        else:
            for i, a in enumerate(rec["attachments"]):
                if not isinstance(a, dict) or not a.get("id") or not a.get("path"):
                    err(f"attachments[{i}]", "needs 'id' and 'path'")
                    continue
                if a["id"] in att_ids:
                    xerr(f"attachments[{i}].id", f"duplicate id {a['id']!r}")
                att_ids.add(a["id"])
                if "sha256" not in a:
                    warn(f"attachments[{i}]", "no sha256: a file found later can't be checked")

    for name, v in (rec.get("structures") or {}).items() if isinstance(rec.get("structures"), dict) else []:
        items = v if isinstance(v, list) else [v]
        for i, s in enumerate(items):
            p = f"structures.{name}" + (f"[{i}]" if isinstance(v, list) else "")
            if not isinstance(s, dict):
                err(p, "must be an object with 'kind'")
            elif s.get("kind") not in STRUCTURE_KINDS:
                if not structural:
                    continue
                err(f"{p}.kind", f"must be one of {', '.join(STRUCTURE_KINDS)}")
            elif s["kind"] == "link" and s.get("attachment") not in att_ids:
                xerr(f"{p}.attachment", f"no attachment with id {s.get('attachment')!r}")

    for ns, block in (rec.get("extensions") or {}).items() if isinstance(rec.get("extensions"), dict) else []:
        if not isinstance(block, dict):
            err(f"extensions.{ns}", "must be an object")
        elif "schema" not in block:
            warn(f"extensions.{ns}", "no 'schema' naming the schema that validates this block")

    if "history" in rec:
        if not isinstance(rec["history"], list):
            err("history", "must be a list")
        else:
            for i, h in enumerate(rec["history"]):
                if not isinstance(h, dict) or not {"when", "by", "action"} <= h.keys():
                    warn(f"history[{i}]", "needs 'when', 'by' and 'action'")
    return issues


def _default_vcov(rec: dict) -> dict | None:
    v = (rec.get("structures") or {}).get("vcov")
    vs = [s for s in (v if isinstance(v, list) else [v]) if isinstance(s, dict)]
    defaults = [s for s in vs if s.get("default")]
    return defaults[0] if defaults else (vs[0] if len(vs) == 1 else None)


def _profile_inference(rec: dict) -> list[Issue]:
    issues = []
    for i, row in enumerate(rec.get("parameters") or []):
        if isinstance(row, dict) and not row.get("dropped"):
            missing = [k for k in ("estimate", "std_error", "p_value") if k not in row]
            if missing:
                issues.append(Issue("error", f"parameters[{i}]", f"missing {', '.join(missing)}"))
    metrics = rec.get("metrics") or {}
    for k in ("nobs", "df_residual"):
        if k not in metrics:
            issues.append(Issue("error", f"metrics.{k}", "required"))
    vc = _default_vcov(rec)
    if vc is None:
        issues.append(Issue("error", "structures.vcov", "a default vcov is required"))
    else:
        if "type" not in vc:
            issues.append(Issue("error", "structures.vcov.type", "required on the default vcov"))
        if not isinstance(vc.get("ssc"), dict) or "convention" not in vc["ssc"]:
            issues.append(Issue("error", "structures.vcov.ssc", "required, with at least 'convention'"))
    return issues


def _profile_linear_fe(rec: dict) -> list[Issue]:
    # Convergence is vocabulary, not required (Ted, 2026-10-24)
    issues = _profile_inference(rec)
    if "n_singletons" not in (rec.get("metrics") or {}):
        issues.append(Issue("error", "metrics.n_singletons", "required"))
    opts = (rec.get("specification") or {}).get("options") or {}
    if "drop_singletons" not in opts:
        issues.append(Issue("error", "specification.options.drop_singletons", "required"))
    fe = (rec.get("structures") or {}).get("fixed_effects")
    if fe is None:
        issues.append(Issue("error", "structures.fixed_effects", "required"))
    else:
        entries = fe.get("data") if isinstance(fe, dict) and "data" in fe else fe
        for i, e in enumerate(entries if isinstance(entries, list) else []):
            if not isinstance(e, dict) or "levels" not in e or "name" not in e:
                issues.append(Issue("error", f"structures.fixed_effects[{i}]", "needs 'name' and 'levels'"))
    return issues


def _profile_dfeols(rec: dict) -> list[Issue]:
    issues = _profile_linear_fe(rec)
    if "dfeols" not in (rec.get("extensions") or {}):
        issues.append(Issue("error", "extensions.dfeols", "required"))
    return issues


# Built-in checks of the decided required sets, used only when schema/ or jsonschema is
# missing. The schemas are the reference; these cover the main required fields.
PROFILES: dict[str, Callable[[dict], list[Issue]]] = {
    "inference/0.1": _profile_inference,
    "linear-fe/0.1": _profile_linear_fe,
    "dfeols/0.1": _profile_dfeols,
}


@dataclass(frozen=True)
class Schemas:
    """The core schema and the profiles, by profile name (each file's `title`)."""

    directory: Path
    core: dict
    profiles: dict[str, dict]

    def documents(self) -> list[dict]:
        return [self.core, *self.profiles.values()]


def find_schema_dir() -> Path | None:
    """$ESTIMATION_SCHEMA_DIR, else schema/ in the repository checkout this module is in."""
    env = os.environ.get("ESTIMATION_SCHEMA_DIR")
    for d in [Path(env)] if env else [Path(__file__).resolve().parents[2] / "schema"]:
        if (d / "estimation-0.1.json").is_file():
            return d
    return None


@functools.lru_cache(maxsize=8)
def _load_schemas(directory: str) -> Schemas:
    d = Path(directory)
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    profiles = {}
    for f in sorted((d / "profiles").glob("*.json")):
        doc = load(f)
        profiles[doc.get("title") or f.stem] = doc
    return Schemas(d, load(d / "estimation-0.1.json"), profiles)


def load_schemas(directory: str | os.PathLike | None = None) -> Schemas | None:
    """The schemas in `directory` (default `find_schema_dir()`), or None if there are none.
    Loaded once per folder; treat the result as read-only."""
    d = Path(directory) if directory is not None else find_schema_dir()
    if d is None or not (d / "estimation-0.1.json").is_file():
        return None
    return _load_schemas(str(d.resolve()))


def _have_jsonschema() -> bool:
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return False
    return True


def jsonschema_check(
    schema: dict | str | os.PathLike, documents: Iterable[dict] = ()
) -> Callable[[dict], list[Issue]]:
    """A schema plug-in for `validate` from a JSON Schema (needs the `jsonschema` package).
    `documents` are other schemas its `$ref`s may name (by `$id`, relative refs included),
    so a profile can refer to the core without a network."""
    try:
        import jsonschema
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("JSON Schema validation needs the 'jsonschema' package") from e
    root: dict = schema if isinstance(schema, dict) else json.loads(Path(schema).read_text(encoding="utf-8"))
    docs = [d for d in [root, *documents] if "$id" in d]
    cls = jsonschema.validators.validator_for(root, default=jsonschema.Draft202012Validator)
    try:
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012

        registry = Registry().with_resources(
            (d["$id"], Resource.from_contents(d, default_specification=DRAFT202012)) for d in docs
        )
        v = cls(root, registry=registry)
    except ImportError:  # jsonschema < 4.18
        store = {d["$id"]: d for d in docs}
        v = cls(root, resolver=jsonschema.RefResolver.from_schema(root, store=store))

    def check(rec: dict) -> list[Issue]:
        out = []
        for e in v.iter_errors(to_json(rec)):
            parts = list(e.absolute_path)
            if e.validator == "required":  # name the missing field, as the built-in checks do
                m = re.match(r"'(.+)' is a required property", e.message)
                parts += [m.group(1)] if m else []
            out.append(Issue("error", _path(parts), _message(e)))
        return out

    return check


def _path(parts: Iterable[Any]) -> str:
    return "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in parts).lstrip(".")


def _message(e: Any) -> str:
    """jsonschema's message, without the value it quotes when that is long; for a failed
    anyOf/oneOf, the closest branch's reason instead."""
    msg = e.message
    r = repr(e.instance)
    if len(r) > 80:
        msg = msg.replace(r, "the value")
    if e.context:
        from jsonschema.exceptions import best_match

        best = best_match(e.context)
        where = _path(list(best.relative_path))
        msg += f" (closest: {where + ': ' if where else ''}{_message(best)})"
    return msg


@functools.lru_cache(maxsize=32)
def _schema_checks(directory: str, profile: str | None) -> Callable[[dict], list[Issue]]:
    """The core check (profile None) or a profile's own rules, without the core again."""
    sc = _load_schemas(directory)
    if profile is None:
        return jsonschema_check(sc.core, sc.documents())
    doc = sc.profiles[profile]
    target = {"$ref": f"{doc['$id']}#/$defs/rules"} if "rules" in doc.get("$defs", {}) else doc
    return jsonschema_check(target, sc.documents())


def validate(
    record: dict,
    *,
    schema: Schemas | dict | str | os.PathLike | Callable[[dict], Iterable[Issue]] | None = "auto",
    profiles: dict[str, Callable[[dict], list[Issue]]] | None = None,
    base_dir: str | os.PathLike | None = None,
) -> list[Issue]:
    """Check a record (as read, or as returned by `write`) and return the issues found.

    Errors make the record invalid; warnings flag missing expected sections, size, a
    digest that doesn't match, or attachment files that don't match their hash; info
    notes what couldn't be checked.

    `schema="auto"` (default) validates against schema/ (`find_schema_dir`): the core
    plus the rules of each profile in `meta.profiles`; without the schemas or the
    `jsonschema` package it falls back to the built-in checks. A `Schemas` uses those
    files; None uses the built-in checks only; any other value (a JSON Schema as dict or
    path, or a callable returning issues) is checked in addition to the built-in ones.
    `profiles` adds or replaces profile checks by name. With `base_dir` (the record's
    folder), attachment files found there are hashed."""
    if not isinstance(record, dict):
        return [Issue("error", "", "an estimation record is a JSON object")]
    issues: list[Issue] = []
    schemas: Schemas | None = None
    extra = None
    if isinstance(schema, str) and schema == "auto":
        schemas = load_schemas()
        why = "no schema/ found" if schemas is None else "the 'jsonschema' package is not installed"
        if schemas is not None and not _have_jsonschema():
            schemas = None
        if schemas is None:
            issues.append(Issue("info", "", f"{why}; used the built-in checks"))
    elif isinstance(schema, Schemas):
        schemas = schema
    elif schema is not None:
        extra = schema
    try:
        enc = to_json(record)
    except (EstimationError, ValueError) as e:
        return [*issues, Issue("error", "", f"not encodable as JSON: {e}")]
    issues += _check_core(enc, structural=schemas is None)
    issues += _size_issues(enc, len(_dumps(enc).encode()))

    try:
        ok = check_digest(record)
    except EstimationError as e:
        ok = False
        issues.append(Issue("warning", "meta.core_digest", f"can't compute: {e}"))
    if ok is None:
        issues.append(Issue("info", "meta.core_digest", "no digest"))
    elif not ok:
        issues.append(
            Issue("warning", "meta.core_digest", "does not match: the result sections changed after writing")
        )

    key = str(schemas.directory.resolve()) if schemas is not None else ""
    if schemas is not None:
        issues += _schema_checks(key, None)(enc)
    claimed = (enc.get("meta") or {}).get("profiles") if isinstance(enc.get("meta"), dict) else None
    seen = {(i.level, i.path, i.message) for i in issues}
    for p in claimed if isinstance(claimed, list) else []:
        if profiles and p in profiles:
            found = profiles[p](enc)
        elif schemas is not None and p in schemas.profiles:
            found = _schema_checks(key, p)(enc)
        elif schemas is None and p in PROFILES:
            found = PROFILES[p](enc)
        else:
            issues.append(Issue("info", "meta.profiles", f"no checks known for profile {p!r}"))
            continue
        # a profile's rules include those it builds on: report each problem once
        for i in found:
            if (i.level, i.path, i.message) not in seen:
                seen.add((i.level, i.path, i.message))
                issues.append(Issue(i.level, i.path, f"[{p}] {i.message}"))

    if extra is not None:
        check = extra if callable(extra) else jsonschema_check(extra)
        issues += list(check(record))

    if base_dir is not None:
        for a in record.get("attachments") or []:
            if not isinstance(a, dict) or "path" not in a or "://" in str(a["path"]):
                continue
            p = attachment_path(Path(base_dir) / "_", a)
            if not p.exists():
                issues.append(Issue("info", f"attachments.{a.get('id')}", f"file not found: {a['path']}"))
            elif a.get("sha256") and _sha256_file(p) != a["sha256"]:
                issues.append(
                    Issue("warning", f"attachments.{a.get('id')}", "file does not match its sha256")
                )
    return issues


# ---------------------------------------------------------------- redactor


def identifying_paths(schema: dict, documents: Iterable[dict] = ()) -> list[str]:
    """Paths of the fields a JSON Schema marks `"x-identifying": true`, in the syntax of
    DEFAULT_IDENTIFYING. Follows properties, items, additionalProperties,
    patternProperties, allOf/anyOf/oneOf and `$ref`s: local ones, and ones to `documents`
    by `$id` (relative to the referring file's `$id`, as a profile refers to the core)."""
    found: list[str] = []
    by_id = {d["$id"]: d for d in [schema, *documents] if isinstance(d, dict) and "$id" in d}

    def resolve(ref: str, doc: dict) -> tuple[Any, dict] | None:
        uri = urljoin(doc.get("$id", ""), ref)
        doc_uri, _, frag = uri.partition("#")
        target = doc if not doc_uri or doc_uri == doc.get("$id") else by_id.get(doc_uri)
        if target is None:
            return None  # outside the documents given
        node: Any = target
        for part in [p for p in frag.split("/") if p]:
            node = node[part.replace("~1", "/").replace("~0", "~")]
        return node, target

    def walk(node: Any, path: str, seen: frozenset, doc: dict) -> None:
        if not isinstance(node, dict):
            return
        if node.get("x-identifying") is True and path:
            found.append(path)
            return
        ref = node.get("$ref")
        if isinstance(ref, str):
            key = urljoin(doc.get("$id", ""), ref)
            hit = resolve(ref, doc) if key not in seen else None
            if hit is not None:
                walk(hit[0], path, seen | {key}, hit[1])
        join = lambda k: f"{path}.{k}" if path else k
        for k, sub in (node.get("properties") or {}).items():
            walk(sub, join(k), seen, doc)
        for sub in (node.get("patternProperties") or {}).values():
            walk(sub, join("*"), seen, doc)
        if isinstance(node.get("additionalProperties"), dict):
            walk(node["additionalProperties"], join("*"), seen, doc)
        if isinstance(node.get("items"), dict):
            walk(node["items"], f"{path}[*]", seen, doc)
        for key in ("allOf", "anyOf", "oneOf"):
            for sub in node.get(key) or []:
                walk(sub, path, seen, doc)

    walk(schema, "", frozenset(), schema)
    return sorted(set(found))


def record_identifying_paths(record: dict, schemas: Schemas) -> list[str]:
    """The identifying paths for one record: the core's marks plus those of each profile
    the record claims."""
    docs = schemas.documents()
    paths = set(identifying_paths(schemas.core, docs))
    for p in (record.get("meta") or {}).get("profiles") or []:
        if p in schemas.profiles:
            paths |= set(identifying_paths(schemas.profiles[p], docs))
    return sorted(paths)


def _split(path: str) -> list[str]:
    return [p for p in path.replace("[*]", ".[*]").split(".") if p]


def _matches(node: Any, parts: list[str], prefix: str) -> Iterable[tuple[Any, Any, str]]:
    """(container, key, concrete path) of every value a pattern path reaches."""
    if not parts:
        return
    head, rest = parts[0], parts[1:]
    if head == "[*]":
        if isinstance(node, list):
            for i, v in enumerate(node):
                p = f"{prefix}[{i}]"
                yield from [(node, i, p)] if not rest else _matches(v, rest, p)
        return
    if not isinstance(node, dict):
        return
    for k in list(node) if head == "*" else [head] if head in node else []:
        p = f"{prefix}.{k}" if prefix else k
        yield from [(node, k, p)] if not rest else _matches(node[k], rest, p)


def _hmac(salt: bytes, value: Any) -> str:
    return "hmac-sha256:" + hmac.new(salt, canonical_json(to_json(value)), hashlib.sha256).hexdigest()


def redact(
    record: dict,
    *,
    paths: Iterable[str] | None = None,
    schema: Schemas | dict | str | None = "auto",
    salt: bytes | str | None = None,
    by: dict | None = None,
) -> dict:
    """A copy of `record` with its identifying fields removed or, with `salt`, replaced
    by an HMAC-SHA256 of the value, so records redacted with the same salt still compare
    equal. Appends a `history` entry listing the fields changed. The result sections
    covered by the core digest are never touched.

    The fields are `paths` if given; else those marked `x-identifying` in the core and
    each profile the record claims (`schema="auto"` finds schema/, or pass `Schemas`),
    or in one JSON Schema given as a dict; else (no schemas, or None) DEFAULT_IDENTIFYING."""
    if paths is None:
        if isinstance(schema, str) and schema == "auto":
            schema = load_schemas()
        if isinstance(schema, Schemas):
            paths = record_identifying_paths(record, schema)
        elif isinstance(schema, dict):
            paths = identifying_paths(schema)
        else:
            paths = DEFAULT_IDENTIFYING
    rec = copy.deepcopy(record)
    if isinstance(salt, str):
        salt = salt.encode()
    changed: list[str] = []
    for pattern in paths:
        parts = _split(pattern)
        if parts and parts[0] in CORE:
            raise EstimationError(f"{pattern}: redaction can't change a digest-covered section")
        # list items from the end, so removing one doesn't shift the others
        hits = sorted(_matches(rec, parts, ""), key=lambda h: (isinstance(h[1], int), h[1]), reverse=True)
        for container, key, p in hits:
            if salt is None:
                del container[key]
            else:
                container[key] = _hmac(salt, container[key])
            changed.append(p)
    entry = {
        "when": _now(),
        "by": by or _producer(),
        "action": "redacted",
        "method": "removed" if salt is None else "hmac-sha256",
        "fields": sorted(changed),
    }
    if salt is not None:
        # names the salt without revealing it, so readers know which records are comparable
        entry["salt_id"] = hashlib.sha256(b"estimation-salt-id:" + salt).hexdigest()[:16]
    rec.setdefault("history", []).append(entry)
    return rec


# ---------------------------------------------------------------- command line


def _cli_schemas(ap: argparse.ArgumentParser, directory: str | None, off: bool) -> Schemas | str | None:
    if off:
        return None
    if directory is None:
        return "auto"
    schemas = load_schemas(directory)
    if schemas is None:
        ap.error(f"{directory}: no estimation-0.1.json there")
    return schemas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="estimation", description="estimation record tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="check records; exit 1 on errors")
    v.add_argument("files", nargs="+")
    v.add_argument("--schema-dir", help="folder with estimation-0.1.json and profiles/ (default: schema/)")
    v.add_argument("--no-schema", action="store_true", help="built-in checks only")
    v.add_argument("--schema", help="another JSON Schema to check against as well")
    v.add_argument("-q", "--quiet", action="store_true", help="errors only")
    r = sub.add_parser("redact", help="remove or hash identifying fields")
    r.add_argument("file")
    r.add_argument("-o", "--output", required=True)
    r.add_argument(
        "--hash",
        action="store_true",
        help="replace values with HMAC-SHA256; the salt comes from --salt-file or $ESTIMATION_REDACT_SALT",
    )
    r.add_argument("--salt-file")
    r.add_argument("--schema-dir", help="take the fields from the core and claimed profiles in this folder")
    r.add_argument("--schema", help="take the fields from this one schema's x-identifying marks")
    r.add_argument("--path", action="append", default=[], help="also redact this path (repeatable)")
    d = sub.add_parser("digest", help="print the core digest and whether it matches")
    d.add_argument("file")
    args = ap.parse_args(argv)

    if args.cmd == "validate":
        bad = 0
        schemas = _cli_schemas(ap, args.schema_dir, args.no_schema)
        for f in args.files:
            rec = loads(Path(f).read_bytes(), nonfinite="string")
            issues = validate(rec, schema=schemas, base_dir=Path(f).parent)
            if args.schema:
                issues += jsonschema_check(args.schema)(rec)
            for i in issues:
                if not args.quiet or i.level == "error":
                    print(f"{f}: {i}")
            errors = sum(i.level == "error" for i in issues)
            bad += errors > 0
            if not args.quiet:
                print(f"{f}: {'INVALID' if errors else 'valid'} ({errors} errors)")
        return 1 if bad else 0

    if args.cmd == "redact":
        rec = loads(Path(args.file).read_bytes(), nonfinite="string")
        salt = None
        if args.hash:
            if args.salt_file:
                salt = Path(args.salt_file).read_bytes().strip()
            else:
                salt = os.environ.get("ESTIMATION_REDACT_SALT", "").encode() or None
            if not salt:
                ap.error("--hash needs a salt: --salt-file or $ESTIMATION_REDACT_SALT")
        if args.schema:
            paths = identifying_paths(json.loads(Path(args.schema).read_text(encoding="utf-8")))
        else:
            schemas = _cli_schemas(ap, args.schema_dir, False)
            paths = (
                record_identifying_paths(rec, schemas)
                if isinstance(schemas, Schemas)
                else list(DEFAULT_IDENTIFYING)
            )
        out = redact(rec, paths=paths + args.path, salt=salt)
        Path(args.output).write_text(_dumps(out, 1) + "\n", encoding="utf-8")
        print(f"{args.output}: {len(out['history'][-1]['fields'])} fields {out['history'][-1]['method']}")
        return 0

    rec = loads(Path(args.file).read_bytes(), nonfinite="string")
    ok = check_digest(rec)
    print(core_digest(rec), {None: "(no digest in record)", True: "matches", False: "DOES NOT MATCH"}[ok])
    return 1 if ok is False else 0


if __name__ == "__main__":
    sys.exit(main())
