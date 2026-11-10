# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Expressions for created variables in sessions: `log(x)`, `a * b`, `x > 0`, `ifelse(x > 0,
y, 0)`, on raw columns, compiled to a Polars expression.

The text is parsed with Python's `ast` (which runs nothing) and only an allowlist of node
types is compiled; anything else is an error. R's spelling is accepted where it differs:
`^` is a power (binding tightest, as in R), `&`, `|`, `!` are and, or, not (binding looser
than comparisons, as in R). The result is Float64; a comparison is 1.0 or 0.0. A null or a
non-finite result (log of 0, a division by 0) counts as missing.

A quoted string is allowed as one side of `==` or `!=` against a column, `region == "west"`,
for non-numeric columns (session i() terms on them). Its text may not hold its own quote
character, a backslash or control characters.
"""

from __future__ import annotations

import ast
import hashlib
import re

import polars as pl

_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9_.+\-*/^()<>=!&|,\s]*$")
FUNCTIONS = {
    "log": lambda e: e.log(),
    "log2": lambda e: e.log(2),
    "log10": lambda e: e.log10(),
    "log1p": lambda e: e.log1p(),
    "exp": lambda e: e.exp(),
    "sqrt": lambda e: e.sqrt(),
    "abs": lambda e: e.abs(),
}
_ARITH = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
_CMP = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


class ExpressionError(ValueError):
    pass


_STRING = re.compile(r"\"([^\"\\\x00-\x1f]*)\"|'([^'\\\x00-\x1f]*)'")
_LIT = "__dfeols_lit{}__"


def _python(text: str) -> str:
    """R spelling to Python's, so precedence matches R's. String literals are set aside
    first, so their text is neither checked against the allowed characters nor rewritten."""
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("an expression is needed, e.g. 'log(x)' or 'a * b'")
    lits: list[str] = []

    def keep(m):
        lits.append(m.group(1) if m.group(1) is not None else m.group(2))
        return _LIT.format(len(lits) - 1)

    s = _STRING.sub(keep, text)
    if "__dfeols_lit" in text:
        raise ExpressionError(f"{text!r}: names starting with __dfeols_lit are reserved")
    if not _ALLOWED_CHARS.match(s):
        bad = sorted(set(re.sub(r"[A-Za-z0-9_.+\-*/^()<>=!&|,\s]", "", s)))
        raise ExpressionError(f"{text!r}: characters not allowed in an expression: {' '.join(bad)}")
    s = s.replace("^", "**")
    s = re.sub(r"&&?", " and ", s)
    s = re.sub(r"\|\|?", " or ", s)
    s = re.sub(r"!(?!=)", " not ", s)
    for i, v in enumerate(lits):
        s = s.replace(_LIT.format(i), repr(v))
    return s


def _string_compare(n: ast.AST) -> bool:
    return isinstance(n, ast.Compare) and any(
        isinstance(v, ast.Constant) and isinstance(v.value, str) for v in [n.left, *n.comparators]
    )


def string_columns(tree: ast.AST) -> set[str]:
    """Columns compared with a string (read as text, not as numbers)."""
    return {
        v.id
        for n in ast.walk(tree)
        if _string_compare(n)
        for v in [n.left, *n.comparators]
        if isinstance(v, ast.Name)
    }


def parse(text: str) -> ast.expr:
    try:
        tree = ast.parse(_python(text).strip(), mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"{text!r} is not an expression: {e.msg}") from None
    _check(tree.body, text)
    return tree.body


def _check(node: ast.AST, text: str) -> None:
    ok = (
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Call,
        ast.Name,
        ast.Constant,
        ast.Load,
        ast.And,
        ast.Or,
        ast.Not,
        ast.USub,
        ast.UAdd,
        *_ARITH,
        *_CMP,
    )
    in_string_compare = set()
    for n in ast.walk(node):
        if _string_compare(n):
            sides = [n.left, *n.comparators]
            if (
                len(n.ops) != 1
                or not isinstance(n.ops[0], ast.Eq | ast.NotEq)
                or not any(isinstance(v, ast.Name) for v in sides)
            ):
                raise ExpressionError(
                    f"{text!r}: a string can only be compared with == or != to a column, e.g. "
                    'region == "west"'
                )
            in_string_compare.update(id(v) for v in sides)
    for n in ast.walk(node):
        if not isinstance(n, ok):
            raise ExpressionError(f"{text!r}: {type(n).__name__} is not allowed in an expression")
        number = (
            isinstance(n.value, int | float) and not isinstance(n.value, bool)
            if isinstance(n, ast.Constant)
            else True
        )
        if not number and not (isinstance(n.value, str) and id(n) in in_string_compare):
            raise ExpressionError(
                f"{text!r}: only numbers are allowed as constants (a string only in == or != "
                "against a column)"
            )
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in (*FUNCTIONS, "ifelse"):
                name = n.func.id if isinstance(n.func, ast.Name) else "?"
                raise ExpressionError(
                    f"{text!r}: unknown function {name}(); allowed: {', '.join([*FUNCTIONS, 'ifelse'])}"
                )
            if n.keywords:
                raise ExpressionError(f"{text!r}: functions take positional arguments only")
            want = 3 if n.func.id == "ifelse" else 1
            if len(n.args) != want:
                raise ExpressionError(f"{text!r}: {n.func.id}() takes {want} argument{'s' * (want > 1)}")


def columns(text: str) -> list[str]:
    """The raw columns an expression reads, sorted."""
    tree = parse(text)
    return sorted(
        {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        - {
            n.func.id  # function names are Names too
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
    )


def canonical(text: str) -> str:
    """The expression's normal form: the same for `log(x)*y` and `log( x ) * y`."""
    return ast.unparse(parse(text))


def digest(text: str) -> str:
    """Names the created column: the same for `a * b` and `b*a` (swapping the two operands
    of one + or * gives the same float64 result, bit for bit), different otherwise."""
    return hashlib.sha256(_ordered(parse(text)).encode()).hexdigest()[:16]


def _ordered(n: ast.AST) -> str:
    if isinstance(n, ast.BinOp):
        a, b = _ordered(n.left), _ordered(n.right)
        if isinstance(n.op, ast.Add | ast.Mult):
            a, b = sorted((a, b))
        return f"BinOp({type(n.op).__name__}, {a}, {b})"
    if isinstance(n, ast.AST):
        fields = ", ".join(f"{k}={_ordered(v)}" for k, v in ast.iter_fields(n))
        return f"{type(n).__name__}({fields})"
    if isinstance(n, list):
        return "[" + ", ".join(_ordered(v) for v in n) + "]"
    return repr(n)


def compile(text: str) -> pl.Expr:
    """The Polars expression, Float64. Null wherever a column it reads is null or not
    finite, so `x > 0 & y > 0` or `ifelse(c > 0, a, b)` cannot hide a missing input."""
    tree = parse(text)
    texts = string_columns(tree)
    e, _ = _compile(tree, text)
    e = e.cast(pl.Float64)
    cols = columns(text)
    if not cols:
        return e

    def missing(c):  # a column compared with a string is text: only null is missing
        return (
            pl.col(c).is_null()
            if c in texts
            else pl.col(c).is_null() | ~pl.col(c).cast(pl.Float64).is_finite()
        )

    bad = pl.any_horizontal([missing(c) for c in cols])
    return pl.when(bad).then(None).otherwise(e)


def _num(e, kind):
    return e.cast(pl.Float64) if kind == "bool" else e


def _bool(e, kind, text):
    if kind != "bool":
        raise ExpressionError(f"{text!r}: and, or, not (&, |, !) need comparisons, e.g. x > 0 & y > 0")
    return e


def _compile(n: ast.AST, text: str) -> tuple[pl.Expr, str]:
    if isinstance(n, ast.Constant):
        return pl.lit(float(n.value), dtype=pl.Float64), "num"
    if isinstance(n, ast.Name):
        return pl.col(n.id).cast(pl.Float64), "num"
    if isinstance(n, ast.BinOp):
        a = _num(*_compile(n.left, text))
        b = _num(*_compile(n.right, text))
        op = _ARITH[type(n.op)]
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b, "**": a.pow(b)}[op], "num"
    if isinstance(n, ast.UnaryOp):
        e, kind = _compile(n.operand, text)
        if isinstance(n.op, ast.Not):
            return ~_bool(e, kind, text), "bool"
        return (-_num(e, kind) if isinstance(n.op, ast.USub) else _num(e, kind)), "num"
    if isinstance(n, ast.BoolOp):
        parts = [_bool(*_compile(v, text), text) for v in n.values]
        out = parts[0]
        for p in parts[1:]:
            out = (out & p) if isinstance(n.op, ast.And) else (out | p)
        return out, "bool"
    if _string_compare(n):  # column == "text" (checked: one == or !=, a column on one side)
        a, b = (
            pl.col(v.id).cast(pl.String) if isinstance(v, ast.Name) else pl.lit(str(v.value))
            for v in (n.left, n.comparators[0])
        )
        return (a == b) if isinstance(n.ops[0], ast.Eq) else (a != b), "bool"
    if isinstance(n, ast.Compare):
        terms = [_num(*_compile(v, text)) for v in [n.left, *n.comparators]]
        out = None
        for op, a, b in zip(n.ops, terms, terms[1:]):
            c = _CMP[type(op)](a, b)
            out = c if out is None else (out & c)
        return out, "bool"
    if isinstance(n, ast.Call):
        if n.func.id == "ifelse":
            c = _bool(*_compile(n.args[0], text), text)
            a = _num(*_compile(n.args[1], text))
            b = _num(*_compile(n.args[2], text))
            return pl.when(c).then(a).otherwise(b), "num"
        return FUNCTIONS[n.func.id](_num(*_compile(n.args[0], text))), "num"
    raise ExpressionError(f"{text!r}: {type(n).__name__} is not allowed")  # _check stops these first
