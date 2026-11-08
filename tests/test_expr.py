"""Expressions for sessions' created variables (dfeols.expr): R's spelling and precedence,
missing inputs stay missing, and nothing outside the allowlist compiles."""

import math

import polars as pl
import pytest

from dfeols import expr as E

DF = pl.DataFrame(
    {
        "x": [1.0, 0.0, -2.0, None, float("inf")],
        "y": [2.0, 3.0, 4.0, 5.0, 6.0],
        "k": [1, 2, 3, 4, 5],
    }
)


def _v(text):
    return DF.select(E.compile(text).alias("v"))["v"].to_list()


def test_arithmetic_and_r_precedence():
    assert _v("x^2 + 1")[:3] == [2.0, 1.0, 5.0]  # ^ binds tightest, as in R
    assert _v("-x^2")[:3] == [-1.0, -0.0, -4.0]
    assert _v("2^3^2")[0] == 512.0  # right-associative
    assert _v("x * y / k")[:2] == [2.0, 0.0]
    assert _v("k")[0] == 1.0  # integers become Float64
    assert _v("log(y)")[0] == pytest.approx(math.log(2))
    assert _v("sqrt(y) + abs(x) + exp(0) + log1p(0) + log10(y) + log2(y)")[0] == pytest.approx(
        math.sqrt(2) + 1 + 1 + 0 + math.log10(2) + 1
    )


def test_comparisons_and_logic():
    assert _v("x > 0 & y > 2")[:3] == [0.0, 0.0, 0.0]  # & looser than >, as in R
    assert _v("x > 0 | y > 3")[:3] == [1.0, 0.0, 1.0]
    assert _v("!(x > 0)")[:3] == [0.0, 1.0, 1.0]
    assert _v("(x > 0) * y")[:3] == [2.0, 0.0, 0.0]
    assert _v("0 < y < 4")[:3] == [1.0, 1.0, 0.0]
    assert _v("y != 3")[:2] == [1.0, 0.0]
    assert _v("ifelse(x > 0, y, 0)")[:3] == [2.0, 0.0, 0.0]


def test_missing_inputs_stay_missing():
    # a null or infinite input is missing whatever the expression does with it
    for t in ("x > 0 & y > 2", "ifelse(x > 0, y, 0)", "x * 0", "(x > 0) | (y > 0)"):
        assert _v(t)[3:] == [None, None], t
    assert _v("log(x)")[1] == -math.inf  # not finite: counted as missing by the session


def test_canonical_digest_columns():
    assert E.canonical("log( x )*y") == "log(x) * y"
    assert E.digest("log( x )*y") == E.digest("log(x) * y") != E.digest("log(y) * x")
    assert E.digest("y * log(x)") == E.digest("log(x)*y") and E.digest("x + k") == E.digest("k+x")
    assert E.digest("x - k") != E.digest("k - x") and E.digest("x / k") != E.digest("k / x")
    assert E.digest("x * y * k") != E.digest("x * (y * k)")  # a different rounding
    assert E.columns("ifelse(x > 0, log(y), k)") == ["k", "x", "y"]
    assert E.columns("2 ^ 3") == []


@pytest.mark.parametrize(
    "text, msg",
    [
        ("__import__('os')", "characters not allowed"),
        ("x.y", "Attribute"),
        ("x if y else 1", "IfExp"),
        ("foo(x)", "unknown function foo"),
        ("x; y", "characters not allowed"),
        ("lambda: 1", "characters not allowed"),
        ("x[0]", "characters not allowed"),
        ("x & 1", "need comparisons"),
        ("log(x, 2)", "takes 1 argument"),
        ("ifelse(x > 0, 1)", "takes 3 arguments"),
        ("", "an expression is needed"),
        ("x +", "not an expression"),
    ],
)
def test_refused(text, msg):
    with pytest.raises(E.ExpressionError, match=msg):
        E.compile(text)
