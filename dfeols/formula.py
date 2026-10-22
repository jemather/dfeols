"""Formula parsing.

Syntax (a fixest-like subset):

    y1 + y2 ~ x1 + x1:x2 + i(week) + feat:i(tbin) + feat:i(tbin, ref=0) | spell + id_a^week

LHS   one or more outcomes, estimated together (shared demeaning).
RHS   terms separated by '+':
        x            continuous variable
        a:b          product of continuous variables
        i(c)         dummies for categorical c, first level dropped (or ref=...)
        x:i(c)       x times a full set of dummies for c (like Stata c.x#ibn.c);
                     add ref=... to drop one level
        0            suppress the intercept (only matters without fixed effects)
FE    absorbed fixed effects separated by '+'; a^b is the interaction of a and b.
      Every FE must have been encoded by prep().
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_I_RE = re.compile(r"^i\(\s*([A-Za-z_][\w]*)\s*(?:,\s*ref\s*=\s*([^)]+?)\s*)?\)$")
_NAME_RE = re.compile(r"^[A-Za-z_][\w]*$")


@dataclass(frozen=True)
class Term:
    conts: tuple[str, ...]  # continuous factors multiplied together
    cat: str | None = None  # categorical variable, if any
    ref: object = None  # level to drop (None = keep all for x:i(c))
    drop_first: bool = False  # i(c) with no ref: drop the first level

    @property
    def label(self) -> str:
        base = ":".join(self.conts)
        if self.cat is None:
            return base
        return f"i({self.cat}){':' + base if base else ''}"


@dataclass(frozen=True)
class Formula:
    text: str
    depvars: tuple[str, ...]
    terms: tuple[Term, ...]
    fes: tuple[str, ...]  # FE names, e.g. "spell", "id_a^week"
    intercept: bool = True

    @property
    def variables(self) -> set[str]:
        out = set(self.depvars)
        for t in self.terms:
            out.update(t.conts)
            if t.cat:
                out.add(t.cat)
        return out

    def fingerprint(self, **extra) -> str:
        s = repr((self.text, sorted(extra.items())))
        return hashlib.sha1(s.encode()).hexdigest()[:12]


def fe_components(name: str) -> list[str]:
    parts = [p.strip() for p in name.split("^")]
    for p in parts:
        if not _NAME_RE.match(p):
            raise ValueError(f"Bad fixed-effect name {name!r}")
    return parts


def fe_column(name: str) -> str:
    """Column that holds the integer codes for an FE (or cluster) spec."""
    return "__fe__" + "__X__".join(fe_components(name))


def _parse_ref(s: str):
    s = s.strip()
    if (s[0] == s[-1]) and s[0] in "'\"":
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def _split_top(s: str, sep: str) -> list[str]:
    """Split on sep, ignoring separators inside parentheses."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [p.strip() for p in out if p.strip()]


def parse(text: str) -> Formula:
    if "~" not in text:
        raise ValueError("Formula needs '~'")
    lhs, rhs = text.split("~", 1)
    depvars = tuple(_split_top(lhs, "+"))
    for d in depvars:
        if not _NAME_RE.match(d):
            raise ValueError(f"Bad outcome name {d!r}")

    fe_part = ""
    if "|" in rhs:
        rhs, fe_part = rhs.split("|", 1)
        if "|" in fe_part:
            raise ValueError("IV syntax is not supported yet")
    fes = tuple(_split_top(fe_part, "+")) if fe_part.strip() else ()
    for f in fes:
        fe_components(f)

    intercept = True
    terms: list[Term] = []
    for raw in _split_top(rhs, "+"):
        if raw in ("0", "-1"):
            intercept = False
            continue
        if raw == "1":
            continue
        conts, cat, ref, drop_first = [], None, None, False
        for piece in _split_top(raw, ":"):
            m = _I_RE.match(piece)
            if m:
                if cat is not None:
                    raise ValueError(f"Only one i() per term: {raw!r}")
                cat = m.group(1)
                ref = _parse_ref(m.group(2)) if m.group(2) else None
            elif _NAME_RE.match(piece):
                conts.append(piece)
            else:
                raise ValueError(f"Cannot parse term piece {piece!r} in {raw!r}")
        if cat is not None and not conts and ref is None:
            drop_first = True
        terms.append(Term(tuple(conts), cat, ref, drop_first))

    if not terms:
        raise ValueError("No regressors")
    return Formula(text.strip(), depvars, tuple(terms), fes, intercept)


def to_pyfixest(text: str) -> str:
    """Translate a dfeols formula to pyfixest syntax (for validation).
    x:i(c) -> i(c, x); x:i(c, ref=r) -> i(c, x, ref=r); i(c) -> C(c)."""
    f = parse(text)
    rhs = []
    for t in f.terms:
        base = ":".join(t.conts)
        if t.cat is None:
            rhs.append(base)
        elif not t.conts:
            rhs.append(f"i({t.cat}, ref={t.ref!r})" if t.ref is not None else f"C({t.cat})")
        else:
            if len(t.conts) > 1:
                raise ValueError("pyfixest i() takes one continuous variable")
            ref = f", ref={t.ref!r}" if t.ref is not None else ""
            rhs.append(f"i({t.cat}, {base}{ref})")
    if not f.intercept:
        rhs.append("0")
    out = f"{' + '.join(f.depvars)} ~ {' + '.join(rhs)}"
    return out + (f" | {' + '.join(f.fes)}" if f.fes else "")
