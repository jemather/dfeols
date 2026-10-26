# SPDX-License-Identifier: GPL-3.0-or-later
"""Carrier bankruptcy periods for the G&S bankruptcy dummy (Table 2 note).

Source: Wikipedia, "List of airline bankruptcies in the United States", revision 1379726640
(2026-10-11), saved by Ted as raw/db1b/airline_bankruptcy/*.wikitext (docs/sources/db1b.md).
Only the Chapter 11 table is used: Chapter 7 is liquidation, so the carrier leaves the data.

`chapter11(path)` parses the table; `periods(path)` maps the carriers in the 1993-2006 panel to
their codes. A period with no exit date runs until the carrier ceased operating or was acquired
(`END_WITHOUT_EXIT`). The page marks itself incomplete; known gaps are listed in MISSING and are
not filled in until Ted chooses a source.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

WIKITEXT = Path("raw/db1b/airline_bankruptcy/wikipedia_airline_bankruptcies_2026-10-26.wikitext")
REVISION = 1379726640

# Wikipedia article title (link target) -> DB1B/T-100 carrier code, for carriers in the panel
CODES = {
    "America West Airlines": "HP",
    "US Airways": "US",
    "United Airlines": "UA",
    "Delta Air Lines": "DL",
    "Northwest Airlines": "NW",
    "Aloha Airlines": "AQ",
    "National Airlines (1999–2002)": "N7",
    "Sun Country Airlines": "SY",
    "Trans World Airlines": "TW",
}
# Filings with no exit date that end inside the panel: the date operations ended (page notes)
END_WITHOUT_EXIT = {
    ("N7", "2000-12-06"): "2002-11-06",  # "Ceased operations on November 6, 2002"
    ("TW", "2001-01-10"): "2001-12-01",  # acquired by American; TWA flights ended 2001-12-01 (to verify)
}
# Known in-panel bankruptcies the page lacks or leaves undated (to source; Ted decides)
MISSING = [
    "CO: Chapter 11 from December 1990 (page prose only), exit date not given",
    "TW: 1992 and 1995 filings mentioned without dates",
    "SY: exit 2003, month not given",
    "HA: Hawaiian Airlines 2003-2005 not listed",
    "TZ: ATA 2004-2006 not listed (only the 2008 filing)",
]


def _iso(cell: str) -> str | None:
    """The sortable ISO date the page hides in each date cell, e.g. 2002-08-11 or 2002-01."""
    m = re.search(r'display:none">(\d{4}(?:-\d{2}(?:-\d{2})?)?)<', cell)
    if m:
        return m.group(1)
    m = re.match(r"\s*(\d{4})\s*$", cell)
    return m.group(1) if m else None


def chapter11(path: Path = WIKITEXT) -> list[dict]:
    """Rows of the Chapter 11 table: article, filed, exited (ISO, possibly partial), notes."""
    text = Path(path).read_text()
    table = text[text.index("====Chapter 11====") :]
    table = table[table.index("{|") : table.index("|}")]
    rows = []
    for block in table.split("\n|-")[1:]:
        block = block.strip("\n")
        if not block.startswith("|"):  # the header row starts with "!"
            continue
        cells = block[1:].split("\n|")  # keep empty cells: an empty exit date is data
        link = re.search(r"\[\[([^\]|]+)", cells[0])
        rows.append(
            {
                "article": link.group(1) if link else cells[0].strip(),
                "filed": _iso(cells[1]) if len(cells) > 1 else None,
                "exited": _iso(cells[2]) if len(cells) > 2 else None,
                "notes": re.sub(r"<ref.*?(</ref>|/>)", "", cells[3]).strip() if len(cells) > 3 else "",
            }
        )
    return rows


def _day(iso: str, end: bool) -> date:
    parts = [int(p) for p in iso.split("-")]
    if len(parts) == 1:
        return date(parts[0], 12, 31) if end else date(parts[0], 1, 1)
    if len(parts) == 2:
        if end:
            nxt = date(parts[0] + (parts[1] == 12), parts[1] % 12 + 1, 1)
            return date.fromordinal(nxt.toordinal() - 1)
        return date(parts[0], parts[1], 1)
    return date(*parts)


def periods(path: Path = WIKITEXT) -> list[dict]:
    """In-panel bankruptcy periods by carrier code: code, start, end (dates), source row."""
    out = []
    for r in chapter11(path):
        code = CODES.get(r["article"])
        if code is None or r["filed"] is None:
            continue
        end = r["exited"] or END_WITHOUT_EXIT.get((code, r["filed"]))
        if end is None:
            continue
        # partial exit dates ("2003") take the latest day they allow
        out.append({"carrier": code, "start": _day(r["filed"], False), "end": _day(end, True), **r})
    return out
