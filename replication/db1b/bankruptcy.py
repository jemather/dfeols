# SPDX-License-Identifier: GPL-3.0-or-later
"""Carrier bankruptcy periods for the G&S bankruptcy dummy (Table 2 note).

Source: Wikipedia, "List of airline bankruptcies in the United States", revision 1379726640
(2026-10-11), saved by Ted as raw/db1b/airline_bankruptcy/*.wikitext (docs/sources/db1b.md).
Only the Chapter 11 table is used: Chapter 7 is liquidation, so the carrier leaves the data.

`chapter11(path)` parses the table; `periods(path)` maps the carriers in the 1993-2006 panel to
their codes. A period with no exit date runs until the carrier ceased operating or was acquired
(`END_WITHOUT_EXIT`). The page marks itself incomplete: SUPPLEMENT adds the in-panel cases it lacks,
from sources Ted supplied (2026-10-27); MISSING lists what is still open.
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
    # AMR bought TWA's assets in April 2001; TWA Airlines LLC (not in bankruptcy) flew the TW
    # code until 2001-12-01 (Ted, 2026-10-27)
    ("TW", "2001-01-10"): "2001-04",
}
# In-panel Chapter 11 periods the page lacks, from sources Ted supplied (docs/sources/db1b.md)
SUPPLEMENT = [
    # filed December 1990 (page prose); emerged Wednesday 1993-04-28 after 29 months (UPI, 1993-04-27)
    {"carrier": "CO", "filed": "1990-12", "exited": "1993-04-28", "source": "UPI 1993-04-27"},
    # second filing 1995-06-30, emerged August 1995 (NYT 1995-06-29; Wikipedia, Trans World Airlines)
    {"carrier": "TW", "filed": "1995-06-30", "exited": "1995-08", "source": "NYT 1995-06-29"},
    # filed March 2003, emerged 2005-06-02 (SEC filing, Hawaiian Holdings)
    {"carrier": "HA", "filed": "2003-03", "exited": "2005-06-02", "source": "SEC EDGAR 1172222"},
    # filed 2004-10-26, emerged 2006-02-28 (Wikipedia, ATA Airlines)
    {"carrier": "TZ", "filed": "2004-10-26", "exited": "2006-02-28", "source": "Wikipedia, ATA Airlines"},
]
# Still open (Ted)
MISSING = [
    "TW: first filing 1992-01-31; exit date not yet sourced (a deal was reached 1992-08-24)",
    "SY: the 2002 involuntary filing may concern the pre-shutdown company (ceased 2001-12-07)",
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
    for r in SUPPLEMENT:
        out.append({**r, "start": _day(r["filed"], False), "end": _day(r["exited"], True)})
    return out
