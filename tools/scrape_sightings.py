"""Pull the sightings register off the Forest Alerts admin site.

The dashboard eats a CSV export. Where no export button exists, the same
rows can be read off the admin listing:

    https://mpforest.forestalerts.com/admin/sightings
        ?beat=&damage_type=&division=&elephant_visible=
        &end_date=<today>&page=<n>&range=&search=&start_date=2025-10-01

Run it:

    export FORESTALERTS_EMAIL=... FORESTALERTS_PASSWORD=...
    python -m tools.scrape_sightings -o data/sightings_scraped.csv

The window is deliberately not a parameter of the job: ``start_date`` is
pinned to the first day of the register and ``end_date`` defaults to
today, so a re-run is a full refresh rather than an increment to be
merged. The register is small enough that this is cheaper than being
clever, and a full pull cannot develop a gap.

Two things here are defensive rather than decorative.

**Headers are read, not assumed.** The listing's column order and
spelling are the site's to change, so every table header is normalised
and looked up in :data:`COLUMN_SYNONYMS` to find the dashboard's name for
it. A column nobody anticipated is carried through under its own name
rather than dropped, because this is meant to be a raw capture that also
happens to load.

**Pagination stops on evidence, not arithmetic.** Asking for a page past
the end returns either nothing or a repeat of an earlier page depending
on the framework, so the loop stops when a page yields no row it has not
already seen. Rows are fingerprinted and de-duplicated globally for the
same reason.

The output is written with :mod:`csv` and validated by feeding it back
through ``core.data_loader``, which is the only test that matters: the
scrape is correct when the app accepts it.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urljoin, urlparse

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://mpforest.forestalerts.com"
SIGHTINGS_PATH = "/admin/sightings"
# Tried in order when --login-path is not given. Guessing at a path is
# cheap; failing with "no form at /login" when the form is at
# /admin/login is a dead end the user has to diagnose themselves.
LOGIN_PATHS = ("/login", "/admin/login", "/auth/login", "/")
LOGIN_PATH = LOGIN_PATHS[0]

# The register opens on 1 October 2025; there is nothing before it to
# fetch, so the start of the window is a constant and not an argument.
START_DATE = "2025-10-01"

# Sent empty, exactly as the listing sends them unfiltered. They are
# named rather than omitted because a filter the site defaults to
# non-empty would otherwise silently narrow the pull.
FIXED_FILTERS: Dict[str, str] = {
    "beat": "",
    "damage_type": "",
    "division": "",
    "elephant_visible": "",
    "range": "",
    "search": "",
}

USER_AGENT = "ED-sightings-scraper/1.0 (Elephant Conflict Intelligence; +contact your MPFD admin)"

# Retry only what a retry can fix: a dropped connection or a 5xx. A 403
# is an answer, and repeating it is just noise on someone's server.
RETRY_ATTEMPTS = 4
RETRY_BACKOFF = [2.0, 4.0, 8.0]
RETRY_STATUS = {429, 500, 502, 503, 504}

DATE_PARAM_FORMAT = "%Y-%m-%d"

# The dashboard's names, in the order the CSV should carry them.
# ``core.data_loader`` requires Date, Latitude, Longitude, Division,
# Range and Beat; the rest unlock features when present.
CANONICAL_ORDER = [
    "ID",
    "Date",
    "Time",
    "Latitude",
    "Longitude",
    "Division",
    "Range",
    "Beat",
    "Sub Beat",
    "Total Count",
    "Male Count",
    "Female Count",
    "Calf Count",
    "Unknown Count",
    "Crop Damage",
    "Grain Damage",
    "House Damage",
    "Injury",
    "Death",
    "Male Death Count",
    "Female Death Count",
    "Children Death Count",
    "Male Injury Count",
    "Female Injury Count",
    "Children Injury Count",
]

# Normalised header spellings -> the dashboard's column name. Normalising
# drops case, spaces and punctuation, so "Total Count", "total_count" and
# "TOTAL COUNT." are one key. Hindi headers are listed where the field is
# one the app reads; anything unlisted survives under its own name.
COLUMN_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "ID": ("id", "sno", "srno", "serialno", "slno", "sightingid", "reportid", "reportno"),
    "Date": (
        "date", "sightingdate", "dateofsighting", "reporteddate", "reportdate",
        "eventdate", "incidentdate", "createdat", "createdon", "datetime",
        "dateandtime", "sightingdatetime", "दिनांक",
    ),
    "Time": ("time", "sightingtime", "reportedtime", "timeofsighting", "incidenttime", "समय"),
    "Latitude": ("latitude", "lat"),
    "Longitude": ("longitude", "long", "lng", "lon"),
    "Division": ("division", "divisionname", "div", "forestdivision", "वनमंडल"),
    "Range": ("range", "rangename", "forestrange", "परिक्षेत्र"),
    "Beat": ("beat", "beatname", "forestbeat", "बीट"),
    "Sub Beat": ("subbeat", "subbeatname", "sub"),
    "Total Count": (
        "totalcount", "total", "totalelephants", "elephantcount", "noofelephants",
        "numberofelephants", "totalnoofelephants", "herdsize", "count",
    ),
    "Male Count": ("male", "malecount", "males", "tusker", "tuskers", "maleelephants"),
    "Female Count": ("female", "femalecount", "females", "femaleelephants"),
    "Calf Count": ("calf", "calfcount", "calves", "calves-count", "children", "childrencount", "juvenile"),
    "Unknown Count": ("unknown", "unknowncount", "unsexed", "unsexedcount", "notidentified"),
    "Crop Damage": ("cropdamage", "crop", "cropdamaged", "cropdamagearea", "fasalnuksan"),
    "Grain Damage": ("graindamage", "grain", "graindamaged", "storedgraindamage"),
    "House Damage": ("housedamage", "house", "housedamaged", "propertydamage", "hutdamage"),
    "Injury": ("injury", "injured", "humaninjury", "personinjured", "injuries"),
    "Death": ("death", "deaths", "humandeath", "humandeaths", "killed", "fatalities"),
    "Male Death Count": ("maledeathcount", "maledeath", "maledeaths", "mendied"),
    "Female Death Count": ("femaledeathcount", "femaledeath", "femaledeaths", "womendied"),
    "Children Death Count": ("childrendeathcount", "childdeathcount", "childdeath", "childrendeath"),
    "Male Injury Count": ("maleinjurycount", "maleinjury", "maleinjured"),
    "Female Injury Count": ("femaleinjurycount", "femaleinjury", "femaleinjured"),
    "Children Injury Count": ("childreninjurycount", "childinjurycount", "childinjury", "childreninjured"),
}

# Without these the dashboard cannot load the file at all, so they decide
# which of several tables on a page is the listing.
REQUIRED_COLUMNS = ("Date", "Latitude", "Longitude", "Division", "Range", "Beat")

# Headers whose cell may hold "23.1234, 81.5678" as one value.
_COORDINATE_HEADERS = ("location", "coordinates", "coordinate", "latlong", "latlng", "geo", "position")
_COORDINATE_PAIR = re.compile(
    r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*[,;/|]\s*(-?\d{1,3}(?:\.\d+)?)\s*$"
)
_TRAILING_TIME = re.compile(
    r"[\s,T]+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap]\.?[Mm]\.?)?)\s*$"
)
_CLOCK = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([APap])\.?[Mm]\.?\s*$|^(\d{1,2}):(\d{2})(?::(\d{2}))?$")
_PAGE_LINK = re.compile(r"[?&]page=(\d+)")


# Where single-page frameworks mount. A page that is one of these and
# little else has no form in its HTML to find.
_MOUNT_IDS = ("app", "root", "__next", "__nuxt", "main", "main-app")


class ScrapeError(RuntimeError):
    """The site did not give us what we asked for, and said why."""


def _slug(url: str) -> str:
    """A filename for a dumped page, from its path."""
    return re.sub(r"[^a-z0-9]+", "-", urlparse(url).path.lower()).strip("-") or "root"


def _normalise(text: str) -> str:
    """Fold a header down to letters and digits for synonym lookup."""
    return re.sub(r"[^0-9a-zऀ-ॿ]+", "", (text or "").strip().lower())


_SYNONYM_INDEX: Dict[str, str] = {
    _normalise(spelling): canonical
    for canonical, spellings in COLUMN_SYNONYMS.items()
    for spelling in spellings
}
# A header that already carries the dashboard's own name maps to itself.
_SYNONYM_INDEX.update({_normalise(name): name for name in CANONICAL_ORDER})


def canonical_header(raw: str) -> Optional[str]:
    """The dashboard's name for a scraped column header, if it has one."""
    return _SYNONYM_INDEX.get(_normalise(raw))


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------


@dataclass
class Table:
    """A table lifted out of a page: rows of cell text, header flagged."""

    rows: List[List[str]] = field(default_factory=list)
    header_index: Optional[int] = None

    @property
    def header(self) -> List[str]:
        if self.header_index is None:
            return self.rows[0] if self.rows else []
        return self.rows[self.header_index]

    @property
    def data_rows(self) -> List[List[str]]:
        start = 0 if self.header_index is None else self.header_index + 1
        return [row for row in self.rows[start:] if any(cell.strip() for cell in row)]


class _TableExtractor(HTMLParser):
    """Collect every ``<table>`` in a document as rows of cell text."""

    _CELLS = ("td", "th")
    _SKIP = ("script", "style")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: List[Table] = []
        self._open: List[Table] = []
        self._row: Optional[List[str]] = None
        self._row_has_header_cell = False
        self._cell: Optional[List[str]] = None
        self._colspan = 1
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        if tag in self._SKIP:
            self._skipping += 1
            return
        if tag == "table":
            self._open.append(Table())
        elif tag == "tr" and self._open:
            self._row = []
            self._row_has_header_cell = False
        elif tag in self._CELLS and self._open:
            if self._row is None:  # a cell outside any <tr>
                self._row = []
                self._row_has_header_cell = False
            self._cell = []
            self._row_has_header_cell = self._row_has_header_cell or tag == "th"
            self._colspan = _positive_int(dict(attrs).get("colspan"), default=1)
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_startendtag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        if tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skipping = max(0, self._skipping - 1)
            return
        if tag in self._CELLS and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            assert self._row is not None
            self._row.append(text)
            # A spanned cell would otherwise shift every column after it.
            self._row.extend("" for _ in range(self._colspan - 1))
            self._cell = None
            self._colspan = 1
        elif tag == "tr" and self._open and self._row is not None:
            table = self._open[-1]
            if self._row_has_header_cell and table.header_index is None:
                table.header_index = len(table.rows)
            table.rows.append(self._row)
            self._row = None
            self._row_has_header_cell = False
        elif tag == "table" and self._open:
            self.tables.append(self._open.pop())
            self._row = None
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None and not self._skipping:
            self._cell.append(data)

    def close(self) -> None:  # unclosed <table> still yields its rows
        super().close()
        while self._open:
            self.tables.append(self._open.pop())


def _positive_int(value: Optional[str], default: int = 1) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if 1 <= parsed <= 50 else default


def extract_tables(html: str) -> List[Table]:
    """Every table in the document, outermost last."""
    parser = _TableExtractor()
    parser.feed(html)
    parser.close()
    return parser.tables


def choose_table(tables: Sequence[Table]) -> Optional[Table]:
    """The one table that looks like the sightings listing.

    Scored on how many of the dashboard's required columns its header
    maps to, then on row count, so a layout or filter-summary table
    cannot win over the listing just by being first.
    """
    best: Optional[Table] = None
    best_score: Tuple[int, int] = (-1, -1)
    for table in tables:
        rows = table.data_rows
        if not rows:
            continue
        mapped = {canonical_header(cell) for cell in table.header}
        score = (sum(1 for col in REQUIRED_COLUMNS if col in mapped), len(rows))
        if score > best_score:
            best, best_score = table, score
    return best


@dataclass
class Form:
    action: str = ""
    method: str = "post"
    fields: Dict[str, str] = field(default_factory=dict)
    # Field names by input type, so the login form can be filled without
    # knowing what the site calls its username box.
    types: Dict[str, str] = field(default_factory=dict)
    # Inputs the page rendered with no name attribute, as (type, id). A
    # script cannot post them -- the server never sees a nameless field
    # -- but they are the whole difference between "this page has no
    # password box" and "this page has one I cannot submit", which are
    # different problems with different answers.
    unnamed: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def password_field(self) -> Optional[str]:
        for name, kind in self.types.items():
            if kind == "password":
                return name
        return None

    @property
    def has_password_input(self) -> bool:
        """A password box is here, submittable or not."""
        return self.password_field is not None or any(
            kind == "password" for kind, _ in self.unnamed
        )


class _FormExtractor(HTMLParser):
    """Collect forms with their inputs, so a login can be replayed."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: List[Form] = []
        # Inputs that belong to no <form> at all. A page whose form is
        # assembled by JavaScript still renders its boxes, and knowing
        # they are there is the difference between diagnosing the site
        # and guessing at it.
        self.loose = Form()
        self._current: Optional[Form] = None

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        values = {key.lower(): (value or "") for key, value in attrs}
        if tag == "form":
            if self._current is not None:  # a form that was never closed
                self.forms.append(self._current)
            self._current = Form(
                action=values.get("action", ""),
                method=(values.get("method") or "post").lower(),
            )
        elif tag in ("input", "select", "textarea"):
            target = self._current if self._current is not None else self.loose
            name = values.get("name")
            kind = (values.get("type") or "text").lower()
            if name:
                target.fields[name] = values.get("value", "")
                target.types[name] = kind
            else:
                # Keyed by id for the report; it cannot be posted.
                target.unnamed.append((kind, values.get("id", "")))

    def handle_startendtag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._current is not None:
            self.forms.append(self._current)
            self._current = None

    def close(self) -> None:
        super().close()
        if self._current is not None:
            self.forms.append(self._current)
            self._current = None


def _parse_inputs(html: str) -> Tuple[List[Form], Form]:
    """Every form on the page, plus the inputs belonging to none of them."""
    parser = _FormExtractor()
    parser.feed(html)
    parser.close()
    return parser.forms, parser.loose


def extract_forms(html: str) -> List[Form]:
    return _parse_inputs(html)[0]


def looks_like_login(html: str) -> bool:
    """Whether a response is the login screen wearing another URL.

    A session that expires mid-scrape is served the login page with a
    200, so the tell is a password box -- any password box, named or
    not, inside a form or loose in the document. Requiring a submittable
    one would read a JavaScript login screen as a successful fetch of an
    empty register.
    """
    forms, loose = _parse_inputs(html)
    return loose.has_password_input or any(f.has_password_input for f in forms)


class _PageProbe(HTMLParser):
    """Enough of a page to say why it is not the login screen."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.scripts = 0
        self.mounts: List[str] = []
        self._text: List[str] = []
        self._in_title = False
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        values = {key.lower(): (value or "") for key, value in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "script":
            self.scripts += 1
            self._skipping += 1
        elif tag == "style":
            self._skipping += 1
        elif values.get("id", "").lower() in _MOUNT_IDS:
            self.mounts.append("#" + values["id"])

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in ("script", "style"):
            self._skipping = max(0, self._skipping - 1)

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skipping:
            self._text.append(data)

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._text)).strip()


@dataclass
class PageReport:
    """What a page actually contains, for when it is not what we wanted."""

    url: str
    status: int
    title: str
    forms: List[Form]
    loose: Form
    scripts: int
    mounts: List[str]
    text_length: int
    mentions_password: bool
    data_rows: int

    @property
    def submittable_password(self) -> bool:
        return any(f.password_field for f in self.forms)

    @property
    def unsubmittable_password(self) -> bool:
        """A password box is on the page but nothing can post it."""
        return not self.submittable_password and (
            self.loose.has_password_input
            or any(f.has_password_input for f in self.forms)
        )

    @property
    def looks_like_an_app_shell(self) -> bool:
        return bool(self.mounts) and self.scripts > 0 and self.text_length < 400

    def lines(self) -> List[str]:
        inputs = sorted(
            {name for f in self.forms for name in f.fields} | set(self.loose.fields)
        )
        nameless = [
            kind for f in (*self.forms, self.loose) for kind, _ in f.unnamed
        ]
        detail = [
            f"{self.url}  HTTP {self.status}" + (f'  "{self.title.strip()}"' if self.title.strip() else ""),
            f"  forms: {len(self.forms)}"
            f"; named inputs: {', '.join(inputs) if inputs else 'none'}"
            f"; nameless inputs: {', '.join(nameless) if nameless else 'none'}",
            f"  scripts: {self.scripts}"
            f"; mount points: {', '.join(self.mounts) if self.mounts else 'none'}"
            f"; visible text: {self.text_length} chars"
            f"; table rows: {self.data_rows}",
        ]
        detail.append("  -> " + self.verdict())
        return detail

    def verdict(self) -> str:
        if self.submittable_password:
            return "a login form this can fill and post."
        if self.unsubmittable_password:
            return ("a password box that cannot be posted -- it has no name "
                    "attribute, or sits outside any <form>. The form is wired "
                    "up in JavaScript.")
        if self.looks_like_an_app_shell:
            return ("a JavaScript application shell: the page's content is not "
                    "in its HTML.")
        if self.mentions_password:
            return ("the word 'password' appears but no password input does, "
                    "so the form is most likely built by script.")
        if self.data_rows:
            return f"a page with {self.data_rows} table row(s) -- not a login screen."
        if self.status >= 400:
            return "not served."
        return "no login form and nothing password-shaped."


def diagnose(html: str, url: str = "", status: int = 200) -> PageReport:
    """Read a page for the purpose of explaining it, not using it."""
    probe = _PageProbe()
    probe.feed(html)
    probe.close()
    forms, loose = _parse_inputs(html)
    table = choose_table(extract_tables(html))
    return PageReport(
        url=url,
        status=status,
        title=probe.title,
        forms=forms,
        loose=loose,
        scripts=probe.scripts,
        mounts=probe.mounts,
        text_length=len(probe.text),
        mentions_password="password" in html.lower(),
        data_rows=len(table.data_rows) if table else 0,
    )


COOKIE_ADVICE = (
    "The way through a form no script can fill is --cookie: sign in with a "
    "browser, copy the session cookie from its developer tools (Application "
    "-> Cookies), and pass that instead of --email/--password. Run with "
    "--probe to see this report again, or --dump-dir to keep the HTML."
)


def _has_listing(html: str) -> bool:
    """Whether the page carries something that reads as the sightings table."""
    table = choose_table(extract_tables(html))
    return bool(table and table.data_rows)


def last_page_number(html: str) -> Optional[int]:
    """The highest ``page=`` in the pagination links, when there are any."""
    pages = [int(match) for match in _PAGE_LINK.findall(html)]
    return max(pages) if pages else None


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


def table_records(table: Table) -> List[Dict[str, str]]:
    """Zip a table's data rows against its header."""
    header = [cell.strip() for cell in table.header]
    if table.header_index is None and not any(canonical_header(cell) for cell in header):
        # No <th> and a first row that maps to nothing: it is data, and
        # the columns have no names we can use.
        header = [f"Column {i + 1}" for i in range(len(header))]
        rows = table.rows
    else:
        rows = table.data_rows

    records: List[Dict[str, str]] = []
    for row in rows:
        record: Dict[str, str] = {}
        for index, value in enumerate(row):
            name = header[index] if index < len(header) else f"Column {index + 1}"
            if not name:
                name = f"Column {index + 1}"
            if name in record and record[name]:
                name = f"{name} ({index + 1})"
            record[name] = value
        records.append(record)
    return records


def normalise_time(text: str) -> str:
    """A clock reading in the 24-hour form ``core.config`` parses.

    Anything that is not a clock is handed back untouched: the loader
    would rather warn about a value it cannot read than be given one we
    invented.
    """
    match = _CLOCK.match((text or "").strip())
    if not match:
        return (text or "").strip()
    if match.group(1) is not None:
        hour, minute, second, meridiem = (
            int(match.group(1)), match.group(2), match.group(3), match.group(4).lower()
        )
        if hour > 12 or hour < 1:
            return text.strip()
        if meridiem == "p" and hour != 12:
            hour += 12
        elif meridiem == "a" and hour == 12:
            hour = 0
    else:
        hour, minute, second = int(match.group(5)), match.group(6), match.group(7)
        if hour > 23:
            return text.strip()
    if int(minute) > 59:
        return text.strip()
    return f"{hour:02d}:{minute}" + (f":{second}" if second else "")


def _split_datetime(record: Dict[str, str]) -> None:
    """Move a time hiding in the date cell into its own column.

    A listing that prints "01-10-2025 19:40" is one that would otherwise
    lose the night/day split, which is half the staffing answer.
    """
    stamp = record.get("Date", "")
    if not stamp:
        return
    match = _TRAILING_TIME.search(stamp)
    if not match:
        return
    if record.get("Time", "").strip():
        record["Date"] = stamp[: match.start()].strip().rstrip(",")
        return
    record["Date"] = stamp[: match.start()].strip().rstrip(",")
    record["Time"] = normalise_time(match.group(1))


def _split_coordinates(record: Dict[str, str], raw: Dict[str, str]) -> None:
    """Fill Latitude/Longitude from a single combined cell.

    The app cannot map a row without both, so a listing that prints one
    "Location" column is worth reading rather than failing on.
    """
    if record.get("Latitude", "").strip() and record.get("Longitude", "").strip():
        return
    named = [
        value for key, value in raw.items()
        if _normalise(key) in _COORDINATE_HEADERS
    ]
    # A column that says it holds coordinates is taken at its word; any
    # other cell has to look like a coordinate pair, and a whole-number
    # pair such as "12, 34" is far likelier to be two counts.
    others = [
        value for value in raw.values()
        if value not in named and "." in (value or "")
    ]
    for value in named + others:
        match = _COORDINATE_PAIR.match(value or "")
        if not match:
            continue
        first, second = match.group(1), match.group(2)
        # Longitude runs past 90 and latitude cannot, which settles the
        # order without trusting a column name we did not write.
        if abs(float(first)) > 90 >= abs(float(second)):
            first, second = second, first
        record["Latitude"], record["Longitude"] = first, second
        return


def map_record(raw: Dict[str, str]) -> Dict[str, str]:
    """Rename a scraped row to the dashboard's columns, losing nothing.

    A header the synonym table knows becomes the dashboard's name; one it
    does not keeps its own, so the CSV stays a raw capture that happens
    to load.
    """
    mapped: Dict[str, str] = {}
    for key, value in raw.items():
        name = canonical_header(key) or key.strip() or "Column"
        if name in mapped and mapped[name].strip():
            # Two headers claiming one column: keep the first non-empty
            # and park the other under its own name rather than lose it.
            if value.strip():
                mapped[key.strip()] = value
            continue
        mapped[name] = value
    _split_datetime(mapped)
    _split_coordinates(mapped, raw)
    if "Time" in mapped:
        mapped["Time"] = normalise_time(mapped["Time"])
    return mapped


def order_columns(records: Sequence[Dict[str, str]]) -> List[str]:
    """Dashboard columns first, then whatever else the site carried."""
    seen: List[str] = []
    for record in records:
        for key in record:
            if key not in seen:
                seen.append(key)
    known = [name for name in CANONICAL_ORDER if name in seen]
    extra = [name for name in seen if name not in CANONICAL_ORDER]
    return known + extra


def fingerprint(record: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    """Identity of a scraped row, for spotting a re-served page."""
    return tuple(sorted((key, value) for key, value in record.items()))


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class ForestAlertsScraper:
    """A logged-in session over the admin listing."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        *,
        sightings_path: str = SIGHTINGS_PATH,
        login_path: Optional[str] = None,
        timeout: float = 30.0,
        delay: float = 1.0,
        dump_dir: Optional[Path] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.sightings_url = urljoin(self.base_url + "/", sightings_path.lstrip("/"))
        paths = [login_path] if login_path else list(LOGIN_PATHS)
        self.login_urls = [
            urljoin(self.base_url + "/", path.lstrip("/")) for path in paths
        ]
        self.timeout = timeout
        self.delay = delay
        self.dump_dir = dump_dir
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._credentials: Optional[Tuple[str, str]] = None

    @property
    def login_url(self) -> str:
        """The first candidate, for messages that name one."""
        return self.login_urls[0]

    # -- plumbing ---------------------------------------------------------

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        last_error: Optional[Exception] = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = self.session.request(method, url, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as error:
                last_error = error
            else:
                if response.status_code not in RETRY_STATUS:
                    return response
                last_error = ScrapeError(
                    f"{method} {url} returned HTTP {response.status_code}"
                )
            if attempt < len(RETRY_BACKOFF):
                pause = RETRY_BACKOFF[attempt]
                logger.warning("%s %s failed (%s); retrying in %.0fs",
                               method, url, last_error, pause)
                time.sleep(pause)
        raise ScrapeError(f"{method} {url} failed after {RETRY_ATTEMPTS} attempts: {last_error}")

    def _dump(self, name: str, text: str) -> None:
        if self.dump_dir is None:
            return
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        (self.dump_dir / name).write_text(text, encoding="utf-8")

    # -- auth -------------------------------------------------------------

    def use_cookie(self, cookie: str) -> None:
        """Authenticate with a session cookie copied out of a browser.

        Some admin panels sit behind an OTP or an SSO redirect that no
        script is going to complete. Pasting the cookie is the honest
        way through that, and keeps this tool useful when the login form
        is not the whole story.
        """
        host = urlparse(self.base_url).hostname or ""
        for part in cookie.split(";"):
            if "=" not in part:
                continue
            name, _, value = part.strip().partition("=")
            self.session.cookies.set(name.strip(), value.strip(), domain=host)

    def login(self, email: str, password: str) -> None:
        """Fill and submit whatever login form the site is serving.

        The form's own hidden fields are posted back untouched, which
        covers Laravel's ``_token``, Django's ``csrfmiddlewaretoken`` and
        Rails' ``authenticity_token`` without this having to know which
        of them it is talking to.
        """
        self._credentials = (email, password)
        reports: List[PageReport] = []
        for url in self.login_urls:
            page = self._request("GET", url)
            self._dump(f"login-{_slug(url)}.html", page.text)
            report = diagnose(page.text, url=page.url, status=page.status_code)
            if report.submittable_password:
                form = next(
                    f for f in report.forms if f.password_field
                )
                return self._submit_login(form, page, email, password)
            reports.append(report)

        raise ScrapeError(
            "No login form this can fill was found. What each candidate page "
            "actually holds:\n\n"
            + "\n\n".join("\n".join(r.lines()) for r in reports)
            + "\n\n"
            + COOKIE_ADVICE
        )

    def probe(self) -> List[PageReport]:
        """Report what the candidate pages contain, without signing in.

        For the case this tool is least able to guess its way out of: a
        login that is not where or what it was expected to be.
        """
        reports = []
        for url in [*self.login_urls, self.sightings_url]:
            response = self._request("GET", url)
            self._dump(f"probe-{_slug(url)}.html", response.text)
            reports.append(
                diagnose(response.text, url=response.url, status=response.status_code)
            )
        return reports

    def _submit_login(
        self, form: Form, page: requests.Response, email: str, password: str
    ) -> None:
        """Fill the form the site served and post it back."""
        data = dict(form.fields)
        data[form.password_field] = password
        user_field = self._user_field(form)
        if user_field is None:
            raise ScrapeError(
                f"The login form at {page.url} has a password box but no "
                "username or email field this recognises."
            )
        data[user_field] = email

        headers = {"Referer": page.url}
        # Laravel hands the CSRF token out as a cookie too, and checks
        # the header on XHR-shaped posts.
        xsrf = self.session.cookies.get("XSRF-TOKEN")
        if xsrf:
            headers["X-XSRF-TOKEN"] = requests.utils.unquote(xsrf)

        action = urljoin(page.url, form.action) if form.action else page.url
        response = self._request("POST", action, data=data, headers=headers)
        self._dump("login-response.html", response.text)
        if response.status_code >= 400:
            raise ScrapeError(f"Login POST to {action} returned HTTP {response.status_code}.")
        if looks_like_login(response.text):
            raise ScrapeError(
                "Login was rejected -- the site served the login form again. "
                "Check FORESTALERTS_EMAIL and FORESTALERTS_PASSWORD."
            )
        logger.info("Logged in as %s", email)

    @staticmethod
    def _user_field(form: Form) -> Optional[str]:
        preferred = ("email", "username", "user", "login", "mobile", "phone", "userid")
        for candidate in preferred:
            for name in form.fields:
                if _normalise(name) == candidate:
                    return name
        for name, kind in form.types.items():
            if kind in ("text", "email", "tel"):
                return name
        return None

    # -- listing ----------------------------------------------------------

    def page_params(self, page: int, start_date: str, end_date: str) -> Dict[str, str]:
        params = dict(FIXED_FILTERS)
        params.update({"start_date": start_date, "end_date": end_date, "page": str(page)})
        return params

    def fetch_page(self, page: int, start_date: str, end_date: str) -> str:
        response = self._request(
            "GET", self.sightings_url, params=self.page_params(page, start_date, end_date)
        )
        self._dump(f"page-{page:04d}.html", response.text)
        if response.status_code >= 400:
            raise ScrapeError(
                f"{self.sightings_url} page {page} returned HTTP {response.status_code}."
            )
        # A password box on a page that also carries the listing is a
        # profile widget, not an expired session.
        if looks_like_login(response.text) and not _has_listing(response.text):
            if self._credentials is None:
                raise ScrapeError(
                    "The listing served the login page. Supply credentials "
                    "(--email/--password) or a fresh --cookie."
                )
            logger.warning("Session expired on page %s; logging in again", page)
            self.login(*self._credentials)
            response = self._request(
                "GET", self.sightings_url, params=self.page_params(page, start_date, end_date)
            )
            if looks_like_login(response.text) and not _has_listing(response.text):
                raise ScrapeError("Session expired and could not be re-established.")
        return response.text

    def iter_rows(
        self,
        start_date: str,
        end_date: str,
        max_pages: int = 500,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[Dict[str, str]]:
        """Walk the listing, yielding every row exactly once.

        ``progress`` is called with ``(page, rows so far)`` as each page
        lands, for a caller with a screen to keep honest.
        """
        total = 0
        seen: set = set()
        duplicates = 0
        expected_last: Optional[int] = None

        for page in range(1, max_pages + 1):
            if page > 1 and self.delay:
                time.sleep(self.delay)
            html = self.fetch_page(page, start_date, end_date)
            advertised = last_page_number(html)
            if advertised and advertised > (expected_last or 0):
                expected_last = advertised
                if page == 1:
                    logger.info("Pagination reports %s page(s)", expected_last)

            table = choose_table(extract_tables(html))
            if table is None:
                if page == 1:
                    raise ScrapeError(
                        "No data table found on the first page. The listing may "
                        "render its rows in JavaScript; re-run with --dump-dir to "
                        "capture the HTML and check."
                    )
                logger.info("Page %s has no table; stopping", page)
                break

            records = [map_record(raw) for raw in table_records(table)]
            fresh: List[Dict[str, str]] = []
            for record in records:
                mark = fingerprint(record)
                if mark in seen:
                    duplicates += 1
                    continue
                seen.add(mark)
                fresh.append(record)

            if not fresh:
                # Either the end of the listing, or the site clamped an
                # out-of-range page back to one we have already read.
                logger.info("Page %s added no new rows; stopping", page)
                if expected_last and page <= expected_last:
                    logger.warning(
                        "Stopped on page %s but the pagination advertised %s. "
                        "Check the pull against the site's own row count.",
                        page, expected_last,
                    )
                break

            logger.info("Page %s: %s row(s)", page, len(fresh))
            total += len(fresh)
            if progress is not None:
                progress(page, total)
            yield from fresh
        else:
            logger.warning(
                "Stopped at the --max-pages limit of %s; the pull may be short.",
                max_pages,
            )

        if duplicates:
            logger.info("Skipped %s duplicate row(s) across pages", duplicates)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def scrape(
    *,
    base_url: str = BASE_URL,
    start_date: str = START_DATE,
    end_date: Optional[str] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
    cookie: Optional[str] = None,
    login_path: Optional[str] = None,
    sightings_path: str = SIGHTINGS_PATH,
    max_pages: int = 500,
    delay: float = 1.0,
    timeout: float = 30.0,
    dump_dir: Optional[Path] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, str]]:
    """Log in, walk the listing, and return every row as a dict."""
    end_date = end_date or today()
    check_date(start_date, "--start-date")
    check_date(end_date, "--end-date")
    if end_date < start_date:
        raise ScrapeError(f"end_date {end_date} is before start_date {start_date}.")

    scraper = ForestAlertsScraper(
        base_url,
        sightings_path=sightings_path,
        login_path=login_path,
        timeout=timeout,
        delay=delay,
        dump_dir=dump_dir,
    )
    if cookie:
        scraper.use_cookie(cookie)
    if email and password:
        scraper.login(email, password)
    elif not cookie:
        raise ScrapeError(
            "No credentials. Set FORESTALERTS_EMAIL and FORESTALERTS_PASSWORD, "
            "or pass --cookie with a browser session cookie."
        )

    logger.info("Scraping %s from %s to %s", scraper.sightings_url, start_date, end_date)
    return list(
        scraper.iter_rows(start_date, end_date, max_pages=max_pages, progress=progress)
    )


def today() -> str:
    return date.today().strftime(DATE_PARAM_FORMAT)


def check_date(value: str, label: str) -> str:
    try:
        datetime.strptime(value, DATE_PARAM_FORMAT)
    except ValueError:
        raise ScrapeError(f"{label} must be YYYY-MM-DD, got {value!r}") from None
    return value


def records_to_csv(records: Sequence[Dict[str, str]]) -> Tuple[str, List[str]]:
    """Render the scrape as CSV text. Returns ``(text, header)``.

    Kept separate from :func:`write_csv` so the app can load a scrape
    straight into memory: a register pulled inside a session has no
    business being written to the server's disk on the way through.
    """
    if not records:
        raise ScrapeError("Nothing was scraped, so there is nothing to write.")
    columns = order_columns(records)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow({column: record.get(column, "") for column in columns})
    return buffer.getvalue(), columns


def write_csv(records: Sequence[Dict[str, str]], path: Path) -> List[str]:
    """Write the scrape, dashboard columns first. Returns the header."""
    text, columns = records_to_csv(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return columns


def verify(path: Path) -> bool:
    """Feed the file back through the app's own loader.

    This is the end-to-end check: a scrape is correct when the dashboard
    accepts it, not when it parsed.
    """
    try:
        from core.data_loader import load_and_validate_csv
        from core.exceptions import DataValidationError
    except ImportError as error:  # pandas absent, e.g. a bare scraping box
        print(f"Skipping verification: {error}")
        return True

    try:
        frame, warnings = load_and_validate_csv(str(path))
    except DataValidationError as error:
        print(f"REJECTED by core.data_loader: {error}")
        return False

    print(f"Accepted by core.data_loader: {len(frame):,} rows, "
          f"{frame['Date'].min():%Y-%m-%d} to {frame['Date'].max():%Y-%m-%d}, "
          f"{frame['Division'].nunique()} division(s), {frame['Beat'].nunique()} beat(s)")
    for warning in warnings:
        print(f"  warning: {warning}")
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Scrape the Forest Alerts sightings listing into a dashboard CSV.",
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("data/sightings_scraped.csv"),
                        help="Where to write the CSV (default: %(default)s)")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--start-date", default=START_DATE,
                        help="Fixed start of the register (default: %(default)s)")
    parser.add_argument("--end-date", default=None,
                        help="Default: today. The sample URL used the following "
                             "day, which matters only if the site's filter is exclusive.")
    parser.add_argument("--email", default=os.environ.get("FORESTALERTS_EMAIL"),
                        help="Prefer the FORESTALERTS_EMAIL environment variable.")
    parser.add_argument("--password", default=os.environ.get("FORESTALERTS_PASSWORD"),
                        help="Prefer the FORESTALERTS_PASSWORD environment variable; "
                             "a password on the command line is visible in ps and history.")
    parser.add_argument("--cookie", default=os.environ.get("FORESTALERTS_COOKIE"),
                        help="Session cookie(s) from a logged-in browser, instead of a login.")
    parser.add_argument("--login-path", default=None,
                        help="Where the login form lives. Default: try "
                             + ", ".join(LOGIN_PATHS) + " in turn.")
    parser.add_argument("--probe", action="store_true",
                        help="Report what the candidate login pages and the "
                             "listing actually contain, then exit. Needs no "
                             "credentials, and is the place to start when the "
                             "login is not where this expects it.")
    parser.add_argument("--sightings-path", default=SIGHTINGS_PATH)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between page requests (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--dump-dir", type=Path, default=None,
                        help="Save every fetched page here, for when the parse fails.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip loading the result back through core.data_loader.")
    args = parser.parse_args(argv)

    if args.probe:
        scraper = ForestAlertsScraper(
            args.base_url,
            sightings_path=args.sightings_path,
            login_path=args.login_path,
            timeout=args.timeout,
            dump_dir=args.dump_dir,
        )
        if args.cookie:
            scraper.use_cookie(args.cookie)
        try:
            reports = scraper.probe()
        except ScrapeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        for report in reports:
            for line in report.lines():
                print(line)
            print()
        print(COOKIE_ADVICE)
        return 0

    try:
        records = scrape(
            base_url=args.base_url,
            start_date=args.start_date,
            end_date=args.end_date,
            email=args.email,
            password=args.password,
            cookie=args.cookie,
            login_path=args.login_path,
            sightings_path=args.sightings_path,
            max_pages=args.max_pages,
            delay=args.delay,
            timeout=args.timeout,
            dump_dir=args.dump_dir,
        )
        columns = write_csv(records, args.out)
    except ScrapeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {len(records):,} row(s) to {args.out}")
    print(f"Columns: {', '.join(columns)}")
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        print(f"warning: the dashboard also needs {', '.join(missing)}", file=sys.stderr)

    if args.no_verify:
        return 0
    return 0 if verify(args.out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
