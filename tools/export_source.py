"""Downloading the site's own export, which beats reading its pages.

The sightings screen has an Export button. Whatever it hands back is the
register as the department itself publishes it: one request, no
pagination to walk, no row shape to infer, and the columns the people
who run the system chose. That makes it the best of the three sources
this tool can read, and the one to try first.

What comes back is not known in advance. A Laravel admin panel exporting
through maatwebsite/excel returns .xlsx; a hand-rolled one returns .csv;
either may arrive with a useless content type. So the format is detected
from the response rather than assumed -- from the filename the site
suggests, from the content type, and failing both from the first bytes,
because a ZIP container and a line of commas are not hard to tell apart.

Everything here converts to CSV text and stops. The rows then go through
the same column mapping as the scraped ones, so all three sources end at
the same ``core.data_loader``.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Excel and its relatives are ZIP containers; the older .xls is not.
_ZIP_MAGIC = b"PK\x03\x04"
_OLE_MAGIC = b"\xd0\xcf\x11\xe0"

_FORMATS_BY_EXTENSION = {
    ".csv": "csv", ".txt": "csv", ".tsv": "tsv",
    ".xlsx": "xlsx", ".xlsm": "xlsx", ".xls": "xls",
    ".json": "json", ".html": "html", ".htm": "html",
}

_FORMATS_BY_CONTENT_TYPE = (
    ("text/csv", "csv"),
    ("application/csv", "csv"),
    ("text/tab-separated", "tsv"),
    ("spreadsheetml.sheet", "xlsx"),
    ("application/vnd.ms-excel", "xls"),
    ("application/json", "json"),
    ("text/html", "html"),
)

_FILENAME = re.compile(r"""filename\*?=(?:UTF-8'')?["']?([^"';]+)""", re.I)

# Encodings to try, in the order the rest of the project tries them:
# cp1252 before latin-1, because it is what Excel on Windows emits.
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "ISO-8859-1")


class ExportFormatError(RuntimeError):
    """What came back is not an export this can read."""


def suggested_filename(content_disposition: str) -> Optional[str]:
    """The filename the server proposed, if it proposed one."""
    match = _FILENAME.search(content_disposition or "")
    return match.group(1).strip() if match else None


def detect_format(
    content: bytes,
    content_type: str = "",
    filename: Optional[str] = None,
) -> str:
    """What the export actually is: csv, tsv, xlsx, xls, json or html.

    The filename is trusted first because it is the server naming its own
    output, then the content type, then the bytes -- a download served as
    ``application/octet-stream`` is common enough that guessing from the
    header alone would fail on it.
    """
    if filename:
        for extension, name in _FORMATS_BY_EXTENSION.items():
            if filename.lower().endswith(extension):
                return name

    lowered = (content_type or "").lower()
    for needle, name in _FORMATS_BY_CONTENT_TYPE:
        if needle in lowered:
            return name

    if content.startswith(_ZIP_MAGIC):
        return "xlsx"
    if content.startswith(_OLE_MAGIC):
        return "xls"

    head = content[:4096].lstrip()
    if head[:1] in (b"{", b"["):
        return "json"
    if head[:1] == b"<":
        return "html"
    if b"," in head or b"\t" in head:
        return "tsv" if head.count(b"\t") > head.count(b",") else "csv"
    raise ExportFormatError(
        "The export is in a format this does not recognise "
        f"(content-type {content_type!r}, first bytes {content[:16]!r})."
    )


def decode(content: bytes) -> str:
    """Text from bytes, trying the encodings a field export arrives in."""
    for encoding in _ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def to_csv_text(content: bytes, fmt: str) -> str:
    """Any readable export, as CSV text.

    Spreadsheets go through pandas, which reads the cell types rather
    than the printed strings -- an Excel date is a serial number, and
    reading it by hand is how every date in the file ends up wrong.
    """
    if fmt in ("csv", "tsv"):
        text = decode(content)
        if fmt == "tsv":
            return _retabulate(text)
        return text
    if fmt in ("xlsx", "xls"):
        return _spreadsheet_to_csv(content, fmt)
    if fmt == "json":
        from tools.api_source import parse_json, records_from_json

        records = records_from_json(parse_json(decode(content)))
        return _records_to_csv_text(records)
    if fmt == "html":
        raise ExportFormatError(
            "The export came back as an HTML page rather than a file, which "
            "usually means the request was not authorised or the path was "
            "wrong. Re-run with --dump-dir to keep it and look."
        )
    raise ExportFormatError(f"Cannot read an export of type {fmt!r}.")


def _retabulate(text: str) -> str:
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    return _rows_to_csv_text(rows)


def _spreadsheet_to_csv(content: bytes, fmt: str) -> str:
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - pandas is a dependency
        raise ExportFormatError(f"Reading a spreadsheet needs pandas: {error}") from None

    engine = "openpyxl" if fmt == "xlsx" else None
    try:
        frame = pd.read_excel(io.BytesIO(content), engine=engine, dtype=str)
    except ImportError as error:
        raise ExportFormatError(
            f"Reading a .{fmt} export needs the {engine or 'xlrd'} package, which "
            f"is not installed here ({error}). Install it, or save the export "
            "as CSV and upload that instead."
        ) from None
    except Exception as error:  # a corrupt or unexpected workbook
        raise ExportFormatError(f"The .{fmt} export could not be read: {error}") from None

    frame = frame.fillna("")
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


def _rows_to_csv_text(rows: List[List[str]]) -> str:
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue()


def _records_to_csv_text(records: List[Dict[str, str]]) -> str:
    if not records:
        return ""
    columns: List[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def records_from_csv_text(text: str) -> List[Dict[str, str]]:
    """An export's rows, keyed by its own headers.

    The header row is taken as-is; the synonym table downstream is what
    turns the site's spelling into the dashboard's.
    """
    reader = csv.DictReader(io.StringIO(text))
    records: List[Dict[str, str]] = []
    for row in reader:
        clean = {
            (key or "").strip(): ("" if value is None else str(value).strip())
            for key, value in row.items()
            if key is not None
        }
        if any(clean.values()):
            records.append(clean)
    return records
