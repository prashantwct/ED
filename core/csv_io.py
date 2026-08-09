"""Resilient CSV reading shared by every loader in the app.

Field data arrives from spreadsheets, GIS exports and hand-edited sheets
that pass through Windows editors, so a file is routinely *almost* UTF-8:
thousands of clean ASCII rows with a handful of stray bytes in a place
name. Refusing the whole file over one byte is the wrong trade -- a real
9,230-village centroid file was rejected because a single ``0xbb`` sat in
one village name, taking 9,229 usable villages down with it.

So: try UTF-8, then the Windows/Latin-1 encodings those tools actually
emit, and report which one worked rather than silently guessing.
"""

from __future__ import annotations

import logging
from typing import BinaryIO, Callable, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Tried in order. cp1252 before latin-1 because it is what Excel on
# Windows produces, and it decodes the 0x80-0x9F range that latin-1 maps
# to unusable control characters.
#
# Note that ISO-8859-1 maps all 256 byte values, so once it is in the
# chain decoding never fails on bytes. Binary junk therefore arrives as
# nonsense *text* and is rejected further on for not having the expected
# columns -- which is the better error to show a user anyway. The final
# raise below is a safety net for a future change to this list, not a
# path a real file reaches.
FALLBACK_ENCODINGS = ["utf-8", "cp1252", "ISO-8859-1"]

Source = Union[str, "BinaryIO"]


def read_csv_resilient(
    source: Source,
    on_error: Optional[Callable[[str], Exception]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Read a CSV, falling back through known field-export encodings.

    Args:
        source: Path or file-like object (rewound between attempts when
            it supports ``seek``).
        on_error: Builds the exception raised when the file cannot be
            read at all. Defaults to ``ValueError``, so callers can map
            failures onto their own error type.

    Returns:
        ``(dataframe, warnings)``. ``warnings`` names the encoding used
        whenever it was not UTF-8, since a fallback means the file has
        characters that may not have survived intact.

    Raises:
        Whatever ``on_error`` builds, if no encoding works or the file is
        not parseable as CSV.
    """
    make_error = on_error or (lambda message: ValueError(message))
    warnings: List[str] = []
    last_decode_error: Optional[Exception] = None

    for encoding in FALLBACK_ENCODINGS:
        _rewind(source)
        try:
            frame = pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_decode_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise make_error("The file is empty.") from exc
        except pd.errors.ParserError as exc:
            raise make_error(
                "The file could not be parsed as CSV. Please confirm the export "
                "format and that it uses standard comma delimiters."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean message
            raise make_error(f"Could not read the file: {exc}") from exc

        if encoding != "utf-8":
            warnings.append(
                f"File is not valid UTF-8; read it as {encoding} instead. A few "
                "characters in names may not be exactly as intended -- worth a "
                "glance if any place name looks wrong."
            )
            logger.info("CSV decoded with fallback encoding %s.", encoding)
        return frame, warnings

    raise make_error(
        "Could not decode the file as text in UTF-8, Windows-1252 or Latin-1. "
        f"Please re-export it as UTF-8 CSV. ({last_decode_error})"
    )


def _rewind(source: Source) -> None:
    if hasattr(source, "seek"):
        source.seek(0)
