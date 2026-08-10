"""Resilient CSV reading shared by the loaders."""

from __future__ import annotations

import logging
from typing import BinaryIO, Callable, List, Optional, Tuple, Union

import pandas as pd

from core.config import FALLBACK_ENCODINGS

logger = logging.getLogger(__name__)

Source = Union[str, "BinaryIO"]


def read_csv_resilient(
    source: Source,
    on_error: Optional[Callable[[str], Exception]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Read a CSV, falling back through the encodings field exports use.

    Files are routinely almost-UTF-8: thousands of clean ASCII rows with a
    stray byte in one place name. Rejecting the whole file over one byte
    is the wrong trade.

    Args:
        source: Path or file-like object, rewound between attempts.
        on_error: Builds the exception raised on failure, so callers get
            their own error type. Defaults to ``ValueError``.

    Returns:
        ``(dataframe, warnings)``. A warning names the encoding whenever
        it was not UTF-8.
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
                "characters in names may not be exactly as intended."
            )
            logger.info("CSV decoded with fallback encoding %s.", encoding)
        return frame, warnings

    # ISO-8859-1 maps all 256 byte values, so this is unreachable while it
    # is in FALLBACK_ENCODINGS. Kept as a guard against that list changing.
    raise make_error(
        "Could not decode the file as text in UTF-8, Windows-1252 or Latin-1. "
        f"Please re-export it as UTF-8 CSV. ({last_decode_error})"
    )


def _rewind(source: Source) -> None:
    if hasattr(source, "seek"):
        source.seek(0)
