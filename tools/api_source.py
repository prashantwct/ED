"""Reading the register out of a JSON API instead of an HTML table.

The Forest Alerts site is a single-page application: ``/`` and ``/login``
serve a four-script shell around ``<div id="app">`` with no text and no
form in the HTML. Nothing renders server-side, so there is no table to
parse -- the rows arrive as JSON over an API the page calls after it
boots, and that API is the only thing there is to scrape.

This holds the parts of that which are pure: turning a payload of some
unknown shape into rows, and reading an application's own JavaScript to
find out where its API lives. Both are testable without a network, which
matters more than usual here -- the site cannot be reached from CI, and
the shape of its payload is still unknown.

The guiding assumption is that the payload's *shape* is unknown but its
*vocabulary* is not: whatever wrapper the rows arrive in, they are
objects whose keys are the register's columns, and
``tools.scrape_sightings`` already knows how to map those onto the
dashboard's names.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Wrappers a list of rows is commonly found under, in the order worth
# trying. Laravel nests twice ({"data": {"data": [...]}}), so each is
# followed down rather than merely read.
ROW_KEYS = (
    "data", "results", "rows", "items", "records", "list", "content",
    "sightings", "entries", "features",
)

# Where a page count hides, in the order of how specific it is.
PAGE_COUNT_KEYS = ("last_page", "lastPage", "total_pages", "totalPages", "pages", "page_count")
NEXT_KEYS = ("next_page_url", "nextPageUrl", "next", "next_url")

# Keys whose value stands in for a nested object: {"division": {"id": 3,
# "name": "Shahdol"}} is a Division of "Shahdol", not of "3".
LABEL_KEYS = ("name", "title", "label", "value", "text", "english_name", "name_en")

_MAX_DEPTH = 4

# A quoted path that looks like an API route, as it appears in a bundle.
_API_PATH = re.compile(
    r"""['"`](/?(?:api|admin|v[0-9]{1,2})/[A-Za-z0-9/_\-.]{1,120})['"`]"""
)
# Ranked to the top: a route that names what we are after.
_INTERESTING = ("sight", "report", "alert", "incident", "conflict", "elephant")


class ApiShapeError(ValueError):
    """The payload held nothing that reads as a list of rows."""


def parse_json(text: str) -> Any:
    """Parse a response body, or say plainly that it was not JSON."""
    try:
        return json.loads(text)
    except ValueError as error:
        raise ApiShapeError(f"The response is not JSON: {error}") from None


def looks_like_json(text: str) -> bool:
    stripped = (text or "").lstrip()
    return stripped[:1] in ("{", "[")


def find_rows(payload: Any, _depth: int = 0) -> Optional[List[Dict[str, Any]]]:
    """The list of row objects inside a payload of unknown shape.

    Named wrappers are followed first and in order, because a payload can
    hold more than one list and the one called ``data`` is the one meant.
    Only if none matches is the rest searched, shallowest first.
    """
    if _depth > _MAX_DEPTH:
        return None
    if _is_row_list(payload):
        return list(payload)
    if not isinstance(payload, dict):
        return None

    for key in ROW_KEYS:
        if key in payload:
            found = find_rows(payload[key], _depth + 1)
            if found is not None:
                return found

    for value in payload.values():
        found = find_rows(value, _depth + 1)
        if found is not None:
            return found
    return None


def _is_row_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) for item in value)
    )


def flatten(row: Dict[str, Any], _prefix: str = "", _depth: int = 0) -> Dict[str, str]:
    """One API object as flat ``{column: text}``, losing nothing readable.

    A nested object contributes both its dotted keys and, where it
    carries a human label, that label under the parent's own name -- so
    ``{"division": {"id": 3, "name": "Shahdol"}}`` yields a ``division``
    of "Shahdol" for the synonym table to find, and ``division.id``
    beside it for anyone who wants the key.
    """
    flat: Dict[str, str] = {}
    for key, value in row.items():
        name = f"{_prefix}{key}"
        if isinstance(value, dict):
            if _depth < _MAX_DEPTH:
                label = _label_of(value)
                if label is not None:
                    flat[name] = label
                flat.update(flatten(value, f"{name}.", _depth + 1))
            continue
        if isinstance(value, list):
            scalars = [_scalar(item) for item in value if not isinstance(item, (dict, list))]
            if scalars:
                flat[name] = "; ".join(scalars)
            continue
        flat[name] = _scalar(value)
    return flat


def _label_of(value: Dict[str, Any]) -> Optional[str]:
    for key in LABEL_KEYS:
        if key in value and not isinstance(value[key], (dict, list)):
            return _scalar(value[key])
    return None


def _scalar(value: Any) -> str:
    """A cell's text. Booleans become 1/0 so damage flags stay numeric."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def records_from_json(payload: Any) -> List[Dict[str, str]]:
    """Every row in a payload, flattened to text."""
    rows = find_rows(payload)
    if rows is None:
        raise ApiShapeError(
            "No list of records was found in the response. Pass --dump-dir "
            "to keep the payload and check what it holds."
        )
    return [flatten(row) for row in rows]


def last_page_from_json(payload: Any) -> Optional[int]:
    """The page count an API reports, when it reports one."""
    for value in _values_by_key(payload, PAGE_COUNT_KEYS):
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return None


def has_next_page(payload: Any) -> Optional[bool]:
    """Whether the API says another page exists; None when it is silent."""
    for value in _values_by_key(payload, NEXT_KEYS):
        if value in (None, "", False):
            return False
        return True
    return None


def _values_by_key(payload: Any, keys: Sequence[str], _depth: int = 0) -> Iterable[Any]:
    if _depth > _MAX_DEPTH or not isinstance(payload, dict):
        return
    for key in keys:
        if key in payload:
            yield payload[key]
    for value in payload.values():
        if isinstance(value, dict):
            yield from _values_by_key(value, keys, _depth + 1)


def find_api_paths(script: str) -> List[str]:
    """API routes quoted inside a page's JavaScript, most promising first.

    A single-page application has to name its own endpoints somewhere,
    and after bundling they survive as plain string literals. This is a
    lead to check by hand, not a discovery: a route here may be one the
    app never calls.
    """
    found = {match.rstrip("/") for match in _API_PATH.findall(script)}
    cleaned = {path if path.startswith("/") else f"/{path}" for path in found if len(path) > 4}
    return sorted(cleaned, key=lambda path: (not _is_interesting(path), path))


def _is_interesting(path: str) -> bool:
    lowered = path.lower()
    return any(word in lowered for word in _INTERESTING)
