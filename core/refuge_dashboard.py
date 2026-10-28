"""The day-refuge dashboard: the atlas plate, filled with live data.

``assets/refuge_dashboard.html`` is the plate itself -- a D3 sheet with
a register panel, a village finder, a watch list and the charts -- with
the data lifted out and five placeholders left in its place. This puts
the current analysis back in, so what the Day refuges view embeds and
what a reader downloads are the same file, built from the same run.

The plate is vendored rather than generated. It is a design, and the
honest way to keep a design is to keep it: rebuilding that sheet out of
Python every release would be a slow way of losing it.

What gets substituted:

``__ED_DATA__``
    One FeatureCollection holding all three kinds the sheet reads --
    refuges, raid areas, and a line per linked raid. The property names
    are :mod:`core.refuges`' own column names, which is why this is a
    pass-through rather than a translation.
``__ED_TODAY__``
    The day every recency band is measured from. Taken from the data
    rather than the clock, so a register that ends in March does not
    show every refuge as two seasons stale.
``__ED_SUB__``, ``__ED_JUMP__``, ``__ED_DIV_LABEL__``
    The standfirst, the "Go to" buttons and the division labels, all
    built from the divisions actually present. ``DIV_LABEL`` matters
    more than it looks: the sheet filters its map labels through it, so
    a division missing from it is a division with no label.

Land cover is what fills the cover, patch and distance columns. Without
it the analysis still produces refuges and the plate still draws them;
the fields it cannot fill arrive as null, and the sheet already renders
a missing value as a dash.
"""

from __future__ import annotations

import json
import logging
import math
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "assets" / "refuge_dashboard.html"

TOKENS = ("__ED_DATA__", "__ED_TODAY__", "__ED_SUB__", "__ED_JUMP__",
          "__ED_DIV_LABEL__")

# Columns that are the geometry, not a property of the thing.
_GEOMETRY_COLUMNS = ("Latitude", "Longitude")

# The sheet reads these off a linked raid, under these names.
_LINK_FIELDS = {
    "Refuge": "refuge",
    "Date": "date",
    "Night Of": "night_of",
    "Hour": "entry_hour",
    "Category": "category",
    "Distance (km)": "distance_km",
}

# A short label for a crowded row of buttons: the division name without
# the words every division in the landscape shares.
_BUTTON_TRIM = (" National Park", " Division", " Tiger Reserve")


def _script_json(value: object) -> str:
    """JSON safe to sit inside a ``<script>`` block.

    The register is an uploaded CSV, so a beat or village name is
    attacker-controllable text, and ``json.dumps`` leaves ``<`` alone: a
    village called ``</script><script>...`` would close the block and
    run. These four escapes are ordinary JSON string escapes, so the
    value a reader gets back is unchanged -- it simply cannot carry a
    tag, and the page is framed with same-origin access to the app.
    """
    out = json.dumps(value, allow_nan=False)
    return (out.replace("<", "\\u003c").replace(">", "\\u003e")
               .replace("&", "\\u0026")
               .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def available() -> bool:
    """Whether the plate is vendored with this checkout."""
    return TEMPLATE_PATH.exists()


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------
def _clean(value: object) -> object:
    """A property the sheet can read, or None.

    NaN is the common case and has to go: it is not JSON, and the sheet
    already draws a missing value as a dash. Timestamps go out as ISO,
    which is what its own date parser slices.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    if hasattr(value, "item"):  # numpy scalar
        try:
            value = value.item()
        except (ValueError, AttributeError):  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _point(row: Dict[str, object], kind: str, skip=()) -> Optional[dict]:
    lat, lon = row.get("Latitude"), row.get("Longitude")
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    properties = {"kind": kind}
    for key, value in row.items():
        if key in _GEOMETRY_COLUMNS or key in skip or str(key).startswith("_"):
            continue
        properties[key] = _clean(value)
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
    }


def _link(row: Dict[str, object], origins: Dict[str, tuple]) -> Optional[dict]:
    """One raid, as the line from its refuge to where it happened."""
    start = origins.get(str(row.get("Refuge")))
    lat, lon = row.get("Latitude"), row.get("Longitude")
    if start is None or lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    properties = {"kind": "linked raid"}
    for column, name in _LINK_FIELDS.items():
        properties[name] = _clean(row.get(column))
    hour = properties.get("entry_hour")
    if hour is not None:
        try:
            properties["entry_hour"] = int(float(hour))
        except (TypeError, ValueError):
            properties["entry_hour"] = None
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "LineString",
            "coordinates": [[start[1], start[0]], [float(lon), float(lat)]],
        },
    }


def feature_collection(result: Dict[str, object]) -> dict:
    """The three kinds the sheet draws, in one collection."""
    refuges = result.get("refuges")
    areas = result.get("areas")
    links = result.get("links")
    refuges = refuges if refuges is not None else pd.DataFrame()
    areas = areas if areas is not None else pd.DataFrame()
    links = links if links is not None else pd.DataFrame()

    features: List[dict] = []
    origins: Dict[str, tuple] = {}
    for row in refuges.to_dict("records"):
        feature = _point(row, "refuge")
        if feature is None:
            continue
        features.append(feature)
        origins[str(row.get("Refuge"))] = (float(row["Latitude"]), float(row["Longitude"]))

    for row in areas.to_dict("records"):
        feature = _point(row, "raid area")
        if feature is not None:
            features.append(feature)

    for row in links.to_dict("records"):
        feature = _link(row, origins)
        if feature is not None:
            features.append(feature)

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# The header, built from what is actually there
# ---------------------------------------------------------------------------
def divisions_in(result: Dict[str, object]) -> List[str]:
    """Division names present, worst-represented last."""
    names: List[str] = []
    for key in ("refuges", "areas"):
        frame = result.get(key)
        if frame is None or not len(frame) or "Division" not in frame:
            continue
        for value in frame["Division"].dropna():
            name = str(value).strip()
            if name and name not in names:
                names.append(name)
    return sorted(names)


def _sentence(names: List[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _button_label(name: str) -> str:
    for suffix in _BUTTON_TRIM:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _jump_buttons(names: List[str]) -> str:
    buttons = ['<button type="button" data-v="all">All</button>']
    for name in names:
        buttons.append(
            f'<button type="button" data-v="{escape(name, quote=True)}">'
            f"{escape(_button_label(name))}</button>"
        )
    return "".join(buttons)


def _as_of(result: Dict[str, object], today: Optional[pd.Timestamp]) -> pd.Timestamp:
    """The day recency is measured from.

    The register's own last day, not the clock: a file that ends in
    March should not open with every refuge two seasons stale.
    """
    if today is not None:
        return pd.Timestamp(today)
    latest = None
    for key, column in (("refuges", "Last Used"), ("areas", "Last Raid")):
        frame = result.get(key)
        if frame is None or not len(frame) or column not in frame:
            continue
        seen = pd.to_datetime(frame[column], errors="coerce").max()
        if pd.notna(seen) and (latest is None or seen > latest):
            latest = seen
    return pd.Timestamp(latest) if latest is not None else pd.Timestamp.today()


# ---------------------------------------------------------------------------
# Building the page
# ---------------------------------------------------------------------------
def build(result: Dict[str, object], today: Optional[pd.Timestamp] = None) -> str:
    """The dashboard as one self-contained page.

    Args:
        result: Output of :func:`core.refuges.find_refuges`.
        today: Override the day recency is measured from. Tests pin it;
            the app lets it follow the register.

    Returns:
        The whole page as HTML, ready to embed or hand over as a file.

    Raises:
        FileNotFoundError: The plate is not vendored with this checkout.
    """
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    names = divisions_in(result)
    where = _sentence(names)
    anchor = _as_of(result, today)

    filled = {
        "__ED_DATA__": _script_json(feature_collection(result)),
        "__ED_TODAY__": (
            f"Date.UTC({anchor.year}, {anchor.month - 1}, {anchor.day})"
        ),
        "__ED_SUB__": escape(
            "Where bulls and herds lie up by day, and the night raids that "
            "leave from them." + (f" {where}." if where else "")
        ),
        "__ED_JUMP__": _jump_buttons(names),
        "__ED_DIV_LABEL__": _script_json({name: name for name in names}),
    }
    for token, value in filled.items():
        if token not in template:
            raise ValueError(f"The vendored plate has no {token} to fill.")
        template = template.replace(token, value, 1)

    logger.info(
        "Built the refuge dashboard: %d feature(s) across %d division(s).",
        len(feature_collection(result)["features"]), len(names),
    )
    return template
