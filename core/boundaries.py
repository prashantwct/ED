"""Forest administrative boundaries as map layers.

Division, range and beat outlines from the MP forest department
shapefiles, vendored as gzipped GeoJSON by ``tools/build_boundaries.py``.
Loading them needs no GIS stack: they are display layers, and the app
keeps its runtime dependencies to what the analysis actually requires.

Which level is drawn follows the zoom, because a beat outline at
landscape zoom is a smudge and a division outline at beat zoom is off
screen. Streamlit's pydeck reports selection events but not viewport
ones, so "zoom" here means the zoom the app computed for the current
filters, not live pinch-zoom. Filtering to a range zooms the view in and
brings beats up with it; the sidebar can also pin a level.

Names carry a wrinkle worth knowing. The shapefiles suffix tiger-reserve
ranges with their zone -- the export's "Kallwah" is "Kallwah Core" here,
"Manpur" is "Manpur Buffer" -- so joining statistics onto a polygon
normalises that away. The geometry itself lines up: every sighting falls
inside a division and a range polygon.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

BOUNDARY_DIR = Path(__file__).resolve().parent.parent / "data" / "boundaries"

DIVISION, RANGE, BEAT, RESERVE = "division", "range", "beat", "reserve"
LEVELS = (DIVISION, RANGE, BEAT)
LEVEL_LABELS = {
    DIVISION: "Division", RANGE: "Range", BEAT: "Beat", RESERVE: "Reserve",
}

# Zoom at which each level takes over. Below the first, only divisions;
# above the last, beats.
ZOOM_TO_RANGE = 8.6
ZOOM_TO_BEAT = 10.8

# Outline colours, from the same muted family as the rest of the map so
# boundaries never compete with the data drawn on top of them.
# Admin levels are drawn as a single line, so the colour has to hold up
# on pale terrain and on satellite imagery alike. A mid-tone amber does;
# the muted greens these started as vanished into both.
LEVEL_STYLE = {
    DIVISION: {"color": [196, 132, 42], "width": 2.4},
    RANGE: {"color": [186, 140, 70], "width": 1.6},
    BEAT: {"color": [176, 148, 96], "width": 1.0},
    RESERVE: {"color": [31, 95, 63], "width": 3.0},
}

# Beats only exist over forest land, so a sighting on farmland between
# two blocks sits in no beat at all. Stated rather than hidden.
BEAT_COVERAGE_NOTE = (
    "Beat outlines cover forest land only, so sightings on farmland "
    "between blocks fall outside every beat."
)

_ZONE_SUFFIX = re.compile(r"\s+(core|buffer|core zone|buffer zone)$", re.IGNORECASE)

_CACHE: Dict[str, dict] = {}


def normalise(name: object) -> str:
    """Comparable form of a boundary name.

    Strips the reserve-zone suffix the shapefiles carry and the export
    does not, so "Kallwah Core" and "Kallwah" are one place.
    """
    text = str(name or "").strip().lower()
    text = _ZONE_SUFFIX.sub("", text)
    return re.sub(r"\s+", " ", text)


def available() -> bool:
    """Whether the vendored boundary files are present."""
    return (BOUNDARY_DIR / f"{DIVISION}.geojson.gz").exists()


def load(level: str) -> dict:
    """Read one boundary layer. Cached for the process."""
    if level in _CACHE:
        return _CACHE[level]

    path = BOUNDARY_DIR / f"{level}.geojson.gz"
    if not path.exists():
        logger.warning("Boundary layer missing: %s", path)
        return {"type": "FeatureCollection", "features": []}

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    for feature in data["features"]:
        feature["properties"]["_key"] = normalise(feature["properties"].get("name"))
    _CACHE[level] = data
    logger.info("Loaded %d %s boundaries", len(data["features"]), level)
    return data


def level_for_zoom(zoom: float) -> str:
    """Which administrative level to draw at this zoom."""
    if zoom >= ZOOM_TO_BEAT:
        return BEAT
    if zoom >= ZOOM_TO_RANGE:
        return RANGE
    return DIVISION


def _bounds(coordinates) -> Tuple[float, float, float, float]:
    """Bounding box of a GeoJSON coordinate tree."""
    xs: List[float] = []
    ys: List[float] = []

    def walk(node):
        if isinstance(node, (int, float)):
            return
        if node and isinstance(node[0], (int, float)):
            xs.append(node[0])
            ys.append(node[1])
            return
        for child in node:
            walk(child)

    walk(coordinates)
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def in_view(
    features: Sequence[dict], west: float, south: float, east: float, north: float
) -> List[dict]:
    """Features whose bounding box overlaps the view.

    1,274 beat polygons is a megabyte of GeoJSON serialised into the
    page. Sending only what the frame can show keeps the map responsive
    when a manager is looking at one range.
    """
    kept = []
    for feature in features:
        box = feature["properties"].get("_bbox")
        if box is None:
            box = _bounds(feature["geometry"]["coordinates"])
            feature["properties"]["_bbox"] = box
        if box[0] <= east and box[2] >= west and box[1] <= north and box[3] >= south:
            kept.append(feature)
    return kept


def annotate(
    features: Sequence[dict], stats: Optional[Dict[str, Dict[str, object]]] = None
) -> List[dict]:
    """Attach the tooltip payload, and beat statistics where they match.

    Returns copies: the cached layer must stay clean, because the next
    call may be for a different filter selection.
    """
    out = []
    for feature in features:
        properties = dict(feature["properties"])
        properties.pop("_bbox", None)
        match = (stats or {}).get(properties.get("_key", ""))
        if match:
            properties.update(match)
        out.append({
            "type": "Feature",
            "geometry": feature["geometry"],
            "properties": properties,
        })
    return out


def stats_by_name(
    table, name_column: str, columns: Iterable[str]
) -> Dict[str, Dict[str, object]]:
    """Index a stats table by normalised boundary name for tooltip joins."""
    if table is None or getattr(table, "empty", True):
        return {}
    wanted = [c for c in columns if c in table.columns]
    indexed: Dict[str, Dict[str, object]] = {}
    for row in table.to_dict("records"):
        key = normalise(row.get(name_column))
        if key and key not in indexed:
            indexed[key] = {c: row[c] for c in wanted}
    return indexed
