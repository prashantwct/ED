"""Static maps for the downloadable brief.

The brief is a single self-contained HTML file, so it cannot host an
interactive deck.gl canvas. Maps are rendered here as inline SVG over an
optional MapTiler static-image basemap embedded as a data URI.

Both use the same Web Mercator maths as the interactive map, so a point
sits in the same place in the brief as it does on screen. If the basemap
cannot be fetched -- no key, no network, an air-gapped deployment -- the
SVG still renders on a plain background rather than failing.
"""

from __future__ import annotations

import base64
import logging
import math
import urllib.error
import urllib.request
from html import escape
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.analytics import classify_conflict
from core.map_engine import (
    BASEMAP_STYLES,
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    DEFAULT_BASEMAP,
    TIER_COLORS,
    maptiler_key,
)

logger = logging.getLogger(__name__)

MAP_WIDTH = 900
MAP_HEIGHT = 520
# Keeps the outermost points off the frame edge.
EDGE_PADDING_PX = 48
MAX_ZOOM = 14
MIN_ZOOM = 3

# Static-image endpoint. @2x keeps it legible when printed.
_STATIC_URL = (
    "https://api.maptiler.com/maps/{style}/static/"
    "{lon},{lat},{zoom}/{w}x{h}@2x.png?key={key}"
)
_FETCH_TIMEOUT_SECONDS = 12

# Points are drawn behind villages and hotspot rings.
SIGHTING_RADIUS_PX = 3.0
VILLAGE_MIN_RADIUS_PX = 4.0
VILLAGE_MAX_RADIUS_PX = 13.0
LABEL_SEPARATION_PX = 78
MAX_LABELS = 8


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def _world_xy(lat: float, lon: float, zoom: float) -> Tuple[float, float]:
    """Web Mercator pixel coordinates at ``zoom``, 256 px tiles."""
    size = 256 * (2**zoom)
    x = (lon + 180.0) / 360.0 * size
    sin_lat = math.sin(math.radians(max(min(lat, 85.05), -85.05)))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * size
    return x, y


def _fit_view(
    lats: List[float], lons: List[float], width: int, height: int
) -> Tuple[float, float, float]:
    """Centre and zoom that fit every point inside the frame."""
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    centre_lat = (lat_min + lat_max) / 2
    centre_lon = (lon_min + lon_max) / 2

    usable_w = max(width - 2 * EDGE_PADDING_PX, 32)
    usable_h = max(height - 2 * EDGE_PADDING_PX, 32)

    zoom = MIN_ZOOM
    for candidate in [z / 2 for z in range(MIN_ZOOM * 2, MAX_ZOOM * 2 + 1)]:
        x0, y0 = _world_xy(lat_max, lon_min, candidate)
        x1, y1 = _world_xy(lat_min, lon_max, candidate)
        if abs(x1 - x0) <= usable_w and abs(y1 - y0) <= usable_h:
            zoom = candidate
        else:
            break
    return centre_lat, centre_lon, zoom


def _projector(centre_lat: float, centre_lon: float, zoom: float,
               width: int, height: int):
    """Return a function mapping lat/lon to pixel coordinates in the frame."""
    cx, cy = _world_xy(centre_lat, centre_lon, zoom)

    def project(lat: float, lon: float) -> Tuple[float, float]:
        x, y = _world_xy(lat, lon, zoom)
        return x - cx + width / 2, y - cy + height / 2

    return project


# ---------------------------------------------------------------------------
# Basemap
# ---------------------------------------------------------------------------
def _basemap_data_uri(
    centre_lat: float, centre_lon: float, zoom: float,
    width: int, height: int, style_name: str,
) -> Optional[str]:
    """Fetch a MapTiler static image and return it as a data URI.

    Returns None on any failure. A brief without terrain behind the
    points is still a usable brief; one that fails to generate is not.
    """
    key = maptiler_key()
    if not key:
        return None

    url = _STATIC_URL.format(
        style=BASEMAP_STYLES.get(style_name, BASEMAP_STYLES[DEFAULT_BASEMAP]),
        lon=round(centre_lon, 6),
        lat=round(centre_lat, 6),
        zoom=round(zoom, 2),
        w=width,
        h=height,
        key=key,
    )
    try:
        with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                logger.warning("Static basemap HTTP %s", response.status)
                return None
            payload = response.read()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("Static basemap unavailable: %s", exc)
        return None

    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------
def sightings_map_svg(
    df: pd.DataFrame,
    hotspots: Optional[pd.DataFrame] = None,
    style_name: str = DEFAULT_BASEMAP,
) -> str:
    """Sightings coloured by conflict type, with hotspot footprints."""
    if df is None or df.empty or not {"Latitude", "Longitude"}.issubset(df.columns):
        return _empty_note("No mappable sightings for the current filters.")
    points = df.dropna(subset=["Latitude", "Longitude"])
    if points.empty:
        return _empty_note("No mappable sightings for the current filters.")

    lats = points["Latitude"].tolist()
    lons = points["Longitude"].tolist()
    if _has_hotspots(hotspots):
        lats += hotspots["Centre Latitude"].tolist()
        lons += hotspots["Centre Longitude"].tolist()

    centre_lat, centre_lon, zoom = _fit_view(lats, lons, MAP_WIDTH, MAP_HEIGHT)
    project = _projector(centre_lat, centre_lon, zoom, MAP_WIDTH, MAP_HEIGHT)

    categories = classify_conflict(points)
    body: List[str] = []

    # Draw presence first so casualties are not hidden underneath.
    order = ["Presence", "Crop", "House", "Injury", "Death"]
    for category in order:
        colour = CATEGORY_COLORS.get(category, (120, 120, 120))
        fill = "rgb({},{},{})".format(*colour)
        radius = SIGHTING_RADIUS_PX * (1.8 if category in ("Death", "Injury") else 1.0)
        for _, row in points[categories == category].iterrows():
            x, y = project(float(row["Latitude"]), float(row["Longitude"]))
            if not _inside(x, y):
                continue
            body.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
                f'fill-opacity="0.85" stroke="#1b1b1b" stroke-width="0.4"/>'
            )

    if _has_hotspots(hotspots):
        body.extend(_hotspot_shapes(hotspots, project, zoom, centre_lat))

    legend = _legend_rows(
        [(CATEGORY_LABELS[c], CATEGORY_COLORS[c]) for c in reversed(order)]
    )
    basemap = _basemap_data_uri(
        centre_lat, centre_lon, zoom, MAP_WIDTH, MAP_HEIGHT, style_name
    )
    return _svg_document(body, basemap, legend)


def village_map_svg(
    village_risk: pd.DataFrame,
    hotspots: Optional[pd.DataFrame] = None,
    style_name: str = DEFAULT_BASEMAP,
) -> str:
    """Villages sized by conflict count and coloured by tier."""
    if (
        village_risk is None
        or village_risk.empty
        or not {"Latitude", "Longitude"}.issubset(village_risk.columns)
    ):
        return _empty_note(
            "No village ranking to map. Village centroids are required."
        )
    villages = village_risk.dropna(subset=["Latitude", "Longitude"])
    if villages.empty:
        return _empty_note("Village rows carry no coordinates.")

    lats = villages["Latitude"].tolist()
    lons = villages["Longitude"].tolist()
    centre_lat, centre_lon, zoom = _fit_view(lats, lons, MAP_WIDTH, MAP_HEIGHT)
    project = _projector(centre_lat, centre_lon, zoom, MAP_WIDTH, MAP_HEIGHT)

    body: List[str] = []
    if _has_hotspots(hotspots):
        body.extend(_hotspot_shapes(hotspots, project, zoom, centre_lat, labels=False))

    events = villages["Conflict Events"].astype(float).clip(lower=0)
    max_events = float(events.max()) or 1.0

    # Worst last, so Critical villages draw on top of Routine ones.
    tier_order = {"Routine": 0, "Watch": 1, "High": 2, "Critical": 3}
    ordered = villages.assign(
        _rank=villages["Tier"].map(tier_order).fillna(0)
    ).sort_values("_rank")

    placed: List[Tuple[float, float]] = []
    labels: List[str] = []
    for _, row in ordered.iterrows():
        x, y = project(float(row["Latitude"]), float(row["Longitude"]))
        if not _inside(x, y):
            continue
        colour = TIER_COLORS.get(str(row["Tier"]), (107, 133, 120))
        fill = "rgb({},{},{})".format(*colour)
        share = float(row["Conflict Events"]) / max_events
        radius = VILLAGE_MIN_RADIUS_PX + math.sqrt(max(share, 0)) * (
            VILLAGE_MAX_RADIUS_PX - VILLAGE_MIN_RADIUS_PX
        )
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
            f'fill-opacity="0.88" stroke="#ffffff" stroke-width="1.3"/>'
        )

        if str(row["Tier"]) in ("Critical", "High") and len(labels) < MAX_LABELS:
            if all(math.hypot(x - px, y - py) >= LABEL_SEPARATION_PX
                   for px, py in placed):
                placed.append((x, y))
                labels.append(_label(x, y - radius - 6, str(row["Village"])))

    body.extend(labels)
    legend = _legend_rows(
        [(tier, TIER_COLORS[tier]) for tier in ("Critical", "High", "Watch", "Routine")]
    )
    basemap = _basemap_data_uri(
        centre_lat, centre_lon, zoom, MAP_WIDTH, MAP_HEIGHT, style_name
    )
    return _svg_document(body, basemap, legend)


# ---------------------------------------------------------------------------
# Shared drawing
# ---------------------------------------------------------------------------
def _hotspot_shapes(
    hotspots: pd.DataFrame, project, zoom: float, centre_lat: float,
    labels: bool = True,
) -> List[str]:
    """Hotspot footprints at true ground radius."""
    metres_per_pixel = 156543.03392 * math.cos(math.radians(centre_lat)) / (2**zoom)
    shapes: List[str] = []
    for _, row in hotspots.iterrows():
        x, y = project(
            float(row["Centre Latitude"]), float(row["Centre Longitude"])
        )
        if not _inside(x, y, margin=60):
            continue
        radius_px = max(float(row["Radius (km)"]) * 1000 / metres_per_pixel, 4)
        colour = TIER_COLORS.get(str(row["Tier"]), (107, 133, 120))
        stroke = "rgb({},{},{})".format(*colour)
        shapes.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius_px:.1f}" '
            f'fill="{stroke}" fill-opacity="0.10" stroke="{stroke}" '
            f'stroke-width="1.6" stroke-opacity="0.85"/>'
        )
        if labels:
            shapes.append(_label(x, y, str(row["Hotspot"]), anchor="middle"))
    return shapes


def _label(x: float, y: float, text: str, anchor: str = "middle") -> str:
    """White text with a dark halo, legible over terrain or imagery."""
    safe = escape(str(text))
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Segoe UI, Arial, sans-serif" font-size="12" '
        f'font-weight="600" fill="#ffffff" stroke="#1a231f" stroke-width="3" '
        f'paint-order="stroke" stroke-linejoin="round">{safe}</text>'
    )


def _has_hotspots(hotspots: Optional[pd.DataFrame]) -> bool:
    return (
        hotspots is not None
        and not hotspots.empty
        and {"Centre Latitude", "Centre Longitude", "Radius (km)"}.issubset(
            hotspots.columns
        )
    )


def _inside(x: float, y: float, margin: float = 4.0) -> bool:
    return -margin <= x <= MAP_WIDTH + margin and -margin <= y <= MAP_HEIGHT + margin


def _legend_rows(entries: List[Tuple[str, Tuple[int, int, int]]]) -> str:
    chips = []
    for index, (label, colour) in enumerate(entries):
        fill = "rgb({},{},{})".format(*colour)
        x = 12 + index * 150
        chips.append(
            f'<circle cx="{x + 7}" cy="14" r="5.5" fill="{fill}" '
            f'stroke="#33413a" stroke-width="0.8"/>'
            f'<text x="{x + 18}" y="18" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11.5" fill="#33413a">{escape(label)}</text>'
        )
    return "".join(chips)


def _svg_document(body: List[str], basemap: Optional[str], legend: str) -> str:
    """Wrap drawing instructions in an SVG, over the basemap if available."""
    if basemap:
        background = (
            f'<image href="{basemap}" x="0" y="0" width="{MAP_WIDTH}" '
            f'height="{MAP_HEIGHT}" preserveAspectRatio="none"/>'
        )
        note = ""
    else:
        background = (
            f'<rect x="0" y="0" width="{MAP_WIDTH}" height="{MAP_HEIGHT}" '
            f'fill="#eef2ef"/>'
        )
        note = (
            '<text x="12" y="20" font-family="Segoe UI, Arial, sans-serif" '
            'font-size="11.5" fill="#7a8a80">Basemap unavailable when this brief '
            "was generated.</text>"
        )

    legend_height = 28
    total_height = MAP_HEIGHT + legend_height
    return (
        f'<div class="map-figure">'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {MAP_WIDTH} '
        f'{total_height}" width="100%" role="img">'
        f"{background}{note}{''.join(body)}"
        f'<rect x="0" y="{MAP_HEIGHT}" width="{MAP_WIDTH}" height="{legend_height}" '
        f'fill="#ffffff"/>'
        f'<g transform="translate(0, {MAP_HEIGHT})">{legend}</g>'
        f"</svg></div>"
    )


def _empty_note(message: str) -> str:
    return f'<p class="empty-note">{escape(message)}</p>'


def filter_summary(filters: Optional[Dict[str, object]]) -> str:
    """One-line description of the filters behind the figures.

    Returns HTML: the separator is markup, so the values it wraps are
    escaped here rather than at the call site. Division, Range and Beat
    names come from the uploaded CSV, and the brief is a file that gets
    forwarded and opened elsewhere.
    """
    if not filters:
        return "All divisions"

    parts: List[str] = []
    for label, key in (("Division", "divisions"), ("Range", "ranges"), ("Beat", "beats")):
        values = filters.get(key) or []
        if values:
            shown = ", ".join(escape(str(v)) for v in list(values)[:4])
            if len(values) > 4:
                shown += f", +{len(values) - 4} more"
            parts.append(f"{label}: {shown}")

    severity = filters.get("severity")
    if severity and severity != "All reports":
        parts.append(f"Filter: {escape(str(severity))}")

    return " &nbsp;|&nbsp; ".join(parts) if parts else "All divisions"
