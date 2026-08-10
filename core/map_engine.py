"""Map rendering, built on pydeck/Deck.GL over MapTiler basemaps.

Two maps: the operational view (sightings, hotspot footprints, villages,
toggleable) and a village-risk view that leads with the settlements.

The MapTiler key is read from Streamlit secrets or the environment, never
hard-coded. Tiles are fetched by the browser, so the key is visible to
anyone using the app -- restrict it by origin in the MapTiler dashboard
rather than trying to hide it. Without a key the maps fall back to a
keyless Carto basemap.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, Optional, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st

from core.analytics import classify_conflict
from core.config import (
    KM_PER_DEG_LAT,
    MAX_RADIUS_M,
    MIN_RADIUS_M,
    RADIUS_MAX_PIXELS,
    RADIUS_MIN_PIXELS,
)
from core.ui import CATEGORY_STYLE, TIER_STYLE, category_legend, map_legend

logger = logging.getLogger(__name__)

# MapTiler styles worth offering a forest manager. Terrain and imagery
# matter more here than street cartography.
BASEMAP_STYLES = {
    "Outdoor (terrain)": "outdoor-v2",
    "Satellite": "satellite",
    "Hybrid (satellite + labels)": "hybrid",
    "Topographic": "topo-v2",
    "Streets": "streets-v2",
    "Minimal (data focus)": "dataviz",
}
DEFAULT_BASEMAP = "Outdoor (terrain)"

_MAPTILER_STYLE_URL = "https://api.maptiler.com/maps/{style}/style.json?key={key}"

# Keyless fallback. Carto's Positron basemap needs no API key, so the map
# still has geographic context when MAPTILER_KEY is unset -- points on a
# blank white background are close to useless for reading a landscape.
_CARTO_FALLBACK_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

# maplibre is the provider for third-party style URLs. Leaving it at the
# 'carto' default asks the frontend to treat a MapTiler URL as a Carto
# basemap.
_MAP_PROVIDER = "maplibre"


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


# Colour by what happened, not by a severity ramp: anchored on a fatality
# at 100, a ramp renders every non-fatal incident the same shade. Palette
# lives in core.ui so map, legend and report agree, and is colourblind-
# safe. Size carries the same signal in parallel.
CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    key: _hex_to_rgb(style["color"]) for key, style in CATEGORY_STYLE.items()
}
CATEGORY_LABELS = {key: style["label"] for key, style in CATEGORY_STYLE.items()}
TIER_COLORS: Dict[str, Tuple[int, int, int]] = {
    key: _hex_to_rgb(style["accent"]) for key, style in TIER_STYLE.items()
}

# Casualty incidents are drawn at full size regardless of the severity
# arithmetic, so the worst points stay findable at landscape zoom.
EMPHASIS_CATEGORIES = ("Death", "Injury")

# Label only what a reader can act on; labelling 231 villages is noise.
LABELLED_TIERS = ("Critical", "High")

# deck.gl's TextLayer has no collision handling, so labels are thinned by
# hand. The separation has to be in *pixels*: at the zoom that fits this
# landscape one pixel is nearly 2 km, so a 4 km rule spaced labels 13 px
# apart while a name like "Kekarpani" is ~70 px wide. Converted to a
# ground distance per zoom level in _label_separation_km.
LABEL_SEPARATION_PX = 70
MAX_LABELS = 8
LABEL_FONT_SIZE = 13


def maptiler_key() -> Optional[str]:
    """Resolve the MapTiler key from Streamlit secrets or the environment."""
    try:
        key = st.secrets.get("MAPTILER_KEY")
        if key:
            return str(key)
    except Exception:  # noqa: BLE001 - no secrets file is a normal state
        pass
    return os.getenv("MAPTILER_KEY") or None


def basemap_style(style_name: str = DEFAULT_BASEMAP) -> str:
    """Basemap style URL. Falls back to the keyless Carto basemap."""
    key = maptiler_key()
    if not key:
        return _CARTO_FALLBACK_STYLE
    style = BASEMAP_STYLES.get(style_name, BASEMAP_STYLES[DEFAULT_BASEMAP])
    return _MAPTILER_STYLE_URL.format(style=style, key=key)


def basemap_warning() -> Optional[str]:
    """Message shown when the selected basemap is unavailable."""
    if maptiler_key():
        return None
    return (
        "No MAPTILER_KEY set, so the basemap selector is inactive and a plain "
        "fallback map is shown. Add the key under Streamlit secrets for terrain "
        "and satellite."
    )


# ---------------------------------------------------------------------------
# Operational map
# ---------------------------------------------------------------------------
def render_map(
    df: pd.DataFrame,
    hotspots: Optional[pd.DataFrame] = None,
    villages: Optional[pd.DataFrame] = None,
    style_name: str = DEFAULT_BASEMAP,
    show_sightings: bool = True,
    show_hotspots: bool = True,
    show_villages: bool = True,
) -> None:
    """Render the operational map with the requested layers."""
    if df.empty:
        st.info("No data available to display on the map with the current filters.")
        return
    if df["Latitude"].isna().all() or df["Longitude"].isna().all():
        st.warning("No valid coordinates available to plot.")
        return

    plot_df = _prepare_sightings(df)
    view_state = _adaptive_view_state(plot_df)

    layers = []
    if show_hotspots and hotspots is not None and not hotspots.empty:
        layers.extend(_hotspot_layers(hotspots))
    if show_sightings:
        layers.append(_sighting_layer(plot_df))
    if show_villages and villages is not None and not villages.empty:
        layers.extend(_village_layers(villages, view_state))

    if not layers:
        st.info("All map layers are switched off. Enable one in the sidebar.")
        return

    _render_deck(layers, view_state, style_name)

    if show_sightings:
        category_legend(plot_df["_category"].value_counts().to_dict())
    if show_villages and villages is not None and not villages.empty:
        map_legend("Villages by tier", TIER_STYLE)
    _basemap_note()


def render_village_map(
    village_risk: pd.DataFrame,
    hotspots: Optional[pd.DataFrame] = None,
    style_name: str = DEFAULT_BASEMAP,
) -> None:
    """Render the village-risk map: settlements over hotspot footprints."""
    if village_risk is None or village_risk.empty:
        st.info(
            "No village-level ranking to map. Load village centroids to enable it."
        )
        return

    view_state = _adaptive_view_state(village_risk)

    layers = []
    if hotspots is not None and not hotspots.empty:
        layers.extend(_hotspot_layers(hotspots, labels=False))
    layers.extend(_village_layers(village_risk, view_state, emphasise=True))

    _render_deck(layers, view_state, style_name)
    map_legend("Villages by tier", TIER_STYLE)
    st.caption(
        "Circle size is conflict count; rings are hotspot footprints. Only the "
        "worst few villages are labelled to keep names legible - hover for the "
        "rest."
    )
    _basemap_note()


def _basemap_note() -> None:
    warning = basemap_warning()
    if warning:
        st.caption(warning)


def _render_deck(layers, view_state, style_name: str) -> None:
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=basemap_style(style_name),
        map_provider=_MAP_PROVIDER,
        tooltip={
            "html": "{_tooltip}",
            "style": {"backgroundColor": "#1f5f3f", "color": "white",
                      "fontSize": "12px"},
        },
    )
    st.pydeck_chart(deck, width="stretch")


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
def _prepare_sightings(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df["_category"] = classify_conflict(plot_df)
    plot_df["_category_label"] = (
        plot_df["_category"].map(CATEGORY_LABELS).fillna("Unknown")
    )

    colors = plot_df["_category"].map(CATEGORY_COLORS)
    colors = colors.where(
        colors.notna(), pd.Series([(120, 120, 120)] * len(plot_df), index=plot_df.index)
    )
    plot_df[["_r", "_g", "_b"]] = pd.DataFrame(colors.tolist(), index=plot_df.index)
    plot_df["_radius"] = _severity_to_radius(plot_df)

    for col in ["Division", "Range", "Beat", "Nearest Village"]:
        if col not in plot_df.columns:
            plot_df[col] = "N/A"

    date = (
        plot_df["Date"].dt.strftime("%d %b %Y")
        if "Date" in plot_df.columns
        else pd.Series("N/A", index=plot_df.index)
    )
    plot_df["_tooltip"] = (
        "<b>" + plot_df["_category_label"].astype(str) + "</b><br/>"
        + date.astype(str) + "<br/>"
        + plot_df["Beat"].astype(str) + " &middot; "
        + plot_df["Range"].astype(str) + " &middot; "
        + plot_df["Division"].astype(str) + "<br/>"
        + "Nearest village: " + plot_df["Nearest Village"].astype(str)
    )
    return plot_df


def _sighting_layer(plot_df: pd.DataFrame) -> pdk.Layer:
    # Ship only what the layer and tooltip use: the frame is serialised to
    # JSON for the browser, and timestamps do not survive that round trip.
    columns = [
        "Longitude", "Latitude", "_radius", "_r", "_g", "_b", "_tooltip",
    ]
    return pdk.Layer(
        "ScatterplotLayer",
        data=plot_df[columns],
        get_position="[Longitude, Latitude]",
        get_radius="_radius",
        get_fill_color="[_r, _g, _b, 190]",
        get_line_color=[30, 30, 30, 200],
        line_width_min_pixels=1,
        radius_min_pixels=RADIUS_MIN_PIXELS,
        radius_max_pixels=RADIUS_MAX_PIXELS,
        pickable=True,
        auto_highlight=True,
    )


def _hotspot_layers(hotspots: pd.DataFrame, labels: bool = True) -> list:
    """Hotspot footprints as true-scale rings, plus optional ID labels."""
    frame = hotspots.copy()
    frame["_radius_m"] = frame["Radius (km)"].astype(float) * 1000
    colors = frame["Tier"].map(TIER_COLORS)
    colors = colors.where(
        colors.notna(), pd.Series([(107, 133, 120)] * len(frame), index=frame.index)
    )
    frame[["_r", "_g", "_b"]] = pd.DataFrame(colors.tolist(), index=frame.index)
    frame["_tooltip"] = (
        "<b>" + frame["Hotspot"].astype(str) + " &middot; "
        + frame["Tier"].astype(str) + "</b><br/>"
        + frame["Divisions"].astype(str) + "<br/>"
        + frame["Sightings"].astype(int).astype(str) + " sightings, "
        + frame["Conflict Events"].astype(int).astype(str) + " conflicts ("
        + frame["Conflict Share %"].round(0).astype(int).astype(str) + "%)<br/>"
        + frame["Human Deaths"].astype(int).astype(str) + " killed, "
        + frame["People Injured"].astype(int).astype(str) + " injured<br/>"
        + "Radius " + frame["Radius (km)"].round(1).astype(str) + " km"
    )

    columns = [
        "Centre Longitude", "Centre Latitude", "_radius_m",
        "_r", "_g", "_b", "_tooltip",
    ]
    out = [
        pdk.Layer(
            "ScatterplotLayer",
            data=frame[columns],
            get_position="[Centre Longitude, Centre Latitude]",
            get_radius="_radius_m",
            get_fill_color="[_r, _g, _b, 38]",
            get_line_color="[_r, _g, _b, 210]",
            line_width_min_pixels=2,
            stroked=True,
            filled=True,
            pickable=True,
        )
    ]
    if labels:
        out.append(
            pdk.Layer(
                "TextLayer",
                data=frame[["Centre Longitude", "Centre Latitude", "Hotspot"]],
                get_position="[Centre Longitude, Centre Latitude]",
                get_text="Hotspot",
                get_size=13,
                get_color=[255, 255, 255, 255],
                get_alignment_baseline="'center'",
                font_settings={"sdf": True},
                outline_width=4,
                outline_color=[20, 28, 24, 255],
                font_weight=600,
                background=True,
                get_background_color=[20, 28, 24, 150],
                background_padding=[5, 2],
            )
        )
    return out


def _village_layers(
    village_risk: pd.DataFrame,
    view_state: Optional[pdk.ViewState] = None,
    emphasise: bool = False,
) -> list:
    """Village markers coloured by tier, sized by conflict count."""
    frame = village_risk.dropna(subset=["Latitude", "Longitude"]).copy()
    if frame.empty:
        return []

    colors = frame["Tier"].map(TIER_COLORS)
    colors = colors.where(
        colors.notna(), pd.Series([(107, 133, 120)] * len(frame), index=frame.index)
    )
    frame[["_r", "_g", "_b"]] = pd.DataFrame(colors.tolist(), index=frame.index)

    # Square-root scaling: conflict counts are long-tailed, so linear
    # radius makes the busiest village swamp everything around it.
    events = frame["Conflict Events"].astype(float).clip(lower=0)
    max_events = float(events.max()) or 1.0
    base = 260 if emphasise else 180
    frame["_radius_m"] = base + (events / max_events) ** 0.5 * (base * 3)

    frame["_tooltip"] = (
        "<b>" + frame["Village"].astype(str) + " &middot; "
        + frame["Tier"].astype(str) + "</b><br/>"
        + frame["Conflict Events"].astype(int).astype(str) + " conflict events<br/>"
        + frame["Human Deaths"].astype(int).astype(str) + " killed, "
        + frame["People Injured"].astype(int).astype(str) + " injured<br/>"
        + "House " + frame["House Damage Events"].astype(int).astype(str)
        + " &middot; crop " + frame["Crop Damage Events"].astype(int).astype(str)
        + "<br/>Nearest hotspot: " + frame["Nearest Hotspot"].astype(str)
    )

    columns = ["Longitude", "Latitude", "_radius_m", "_r", "_g", "_b", "_tooltip"]
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=frame[columns],
            get_position="[Longitude, Latitude]",
            get_radius="_radius_m",
            get_fill_color="[_r, _g, _b, 205]",
            get_line_color=[255, 255, 255, 230],
            line_width_min_pixels=1.5,
            stroked=True,
            filled=True,
            radius_min_pixels=5,
            radius_max_pixels=26,
            pickable=True,
            auto_highlight=True,
        )
    ]

    if not emphasise or view_state is None:
        return layers

    separation = _label_separation_km(view_state.zoom, view_state.latitude)
    labelled = _select_labels(frame, separation)
    if not labelled.empty:
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=labelled[["Longitude", "Latitude", "Village"]],
                get_position="[Longitude, Latitude]",
                get_text="Village",
                get_size=LABEL_FONT_SIZE,
                get_color=[255, 255, 255, 255],
                get_pixel_offset=[0, -18],
                font_settings={"sdf": True},
                outline_width=4,
                outline_color=[20, 28, 24, 255],
                font_weight=600,
                background=True,
                get_background_color=[20, 28, 24, 140],
                background_padding=[5, 2],
            )
        )
    return layers


def _label_separation_km(zoom: float, latitude: float) -> float:
    """Ground distance matching LABEL_SEPARATION_PX at the given zoom.

    Web-Mercator resolution: 156543 m/px at zoom 0 on the equator,
    halving per zoom level and shrinking by cos(latitude).
    """
    metres_per_pixel = 156543.03 * math.cos(math.radians(latitude)) / (2**zoom)
    return LABEL_SEPARATION_PX * metres_per_pixel / 1000


def _select_labels(frame: pd.DataFrame, separation_km: float) -> pd.DataFrame:
    """Pick labels that will not overprint at the map's starting zoom.

    Walks villages worst-first and keeps a label only where it clears
    every label already placed.
    """
    candidates = frame[frame["Tier"].isin(LABELLED_TIERS)].copy()
    if candidates.empty:
        return candidates

    tier_rank = {tier: i for i, tier in enumerate(TIER_STYLE)}
    candidates["_rank"] = candidates["Tier"].map(tier_rank).fillna(len(tier_rank))
    candidates = candidates.sort_values(
        ["_rank", "Human Deaths", "Conflict Events"], ascending=[True, False, False]
    )

    lat_scale = KM_PER_DEG_LAT
    lon_scale = KM_PER_DEG_LAT * math.cos(
        math.radians(float(candidates["Latitude"].mean()))
    )

    kept: list = []
    keep_index = []
    for idx, row in candidates.iterrows():
        x = float(row["Longitude"]) * lon_scale
        y = float(row["Latitude"]) * lat_scale
        if all(math.hypot(x - kx, y - ky) >= separation_km for kx, ky in kept):
            kept.append((x, y))
            keep_index.append(idx)
        if len(keep_index) >= MAX_LABELS:
            break

    return candidates.loc[keep_index]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _severity_to_radius(df: pd.DataFrame) -> pd.Series:
    """Scale severity into a metre radius, casualties at full size.

    Log-scaled: a fatality scores ~200x a presence sighting, so linear
    scaling collapses every non-fatal incident onto the minimum radius.
    """
    scores = pd.to_numeric(df.get("Severity Score"), errors="coerce").fillna(0.0)

    log_scores = scores.clip(lower=0).apply(math.log1p)
    max_log = float(log_scores.max())
    normalised = log_scores / max_log if max_log > 0 else log_scores * 0.0

    radius = MIN_RADIUS_M + normalised * (MAX_RADIUS_M - MIN_RADIUS_M)
    if "_category" in df.columns:
        radius = radius.mask(df["_category"].isin(EMPHASIS_CATEGORIES), MAX_RADIUS_M)
    return radius


def _adaptive_view_state(df: pd.DataFrame) -> pdk.ViewState:
    """Centre and zoom that fit all points, within sane bounds."""
    lat_min, lat_max = df["Latitude"].min(), df["Latitude"].max()
    lon_min, lon_max = df["Longitude"].min(), df["Longitude"].max()

    lat_span = max(lat_max - lat_min, 0.01)
    # Compare spans in comparable units before picking the limiting one.
    mean_lat = float((lat_min + lat_max) / 2)
    lon_span = max((lon_max - lon_min) * math.cos(math.radians(mean_lat)), 0.01)
    span = max(lat_span, lon_span)

    # One zoom level per doubling of span, anchored so ~0.05 deg is z12.
    zoom = 12 - math.log2(max(span / 0.05, 1))
    zoom = min(max(zoom, 5), 14)

    return pdk.ViewState(
        latitude=mean_lat,
        longitude=float((lon_min + lon_max) / 2),
        zoom=zoom,
        pitch=0,
    )
