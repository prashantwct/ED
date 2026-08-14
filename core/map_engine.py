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

import json
import logging
import math
import os
from functools import wraps
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st

from core import boundaries
from core.analytics import classify_conflict, classify_group
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


# ---------------------------------------------------------------------------
# Tooltips
# ---------------------------------------------------------------------------
# The markup lives in the template below, once per chart, and the layers
# supply plain text into it.
#
# It cannot be the other way round. Streamlit interpolates {field} into
# the tooltip's html and HTML-escapes the value first -- in the frontend,
#
#     xie = e => String(e).replace(/[&<>"']/g, e => bie[e])
#     createTooltip: e => (t.html = X9(e, t.html, true), t)
#
# which is an XSS guard on values that came out of an uploaded CSV, and
# not something to defeat. Shipping a built card per row put the whole of
# that card through the escaper, so the reader got `<div class="mt">...`
# as visible text instead of a tooltip.
#
# Two consequences worth keeping in mind:
#
# * Do not escape on this side. Streamlit does it at interpolation, and
#   escaping twice renders "&amp;amp;" in a village name.
# * Every pickable layer must supply every field named in the template.
#   The interpolator leaves a field the hovered object does not have
#   exactly as it found it, so a missing one shows the reader "{_v3}".
#   There is a test holding that.
_TOOLTIP_STYLE = {
    "backgroundColor": "transparent",
    "color": "inherit",
    "padding": "0",
    "margin": "0",
    "boxShadow": "none",
    "fontSize": "12px",
}
# Above this the header takes dark ink. The crop category is pale yellow
# and white on it is unreadable.
_LIGHT_HEADER_LUMINANCE = 150

# Rows the card can hold. Cards are built from what actually happened, so
# five is comfortably above what any layer fills in practice.
_TOOLTIP_ROWS = 5

# Class toggled onto a slot that has nothing to say. A row of two empty
# spans is not :empty, so the CSS cannot find it on its own -- and a
# class name survives the escaper untouched, having no special
# characters in it.
_OFF = "mt-off"


def _tooltip_html() -> str:
    """The one card, as a template Streamlit fills per feature."""
    rows = "".join(
        f'<div class="mt-r {{_h{i}}}">'
        f'<span class="mt-l">{{_l{i}}}</span>'
        f'<span class="mt-v">{{_v{i}}}</span></div>'
        for i in range(1, _TOOLTIP_ROWS + 1)
    )
    return (
        '<div class="mt">'
        '<div class="mt-h {_k}" style="background:{_a}">{_t}'
        '<div class="mt-s {_hs}">{_s}</div></div>'
        f'<div class="mt-b">{rows}</div>'
        '<div class="mt-f {_hf}">{_f}</div>'
        "</div>"
    )


_TOOLTIP_HTML = _tooltip_html()

# Every field the template names, so layers can be checked against it.
TOOLTIP_FIELDS: Tuple[str, ...] = tuple(
    ["_t", "_s", "_a", "_k", "_f", "_hs", "_hf"]
    + [f"_{part}{i}" for i in range(1, _TOOLTIP_ROWS + 1) for part in ("l", "v", "h")]
)


def _slots(
    title: str,
    accent: Tuple[int, int, int],
    rows: List[Tuple[str, object]],
    subtitle: str = "",
    footer: str = "",
) -> Dict[str, str]:
    """One tooltip's worth of plain values for the template.

    Labelled rows rather than a run-on sentence: a hover is read in
    about a second, and a column of label/value pairs survives that
    where prose does not. Rows with nothing to report are switched off
    rather than shown as a zero.
    """
    luminance = 0.299 * accent[0] + 0.587 * accent[1] + 0.114 * accent[2]
    kept = [(l, v) for l, v in rows if v not in (None, "")][:_TOOLTIP_ROWS]

    slots = {
        "_t": str(title),
        "_a": f"rgb({accent[0]},{accent[1]},{accent[2]})",
        "_k": "mt-dark" if luminance > _LIGHT_HEADER_LUMINANCE else "",
        "_s": str(subtitle),
        "_hs": "" if subtitle else _OFF,
        "_f": str(footer),
        "_hf": "" if footer else _OFF,
    }
    for i in range(1, _TOOLTIP_ROWS + 1):
        label, value = kept[i - 1] if i <= len(kept) else ("", "")
        slots[f"_l{i}"] = str(label)
        slots[f"_v{i}"] = str(value)
        slots[f"_h{i}"] = "" if i <= len(kept) else _OFF
    return slots


def _pct(value: object) -> Optional[str]:
    """Percentage, or nothing at all when it was never measured.

    ``float(x or 0)`` does not help here: NaN is truthy, so a missing
    night share sailed through and rendered as "nan%".
    """
    if value is None or pd.isna(value):
        return None
    return f"{float(value):.0f}%"


# Plain ASCII throughout. Em dashes and middots rendered as stray marks
# in a hover the reader has about a second for, and a slash reads the
# same in every font the field offices have.
_DASH = "-"
_SEP = " / "


def _trail(*parts: object) -> str:
    """Breadcrumb of whatever is actually known."""
    known = [_clean(p, "") for p in parts]
    return _SEP.join(p for p in known if p)


def _clean(value: object, dash: str = _DASH) -> str:
    """Blank-safe display value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return dash
    text = str(value).strip()
    return text if text and text.lower() not in ("nan", "none", "unknown") else dash


def _count(value: object) -> Optional[str]:
    """A count worth a row, or nothing.

    Most villages killed nobody and most hotspots damaged no houses. A
    card that lists five zeros to reach the one number that matters
    reads as a wall of text, so a zero drops the row entirely and the
    absence carries the same meaning in less space.
    """
    if value is None or pd.isna(value):
        return None
    number = int(float(value))
    return f"{number:,}" if number else None


def _seen_between(row: dict) -> str:
    """First-to-last-seen line for a hotspot."""
    first, last = row.get("First Seen"), row.get("Last Seen")
    if pd.isna(first) or pd.isna(last):
        return ""
    return f"{pd.Timestamp(first):%d %b} to {pd.Timestamp(last):%d %b %Y}"


def _hotspot_relation(row: dict) -> str:
    """Where a village sits relative to the nearest hotspot."""
    name = _clean(row.get("Nearest Hotspot"), "")
    if not name:
        return ""
    if row.get("Inside Hotspot"):
        return f"Inside hotspot {name}"
    distance = row.get("Distance to Hotspot (km)")
    if pd.notna(distance):
        return f"{float(distance):.1f} km from hotspot {name}"
    return f"Nearest hotspot {name}"


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
def _never_breaks_the_page(render):
    """Contain a map failure to its own section.

    The maps sit in the middle of a long page. A crash in one takes the
    beat table, the brief and the download with it, which is a bad trade
    for a layer that failed to draw -- the numbers are what the manager
    came for. Report it where the map would have been and carry on.

    This also covers the deployment case where the entry point has been
    reloaded but an imported module has not, so a freshly added argument
    hits an older function.
    """

    @wraps(render)
    def guarded(*args, **kwargs):
        try:
            return render(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the page must survive
            logger.exception("%s failed", render.__name__)
            st.error(
                f"The map could not be drawn: {type(exc).__name__}. Everything "
                "else on this page is unaffected. If this followed an update, "
                "reboot the app rather than rerunning it -- a rerun reloads "
                "the page script but keeps the old modules in memory."
            )
            return None

    return guarded


@_never_breaks_the_page
def render_map(
    df: pd.DataFrame,
    hotspots: Optional[pd.DataFrame] = None,
    villages: Optional[pd.DataFrame] = None,
    style_name: str = DEFAULT_BASEMAP,
    show_sightings: bool = True,
    show_hotspots: bool = True,
    show_villages: bool = True,
    show_boundaries: bool = True,
    boundary_level: Optional[str] = None,
    beat_stats: Optional[pd.DataFrame] = None,
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
    layers.extend(boundary_layers(
        view_state, boundary_level, beat_stats, show_reserves=show_boundaries
    ) if show_boundaries else [])
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

    if show_boundaries:
        _boundary_note(view_state, boundary_level)
    if show_sightings:
        category_legend(plot_df["_category"].value_counts().to_dict())
    if show_villages and villages is not None and not villages.empty:
        map_legend("Villages by tier", TIER_STYLE)
    _basemap_note()


@_never_breaks_the_page
def render_village_map(
    village_risk: pd.DataFrame,
    hotspots: Optional[pd.DataFrame] = None,
    style_name: str = DEFAULT_BASEMAP,
    show_boundaries: bool = True,
    boundary_level: Optional[str] = None,
) -> None:
    """Render the village-risk map: settlements over hotspot footprints."""
    if village_risk is None or village_risk.empty:
        st.info(
            "No village-level ranking to map. Load village centroids to enable it."
        )
        return

    view_state = _adaptive_view_state(village_risk)

    layers = list(boundary_layers(view_state, boundary_level)) if show_boundaries else []
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


def _boundary_note(view_state: pdk.ViewState, level: Optional[str]) -> None:
    """Say which level is drawn and why, so it is not a surprise."""
    chosen = level or boundaries.level_for_zoom(float(view_state.zoom))
    how = "pinned" if level else "chosen for this zoom"
    note = f"{boundaries.LEVEL_LABELS[chosen]} boundaries ({how})."
    if chosen == boundaries.BEAT:
        note += " " + boundaries.BEAT_COVERAGE_NOTE
    st.caption(note)


def _basemap_note() -> None:
    warning = basemap_warning()
    if warning:
        st.caption(warning)


def _compact(deck: pdk.Deck) -> pdk.Deck:
    """Strip the pretty-printing out of the deck spec.

    pydeck serialises with ``indent=2`` and Streamlit puts exactly that
    string on the wire. Across 1,761 sightings and a boundary layer the
    whitespace is about two thirds of the payload, which is a lot to
    push through a websocket on every rerun for something no one reads.
    """
    spec = json.loads(deck.to_json())
    deck.to_json = lambda: json.dumps(spec, separators=(",", ":"))
    return deck


def _render_deck(layers, view_state, style_name: str) -> None:
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=basemap_style(style_name),
        map_provider=_MAP_PROVIDER,
        tooltip={"html": _TOOLTIP_HTML, "style": _TOOLTIP_STYLE},
    )
    st.pydeck_chart(_compact(deck), width="stretch")


# ---------------------------------------------------------------------------
# Administrative boundaries
# ---------------------------------------------------------------------------
def boundary_layers(
    view_state: pdk.ViewState,
    level: Optional[str] = None,
    beat_stats: Optional[pd.DataFrame] = None,
    show_reserves: bool = True,
) -> list:
    """Outline layers for the current view.

    ``level`` of None picks one from the zoom: divisions across a
    landscape, beats once the view is tight enough for them to read.
    """
    if not boundaries.available():
        return []

    chosen = level or boundaries.level_for_zoom(float(view_state.zoom))
    layers = []

    if show_reserves:
        layers.extend(_outline_layers(
            boundaries.RESERVE,
            boundaries.annotate(boundaries.load(boundaries.RESERVE)["features"]),
        ))

    features = boundaries.load(chosen)["features"]
    features = boundaries.in_view(features, *_view_bounds(view_state))
    stats = boundaries.stats_by_name(
        beat_stats, "Beat",
        ["Priority Tier", "Reports", "Conflict Events", "Human Deaths"],
    ) if chosen == boundaries.BEAT else {}
    layers.extend(_outline_layers(chosen, boundaries.annotate(features, stats)))
    return layers


def _view_bounds(view_state: pdk.ViewState) -> Tuple[float, float, float, float]:
    """Rough lon/lat box for the viewport, generous at the edges.

    Half a degree of slack at the widest zoom keeps polygons that only
    just intrude on the frame from popping in and out.
    """
    span = max(0.35, 42.0 / (2 ** float(view_state.zoom)) * 8)
    lat, lon = float(view_state.latitude), float(view_state.longitude)
    return (lon - span, lat - span, lon + span, lat + span)


def _outline_layers(level: str, features: list) -> List[pdk.Layer]:
    """A boundary drawn as a light casing under a dark line.

    The basemap is selectable and runs from pale terrain to satellite
    imagery, so a single-colour outline is invisible against one or the
    other. A pale wide line beneath a narrow dark one reads on both, and
    is what a paper forest map does with a boundary anyway.
    """
    if not features:
        return []
    style = boundaries.LEVEL_STYLE[level]
    collection = {"type": "FeatureCollection", "features": features}
    # The casing doubles the payload, because deck.gl serialises the
    # geometry once per layer. Worth it for a dozen reserve outlines,
    # not for 1,274 beats -- the admin levels take a single line and a
    # colour picked to survive both a pale and a dark basemap.
    cased = level == boundaries.RESERVE

    def layer(suffix: str, colour: list, width: float, pickable: bool) -> pdk.Layer:
        return pdk.Layer(
            "GeoJsonLayer",
            data=collection,
            id=f"boundary-{level}-{suffix}",
            stroked=True,
            filled=False,
            get_line_color=colour,
            get_line_width=width,
            # Quoted deliberately. pydeck turns a bare string prop into a
            # deck.gl accessor -- "pixels" became "@@=pixels", a function
            # where a unit belongs, and the width it produced rendered
            # the outlines as rounded blobs or not at all. Quotes are
            # pydeck's escape hatch for a literal.
            line_width_units='"pixels"',
            line_width_min_pixels=width,
            line_joint_rounded=True,
            pickable=pickable,
            auto_highlight=pickable,
            # Default highlight floods the whole polygon; a division
            # filling the screen on hover buries the data on top of it.
            highlight_color=[255, 255, 255, 30],
        )

    for feature in features:
        # Onto the feature itself, not its properties: the interpolator
        # looks at the hovered object first and falls back to properties,
        # and the outline layers carry nothing else at the top level.
        feature.update(_boundary_tooltip(level, feature["properties"]))

    drawn = []
    if cased:
        drawn.append(layer("casing", [255, 255, 255, 150], style["width"] + 2.2, False))
    drawn.append(layer("line", style["color"] + [230], style["width"], True))
    return drawn


def _boundary_tooltip(level: str, properties: dict) -> Dict[str, str]:
    """Slot values for one boundary polygon."""
    rows: List[Tuple[str, object]] = []
    if properties.get("Reports"):
        rows.append(("Reports", f"{int(properties['Reports']):,}"))
    if properties.get("Conflict Events") is not None:
        rows.append(("Conflicts", f"{int(properties['Conflict Events']):,}"))
    if properties.get("Human Deaths"):
        rows.append(("Killed", _count(properties.get("Human Deaths"))))
    if properties.get("Priority Tier"):
        rows.append(("Tier", properties["Priority Tier"]))

    parent = _trail(properties.get("parent"), properties.get("grandparent"))
    return _slots(
        _clean(properties.get("name"), boundaries.LEVEL_LABELS[level]),
        tuple(boundaries.LEVEL_STYLE[level]["color"]),
        rows,
        subtitle=boundaries.LEVEL_LABELS[level],
        footer=parent,
    )


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
    plot_df["_radius"] = _severity_to_radius(plot_df).round(1)

    for col in ["Division", "Range", "Beat", "Nearest Village"]:
        if col not in plot_df.columns:
            plot_df[col] = "N/A"

    plot_df["_group"] = classify_group(plot_df)
    return plot_df.join(
        pd.DataFrame(_sighting_tooltips(plot_df), index=plot_df.index)
    )


def _sighting_tooltips(plot_df: pd.DataFrame) -> List[Dict[str, str]]:
    """What a manager needs off one hover: what, when, who, where."""
    date = (
        plot_df["Date"].dt.strftime("%d %b %Y")
        if "Date" in plot_df.columns
        else pd.Series("", index=plot_df.index)
    )
    hour = plot_df["Hour"] if "Hour" in plot_df.columns else pd.Series(pd.NA, index=plot_df.index)

    tooltips = []
    for i, row in enumerate(plot_df.to_dict("records")):
        when = _clean(date.iloc[i], "")
        clock = hour.iloc[i]
        if pd.notna(clock):
            when = f"{when}, {int(clock):02d}:00" if when else f"{int(clock):02d}:00"

        total = row.get("Total Count")
        group = _clean(row.get("_group"))
        if pd.notna(total) and float(total) > 0:
            group = f"{group} ({int(float(total))})"

        rows: List[Tuple[str, object]] = [("Group", group)]
        damage = [
            name for name, column in (
                ("crop", "Crop Damage"), ("grain", "Grain Damage"),
                ("house", "House Damage"),
            )
            if float(row.get(column) or 0) > 0
        ]
        if damage:
            rows.append(("Damage", ", ".join(damage)))
        for label, column in (("Killed", "Death"), ("Injured", "Injury")):
            if float(row.get(column) or 0) > 0:
                rows.append((label, "yes"))

        village = _clean(row.get("Nearest Village"))
        distance = row.get("Distance to Village (km)")
        if village != _DASH and pd.notna(distance):
            village = f"{village}, {float(distance):.1f} km"
        rows.append(("Nearest village", village))

        tooltips.append(_slots(
            _clean(row.get("_category_label"), "Sighting"),
            CATEGORY_COLORS.get(row.get("_category"), (120, 120, 120)),
            rows,
            subtitle=when,
            footer=_trail(*(row.get(c) for c in ("Beat", "Range", "Division"))),
        ))
    return tooltips


def _sighting_layer(plot_df: pd.DataFrame) -> pdk.Layer:
    # Ship only what the layer and tooltip use: the frame is serialised to
    # JSON for the browser, and timestamps do not survive that round trip.
    columns = [
        "Longitude", "Latitude", "_radius", "_r", "_g", "_b", *TOOLTIP_FIELDS,
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
    frame = frame.join(pd.DataFrame([
        _slots(
            _clean(row.get("Hotspot"), "Hotspot"),
            TIER_COLORS.get(str(row.get("Tier")), (107, 133, 120)),
            [
                ("Sightings", f"{int(row.get('Sightings') or 0):,}"),
                ("Conflicts", f"{int(row.get('Conflict Events') or 0):,} "
                              f"({float(row.get('Conflict Share %') or 0):.0f}%)"),
                ("Killed", _count(row.get("Human Deaths"))),
                ("Injured", _count(row.get("People Injured"))),
                # No radius row: the footprint is drawn at true scale,
                # so the ring on the map already states it.
                ("Night", _pct(row.get("Night Share %"))),
            ],
            # Tier in words as well as in the header colour, and the
            # date range folded in beside it rather than taking a row.
            subtitle=_trail(f"{_clean(row.get('Tier'))} hotspot", _seen_between(row)),
            footer=_trail(row.get("Beats"), row.get("Divisions")),
        )
        for row in frame.to_dict("records")
    ], index=frame.index))

    columns = [
        "Centre Longitude", "Centre Latitude", "_radius_m",
        "_r", "_g", "_b", *TOOLTIP_FIELDS,
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
    frame["_radius_m"] = (base + (events / max_events) ** 0.5 * (base * 3)).round(1)

    frame = frame.join(pd.DataFrame([
        _slots(
            _clean(row.get("Village"), "Village"),
            TIER_COLORS.get(str(row.get("Tier")), (107, 133, 120)),
            [
                ("Conflicts", f"{int(row.get('Conflict Events') or 0):,}"),
                ("Killed", _count(row.get("Human Deaths"))),
                ("Injured", _count(row.get("People Injured"))),
                ("House", _count(row.get("House Damage Events"))),
                ("Crop", _count(row.get("Crop Damage Events"))),
                ("Night", _pct(row.get("Night Share %"))),
            ],
            subtitle=f"{_clean(row.get('Tier'))} tier",
            footer=_hotspot_relation(row),
        )
        for row in frame.to_dict("records")
    ], index=frame.index))

    columns = ["Longitude", "Latitude", "_radius_m", "_r", "_g", "_b", *TOOLTIP_FIELDS]
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
