"""Spatial map rendering for the dashboard, built on pydeck/Deck.GL."""

from __future__ import annotations

import math
from typing import List, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st

# Green -> yellow -> red, used to colour points by normalised severity.
_COLOR_STOPS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.0, (46, 168, 82)),
    (0.5, (240, 200, 30)),
    (1.0, (214, 39, 40)),
]

MIN_RADIUS_M = 60
MAX_RADIUS_M = 500


def render_map(df: pd.DataFrame) -> None:
    """Render an interactive severity map, or a friendly message if empty.

    Points are colour-scaled green -> yellow -> red by normalised
    ``Severity Score``, sized the same way, and the initial view adapts
    its zoom level to the geographic spread of the filtered data.

    Args:
        df: Filtered/enriched dataframe with ``Latitude``, ``Longitude``,
            and ``Severity Score`` columns.
    """
    if df.empty:
        st.info("No data available to display on the map with the current filters.")
        return

    if df["Latitude"].isna().all() or df["Longitude"].isna().all():
        st.warning("No valid coordinates available to plot.")
        return

    plot_df = df.copy()
    max_severity = float(plot_df["Severity Score"].max())
    colors = plot_df["Severity Score"].apply(lambda s: _severity_to_color(s, max_severity))
    plot_df[["_r", "_g", "_b"]] = pd.DataFrame(colors.tolist(), index=plot_df.index)
    plot_df["_radius"] = _severity_to_radius(plot_df["Severity Score"])

    for optional_col in ["Division", "Range", "Beat", "Nearest Village"]:
        if optional_col not in plot_df.columns:
            plot_df[optional_col] = "N/A"

    plot_df["_crop"] = _flag_label(plot_df, "Crop Damage")
    plot_df["_house"] = _flag_label(plot_df, "House Damage")
    plot_df["_injury"] = _flag_label(plot_df, "Injury")

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[Longitude, Latitude]",
        get_radius="_radius",
        get_fill_color="[_r, _g, _b, 180]",
        get_line_color=[40, 40, 40],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )

    view_state = _adaptive_view_state(plot_df)

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=None,
        tooltip={
            "html": (
                "<b>Severity:</b> {Severity Score}<br/>"
                "<b>Division:</b> {Division} &nbsp; "
                "<b>Range:</b> {Range} &nbsp; "
                "<b>Beat:</b> {Beat}<br/>"
                "<b>Nearest Village:</b> {Nearest Village}<br/>"
                "<b>Crop damage:</b> {_crop} &nbsp; "
                "<b>House damage:</b> {_house} &nbsp; "
                "<b>Injury:</b> {_injury}"
            ),
            "style": {"backgroundColor": "steelblue", "color": "white"},
        },
    )

    st.pydeck_chart(deck, use_container_width=True)
    st.caption("🟢 Low severity → 🟡 Moderate → 🔴 High severity")


def _flag_label(df: pd.DataFrame, column: str) -> pd.Series:
    """Return 'Yes'/'No' labels for an optional boolean-ish flag column."""
    if column not in df.columns:
        return pd.Series("N/A", index=df.index)
    return (pd.to_numeric(df[column], errors="coerce").fillna(0) > 0).map({True: "Yes", False: "No"})


def _severity_to_color(score: float, max_score: float) -> Tuple[int, int, int]:
    """Map a raw severity score to an RGB colour via green-yellow-red stops.

    Args:
        score: This row's severity score.
        max_score: The maximum severity score across the plotted data,
            used to normalise ``score`` into a 0-1 range.
    """
    normalised = 0.0 if max_score <= 0 else min(max(score / max_score, 0.0), 1.0)

    for (lo_t, lo_c), (hi_t, hi_c) in zip(_COLOR_STOPS, _COLOR_STOPS[1:]):
        if lo_t <= normalised <= hi_t:
            span = hi_t - lo_t or 1.0
            frac = (normalised - lo_t) / span
            return tuple(int(lo_c[i] + (hi_c[i] - lo_c[i]) * frac) for i in range(3))
    return _COLOR_STOPS[-1][1]


def _severity_to_radius(scores: pd.Series) -> pd.Series:
    """Scale severity scores into a reasonable pixel-radius range (metres)."""
    max_score = scores.max()
    if max_score <= 0:
        return pd.Series(MIN_RADIUS_M, index=scores.index)
    normalised = (scores / max_score).clip(lower=0, upper=1)
    return MIN_RADIUS_M + normalised * (MAX_RADIUS_M - MIN_RADIUS_M)


def _adaptive_view_state(df: pd.DataFrame) -> pdk.ViewState:
    """Pick a centre and zoom level that fits all points, with sane bounds."""
    lat_min, lat_max = df["Latitude"].min(), df["Latitude"].max()
    lon_min, lon_max = df["Longitude"].min(), df["Longitude"].max()

    lat_span = max(lat_max - lat_min, 0.01)
    lon_span = max(lon_max - lon_min, 0.01)
    span = max(lat_span, lon_span)

    # Rough heuristic: halve zoom level for each doubling of angular span,
    # anchored so a ~0.05 degree spread (a couple of km) reads as zoom 12.
    zoom = 12 - math.log2(max(span / 0.05, 1))
    zoom = min(max(zoom, 5), 14)

    return pdk.ViewState(
        latitude=float((lat_min + lat_max) / 2),
        longitude=float((lon_min + lon_max) / 2),
        zoom=zoom,
        pitch=0,
    )

