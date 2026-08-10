"""Spatial map rendering for the dashboard, built on pydeck/Deck.GL."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st

from core.analytics import classify_conflict
from core.config import (
    MAX_RADIUS_M,
    MIN_RADIUS_M,
    RADIUS_MAX_PIXELS,
    RADIUS_MIN_PIXELS,
)
from core.ui import CATEGORY_STYLE, category_legend


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


# Colour by what happened, not by a severity ramp: anchored on a
# fatality at 100, a ramp renders every non-fatal incident the same
# shade. Palette lives in core.ui so map, legend and report agree, and is
# colourblind-safe. Size carries the same signal in parallel.
CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    key: _hex_to_rgb(style["color"]) for key, style in CATEGORY_STYLE.items()
}
CATEGORY_LABELS = {key: style["label"] for key, style in CATEGORY_STYLE.items()}


# Categories that get drawn larger regardless of severity arithmetic.
EMPHASIS_CATEGORIES = ("Death", "Injury")


def render_map(df: pd.DataFrame) -> None:
    """Render an interactive conflict map, or a friendly message if empty.

    Points are coloured by conflict category (fatality through
    presence-only) and sized by severity with a pixel floor so they stay
    visible at landscape zoom.

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
    plot_df["_category"] = classify_conflict(plot_df)
    plot_df["_category_label"] = plot_df["_category"].map(CATEGORY_LABELS).fillna("Unknown")

    colors = plot_df["_category"].map(CATEGORY_COLORS)
    colors = colors.where(colors.notna(), pd.Series([(120, 120, 120)] * len(plot_df), index=plot_df.index))
    plot_df[["_r", "_g", "_b"]] = pd.DataFrame(colors.tolist(), index=plot_df.index)
    plot_df["_radius"] = _severity_to_radius(plot_df)

    for optional_col in ["Division", "Range", "Beat", "Nearest Village"]:
        if optional_col not in plot_df.columns:
            plot_df[optional_col] = "N/A"

    plot_df["_date"] = (
        plot_df["Date"].dt.strftime("%d %b %Y")
        if "Date" in plot_df.columns
        else "N/A"
    )

    # Only what the layer and tooltip use: the frame is serialised to
    # JSON and shipped to the browser. Timestamps do not survive that
    # round trip, hence the pre-formatted `_date` string.
    layer_df = plot_df[
        [
            "Longitude", "Latitude", "_radius", "_r", "_g", "_b",
            "_category_label", "_date", "Severity Score",
            "Division", "Range", "Beat", "Nearest Village",
        ]
    ]

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=layer_df,
        get_position="[Longitude, Latitude]",
        get_radius="_radius",
        get_fill_color="[_r, _g, _b, 190]",
        get_line_color=[40, 40, 40],
        line_width_min_pixels=1,
        radius_min_pixels=RADIUS_MIN_PIXELS,
        radius_max_pixels=RADIUS_MAX_PIXELS,
        pickable=True,
        auto_highlight=True,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=_adaptive_view_state(plot_df),
        map_style=None,
        tooltip={
            "html": (
                "<b>{_category_label}</b><br/>"
                "<b>Date:</b> {_date} &nbsp; "
                "<b>Severity:</b> {Severity Score}<br/>"
                "<b>Division:</b> {Division} &nbsp; "
                "<b>Range:</b> {Range} &nbsp; "
                "<b>Beat:</b> {Beat}<br/>"
                "<b>Nearest Village:</b> {Nearest Village}"
            ),
            "style": {"backgroundColor": "#1f5f3f", "color": "white"},
        },
    )

    st.pydeck_chart(deck, width="stretch")
    category_legend(plot_df["_category"].value_counts().to_dict())
    st.caption(
        "Casualty incidents are drawn at full size regardless of severity "
        "arithmetic, so the most serious points stay findable at landscape zoom."
    )


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
        emphasised = df["_category"].isin(EMPHASIS_CATEGORIES)
        radius = radius.mask(emphasised, MAX_RADIUS_M)
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
