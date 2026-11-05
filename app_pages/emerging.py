"""New or growing? Emerging hotspot patterns on a hexagon grid."""

from __future__ import annotations

import logging

import streamlit as st

from app_pages.common import _plot
from core import charts, spacetime
from core.map_engine import MAP_MODES, render_pattern_map
from core.ui import map_legend


logger = logging.getLogger(__name__)


def render(c) -> None:
    basemap = c.basemap
    boundary_level = c.boundary_level
    cube = c.cube
    cube_cell_km = c.cube_cell_km
    cube_fdr = c.cube_fdr
    cube_metric = c.cube_metric
    cube_step = c.cube_step
    emerging = c.emerging
    layer_boundaries = c.layer_boundaries

    st.caption(
        f"Reports binned into {cube_cell_km:g} km hexagons by {cube_step.lower()}, then each "
        "place tested for whether it is a hotspot at each step and whether that is "
        "changing. Settings are on the Analyst page."
    )
    counts = spacetime.pattern_counts(emerging)
    if emerging.empty:
        for note in spacetime.pattern_caveats(cube, emerging, cube_fdr):
            st.info(note)
    else:
        shown_patterns = [p for p in spacetime.PATTERN_ORDER[:-1] if counts.get(p)]
        if shown_patterns:
            for col, pattern in zip(st.columns(len(shown_patterns)), shown_patterns):
                col.metric(pattern, f"{counts[pattern]:,}", border=True,
                           help=spacetime.PATTERN_MEANING[pattern])
        map_legend("Pattern", {p: {"accent": spacetime.PATTERN_COLORS[p], "glyph": "",
                                   "label": p} for p in shown_patterns})

        named = emerging[emerging["Place"] != ""]
        cube_hot = spacetime.hot_bins(cube, fdr=cube_fdr)
        table_col, map_col = st.columns([1, 1], gap="medium")
        with table_col:
            if named.empty:
                st.info("No place is a significant hotspot in any step. Try a larger cell, "
                        "a longer step, or switch the correction off.")
            else:
                date_fmt = "MMM YYYY" if cube_step == "Month" else "D MMM YYYY"
                st.dataframe(
                    named.drop(columns=["Cell", "Centre Latitude", "Centre Longitude"]),
                    width="stretch", height=min(520, 38 + 35 * len(named)), hide_index=True,
                    column_config={
                        "Recent Events": st.column_config.NumberColumn(
                            "Recent", help="Events in the last three steps."),
                        "Hot Steps": st.column_config.NumberColumn(
                            "Hot steps", help=f"{cube_step}s in which this place was a "
                                              "significant hotspot."),
                        "First Hot": st.column_config.DateColumn("First hot", format=date_fmt),
                        "Last Hot": st.column_config.DateColumn("Last hot", format=date_fmt),
                    },
                )
                st.download_button(
                    "Download emerging hotspots (CSV)",
                    named.to_csv(index=False).encode("utf-8"),
                    "emerging_hotspots.csv", mime="text/csv",
                )
        with map_col:
            mode_col, height_col = st.columns([3, 2], vertical_alignment="bottom")
            map_mode = mode_col.segmented_control(
                "Map", MAP_MODES, default=MAP_MODES[0], key="emerging_map_mode",
                help="Columns raise each place by its events; Stacked builds the cube on the map.",
            ) or MAP_MODES[0]
            height_by = "Events"
            if map_mode == MAP_MODES[1]:
                height_by = height_col.segmented_control(
                    "Height", ["Events", "Recent Events"], default="Events",
                    key="emerging_height",
                    format_func=lambda h: "All events" if h == "Events"
                    else "Last 3 steps",
                ) or "Events"
            render_pattern_map(emerging, cube.cell_km, cube.reference_lat, cube.step,
                               style_name=basemap, show_boundaries=layer_boundaries,
                               boundary_level=boundary_level, mode=map_mode,
                               height_by=height_by, counts=cube.counts,
                               hot=cube_hot)
            if map_mode == MAP_MODES[1]:
                st.caption("Colour is the pattern, height the events. Ctrl-drag (or two "
                           "fingers) to rotate.")
            elif map_mode == MAP_MODES[2]:
                st.caption("Each tower is a place's history, oldest step at the ground and "
                           "latest on top: coloured slabs were hotspot steps, grey ones not. "
                           "Ctrl-drag to rotate.")

        matrix = spacetime.location_time_matrix(cube, emerging)
        if not matrix.empty:
            st.markdown(f"**When each place lit up** -- {cube_metric.lower()} per "
                        f"{cube_step.lower()}, worst places first")
            _plot(charts.place_time_heatmap(matrix, cube_step))
            st.caption("Read along a row to see whether a place is new, steady or fading; "
                       "down a column to see which places were active together.")


        with st.expander("How to read this", expanded=False):
            for pattern in spacetime.PATTERN_ORDER:
                st.markdown(f"- **{pattern}**: {spacetime.PATTERN_MEANING[pattern]}")
            for note in spacetime.pattern_caveats(cube, emerging, cube_fdr):
                st.markdown(f"- {note}")
