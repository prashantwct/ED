"""Analyst: the model settings, the 3D space-time cube, and diagnostics.

For whoever tunes the analysis. The standard settings are what every
officer should see; anything changed here is named in a badge on every
page and in the brief's scope line (core.analysis_settings), and *Reset
to standard* puts them all back. The values survive moving between
pages: the entry point writes them back to session state on every run.
"""

from __future__ import annotations

import streamlit as st

from app_pages import cube
from core import analysis_settings as cfg
from core import spacetime
from core.intelligence import CRITICAL_CASUALTY_WINDOW_DAYS
from core.map_engine import basemap_warning
from core.ui import section


def _reset() -> None:
    for setting in cfg.SETTINGS:
        st.session_state[setting.key] = setting.default


def render(c) -> None:
    section("assessment", "Analysis settings",
            "The standard settings suit most reviews; changes are named on every page "
            "and in the brief")
    rank_col, place_col = st.columns(2, gap="large")
    with rank_col:
        st.markdown("**Ranking**")
        st.slider(
            "Compare the last ... days", min_value=30, max_value=180, step=30,
            key=cfg.RECENT_DAYS.key,
            help="Trends compare these days with the same days a year earlier. Critical "
            f"always uses {CRITICAL_CASUALTY_WINDOW_DAYS} days.",
        )
        st.slider(
            "Count incidents within ... km of a village", min_value=1.0, max_value=10.0,
            step=0.5, key=cfg.VILLAGE_RADIUS.key,
            help="Incidents this close to a village count towards its risk (the village "
            "exposure radius).",
        )
        st.slider(
            "Merge activity closer than ... km", min_value=0.5, max_value=3.0, step=0.25,
            key=cfg.EPS_KM.key,
            help="Reports this close together join one hotspot (the DBSCAN neighbour "
            "distance).",
        )
        st.slider(
            "Smallest hotspot (reports)", min_value=5, max_value=40, step=5,
            key=cfg.MIN_SAMPLES.key,
            help="A cluster needs at least this many reports to count as a hotspot.",
        )
    with place_col:
        st.markdown("**New or growing places**")
        st.selectbox(
            "Time step", list(spacetime.STEPS), key=cfg.CUBE_STEP.key,
            help="Shorter steps react faster but need more reports per place.",
        )
        st.slider(
            "Hexagon size (km)", min_value=1.0, max_value=5.0, step=0.5,
            key=cfg.CUBE_CELL_KM.key, help="Distance between neighbouring hexagon centres.",
        )
        st.radio(
            "Count", spacetime.METRICS, horizontal=True, key=cfg.CUBE_METRIC.key,
            help="All reports adds plain sightings, which mostly track where people report.",
        )
        st.checkbox(
            "Strict test (fewer false alarms)", key=cfg.CUBE_FDR.key,
            help="Corrects for testing many places at once (false discovery rate). Off "
            "finds more, some of them by chance.",
        )
    changed = cfg.non_standard(st.session_state)
    st.button("Reset to standard", on_click=_reset, disabled=not changed,
              icon=":material/restart_alt:")
    if changed:
        st.caption("Changed from standard: " + "; ".join(changed) + ".")

    section("hotspot", "The space-time cube",
            "The ground on two axes, time rising on the third")
    cube.render(c)

    section("map", "Diagnostics", "What the app could and could not reach")
    note = basemap_warning()
    st.markdown(f"- Basemap: {note if note else 'MapTiler key set; all basemaps available.'}")
    st.markdown(f"- Village centroids: {len(c.villages):,} loaded."
                if c.villages is not None else "- Village centroids: none loaded.")
    st.markdown(f"- Register: {len(c.df):,} valid reports, "
                f"{c.register_start.day} {c.register_start:%b %Y} to "
                f"{c.register_end.day} {c.register_end:%b %Y}.")
