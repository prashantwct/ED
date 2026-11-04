"""What every page shares: chart helpers, the map pick, and the cached analysis.

The analysis steps here are cached on the frames they read, so moving
between pages reuses them instead of re-ranking 222 beats and 9,000
villages on every click.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import pandas as pd
import streamlit as st

from core import map_select, spacetime
from core.exports import download_menu
from core.hotspots import detect_hotspots, villages_at_risk
from core.intelligence import management_brief

# A raw 0-300 severity slider is unusable once fatalities weigh 100.
SEVERITY_FILTERS = {
    "All reports": 0.0,
    "Conflict only (damage or worse)": 1.0,
    "Property damage or worse": 5.0,
    "Human harm only (injury or death)": 25.0,
    "Fatalities only": 100.0,
}

# Plot colours drawn from the shared tokens so charts, map and report agree.
BRAND = "#1f5f3f"
BRAND_ALT = "#C2570C"

# Hover stays, the toolbar goes: field staff never use it and it covers
# the legend on a narrow screen.
CHART_CONFIG = {"displayModeBar": False}
# The 3D cube is the exception, with two buttons: once rotated there is
# no other way back to the preset, and a rotated view is the one worth
# saving for a report.
CUBE_CONFIG = {
    "displaylogo": False,
    "modeBarButtons": [["resetCameraDefault3d", "toImage"]],
    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "space_time_cube"},
}

MAP_FOCUS_KEY = "map_focus"
# A link that should open the Where to act map on one place carries it as
# ``?focus=beat:<name>`` (or hotspot:, village:).
FOCUS_PARAM = "focus"
FOCUS_KINDS = ("beat", "hotspot", "village")


def _plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)


def _change(recent: float, prior: float, unit: str = "") -> str | None:
    """Signed change for a metric delta, or None when there is no prior."""
    if prior != prior or recent != recent:
        return None
    diff = recent - prior
    if unit == "pt":
        return f"{diff:+.1f} pts"
    if prior > 0:
        return f"{diff:+,.0f} ({diff / prior:+.0%})"
    return f"{diff:+,.0f}"


def _figure(title: str, fig, download_name: str, make_svg, note: Optional[str] = None,
            subtitle: str = "") -> None:
    """A chart under one header row: its title, and its Download menu at the right."""
    head, menu = st.columns([14, 1], vertical_alignment="bottom")
    head.markdown(f"**{title}**" + (f" -- {subtitle}" if subtitle else ""))
    with menu:
        download_menu(download_name, make_svg, compact=True)
    _plot(fig)
    if note:
        st.caption(note)


def _ews():
    """The early-warning coverage subsystem, imported on first use.

    Nothing in it runs until a registry is uploaded and most sessions
    never upload one, so it is not a top-level import. A dashboard that
    cannot rank beats because an optional module failed to load is worse
    than one without the optional module.
    """
    from core import coverage

    return coverage


# ---------------------------------------------------------------------------
# The map pick: a table row, or a link from the Overview, focuses the map
# ---------------------------------------------------------------------------
def _map_pick(event, key: str, kind: str, values: pd.Series, noun: str) -> None:
    """Send the row picked in a table to the spatial view.

    The latest pick in any table wins; clearing a table's pick clears the
    map focus only if that table set it.
    """
    rows = event.selection.rows if event is not None else []
    picked = values.iloc[rows[0]] if rows else None
    previous = st.session_state.get(f"_pick_{key}")
    if picked != previous:
        st.session_state[f"_pick_{key}"] = picked
        if picked is not None:
            st.session_state[MAP_FOCUS_KEY] = (kind, picked)
        elif st.session_state.get(MAP_FOCUS_KEY, (None, None))[0] == kind:
            st.session_state.pop(MAP_FOCUS_KEY, None)


def focus_map_on(kind: str, value: object) -> None:
    """Point the Where to act map at one beat, hotspot or village.

    Clears the tables' remembered picks too: the tables come back with
    nothing ticked, and a stale pick would read as "unticked" and undo
    the focus on arrival.
    """
    for key in [k for k in st.session_state if str(k).startswith("_pick_")]:
        st.session_state[key] = None
    st.session_state[MAP_FOCUS_KEY] = (kind, value)


def _map_focus(frame: pd.DataFrame, hotspot_table: pd.DataFrame,
               village_table: pd.DataFrame, village_km: float) -> Optional[dict]:
    """The focus the spatial view should draw, from the latest table pick."""
    picked = st.session_state.get(MAP_FOCUS_KEY)
    if not picked:
        return None
    return map_select.focus_for(picked[0], picked[1], frame, hotspot_table,
                                village_table, village_km)


# ---------------------------------------------------------------------------
# Cached analysis
# ---------------------------------------------------------------------------
def _frame_key(frame: pd.DataFrame) -> str:
    """Cheap, stable cache key for a DataFrame.

    The coordinates, dates, severity and row count identify a selection
    without pickling the whole frame on every rerun just to build a key.
    """
    columns = [c for c in ("Latitude", "Longitude", "Date", "Severity Score",
                           "Centre Latitude", "Centre Longitude", "Radius (km)")
               if c in frame.columns]
    digest = hashlib.blake2b(digest_size=16)
    if columns:
        for column in columns:
            digest.update(frame[column].to_numpy().tobytes())
    else:  # a small table with none of those: hash all of it
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest() + f":{len(frame)}:{','.join(map(str, frame.columns))}"


_FRAMES = {pd.DataFrame: _frame_key}


@st.cache_data(show_spinner="Finding movement hotspots...", hash_funcs=_FRAMES, max_entries=16)
def _hotspots(frame: pd.DataFrame, eps_km: float, min_samples: int, as_of):
    """Cached hotspot detection."""
    return detect_hotspots(frame, eps_km=eps_km, min_samples=min_samples, as_of=as_of)


@st.cache_data(show_spinner="Building the space-time cube...", hash_funcs=_FRAMES,
               max_entries=16)
def _emerging(frame: pd.DataFrame, cell_km: float, step: str, metric: str, fdr: bool,
              as_of, centroids: Optional[pd.DataFrame]):
    """Cached cube and emerging hotspot labels for the current filters."""
    cube = spacetime.build_cube(frame, cell_km=cell_km, step=step, metric=metric, as_of=as_of)
    return cube, spacetime.emerging_hotspots(cube, fdr=fdr, villages=centroids)


@st.cache_data(show_spinner="Ranking beats...", hash_funcs=_FRAMES, max_entries=16)
def brief_for(frame: pd.DataFrame, recent_days: int, as_of):
    """Cached beat ranking and findings."""
    return management_brief(frame, recent_days=recent_days, as_of=as_of)


@st.cache_data(show_spinner="Ranking villages...", hash_funcs=_FRAMES, max_entries=16)
def villages_for(frame: pd.DataFrame, villages: Optional[pd.DataFrame],
                 hotspots: pd.DataFrame, radius_km: float, as_of):
    """Cached village ranking: the slowest step on a rerun."""
    return villages_at_risk(frame, villages, hotspots, radius_km=radius_km, as_of=as_of)
