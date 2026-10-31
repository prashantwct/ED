"""Elephant Conflict Intelligence -- Streamlit entry point.

All data logic lives in core/; this file handles layout, widgets and
wiring only. The headline numbers and assessment sit on top; the rest is
one tab per question a manager asks: is it getting worse, where to act,
which villages, when, and the data behind it.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
import time
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# Set on each of the app's modules when this process first sees it loaded.
_LOADED_AT = "_ed_loaded_at"
# The app's own packages: core/ for the page, tools/ for the Forest
# Alerts client the fetch imports from.
_OUR_PACKAGES = ("core", "tools")


def _is_core(name: str) -> bool:
    return name.split(".", 1)[0] in _OUR_PACKAGES


def _forget_stale_core_modules() -> bool:
    """Drop all the app's modules if any has changed on disk since it was loaded.

    Streamlit re-reads this file on every run but keeps imported modules
    in memory. After a deploy that pairs a new call here with a new
    argument there, the page calls the old function and dies with a
    TypeError until the app is rebooted. Forgetting the modules makes the
    imports below load the current code, as Streamlit's own watcher does
    in local development. All of them go, not just the changed one,
    because the others hold names imported from it -- core.fetch from
    tools.fetch_sightings, which is why tools/ is watched too.
    """
    stale = False
    for name, module in list(sys.modules.items()):
        loaded = getattr(module, _LOADED_AT, None) if _is_core(name) else None
        path = getattr(module, "__file__", None)
        if loaded is not None and path:
            try:
                stale = os.path.getmtime(path) > loaded
            except OSError:  # deleted by the deploy: certainly stale
                stale = True
            if stale:
                break
    if stale:
        for name in [n for n in sys.modules if _is_core(n)]:
            del sys.modules[name]
    return stale


def _stamp_core_modules() -> None:
    now = time.time()
    for name, module in list(sys.modules.items()):
        if _is_core(name) and getattr(module, _LOADED_AT, None) is None:
            setattr(module, _LOADED_AT, now)


_reloaded_core = _forget_stale_core_modules()

from core.analytics import (
    compute_is_night,
    compute_kpis,
    compute_severity,
    division_conflict_rate,
    division_monthly_conflict,
    filter_dataframe,
    monthly_conflict_breakdown,
    monthly_trend,
    night_day_comparison,
    seasonal_matrix,
    severity_distribution,
    window_comparison,
)
from core import boundaries, charts, landing, map_select, report_charts, spacetime
from core.exports import download_menu
from core.config import (
    DEFAULT_CUBE_CELL_KM,
    DEFAULT_CUBE_STEP,
    LOG_PATH,
    NIGHT_HOUR_END,
    NIGHT_HOUR_START,
)
from core.data_loader import load_and_validate_csv
from core.exceptions import DataValidationError, SpatialEnrichmentError
from core.hotspots import (
    DEFAULT_EPS_KM,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_VILLAGE_RADIUS_KM,
    detect_hotspots,
    hotspot_caveats,
    village_caveats,
    villages_at_risk,
)
from core.intelligence import (
    CRITICAL_CASUALTY_WINDOW_DAYS,
    DEFAULT_RECENT_DAYS,
    TIER_CRITICAL,
    TIER_HIGH,
    TIER_ORDER,
    format_window,
    management_brief,
)
from core.map_engine import (
    BASEMAP_STYLES,
    DEFAULT_BASEMAP,
    MAP_MODES,
    render_map,
    render_pattern_map,
    render_village_map,
)
from core.map_export import sightings_map_svg, village_map_svg
from core.report import generate_html_report
from core.spatial import attach_nearest_village, load_village_centroids
from core.ui import (
    CATEGORY_STYLE,
    TIER_STYLE,
    findings,
    inject_theme,
    map_legend,
    section,
    tier_summary,
)

_stamp_core_modules()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
if _reloaded_core:
    logger.info("Core modules changed on disk since they were loaded; reloaded them.")

st.set_page_config(
    page_title="Elephant Conflict Intelligence",
    layout="wide",
    page_icon="🐘",  # browser tab only; in-page icons are SVG
)
inject_theme()

# A raw 0-300 severity slider is unusable once fatalities weigh 100.
SEVERITY_FILTERS = {
    "All reports": 0.0,
    "Conflict only (damage or worse)": 1.0,
    "Property damage or worse": 5.0,
    "Human harm only (injury or death)": 25.0,
    "Fatalities only": 100.0,
}

FILTER_KEYS = ["flt_dates", "flt_divisions", "flt_ranges", "flt_beats", "flt_severity"]

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


def _frame_key(frame: pd.DataFrame) -> str:
    """Cheap, stable cache key for a DataFrame.

    Hashing the coordinates and row count is enough to identify the
    filtered selection, and avoids pickling the whole frame on every
    rerun just to build a key.
    """
    coords = frame[["Latitude", "Longitude"]].to_numpy().tobytes()
    return hashlib.blake2b(coords, digest_size=16).hexdigest() + f":{len(frame)}"


@st.cache_data(show_spinner="Loading and validating the export...")
def _load(file_bytes: bytes, filename: str):
    """Parse an uploaded CSV. Cached on content, so filters don't re-parse."""
    return load_and_validate_csv(io.BytesIO(file_bytes))


def _ews():
    """The early-warning coverage subsystem, imported on first use.

    Nothing in it runs until a registry is uploaded and most sessions
    never upload one, so it is not a top-level import. A dashboard that
    cannot rank beats because an optional module failed to load is worse
    than one without the optional module.
    """
    from core import coverage

    return coverage


@st.cache_data(show_spinner="Reading the early-warning registry...")
def _registry(file_bytes: bytes, filename: str):
    """De-identify the registry once per upload, not once per rerun."""
    return _ews().load_ews_registry(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Reading the registered user list...")
def _roster(file_bytes: bytes, filename: str):
    return _ews().load_staff_roster(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Finding movement hotspots...", hash_funcs={pd.DataFrame: _frame_key})
def _hotspots(frame: pd.DataFrame, eps_km: float, min_samples: int, as_of):
    """Cached hotspot detection -- the slowest step on a rerun."""
    return detect_hotspots(frame, eps_km=eps_km, min_samples=min_samples, as_of=as_of)


@st.cache_data(show_spinner="Building the space-time cube...",
               hash_funcs={pd.DataFrame: _frame_key})
def _emerging(frame: pd.DataFrame, cell_km: float, step: str, metric: str, fdr: bool,
              as_of, centroids: Optional[pd.DataFrame]):
    """Cached cube and emerging hotspot labels for the current filters."""
    cube = spacetime.build_cube(frame, cell_km=cell_km, step=step, metric=metric, as_of=as_of)
    return cube, spacetime.emerging_hotspots(cube, fdr=fdr, villages=centroids)


def _fetch(request: landing.FetchRequest):
    """Run a Forest Alerts pull behind a status panel.

    Deliberately not cached. A Streamlit cache keyed on this call would
    be a cache keyed on a password; the CSV it returns goes through the
    same cached ``_load`` as an upload instead.
    """
    from core.fetch import SOURCE_EXPORT, FetchError, fetch_sightings

    with st.status("Signing in to Forest Alerts...", expanded=True) as status:
        def progress(page: int, rows: int) -> None:
            status.update(label=f"Reading page {page} -- {rows:,} row(s) so far")

        def stage(text: str) -> None:
            status.update(label=text)

        try:
            result = fetch_sightings(
                email=request.email,
                password=request.password,
                start_date=request.start_date,
                end_date=request.end_date,
                today=request.today,
                export_path=_setting("FORESTALERTS_EXPORT_PATH"),
                progress=progress,
                stage=stage,
            )
        except FetchError as exc:
            status.update(label="Could not fetch the register", state="error")
            # The diagnosis is several lines of page report; markdown
            # would run them together, so it goes in a block.
            headline, _, detail = str(exc).partition("\n")
            st.error(headline)
            if detail.strip():
                st.code(detail.strip())
            return None
        if result.source == SOURCE_EXPORT:
            label = f"Fetched {result.rows:,} row(s) from the export"
        else:
            label = (f"Fetched {result.rows:,} row(s) from {result.pages} page(s) "
                     "-- no export found")
            st.warning(result.note)
        status.update(label=label, state="complete")
    return result


def _setting(name: str) -> Optional[str]:
    """An optional setting, from Streamlit secrets or the environment."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:  # noqa: BLE001 - no secrets file is a normal state
        pass
    return os.getenv(name) or None


# ---------------------------------------------------------------------------
# 1. Upload, or fetch, & load
# ---------------------------------------------------------------------------
# A fetch survives reruns under this key; nothing else about it does.
FETCHED_KEY = "fetched_register"
# The last register that loaded, so the damage view can cross-check
# against it without the upload widget being on its page.
REGISTER_KEY = "loaded_register"

# Two views over two kinds of record. The conflict dashboard reads the
# Gaj Rakshak sighting register; the damage view reads the household
# damage surveys. Different questions, so different pages.
VIEW_KEY = "view"
CONFLICT_VIEW, DAMAGE_VIEW = "Conflict dashboard", "Damage & compensation"
REFUGE_VIEW = "Day refuges"
view = st.sidebar.radio("View", [CONFLICT_VIEW, REFUGE_VIEW, DAMAGE_VIEW], key=VIEW_KEY)


def _stashed_register() -> Optional[pd.DataFrame]:
    """The register last loaded in the conflict view, if any."""
    stash = st.session_state.get(REGISTER_KEY)
    if stash is None:
        return None
    try:
        return _load(*stash)[0]
    except DataValidationError:
        return None


if view == REFUGE_VIEW:
    from core import refuge_view

    landing.header()
    refuge_register = _stashed_register()
    try:
        refuge_villages, _ = load_village_centroids(None)
    except SpatialEnrichmentError:
        refuge_villages = None
    if refuge_register is not None:
        refuge_register, _ = attach_nearest_village(refuge_register, refuge_villages)
        chosen = st.sidebar.multiselect(
            "Division", sorted(refuge_register["Division"].unique()), key="refuge_divisions",
            help="Leave empty for the whole register.")
        if chosen:
            refuge_register = refuge_register[refuge_register["Division"].isin(chosen)]
    refuge_basemap = st.sidebar.selectbox(
        "Basemap", list(BASEMAP_STYLES),
        index=list(BASEMAP_STYLES).index(DEFAULT_BASEMAP), key="refuge_basemap",
    )
    refuge_view.render(refuge_register, refuge_villages, refuge_basemap)
    st.stop()

if view == DAMAGE_VIEW:
    from core import damage_view

    landing.header()
    damage_df, damage_notes = damage_view.sources()
    if damage_df is None:
        st.info("Upload a herd-crop-damage or herd-house-damage export to begin.")
        st.stop()
    register_df = _stashed_register()
    damage_basemap = st.sidebar.selectbox(
        "Basemap", list(BASEMAP_STYLES),
        index=list(BASEMAP_STYLES).index(DEFAULT_BASEMAP), key="damage_basemap",
    )
    damage_view.render(damage_df, damage_notes, register_df, damage_basemap)
    st.stop()


MAP_FOCUS_KEY = "map_focus"


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


def _map_focus(frame: pd.DataFrame, hotspot_table: pd.DataFrame,
               village_table: pd.DataFrame, village_km: float) -> Optional[dict]:
    """The focus the spatial view should draw, from the latest table pick."""
    picked = st.session_state.get(MAP_FOCUS_KEY)
    if not picked:
        return None
    return map_select.focus_for(picked[0], picked[1], frame, hotspot_table,
                                village_table, village_km)


def _open_damage_view() -> None:
    st.session_state[VIEW_KEY] = DAMAGE_VIEW

# The masthead sits above the uploader but depends on whether a file
# arrived, which is only known after the widget is created. A container
# reserves the slot now and is filled once that is settled.
masthead = st.container()
uploaded = st.file_uploader(
    "Upload the Gajrakshak sightings export (CSV)", type="csv"
)
fetched = st.session_state.get(FETCHED_KEY)

with masthead:
    if uploaded or fetched:
        landing.header()
    else:
        landing.hero()

if not uploaded and not fetched:
    request = landing.fetch_panel()
    if request is not None:
        refusal = request.is_usable()
        if refusal:
            st.warning(refusal)
        else:
            result = _fetch(request)
            if result is not None:
                st.session_state[FETCHED_KEY] = result
                st.rerun()
    st.button(
        "Crop and house damage surveys (Epicollect5)",
        help="Losses, compensation and the Gaj Rakshak gap, from the damage surveys.",
        on_click=_open_damage_view,
    )
    landing.details()
    st.stop()

# A fetch is the source until it is cleared, and an upload overrides it.
# The escape hatch is drawn before the load so that a fetch the loader
# rejects does not strand the session on an error with no way back.
if fetched:
    source, clear = st.columns([6, 1])
    with source:
        if uploaded:
            landing.source_note(
                "Uploaded file",
                f"Using {uploaded.name}. The register fetched on "
                f"{fetched.fetched_at} is still held -- remove the upload to "
                f"go back to it.",
            )
        else:
            # A fetch held from before this field existed has no source.
            how = ("from the export" if getattr(fetched, "source", "") == "export"
                   else f"over {fetched.pages} page(s) of the listing")
            landing.source_note(
                "Forest Alerts",
                f"{fetched.rows:,} rows {how}, {fetched.start_date} to "
                f"{fetched.end_date}, fetched {fetched.fetched_at}.",
            )
    with clear:
        if st.button("Clear", help="Discard the fetched register."):
            del st.session_state[FETCHED_KEY]
            st.rerun()

if uploaded:
    source_bytes, source_name = uploaded.getvalue(), uploaded.name
else:
    source_bytes, source_name = fetched.data, fetched.name

try:
    raw_df, load_warnings = _load(source_bytes, source_name)
except DataValidationError as exc:
    st.error(f"Could not load the file: {exc}")
    st.stop()
st.session_state[REGISTER_KEY] = (source_bytes, source_name)

if load_warnings:
    with st.expander(f"{len(load_warnings)} data quality note(s)", expanded=False):
        for warning in load_warnings:
            st.warning(warning)

st.success(f"Loaded {len(raw_df):,} valid rows.")


# ---------------------------------------------------------------------------
# 2. Sidebar: centroids, filters, tuning
# ---------------------------------------------------------------------------
with st.sidebar.expander("Village centroids", expanded=False):
    st.caption(
        "Shahdol-Anuppur centroids ship with the app and load automatically. "
        "Upload only to override them. Columns: Village, Latitude, Longitude."
    )
    centroid_upload = st.file_uploader(
        "Override centroids", type="csv", key="centroids_upload"
    )

with st.sidebar.expander("Early-warning coverage", expanded=False):
    st.caption(
        "Optional. Upload the early-warning villager registry to see which "
        "exposed villages have nobody enrolled, and the app-user list to see "
        "reporting effort against registered staff."
    )
    registry_upload = st.file_uploader(
        "Villager registry (CSV)", type="csv", key="registry_upload",
        help="Columns: Latitude, Longitude, and ideally Village and Division.",
    )
    roster_upload = st.file_uploader(
        "Registered app users (CSV)", type="csv", key="roster_upload",
        help="Columns: Division, and ideally Latitude and Longitude.",
    )
    st.caption(
        "Both files carry personal data. Names, mobile numbers and email "
        "addresses are discarded as the file is read and coordinates are "
        "rounded to about 100 m, so only per-village counts reach the page or "
        "any download. Neither file is stored."
    )

try:
    villages, centroid_warnings = load_village_centroids(
        centroid_upload if centroid_upload else None
    )
except SpatialEnrichmentError as exc:
    st.sidebar.warning(f"Village centroids not applied: {exc}")
    villages, centroid_warnings = None, []

for warning in centroid_warnings:
    st.sidebar.warning(warning)

df, spatial_warnings = attach_nearest_village(raw_df, villages)
for warning in spatial_warnings:
    st.warning(warning)
if villages is not None:
    st.sidebar.success(f"Enriched with {len(villages):,} village centroids.")

registry = roster = None
for source, loader, label in (
    (registry_upload, _registry, "registry"),
    (roster_upload, _roster, "user list"),
):
    if source is None:
        continue
    try:
        loaded, load_notes = loader(source.getvalue(), source.name)
    except DataValidationError as exc:
        st.sidebar.warning(f"Could not read the {label}: {exc}")
        continue
    except ImportError:
        # Stale interpreter after a deploy: the entry point has been
        # reloaded from disk but an already-imported module has not.
        logger.exception("Coverage subsystem failed to import")
        st.sidebar.error(
            "Early-warning coverage could not load. If the app was updated "
            "just now, reboot it so the running process picks up the new "
            "modules. Everything else on this page is unaffected."
        )
        break
    for note in load_notes:
        st.sidebar.caption(note)
    if label == "registry":
        registry = loaded
        st.sidebar.success(f"{len(registry):,} registrations read.")
    else:
        roster = loaded
        st.sidebar.success(f"{len(roster):,} registered users read.")

df = df.copy()
df["Severity Score"] = compute_severity(df)
df["Is_Night"] = compute_is_night(df)

st.sidebar.header("Filters")

if st.sidebar.button("Reset filters"):
    # Clearing the widget keys is what resets them; a bare st.rerun()
    # re-runs with every selection still in session state.
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)
    st.rerun()

min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="flt_dates",
)
if isinstance(date_range, tuple) and len(date_range) != 2:
    date_range = (min_date, max_date)

division_options = sorted(df["Division"].unique())
divisions = st.sidebar.multiselect("Division", division_options, key="flt_divisions")

# Prune stale selections before creating the dependent widget. Widening
# Division back out otherwise leaves Range holding values no longer
# offered, silently excluding rows while the UI looks reset.
range_pool = df[df["Division"].isin(divisions)] if divisions else df
range_options = sorted(range_pool["Range"].unique())
st.session_state["flt_ranges"] = [
    r for r in st.session_state.get("flt_ranges", []) if r in range_options
]
ranges = st.sidebar.multiselect("Range", range_options, key="flt_ranges")

beat_pool = range_pool[range_pool["Range"].isin(ranges)] if ranges else range_pool
beat_options = sorted(beat_pool["Beat"].unique())
st.session_state["flt_beats"] = [
    b for b in st.session_state.get("flt_beats", []) if b in beat_options
]
beats = st.sidebar.multiselect("Beat", beat_options, key="flt_beats")

severity_choice = st.sidebar.selectbox(
    "Minimum incident type",
    list(SEVERITY_FILTERS),
    key="flt_severity",
    help="Filters on the severity score behind each report.",
)

st.sidebar.divider()
st.sidebar.subheader("Analysis settings")
recent_days = st.sidebar.slider(
    "Escalation window (days)",
    min_value=30, max_value=180, value=DEFAULT_RECENT_DAYS, step=30,
    help="Compares the recent N days against the N before. Critical uses a "
    f"fixed {CRITICAL_CASUALTY_WINDOW_DAYS}-day window.",
)
eps_km = st.sidebar.slider(
    "Hotspot neighbour distance (km)",
    min_value=0.5, max_value=3.0, value=DEFAULT_EPS_KM, step=0.25,
    help="Larger values merge nearby activity into one region.",
)
min_samples = st.sidebar.slider(
    "Minimum sightings per hotspot",
    min_value=5, max_value=40, value=DEFAULT_MIN_SAMPLES, step=5,
)
village_radius = st.sidebar.slider(
    "Village exposure radius (km)",
    min_value=1.0, max_value=10.0, value=DEFAULT_VILLAGE_RADIUS_KM, step=0.5,
    help="Incidents within this distance of a village count towards its risk.",
)
with st.sidebar.expander("Space-time trend"):
    cube_step = st.selectbox(
        "Time step", list(spacetime.STEPS), index=list(spacetime.STEPS).index(DEFAULT_CUBE_STEP),
        help="Shorter steps react faster but need more reports per place.",
    )
    cube_cell_km = st.slider(
        "Cell size (km)", min_value=1.0, max_value=5.0, value=DEFAULT_CUBE_CELL_KM, step=0.5,
        help="Distance between neighbouring hexagon centres.",
    )
    cube_metric = st.radio(
        "Count", spacetime.METRICS, horizontal=True,
        help="All reports adds plain sightings, which mostly track where people report.",
    )
    cube_fdr = st.checkbox(
        "Correct for multiple testing", value=True,
        help="False discovery rate correction. Off finds more, some of them by chance.",
    )

st.sidebar.divider()
st.sidebar.subheader("Map")
basemap = st.sidebar.selectbox(
    "Basemap",
    list(BASEMAP_STYLES),
    index=list(BASEMAP_STYLES).index(DEFAULT_BASEMAP),
    help="Terrain and imagery read better than street cartography here.",
)

if boundaries.available():
    layer_boundaries = st.sidebar.checkbox("Layer: forest boundaries", value=True)
    boundary_choice = st.sidebar.selectbox(
        "Boundary level",
        ["Auto (by zoom)"] + [boundaries.LEVEL_LABELS[k] for k in boundaries.LEVELS],
        help="Auto picks the level from the view your filters produce. "
        "Pin one to override it.",
    )
    boundary_level = None if boundary_choice.startswith("Auto") else {
        boundaries.LEVEL_LABELS[k]: k for k in boundaries.LEVELS
    }[boundary_choice]
else:
    layer_boundaries, boundary_level = False, None

filtered = filter_dataframe(
    df,
    date_range=date_range,
    divisions=divisions,
    ranges=ranges,
    beats=beats,
    min_severity=SEVERITY_FILTERS[severity_choice],
)


# ---------------------------------------------------------------------------
# 3. Headline numbers
# ---------------------------------------------------------------------------
kpis = compute_kpis(filtered)

# Anchor casualty recency to the selected period's end, not to whatever
# the current Division/Beat selection happens to end at.
period_end = pd.Timestamp(date_range[1]) if len(date_range) == 2 else df["Date"].max()

monthly = monthly_conflict_breakdown(filtered)
change = window_comparison(filtered, recent_days, as_of=period_end)
window_note = f"vs prior {recent_days}d"


def _spark(column: str):
    """Monthly series behind a headline number, when there is a trend to show."""
    return monthly[column].tolist() if len(monthly) >= 3 else None


# Each headline carries its monthly sparkline and the change over the
# escalation window. Rises in conflict are bad news, so they read red.
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Reports", f"{kpis['entries']:,}", chart_data=_spark("Sightings"),
          chart_type="bar", border=True)
c2.metric(
    "Conflict events",
    f"{kpis['conflicts']:,}",
    delta=_change(change["conflicts_recent"], change["conflicts_prior"]),
    delta_color="inverse",
    delta_description=window_note,
    chart_data=_spark("Conflict Events"),
    chart_type="area",
    border=True,
)
c3.metric(
    "Conflict rate",
    "N/A" if pd.isna(kpis["conflict_rate"]) else f"{kpis['conflict_rate']:.1f}%",
    delta=_change(change["rate_recent"], change["rate_prior"], unit="pt"),
    delta_color="inverse",
    delta_description=window_note,
    chart_data=_spark("Conflict Rate %") if len(monthly) >= 3
    and monthly["Conflict Rate %"].notna().all() else None,
    border=True,
)
c4.metric(
    "Killed / injured",
    f"{int(kpis['human_deaths'])} / {int(kpis['human_injuries'])}",
    delta=_change(change["casualties_recent"], change["casualties_prior"]),
    delta_color="inverse",
    delta_description=window_note,
    chart_data=(monthly["Human Deaths"] + monthly["People Injured"]).tolist()
    if len(monthly) >= 3 else None,
    chart_type="bar",
    border=True,
)
c5.metric(
    "Night share",
    "N/A" if pd.isna(kpis["night_pct"]) else f"{kpis['night_pct']:.1f}%",
    delta=f"of {kpis['night_known']:,} timed reports" if kpis["night_known"] else None,
    delta_color="off",
    border=True,
)

if filtered.empty:
    st.warning("No rows match the current filters. Adjust the filters in the sidebar.")
    st.stop()

brief = management_brief(filtered, recent_days=recent_days, as_of=period_end)

# Everything the tabs share is worked out once, before any tab draws.
hotspots = _hotspots(filtered, eps_km, min_samples, period_end)
village_risk = villages_at_risk(
    filtered, villages, hotspots, radius_km=village_radius, as_of=period_end
)
coverage_table = pd.DataFrame()
coverage_stats: dict = {}
if registry is not None:
    coverage_table, coverage_stats = _ews().village_coverage(
        village_risk, villages, registry
    )
cube, emerging = _emerging(filtered, cube_cell_km, cube_step, cube_metric, cube_fdr,
                           period_end, villages)
beats_table = brief["beats"]
composition = brief["composition"]
escalating = brief["escalating"]
temporal = brief["temporal"]
division_palette = charts.division_colors(df["Division"].unique())


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


# ---------------------------------------------------------------------------
# 4. Assessment, and the brief beside it
# ---------------------------------------------------------------------------
section("assessment", "Assessment", f"{len(filtered):,} reports in the selected period")
emerging_line = spacetime.headline(emerging)
findings(brief["headlines"] + ([emerging_line] if emerging_line else []))

how_col, brief_col = st.columns([3, 1], vertical_alignment="top")
with how_col:
    with st.expander("How to read this", expanded=False):
        for caveat in brief["caveats"]:
            st.markdown(f"- {caveat}")
with brief_col:
    if st.button("Generate brief", type="primary", width="stretch",
                 help="Self-contained HTML with the maps embedded, for printing or email."):
        from core import refuge_view

        try:
            try:
                brief_refuges = refuge_view.for_brief(filtered, villages)
            except Exception:  # noqa: BLE001 - the brief goes out without the section
                logger.exception("Day refuges for the brief failed")
                brief_refuges = None
            html = generate_html_report(
                filtered,
                filtered["Date"].min(),
                filtered["Date"].max(),
                recent_days=recent_days,
                hotspots=hotspots,
                village_risk=village_risk,
                filters={
                    "divisions": divisions,
                    "ranges": ranges,
                    "beats": beats,
                    "severity": severity_choice,
                },
                basemap_style_name=basemap,
                coverage=coverage_table,
                coverage_stats=coverage_stats,
                refuges=brief_refuges,
                emerging=emerging,
                emerging_step=cube_step,
            )
        except Exception as exc:  # noqa: BLE001 - report generation must never 500
            logger.exception("Report generation failed")
            st.error(f"Could not generate the brief: {exc}")
        else:
            st.download_button(
                "Download brief (HTML)", html, "Elephant_Conflict_Brief.html",
                mime="text/html", width="stretch",
            )
            st.success("Brief ready: download it above.")


# ---------------------------------------------------------------------------
# 5. One tab per question a manager asks
# ---------------------------------------------------------------------------
tab_trend, tab_where, tab_emerging, tab_villages, tab_when, tab_data = st.tabs([
    ":material/trending_up: Is it getting worse?",
    ":material/map: Where to act",
    ":material/timeline: New or growing?",
    ":material/home: Villages at risk",
    ":material/schedule: When",
    ":material/table_chart: Data",
])


# ---- Trends ---------------------------------------------------------------
with tab_trend:
    if len(monthly) < 2:
        st.info("The selected period covers less than two months, so there is no trend "
                "to draw.")
    else:
        # Each figure downloads from its SVG twin. Values are bound as
        # default arguments: the callable runs on click, after this script
        # has moved on and reassigned names like ``monthly``.
        _figure(
            "Conflict events per month, by type",
            charts.conflict_type_trend(monthly, recent_days=recent_days, as_of=period_end),
            "Conflict events per month by type",
            lambda m=monthly, d=recent_days, e=period_end:
            report_charts.conflict_type_trend_svg(m, d, e),
            note="Darkest at the base is the most severe; the shaded band is the "
                 "escalation window set in the sidebar.",
        )

        rate_col, casualty_col = st.columns(2)
        with rate_col:
            _figure("Conflict rate", charts.conflict_rate_trend(monthly, kpis["conflict_rate"]),
                    "Conflict rate per month",
                    lambda m=monthly, r=kpis["conflict_rate"]:
                    report_charts.conflict_rate_svg(m, r),
                    subtitle="share of reports that were conflict")
        with casualty_col:
            if float((monthly["Human Deaths"] + monthly["People Injured"]).sum()) == 0:
                st.markdown("**People killed and injured**")
                st.success("No casualties recorded in the selected period.")
            else:
                _figure("People killed and injured", charts.casualty_trend(monthly),
                        "People killed and injured per month",
                        lambda m=monthly: report_charts.casualty_svg(m))

        season_col, division_col = st.columns(2)
        with season_col:
            _figure("Seasonal pattern", charts.seasonal_heatmap(seasonal_matrix(filtered)),
                    "Seasonal pattern",
                    lambda f=filtered: report_charts.seasonal_heatmap_svg(seasonal_matrix(f)),
                    note="Read down a column to compare the same month across years.")
        with division_col:
            division_long = division_monthly_conflict(filtered)
            if division_long["Division"].nunique() < 2:
                st.markdown("**By division**")
                st.info("One division selected. Widen the Division filter to compare.")
            else:
                _figure("By division", charts.division_trend(division_long, division_palette),
                        "Conflict events per month by division",
                        lambda d=division_long, c=division_palette:
                        report_charts.division_trend_svg(d, c),
                        subtitle="three-month average")

    esc_col, rate_div_col = st.columns(2)
    with esc_col:
        if escalating.empty:
            st.markdown(f"**Escalating beats** -- last {recent_days} days vs the "
                        f"{recent_days} before")
            st.info("No beat shows a materially higher conflict count than the prior window.")
        else:
            _figure("Escalating beats", charts.escalation_dumbbell(escalating),
                    "Escalating beats", lambda e=escalating: report_charts.escalation_svg(e),
                    subtitle=f"last {recent_days} days vs the {recent_days} before",
                    note=f"Grey is the earlier window, red the recent one; "
                         f"{len(escalating)} beat(s) escalating.")
    with rate_div_col:
        rates = division_conflict_rate(filtered)
        if rates.empty:
            st.markdown("**Conflict rate by division**")
            st.info("No division data for the current filters.")
        else:
            _figure("Conflict rate by division",
                    charts.division_rate_bars(rates, kpis["conflict_rate"], division_palette),
                    "Conflict rate by division",
                    lambda r=rates, k=kpis["conflict_rate"], c=division_palette:
                    report_charts.division_rate_svg(r, k, c),
                    note="Rate, not raw volume: sighting counts mostly track reporting "
                         "effort.")


# ---- Where to act -------------------------------------------------------------
with tab_where:
    if not beats_table.empty:
        tier_summary(beats_table["Priority Tier"].value_counts().to_dict())

    # Layer filters: what happened, by which animals, and which villages.
    # They cut the data before anything is drawn, so "crop damage by lone
    # bulls" is a map of exactly those sightings and the places near them.
    event_counts, group_counts = map_select.class_counts(filtered)
    ev_col, grp_col, vil_col = st.columns([5, 5, 4])
    with ev_col:
        events = st.pills(
            "Events", [k for k in map_select.EVENT_CLASSES if event_counts.get(k)],
            selection_mode="multi", key="layer_events",
            default=[k for k in map_select.EVENT_CLASSES if event_counts.get(k)],
            format_func=lambda k, c=event_counts:
            f"{CATEGORY_STYLE[k]['label']} ({c.get(k, 0):,})",
            help="Untick a class to drop it from the map. None ticked hides sightings.",
        ) or []
    with grp_col:
        groups = st.pills(
            "Animals", [g for g in map_select.GROUP_CLASSES if group_counts.get(g)],
            selection_mode="multi", key="layer_groups",
            default=[g for g in map_select.GROUP_CLASSES if group_counts.get(g)],
            format_func=lambda g, c=group_counts: f"{g} ({c.get(g, 0):,})",
            help="Combine with Events: e.g. Human injury + Lone bull alone.",
        ) or []
    with vil_col:
        tier_options = [t for t in TIER_STYLE
                        if not village_risk.empty and (village_risk["Tier"] == t).any()]
        tiers = st.pills(
            "Villages by tier", tier_options, selection_mode="multi", key="layer_tiers",
            default=[t for t in tier_options if t != "Routine"],
            help="Village layer. Routine villages start off to keep the map readable.",
        ) or []
        show_hotspots = st.toggle("Hotspot footprints", value=True, key="layer_hotspots")

    table_col, map_col = st.columns([1, 1], gap="medium")

    # The table draws first, so a ticked row reaches the map in the same
    # run: one click, not two.
    with table_col:
        table_options = ["Beats", "Hotspots", "Villages", "Animals"]
        picker = st.segmented_control(
            "Tick a row to show only that on the map", table_options,
            default="Beats", key="where_table",
        ) or "Beats"

        if picker == "Beats":
            if beats_table.empty:
                st.info("Not enough data to rank beats.")
            else:
                display_cols = [
                    "Beat", "Range", "Priority Tier", "Priority Score",
                    "Conflict Events", "Recent Deaths", "Recent Injuries",
                    "Bull-Type Conflict %", "Trend", "Recommended Action",
                ]
                beat_pick = st.dataframe(
                    beats_table[display_cols],
                    width="stretch",
                    height=520,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="beat_table",
                    column_config={
                        "Priority Tier": st.column_config.TextColumn("Tier", width="small"),
                        "Priority Score": st.column_config.ProgressColumn(
                            "Score", min_value=0, max_value=100, format="%.0f",
                            help="Orders beats within a tier only. Moves when filters "
                            "change.",
                        ),
                        "Conflict Events": st.column_config.NumberColumn("Conflicts"),
                        "Recent Deaths": st.column_config.NumberColumn(
                            "Killed", format="%.0f",
                            help=f"In the last {CRITICAL_CASUALTY_WINDOW_DAYS} days. Drives "
                            "the Critical tier.",
                        ),
                        "Recent Injuries": st.column_config.NumberColumn(
                            "Injured", format="%.0f",
                            help=f"In the last {CRITICAL_CASUALTY_WINDOW_DAYS} days.",
                        ),
                        "Bull-Type Conflict %": st.column_config.NumberColumn(
                            "Bull-driven", format="%.0f%%",
                            help="Share of this beat's conflict from lone bulls or small "
                            "all-male parties.",
                        ),
                        "Recommended Action": st.column_config.TextColumn(width="large"),
                    },
                )
                _map_pick(beat_pick, "beat_table", "beat", beats_table["Beat"], "Beat")
                st.caption(f"Critical = a person killed or injured in the last "
                           f"{CRITICAL_CASUALTY_WINDOW_DAYS} days. Tiers compare across "
                           "periods; the score only orders beats within a tier.")
                st.download_button(
                    "Download all beat figures (CSV)",
                    beats_table.to_csv(index=False).encode("utf-8"),
                    "beat_priorities.csv",
                    mime="text/csv",
                    help="Every column, including period casualties, confidence, rates, "
                    "night and village shares.",
                )

        elif picker == "Hotspots":
            if hotspots.empty:
                st.info("No hotspots at these settings. Try a larger neighbour distance or "
                        "fewer minimum sightings in the sidebar.")
            else:
                hotspot_cols = [c for c in [
                    "Hotspot", "Tier", "Beats", "Sightings", "Conflict Events",
                    "Conflict Share %", "Human Deaths", "People Injured", "Night Share %",
                    "Radius (km)",
                ] if c in hotspots.columns]
                hotspot_pick = st.dataframe(
                    hotspots[hotspot_cols], width="stretch", height=520, hide_index=True,
                    on_select="rerun", selection_mode="single-row", key="hotspot_table",
                    column_config={
                        "Tier": st.column_config.TextColumn(width="small"),
                        "Conflict Share %": st.column_config.NumberColumn(
                            "Conflict share", format="%.0f%%"),
                        "Night Share %": st.column_config.NumberColumn("Night",
                                                                       format="%.0f%%"),
                        "Human Deaths": st.column_config.NumberColumn("Killed", format="%.0f"),
                        "People Injured": st.column_config.NumberColumn("Injured",
                                                                        format="%.0f"),
                        "Radius (km)": st.column_config.NumberColumn("Radius",
                                                                     format="%.1f km"),
                    },
                )
                _map_pick(hotspot_pick, "hotspot_table", "hotspot", hotspots["Hotspot"],
                          "Hotspot")
                st.download_button(
                    "Download hotspots (CSV)",
                    hotspots.to_csv(index=False).encode("utf-8"),
                    "hotspots.csv",
                    mime="text/csv",
                )
            for note in hotspot_caveats(hotspots, filtered, villages):
                st.caption(note)

        elif picker == "Villages":
            if village_risk.empty:
                st.info("No village ranking: load a village centroids file in the sidebar.")
            else:
                village_cols = [c for c in [
                    "Village", "Tier", "Conflict Events", "Human Deaths", "People Injured",
                    "House Damage Events", "Crop Damage Events", "Nearest Hotspot",
                ] if c in village_risk.columns]
                village_pick = st.dataframe(
                    village_risk[village_cols], width="stretch", height=520,
                    hide_index=True, on_select="rerun", selection_mode="single-row",
                    key="village_table",
                    column_config={
                        "Tier": st.column_config.TextColumn(width="small"),
                        "Conflict Events": st.column_config.NumberColumn("Conflicts"),
                        "Human Deaths": st.column_config.NumberColumn("Killed"),
                        "People Injured": st.column_config.NumberColumn("Injured"),
                        "House Damage Events": st.column_config.NumberColumn("House"),
                        "Crop Damage Events": st.column_config.NumberColumn("Crop"),
                        "Nearest Hotspot": st.column_config.TextColumn("Hotspot"),
                    },
                )
                _map_pick(village_pick, "village_table", "village", village_risk["Village"],
                          "Village")
                st.caption(f"A picked village shows the sightings within "
                           f"{village_radius:g} km of it, the exposure radius in the sidebar.")

        else:
            if composition.empty or composition["Conflict Events"].sum() == 0:
                st.info("No group composition recorded for the current filters.")
            else:
                st.dataframe(
                    composition,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Damage Rate %": st.column_config.NumberColumn(
                            "Damage rate", format="%.1f%%",
                            help="Share of that group's sightings involving damage, "
                            "injury or death.",
                        ),
                        "Human Deaths": st.column_config.NumberColumn("Killed", format="%.0f"),
                    },
                )
                st.caption(
                    "Map a group with the Animals filter above. Only bulls carry tusks, so "
                    "Male Count is the tusker count. A lone bull or small all-male party "
                    "raids and is tracked and deterred; a breeding herd with calves is "
                    "given passage and not driven. Unrecorded groups are kept apart "
                    "rather than guessed."
                )

    map_focus = _map_focus(filtered, hotspots, village_risk, village_radius)
    shown, shown_hotspots, shown_villages = map_select.select(
        filtered, hotspots, village_risk, events=events, groups=groups, tiers=tiers,
        focus=map_focus, village_km=village_radius,
    )

    with map_col:
        head, menu = st.columns([10, 1], vertical_alignment="bottom")
        head.markdown(f"**Spatial view** -- {len(shown):,} of {len(filtered):,} sightings")
        with menu:
            download_menu(
                "Spatial view",
                lambda f=shown, h=shown_hotspots if show_hotspots else None, b=basemap:
                sightings_map_svg(f, h, b),
                compact=True,
            )
        if map_focus:
            focus_col, clear_col = st.columns([3, 1], vertical_alignment="center")
            focus_col.info(f"Showing {map_focus['label']} only. Untick the row to show "
                           "everything again.")
            if clear_col.button("Show everything", key="clear_map_focus", width="stretch"):
                st.session_state.pop(MAP_FOCUS_KEY, None)
                st.rerun()
        if shown.empty:
            st.caption("No sightings match the Events and Animals filters"
                    + (f" in {map_focus['label']}." if map_focus else "."))
        render_map(
            shown,
            frame_df=filtered[map_focus["mask"]] if map_focus else filtered,
            focus=map_focus,
            hotspots=shown_hotspots,
            villages=shown_villages,
            style_name=basemap,
            show_sightings=not shown.empty,
            show_hotspots=show_hotspots,
            show_villages=not shown_villages.empty,
            show_boundaries=layer_boundaries,
            boundary_level=boundary_level,
            beat_stats=beats_table,
        )


# ---- Emerging hotspots ------------------------------------------------------
with tab_emerging:
    st.caption(
        f"Reports binned into {cube_cell_km:g} km hexagons by {cube_step.lower()}, then each "
        "place tested for whether it is a hotspot at each step and whether that is "
        "changing. Settings are under Space-time trend in the sidebar."
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

        if not named.empty:
            st.markdown("**The space-time cube** -- the ground on two axes, time rising "
                        "on the third")
            view_col, focus_col = st.columns([3, 2], vertical_alignment="bottom")
            camera = view_col.segmented_control(
                "View", list(charts.CUBE_CAMERAS), default="Oblique", key="cube_camera",
            ) or "Oblique"
            focus_names = {
                row["Place"]: f"{row['Place']} {row['Pattern']}"
                + (f" - {str(row['Nearest Village']).split(' (')[0]}"
                   if row["Nearest Village"] else "")
                for row in named.to_dict("records")
            }
            # A string, not None, for "all": Streamlit reads a None option as
            # nothing selected and shows an empty box.
            all_places = "All places"
            picked_focus = focus_col.selectbox(
                "Focus", [all_places, *focus_names], key="cube_focus",
                format_func=lambda p, n=focus_names: n.get(p, p),
                help="Zoom the cube onto one place and its neighbours.",
            )
            focus = None if picked_focus == all_places else picked_focus
            step_col, ctx_col = st.columns([6, 1], vertical_alignment="bottom")
            step_fmt = "%b %Y" if cube.step == "Month" else "%d %b %Y"
            slice_labels = ["None"] + [f"{s:{step_fmt}}" for s in cube.steps]
            picked_slice = step_col.select_slider(
                "Highlight a step", slice_labels, value="None", key="cube_slice",
            )
            show_context = ctx_col.toggle("Others", value=True, key="cube_context",
                                          help="Faint dots for places with no pattern.")
            highlight = (None if picked_slice == "None"
                         else cube.steps[slice_labels.index(picked_slice) - 1])
            # Like the maps, a cube that fails to draw says so in its own
            # place: an exception here would stop the script and take the
            # Villages, When and Data tabs down with it.
            try:
                cube_figure = charts.space_time_cube(
                    cube, emerging, cube_hot, villages,
                    camera=camera, highlight_step=highlight, context=show_context,
                    floor_boundaries=layer_boundaries, focus=focus,
                )
            except Exception as exc:  # noqa: BLE001 - the page must survive
                logger.exception("Space-time cube failed")
                st.error(
                    f"The space-time cube could not be drawn: {type(exc).__name__}. "
                    "Everything else on this page is unaffected."
                )
            else:
                st.plotly_chart(cube_figure, width="stretch", config=CUBE_CONFIG,
                                key="cube_chart")
            st.caption(
                "Each column is a place; each hexagon, a step with events, its area "
                "following the count. Coloured hexagons are steps the place was a "
                "significant hotspot, grey ones steps it was not, so a New place is grey "
                "with a coloured top and a Diminishing one the reverse. The floor carries "
                "the forest boundaries. Drag to rotate, scroll to zoom, click a legend "
                "entry to hide it; the buttons at the top right reset the view and save "
                "it as an image."
            )

        with st.expander("How to read this", expanded=False):
            for pattern in spacetime.PATTERN_ORDER:
                st.markdown(f"- **{pattern}**: {spacetime.PATTERN_MEANING[pattern]}")
            for note in spacetime.pattern_caveats(cube, emerging, cube_fdr):
                st.markdown(f"- {note}")


# ---- Villages --------------------------------------------------------------
with tab_villages:
    st.caption(f"Conflict within {village_radius:g} km of each village, so a settlement "
               "between two hotspots counts both.")
    if village_risk.empty:
        st.info(
            "No village ranking. Sighting exports carry no village field, so this "
            "needs a centroids file (Village, Latitude, Longitude) -- the Census "
            "village directory, a forest department layer, or OpenStreetMap places."
        )
    else:
        table_col, map_col = st.columns([1, 1])
        with table_col:
            village_cols = [c for c in [
                "Village", "Tier", "Conflict Events", "Human Deaths", "People Injured",
                "House Damage Events", "Crop Damage Events", "Nearest Hotspot",
            ] if c in village_risk.columns]
            st.dataframe(
                village_risk[village_cols],
                width="stretch",
                height=520,
                hide_index=True,
                column_config={
                    "Tier": st.column_config.TextColumn(width="small"),
                    "Conflict Events": st.column_config.NumberColumn("Conflicts"),
                    "Human Deaths": st.column_config.NumberColumn("Killed"),
                    "People Injured": st.column_config.NumberColumn("Injured"),
                    "House Damage Events": st.column_config.NumberColumn("House"),
                    "Crop Damage Events": st.column_config.NumberColumn("Crop"),
                    "Nearest Hotspot": st.column_config.TextColumn("Hotspot"),
                },
            )
            st.download_button(
                "Download all village figures (CSV)",
                village_risk.to_csv(index=False).encode("utf-8"),
                "villages_at_risk.csv",
                mime="text/csv",
            )
        with map_col:
            head, menu = st.columns([8, 1], vertical_alignment="bottom")
            head.markdown("**Village risk map**")
            with menu:
                download_menu("Village risk map",
                              lambda v=village_risk, h=hotspots, b=basemap:
                              village_map_svg(v, h, b), compact=True)
            render_village_map(
                village_risk, hotspots, style_name=basemap,
                show_boundaries=layer_boundaries, boundary_level=boundary_level,
            )
        for note in village_caveats(village_risk, filtered, village_radius):
            st.caption(note)

    if not coverage_table.empty:
        section(
            "broadcast",
            "Early-warning coverage",
            "The villages the ranking says are exposed, against the people "
            "registered to be warned there.",
        )
        ews = _ews()
        min_contacts = int(coverage_stats["min_contacts"])
        findings(ews.coverage_headlines(coverage_table, coverage_stats))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Registrations placed", f"{coverage_stats['matched']:,}")
        m2.metric("Villages reached", f"{coverage_stats['villages_reached']:,}")
        m3.metric(
            "Exposed, no contact",
            f"{coverage_stats['no_contact']:,}",
            delta=f"of {coverage_stats['exposed']:,} exposed villages",
            delta_color="off",
        )
        m4.metric(
            "Critical / High short of contacts",
            f"{coverage_stats['urgent']:,}",
            delta=f"of {coverage_stats['at_risk']:,} in those tiers",
            delta_color="off",
        )

        gaps = ews.coverage_gaps(coverage_table)
        gap_cols = [
            "Village", "Tier", "Registered Contacts", "Coverage", "Conflict Events",
            "Human Deaths", "People Injured", "Nearest Hotspot",
        ]
        coverage_config = {
            "Tier": st.column_config.TextColumn(width="small"),
            "Registered Contacts": st.column_config.NumberColumn(
                "Registered", format="%d",
                help="People enrolled for early warning within "
                f"{coverage_stats['radius_km']:g} km of the village centroid.",
            ),
            "Coverage": st.column_config.TextColumn(
                width="small",
                help=f"Fewer than {min_contacts} registrations counts as thin.",
            ),
        }

        if gaps.empty:
            st.success(
                "Every Critical and High village has at least "
                f"{min_contacts} registered contacts."
            )
        else:
            st.markdown("**Enrolment queue** -- worst tier and thinnest cover first")
            st.dataframe(
                gaps[[c for c in gap_cols if c in gaps.columns]],
                width="stretch", hide_index=True, column_config=coverage_config,
            )
            st.download_button(
                "Download enrolment queue (CSV)",
                gaps.to_csv(index=False).encode("utf-8"),
                "ews_enrolment_queue.csv",
                mime="text/csv",
            )

        with st.expander(
            f"All {len(coverage_table)} exposed villages with their coverage",
            expanded=False,
        ):
            st.dataframe(
                coverage_table[[c for c in gap_cols if c in coverage_table.columns]],
                width="stretch", hide_index=True, column_config=coverage_config,
            )

        enrol_col, staff_col = st.columns(2)
        labels = ews.division_labels(filtered)

        with enrol_col:
            st.markdown("**Enrolment by division**")
            enrolment = ews.enrolment_by_division(registry, labels)
            st.dataframe(enrolment, width="stretch", hide_index=True)
            st.caption(
                "Whole registry, not the filtered period -- enrolment is a standing "
                "figure, and the divisions it is thinnest in are the ones to work on."
            )

        with staff_col:
            st.markdown("**Reporting against registered users**")
            if roster is None:
                st.info(
                    "Upload the registered app-user list in the sidebar to compare "
                    "reporting volume against the staff enrolled in each division."
                )
            else:
                staffing = ews.division_staffing(filtered, roster)
                st.dataframe(
                    staffing,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Reports per User": st.column_config.NumberColumn(
                            "Reports / user", format="%.1f",
                            help="Blank where reports arrived from a division with "
                            "no registered user.",
                        ),
                    },
                )
                st.caption(
                    "Division only: Range was a placeholder for almost every user "
                    "in this export, and Beat for all of them. A low figure is as "
                    "likely to mean accounts created and unused as a quiet division."
                )

        for note in ews.coverage_caveats(coverage_stats):
            st.caption(note)

    elif registry is not None and not village_risk.empty:
        st.info(
            "The registry loaded but nobody could be placed against the exposed "
            "villages. Check that its coordinates cover the same landscape as the "
            "centroids file."
        )
    elif registry is None:
        st.caption("Upload the early-warning villager registry in the sidebar to see which "
                   "exposed villages have nobody enrolled to be warned.")


# ---- When ------------------------------------------------------------------
with tab_when:
    months = temporal["peak_months"]
    w1, w2, w3 = st.columns(3)
    w1.metric("Injuries and deaths peak",
              format_window(temporal["peak_window"]).split(" (")[0]
              if temporal["peak_window"] else "Not enough timed casualties",
              border=True)
    w2.metric("Counted as night raids", f"{temporal.get('night_damage', 0):,}",
              delta="damage and indirect reports", delta_color="off", border=True)
    w3.metric("Seasonal peak", ", ".join(months) if months else "Spread through the year",
              border=True)
    st.caption(
        f"Damage and indirect signs are recorded after the night, so they count as night "
        f"raids ({NIGHT_HOUR_START:02d}:00-{NIGHT_HOUR_END:02d}:00) whatever hour they were "
        "entered; only direct sightings are timed. "
        + (f"Peak window detail: {format_window(temporal['peak_window'])}."
           if temporal["peak_window"] else "")
    )

    hour_col, nd_col = st.columns(2)
    with hour_col:
        hourly = temporal["hourly"]
        if float(hourly.sum()) > 0:
            _figure("Injuries and deaths by hour",
                    charts.hourly_profile(hourly, NIGHT_HOUR_START, NIGHT_HOUR_END),
                    "Injuries and deaths by hour",
                    lambda h=hourly: report_charts.hourly_svg(h, NIGHT_HOUR_START,
                                                              NIGHT_HOUR_END))
        else:
            st.markdown("**Injuries and deaths by hour**")
            st.info("No timed injuries or deaths, so no hour-of-day risk window can be "
                    "identified. Damage is counted as night conflict.")
    with nd_col:
        nd = night_day_comparison(filtered)
        if nd.empty:
            st.markdown("**Night and day**")
            st.info("Night/day classification unavailable (no Hour/Time column).")
        else:
            fig = px.bar(nd, x="Period", y="Entries", text="Entries")
            fig.update_traces(marker_color=BRAND)
            fig.update_layout(
                height=320, margin=dict(t=10, b=10), showlegend=False,
                xaxis_title="", yaxis_title="Reports",
                yaxis_gridcolor="rgba(128,128,128,0.2)",
            )
            _figure("Night and day", fig, "Night and day",
                    lambda n=nd: report_charts.bars_svg(
                        n["Period"].tolist(), n["Entries"].tolist(), "Night and day"),
                    subtitle="all reports, stale ones counted as night")


# ---- Data ------------------------------------------------------------------
with tab_data:
    trend_col, sev_col = st.columns(2)
    with trend_col:
        trend = monthly_trend(filtered)
        if trend.empty:
            st.markdown("**Monthly sightings and conflict events**")
            st.info("Not enough data to chart a monthly trend.")
        else:
            fig = px.line(
                trend.reset_index(), x="Month", y=["Sightings", "Conflict Events"],
                markers=True, color_discrete_sequence=[BRAND, BRAND_ALT],
            )
            fig.update_layout(
                height=320, margin=dict(t=10, b=10), legend_title_text="",
                yaxis_title="Reports", xaxis_title="Month",
                yaxis_gridcolor="rgba(128,128,128,0.2)",
            )
            _figure("Monthly sightings and conflict events", fig,
                    "Monthly sightings and conflict events",
                    lambda t=trend: report_charts.sightings_vs_conflict_svg(t),
                    note="Sightings mostly track reporting effort; read the conflict "
                         "line, or the rate on the first tab.")
    with sev_col:
        bands = severity_distribution(filtered)
        if bands.empty or int(bands["Count"].sum()) == 0:
            st.markdown("**Severity distribution**")
            st.info("No severity data to chart.")
        else:
            fig = px.bar(bands, x="Band", y="Count")
            fig.update_traces(marker_color=BRAND)
            fig.update_layout(
                height=320, margin=dict(t=10, b=10), showlegend=False,
                xaxis_title="", yaxis_title="Reports",
                yaxis_gridcolor="rgba(128,128,128,0.2)",
            )
            _figure("Severity distribution", fig, "Severity distribution",
                    lambda b=bands: report_charts.bars_svg(
                        b["Band"].tolist(), b["Count"].tolist(), "Severity distribution"))

    rates = division_conflict_rate(filtered)
    if not rates.empty:
        with st.expander("Division figures as a table", expanded=False):
            st.dataframe(rates, width="stretch")
    if not escalating.empty:
        with st.expander("Escalating beats as a table", expanded=False):
            st.dataframe(
                escalating[["Beat", "Division", "Recent vs Prior", "Human Deaths",
                            "People Injured"]],
                width="stretch", hide_index=True,
            )

    st.markdown(f"**Filtered reports** -- {len(filtered):,} rows after filters")
    preview_cols = [
        c
        for c in [
            "Date", "Division", "Range", "Beat", "Latitude", "Longitude",
            "Severity Score", "Is_Night", "Nearest Village", "Distance to Village (km)",
            "Crop Damage", "Grain Damage", "House Damage", "Injury", "Death",
        ]
        if c in filtered.columns
    ]
    st.dataframe(filtered[preview_cols].head(500), width="stretch", height=360)
    st.caption(f"Showing up to 500 of {len(filtered):,} filtered rows.")
    st.download_button(
        "Download filtered data (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        "filtered_elephant_data.csv",
        mime="text/csv",
    )
    st.warning(
        "Field exports carry victim and reporter names, often in the free-text "
        "description. This app has no access control -- check before sharing."
    )
