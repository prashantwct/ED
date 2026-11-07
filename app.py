"""Elephant Conflict Intelligence -- Streamlit entry point.

All data logic lives in core/ and each page's layout in app_pages/; this
file is the frame around them. It runs on every page: the masthead, the
line saying which register is loaded, the scope bar (area, period,
incident type), the sidebar, and the analysis every page shares, worked
out once per scope. Then it runs the page the reader is on.

The pages, along the top: Overview (what changed, where to act, whom to
warn), Where to act, Villages, Patterns, Day refuges, Damage, Data &
brief, and Analyst.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from types import SimpleNamespace
from typing import Optional

import pandas as pd
import streamlit as st

# Set on each of the app's modules when this process first sees it loaded.
_LOADED_AT = "_ed_loaded_at"
# The app's own packages: core/ for the analysis, app_pages/ for the
# pages, tools/ for the Forest Alerts client the fetch imports from.
_OUR_PACKAGES = ("core", "app_pages", "tools")


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

from core import analysis_settings, boundaries, charts, landing, scope  # noqa: E402
from core.analytics import (  # noqa: E402
    compute_is_night,
    compute_kpis,
    compute_severity,
    monthly_conflict_breakdown,
    window_comparison,
)
from core.config import LOG_PATH  # noqa: E402
from core.data_loader import load_and_validate_csv  # noqa: E402
from core.exceptions import DataValidationError, SpatialEnrichmentError  # noqa: E402
from core.map_engine import BASEMAP_STYLES, DEFAULT_BASEMAP, basemap_warning  # noqa: E402
from core.spatial import attach_nearest_village, load_village_centroids  # noqa: E402
from core.ui import inject_theme  # noqa: E402

from app_pages import common  # noqa: E402

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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and validating the export...")
def _load(file_bytes: bytes, filename: str):
    """Parse an uploaded CSV. Cached on content, so filters don't re-parse."""
    return load_and_validate_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Reading the early-warning registry...")
def _registry(file_bytes: bytes, filename: str):
    """De-identify the registry once per upload, not once per rerun."""
    return common._ews().load_ews_registry(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Reading the registered user list...")
def _roster(file_bytes: bytes, filename: str):
    return common._ews().load_staff_roster(io.BytesIO(file_bytes))


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


def _day(stamp) -> str:
    """A date as people write it: 1 Oct 2025."""
    return f"{stamp.day} {stamp:%b %Y}"


# A fetch survives reruns under this key; nothing else about it does.
FETCHED_KEY = "fetched_register"
# The last register that loaded, so the refuge and damage pages can read
# it even in a session that never drew the uploader.
REGISTER_KEY = "loaded_register"
UPLOAD_KEY = "register_upload"

# The scope bar's widgets.
SCOPE_KEYS = ("flt_divisions", "flt_ranges", "flt_beats", "flt_period", "flt_dates",
              "flt_severity")
MAP_KEYS = ("map_basemap", "map_boundaries", "map_boundary_level")
BOUNDARY_AUTO = "Auto (by zoom)"


# ---------------------------------------------------------------------------
# 0. Keep settings alive across pages
# ---------------------------------------------------------------------------
# Streamlit forgets a widget's value on any run that does not draw it, so
# a setting made on the Analyst page would be gone after one visit to the
# Overview. Writing each value back to itself at the top of every run
# keeps it; the widgets are then created without a default of their own.
for _key in SCOPE_KEYS + MAP_KEYS + analysis_settings.KEYS:
    if _key in st.session_state:
        st.session_state[_key] = st.session_state[_key]
for _setting_ in analysis_settings.SETTINGS:
    st.session_state.setdefault(_setting_.key, _setting_.default)
st.session_state.setdefault("map_basemap", DEFAULT_BASEMAP)
st.session_state.setdefault("map_boundaries", True)
st.session_state.setdefault("map_boundary_level", BOUNDARY_AUTO)


# ---------------------------------------------------------------------------
# 1. Pages
# ---------------------------------------------------------------------------
# Pages are functions that read one shared context, built below once the
# scope is known. They are defined here and run last.
ctx = SimpleNamespace()


def _page(module: str):
    def run() -> None:
        import importlib

        importlib.import_module(f"app_pages.{module}").render(ctx)

    run.__name__ = module
    return run


PAGES = {
    "": st.Page(_page("home"), title="Overview", icon=":material/dashboard:",
                url_path="overview", default=True),
    "where": st.Page(_page("where"), title="Where to act", icon=":material/map:",
                     url_path="where"),
    "next": st.Page(_page("next"), title="Next moves", icon=":material/near_me:",
                    url_path="next"),
    "range": st.Page(_page("range"), title="Range & corridors", icon=":material/route:",
                     url_path="range"),
    "villages": st.Page(_page("villages"), title="Villages", icon=":material/home:",
                        url_path="villages"),
    "patterns": st.Page(_page("patterns"), title="Patterns",
                        icon=":material/trending_up:", url_path="patterns"),
    "refuges": st.Page(lambda: _refuge_page(), title="Day refuges",
                       icon=":material/forest:", url_path="refuges"),
    "damage": st.Page(lambda: _damage_page(), title="Damage",
                      icon=":material/agriculture:", url_path="damage"),
    "data": st.Page(_page("data"), title="Data & brief", icon=":material/description:",
                    url_path="data"),
    "analyst": st.Page(_page("analyst"), title="Analyst", icon=":material/tune:",
                       url_path="analyst"),
}
ctx.pages = PAGES
CONFLICT_PAGES = {"", "where", "next", "range", "villages", "patterns", "data", "analyst"}

nav = st.navigation(list(PAGES.values()), position="top")
current = next(k for k, p in PAGES.items() if p.url_path == nav.url_path)


# ---------------------------------------------------------------------------
# 2. The register: upload or fetch, and the line that says what loaded
# ---------------------------------------------------------------------------
masthead = st.container()
fetched = st.session_state.get(FETCHED_KEY)
# Once a register is loaded the uploader moves into a "Replace" popover on
# the source line, drawn on every page so the file survives moving between
# them. A keyed uploader's identity is its key, not where it is drawn, and
# session state already holds the new value when the run starts, so the
# layout can be chosen before the widget exists.
held = st.session_state.get(UPLOAD_KEY) is not None or fetched is not None
note_slot = notes_slot = None
uploaded = None
if held:
    note_col, notes_col, replace_col, *clear_col = st.columns(
        [7, 2, 2, 1] if fetched else [8, 2, 2], vertical_alignment="center")
    note_slot, notes_slot = note_col.empty(), notes_col.empty()
    replacing = st.session_state.get(UPLOAD_KEY) is not None
    with replace_col.popover("Replace file" if replacing else "Upload a file",
                             icon=":material/upload_file:", width="stretch"):
        uploaded = st.file_uploader(
            "Upload a sightings export (CSV)", type="csv", key=UPLOAD_KEY,
            help="Replaces the register on screen. Nothing is stored.",
        )
    if clear_col:
        with clear_col[0]:
            if st.button("Clear", help="Discard the fetched register."):
                del st.session_state[FETCHED_KEY]
                st.rerun()
elif current in CONFLICT_PAGES:
    uploaded = st.file_uploader(
        "Upload the Gajrakshak sightings export (CSV)", type="csv", key=UPLOAD_KEY,
    )

with masthead:
    if uploaded or fetched or current not in CONFLICT_PAGES:
        landing.header()
    else:
        landing.hero()


def _stashed_register() -> Optional[pd.DataFrame]:
    """The register last loaded in this session, if any."""
    stash = st.session_state.get(REGISTER_KEY)
    if stash is None:
        return None
    try:
        return _load(*stash)[0]
    except DataValidationError:
        return None


def _optional_data() -> SimpleNamespace:
    """The sidebar's optional uploads: centroid override, registry, user list.

    Drawn on every page that holds a register, so a file added once stays
    while the reader moves around.
    """
    out = SimpleNamespace(centroid_upload=None, registry_upload=None, roster_upload=None,
                          panel=st.sidebar.container())
    out.panel.divider()
    out.panel.subheader("Optional data")
    with out.panel.expander("Village centroids", expanded=False):
        st.caption(
            "Shahdol-Anuppur centroids ship with the app and load automatically. "
            "Upload only to override them. Columns: Village, Latitude, Longitude."
        )
        out.centroid_upload = st.file_uploader(
            "Override centroids", type="csv", key="centroids_upload"
        )
    with out.panel.expander("Early-warning coverage", expanded=False):
        st.caption(
            "Optional. Upload the early-warning villager registry to see which "
            "exposed villages have nobody enrolled, and the app-user list to see "
            "reporting effort against registered staff."
        )
        out.registry_upload = st.file_uploader(
            "Villager registry (CSV)", type="csv", key="registry_upload",
            help="Columns: Latitude, Longitude, and ideally Village and Division.",
        )
        out.roster_upload = st.file_uploader(
            "Registered app users (CSV)", type="csv", key="roster_upload",
            help="Columns: Division, and ideally Latitude and Longitude.",
        )
        st.caption(
            "Both files carry personal data. Names, mobile numbers and email "
            "addresses are discarded as the file is read and coordinates are "
            "rounded to about 100 m, so only per-village counts reach the page or "
            "any download. Neither file is stored."
        )
    return out


def _refuge_page() -> None:
    from core import refuge_view

    register = _stashed_register()
    try:
        refuge_villages, _ = load_village_centroids(None)
    except SpatialEnrichmentError:
        refuge_villages = None
    if register is not None:
        register, _ = attach_nearest_village(register, refuge_villages)
        chosen = st.sidebar.multiselect(
            "Division", sorted(register["Division"].unique()), key="refuge_divisions",
            help="Leave empty for the whole register.")
        if chosen:
            register = register[register["Division"].isin(chosen)]
    basemap_note = basemap_warning()
    refuge_basemap = st.sidebar.selectbox(
        "Basemap", list(BASEMAP_STYLES),
        index=list(BASEMAP_STYLES).index(DEFAULT_BASEMAP), key="refuge_basemap",
        disabled=basemap_note is not None, help=basemap_note,
    )
    refuge_view.render(register, refuge_villages, refuge_basemap)


def _damage_page() -> None:
    from core import damage_view

    damage_df, damage_notes = damage_view.sources()
    if damage_df is None:
        st.info("Upload a herd-crop-damage or herd-house-damage export to begin.")
        return
    basemap_note = basemap_warning()
    damage_basemap = st.sidebar.selectbox(
        "Basemap", list(BASEMAP_STYLES),
        index=list(BASEMAP_STYLES).index(DEFAULT_BASEMAP), key="damage_basemap",
        disabled=basemap_note is not None, help=basemap_note,
    )
    damage_view.render(damage_df, damage_notes, _stashed_register(), damage_basemap)


# Nothing loaded yet: the conflict pages show the landing page.
if not uploaded and not fetched:
    if current not in CONFLICT_PAGES:
        nav.run()
        st.stop()
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
    if st.button(
        "Crop and house damage surveys (Epicollect5)",
        help="Losses, compensation and the Gaj Rakshak gap, from the damage surveys.",
    ):
        st.switch_page(PAGES["damage"])
    landing.details()
    st.stop()

if note_slot is None:  # a source the start of the run did not know about
    _note_col, _notes_col = st.columns([10, 2], vertical_alignment="center")
    note_slot, notes_slot = _note_col.empty(), _notes_col.empty()


def _source_note(rows: Optional[int] = None, span: str = "") -> None:
    counted = f", {rows:,} reports" if rows is not None else ""
    if uploaded:
        held_too = (f" The register fetched {fetched.fetched_at} is still held; remove "
                    f"this file to go back to it." if fetched else "")
        with note_slot.container():
            landing.source_note("Uploaded file",
                                f"{uploaded.name}{counted}{span}.{held_too}")
        return
    # A fetch held from before this field existed has no source.
    how = ("from the export" if getattr(fetched, "source", "") == "export"
           else f"over {fetched.pages} page(s) of the listing")
    valid = (f"; {rows:,} valid" if rows is not None and rows != fetched.rows else "")
    with note_slot.container():
        landing.source_note(
            "Forest Alerts",
            f"{fetched.rows:,} rows {how}{valid}, {fetched.start_date} to "
            f"{fetched.end_date}, fetched {fetched.fetched_at}.",
        )


# The source line is drawn before the load, so a file the loader rejects
# does not strand the session on an error with no way back.
_source_note()

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
_source_note(len(raw_df),
             f", {_day(raw_df['Date'].min())} to {_day(raw_df['Date'].max())}")

# Anything worth knowing about the data, gathered as it is read and shown
# behind one button on the source line rather than as banners around it.
data_notes: list = list(load_warnings)


def _notes_popover(notes: list, about: list) -> None:
    with notes_slot.container():
        with st.popover(f"Data notes ({len(notes)})" if notes else "About the data",
                        icon=":material/info:", width="stretch",
                        help="What was found while reading the register."):
            for note in notes:
                st.markdown(f"- {note}")
            if not notes:
                st.markdown("Nothing unusual was found while reading the register.")
            for line in about:
                st.caption(line)


# The refuge and damage pages read the register on their own terms.
if current not in CONFLICT_PAGES:
    _notes_popover(data_notes, [])
    nav.run()
    st.stop()

# Map styling first in the sidebar, then the optional uploads below it.
sidebar_map = st.sidebar.container()
optional = _optional_data()

try:
    villages, centroid_warnings = load_village_centroids(
        optional.centroid_upload if optional.centroid_upload else None
    )
except SpatialEnrichmentError as exc:
    # Beside the uploader when it was the user's file; a data note when
    # the bundled file is what failed.
    failed = f"Village centroids not applied: {exc}"
    if optional.centroid_upload:
        optional.panel.warning(failed)
    else:
        data_notes.append(failed)
    villages, centroid_warnings = None, []

df, spatial_warnings = attach_nearest_village(raw_df, villages)
data_notes += list(centroid_warnings) + list(spatial_warnings)
data_about = []
if villages is not None:
    data_about.append(
        f"Villages are named against {len(villages):,} "
        f"{'uploaded' if optional.centroid_upload else 'bundled'} village centroids."
    )

_notes_popover(data_notes, data_about)

registry = roster = None
for source, loader, label in (
    (optional.registry_upload, _registry, "registry"),
    (optional.roster_upload, _roster, "user list"),
):
    if source is None:
        continue
    try:
        loaded, load_notes = loader(source.getvalue(), source.name)
    except DataValidationError as exc:
        optional.panel.warning(f"Could not read the {label}: {exc}")
        continue
    except ImportError:
        # Stale interpreter after a deploy: the entry point has been
        # reloaded from disk but an already-imported module has not.
        logger.exception("Coverage subsystem failed to import")
        optional.panel.error(
            "Early-warning coverage could not load. If the app was updated "
            "just now, reboot it so the running process picks up the new "
            "modules. Everything else on this page is unaffected."
        )
        break
    for note in load_notes:
        optional.panel.caption(note)
    if label == "registry":
        registry = loaded
        optional.panel.success(f"{len(registry):,} registrations read.")
    else:
        roster = loaded
        optional.panel.success(f"{len(roster):,} registered users read.")

df = df.copy()
df["Severity Score"] = compute_severity(df)
df["Is_Night"] = compute_is_night(df)


# ---------------------------------------------------------------------------
# 3. The scope bar: area, period, incident type -- on every conflict page
# ---------------------------------------------------------------------------
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
division_options = sorted(df["Division"].unique())
incident_options = list(common.SEVERITY_FILTERS)

# First run with this register: start from the link, if it carries a scope.
if "flt_period" not in st.session_state:
    shared = scope.from_query(
        {k: st.query_params.get_all(k) for k in st.query_params},
        division_options, sorted(df["Range"].unique()), sorted(df["Beat"].unique()),
        incident_options)
    st.session_state["flt_divisions"] = shared["divisions"]
    st.session_state["flt_ranges"] = shared["ranges"]
    st.session_state["flt_beats"] = shared["beats"]
    st.session_state["flt_period"] = shared["period"]
    st.session_state["flt_severity"] = shared["incident"]
    if shared["custom"]:
        st.session_state["flt_dates"] = shared["custom"]


def _reset_scope() -> None:
    for key in ("flt_divisions", "flt_ranges", "flt_beats"):
        st.session_state[key] = []
    st.session_state["flt_period"] = scope.DEFAULT_PERIOD
    st.session_state["flt_severity"] = incident_options[0]
    st.session_state.pop("flt_dates", None)


def _keep_a_period() -> None:
    # A segmented control can be clicked off; a page always has a period.
    if not st.session_state.get("flt_period"):
        st.session_state["flt_period"] = scope.DEFAULT_PERIOD


# Prune stale selections before creating the dependent widgets. Widening
# Division back out otherwise leaves Range holding values no longer
# offered, silently excluding rows while the UI looks reset.
st.session_state["flt_divisions"] = [
    d for d in st.session_state.get("flt_divisions", []) if d in division_options]
divisions = st.session_state["flt_divisions"]
range_pool = df[df["Division"].isin(divisions)] if divisions else df
range_options = sorted(range_pool["Range"].unique())
st.session_state["flt_ranges"] = [
    r for r in st.session_state.get("flt_ranges", []) if r in range_options]
ranges = st.session_state["flt_ranges"]
beat_pool = range_pool[range_pool["Range"].isin(ranges)] if ranges else range_pool
beat_options = sorted(beat_pool["Beat"].unique())
st.session_state["flt_beats"] = [
    b for b in st.session_state.get("flt_beats", []) if b in beat_options]
if st.session_state.get("flt_severity") not in incident_options:
    st.session_state["flt_severity"] = incident_options[0]
if st.session_state.get("flt_period") not in scope.PERIODS:
    st.session_state["flt_period"] = scope.DEFAULT_PERIOD

area_col, period_col, type_col, act_col = st.columns([2.4, 5.4, 2.4, 3.2],
                                                    vertical_alignment="bottom")
with area_col.popover(
        scope.area_label(divisions, ranges, st.session_state["flt_beats"]),
        icon=":material/location_on:", width="stretch",
        help="Division, range and beat. Leave empty for the whole register."):
    divisions = st.multiselect("Division", division_options, key="flt_divisions")
    ranges = st.multiselect("Range", range_options, key="flt_ranges")
    beats = st.multiselect("Beat", beat_options, key="flt_beats")
with period_col:
    period = st.segmented_control(
        "Period", list(scope.PERIODS), key="flt_period", on_change=_keep_a_period,
        format_func=lambda p: scope.SHORT_LABELS.get(p, p),
        help="What the headline numbers count and the map draws. Rankings use the "
        "register up to its end.",
    ) or scope.DEFAULT_PERIOD
    custom = None
    if period == scope.CUSTOM:
        st.session_state.setdefault("flt_dates", (min_date, max_date))
        custom = st.date_input("Custom period", min_value=min_date, max_value=max_date,
                               key="flt_dates", format="DD/MM/YYYY")
severity_choice = type_col.selectbox(
    "Incident type", incident_options, key="flt_severity",
    help="Show only reports at least this serious.")
actions = act_col.container(horizontal=True, horizontal_alignment="right")
actions.button("Reset", icon=":material/restart_alt:", help="Back to all divisions, "
               "the last 90 days and all reports.", on_click=_reset_scope)

window = scope.focus_window(period, min_date, max_date, custom)
query = scope.to_query(divisions, ranges, beats, period, custom, severity_choice,
                       incident_options)
# A link from the Overview names the place to focus the map on. Apply it
# once, then leave it out of the address so a reload does not re-apply it.
_kind, _, _value = (st.query_params.get(common.FOCUS_PARAM) or "").partition(":")
if _kind in common.FOCUS_KINDS and _value:
    common.focus_map_on(_kind, _value)
st.query_params.from_dict(query)


def _share_link() -> str:
    """This page and scope as a link. The register is never in it."""
    from urllib.parse import urlencode, urlsplit

    page = PAGES[current].url_path
    page = "" if current == "" else page
    try:
        parts = urlsplit(st.context.url or "")
    except Exception:  # noqa: BLE001 - no browser context, e.g. under test
        parts = None
    base = f"{parts.scheme}://{parts.netloc}" if parts and parts.netloc else ""
    root = (st.get_option("server.baseUrlPath") or "").strip("/")
    path = "/".join(p for p in (root, page) if p)
    tail = ("?" + urlencode(query, doseq=True)) if query else ""
    return f"{base}/{path}{tail}"


with actions.popover("Share", icon=":material/link:",
                     help="A link to this page with this area and period."):
    st.markdown("**Share this view**")
    st.code(_share_link(), language=None, wrap_lines=True)
    st.caption("The link opens this page with the same area, period and incident "
               "type. It carries no data: whoever opens it loads the register "
               "themselves, under their own sign-in.")

history = scope.history(df, window, divisions, ranges, beats,
                        common.SEVERITY_FILTERS[severity_choice])
focus = scope.focus(history, window)
period_end = pd.Timestamp(window.end)
st.caption(
    f"Showing {window.label()} ({window.days:,} days, {len(focus):,} reports). "
    + ("" if scope.covers_register(window, min_date) else
       f"Rankings and trends use the register from {_day(min_date)} to "
       f"{_day(window.end)}.")
)


# ---------------------------------------------------------------------------
# 4. Sidebar: map styling
# ---------------------------------------------------------------------------
map_panel = sidebar_map
map_panel.subheader("Map")
basemap_unavailable = basemap_warning()
basemap = map_panel.selectbox(
    "Basemap", list(BASEMAP_STYLES), key="map_basemap",
    disabled=basemap_unavailable is not None,
    help=basemap_unavailable or "Terrain and imagery read better than street "
    "cartography here.",
)
if boundaries.available():
    layer_boundaries = map_panel.checkbox("Forest boundaries", key="map_boundaries")
    boundary_choice = map_panel.selectbox(
        "Boundary level",
        [BOUNDARY_AUTO] + [boundaries.LEVEL_LABELS[k] for k in boundaries.LEVELS],
        key="map_boundary_level",
        help="Auto picks the level from the view your filters produce. "
        "Pin one to override it.",
    )
    boundary_level = None if boundary_choice == BOUNDARY_AUTO else {
        boundaries.LEVEL_LABELS[k]: k for k in boundaries.LEVELS
    }[boundary_choice]
else:
    layer_boundaries, boundary_level = False, None


# ---------------------------------------------------------------------------
# 5. The shared analysis, worked out once per scope and cached
# ---------------------------------------------------------------------------
settings = {s.key: st.session_state[s.key] for s in analysis_settings.SETTINGS}
recent_days = int(settings[analysis_settings.RECENT_DAYS.key])
eps_km = float(settings[analysis_settings.EPS_KM.key])
min_samples = int(settings[analysis_settings.MIN_SAMPLES.key])
village_radius = float(settings[analysis_settings.VILLAGE_RADIUS.key])
cube_step = settings[analysis_settings.CUBE_STEP.key]
cube_cell_km = float(settings[analysis_settings.CUBE_CELL_KM.key])
cube_metric = settings[analysis_settings.CUBE_METRIC.key]
cube_fdr = bool(settings[analysis_settings.CUBE_FDR.key])
custom_analysis = analysis_settings.non_standard(settings)

if custom_analysis:
    st.badge("Custom analysis: " + "; ".join(custom_analysis), icon=":material/tune:",
             color="orange",
             help="Set on the Analyst page. Reset to standard there.")

if history.empty:
    st.warning("No reports match this area and incident type. Widen the area or reset "
               "the scope.")
    st.stop()

compare_days = scope.comparison_days(window, min_date, recent_days)
brief = common.brief_for(history, recent_days, period_end)
hotspots = common._hotspots(history, eps_km, min_samples, period_end)
village_risk = common.villages_for(history, villages, hotspots, village_radius, period_end)
coverage_table = pd.DataFrame()
coverage_stats: dict = {}
if registry is not None:
    coverage_table, coverage_stats = common._ews().village_coverage(
        village_risk, villages, registry)
cube, emerging = common._emerging(history, cube_cell_km, cube_step, cube_metric,
                                  cube_fdr, period_end, villages)

ctx.__dict__.update(
    query=query, df=df, history=history, focus=focus, window=window, period_end=period_end,
    register_start=min_date, register_end=max_date,
    villages=villages, registry=registry, roster=roster,
    coverage_table=coverage_table, coverage_stats=coverage_stats,
    kpis=compute_kpis(focus), kpis_history=compute_kpis(history),
    monthly=monthly_conflict_breakdown(history),
    change=window_comparison(history, compare_days, as_of=period_end),
    compare_days=compare_days, brief=brief, beats_table=brief["beats"],
    composition=brief["composition"], escalating=brief["escalating"],
    temporal=brief["temporal"], hotspots=hotspots, village_risk=village_risk,
    cube=cube, emerging=emerging,
    division_palette=charts.division_colors(df["Division"].unique()),
    recent_days=recent_days, eps_km=eps_km, min_samples=min_samples,
    village_radius=village_radius, cube_step=cube_step, cube_cell_km=cube_cell_km,
    cube_metric=cube_metric, cube_fdr=cube_fdr, custom_analysis=custom_analysis,
    basemap=basemap, layer_boundaries=layer_boundaries, boundary_level=boundary_level,
    divisions=divisions, ranges=ranges, beats=beats, severity_choice=severity_choice,
    data_notes=data_notes, data_about=data_about, load_warnings=load_warnings,
)

nav.run()
