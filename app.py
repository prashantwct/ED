"""Elephant Conflict Intelligence -- Streamlit entry point.

All data logic lives in core/; this file handles layout, widgets and
wiring only. Page order follows the order a manager decides in:
situation, which beats, where exactly, which villages, then evidence.
"""

from __future__ import annotations

import hashlib
import io
import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from core.analytics import (
    compute_is_night,
    compute_kpis,
    compute_severity,
    division_conflict_rate,
    filter_dataframe,
    monthly_trend,
    night_day_comparison,
    severity_distribution,
)
from core.config import LOG_PATH
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
from core.map_engine import render_map
from core.report import generate_html_report
from core.spatial import attach_nearest_village, load_village_centroids
from core.ui import (
    TIER_STYLE,
    findings,
    hotspot_card,
    inject_theme,
    section,
    tier_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Elephant Conflict Intelligence",
    layout="wide",
    page_icon="🐘",  # browser tab only; in-page icons are SVG
)
inject_theme()

st.title("Elephant Conflict Intelligence")
st.caption(
    "Sighting and conflict reporting turned into deployment decisions: "
    "which beats, which shift, which villages."
)

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


@st.cache_data(show_spinner="Finding movement hotspots...", hash_funcs={pd.DataFrame: _frame_key})
def _hotspots(frame: pd.DataFrame, eps_km: float, min_samples: int, as_of):
    """Cached hotspot detection -- the slowest step on a rerun."""
    return detect_hotspots(frame, eps_km=eps_km, min_samples=min_samples, as_of=as_of)


# ---------------------------------------------------------------------------
# 1. Upload & load
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Upload sightings/conflict CSV", type="csv")

if not uploaded:
    st.info(
        "Please upload a CSV file to continue. Required columns: "
        "**Date, Latitude, Longitude, Division, Range, Beat**. "
        "Optional columns (unlock extra features): Time or Hour, Total Count, "
        "Crop Damage, Grain Damage, House Damage, Injury, Death, "
        "Male/Female/Children Death Count."
    )
    st.stop()

try:
    raw_df, load_warnings = _load(uploaded.getvalue(), uploaded.name)
except DataValidationError as exc:
    st.error(f"Could not load the file: {exc}")
    st.stop()

if load_warnings:
    with st.expander(f"{len(load_warnings)} data quality note(s)", expanded=False):
        for warning in load_warnings:
            st.warning(warning)

st.success(f"Loaded {len(raw_df):,} valid rows.")


# ---------------------------------------------------------------------------
# 2. Sidebar: centroids, filters, tuning
# ---------------------------------------------------------------------------
with st.sidebar.expander("Village centroids (optional)", expanded=False):
    st.caption(
        "Sighting exports carry no village field, so villages can only be named "
        "against a centroids file. Needs columns: Village, Latitude, Longitude."
    )
    centroid_upload = st.file_uploader(
        "Upload centroids.csv", type="csv", key="centroids_upload"
    )
    st.caption("Falls back to a local `centroids.csv` next to app.py if present.")

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
    help="Each beat's most recent N days are compared against the N days before "
    "them. This does not affect the Critical tier, which uses a fixed "
    f"{CRITICAL_CASUALTY_WINDOW_DAYS}-day casualty window.",
)
eps_km = st.sidebar.slider(
    "Hotspot neighbour distance (km)",
    min_value=0.5, max_value=3.0, value=DEFAULT_EPS_KM, step=0.25,
    help="Larger values merge nearby activity. Push it too far and separate "
    "concentrations chain into one broad region rather than a patrol target.",
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

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Reports", f"{kpis['entries']:,}")
c2.metric(
    "Conflict events",
    f"{kpis['conflicts']:,}",
    delta=None if pd.isna(kpis["conflict_rate"]) else f"{kpis['conflict_rate']:.1f}% of reports",
    delta_color="off",
)
c3.metric("People killed", f"{int(kpis['human_deaths'])}")
c4.metric("People injured", f"{int(kpis['human_injuries'])}")
c5.metric(
    "Night share",
    "N/A" if pd.isna(kpis["night_pct"]) else f"{kpis['night_pct']:.1f}%",
    delta=f"of {kpis['night_known']:,} timed reports" if kpis["night_known"] else None,
    delta_color="off",
)

if filtered.empty:
    st.warning("No rows match the current filters. Adjust the filters in the sidebar.")
    st.stop()

# Anchor casualty recency to the selected period's end, not to whatever
# the current Division/Beat selection happens to end at.
period_end = pd.Timestamp(date_range[1]) if len(date_range) == 2 else df["Date"].max()

brief = management_brief(filtered, recent_days=recent_days, as_of=period_end)


# ---------------------------------------------------------------------------
# 4. Assessment
# ---------------------------------------------------------------------------
section("assessment", "Assessment", f"{len(filtered):,} reports in the selected period")
findings(brief["headlines"])

with st.expander("How to read this", expanded=False):
    for caveat in brief["caveats"]:
        st.markdown(f"- {caveat}")


# ---------------------------------------------------------------------------
# 5. Beat priorities
# ---------------------------------------------------------------------------
section(
    "target",
    "Beat priorities",
    "Tier is set by fixed rules and means the same thing across periods and "
    f"filters. Critical = a casualty in the last {CRITICAL_CASUALTY_WINDOW_DAYS} days. "
    "The score only orders beats within a tier.",
)

beats_table = brief["beats"]
if beats_table.empty:
    st.info("Not enough data to rank beats.")
else:
    tier_summary(beats_table["Priority Tier"].value_counts().to_dict())

    display_cols = [
        "Beat", "Division", "Range", "Priority Tier", "Priority Score", "Confidence",
        "Reports", "Conflict Events", "Adj. Conflict Rate %",
        "Recent Deaths", "Recent Injuries", "Human Deaths", "People Injured",
        "Night Conflict %", "Near Village %", "Trend", "Recent vs Prior",
        "Recommended Action",
    ]
    st.dataframe(
        beats_table[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "Priority Tier": st.column_config.TextColumn("Tier", width="small"),
            "Priority Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.0f",
                help="Orders beats within a tier only. Moves when filters change.",
            ),
            "Adj. Conflict Rate %": st.column_config.NumberColumn(
                "Adj. rate", format="%.1f%%",
                help="Conflict rate shrunk toward the landscape average, so beats "
                "with very few reports cannot top the ranking on one incident.",
            ),
            "Recent Deaths": st.column_config.NumberColumn(
                "Killed (recent)", format="%.0f",
                help=f"In the last {CRITICAL_CASUALTY_WINDOW_DAYS} days. Drives the "
                "Critical tier.",
            ),
            "Recent Injuries": st.column_config.NumberColumn("Injured (recent)", format="%.0f"),
            "Human Deaths": st.column_config.NumberColumn(
                "Killed (period)", format="%.0f",
                help="Across the whole filtered period, however long ago.",
            ),
            "People Injured": st.column_config.NumberColumn("Injured (period)", format="%.0f"),
            "Night Conflict %": st.column_config.NumberColumn("Night", format="%.0f%%"),
            "Near Village %": st.column_config.NumberColumn("Near village", format="%.0f%%"),
            "Recommended Action": st.column_config.TextColumn(width="large"),
        },
    )
    st.download_button(
        "Download beat priorities (CSV)",
        beats_table.to_csv(index=False).encode("utf-8"),
        "beat_priorities.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# 6. Movement hotspots
# ---------------------------------------------------------------------------
section(
    "hotspot",
    "Movement hotspots",
    "Density clusters in the point data itself. A herd working a corridor along "
    "a beat boundary reads as moderate pressure in two beats, but as one hotspot "
    "here.",
)

hotspots = _hotspots(filtered, eps_km, min_samples, period_end)

if hotspots.empty:
    st.info(
        "No density hotspots at the current settings. Try increasing the neighbour "
        "distance or lowering the minimum sightings per hotspot in the sidebar."
    )
else:
    notable = hotspots[hotspots["Tier"] != "Routine"]
    shown = notable if not notable.empty else hotspots.head(3)

    for chunk_start in range(0, len(shown), 2):
        cols = st.columns(2)
        for col, (_, row) in zip(cols, shown.iloc[chunk_start : chunk_start + 2].iterrows()):
            with col:
                hotspot_card(row.to_dict())

    with st.expander(f"All {len(hotspots)} hotspots as a table", expanded=False):
        st.dataframe(hotspots, width="stretch", hide_index=True)
        st.download_button(
            "Download hotspots (CSV)",
            hotspots.to_csv(index=False).encode("utf-8"),
            "hotspots.csv",
            mime="text/csv",
        )

for note in hotspot_caveats(hotspots, filtered, villages):
    st.caption(note)


# ---------------------------------------------------------------------------
# 7. Villages at risk
# ---------------------------------------------------------------------------
section(
    "village",
    "Villages at risk",
    f"Every conflict incident within {village_radius:g} km of each village, so a "
    "settlement between two hotspots is credited with both.",
)

village_risk = villages_at_risk(
    filtered, villages, hotspots, radius_km=village_radius, as_of=period_end
)

if village_risk.empty:
    st.info(
        "No village-level ranking available. This export contains no village field, "
        "so villages can only be identified from a centroids file — upload one in "
        "the sidebar (columns: Village, Latitude, Longitude). Sources include the "
        "Census village directory, the state forest department's own village "
        "layer, or OpenStreetMap place nodes."
    )
else:
    st.dataframe(
        village_risk,
        width="stretch",
        hide_index=True,
        column_config={
            "Tier": st.column_config.TextColumn(width="small"),
            "Night Share %": st.column_config.NumberColumn("Night", format="%.0f%%"),
            "Distance to Hotspot (km)": st.column_config.NumberColumn(
                "To hotspot", format="%.1f km"
            ),
            "Inside Hotspot": st.column_config.CheckboxColumn("In hotspot"),
        },
    )
    st.download_button(
        "Download village risk (CSV)",
        village_risk.to_csv(index=False).encode("utf-8"),
        "villages_at_risk.csv",
        mime="text/csv",
    )
    for note in village_caveats(village_risk, filtered, village_radius):
        st.caption(note)


# ---------------------------------------------------------------------------
# 8. Escalation & timing
# ---------------------------------------------------------------------------
esc_col, time_col = st.columns([1, 1])

with esc_col:
    section("trend", "Escalating", f"Last {recent_days} days vs the {recent_days} before")
    escalating = brief["escalating"]
    if escalating.empty:
        st.info("No beat shows a materially higher conflict count than the prior window.")
    else:
        st.dataframe(
            escalating[
                ["Beat", "Division", "Recent vs Prior", "Human Deaths", "People Injured"]
            ],
            width="stretch",
            hide_index=True,
        )

with time_col:
    section("clock", "Timing", "When to staff the shift")
    temporal = brief["temporal"]
    st.markdown(f"**Peak window:** {format_window(temporal['peak_window'])}")
    months = temporal["peak_months"]
    st.markdown(
        f"**Seasonal peak:** {', '.join(months)}"
        if months
        else "**Seasonal peak:** none marked - conflict is spread through the year."
    )
    hourly = temporal["hourly"]
    if float(hourly.sum()) > 0:
        fig = px.bar(
            x=hourly.index.astype(str),
            y=hourly.to_numpy(),
            labels={"x": "Hour of day (24h)", "y": "Conflict events"},
        )
        fig.update_traces(marker_color=BRAND)
        fig.update_layout(
            height=260,
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis_gridcolor="rgba(128,128,128,0.2)",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No timed conflict records, so no risk window can be identified.")


# ---------------------------------------------------------------------------
# 9. Map
# ---------------------------------------------------------------------------
section("map", "Spatial view", "Coloured by what happened, sized by severity")
render_map(filtered)


# ---------------------------------------------------------------------------
# 10. Trends & distributions
# ---------------------------------------------------------------------------
section("chart", "Supporting evidence", "The detail behind the rankings above")

trend_col1, trend_col2 = st.columns(2)

with trend_col1:
    st.markdown("**Monthly sightings vs. conflict events**")
    trend = monthly_trend(filtered)
    if trend.empty:
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
        st.plotly_chart(fig, width="stretch")

with trend_col2:
    st.markdown("**Conflict rate by division**")
    rates = division_conflict_rate(filtered)
    if rates.empty:
        st.info("No division data for the current filters.")
    else:
        st.dataframe(rates, width="stretch")
        st.caption(
            "Rate, not raw volume: sighting counts mostly track reporting effort."
        )

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    st.markdown("**Severity distribution**")
    bands = severity_distribution(filtered)
    if bands.empty or int(bands["Count"].sum()) == 0:
        st.info("No severity data to chart.")
    else:
        fig = px.bar(bands, x="Band", y="Count")
        fig.update_traces(marker_color=BRAND)
        fig.update_layout(
            height=320, margin=dict(t=10, b=10), showlegend=False,
            xaxis_title="", yaxis_title="Reports",
            yaxis_gridcolor="rgba(128,128,128,0.2)",
        )
        st.plotly_chart(fig, width="stretch")

with dist_col2:
    st.markdown("**Night vs. day**")
    nd = night_day_comparison(filtered)
    if nd.empty:
        st.info("Night/day classification unavailable (no Hour/Time column).")
    else:
        fig = px.bar(nd, x="Period", y="Entries", text="Entries")
        fig.update_traces(marker_color=BRAND)
        fig.update_layout(
            height=320, margin=dict(t=10, b=10), showlegend=False,
            xaxis_title="", yaxis_title="Reports",
            yaxis_gridcolor="rgba(128,128,128,0.2)",
        )
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# 11. Data preview + download
# ---------------------------------------------------------------------------
section("table", "Filtered data", f"{len(filtered):,} rows after filters")

preview_cols = [
    c
    for c in [
        "Date", "Division", "Range", "Beat", "Latitude", "Longitude",
        "Severity Score", "Is_Night", "Nearest Village", "Distance to Village (km)",
        "Crop Damage", "Grain Damage", "House Damage", "Injury", "Death",
    ]
    if c in filtered.columns
]
st.dataframe(filtered[preview_cols].head(500), width="stretch")
st.caption(f"Showing up to 500 of {len(filtered):,} filtered rows.")
st.download_button(
    "Download filtered data (CSV)",
    filtered.to_csv(index=False).encode("utf-8"),
    "filtered_elephant_data.csv",
    mime="text/csv",
)
st.warning(
    "Field exports of this kind carry victim and reporter names, often in the "
    "free-text description. Check before sharing a download outside the "
    "department — this app has no access control."
)


# ---------------------------------------------------------------------------
# 12. Report generation
# ---------------------------------------------------------------------------
section("report", "Intelligence brief", "Self-contained HTML for printing or email")

if st.button("Generate brief", type="primary"):
    try:
        html = generate_html_report(
            filtered,
            filtered["Date"].min(),
            filtered["Date"].max(),
            recent_days=recent_days,
            hotspots=hotspots,
            village_risk=village_risk,
        )
    except Exception as exc:  # noqa: BLE001 - report generation must never 500
        logger.exception("Report generation failed")
        st.error(f"Could not generate the brief: {exc}")
    else:
        st.download_button(
            "Download brief (HTML)", html, "Elephant_Conflict_Brief.html",
            mime="text/html",
        )
        st.success("Brief generated. Click the button above to download.")
