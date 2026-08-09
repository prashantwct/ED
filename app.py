"""Elephant Sighting & Conflict Dashboard.

Streamlit entry point. All data logic lives in core/ - this file is
responsible only for layout, widgets, and wiring user input to those
pure functions.
"""

from __future__ import annotations

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
from core.data_loader import load_and_validate_csv
from core.exceptions import DataValidationError, SpatialEnrichmentError
from core.intelligence import (
    DEFAULT_RECENT_DAYS,
    TIER_CRITICAL,
    TIER_HIGH,
    TIER_ORDER,
    TIER_WATCH,
    format_window,
    management_brief,
)
from core.map_engine import render_map
from core.report import generate_html_report
from core.spatial import attach_nearest_village, load_village_centroids

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Elephant Conflict Intelligence", layout="wide", page_icon="🐘"
)
st.title("🐘 Elephant Conflict Intelligence")
st.caption(
    "Sighting and conflict reporting turned into deployment decisions: which beats, "
    "which shift, which villages."
)

# Severity thresholds behind the "minimum incident type" filter. Exposing
# a raw 0-300 severity slider is unusable once fatalities are weighted at
# 100 points -- these bands are what a manager actually filters by.
SEVERITY_FILTERS = {
    "All reports": 0.0,
    "Conflict only (damage or worse)": 1.0,
    "Property damage or worse": 5.0,
    "Human harm only (injury or death)": 25.0,
    "Fatalities only": 100.0,
}

TIER_COLORS = {
    TIER_CRITICAL: "#b3261e",
    TIER_HIGH: "#e8751a",
    TIER_WATCH: "#c9a227",
    "Routine": "#46614f",
}

FILTER_KEYS = ["flt_dates", "flt_divisions", "flt_ranges", "flt_beats", "flt_severity"]


@st.cache_data(show_spinner="Loading and validating the export...")
def _load(file_bytes: bytes, filename: str):
    """Parse an uploaded CSV. Cached on content so filter changes don't re-parse."""
    import io

    return load_and_validate_csv(io.BytesIO(file_bytes))


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
    st.error(f"❌ Could not load the file: {exc}")
    st.stop()

if load_warnings:
    with st.expander(f"⚠️ {len(load_warnings)} data quality note(s)", expanded=True):
        for warning in load_warnings:
            st.warning(warning)

st.success(f"Loaded {len(raw_df):,} valid rows.")


# ---------------------------------------------------------------------------
# 2. Optional village-centroid enrichment
# ---------------------------------------------------------------------------
with st.sidebar.expander("📍 Village centroids (optional)", expanded=False):
    st.caption(
        "Attaches the nearest village to each sighting. Enables village-level "
        "exposure and the proximity component of the priority score."
    )
    centroid_upload = st.file_uploader(
        "Upload centroids.csv", type="csv", key="centroids_upload"
    )
    st.caption("Falls back to a local `centroids.csv` next to app.py if present.")

try:
    villages = load_village_centroids(centroid_upload if centroid_upload else None)
except SpatialEnrichmentError as exc:
    st.sidebar.warning(f"⚠️ Village centroids not applied: {exc}")
    villages = None

df, spatial_warnings = attach_nearest_village(raw_df, villages)
for warning in spatial_warnings:
    st.warning(f"⚠️ {warning}")
if villages is not None:
    st.sidebar.success(f"✅ Enriched with {len(villages):,} village centroids.")
else:
    st.sidebar.info("No village centroids loaded - village exposure unavailable.")

# ---------------------------------------------------------------------------
# 3. Derived columns
# ---------------------------------------------------------------------------
df = df.copy()
df["Severity Score"] = compute_severity(df)
df["Is_Night"] = compute_is_night(df)


# ---------------------------------------------------------------------------
# 4. Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔎 Filters")

if st.sidebar.button("Reset filters"):
    # Clearing the widget keys is what actually resets them; a bare
    # st.rerun() re-runs the script with every selection still in
    # session state, so the button appears to do nothing.
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

# Cascading filters need their stale selections pruned *before* the
# dependent widget is created. Widening Division back out otherwise
# leaves Range holding values that are no longer offered, silently
# excluding rows while the UI looks reset.
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
recent_days = st.sidebar.slider(
    "Escalation window (days)",
    min_value=30,
    max_value=180,
    value=DEFAULT_RECENT_DAYS,
    step=30,
    help="Each beat's most recent N days are compared against the N days before them.",
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
# 5. KPIs
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

brief = management_brief(filtered, recent_days=recent_days)


# ---------------------------------------------------------------------------
# 6. The brief
# ---------------------------------------------------------------------------
st.subheader("🧭 Assessment")
st.markdown("\n".join(f"- {line}" for line in brief["headlines"]))

with st.expander("How to read this", expanded=False):
    for caveat in brief["caveats"]:
        st.markdown(f"- {caveat}")


# ---------------------------------------------------------------------------
# 7. Beat priorities
# ---------------------------------------------------------------------------
st.subheader("🎯 Beat Priorities")
st.caption(
    "Tier is set by fixed rules and means the same thing across periods and filters. "
    "The score only orders beats within a tier."
)

beats_table = brief["beats"]
if beats_table.empty:
    st.info("Not enough data to rank beats.")
else:
    tier_counts = beats_table["Priority Tier"].value_counts()
    tier_cols = st.columns(len(TIER_ORDER))
    for col, tier in zip(tier_cols, TIER_ORDER):
        col.markdown(
            f"<div style='border-left:4px solid {TIER_COLORS[tier]};padding-left:10px;'>"
            f"<div style='font-size:12px;color:#667;text-transform:uppercase;'>{tier}</div>"
            f"<div style='font-size:22px;font-weight:700;'>{int(tier_counts.get(tier, 0))}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    display_cols = [
        "Beat", "Division", "Range", "Priority Tier", "Priority Score", "Confidence",
        "Reports", "Conflict Events", "Conflict Rate %", "Adj. Conflict Rate %",
        "Human Deaths", "People Injured", "Night Conflict %", "Near Village %",
        "Trend", "Recent vs Prior", "Recommended Action",
    ]
    st.dataframe(
        beats_table[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "Priority Score": st.column_config.ProgressColumn(
                "Priority Score", min_value=0, max_value=100, format="%.0f"
            ),
            "Conflict Rate %": st.column_config.NumberColumn(format="%.1f%%"),
            "Adj. Conflict Rate %": st.column_config.NumberColumn(
                "Adj. rate %",
                format="%.1f%%",
                help="Conflict rate shrunk toward the landscape average, so beats "
                "with very few reports cannot top the ranking on one incident.",
            ),
            "Night Conflict %": st.column_config.NumberColumn(format="%.0f%%"),
            "Near Village %": st.column_config.NumberColumn(format="%.0f%%"),
            "Recommended Action": st.column_config.TextColumn(width="large"),
        },
    )
    st.download_button(
        "⬇️ Download beat priorities (CSV)",
        beats_table.to_csv(index=False).encode("utf-8"),
        "beat_priorities.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# 8. Escalation & timing
# ---------------------------------------------------------------------------
esc_col, time_col = st.columns([1, 1])

with esc_col:
    st.subheader("📈 Escalating")
    st.caption(f"Last {recent_days} days vs the {recent_days} days before.")
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
    st.subheader("🕒 Timing")
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
            labels={"x": "Hour of day", "y": "Conflict events"},
        )
        fig.update_traces(marker_color="#1f5f3f")
        fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No timed conflict records, so no risk window can be identified.")


# ---------------------------------------------------------------------------
# 9. Village exposure
# ---------------------------------------------------------------------------
st.subheader("🏘️ Village Exposure")
village_table = brief["villages"]
if village_table.empty:
    st.info(
        "No village-centroid data applied. Upload a centroids file in the sidebar to "
        "rank villages by the conflict recorded around them."
    )
else:
    st.dataframe(village_table, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# 10. Map
# ---------------------------------------------------------------------------
st.subheader("📍 Spatial View")
render_map(filtered)


# ---------------------------------------------------------------------------
# 11. Trends & distributions
# ---------------------------------------------------------------------------
st.subheader("📊 Trends")
trend_col1, trend_col2 = st.columns(2)

with trend_col1:
    st.markdown("**Monthly sightings vs. conflict events**")
    trend = monthly_trend(filtered)
    if trend.empty:
        st.info("Not enough data to chart a monthly trend.")
    else:
        fig = px.line(
            trend.reset_index(),
            x="Month",
            y=["Sightings", "Conflict Events"],
            markers=True,
        )
        fig.update_layout(
            height=320, margin=dict(t=10, b=10), legend_title_text="", yaxis_title=""
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
    if bands.empty:
        st.info("No severity data to chart.")
    else:
        fig = px.bar(bands, x="Band", y="Count", color="Count", color_continuous_scale="OrRd")
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

with dist_col2:
    st.markdown("**Night vs. day**")
    nd = night_day_comparison(filtered)
    if nd.empty:
        st.info("Night/day classification unavailable (no Hour/Time column).")
    else:
        fig = px.bar(nd, x="Period", y="Entries", color="Period", text="Entries")
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# 12. Data preview + download
# ---------------------------------------------------------------------------
st.subheader("📋 Filtered Data")
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
    "⬇️ Download filtered data (CSV)",
    filtered.to_csv(index=False).encode("utf-8"),
    "filtered_elephant_data.csv",
    mime="text/csv",
)
st.caption(
    "⚠️ Field exports of this kind often carry victim and reporter names. Check before "
    "sharing a download outside the department - this app has no access control."
)


# ---------------------------------------------------------------------------
# 13. Report generation
# ---------------------------------------------------------------------------
st.subheader("🧾 Intelligence Brief")
if st.button("Generate brief"):
    try:
        html = generate_html_report(
            filtered,
            filtered["Date"].min(),
            filtered["Date"].max(),
            recent_days=recent_days,
        )
    except Exception as exc:  # noqa: BLE001 - report generation must never 500
        logger.exception("Report generation failed")
        st.error(f"❌ Could not generate the brief: {exc}")
    else:
        st.download_button(
            "⬇️ Download brief (HTML)",
            html,
            "Elephant_Conflict_Brief.html",
            mime="text/html",
        )
        st.success("Brief generated. Click the button above to download.")
