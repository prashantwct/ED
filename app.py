import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.analytics import (
    classify_conflict,
    compute_kpis,
    compute_severity,
    division_conflict_rate,
    hourly_conflict_profile,
    monthly_trend,
)
from core.data_loader import load_and_validate_csv
from core.map_engine import CATEGORY_COLORS, render_map
from core.report import generate_html_report
from core.risk_engine import beat_risk_scores
from core.spatial import attach_nearest_village

st.set_page_config(page_title="Elephant Dashboard", layout="wide", page_icon="🐘")
st.title("🐘 Elephant Sighting & Conflict Dashboard")

uploaded = st.file_uploader("Upload CSV", type="csv")

if not uploaded:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df, load_warnings = load_and_validate_csv(uploaded)

if load_warnings:
    with st.expander(f"⚠️ Data quality notes ({len(load_warnings)})", expanded=False):
        for w in load_warnings:
            st.warning(w)

if df.empty:
    st.error("No usable rows remained after validation. See the data quality notes above.")
    st.stop()

# ---------------------------------------------------------------- filters
st.sidebar.header("Filters")

divisions = sorted(df["Division"].unique())
if "division_filter" not in st.session_state:
    st.session_state.division_filter = divisions


def _sync_ranges_to_division():
    # Without this, widening Division back out after narrowing it leaves
    # Range holding its stale, narrower selection -- the filter looks
    # "reset" but silently keeps excluding rows.
    pool = df[df["Division"].isin(st.session_state.division_filter)] if st.session_state.division_filter else df
    st.session_state.range_filter = sorted(pool["Range"].unique())


selected_divisions = st.sidebar.multiselect(
    "Division", divisions, key="division_filter", on_change=_sync_ranges_to_division
)

range_pool = df[df["Division"].isin(selected_divisions)] if selected_divisions else df
ranges = sorted(range_pool["Range"].unique())
if "range_filter" not in st.session_state:
    st.session_state.range_filter = ranges
# Drop any selected range that's no longer valid for the current division set.
st.session_state.range_filter = [r for r in st.session_state.range_filter if r in ranges]
selected_ranges = st.sidebar.multiselect("Range", ranges, key="range_filter")

min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
date_selection = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
if isinstance(date_selection, tuple) and len(date_selection) == 2:
    start_date, end_date = date_selection
else:
    start_date, end_date = min_date, max_date

mask = (
    df["Division"].isin(selected_divisions)
    & df["Range"].isin(selected_ranges)
    & (df["Date"].dt.date >= start_date)
    & (df["Date"].dt.date <= end_date)
)
fdf = df.loc[mask].copy()

if fdf.empty:
    st.warning("No sightings match the current filters.")
    st.stop()

# ---------------------------------------------------------------- pipeline
fdf["Severity Score"] = compute_severity(fdf)
fdf["Conflict Category"] = classify_conflict(fdf)
fdf = attach_nearest_village(fdf)
if "Near Village" not in fdf.columns:
    st.caption("Village-proximity scoring is inactive: no centroids reference file was found.")

kpis = compute_kpis(fdf)
div_rates = division_conflict_rate(fdf)
trend = monthly_trend(fdf)
hourly = hourly_conflict_profile(fdf)
priority_beats = beat_risk_scores(fdf)

# ---------------------------------------------------------------- KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Entries", kpis["entries"])
c2.metric("Conflict-related", kpis["conflicts"])
c3.metric("Human deaths", f"{kpis['human_deaths']:.0f}", help=f"{kpis['human_death_incidents']} incident(s)")
c4.metric("Conflict at night", f"{kpis['night_pct']:.0f}%", help="Share of conflict events between 7 PM and 2 AM")

# ---------------------------------------------------------------- trends
st.subheader("📈 Trends")
t1, t2 = st.columns(2)

with t1:
    fig = go.Figure()
    fig.add_bar(x=trend.index, y=trend["Sightings"], name="All sightings", marker_color="#8A9A5B")
    fig.add_bar(x=trend.index, y=trend["Conflict Events"], name="Conflict-related", marker_color="#C97A2B")
    fig.update_layout(barmode="overlay", title="Monthly volume", height=340, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width='stretch')

with t2:
    colors = ["#C97A2B" if (h >= 19 or h <= 2) else "#8A9A5B" for h in hourly.index]
    fig2 = go.Figure(go.Bar(x=list(hourly.index), y=hourly.values, marker_color=colors))
    fig2.update_layout(
        title="Conflict events by hour of day", xaxis_title="Hour", height=340, margin=dict(t=40, b=10)
    )
    st.plotly_chart(fig2, width='stretch')

# ---------------------------------------------------------------- priority beats
st.subheader("🎯 Priority beats")
st.caption("Ranked by accumulated severity, night share, and village proximity -- highest risk first.")
st.dataframe(priority_beats.head(15), width='stretch', hide_index=True)

# ---------------------------------------------------------------- map
st.subheader("📍 Spatial view")
render_map(fdf)

# ---------------------------------------------------------------- division rates
st.subheader("🏞️ Conflict rate by division")
st.caption("Share of sightings involving conflict -- not just raw sighting volume.")
st.dataframe(div_rates, width='stretch')

# ---------------------------------------------------------------- report
if st.button("Generate Report"):
    html = generate_html_report(
        fdf, start_date, end_date, kpis, division_rates=div_rates, top_beats=priority_beats
    )
    st.download_button("Download Report", html, "Elephant_Report.html", mime="text/html")
