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
    filter_dataframe,
    night_day_comparison,
    severity_distribution,
)
from core.data_loader import load_and_validate_csv
from core.exceptions import DataValidationError, SpatialEnrichmentError
from core.map_engine import render_map
from core.report import generate_html_report
from core.risk_engine import beat_risk_scores
from core.spatial import attach_nearest_village, load_village_centroids

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Elephant Dashboard", layout="wide", page_icon="🐘")
st.title("🐘 Elephant Sighting & Conflict Dashboard")


# ---------------------------------------------------------------------------
# 1. Upload & load
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Upload sightings/conflict CSV", type="csv")

if not uploaded:
    st.info(
        "Please upload a CSV file to continue. Required columns: "
        "**Date, Latitude, Longitude, Division, Range, Beat**. "
        "Optional columns (unlock extra features): "
        "Time or Hour, Total Count, Crop Damage, House Damage, Injury."
    )
    st.stop()

try:
    raw_df, load_warnings = load_and_validate_csv(uploaded)
except DataValidationError as exc:
    st.error(f"❌ Could not load the file: {exc}")
    st.stop()

for warning in load_warnings:
    st.warning(f"⚠️ {warning}")

st.success(f"Loaded {len(raw_df):,} valid rows.")


# ---------------------------------------------------------------------------
# 2. Optional village-centroid enrichment
# ---------------------------------------------------------------------------
with st.sidebar.expander("📍 Village centroids (optional)", expanded=False):
    st.caption(
        "Attaches the nearest village to each sighting, used in map tooltips "
        "and the beat risk score. Skipped entirely if not provided."
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

# ---------------------------------------------------------------------------
# 3. Derived columns
# ---------------------------------------------------------------------------
df["Severity Score"] = compute_severity(df)
df["Is_Night"] = compute_is_night(df)


# ---------------------------------------------------------------------------
# 4. Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔎 Filters")

min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) != 2:
    date_range = (min_date, max_date)

division_options = sorted(df["Division"].unique())
divisions = st.sidebar.multiselect("Division", division_options)

range_pool = df[df["Division"].isin(divisions)] if divisions else df
range_options = sorted(range_pool["Range"].unique())
ranges = st.sidebar.multiselect("Range", range_options)

beat_pool = range_pool[range_pool["Range"].isin(ranges)] if ranges else range_pool
beat_options = sorted(beat_pool["Beat"].unique())
beats = st.sidebar.multiselect("Beat", beat_options)

max_possible_severity = float(df["Severity Score"].max()) if not df.empty else 0.0
min_severity = st.sidebar.slider(
    "Minimum severity",
    min_value=0.0,
    max_value=max(max_possible_severity, 1.0),
    value=0.0,
    step=0.5,
)

filtered = filter_dataframe(
    df,
    date_range=date_range,
    divisions=divisions,
    ranges=ranges,
    beats=beats,
    min_severity=min_severity,
)

if st.sidebar.button("Reset filters"):
    st.rerun()


# ---------------------------------------------------------------------------
# 5. KPIs
# ---------------------------------------------------------------------------
kpis = compute_kpis(filtered)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Entries", f"{kpis['entries']:,}")
c2.metric("Conflicts", f"{kpis['conflicts']:,}")
c3.metric("Total Severity", f"{kpis['severity']:.1f}")
c4.metric(
    "Night Incidents",
    "N/A" if pd.isna(kpis["night_pct"]) else f"{kpis['night_pct']:.1f}%",
)

if filtered.empty:
    st.warning("No rows match the current filters. Adjust the filters in the sidebar.")
    st.stop()


# ---------------------------------------------------------------------------
# 6. Data preview + download
# ---------------------------------------------------------------------------
st.subheader("📋 Filtered Data Preview")
preview_cols = [
    c
    for c in [
        "Date", "Division", "Range", "Beat", "Latitude", "Longitude",
        "Severity Score", "Is_Night", "Nearest Village", "Distance to Village (km)",
        "Crop Damage", "House Damage", "Injury",
    ]
    if c in filtered.columns
]
st.dataframe(filtered[preview_cols].head(500), use_container_width=True)
st.caption(f"Showing up to 500 of {len(filtered):,} filtered rows.")
st.download_button(
    "⬇️ Download filtered data (CSV)",
    filtered.to_csv(index=False).encode("utf-8"),
    "filtered_elephant_data.csv",
    mime="text/csv",
)


# ---------------------------------------------------------------------------
# 7. Map
# ---------------------------------------------------------------------------
st.subheader("📍 Spatial View")
render_map(filtered)


# ---------------------------------------------------------------------------
# 8. Beat risk ranking
# ---------------------------------------------------------------------------
st.subheader("⚠️ Beat Risk Ranking")
risk_table = beat_risk_scores(filtered)
if risk_table.empty:
    st.info("Not enough data to compute beat risk scores.")
else:
    st.dataframe(
        risk_table.style.background_gradient(subset=["Risk Score"], cmap="RdYlGn_r"),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# 9. Charts
# ---------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Severity Distribution**")
    bands = severity_distribution(filtered)
    if bands.empty:
        st.info("Not enough variation in severity to chart.")
    else:
        fig = px.bar(bands, x="Band", y="Count", color="Count", color_continuous_scale="OrRd")
        fig.update_layout(showlegend=False, height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.markdown("**Night vs. Day Comparison**")
    nd = night_day_comparison(filtered)
    if nd.empty:
        st.info("Night/day classification unavailable (no Hour/Time column).")
    else:
        fig = px.bar(nd, x="Period", y="Entries", color="Period", text="Entries")
        fig.update_layout(showlegend=False, height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 10. Report generation
# ---------------------------------------------------------------------------
st.subheader("🧾 Report")
if st.button("Generate Report"):
    try:
        html = generate_html_report(
            filtered, filtered["Date"].min(), filtered["Date"].max()
        )
    except Exception as exc:  # noqa: BLE001 - report generation must never 500
        logger.exception("Report generation failed")
        st.error(f"❌ Could not generate the report: {exc}")
    else:
        st.download_button(
            "⬇️ Download Report",
            html,
            "Elephant_Report.html",
            mime="text/html",
        )
        st.success("Report generated. Click the button above to download.")
