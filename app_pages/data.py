"""Data & brief: the brief, the full assessment, what the numbers mean, the rows.

Everything here was on the old single page around the tabs -- the
findings list, "How to read this", the data-quality notes, the Generate
brief button -- gathered where someone checking the figures looks for it.
"""

from __future__ import annotations

import logging

import plotly.express as px
import streamlit as st

from app_pages.brief import brief_button
from app_pages.common import BRAND, BRAND_ALT, _figure
from core import report_charts, spacetime
from core.map_export import filter_summary
from core.ui import findings, section
from core.analytics import division_conflict_rate, monthly_trend, severity_distribution


logger = logging.getLogger(__name__)


def render(c) -> None:
    escalating = c.escalating
    filtered = c.focus
    history = c.history

    section("report", "The brief",
            "One HTML file with the maps embedded, for printing or email.")
    scope_col, brief_col = st.columns([3, 1], vertical_alignment="center")
    # filter_summary returns escaped HTML; the separators are markup.
    scope_col.markdown(
        f"Covers **{c.window.label()}** as of its last day, with rankings and trends "
        f"from the register before it. Scope: {filter_summary({'divisions': c.divisions, 'ranges': c.ranges, 'beats': c.beats, 'severity': c.severity_choice, 'analysis': c.custom_analysis})}.",  # noqa: E501
        unsafe_allow_html=True,
    )
    with brief_col:
        brief_button(c, "data_brief")

    section("assessment", "Assessment",
            f"{len(history):,} reports up to {c.window.end.day} {c.window.end:%b %Y}, "
            "as the brief states it")
    emerging_line = spacetime.headline(c.emerging)
    findings(c.brief["headlines"] + ([emerging_line] if emerging_line else []))

    with st.expander("About these numbers", expanded=False):
        for caveat in c.brief["caveats"]:
            st.markdown(f"- {caveat}")

    with st.expander(f"Data notes ({len(c.data_notes)})", expanded=False):
        for note in c.data_notes:
            st.markdown(f"- {note}")
        if not c.data_notes:
            st.markdown("Nothing unusual was found while reading the register.")
        for line in c.data_about:
            st.caption(line)

    section("table", "The reports",
            f"{len(filtered):,} in {c.window.label()}; trends over the whole history")

    trend_col, sev_col = st.columns(2)
    with trend_col:
        trend = monthly_trend(history)
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
                         "line, or the rate on the Patterns page.")
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

    rates = division_conflict_rate(history)
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

    st.markdown(f"**Reports in this period** -- {len(filtered):,} rows")
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
