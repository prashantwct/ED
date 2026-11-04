"""When? Peak hours, night raids, the seasonal peak."""

from __future__ import annotations

import logging

import plotly.express as px
import streamlit as st

from app_pages.common import BRAND, _figure
from core import charts, report_charts
from core.analytics import night_day_comparison
from core.config import NIGHT_HOUR_END, NIGHT_HOUR_START
from core.intelligence import format_window


logger = logging.getLogger(__name__)


def render(c) -> None:
    filtered = c.history
    temporal = c.temporal

    months = temporal["peak_months"]
    kpis = c.kpis_history
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Injuries and deaths peak",
              format_window(temporal["peak_window"]).split(" (")[0]
              if temporal["peak_window"] else "Not enough timed casualties",
              border=True)
    w2.metric("Counted as night raids", f"{temporal.get('night_damage', 0):,}",
              delta="damage and indirect reports", delta_color="off", border=True)
    w3.metric("Seasonal peak", ", ".join(months) if months else "Spread through the year",
              border=True)
    # Measured on direct sightings with an hour. Signs and damage are night
    # by rule (they are written up the morning after), so counting them would
    # report the rule back as a finding.
    w4.metric(
        "Night sightings",
        "N/A" if kpis["sighting_night_pct"] != kpis["sighting_night_pct"]
        else f"{kpis['sighting_night_pct']:.0f}%",
        delta=(f"of {kpis['sightings_timed']:,} timed" if kpis["sightings_timed"] else None),
        delta_color="off", border=True,
        help="Direct sightings 18:00-06:00; signs and damage are left out, their hour is "
        "the write-up.",
    )
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
