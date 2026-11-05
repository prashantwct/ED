"""The brief: one self-contained HTML file, from the Overview or Data & brief."""

from __future__ import annotations

import logging

import streamlit as st

from core.report import generate_html_report

logger = logging.getLogger(__name__)


def brief_button(c, key: str) -> None:
    """Generate brief, then a download for it, in the same place.

    The brief analyses the same history the page ranks from, so its tiers
    and trends match what is on screen, and it says so in its scope line.
    """
    if not st.button("Generate brief", type="primary", width="stretch", key=key,
                     help="Self-contained HTML with the maps embedded, for printing or "
                     "email."):
        return
    from core import refuge_view

    try:
        try:
            refuges = refuge_view.for_brief(c.history, c.villages)
        except Exception:  # noqa: BLE001 - the brief goes out without the section
            logger.exception("Day refuges for the brief failed")
            refuges = None
        html = generate_html_report(
            c.history,
            c.history["Date"].min(),
            c.history["Date"].max(),
            recent_days=c.recent_days,
            hotspots=c.hotspots,
            village_risk=c.village_risk,
            filters={
                "divisions": c.divisions,
                "ranges": c.ranges,
                "beats": c.beats,
                "severity": c.severity_choice,
                "analysis": c.custom_analysis,
            },
            basemap_style_name=c.basemap,
            coverage=c.coverage_table,
            coverage_stats=c.coverage_stats,
            refuges=refuges,
            emerging=c.emerging,
            emerging_step=c.cube_step,
        )
    except Exception as exc:  # noqa: BLE001 - report generation must never 500
        logger.exception("Report generation failed")
        st.error(f"Could not generate the brief: {exc}")
    else:
        st.download_button(
            "Download brief (HTML)", html, "Elephant_Conflict_Brief.html",
            mime="text/html", width="stretch", key=f"{key}_download",
        )
        st.success("Brief ready: download it above.")
