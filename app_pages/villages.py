"""Villages at risk, and early-warning coverage when a registry is loaded."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from app_pages.common import _ews
from core.exports import download_menu
from core.hotspots import village_caveats
from core.map_engine import render_village_map
from core.map_export import village_map_svg
from core.ui import findings, section


logger = logging.getLogger(__name__)


def render(c) -> None:
    basemap = c.basemap
    boundary_level = c.boundary_level
    coverage_stats = c.coverage_stats
    coverage_table = c.coverage_table
    filtered = c.history
    hotspots = c.hotspots
    layer_boundaries = c.layer_boundaries
    registry = c.registry
    roster = c.roster
    village_radius = c.village_radius
    village_risk = c.village_risk

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
            # The columns a warning decision needs, in half the page's width
            # without a scrollbar; house and crop counts are in the CSV and
            # on the map. Hotspot shows only when there are hotspots to name,
            # with the distance, since "nearest" alone may be far away.
            shown = village_risk.copy()
            village_cols = ["Village", "Tier", "Conflict Events", "Human Deaths",
                            "People Injured"]
            if (shown.get("Nearest Hotspot", pd.Series(dtype=object)) != "N/A").any() \
                    and "Distance to Hotspot (km)" in shown.columns:
                shown["Hotspot"] = [
                    f"inside {h}" if inside else f"{h}, {d:g} km"
                    for h, d, inside in zip(shown["Nearest Hotspot"],
                                            shown["Distance to Hotspot (km)"],
                                            shown.get("Inside Hotspot",
                                                      pd.Series(False, index=shown.index)))
                ]
                village_cols.append("Hotspot")
            st.dataframe(
                shown[[c for c in village_cols if c in shown.columns]],
                width="stretch",
                height=520,
                hide_index=True,
                column_config={
                    "Tier": st.column_config.TextColumn(width="small"),
                    "Conflict Events": st.column_config.NumberColumn(
                        "Incidents",
                        help=f"Conflict within {village_radius:g} km. Neighbouring villages "
                        "share incidents, so the column does not add up to the register."),
                    "Human Deaths": st.column_config.NumberColumn(
                        "Killed",
                        help=f"Deaths within {village_radius:g} km. One death counts for "
                        "every village near it, so a row means exposure, not loss."),
                    "People Injured": st.column_config.NumberColumn("Injured"),
                    "Hotspot": st.column_config.TextColumn(
                        "Nearest hotspot", help="The hotspot the village sits in, or the "
                        "nearest one and its distance."),
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
        # The overlap caveat is on the Incidents and Killed headers instead.
        for note in village_caveats(village_risk, filtered, village_radius, overlap=False):
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
