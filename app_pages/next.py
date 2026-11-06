"""Next moves: where groups reported in the last few days may be next.

A forecast from the register alone, with its own track record beside it.
Groups are movement units chained from reports (core.movement), not
identified animals. The forecast reads every report up to the end of the
period, whatever the incident type, because a group's quiet sightings
say as much about where it goes as its raids do; the area in the scope
bar decides which groups are shown.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from app_pages.common import movement_record, next_moves_for
from core import movement as mv
from core.brief_insights import names, plural
from core.map_engine import ZONE_COLORS, render_forecast_map
from core.ui import map_legend, section

HORIZONS = {"Next 24 hours": 1, "Next 2 days": 2, "Next 3 days": 3}


def _in_scope(groups: pd.DataFrame, c) -> pd.DataFrame:
    for column, chosen in (("Division", c.divisions), ("Range", c.ranges), ("Beat", c.beats)):
        if chosen and column in groups:
            groups = groups[groups[column].isin(chosen)]
    return groups


def _track_record(record: mv.TrackRecord) -> None:
    if record.tested == 0:
        st.info("Not enough history to test the forecast on this register yet: it needs "
                "about three months of reports. Treat the zones as a rough guide.")
        return
    said = (f"Tested on this register: over {plural(record.months, 'month')}, the likely "
            f"zone (on average {record.mean_area_km2:.0f} km²) held the group's next "
            f"report **{record.forecast_rate:.0%}** of the time, against "
            f"**{record.stay_rate:.0%}** for a circle of the same size around where it was "
            f"last seen. {record.tested:,} moves, each forecast only from the months "
            "before it.")
    if record.beats_staying_put:
        st.markdown(f":material/verified: {said}")
    else:
        st.warning(said + " On this register the forecast is no better than assuming a "
                   "group stays near where it was last seen: read each zone as an area "
                   "around the last report, not a direction.", icon=":material/info:")


def render(c) -> None:
    section("hotspot", "Next moves",
            f"Where groups reported in the {mv.ACTIVE_DAYS} days to "
            f"{c.window.end.day} {c.window.end:%b %Y} may be next")
    choice = st.segmented_control("Forecast for", list(HORIZONS), default="Next 24 hours",
                                  key="next_horizon") or "Next 24 hours"
    days = HORIZONS[choice]

    register = c.df[c.df["Date"] < c.period_end + pd.Timedelta(days=1)]
    record = movement_record(register, days)
    out = next_moves_for(register, c.period_end, days, c.villages, record.prior_weight)
    groups = _in_scope(out["groups"], c) if len(out["groups"]) else out["groups"]
    _track_record(record)

    if groups.empty:
        st.info(f"No group was reported in this area in the {mv.ACTIVE_DAYS} days to "
                f"{c.window.end.day} {c.window.end:%b %Y}, so there is nothing to forecast. "
                "Widen the area, or move the period's end to a day with reports.")
        return

    shown = set(groups["Group"])
    villages = out["villages"]
    villages = villages[villages["Group"].isin(shown)] if len(villages) else villages
    trails = out["trails"]
    trails = trails[trails["Group"].isin(shown)] if len(trails) else trails
    likely = villages[villages["Band"] == "Likely"] if len(villages) else villages
    headline = (f"{plural(len(groups), 'group')} reported in the last {mv.ACTIVE_DAYS} days. "
                + (f"Their likely zones reach {plural(len(likely), 'village')}: warn "
                   f"{names(likely['Village'], limit=6)}." if len(likely)
                   else "No village sits in a likely zone."))
    st.markdown(f'<p class="ci-headline">{escape(headline)}</p>', unsafe_allow_html=True)

    map_col, table_col = st.columns([3, 2], gap="medium")
    with map_col:
        render_forecast_map(out["cells"], groups, trails, villages, out.get("cell_km", 1.0),
                            style_name=c.basemap, show_boundaries=c.layer_boundaries,
                            boundary_level=c.boundary_level)
        map_legend("Zones", {band: {"accent": "#%02x%02x%02x" % rgb, "glyph": "",
                                    "label": f"{band} (holds {mass:.0%})"}
                             for (band, rgb), mass in zip(ZONE_COLORS.items(),
                                                          (mv.LIKELY_MASS, mv.POSSIBLE_MASS))})
        st.caption("Green dots are each group's last report, with its recent trail. "
                   "Villages in a likely zone are labelled.")
    with table_col:
        st.markdown("**Groups**")
        st.dataframe(
            groups[["Group", "Class", "Size", "Last Seen", "Near", "Likely Zone (km2)",
                    "Villages in Likely Zone"]],
            hide_index=True, width="stretch",
            column_config={
                "Last Seen": st.column_config.DateColumn("Last seen", format="D MMM"),
                "Likely Zone (km2)": st.column_config.NumberColumn(
                    "Likely zone", format="%.0f km²",
                    help="The smallest area holding half the forecast."),
                "Villages in Likely Zone": st.column_config.TextColumn(
                    "Villages in it", width="large"),
            },
        )
        if len(villages):
            st.markdown("**Villages to warn**")
            st.dataframe(
                villages[["Village", "Band", "Group", "From last report (km)"]],
                hide_index=True, width="stretch",
                column_config={"Band": st.column_config.TextColumn(
                    "Zone", help="Likely: in the area holding half the forecast. Possible: "
                    "in the area holding 80%.")},
            )
            st.download_button("Download villages to warn (CSV)",
                               villages.to_csv(index=False).encode("utf-8"),
                               "next_moves_villages.csv", mime="text/csv")

    with st.expander("How this is worked out, and what it cannot do", expanded=False):
        st.markdown(
            f"- **Groups** are chained from reports by time, place and group composition: "
            f"a report joins a track if it is within reach for the time elapsed (up to "
            f"{mv.MAX_SPEED_KM_PER_DAY:g} km a day) and the group could be the same. They "
            "are plausible movement units, not identified animals: two bulls in one valley "
            "can be one group here, and a group is lost after "
            f"{mv.MAX_GAP_DAYS} days without a report.\n"
            f"- **How far**: every move between a group's reports a day or more apart, "
            f"{out.get('steps_learned', 0):,} of them on this register, sets how far a group "
            "goes in a day.\n"
            "- **Where**: places with earlier reports, recent ones counting more, are where "
            "groups tend to go. How much that counts is chosen from past months: "
            f"{record.prior_weight:g} on a scale from 0 (ignored) to 1.\n"
            "- **The zones** hold half (likely) and 80% (possible) of the forecast. They "
            "are areas to watch, not promises: a forecast cannot see a first visit to a "
            "place nobody has reported elephants in.\n"
            "- **The track record** forecasts each month from the months before it and "
            "checks where each group was reported next.")
