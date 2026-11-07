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

import numpy as np
import pandas as pd
import streamlit as st

from app_pages.common import movement_training, next_moves_for
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
                "about three months of reports. Until then the zones are a circle of typical "
                "daily travel around each group's last report, not a tested forecast.")
        return
    uses = record.config.parts()
    said = (f"Tested on this register: {record.tested:,} moves over "
            f"{plural(record.months, 'month')}, each forecast only from the months before. "
            f"The likely zone held the group's next report **{record.forecast_rate:.0%}** of "
            f"the time (it aims for half), averaging {record.mean_area_km2:.0f} km\u00b2")
    if record.circle_area_km2 == record.circle_area_km2:
        said += (f"; a circle around the last report needed "
                 f"{record.circle_area_km2:.0f} km\u00b2 to do as well")
    said += (f". At the same size, the circle held {record.stay_rate:.0%}. "
             + ("It uses " + ", ".join(uses) + "." if uses else
                "On this register only the distance groups travel helped."))
    if record.beats_staying_put:
        st.markdown(f":material/verified: {said}")
    else:
        st.warning(said + " That is no better than assuming a group stays near where it "
                   "was last seen: read each zone as an area around the last report, not a "
                   "direction.", icon=":material/info:")


def _model_status(training: mv.Training) -> None:
    """What the forecast was trained on, and what this load had to retrain."""
    if not training.reports:
        return
    span = (f"{training.first_report.day} {training.first_report:%b %Y} to "
            f"{training.last_report.day} {training.last_report:%b %Y}")
    said = (f"Trained on {training.reports:,} reports from {span}: "
            f"{training.moves:,} moves, {plural(len(training.months), 'month')} forecast "
            "and checked. ")
    if not training.months:
        said += "It retrains whenever the register changes."
    elif training.retrained_months == 0:
        said += "Up to date with the register on screen."
    elif training.reused_months == 0:
        said += "Trained from scratch on this register; it retrains whenever entries change."
    else:
        said += (f"New or changed entries retrained the last "
                 f"{plural(training.retrained_months, 'month')}; the "
                 f"{plural(training.reused_months, 'earlier month')} carried over unchanged.")
    st.caption(said)


def _turn_phrase(turns: dict) -> str:
    """What the register says about direction, in words."""
    if not turns or 1 not in turns:
        return "too few moves to say which way groups go next"
    mu, kappa = turns[1]
    if kappa < 0.3:
        return "no settled direction from one day to the next"
    degrees = abs(np.degrees(mu))
    if degrees < 45:
        return "groups mostly keep going the way they were heading"
    if degrees > 135:
        return "groups mostly come back the way they went, as between a day refuge and fields"
    return "groups mostly turn off their last direction"


def render(c) -> None:
    section("hotspot", "Next moves",
            f"Where groups reported in the {mv.ACTIVE_DAYS} days to "
            f"{c.window.end.day} {c.window.end:%b %Y} may be next")
    choice = st.segmented_control("Forecast for", list(HORIZONS), default="Next 24 hours",
                                  key="next_horizon") or "Next 24 hours"
    days = HORIZONS[choice]

    register = c.df[c.df["Date"] < c.period_end + pd.Timedelta(days=1)]
    training = movement_training(register, days)
    record = training.record
    out = next_moves_for(register, c.period_end, days, c.villages, record,
                         (record.config, record.masses, record.tested))
    groups = _in_scope(out["groups"], c) if len(out["groups"]) else out["groups"]
    _track_record(record)
    _model_status(training)

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
                                    "label": f"{band} (catches about {share})"}
                             for (band, rgb), share in zip(ZONE_COLORS.items(),
                                                           ("1 in 2", "4 in 5"))})
        st.caption("Green dots are each group's last report, with its recent trail; the "
                   "short line points to its likeliest spot. Villages in a likely zone are "
                   "labelled.")
    with table_col:
        st.markdown("**Groups**")
        st.dataframe(
            groups[["Group", "Most Likely", "Heading", "Last Seen", "Class", "Size",
                    "Likely Zone (km2)", "Villages in Likely Zone"]],
            hide_index=True, width="stretch",
            column_config={
                "Last Seen": st.column_config.DateColumn("Last seen", format="D MMM"),
                "Heading": st.column_config.TextColumn(
                    "Heading", help="Direction of its last move, a day or more apart."),
                "Most Likely": st.column_config.TextColumn(
                    "Most likely", help="The single likeliest kilometre square, from its "
                    "last report.", width="medium"),
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
        likely_mass, possible_mass = out.get("masses", (mv.LIKELY_MASS, mv.POSSIBLE_MASS))
        st.markdown(
            f"- **Groups** are chained from reports by time, place and group composition: "
            f"a report joins a track if it is within reach for the time elapsed (up to "
            f"{mv.MAX_SPEED_KM_PER_DAY:g} km a day) and the group could be the same. They "
            "are plausible movement units, not identified animals: two bulls in one valley "
            "can be one group here, and a group is lost after "
            f"{mv.MAX_GAP_DAYS} days without a report.\n"
            f"- **How far**: {out.get('steps_learned', 0):,} moves between a group's reports "
            "a day or more apart set how far a group goes in a day, separately for lone "
            "bulls, bull parties and family herds once each has enough moves of its own.\n"
            f"- **Which way**: on this register, {_turn_phrase(out.get('turns', {}))}.\n"
            "- **Where**: places with earlier reports, recent ones counting more and "
            "optionally the same weeks of earlier years, and each group's own earlier "
            "reports.\n"
            "- **Which of these to use** is chosen from past months: the combination that "
            "forecast them best. On this register: "
            + (", ".join(record.config.parts()) if record.config.parts()
               else "distance alone") + ".\n"
            f"- **The zones** are sized from past months so that they catch about half "
            f"(likely) and 80% (possible) of next reports: here they hold "
            f"{likely_mass:.0%} and {possible_mass:.0%} of each forecast. They are areas to "
            "watch, not promises: a forecast cannot see a first visit to a place nobody "
            "has reported elephants in.")
