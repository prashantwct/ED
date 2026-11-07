"""Overview: what changed, where to act, whom to warn -- on one screen.

The home page. A sentence first, then four numbers, then the beats to act
on beside the map, then what changed in the last month. Every line comes
from the same tables the other pages and the brief use (core.overview),
and each item leads to the page that shows it in full.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from app_pages.brief import brief_button
from app_pages.common import FOCUS_PARAM, _change, brief_for, recent_groups
from core import map_select, overview
from core.brief_insights import action_plan, plural
from core.movement import ACTIVE_DAYS
from core.intelligence import (
    CRITICAL_CASUALTY_WINDOW_DAYS,
    TIER_CRITICAL,
    TIER_HIGH,
)
from core.map_engine import render_map
from core.ui import tier_badge


def _beats_before(c) -> pd.DataFrame:
    """The beat tiers ``CHANGE_DAYS`` before the period's end, for "now Critical"."""
    then = c.period_end - pd.Timedelta(days=overview.CHANGE_DAYS)
    earlier = c.history[c.history["Date"] <= then + pd.Timedelta(days=1)]
    if earlier.empty:
        return pd.DataFrame()
    return brief_for(earlier, c.recent_days, then)["beats"]


def _spark(c, column: str):
    return c.monthly[column].tolist() if len(c.monthly) >= 3 else None


def _headline_numbers(c) -> None:
    change, days = c.change, c.compare_days
    base = "year" if change.get("conflicts_year") == change.get("conflicts_year") \
        else "prior"
    note = "vs last yr" if base == "year" else f"vs prior {days}d"
    window_help = (f"Change over the last {days} days against "
                   + (f"the same {days} days a year earlier." if base == "year"
                      else f"the {days} days before them."))
    beats = c.beats_table
    critical = int((beats["Priority Tier"] == TIER_CRITICAL).sum()) if len(beats) else 0
    high = int((beats["Priority Tier"] == TIER_HIGH).sum()) if len(beats) else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Conflict events", f"{c.kpis['conflicts']:,}",
        delta=_change(change["conflicts_recent"], change[f"conflicts_{base}"]),
        delta_color="inverse", delta_description=note,
        help=f"Damage, injury or death in {c.window.label()}. " + window_help,
        chart_data=_spark(c, "Conflict Events"), chart_type="area", border=True,
    )
    k2.metric(
        "Killed / injured",
        f"{int(c.kpis['human_deaths'])} / {int(c.kpis['human_injuries'])}",
        delta=_change(change["casualties_recent"], change[f"casualties_{base}"]),
        delta_color="inverse", delta_description=note,
        help=f"People killed and injured in {c.window.label()}. " + window_help,
        chart_data=(c.monthly["Human Deaths"] + c.monthly["People Injured"]).tolist()
        if len(c.monthly) >= 3 else None,
        chart_type="bar", border=True,
    )
    k3.metric(
        "Beats Critical / High", f"{critical} / {high}", border=True,
        help=f"Critical: a person killed or injured in the last "
        f"{CRITICAL_CASUALTY_WINDOW_DAYS} days. High: an older casualty, or sustained "
        "damage.",
    )
    k4.metric(
        "Villages to warn", f"{overview.villages_to_warn(c.village_risk):,}", border=True,
        help=f"Villages in the Critical or High tier, from conflict within "
        f"{c.village_radius:g} km.",
    )


def _link(c, page: str, label: str, *, focus: str = "", icon: str = "", help: str = "",
          width="content") -> None:
    """A real link to another page, carrying the scope and, optionally, a map focus.

    A link and not a button with ``st.switch_page``: navigating from inside
    a run drops the uploaded register in the browser, while a link keeps
    it, as the page menu does. The focus travels as ``focus=beat:<name>``
    and the entry point applies it on arrival (common.FOCUS_PARAM).
    """
    query = dict(c.query)
    if focus:
        query[FOCUS_PARAM] = focus
    st.page_link(c.pages[page], label=label, icon=icon or None, help=help or None,
                 query_params=query, width=width)


def _act_now(c) -> None:
    head, mode_col = st.columns([1, 1], vertical_alignment="center")
    head.markdown("**Act now**")
    mode = mode_col.segmented_control(
        "Group", ["Beats", "Actions"], default="Beats", key="act_now_mode",
        label_visibility="collapsed",
        help="Beats: the worst beats first. Actions: the same signals grouped by what to do.",
    ) or "Beats"

    if mode == "Beats":
        rows = overview.act_now(c.beats_table)
        if not rows:
            st.success("No beat is Critical or High in this area.")
        for row in rows:
            with st.container(border=True):
                name_col, map_col = st.columns([4, 1], vertical_alignment="center")
                # Beat and range names come from the CSV: escaped before HTML.
                name_col.markdown(
                    f"{tier_badge(row['tier'])} &nbsp;**{escape(row['beat'])}** "
                    f"<span style='color:var(--ci-text-muted)'>"
                    f"{escape(row['range'])}</span>", unsafe_allow_html=True)
                with map_col:
                    _link(c, "where", "Map", focus=f"beat:{row['beat']}",
                          icon=":material/map:", help=f"Show {row['beat']} on the Where "
                          "to act map.")
                st.markdown(f"{row['do']}  \n:gray[{row['why']}]")
        urgent = c.beats_table["Priority Tier"].isin([TIER_CRITICAL, TIER_HIGH]).sum() \
            if len(c.beats_table) else 0
        if urgent > len(rows):
            _link(c, "where", f"All {urgent} Critical and High beats",
                  icon=":material/arrow_forward:")
    else:
        plan = action_plan(c.brief, c.recent_days, c.emerging, None, c.village_risk)
        if not plan:
            st.success("Nothing in this area calls for action beyond routine monitoring.")
        for action in plan[:5]:
            with st.container(border=True):
                st.markdown(f"**{action.action}**")
                st.markdown(f"Where: {action.where}")
                st.caption(f"{action.why}. When: {action.when}.")

    warn = overview.warn_list(c.village_risk)
    if warn:
        st.markdown(f"**Warn first:** {warn}")
        _link(c, "villages", "All villages at risk", icon=":material/arrow_forward:")

    register = c.df[c.df["Date"] < c.period_end + pd.Timedelta(days=1)]
    active = recent_groups(register, c.period_end)
    if active:
        st.markdown(f"**On the move:** {plural(active, 'group')} reported in the last "
                    f"{ACTIVE_DAYS} days.")
        _link(c, "next", "Where they may be next", icon=":material/near_me:")


def _map(c) -> None:
    """Conflict in the period, Critical and High villages, hotspot footprints."""
    focus = c.focus
    events = [k for k in map_select.EVENT_CLASSES if k != "Presence"]
    shown, shown_hotspots, shown_villages = map_select.select(
        focus, c.hotspots, c.village_risk, events=events,
        groups=list(map_select.GROUP_CLASSES), tiers=[TIER_CRITICAL, TIER_HIGH],
        focus=None, village_km=c.village_radius,
    )
    st.markdown(f"**Conflict in this period** -- {len(shown):,} reports, with the "
                f"Critical and High villages")
    render_map(
        shown, frame_df=focus if len(shown) == 0 else shown, hotspots=shown_hotspots,
        villages=shown_villages, style_name=c.basemap, show_sightings=not shown.empty,
        show_hotspots=True, show_villages=not shown_villages.empty,
        show_boundaries=c.layer_boundaries, boundary_level=c.boundary_level,
        beat_stats=c.beats_table,
    )


def _what_changed(c, before: pd.DataFrame) -> None:
    changes = overview.what_changed(c.beats_table, before, c.escalating, c.emerging)
    st.markdown(f"**What changed** -- the last {overview.CHANGE_DAYS} days")
    if not changes:
        st.caption("No beat became Critical, none is rising, and no new hotspot place "
                   "appeared.")
        return
    cols = st.columns(min(len(changes), 3))
    for i, change in enumerate(changes):
        with cols[i % len(cols)]:
            with st.container(border=True):
                if change.kind == "emerging":
                    _link(c, "patterns", change.label, icon=":material/timeline:",
                          help="See it on the Patterns page.", width="stretch")
                else:
                    _link(c, "where", change.label, focus=f"beat:{change.target}",
                          icon=":material/priority_high:" if change.kind == "critical"
                          else ":material/trending_up:",
                          help=f"Show {change.target} on the Where to act map.",
                          width="stretch")


def render(c) -> None:
    before = _beats_before(c)
    said = overview.headline(c.change, c.compare_days, c.beats_table,
                             overview.new_critical(c.beats_table, before), c.village_risk)
    text_col, brief_col = st.columns([4, 1], vertical_alignment="center")
    text_col.markdown(f'<p class="ci-headline">{escape(said)}</p>', unsafe_allow_html=True)
    with brief_col:
        brief_button(c, "overview_brief")

    _headline_numbers(c)

    act_col, map_col = st.columns([2, 3], gap="medium")
    with act_col:
        _act_now(c)
    with map_col:
        _map(c)

    _what_changed(c, before)
