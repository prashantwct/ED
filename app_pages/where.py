"""Where to act: beat tiers, layer filters, a table beside the spatial view."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from app_pages.common import _map_focus, MAP_FOCUS_KEY, _map_pick
from core import map_select
from core.exports import download_menu
from core.hotspots import hotspot_caveats
from core.intelligence import CRITICAL_CASUALTY_WINDOW_DAYS
from core.map_engine import render_map
from core.map_export import sightings_map_svg
from core.ui import CATEGORY_STYLE, TIER_STYLE, tier_summary


logger = logging.getLogger(__name__)


def render(c) -> None:
    basemap = c.basemap
    beats_table = c.beats_table
    boundary_level = c.boundary_level
    composition = c.composition
    # Points on the map, and the counts on the layer filters, are the focus
    # window; tiers, hotspots and villages come from the analysis history.
    filtered = c.focus
    history = c.history
    hotspots = c.hotspots
    layer_boundaries = c.layer_boundaries
    village_radius = c.village_radius
    village_risk = c.village_risk
    villages = c.villages

    if not beats_table.empty:
        tier_summary(beats_table["Priority Tier"].value_counts().to_dict())

    # Layer filters: what happened, by which animals, and which villages.
    # They cut the data before anything is drawn, so "crop damage by lone
    # bulls" is a map of exactly those sightings and the places near them.
    event_counts, group_counts = map_select.class_counts(filtered)
    ev_col, grp_col, vil_col = st.columns([5, 5, 4])
    with ev_col:
        events = st.pills(
            "Events", [k for k in map_select.EVENT_CLASSES if event_counts.get(k)],
            selection_mode="multi", key="layer_events",
            default=[k for k in map_select.EVENT_CLASSES if event_counts.get(k)],
            format_func=lambda k, c=event_counts:
            f"{CATEGORY_STYLE[k]['label']} ({c.get(k, 0):,})",
            help="Untick a class to drop it from the map. None ticked hides sightings.",
        ) or []
    with grp_col:
        groups = st.pills(
            "Animals", [g for g in map_select.GROUP_CLASSES if group_counts.get(g)],
            selection_mode="multi", key="layer_groups",
            default=[g for g in map_select.GROUP_CLASSES if group_counts.get(g)],
            format_func=lambda g, c=group_counts: f"{g} ({c.get(g, 0):,})",
            help="Combine with Events: e.g. Human injury + Lone bull alone.",
        ) or []
    with vil_col:
        tier_options = [t for t in TIER_STYLE
                        if not village_risk.empty and (village_risk["Tier"] == t).any()]
        tiers = st.pills(
            "Villages by tier", tier_options, selection_mode="multi", key="layer_tiers",
            default=[t for t in tier_options if t != "Routine"],
            help="Village layer. Routine villages start off to keep the map readable.",
        ) or []
        show_hotspots = st.toggle("Hotspot footprints", value=True, key="layer_hotspots")

    table_col, map_col = st.columns([1, 1], gap="medium")

    # The table draws first, so a ticked row reaches the map in the same
    # run: one click, not two.
    with table_col:
        table_options = ["Beats", "Hotspots", "Villages", "Animals"]
        picker = st.segmented_control(
            "Tick a row to show only that on the map", table_options,
            default="Beats", key="where_table",
        ) or "Beats"

        if picker == "Beats":
            if beats_table.empty:
                st.info("Not enough data to rank beats.")
            else:
                display_cols = [
                    "Beat", "Range", "Priority Tier", "Priority Score",
                    "Conflict Events", "Recent Deaths", "Recent Injuries",
                    "Bull-Type Conflict %", "Trend", "Recommended Action",
                ]
                beat_pick = st.dataframe(
                    beats_table[display_cols],
                    width="stretch",
                    height=520,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="beat_table",
                    column_config={
                        "Priority Tier": st.column_config.TextColumn("Tier", width="small"),
                        "Priority Score": st.column_config.ProgressColumn(
                            "Score", min_value=0, max_value=100, format="%.0f",
                            help="Orders beats within a tier only. Moves when filters "
                            "change.",
                        ),
                        "Conflict Events": st.column_config.NumberColumn("Conflicts"),
                        "Recent Deaths": st.column_config.NumberColumn(
                            "Killed", format="%.0f",
                            help=f"In the last {CRITICAL_CASUALTY_WINDOW_DAYS} days. Drives "
                            "the Critical tier.",
                        ),
                        "Recent Injuries": st.column_config.NumberColumn(
                            "Injured", format="%.0f",
                            help=f"In the last {CRITICAL_CASUALTY_WINDOW_DAYS} days.",
                        ),
                        "Bull-Type Conflict %": st.column_config.NumberColumn(
                            "Bull-driven", format="%.0f%%",
                            help="Share of this beat's conflict from lone bulls or small "
                            "all-male parties.",
                        ),
                        "Recommended Action": st.column_config.TextColumn(width="large"),
                    },
                )
                _map_pick(beat_pick, "beat_table", "beat", beats_table["Beat"], "Beat")
                st.caption(f"Critical = a person killed or injured in the last "
                           f"{CRITICAL_CASUALTY_WINDOW_DAYS} days. Tiers compare across "
                           "periods; the score only orders beats within a tier.")
                st.download_button(
                    "Download all beat figures (CSV)",
                    beats_table.to_csv(index=False).encode("utf-8"),
                    "beat_priorities.csv",
                    mime="text/csv",
                    help="Every column, including period casualties, confidence, rates, "
                    "night and village shares.",
                )

        elif picker == "Hotspots":
            if hotspots.empty:
                st.info("No hotspots at these settings. On the Analyst page, try a larger "
                        "merge distance or a smaller minimum hotspot.")
            else:
                hotspot_cols = [c for c in [
                    "Hotspot", "Tier", "Beats", "Sightings", "Conflict Events",
                    "Conflict Share %", "Human Deaths", "People Injured", "Night Share %",
                    "Radius (km)",
                ] if c in hotspots.columns]
                hotspot_pick = st.dataframe(
                    hotspots[hotspot_cols], width="stretch", height=520, hide_index=True,
                    on_select="rerun", selection_mode="single-row", key="hotspot_table",
                    column_config={
                        "Tier": st.column_config.TextColumn(width="small"),
                        "Conflict Share %": st.column_config.NumberColumn(
                            "Conflict share", format="%.0f%%"),
                        "Night Share %": st.column_config.NumberColumn("Night",
                                                                       format="%.0f%%"),
                        "Human Deaths": st.column_config.NumberColumn("Killed", format="%.0f"),
                        "People Injured": st.column_config.NumberColumn("Injured",
                                                                        format="%.0f"),
                        "Radius (km)": st.column_config.NumberColumn("Radius",
                                                                     format="%.1f km"),
                    },
                )
                _map_pick(hotspot_pick, "hotspot_table", "hotspot", hotspots["Hotspot"],
                          "Hotspot")
                st.download_button(
                    "Download hotspots (CSV)",
                    hotspots.to_csv(index=False).encode("utf-8"),
                    "hotspots.csv",
                    mime="text/csv",
                )
            for note in hotspot_caveats(hotspots, history, villages):
                st.caption(note)

        elif picker == "Villages":
            if village_risk.empty:
                st.info("No village ranking: load a village centroids file in the sidebar.")
            else:
                village_cols = [c for c in [
                    "Village", "Tier", "Conflict Events", "Human Deaths", "People Injured",
                    "House Damage Events", "Crop Damage Events", "Nearest Hotspot",
                ] if c in village_risk.columns]
                village_pick = st.dataframe(
                    village_risk[village_cols], width="stretch", height=520,
                    hide_index=True, on_select="rerun", selection_mode="single-row",
                    key="village_table",
                    column_config={
                        "Tier": st.column_config.TextColumn(width="small"),
                        "Conflict Events": st.column_config.NumberColumn("Conflicts"),
                        "Human Deaths": st.column_config.NumberColumn("Killed"),
                        "People Injured": st.column_config.NumberColumn("Injured"),
                        "House Damage Events": st.column_config.NumberColumn("House"),
                        "Crop Damage Events": st.column_config.NumberColumn("Crop"),
                        "Nearest Hotspot": st.column_config.TextColumn("Hotspot"),
                    },
                )
                _map_pick(village_pick, "village_table", "village", village_risk["Village"],
                          "Village")
                st.caption(f"A picked village shows the sightings within "
                           f"{village_radius:g} km of it (the village radius on the Analyst "
                           f"page).")

        else:
            if composition.empty or composition["Conflict Events"].sum() == 0:
                st.info("No group composition recorded for the current filters.")
            else:
                st.dataframe(
                    composition,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Damage Rate %": st.column_config.NumberColumn(
                            "Damage rate", format="%.1f%%",
                            help="Share of that group's sightings involving damage, "
                            "injury or death.",
                        ),
                        "Human Deaths": st.column_config.NumberColumn("Killed", format="%.0f"),
                    },
                )
                st.caption(
                    "Map a group with the Animals filter above. Only bulls carry tusks, so "
                    "Male Count is the tusker count. A lone bull or small all-male party "
                    "raids and is tracked and deterred; a breeding herd with calves is "
                    "given passage and not driven. Unrecorded groups are kept apart "
                    "rather than guessed."
                )

    map_focus = _map_focus(filtered, hotspots, village_risk, village_radius)
    frame_df = filtered[map_focus["mask"]] if map_focus else filtered
    if map_focus is None and st.session_state.get(MAP_FOCUS_KEY):
        # A beat picked on the Overview may have no reports in a short focus
        # window: frame it from its history and say why the map is empty.
        earlier = _map_focus(history, hotspots, village_risk, village_radius)
        if earlier is not None:
            frame_df = history[earlier["mask"]]
            map_focus = {**earlier, "mask": pd.Series(False, index=filtered.index),
                         "label": earlier["label"] + " (no reports in this period)"}
    shown, shown_hotspots, shown_villages = map_select.select(
        filtered, hotspots, village_risk, events=events, groups=groups, tiers=tiers,
        focus=map_focus, village_km=village_radius,
    )

    with map_col:
        head, menu = st.columns([10, 1], vertical_alignment="bottom")
        head.markdown(f"**Spatial view** -- {len(shown):,} of {len(filtered):,} sightings")
        with menu:
            download_menu(
                "Spatial view",
                lambda f=shown, h=shown_hotspots if show_hotspots else None, b=basemap:
                sightings_map_svg(f, h, b),
                compact=True,
            )
        if map_focus:
            focus_col, clear_col = st.columns([3, 1], vertical_alignment="center")
            focus_col.info(f"Showing {map_focus['label']} only. Untick the row to show "
                           "everything again.")
            if clear_col.button("Show everything", key="clear_map_focus", width="stretch"):
                st.session_state.pop(MAP_FOCUS_KEY, None)
                st.rerun()
        if shown.empty:
            st.caption("No sightings match the Events and Animals filters"
                    + (f" in {map_focus['label']}." if map_focus else "."))
        render_map(
            shown,
            frame_df=frame_df,
            focus=map_focus,
            hotspots=shown_hotspots,
            villages=shown_villages,
            style_name=basemap,
            show_sightings=not shown.empty,
            show_hotspots=show_hotspots,
            show_villages=not shown_villages.empty,
            show_boundaries=layer_boundaries,
            boundary_level=boundary_level,
            beat_stats=beats_table,
        )
