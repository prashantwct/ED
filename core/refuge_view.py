"""The day-refuge view: Streamlit layout over ``core.refuges``.

Reads the Gaj Rakshak register loaded in the conflict view, the village
centroids, and Esri land cover -- uploaded GeoTIFFs, or the published
files read online -- and shows where elephants lie up by day, which
raids each refuge feeds, and what to do about it.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from core import esri_landcover as lc
from core import refuges as rf
from core.refuge_report import AREA_COLOR, CONFIDENCE_COLORS, RAID_COLOR, map_svg
from core.ui import findings, section

# The source and settings last used in this view, for the brief to reuse.
LAST_USED_KEY = "refuge_last_used"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "ed_landcover"

# ---------------------------------------------------------------------------
# What the map can be coloured by
# ---------------------------------------------------------------------------
# Confidence is how much the site is believed; group is who uses it; last
# used is whether it is live. They answer different questions and the
# reader picks, rather than the map deciding for them.
BY_CONFIDENCE, BY_GROUP, BY_RECENCY = "Confidence", "Group", "Last used"
COLOUR_MODES = (BY_CONFIDENCE, BY_GROUP, BY_RECENCY)

# Okabe-Ito, as the rest of the app uses for identity rather than order.
GROUP_COLORS = {
    "Lone bull": "#0072B2",
    "Bull party": "#009E73",
    "Family herd": "#CC79A7",
    "Mixed / unsexed": "#E69F00",
    "Unrecorded": "#7B857C",
}
# One hue, dark = used most recently. An ordinal question gets an ordinal
# ramp, not five unrelated colours.
RECENCY_BANDS = (
    (7, "Last 7 days", "#00474D"),
    (30, "8-30 days", "#2B8187"),
    (60, "31-60 days", "#78B2AD"),
    (float("inf"), "Over 60 days", "#B2C4BA"),
)
WITHIN_CHOICES = {"Any": None, "7 days": 7, "30": 30, "60": 60}
PERIOD_CHOICES = {"All": None, "90 days": 90, "30 days": 30}
WATCH_DAYS = 7


def recency_band(days_since: object) -> Tuple[str, str]:
    """Which recency band a refuge falls in, as (label, colour)."""
    if days_since is None or pd.isna(days_since):
        return RECENCY_BANDS[-1][1], RECENCY_BANDS[-1][2]
    for limit, label, color in RECENCY_BANDS:
        if float(days_since) <= limit:
            return label, color
    return RECENCY_BANDS[-1][1], RECENCY_BANDS[-1][2]


def refuge_color(row: Dict[str, object], mode: str) -> str:
    """The colour one refuge takes under the chosen mode."""
    if mode == BY_GROUP:
        return GROUP_COLORS.get(str(row.get("Group")), GROUP_COLORS["Unrecorded"])
    if mode == BY_RECENCY:
        return recency_band(row.get("Days Since Last Use"))[1]
    return CONFIDENCE_COLORS.get(str(row.get("Confidence")), RAID_COLOR)


def legend_entries(table: pd.DataFrame, mode: str) -> List[Tuple[str, str]]:
    """Legend pairs for the mode, carrying only what is on the map."""
    if mode == BY_GROUP:
        present = set(table["Group"].astype(str)) if len(table) else set()
        return [(n, c) for n, c in GROUP_COLORS.items() if n in present]
    if mode == BY_RECENCY:
        present = {recency_band(d)[0] for d in table["Days Since Last Use"]} if len(table) else set()
        return [(label, color) for _, label, color in RECENCY_BANDS if label in present]
    present = set(table["Confidence"].astype(str)) if len(table) else set()
    return [(n, c) for n, c in CONFIDENCE_COLORS.items() if n in present]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _saved_uploads(files) -> Tuple[str, ...]:
    """Uploaded GeoTIFFs written to disk once each, by content hash.

    rasterio reads from a path; the hash makes the path stable across
    reruns, so the cached analysis is not redone for the same file.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        data = f.getvalue()
        path = UPLOAD_DIR / (hashlib.blake2b(data, digest_size=10).hexdigest() + ".tif")
        if not path.exists():
            path.write_bytes(data)
        paths.append(str(path))
    return tuple(paths)


def land_cover_source() -> Tuple[str, object]:
    """Sidebar choice of land cover: ('online', year), ('files', paths) or ('none', None)."""
    with st.sidebar.expander("Land cover (Esri 10 m)", expanded=True):
        choice = st.radio(
            "Source", ["Read Esri land cover online", "Upload Esri GeoTIFF", "None"],
            key="refuge_lc_source",
            help="Esri Sentinel-2 land use / land cover, 10 m, yearly.",
        )
        if choice.startswith("Read"):
            years = list(lc.YEARS)[::-1]
            year = st.selectbox("Year", years, index=years.index(lc.DEFAULT_YEAR),
                                key="refuge_lc_year",
                                help="A year Esri has not published falls back to the "
                                     "latest earlier one.")
            st.caption("Fetches only the few hundred kilobytes around each refuge from "
                       "Esri's published files. Needs internet where the app runs.")
            return "online", int(year)
        if choice.startswith("Upload"):
            files = st.file_uploader(
                "GeoTIFF(s) from the Esri Land Cover Explorer", type=["tif", "tiff"],
                accept_multiple_files=True, key="refuge_lc_files")
            st.caption("Download the tile(s) covering your area from Esri's Sentinel-2 "
                       "Land Cover Explorer. UTM or lat/lon projections both work.")
            return ("files", _saved_uploads(files)) if files else ("none", None)
        return "none", None


def settings() -> Dict[str, object]:
    with st.sidebar.expander("Refuge settings", expanded=False):
        forest = st.radio(
            "Forest from", list(FOREST_CHOICES), key="refuge_forest",
            help="Only sightings and signs inside forest are refuge evidence.")
        eps = st.slider("Reports within (km) form one block", 0.25, 2.0, rf.EPS_KM, 0.25,
                        key="refuge_eps")
        days = st.slider("Minimum separate days of use", 2, 10, rf.MIN_DAYS, key="refuge_days")
        link = st.slider("Night raid reach from a refuge (km)", 1.0, 10.0, rf.LINK_KM, 0.5,
                         key="refuge_link")
        link_days = st.slider(
            "Raid within (days) of forest evidence", 1, 15, rf.LINK_DAYS,
            key="refuge_link_days",
            help="Wider windows link more raids, but mostly by chance: see the baseline.")
    return {"forest": FOREST_CHOICES[forest], "eps_km": eps, "min_days": days,
            "link_km": link, "link_days": link_days}


FOREST_CHOICES = {
    "Forest boundaries (notified forest)": "auto",
    "Esri land cover": "landcover",
    "No forest test": "none",
}


@st.cache_data(show_spinner="Finding day refuges...", max_entries=8)
def analyse(df: pd.DataFrame, villages: Optional[pd.DataFrame], source: Tuple[str, object],
            params: Tuple) -> Dict[str, object]:
    kind, value = source
    cover = None
    if kind == "online":
        cover = lc.LandCover(remote_year=value)
    elif kind == "files":
        cover = lc.LandCover(list(value))
    forest, eps_km, min_days, link_km, link_days = params
    try:
        result = rf.find_refuges(df, villages, cover, forest=forest, eps_km=eps_km,
                                 min_days=min_days, link_km=link_km, link_days=link_days)
    finally:
        if cover is not None:
            cover.close()
    result["land_cover"] = cover.label if cover is not None else "None"
    result["land_cover_failures"] = list(cover.failures) if cover is not None else []
    result["land_cover_notes"] = [
        f"Esri has not published {value} land cover for tile {tile}; {year} was used there."
        for tile, year in sorted(getattr(cover, "fallbacks", {}).items())]
    return result


def for_brief(df: pd.DataFrame, villages: Optional[pd.DataFrame]) -> Dict[str, object]:
    """Refuges for the brief, with the land cover and settings last used in
    the Day refuges view, or the defaults and Esri's latest year online."""
    source, params = st.session_state.get(LAST_USED_KEY, (
        ("online", lc.DEFAULT_YEAR), ("auto", rf.EPS_KM, rf.MIN_DAYS, rf.LINK_KM,
                                      rf.LINK_DAYS)))
    return analyse(df, villages, source, params)


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------
def _render_map(table: pd.DataFrame, links: pd.DataFrame, basemap: str,
                areas: Optional[pd.DataFrame] = None,
                colour_mode: str = BY_CONFIDENCE) -> None:
    from core import map_engine as me

    points = table.copy()
    areas = areas if areas is not None else pd.DataFrame()
    view = me._adaptive_view_state(pd.concat(
        [f[["Latitude", "Longitude"]] for f in (points, links, areas) if len(f)]))
    layers = list(me.boundary_layers(view, None))
    if len(areas):
        area_tips = [me._slots(
            title=f"{r['Area']} - raid area, no known refuge",
            accent=me._hex_to_rgb(AREA_COLOR),
            subtitle=me._trail(r.get("Beat"), r.get("Range")),
            rows=[("Raids", f"{r['Raids']} on {r['Nights']} nights"),
                  ("Last raid", f"{r['Last Raid']:%d %b %Y}"),
                  ("Villages", r["Villages Raided"] or None),
                  ("Search", f"{r['Search Patch (ha)']:.0f} ha tree patch, "
                             f"{r['Search Distance (km)']} km {r['Search Direction']}"
                   if r.get("Search Patch (ha)") == r.get("Search Patch (ha)")
                   and "Search Patch (ha)" in r else None)],
            footer=r["Recommended Action"],
        ) for r in areas.to_dict("records")]
        area_data = pd.concat([areas[["Latitude", "Longitude"]].reset_index(drop=True),
                               pd.DataFrame(area_tips)], axis=1)
        area_data["_r"] = 300 + 40 * areas["Raids"].clip(upper=40).to_numpy()
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=area_data, get_position="[Longitude, Latitude]",
            get_radius="_r", radius_min_pixels=6,
            get_fill_color=list(me._hex_to_rgb(AREA_COLOR)) + [60],
            get_line_color=list(me._hex_to_rgb(AREA_COLOR)) + [230],
            stroked=True, line_width_min_pixels=2, pickable=True,
        ))
    if points.empty:
        me._render_deck(layers, view, basemap)
        return
    if len(links):
        ref = table.set_index("Refuge")
        lines = links.assign(
            from_lon=links["Refuge"].map(ref["Longitude"]),
            from_lat=links["Refuge"].map(ref["Latitude"]))
        layers.append(pdk.Layer(
            "LineLayer", data=lines[["from_lon", "from_lat", "Longitude", "Latitude"]],
            get_source_position="[from_lon, from_lat]",
            get_target_position="[Longitude, Latitude]",
            get_color=[91, 107, 98, 110], get_width=1.5, width_units="'pixels'",
        ))
        raid_tips = [me._slots(
            title=f"Night raid linked to {r['Refuge']}",
            accent=me._hex_to_rgb(RAID_COLOR),
            subtitle=f"Night of {r['Night Of']:%d %b %Y} ({r['Link'].lower()})",
            rows=[("Damage", r["Category"]), ("From refuge", f"{r['Distance (km)']:.1f} km"),
                  ("Village", r["Village"] or None)],
        ) for r in links.to_dict("records")]
        raid_data = pd.concat([links[["Latitude", "Longitude"]].reset_index(drop=True),
                               pd.DataFrame(raid_tips)], axis=1)
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=raid_data, get_position="[Longitude, Latitude]",
            get_radius=120, radius_min_pixels=2.5, get_fill_color=[91, 107, 98, 200],
            pickable=True,
        ))
    tips = []
    for r in table.to_dict("records"):
        tips.append(me._slots(
            title=f"{r['Refuge']} - {r['Confidence']} day refuge",
            accent=me._hex_to_rgb(refuge_color(r, colour_mode)),
            subtitle=me._trail(r.get("Beat"), r.get("Range")),
            rows=[("Days used", f"{r['Days Used']} (last {r['Last Used']:%d %b})"),
                  ("Cover", f"{r['Cover Share %']:.0f}% within 250 m"
                   if r.get("Cover Share %") == r.get("Cover Share %") and "Cover Share %" in r
                   else None),
                  ("Linked raids", f"{r['Linked Raids']}, heading {r['Raids Head']}"
                   if r["Linked Raids"] else "none"),
                  ("Group", r["Group"]),
                  ("Nearest village", me._trail(r.get("Nearest Village"),
                                                f"{r['To Nearest Village (km)']:.1f} km"
                                                if r.get("To Nearest Village (km)") ==
                                                r.get("To Nearest Village (km)") else ""))],
            footer=r["Flags"],
        ))
    data = pd.concat([points[["Latitude", "Longitude"]].reset_index(drop=True),
                      pd.DataFrame(tips)], axis=1)
    data["_r"] = np.maximum(points["Radius (km)"].to_numpy() * 1000, 300)
    data["_c"] = [list(me._hex_to_rgb(refuge_color(r, colour_mode))) + [150]
                  for r in points.to_dict("records")]
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=data, get_position="[Longitude, Latitude]",
        get_radius="_r", radius_min_pixels=7, get_fill_color="_c",
        get_line_color=[255, 255, 255, 255], line_width_min_pixels=2, stroked=True,
        pickable=True,
    ))
    me._render_deck(layers, view, basemap)


# ---------------------------------------------------------------------------
# The register panel
# ---------------------------------------------------------------------------
def _key_panel(table: pd.DataFrame, areas: pd.DataFrame) -> Dict[str, object]:
    """The map key, as the controls that drive the map.

    A legend that only explains is a waste of the space beside a map.
    Each block here says what a mark means and lets the reader take it
    off, which is the difference between reading the plate and working
    it.
    """
    st.markdown("##### Map key")
    mode = st.segmented_control(
        "Colour refuges by", COLOUR_MODES, default=BY_CONFIDENCE,
        key="refuge_colour_mode",
    ) or BY_CONFIDENCE

    confidences = [c for c in (rf.CONFIRMED, rf.PROBABLE, rf.POSSIBLE)
                   if len(table) and (table["Confidence"] == c).any()]
    keep_conf = st.segmented_control(
        "Confidence", confidences, selection_mode="multi", default=confidences,
        key="refuge_conf_pick",
        help="How much the site is believed: separate days of forest evidence.",
    ) or confidences

    groups = [g for g in GROUP_COLORS if len(table) and (table["Group"] == g).any()]
    keep_groups = st.segmented_control(
        "Group", groups, selection_mode="multi", default=groups,
        key="refuge_group_pick",
    ) or groups

    within = st.segmented_control(
        "Last used within", list(WITHIN_CHOICES), default="Any", key="refuge_within",
    ) or "Any"

    st.markdown("##### Raids")
    period = st.segmented_control(
        "Raid period", list(PERIOD_CHOICES), default="All", key="refuge_period",
    ) or "All"
    show_links = st.checkbox("Lines from refuge to raid", value=True, key="refuge_tg_links")
    show_areas = st.checkbox(
        "Raid areas with no known refuge", value=True, key="refuge_tg_areas",
        disabled=areas is None or areas.empty,
    )
    return {"mode": mode, "confidence": list(keep_conf), "groups": list(keep_groups),
            "within": WITHIN_CHOICES[within], "period": PERIOD_CHOICES[period],
            "links": show_links, "areas": show_areas}


def _apply(table: pd.DataFrame, links: pd.DataFrame, areas: pd.DataFrame,
           pick: Dict[str, object]):
    """Narrow the three frames to what the key is currently showing."""
    shown = table
    if len(shown):
        shown = shown[shown["Confidence"].isin(pick["confidence"])
                      & shown["Group"].isin(pick["groups"])]
        if pick["within"] is not None:
            days = pd.to_numeric(shown["Days Since Last Use"], errors="coerce")
            shown = shown[days <= pick["within"]]

    shown_links = links
    if len(shown_links):
        shown_links = shown_links[shown_links["Refuge"].isin(shown["Refuge"])]
        if pick["period"] is not None and "Night Of" in shown_links:
            nights = pd.to_datetime(shown_links["Night Of"], errors="coerce")
            if nights.notna().any():
                cutoff = nights.max() - pd.Timedelta(days=int(pick["period"]))
                shown_links = shown_links[nights >= cutoff]
        if not pick["links"]:
            shown_links = shown_links.iloc[0:0]

    shown_areas = areas if pick["areas"] and areas is not None else pd.DataFrame()
    return shown, shown_links, shown_areas


def _legend(table: pd.DataFrame, pick: Dict[str, object], areas: pd.DataFrame) -> None:
    items = [f'<span class="ci-legend__item"><span class="ci-legend__dot" '
             f'style="background:{c};"></span>{n}</span>'
             for n, c in legend_entries(table, str(pick["mode"]))]
    if len(table) and pick["links"]:
        items.append('<span class="ci-legend__item"><span class="ci-legend__dot" '
                     f'style="background:{RAID_COLOR};"></span>Linked night raid</span>')
    if len(areas):
        items.append('<span class="ci-legend__item"><span class="ci-legend__dot" '
                     f'style="background:{AREA_COLOR};border-radius:2px;"></span>'
                     'Raid area, no known refuge</span>')
    if items:
        st.markdown(f'<div class="ci-legend">{"".join(items)}</div>',
                    unsafe_allow_html=True)


def _watch_block(table: pd.DataFrame, days: int = WATCH_DAYS) -> None:
    """Refuges used in the last week: what to act on tonight.

    The operational output of the whole view. Newest first, each line
    carrying the edge to watch at dusk and the villages to warn before
    dark, and the lot copyable as a message because that is how it
    reaches a range officer.
    """
    st.markdown(f"##### Occupied in the last {days} days")
    if table.empty:
        st.caption("No refuges in the current selection.")
        return
    since = pd.to_numeric(table["Days Since Last Use"], errors="coerce")
    watch = table[since <= days].sort_values("Days Since Last Use")
    if watch.empty:
        st.caption(
            f"None used in the last {days} days. The register below covers the "
            "whole period."
        )
        return

    lines = []
    for r in watch.to_dict("records"):
        ago = int(r["Days Since Last Use"])
        when = "today" if ago == 0 else f"{ago} day{'s' if ago > 1 else ''} ago"
        head = f", raids head {r['Raids Head']}" if r.get("Raids Head") else ""
        warn = str(r.get("Villages Within Reach") or "").strip()
        st.markdown(
            f"**{r['Refuge']}** &middot; {r['Confidence']} &middot; {r.get('Beat', '')}  \n"
            f"Used {when}{head}. "
            + (f"Warn {warn}." if warn else "No village within reach.")
        )
        lines.append(
            f"{r['Refuge']} ({r.get('Beat', '')}): used {when}{head}. "
            + (f"Warn before dark: {warn}." if warn else "No village within reach.")
        )

    st.text_area(
        "Copy as message", "\n".join(lines), height=120, key="refuge_watch_text",
        help="Paste into WhatsApp or SMS for the range office.",
    )


def _panel_charts(links: pd.DataFrame) -> None:
    from core.charts import raid_start_hours, raids_by_month
    from core.config import NIGHT_HOUR_END, NIGHT_HOUR_START

    st.markdown("##### When raids start")
    if links.empty:
        st.caption("No linked raids in the current selection.")
        return
    st.plotly_chart(raid_start_hours(links, NIGHT_HOUR_START, NIGHT_HOUR_END),
                    width="stretch", key="refuge_hours")
    st.markdown("##### Raids by month")
    st.plotly_chart(raids_by_month(links), width="stretch", key="refuge_months")


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------
def render(register: Optional[pd.DataFrame], villages: Optional[pd.DataFrame],
           basemap: str) -> None:
    section("hotspot", "Day refuges",
            "Where elephants lie up by day, the raids each refuge feeds, and who to warn.")
    if register is None or register.empty:
        st.info("Load the Gaj Rakshak register in the Conflict dashboard view first; this "
                "view reads the same data.")
        return
    source = land_cover_source()
    params = settings()
    key = (params["forest"], params["eps_km"], params["min_days"], params["link_km"],
           params["link_days"])
    result = analyse(register, villages, source, key)
    st.session_state[LAST_USED_KEY] = (source, key)
    table, links, s = result["refuges"], result["links"], result["summary"]
    areas = result.get("areas", pd.DataFrame())

    st.caption(f"Land cover: {result['land_cover']}.")
    for note in result["land_cover_notes"]:
        st.info(note)
    for failure in result["land_cover_failures"][:3]:
        st.warning(f"Land cover could not be read: {failure}. Upload the tile from Esri's "
                   "Land Cover Explorer instead.")

    note = dict(delta_color="off", delta_arrow="off", border=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Night raids", f"{s['night_raids']:,}",
              delta=f"{s['scattered_raids']:,} scattered", **note)
    c2.metric("Linked to a forest refuge", f"{s['linked_raids']:,}",
              delta=f"{s['same_night']:,} same night; ~{s['chance_links']:.0f} by chance",
              **note)
    c3.metric("In raid areas, no known refuge", f"{s['area_raids']:,}",
              delta=f"{s['areas']} area(s) to search", **note)
    c4.metric("Forest refuges", f"{s['refuges']}",
              delta=f"{s['confirmed']} confirmed, {s['probable']} probable", **note)

    findings(rf.headlines(result))
    if table.empty and areas.empty:
        for c in result["caveats"]:
            st.caption(c)
        return

    # The plate: map sheet on the left, register panel beside it. The
    # panel's key drives the map, so its widgets are created before the
    # map is drawn; the watch list reads first on the page, so its slot
    # is reserved now and filled once the selection is known.
    sheet, panel = st.columns([1.7, 1], gap="medium")
    with panel:
        watch_slot = st.container()
        pick = _key_panel(table, areas)
    shown, shown_links, shown_areas = _apply(table, links, areas, pick)

    with sheet:
        st.markdown("**Refuges, the raids linked to them, and raid areas with no "
                    "known refuge**")
        _render_map(shown, shown_links, basemap, shown_areas, str(pick["mode"]))
        _legend(shown, pick, shown_areas)
        if len(shown) < len(table) or len(shown_areas) < len(areas):
            st.caption(f"Showing {len(shown)} of {len(table)} refuges and "
                       f"{len(shown_links):,} of {len(links):,} linked raids. "
                       "The figures above cover the whole period.")
        from core.exports import download_menu

        download_menu("Day refuge map", lambda t=shown, l=shown_links, a=shown_areas,
                      b=basemap: map_svg(t, l, areas=a, style_name=b))
    with watch_slot:
        _watch_block(shown)
    with panel:
        _panel_charts(shown_links)

    _areas_table(shown_areas if len(shown_areas) else areas)
    if table.empty:
        _method(result)
        return

    st.markdown("**Refuge register** -- strongest evidence first")
    register = shown if len(shown) else table
    cols = [c for c in [
        "Refuge", "Confidence", "Beat", "Range", "Days Used", "Direct Sightings",
        "Indirect Signs", "Last Used", "Months Used",
        "Group", "Cover Share %", "Tree Patch (ha)", "Linked Raids", "Same-Night Raids",
        "Raids Head",
        "Raid Distance (km, median)", "Villages Raided", "Nearest Village",
        "To Nearest Village (km)", "Flags", "Recommended Action"] if c in table]
    st.dataframe(register[cols], width="stretch", hide_index=True, column_config={
        "Last Used": st.column_config.DateColumn(format="DD MMM YYYY"),
        "Cover Share %": st.column_config.ProgressColumn("Cover", min_value=0, max_value=100,
                                                         format="%.0f%%"),
        "Recommended Action": st.column_config.TextColumn(width="large"),
    })
    d1, d2, d3, d4 = st.columns(4)
    # The downloads follow the key, so a file matches the plate it came
    # from rather than quietly carrying rows the reader had switched off.
    out_links = shown_links if len(shown) else links
    out_areas = shown_areas if len(shown_areas) else areas
    d1.download_button("Refuges (CSV)", register.to_csv(index=False).encode(),
                       "day_refuges.csv", mime="text/csv", width="stretch")
    d2.download_button("Refuges + raids (GeoJSON)",
                       rf.to_geojson(register, out_links, out_areas).encode(),
                       "day_refuges.geojson", mime="application/geo+json", width="stretch")
    d3.download_button("Refuges (KML, Google Earth)",
                       rf.to_kml(register, out_areas).encode(),
                       "day_refuges.kml", mime="application/vnd.google-earth.kml+xml",
                       width="stretch")
    d4.download_button("Linked raids (CSV)", out_links.to_csv(index=False).encode(),
                       "refuge_linked_raids.csv", mime="text/csv", width="stretch")
    st.caption("GeoJSON opens in QGIS; KML in Google Earth or a phone map app for field teams.")

    _cards(register, out_links, result["windows"])
    _method(result)


def _method(result: Dict[str, object]) -> None:
    with st.expander("Method, and what this cannot say", expanded=False):
        st.markdown(rf.__doc__.split("\n", 1)[1])
        for c in result["caveats"]:
            st.markdown(f"- {c}")


def _areas_table(areas: pd.DataFrame) -> None:
    if areas is None or areas.empty:
        return
    st.markdown("**Raid areas with no known refuge** -- raids no reported day refuge accounts "
                "for; an unreported refuge lies within a night's walk of each")
    shown = [c for c in [
        "Area", "Beat", "Range", "Raids", "Nights", "Last Raid", "Months", "House Damage",
        "Casualties", "Villages Raided", "To Nearest Known Refuge (km)", "Search Patch (ha)",
        "Search Distance (km)", "Search Direction", "Recommended Action"] if c in areas]
    st.dataframe(areas[shown], width="stretch", hide_index=True, column_config={
        "Last Raid": st.column_config.DateColumn(format="DD MMM YYYY"),
        "Recommended Action": st.column_config.TextColumn(width="large"),
    })
    st.download_button("Raid areas to search (CSV)", areas.to_csv(index=False).encode(),
                       "raid_areas_no_refuge.csv", mime="text/csv",
                       key="refuge_areas_csv")


def _cards(table: pd.DataFrame, links: pd.DataFrame, windows: Dict[str, object]) -> None:
    if not windows:
        return
    st.markdown("**Refuges on the land cover** -- 6 km around each, Esri classes; "
                "white ring is the refuge, dots are linked raids")
    legend = " ".join(
        f'<span class="ci-legend__item"><span class="ci-legend__dot" style="background:{c};'
        f'border-radius:2px;"></span>{lc.CLASSES[k]}</span>'
        for k, c in lc.COLORS.items() if k not in (lc.SNOW, lc.CLOUDS))
    st.markdown(f'<div class="ci-legend">{legend}</div>', unsafe_allow_html=True)
    shown = [r for r in table.to_dict("records") if r["Refuge"] in windows][:12]
    for start in range(0, len(shown), 3):
        cols = st.columns(3)
        for col, r in zip(cols, shown[start:start + 3]):
            window = windows[r["Refuge"]]
            raids = links[links["Refuge"] == r["Refuge"]]
            marks = [window.pixel_of(a, b) for a, b in zip(raids["Latitude"], raids["Longitude"])]
            with col:
                st.image(lc.thumbnail_png(window, 360, marks), width="stretch")
                st.markdown(
                    f"**{r['Refuge']}** -- {r['Confidence']} &middot; {r.get('Beat', '')}  \n"
                    f"Used {r['Days Used']} days, last {r['Last Used']:%d %b %Y} &middot; "
                    f"{r['Cover Share %']:.0f}% cover, tree patch {r['Tree Patch (ha)']:,.0f} ha"
                    + ("+" if r.get("Patch Extends Beyond Window") else "")
                    + f"  \n{r['Linked Raids']} linked raid(s)"
                    + (f", heading {r['Raids Head']}" if r["Linked Raids"] else ""))
