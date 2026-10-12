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
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from core import esri_landcover as lc
from core import refuges as rf
from core.ui import findings, section

CONFIDENCE_COLORS = {rf.CONFIRMED: "#8C1D18", rf.PROBABLE: "#C2570C", rf.POSSIBLE: "#C9A227"}
RAID_COLOR = "#5B6B62"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "ed_landcover"


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
            year = st.selectbox("Year", list(lc.YEARS)[::-1], key="refuge_lc_year")
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
        start, end = st.slider("Daytime rest window (hours)", 6, 18,
                               (rf.DAY_START, rf.DAY_END), key="refuge_window",
                               help="Dawn and dusk are left out: animals are moving then.")
        eps = st.slider("Sightings within (km) form one place", 0.25, 2.0, rf.EPS_KM, 0.25,
                        key="refuge_eps")
        days = st.slider("Minimum separate days of use", 2, 10, rf.MIN_DAYS, key="refuge_days")
        link = st.slider("Night raid reach from a refuge (km)", 1.0, 10.0, rf.LINK_KM, 0.5,
                         key="refuge_link")
    return {"day_window": (start, end), "eps_km": eps, "min_days": days, "link_km": link}


@st.cache_data(show_spinner="Finding day refuges...", max_entries=8)
def analyse(df: pd.DataFrame, villages: Optional[pd.DataFrame], source: Tuple[str, object],
            params: Tuple) -> Dict[str, object]:
    kind, value = source
    cover = None
    if kind == "online":
        cover = lc.LandCover(remote_year=value)
    elif kind == "files":
        cover = lc.LandCover(list(value))
    day_window, eps_km, min_days, link_km = params
    try:
        result = rf.find_refuges(df, villages, cover, day_window=day_window, eps_km=eps_km,
                                 min_days=min_days, link_km=link_km)
    finally:
        if cover is not None:
            cover.close()
    result["land_cover"] = cover.label if cover is not None else "None"
    result["land_cover_failures"] = list(cover.failures) if cover is not None else []
    return result


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------
def _render_map(table: pd.DataFrame, links: pd.DataFrame, basemap: str) -> None:
    from core import map_engine as me

    points = table.copy()
    view = me._adaptive_view_state(pd.concat([points[["Latitude", "Longitude"]],
                                              links[["Latitude", "Longitude"]]]))
    layers = list(me.boundary_layers(view, None))
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
            subtitle=f"{r['Date']:%d %b %Y}, {int(r['Hour']):02d}:00",
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
            accent=me._hex_to_rgb(CONFIDENCE_COLORS[r["Confidence"]]),
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
    data["_c"] = [list(me._hex_to_rgb(CONFIDENCE_COLORS[c])) + [150]
                  for c in points["Confidence"]]
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=data, get_position="[Longitude, Latitude]",
        get_radius="_r", radius_min_pixels=7, get_fill_color="_c",
        get_line_color=[255, 255, 255, 255], line_width_min_pixels=2, stroked=True,
        pickable=True,
    ))
    me._render_deck(layers, view, basemap)


def map_svg(table: pd.DataFrame, links: pd.DataFrame, width: int = 900, height: int = 600) -> str:
    """The refuge map as vector SVG, for the download menu."""
    from html import escape

    from core.damage_report import _svg, _text, outline_parts, scale_bar
    from core.map_export import _fit_view, _projector

    lats = table["Latitude"].tolist() + links["Latitude"].tolist()
    lons = table["Longitude"].tolist() + links["Longitude"].tolist()
    lat, lon, zoom = _fit_view(lats, lons, width, height)
    project = _projector(lat, lon, zoom, width, height)
    parts = [f'<rect width="{width}" height="{height}" fill="#f4f6f3"/>']
    parts += outline_parts(project, width, height)
    ref = table.set_index("Refuge")
    for r in links.to_dict("records"):
        x0, y0 = project(ref.at[r["Refuge"], "Latitude"], ref.at[r["Refuge"], "Longitude"])
        x1, y1 = project(r["Latitude"], r["Longitude"])
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                     f'stroke="{RAID_COLOR}" stroke-opacity="0.45" stroke-width="1"/>')
        parts.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="2.2" fill="{RAID_COLOR}"/>')
    for r in table.to_dict("records"):
        x, y = project(r["Latitude"], r["Longitude"])
        color = CONFIDENCE_COLORS[r["Confidence"]]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{6 + min(r["Days Used"], 20) * 0.5:.1f}" '
                     f'fill="{color}" fill-opacity="0.85" stroke="#ffffff" stroke-width="1.5">'
                     f'<title>{escape(r["Refuge"])}</title></circle>')
        parts.append(_text(x + 10, y - 8, r["Refuge"], 11, weight="bold"))
    parts += scale_bar(lat, zoom, height)
    legend_x = width - 150
    for k, (name, color) in enumerate(CONFIDENCE_COLORS.items()):
        y = 20 + k * 18
        parts.append(f'<circle cx="{legend_x}" cy="{y - 4}" r="5" fill="{color}"/>')
        parts.append(_text(legend_x + 10, y, f"{name} refuge", 10.5))
    parts.append(f'<line x1="{legend_x - 6}" x2="{legend_x + 6}" y1="{20 + 3 * 18 - 4}" '
                 f'y2="{20 + 3 * 18 - 4}" stroke="{RAID_COLOR}"/>')
    parts.append(_text(legend_x + 10, 20 + 3 * 18, "Linked night raid", 10.5))
    return _svg(width, height, parts, "Elephant day refuges and linked raids")


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
    if "Hour" not in register or register["Hour"].notna().sum() == 0:
        st.warning("The register has no time of sighting, so day and night cannot be told "
                   "apart and no refuge can be identified.")
        return

    source = land_cover_source()
    params = settings()
    key = (params["day_window"], params["eps_km"], params["min_days"], params["link_km"])
    result = analyse(register, villages, source, key)
    table, links, s = result["refuges"], result["links"], result["summary"]

    st.caption(f"Land cover: {result['land_cover']}.")
    for failure in result["land_cover_failures"][:3]:
        st.warning(f"Land cover could not be read: {failure}")

    note = dict(delta_color="off", delta_arrow="off", border=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Day refuges", f"{s['refuges']}",
              delta=f"{s['confirmed']} confirmed, {s['probable']} probable", **note)
    c2.metric("Linked night raids", f"{s['linked_raids']:,}",
              delta=f"of {s['night_raids']:,} night conflict reports", **note)
    c3.metric("Villages within reach", f"{s['villages_to_warn']:,}", delta="of a refuge", **note)
    c4.metric("Daytime reports used", f"{s['day_sightings']:,}",
              delta=f"of {s['timed']:,} timed", **note)

    findings(rf.headlines(result))
    if table.empty:
        for c in result["caveats"]:
            st.caption(c)
        return

    st.markdown("**Refuges and the raids linked to them**")
    _render_map(table, links, basemap)
    from core.exports import download_menu

    download_menu("Day refuge map", lambda t=table, l=links: map_svg(t, l))
    legend = " ".join(
        f'<span class="ci-legend__item"><span class="ci-legend__dot" style="background:{c};">'
        f"</span>{n}</span>" for n, c in CONFIDENCE_COLORS.items())
    st.markdown(f'<div class="ci-legend">{legend}<span class="ci-legend__item">'
                f'<span class="ci-legend__dot" style="background:{RAID_COLOR};"></span>'
                "Linked night raid</span></div>", unsafe_allow_html=True)

    st.markdown("**Refuge register** -- strongest evidence first")
    shown = [c for c in [
        "Refuge", "Confidence", "Beat", "Range", "Days Used", "Last Used", "Months Used",
        "Group", "Cover Share %", "Tree Patch (ha)", "Linked Raids", "Raids Head",
        "Raid Distance (km, median)", "Villages Raided", "Nearest Village",
        "To Nearest Village (km)", "Flags", "Recommended Action"] if c in table]
    st.dataframe(table[shown], width="stretch", hide_index=True, column_config={
        "Last Used": st.column_config.DateColumn(format="DD MMM YYYY"),
        "Cover Share %": st.column_config.ProgressColumn("Cover", min_value=0, max_value=100,
                                                         format="%.0f%%"),
        "Recommended Action": st.column_config.TextColumn(width="large"),
    })
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Refuges (CSV)", table.to_csv(index=False).encode(), "day_refuges.csv",
                       mime="text/csv", width="stretch")
    d2.download_button("Refuges + raids (GeoJSON)", rf.to_geojson(table, links).encode(),
                       "day_refuges.geojson", mime="application/geo+json", width="stretch")
    d3.download_button("Refuges (KML, Google Earth)", rf.to_kml(table).encode(),
                       "day_refuges.kml", mime="application/vnd.google-earth.kml+xml",
                       width="stretch")
    d4.download_button("Linked raids (CSV)", links.to_csv(index=False).encode(),
                       "refuge_linked_raids.csv", mime="text/csv", width="stretch")
    st.caption("GeoJSON opens in QGIS; KML in Google Earth or a phone map app for field teams.")

    _cards(table, links, result["windows"])
    with st.expander("Method, and what this cannot say", expanded=False):
        st.markdown(rf.__doc__.split("\n", 1)[1])
        for c in result["caveats"]:
            st.markdown(f"- {c}")


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
