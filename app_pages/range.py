"""Range & corridors: where elephants range, the next fortnight, and the
corridors between the places they use most.

The register holds presence reports only, filed where people are. The
model behind this page (core.presence) separates the two as far as the
register allows. Its next-fortnight forecast is tested on past
fortnights. Its map of use is corrected for reporting effort, and the
corridors are traced across that map (core.corridors). Like Next moves,
it reads the whole register up to the end of the period, whatever the
area in the scope bar.
"""

from __future__ import annotations

import json
from html import escape
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from app_pages.common import corridors_for, presence_for
from core import boundaries
from core import corridors as cr
from core import presence as pr
from core.brief_insights import names, plural
from core.map_engine import (CORE_COLOR, CORRIDOR_COLOR, FORECAST_BANDS, USE_BANDS,
                             render_presence_map)
from core.ui import map_legend, section

NEXT, RANGE, CORRIDORS = "Next fortnight", "Where they range", "Corridors"
VIEWS = (NEXT, RANGE, CORRIDORS)
EFFORT_UPLOAD = "range_effort_upload"
VILLAGE_EFFORT = "range_village_effort"

Villages = Optional[pd.DataFrame]

_RIVAL_PHRASE = {
    "Last fortnight's reports": "a map of last fortnight's reports",
    "All past reports": "a map of all past reports",
}


def _day(ts: pd.Timestamp) -> str:
    return f"{ts.day} {ts:%b %Y}"


def _uploaded_effort() -> Optional[pd.DataFrame]:
    upload = st.session_state.get(EFFORT_UPLOAD)
    if upload is None:
        return None
    try:
        return pr.read_effort(upload)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Could not read the effort file: {exc}")
        return None


def _track_record(result: pr.Result) -> None:
    record = result.record
    if record.tested < pr.MIN_TESTED:
        st.info("Not enough history to test the forecast on past fortnights yet: it needs "
                f"about {pr.BURN_IN + pr.MIN_FIT_BLOCKS + pr.MIN_TESTED} fortnights with "
                "reports. The maps below are fitted, not tested.")
        return
    rival = record.best_baseline()
    other = next(name for name in pr.BASELINES if name != rival)
    said = (f"Tested on {plural(record.tested, 'past fortnight')} ({record.records:,} "
            "records), each forecast only from the fortnights before it. Half of each "
            f"fortnight's records fell in the forecast's top **{record.half_km2():,.0f} "
            f"km²**; {_RIVAL_PHRASE[rival]} needed {record.half_km2(rival):,.0f} km² "
            f"and {_RIVAL_PHRASE[other]} {record.half_km2(other):,.0f} km². It did better "
            f"than {_RIVAL_PHRASE[rival]} in {record.wins()} of {record.tested}.")
    if record.beats_baselines:
        st.markdown(f":material/verified: {said}")
    else:
        st.warning(said + f" That is no better than {_RIVAL_PHRASE[rival]}: read the "
                   "forecast as that map, smoothed.", icon=":material/info:")


def _legend(result: pr.Result, colours: Dict[str, tuple], tested: bool) -> None:
    curve = result.record.curve() if tested else {}
    held, before = {}, 0.0
    for (label, area), share in zip(pr.BANDS, pr.AREA_SHARES):
        if share in curve:
            held[label] = curve[share] - before
            before = curve[share]
    entries = {}
    for label, _ in pr.BANDS:
        text = f"{label} of the area"
        if label in held:
            text += f" (held {held[label]:.0%} of records in tests)"
        entries[label] = {"accent": "#%02x%02x%02x" % colours[label], "glyph": "",
                          "label": text}
    map_legend("Bands", entries)


def _cells(result: pr.Result, villages: Villages, share: np.ndarray) -> pd.DataFrame:
    lattice = result.inputs.lattice
    band = pr.bands(share)
    shown = band != ""
    village, _ = pr.nearest_village(lattice, villages)
    frame = pd.DataFrame({
        "Latitude": lattice.lat[shown], "Longitude": lattice.lon[shown], "Band": band[shown],
        "Share": [f"{s:.2%}" for s in share[shown]],
        "Place": village[shown],
    })
    if boundaries.available():
        frame["Beat"] = boundaries.locate(boundaries.BEAT, frame["Longitude"], frame["Latitude"])
    return frame


def _places_table(result: pr.Result, villages: Villages, share: np.ndarray,
                  title: str) -> pd.DataFrame:
    table = pr.places(result.inputs.lattice, share, villages)
    st.markdown(f"**{title}**")
    if table.empty:
        st.caption("No village lies within 5 km of the banded cells.")
        return table
    shown = table[["Village", "Share", "Band"]].copy()
    shown["Share"] = shown["Share"] * 100
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        "Share": st.column_config.NumberColumn(
            "Share", format="%.1f%%",
            help="Share of the map's total held by the cells nearest this village."),
        "Band": st.column_config.TextColumn("Band", help="Its best cell's band."),
    })
    return table


def _effect_of_correction(result: pr.Result, villages: Villages, season: str) -> str:
    """How much use the correction moves away from villages: the share in
    the half of the landscape farthest from one, before and after."""
    lattice = result.inputs.lattice
    _, km = pr.nearest_village(lattice, villages, within_km=np.inf)
    if not np.isfinite(km).any() or pr.VILLAGE_KM not in result.inputs.effort:
        return ""
    cut = float(np.nanmedian(km))
    far = km > cut
    corrected = result.use(season, corrected=True)[far].sum()
    reported = result.use(season, corrected=False)[far].sum()
    return (f"Corrected for reporting effort, {corrected:.0%} of use lies in the half of the "
            f"landscape farthest from villages (over {cut:.1f} km from one), against "
            f"{reported:.0%} from reports alone.")


def _core_names(result: pr.Result, villages: Villages, net: cr.Network) -> pd.DataFrame:
    lattice = result.inputs.lattice
    rows = []
    for core in net.cores:
        label = chr(ord("A") + core.id)
        near = ""
        if villages is not None and len(villages):
            vx, vy = lattice.xy(villages["Latitude"].to_numpy(float),
                                villages["Longitude"].to_numpy(float))
            cx, cy = lattice.xy(core.latitude, core.longitude)
            k = int(np.nanargmin((vx - cx) ** 2 + (vy - cy) ** 2))
            near = str(villages["Village"].iloc[k])
        rows.append({"Core": f"Core {label}", "Latitude": core.latitude,
                     "Longitude": core.longitude, "Area (km2)": core.area_km2,
                     "Share": f"{core.use:.0%}", "Place": near})
    return pd.DataFrame(rows)


def _corridor_cells(result: pr.Result, villages: Villages, net: cr.Network,
                    cores: pd.DataFrame) -> pd.DataFrame:
    lattice = result.inputs.lattice
    owner = net.corridor_cells()
    inside = np.flatnonzero(owner >= 0)
    if not len(inside):
        return pd.DataFrame(columns=["Latitude", "Longitude", "Current"])
    current = net.current[inside] if net.current is not None else np.ones(len(inside))
    top = np.quantile(current, 0.98) if len(current) > 10 else current.max()
    scaled = np.clip(current / top, 0, 1) if top > 0 else np.zeros(len(current))
    village, _ = pr.nearest_village(lattice, villages)
    label = {c.id: name for c, name in zip(net.cores, cores["Core"])}
    links = [f"{label[net.links[k].a]} and {label[net.links[k].b]}" for k in owner[inside]]
    return pd.DataFrame({
        "Latitude": lattice.lat[inside], "Longitude": lattice.lon[inside],
        "Current": scaled, "Link": links, "Place": village[inside],
        "Flow": np.where(scaled >= 0.66, "Pinch point: most movement",
                         np.where(scaled >= 0.33, "Moderate", "Light")),
    })


def _links_table(result: pr.Result, villages: Villages, net: cr.Network,
                 cores: pd.DataFrame) -> pd.DataFrame:
    label = {c.id: name for c, name in zip(net.cores, cores["Core"])}
    place = {c.id: p for c, p in zip(net.cores, cores["Place"])}
    rows = []
    for link in net.links:
        on = cr.villages_on(result.inputs.lattice, link, villages)
        rows.append({
            "Between": f"{label[link.a]} ({place[link.a]}) and {label[link.b]} "
                       f"({place[link.b]})",
            "Length (km)": round(link.length_km, 1),
            "Straight (km)": round(link.straight_km, 1),
            "Villages on the way": names(on["Village"], limit=8) if len(on) else "",
        })
    return pd.DataFrame(rows)


def _geojson(result: pr.Result, net: cr.Network, cores: pd.DataFrame) -> bytes:
    lattice = result.inputs.lattice
    label = {c.id: name for c, name in zip(net.cores, cores["Core"])}
    features = []
    for link in net.links:
        features.append({"type": "Feature", "properties": {
            "kind": "least-cost path", "from": label[link.a], "to": label[link.b],
            "length_km": round(link.length_km, 2), "cost": round(link.cost, 1)},
            "geometry": {"type": "LineString", "coordinates": [
                [round(float(lon), 5), round(float(lat), 5)]
                for lat, lon in zip(lattice.lat[link.path], lattice.lon[link.path])]}})
    for row in cores.to_dict("records"):
        features.append({"type": "Feature", "properties": {
            "kind": "core area", "name": row["Core"], "area_km2": row["Area (km2)"]},
            "geometry": {"type": "Point", "coordinates": [round(row["Longitude"], 5),
                                                          round(row["Latitude"], 5)]}})
    return json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")


def _surfaces_csv(result: pr.Result) -> bytes:
    lattice = result.inputs.lattice
    frame = pd.DataFrame({"cell": np.arange(lattice.size),
                          "latitude": np.round(lattice.lat, 5),
                          "longitude": np.round(lattice.lon, 5),
                          "next_fortnight": result.next_fortnight(),
                          "next_fortnight_effort_corrected": result.next_fortnight(True)})
    for season in result.seasons():
        key = season.split(" (")[0].lower().replace(" ", "_")
        frame[f"use_{key}"] = result.use(season, corrected=True)
    return frame.to_csv(index=False).encode("utf-8")


def _effort_controls(result: pr.Result, c) -> None:
    in_use = [result.inputs.labels[n] for n in result.inputs.effort]
    st.markdown("Corrected for: " + (", ".join(in_use) if in_use else "nothing yet") + ".")
    st.checkbox("Count distance from villages as reporting effort", key=VILLAGE_EFFORT,
                help="People report from villages, so reports cluster there. But crops "
                     "draw elephants to villages too. Correcting for it removes both.")
    st.file_uploader("Add reporting effort (CSV of Latitude, Longitude)", type="csv",
                     key=EFFORT_UPLOAD,
                     help="App-open pings, patrol track points or app users' homes: where "
                          "people who report are. Only the coordinates are read.")
    st.caption("The villager registry and the app-user list, when loaded under Optional "
               "data, are used as effort too. Only coordinates reach the model, rounded to "
               "about 100 m, and only per-cell densities leave it.")


def _how(result: pr.Result) -> None:
    inputs = result.inputs
    record = result.record
    chosen = record.spec.parts() if record.tested else (
        result.forecast.spec.parts() if result.forecast else [])
    bits = (f" Per record, the forecast scored {record.gain_bits():+.2f} bits against the "
            f"better simple map (positive is sharper)." if record.tested else "")
    st.markdown(
        f"- **Thinning**: {inputs.reports:,} reports became {inputs.records:,} records, one "
        f"per {inputs.lattice.cell_km:g} km cell per day. Five reports of one herd in one "
        "square on one day are one piece of evidence.\n"
        f"- **Fortnights**: {inputs.blocks.count} blocks of {inputs.blocks.days} days ending "
        f"{_day(inputs.blocks.end - pd.Timedelta(days=1))}, over {inputs.lattice.size:,} cells "
        f"within {pr.DOMAIN_KM:g} km of a report.\n"
        "- **The model** is a Poisson point process on the cells. Its log intensity adds "
        "habitat, reporting effort, a smooth surface of its own, seasonal shifts of that "
        "surface and, optionally, the recent past. It is fitted conditional on each "
        "fortnight's total, so a register that grows as more people use the app does not "
        "bend the map. MaxEnt fits the same model, so it is not run separately.\n"
        "- **Effort-corrected** maps hold every effort term level everywhere: where "
        "elephants are, not where they are seen from. They cannot be tested against "
        "reports, which carry the bias they remove.\n"
        "- **Which version forecasts** is chosen by the tests: on this register, "
        + (", ".join(chosen) or "none yet") + "." + bits + "\n"
        "- **Corridors** join the core areas (touching cells in the top 5% of use) by "
        "their cheapest routes across the effort-corrected map, where less use means "
        "harder going. Each corridor holds every route within 5% of the cheapest; its "
        "shading is the current a circuit model sends through it (Circuitscape's "
        "pairwise mode), so the darkest stretches are pinch points.\n"
        "- **What it cannot do**: nobody reports \"looked, saw none\", so this is relative "
        "use, not the chance a place is occupied. Dynamic occupancy models (colonisation "
        "and extinction) need those non-detections. The recent-past terms are their "
        "presence-only stand-in. Graph neural networks need far more data than a "
        "district register holds. Distance from villages is effort and a real pull (crops) "
        "at once; app-open pings or patrol tracks would separate them."
    )


def _headline(text: str) -> None:
    st.markdown(f'<p class="ci-headline">{escape(text)}</p>', unsafe_allow_html=True)


def _next_view(c, result: pr.Result, villages: Villages, corrected: bool) -> None:
    share = result.next_fortnight(corrected=corrected)
    top = pr.places(result.inputs.lattice, share, villages, limit=6)
    what = "elephants are most likely" if corrected else "most records are expected"
    _headline(f"Next fortnight ({_day(result.next_start)} to {_day(result.next_end)}): "
              + (f"{what} around {names(top['Village'], limit=6)}." if len(top)
                 else "no village lies near the most expected cells."))
    map_col, table_col = st.columns([3, 2], gap="medium")
    with map_col:
        render_presence_map(_cells(result, villages, share), result.inputs.lattice.cell_km,
                            FORECAST_BANDS, "Next fortnight", style_name=c.basemap,
                            show_boundaries=c.layer_boundaries,
                            boundary_level=c.boundary_level)
        _legend(result, FORECAST_BANDS,
                tested=not corrected and result.record.tested >= pr.MIN_TESTED)
        if corrected:
            st.caption("Effort-corrected: where elephants are likely to be, not where "
                       "reports will come from. Not tested: reports carry the bias it "
                       "removes.")
    with table_col:
        _places_table(result, villages, share, "Villages to watch")


def _range_view(c, result: pr.Result, villages: Villages, season: str,
                corrected: bool) -> None:
    share = result.use(season, corrected=corrected)
    top = pr.places(result.inputs.lattice, share, villages, limit=6)
    when = "" if season == pr.ALL_YEAR else f" in {season.split(' (')[0].lower()}"
    _headline(f"Most use{when} lies around {names(top['Village'], limit=6)}."
              if len(top) else "No village lies near the most-used cells.")
    map_col, table_col = st.columns([3, 2], gap="medium")
    with map_col:
        render_presence_map(_cells(result, villages, share), result.inputs.lattice.cell_km,
                            USE_BANDS, "Use", style_name=c.basemap,
                            show_boundaries=c.layer_boundaries,
                            boundary_level=c.boundary_level)
        _legend(result, USE_BANDS, tested=False)
        note = _effect_of_correction(result, villages, season)
        if note:
            st.caption(note)
    with table_col:
        _places_table(result, villages, share, "Most-used places")


def _corridor_view(c, result: pr.Result, villages: Villages, season: str,
                   corrected: bool) -> None:
    net = corridors_for(result, pr.result_key(result), season, corrected)
    cores = _core_names(result, villages, net)
    if not net.links:
        st.info("The most-used places form one connected block on this map, so there is "
                "no corridor between separate core areas to trace. Try another season.")
    else:
        joins = "joins" if len(net.links) == 1 else "join"
        _headline(f"{plural(len(net.links), 'corridor')} {joins} {len(net.cores)} core areas"
                  + ("" if season == pr.ALL_YEAR
                     else f" in {season.split(' (')[0].lower()}") + ".")
    lattice = result.inputs.lattice
    paths = [{"path": [[round(float(lon), 5), round(float(lat), 5)]
                       for lat, lon in zip(lattice.lat[link.path], lattice.lon[link.path])]}
             for link in net.links]
    map_col, table_col = st.columns([3, 2], gap="medium")
    with map_col:
        render_presence_map(_cells(result, villages, result.use(season, corrected=corrected)),
                            lattice.cell_km, USE_BANDS, "Use", cores=cores, paths=paths,
                            corridor=_corridor_cells(result, villages, net, cores),
                            style_name=c.basemap, show_boundaries=c.layer_boundaries,
                            boundary_level=c.boundary_level)
        map_legend("Corridors", {
            "core": {"accent": "#%02x%02x%02x" % CORE_COLOR, "glyph": "",
                     "label": "Core area (top 5% of use)"},
            "corridor": {"accent": "#%02x%02x%02x" % CORRIDOR_COLOR, "glyph": "",
                         "label": "Corridor, darker where more movement passes"},
        })
        st.caption("Dark lines are the cheapest routes between core areas across the map "
                   "of use. Hypotheses to check on the ground, not tracks.")
    with table_col:
        if net.links:
            st.markdown("**Corridors**")
            st.dataframe(_links_table(result, villages, net, cores), hide_index=True,
                         width="stretch", column_config={
                             "Between": st.column_config.TextColumn(width="medium"),
                             "Villages on the way": st.column_config.TextColumn(
                                 width="large", help="Within about a cell of the cheapest "
                                                     "route."),
                         })
            st.download_button("Download corridors (GeoJSON)", _geojson(result, net, cores),
                               "corridors.geojson", mime="application/geo+json",
                               on_click="ignore")


def render(c) -> None:
    section("map", "Range & corridors",
            f"Where elephants range and move, from reports to {_day(c.period_end)}")
    register = c.df[c.df["Date"] < c.period_end + pd.Timedelta(days=1)]
    if register[["Latitude", "Longitude"]].dropna().empty:
        st.info("No report with coordinates up to this date.")
        return
    villages = c.villages
    st.session_state.setdefault(VILLAGE_EFFORT, True)
    users = c.roster if c.roster is not None and {"Latitude", "Longitude"} <= set(
        getattr(c.roster, "columns", [])) else None
    result = presence_for(register, c.period_end, c.villages, c.registry, users,
                          _uploaded_effort(), bool(st.session_state[VILLAGE_EFFORT]))
    if not result.fitted:
        st.info(f"Not enough history to model where elephants range: it needs at least "
                f"{pr.BURN_IN + pr.MIN_FIT_BLOCKS} fortnights and {pr.MIN_RECORDS} records "
                f"after thinning. This register has {result.inputs.blocks.count} "
                f"fortnight(s) and {result.inputs.records:,} record(s).")
        with st.expander("Reporting effort", expanded=False):
            _effort_controls(result, c)
        return
    _track_record(result)
    st.caption(f"{result.inputs.reports:,} reports thinned to {result.inputs.records:,} "
               f"records over {result.inputs.blocks.count} fortnights. It refits whenever "
               "the register changes.")

    view = st.segmented_control("Show", VIEWS, default=NEXT, key="range_view") or NEXT
    controls = st.columns([3, 2], vertical_alignment="bottom")
    season = pr.ALL_YEAR
    if view != NEXT and len(result.seasons()) > 1:
        with controls[0]:
            season = st.segmented_control("Season", result.seasons(), default=pr.ALL_YEAR,
                                          key="range_season") or pr.ALL_YEAR
    with controls[1]:
        corrected = st.toggle("Correct for reporting effort", value=view != NEXT,
                              key=f"range_corrected_{view}",
                              help="Hold every reporting-effort term level everywhere: "
                                   "where elephants are, not where they are seen from.")

    if view == NEXT:
        _next_view(c, result, villages, corrected)
    elif view == RANGE:
        _range_view(c, result, villages, season, corrected)
    else:
        _corridor_view(c, result, villages, season, corrected)

    with st.expander("What drives it", expanded=False):
        effects = result.effects()
        if len(effects):
            st.dataframe(effects[["Term", "Kind", "Effect"]], hide_index=True, width="stretch")
            st.caption("Each effect holds the others fixed. The recent-past terms overlap; "
                       "read them together. Patterns, not proof of cause.")
    with st.expander("Reporting effort", expanded=False):
        _effort_controls(result, c)
    with st.expander("How this is worked out, and what it cannot do", expanded=False):
        _how(result)
        left, right = st.columns(2)
        left.download_button(
            "Model inputs for R (CSV)", lambda: result.model_table().to_csv(index=False),
            "presence_model_inputs.csv", mime="text/csv", on_click="ignore",
            help="One row per cell per fortnight: records, covariates and recent-past "
                 "terms, ready for inlabru, R-INLA or mgcv.")
        right.download_button("Surfaces (CSV)", lambda: _surfaces_csv(result),
                              "presence_surfaces.csv", mime="text/csv", on_click="ignore")
