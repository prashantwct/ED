"""Daytime refuges: where elephants lie up by day, and what they raid by night.

In this landscape elephants feed in fields at night and spend the day in
cover -- a forest patch, a scrubby nala, a plantation -- usually within a
night's walk of the fields they raid. A refuge used again and again is a
fixed point in an otherwise moving problem: a place to watch at dusk,
villages to warn before dark, people to keep out of it by day.

A refuge here is not a cluster of sightings. It has to pass five tests,
each of which removes a known way of being wrong:

1. **Rest hours only.** Sightings between ``DAY_START`` and ``DAY_END``
   (09:00-16:00 by default). Dawn and dusk sightings are left out: they
   catch animals moving between field and cover, and would put "refuges"
   on the paths in between.
2. **Recurrence.** Day sightings on at least ``MIN_DAYS`` separate days.
   One afternoon of reports, however many, is a herd passing through.
3. **Day, not night, use.** Of all timed sightings at the place, at least
   half by day. A place seen mostly at night is a feeding ground.
4. **Stability.** The clustering is re-run over a grid of neighbour
   distances and rest windows; a refuge must reappear within 1 km in
   most of them. One that only exists at one setting is an artefact of
   that setting.
5. **Cover.** With Esri land cover, the share of tree, rangeland and
   flooded-vegetation pixels within 250 m, and the size of the tree patch
   the refuge sits in. Day use with no cover is flagged, not hidden: it
   is either a position error or animals lying up in tall crop, and both
   matter.

Each refuge is then linked to the night conflict it feeds: conflict
reported between ``NIGHT_START`` and ``NIGHT_END`` within ``LINK_KM`` of
the refuge on the night before or after a day it was in use. A raid near
two refuges goes to the nearer one in use, so nothing is counted twice.

**What this cannot say.** Positions are the observer's, not the
elephant's, often a few hundred metres off -- hence the 250 m cover disc
rather than the single pixel. Daytime reports depend on where trackers
go: a refuge nobody visits is invisible here, so absence of a refuge is
not evidence of absence. And the linking is by place and date, not
identity: a raid near a refuge on a night it was in use is consistent
with coming from it, not proof.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.analytics import classify_group, conflict_mask, human_deaths, human_injuries
from core.hotspots import _dbscan, _to_km_plane

DAY_START, DAY_END = 9, 16            # rest window [start, end)
NIGHT_START, NIGHT_END = 18, 6        # raid window, wraps midnight
EPS_KM = 0.75                         # neighbour distance for day clusters
MIN_SIGHTINGS = 3
MIN_DAYS = 3
LINK_KM = 5.0                         # a night's foray from the refuge
LINK_DAYS = 1                         # raid on the night before, or the evening of, a day in use
COVER_DISC_M = 250.0
COVER_PASS = 40.0       # % cover within the disc for Confirmed
COVER_FAIL = 25.0       # below this the cover test fails
CONFIRM_DAYS = 5
MIN_DAY_SHARE = 40.0    # below this a place is a night feeding ground
WINDOW_HALF_M = 3000.0

# Stability grid: neighbour distance x rest window.
STABILITY_EPS = (0.5, 0.75, 1.0)
STABILITY_WINDOWS = ((8, 17), (9, 16), (10, 15))
STABILITY_MATCH_KM = 1.0

CONFIRMED, PROBABLE, POSSIBLE = "Confirmed", "Probable", "Possible"
CONFIDENCE_ORDER = [CONFIRMED, PROBABLE, POSSIBLE]

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _bearing(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def compass(degrees: float) -> str:
    return COMPASS[int((degrees + 22.5) // 45) % 8]


def _circular_mean(degrees: np.ndarray) -> Tuple[float, float]:
    """Mean bearing and its concentration (0 = all directions, 1 = one)."""
    rad = np.radians(degrees)
    s, c = np.sin(rad).mean(), np.cos(rad).mean()
    return float((np.degrees(np.arctan2(s, c)) + 360) % 360), float(math.hypot(s, c))


# ---------------------------------------------------------------------------
# Day and night
# ---------------------------------------------------------------------------
def is_day(hours: pd.Series, start: int = DAY_START, end: int = DAY_END) -> pd.Series:
    return hours.notna() & (hours >= start) & (hours < end)


def is_night(hours: pd.Series, start: int = NIGHT_START, end: int = NIGHT_END) -> pd.Series:
    return hours.notna() & ((hours >= start) | (hours < end))


def _candidates(df: pd.DataFrame, eps_km: float, window: Tuple[int, int],
                min_sightings: int, min_days: int) -> List[pd.DataFrame]:
    """Day-sighting clusters seen on enough separate days."""
    day = df[is_day(df["Hour"], *window)]
    if len(day) < min_sightings:
        return []
    labels = _dbscan(_to_km_plane(day["Latitude"].to_numpy(float),
                                  day["Longitude"].to_numpy(float)),
                     eps=eps_km, min_samples=min_sightings)
    out = []
    for label in sorted(set(labels) - {-1}):
        members = day[labels == label]
        if members["Date"].dt.normalize().nunique() >= min_days:
            out.append(members)
    return out


def _centre(members: pd.DataFrame) -> Tuple[float, float]:
    return float(members["Latitude"].median()), float(members["Longitude"].median())


# ---------------------------------------------------------------------------
# Land cover at a refuge
# ---------------------------------------------------------------------------
def cover_metrics(window) -> Dict[str, float]:
    """What the Esri land cover says about the place."""
    from scipy import ndimage

    from core import esri_landcover as lc

    codes = window.codes
    valid = codes > 0
    disc = window.disc(COVER_DISC_M) & valid
    n = max(int(disc.sum()), 1)
    share = {name: float(((codes == code) & disc).sum()) / n
             for code, name in ((lc.TREES, "trees"), (lc.RANGELAND, "rangeland"),
                                (lc.FLOODED, "flooded"), (lc.CROPS, "crops"),
                                (lc.BUILT, "built"), (lc.WATER, "water"))}
    cover = share["trees"] + share["rangeland"] + share["flooded"]

    # The tree patch the refuge sits in: the component holding the
    # nearest tree pixel within the cover disc.
    trees = codes == lc.TREES
    patch_ha, patch_open = 0.0, False
    near = trees & window.disc(COVER_DISC_M)
    if near.any():
        labels, _ = ndimage.label(trees, structure=np.ones((3, 3)))
        rows, cols = np.nonzero(near)
        k = np.argmin((rows - window.row) ** 2 + (cols - window.col) ** 2)
        component = labels == labels[rows[k], cols[k]]
        patch_ha = float(component.sum() * window.pixel_ha)
        patch_open = bool(component[0, :].any() or component[-1, :].any()
                          or component[:, 0].any() or component[:, -1].any())

    def distance_m(code: int) -> float:
        mask = codes == code
        if not mask.any():
            return float("nan")
        edt = ndimage.distance_transform_edt(~mask, sampling=(window.dy_m, window.dx_m))
        return float(round(edt[window.row, window.col]))

    crop_2km = window.disc(2000.0) & valid
    return {
        "Cover Share %": round(cover * 100, 1),
        "Tree Share %": round(share["trees"] * 100, 1),
        "Rangeland Share %": round(share["rangeland"] * 100, 1),
        "Crop Share %": round(share["crops"] * 100, 1),
        "Built Share %": round(share["built"] * 100, 1),
        "Water Share %": round(share["water"] * 100, 1),
        "Tree Patch (ha)": round(patch_ha, 1),
        "Patch Extends Beyond Window": patch_open,
        "To Cropland (m)": distance_m(lc.CROPS),
        "To Built Area (m)": distance_m(lc.BUILT),
        "To Water (m)": distance_m(lc.WATER),
        "Cropland Within 2 km %": round(
            float(((codes == lc.CROPS) & crop_2km).sum()) / max(int(crop_2km.sum()), 1) * 100, 1),
    }


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------
def find_refuges(
    df: pd.DataFrame,
    villages: Optional[pd.DataFrame] = None,
    land_cover=None,
    *,
    day_window: Tuple[int, int] = (DAY_START, DAY_END),
    eps_km: float = EPS_KM,
    min_days: int = MIN_DAYS,
    link_km: float = LINK_KM,
    as_of: Optional[pd.Timestamp] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Find day refuges in the register and link each to its night raids.

    Returns ``refuges`` (one row per refuge, most-used first), ``links``
    (one row per linked raid), ``windows`` (land-cover windows by refuge
    id, for thumbnails), ``summary`` and ``caveats``.
    """
    timed = df[df["Hour"].notna() & df["Date"].notna()
               & df["Latitude"].notna() & df["Longitude"].notna()].copy()
    timed["Hour"] = timed["Hour"].astype(int)
    caveats = []
    untimed = len(df) - len(timed)
    if untimed:
        caveats.append(f"{untimed:,} report(s) without an hour cannot be placed in day or "
                       "night and are left out.")
    day_n = int(is_day(timed["Hour"], *day_window).sum())
    if not len(timed) or day_n == 0:
        return _empty(caveats + ["No reports fall in the daytime rest window, so no "
                                 "refuge can be identified."])

    as_of = pd.Timestamp(as_of) if as_of is not None else timed["Date"].max()
    base = _candidates(timed, eps_km, day_window, MIN_SIGHTINGS, min_days)

    # Stability: how many settings in the grid find a refuge here too.
    grid = [(e, w) for e in STABILITY_EPS for w in STABILITY_WINDOWS]
    centres_by_setting = []
    for e, w in grid:
        found = _candidates(timed, e, w, MIN_SIGHTINGS, min_days)
        centres_by_setting.append(np.array([_centre(m) for m in found]) if found
                                  else np.empty((0, 2)))

    night = timed[is_night(timed["Hour"])]
    raids = night[conflict_mask(night)].copy() if len(night) else night
    if len(raids):
        raids["_deaths"] = human_deaths(raids)
        raids["_injuries"] = human_injuries(raids)

    rows, windows, use_days = [], {}, {}
    for i, members in enumerate(base):
        if progress:
            progress(f"Assessing candidate {i + 1} of {len(base)}")
        lat, lon = _centre(members)
        d = _km(lat, lon, members["Latitude"].to_numpy(float), members["Longitude"].to_numpy(float))
        radius = float(max(np.percentile(d, 90), 0.25))

        near_all = _km(lat, lon, timed["Latitude"].to_numpy(float),
                       timed["Longitude"].to_numpy(float)) <= radius + 0.25
        local = timed[near_all]
        local_day = int(is_day(local["Hour"], *day_window).sum())
        local_night = int(is_night(local["Hour"]).sum())
        day_share = local_day / max(local_day + local_night, 1)

        stable = sum(
            1 for centres in centres_by_setting
            if len(centres) and _km(lat, lon, centres[:, 0], centres[:, 1]).min()
            <= max(STABILITY_MATCH_KM, radius)
        ) / len(grid)

        days = members["Date"].dt.normalize()
        use_days[i] = set(days)
        groups = classify_group(members)
        known = groups[groups != "Unrecorded"]
        months = sorted(set(days.dt.month))

        row = {
            "_i": i,
            "Latitude": round(lat, 5), "Longitude": round(lon, 5),
            "Radius (km)": round(radius, 2),
            "Day Sightings": len(members),
            "Days Used": int(days.nunique()),
            "First Used": days.min(), "Last Used": days.max(),
            "Days Since Last Use": int((as_of.normalize() - days.max()).days),
            "Months Used": ", ".join(pd.Timestamp(2000, m, 1).strftime("%b") for m in months),
            "Day Share %": round(day_share * 100),
            "Stability %": round(stable * 100),
            "Group": known.mode().iat[0] if len(known) else "Unrecorded",
            "Bull-Type Share %": round(float(groups.isin(["Lone bull", "Bull party"]).mean()) * 100),
            "Herd Size (median)": float(pd.to_numeric(members["Total Count"], errors="coerce")
                                        .median()) if "Total Count" in members else float("nan"),
        }
        if land_cover is not None:
            window = land_cover.window(lat, lon, WINDOW_HALF_M)
            if window is not None:
                windows[i] = window
                row.update(cover_metrics(window))
        rows.append(row)

    feeding = [r for r in rows if r["Day Share %"] < MIN_DAY_SHARE]
    rows = [r for r in rows if r["Day Share %"] >= MIN_DAY_SHARE]
    if feeding:
        caveats.append(f"{len(feeding)} place(s) with repeated day sightings were seen mostly at "
                       "night -- feeding grounds, not refuges -- and are left out.")
    if not rows:
        return _empty(caveats + [f"No place had daytime sightings on {min_days} or more "
                                 "separate days and mostly daytime use."], len(timed), day_n)

    table = pd.DataFrame(rows)
    if len(raids) and "Nearest Village" not in raids and villages is not None and len(villages):
        from core.spatial import attach_nearest_village

        raids, _ = attach_nearest_village(raids, villages)
    links = _link_raids(table, use_days, raids, link_km)
    table = _attach_links(table, links)
    table = _attach_villages(table, villages, link_km)
    table = _attach_admin(table)
    table["Confidence"] = table.apply(_confidence, axis=1, land_cover=land_cover is not None)
    table["Flags"] = table.apply(_flags, axis=1)
    table["Recommended Action"] = table.apply(_actions, axis=1)
    table["_o"] = table["Confidence"].map({c: k for k, c in enumerate(CONFIDENCE_ORDER)})
    table = table.sort_values(["_o", "Linked Raids", "Days Used"],
                              ascending=[True, False, False]).reset_index(drop=True)
    table.insert(0, "Refuge", [f"R{k + 1:02d}" for k in range(len(table))])
    ids = dict(zip(table["_i"], table["Refuge"]))
    links["Refuge"] = links["_i"].map(ids)
    windows = {ids[k]: v for k, v in windows.items() if k in ids}
    table = table.drop(columns=["_i", "_o"])
    links = links.drop(columns=["_i"])

    if land_cover is None:
        caveats.append("No land cover loaded: refuges rest on sighting evidence alone, and "
                       "none can reach Confirmed without the cover check.")
    elif getattr(land_cover, "failures", None):
        caveats.append(f"Land cover could not be read for {len(land_cover.failures)} "
                       "source(s); refuges there carry no cover figures.")
    missing_lc = int(table.get("Cover Share %", pd.Series(dtype=float)).isna().sum()) \
        if land_cover is not None else 0
    if missing_lc:
        caveats.append(f"{missing_lc} refuge(s) fall outside the land cover provided.")
    caveats += [
        "Positions are the observer's, not the elephant's; cover is measured over a "
        f"{COVER_DISC_M:.0f} m disc rather than the single pixel for that reason.",
        "A refuge appears only where trackers go by day. No refuge in an area is not "
        "evidence that elephants do not lie up there.",
        f"Raids are linked by place ({link_km:g} km) and night (the one before or after a "
        "day in use), not identity: consistent with coming from the refuge, not proof.",
    ]
    summary = {
        "timed": len(timed), "day_sightings": day_n, "refuges": len(table),
        "confirmed": int((table["Confidence"] == CONFIRMED).sum()),
        "probable": int((table["Confidence"] == PROBABLE).sum()),
        "linked_raids": int(len(links)),
        "night_raids": int(len(raids)),
        "villages_to_warn": int(len(set(", ".join(table["Villages Within Reach"].dropna())
                                       .split(", ")) - {""})),
    }
    return {"refuges": table, "links": links, "windows": windows,
            "summary": summary, "caveats": caveats}


def _empty(caveats, timed=0, day=0) -> Dict[str, object]:
    return {"refuges": pd.DataFrame(), "links": pd.DataFrame(), "windows": {},
            "summary": {"timed": timed, "day_sightings": day, "refuges": 0, "confirmed": 0,
                        "probable": 0, "linked_raids": 0, "night_raids": 0,
                        "villages_to_warn": 0},
            "caveats": caveats}


def _link_raids(table: pd.DataFrame, use_days: Dict[int, set], raids: pd.DataFrame,
                link_km: float) -> pd.DataFrame:
    """Each night raid to the nearest refuge in use the day before or after."""
    cols = ["_i", "Date", "Hour", "Latitude", "Longitude", "Distance (km)", "Bearing",
            "Village", "Deaths", "Injuries", "Category"]
    if raids is None or not len(raids):
        return pd.DataFrame(columns=cols)
    from core.analytics import classify_conflict

    cats = classify_conflict(raids)
    out = []
    for idx, raid in raids.iterrows():
        # A raid after midnight belongs to the evening before it.
        night_of = raid["Date"].normalize() - pd.Timedelta(days=1 if raid["Hour"] < 12 else 0)
        best = None
        for _, ref in table.iterrows():
            # Lying up on day D, the animals raid the evening of D (leaving at
            # dusk) or came in from the night before (D - 1).
            if not any(-LINK_DAYS <= (night_of - d).days <= 0 for d in use_days[ref["_i"]]):
                continue
            dist = float(_km(ref["Latitude"], ref["Longitude"], raid["Latitude"], raid["Longitude"]))
            if dist <= link_km and (best is None or dist < best[1]):
                best = (ref, dist)
        if best is None:
            continue
        ref, dist = best
        out.append({
            "_i": ref["_i"], "Date": raid["Date"], "Hour": int(raid["Hour"]),
            "Latitude": raid["Latitude"], "Longitude": raid["Longitude"],
            "Distance (km)": round(dist, 2),
            "Bearing": float(_bearing(ref["Latitude"], ref["Longitude"],
                                      raid["Latitude"], raid["Longitude"])),
            "Village": raid.get("Nearest Village", ""),
            "Deaths": float(raid["_deaths"]), "Injuries": float(raid["_injuries"]),
            "Category": cats.loc[idx],
        })
    return pd.DataFrame(out, columns=cols)


def _attach_links(table: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    linked, deaths, injuries, houses, direction, reach, villages = [], [], [], [], [], [], []
    for i in table["_i"]:
        part = links[links["_i"] == i] if len(links) else links
        linked.append(len(part))
        deaths.append(int(part["Deaths"].sum()) if len(part) else 0)
        injuries.append(int(part["Injuries"].sum()) if len(part) else 0)
        houses.append(int((part["Category"] == "House").sum()) if len(part) else 0)
        if len(part):
            mean, spread = _circular_mean(part["Bearing"].to_numpy(float))
            direction.append(compass(mean) if spread >= 0.5 else "Several directions")
            reach.append(round(float(part["Distance (km)"].median()), 1))
            names = part["Village"].replace("", np.nan).dropna()
            villages.append(", ".join(names.value_counts().index[:4]))
        else:
            direction.append("")
            reach.append(np.nan)
            villages.append("")
    table["Linked Raids"] = linked
    table["Linked Deaths"] = deaths
    table["Linked Injuries"] = injuries
    table["Linked House Damage"] = houses
    table["Raids Head"] = direction
    table["Raid Distance (km, median)"] = reach
    table["Villages Raided"] = villages
    return table


def _attach_villages(table: pd.DataFrame, villages: Optional[pd.DataFrame],
                     link_km: float) -> pd.DataFrame:
    table = table.copy()
    if villages is None or villages.empty:
        table["Nearest Village"] = ""
        table["To Nearest Village (km)"] = np.nan
        table["Villages Within Reach"] = ""
        return table
    vlat = villages["Latitude"].to_numpy(float)
    vlon = villages["Longitude"].to_numpy(float)
    names = villages["Village"].astype(str).to_numpy()
    nearest, near_km, within = [], [], []
    for lat, lon in zip(table["Latitude"], table["Longitude"]):
        d = _km(lat, lon, vlat, vlon)
        order = np.argsort(d)
        nearest.append(names[order[0]])
        near_km.append(round(float(d[order[0]]), 2))
        close = order[d[order] <= link_km]
        within.append(", ".join(names[close[:6]]) + (f" (+{len(close) - 6} more)"
                                                       if len(close) > 6 else ""))
    table["Nearest Village"] = nearest
    table["To Nearest Village (km)"] = near_km
    table["Villages Within Reach"] = within
    return table


def _attach_admin(table: pd.DataFrame) -> pd.DataFrame:
    from core import boundaries

    table = table.copy()
    if not boundaries.available():
        table["Division"] = table["Range"] = table["Beat"] = ""
        return table
    for col, level in (("Division", boundaries.DIVISION), ("Range", boundaries.RANGE),
                       ("Beat", boundaries.BEAT)):
        names, _ = boundaries.nearest(level, table["Longitude"], table["Latitude"])
        table[col] = np.where(names == "", "Unknown", names)
    return table


def _confidence(row, land_cover: bool) -> str:
    """Confirmed needs every test passed; failing the cover test caps at Possible."""
    cover = row.get("Cover Share %", np.nan)
    assessed = cover == cover
    if assessed and cover < COVER_FAIL:
        return POSSIBLE
    if (row["Days Used"] >= CONFIRM_DAYS and row["Stability %"] >= 67
            and row["Day Share %"] >= 50 and land_cover and assessed and cover >= COVER_PASS):
        return CONFIRMED
    if row["Days Used"] >= MIN_DAYS and row["Stability %"] >= 34:
        return PROBABLE
    return POSSIBLE


def _flags(row) -> str:
    flags = []
    cover = row.get("Cover Share %", np.nan)
    if cover == cover and cover < COVER_FAIL:
        flags.append("Little cover at the reported position")
    if row["Day Share %"] < 50:
        flags.append("Also used at night")
    near = row.get("To Nearest Village (km)", np.nan)
    if near == near and near < 0.5:
        flags.append("Within 500 m of a village")
    patch = row.get("Tree Patch (ha)", np.nan)
    if patch == patch and 0 < patch < 25 and not row.get("Patch Extends Beyond Window", False):
        flags.append("Small isolated patch")
    if row["Days Since Last Use"] > 60:
        flags.append("Not used in the last 60 days")
    return "; ".join(flags)


def _actions(row) -> str:
    acts = []
    villages = row.get("Villages Raided") or row.get("Nearest Village") or ""
    if row["Linked Raids"]:
        edge = (f"on the {row['Raids Head']} edge" if row["Raids Head"] in COMPASS
                else "around the refuge (raids go several ways)")
        acts.append(f"Dusk watch {edge}; warn {villages or 'nearby villages'} before dark on "
                    "days it is occupied")
    if row["Group"] == "Family herd":
        acts.append("Breeding herd lies up here: keep people out by day, do not drive it")
    elif row["Bull-Type Share %"] >= 50:
        acts.append("Bull refuge: track by day to anticipate the night's raid")
    near = row.get("To Nearest Village (km)", np.nan)
    if near == near and near < 1.0:
        acts.append(f"Warn {row['Nearest Village']}: fuelwood and grazing trips meet "
                    "elephants at rest here")
    cover = row.get("Cover Share %", np.nan)
    if cover == cover and cover < COVER_FAIL:
        acts.append("Verify on the ground: little cover at the reported position "
                    "(tall crop, plantation, or a position error)")
    if not acts:
        acts.append("Keep under daytime watch; confirm with a site visit")
    return "; ".join(acts)


# ---------------------------------------------------------------------------
# Outputs for the field
# ---------------------------------------------------------------------------
def to_geojson(refuges: pd.DataFrame, links: pd.DataFrame) -> str:
    """Refuges as points and linked raids as lines, for QGIS or a phone map."""
    import json

    features = []
    for row in refuges.to_dict("records"):
        props = {k: (v.isoformat() if isinstance(v, pd.Timestamp) else
                     (None if isinstance(v, float) and v != v else
                      (v.item() if hasattr(v, "item") else v)))
                 for k, v in row.items() if k not in ("Latitude", "Longitude")}
        features.append({"type": "Feature", "properties": {"kind": "refuge", **props},
                         "geometry": {"type": "Point",
                                      "coordinates": [row["Longitude"], row["Latitude"]]}})
    by_id = refuges.set_index("Refuge")
    for row in links.to_dict("records"):
        ref = by_id.loc[row["Refuge"]]
        features.append({
            "type": "Feature",
            "properties": {"kind": "linked raid", "refuge": row["Refuge"],
                           "date": row["Date"].isoformat(), "hour": row["Hour"],
                           "category": row["Category"], "distance_km": row["Distance (km)"]},
            "geometry": {"type": "LineString",
                         "coordinates": [[ref["Longitude"], ref["Latitude"]],
                                         [row["Longitude"], row["Latitude"]]]},
        })
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=1)


def to_kml(refuges: pd.DataFrame) -> str:
    """Refuges as placemarks, for Google Earth or a field GPS."""
    from html import escape

    colours = {CONFIRMED: "ff1e1e8c", PROBABLE: "ff0c57c2", POSSIBLE: "ff27a2c9"}
    styles = "".join(
        f'<Style id="{c}"><IconStyle><color>{colours[c]}</color><scale>1.2</scale>'
        "<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>"
        "</Icon></IconStyle></Style>" for c in CONFIDENCE_ORDER)
    marks = []
    for r in refuges.to_dict("records"):
        desc = (f"{r['Confidence']} day refuge. Used on {r['Days Used']} day(s), last "
                f"{r['Last Used']:%d %b %Y}. {r['Linked Raids']} linked night raid(s). "
                f"Beat: {r.get('Beat', '')}. {r['Recommended Action']}")
        marks.append(
            f"<Placemark><name>{escape(r['Refuge'])}</name>"
            f"<description>{escape(desc)}</description>"
            f"<styleUrl>#{r['Confidence']}</styleUrl>"
            f"<Point><coordinates>{r['Longitude']},{r['Latitude']},0</coordinates></Point>"
            "</Placemark>")
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            "<name>Elephant day refuges</name>" + styles + "".join(marks)
            + "</Document></kml>")


def headlines(result: Dict[str, object]) -> List[str]:
    s, table = result["summary"], result["refuges"]
    lines = [f"{s['day_sightings']:,} of {s['timed']:,} timed reports fall in the daytime "
             "rest window."]
    if not s["refuges"]:
        return lines
    lines.append(f"{s['refuges']} day refuge(s): {s['confirmed']} confirmed, "
                 f"{s['probable']} probable, the rest possible.")
    if s["night_raids"]:
        lines.append(f"{s['linked_raids']} of {s['night_raids']} night conflict reports link to "
                     "a refuge in use the day before or after.")
    top = table.sort_values("Linked Raids", ascending=False).iloc[0]
    if top["Linked Raids"]:
        lines.append(
            f"{top['Refuge']} ({top.get('Beat', '')}) feeds the most raids: "
            f"{top['Linked Raids']}, heading {top['Raids Head']} a median "
            f"{top['Raid Distance (km, median)']} km, into {top['Villages Raided'] or 'nearby fields'}.")
    if "Tree Patch (ha)" in table:
        small = table[(table["Tree Patch (ha)"] > 0) & (table["Tree Patch (ha)"] < 25)
                      & ~table["Patch Extends Beyond Window"].fillna(False).astype(bool)]
        if len(small):
            lines.append(f"{len(small)} refuge(s) are isolated tree patches under 25 ha. "
                         "Elephants resting in a small patch are close to the people who "
                         "use it by day.")
    if s["villages_to_warn"]:
        lines.append(f"{s['villages_to_warn']} village(s) lie within a night's reach of a "
                     "refuge; the table lists them per refuge.")
    return lines
