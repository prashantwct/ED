"""Daytime refuges: where elephants lie up by day, and what they raid by night.

In this landscape elephants feed in fields at night and spend the day in
cover -- a forest patch, a scrubby nala, a plantation -- usually within a
night's walk of the fields they raid. A refuge used again and again is a
fixed point in an otherwise moving problem: a place to watch at dusk,
villages to warn before dark, people to keep out of it by day.

A refuge here is not a cluster of sightings. It has to pass five tests,
each of which removes a known way of being wrong:

1. **Rest hours only.** Direct sightings between ``DAY_START`` and ``DAY_END``
   (09:00-16:00 by default). Dawn and dusk sightings are left out: they
   catch animals moving between field and cover, and would put "refuges"
   on the paths in between. Only a direct sighting counts: pug marks,
   dung, broken branches, sounds and damage are stale traces of an
   earlier visit, never sightings, whatever their hour (see below).
2. **Recurrence.** Day sightings on at least ``MIN_DAYS`` separate days.
   One afternoon of reports, however many, is a herd passing through.
3. **Day, not night, use.** Of all use of the place -- timed direct
   sightings, plus the damage and indirect signs found there, which count
   as night use -- at least ``MIN_DAY_SHARE`` per cent by day. A place seen mostly at night, or raided, is a feeding ground.
4. **Stability.** The clustering is re-run over a grid of neighbour
   distances and rest windows; a refuge must reappear within 1 km in
   most of them. One that only exists at one setting is an artefact of
   that setting.
5. **Cover.** With Esri land cover, the share of tree, rangeland and
   flooded-vegetation pixels within 250 m, and the size of the tree patch
   the refuge sits in. Day use with no cover is flagged, not hidden: it
   is either a position error or animals lying up in tall crop, and both
   matter.

**Damage is entered after the raid.** Crop, grain and house damage is
found and recorded the next morning, so the entry's hour is when it was
written down, not when the elephants were there. A damage report entered
on day D before ``NIGHT_START`` (or with no hour) is a raid on the night
that began on D - 1; one entered at or after ``NIGHT_START`` is that same
night. Such reports are taken out of the day sightings -- a 10 am damage
entry in a field is not elephants resting in it -- and count as night use
of the place. Injuries and deaths are reported as they happen, so they
keep their hour and count as raids only at night -- unless reported
indirectly, when they are stale like damage. The same holds for any
report that is not a direct sighting: its hour is when it was written
down, so it is dated to the night before.

Each refuge is then linked to the raids it feeds: raids within
``LINK_KM`` of the refuge on a night within ``LINK_DAYS`` of a day it was
seen in use. Herds stay in one block of cover for days at a stretch but
are not reported there every day, so the night-before-or-after rule alone
links only the raids that happen to fall on a reported day. Each link
records the gap: "Same night" when the refuge was reported the day before
or after, otherwise the number of days. A raid near two refuges goes to
one in use the same night if there is one, else to the nearer one, so
nothing is counted twice.

**Raids with no known refuge.** Every raid is accounted for. Raids that
link to no refuge are grouped into raid areas (``AREA_EPS_KM``,
``AREA_MIN_RAIDS`` or more); each is a place where elephants raid but no
daytime refuge is on record within reach -- a refuge nobody has reported.
With land cover, the nearest tree patch of ``SEARCH_MIN_HA`` or more
within reach is named as the place to search by day.

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
LINK_DAYS = 7                         # days between refuge use and the raid night
AREA_EPS_KM = 1.5                     # raids this close form one raid area
AREA_MIN_RAIDS = 3
SEARCH_MIN_HA = 20.0                  # tree patch big enough to hide a herd by day
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


def raids_and_nights(df: pd.DataFrame) -> pd.DataFrame:
    """Conflict reports that are night raids, each with the night it belongs to.

    ``Night Of`` is the date the raid's night began. Property damage is
    taken to precede its entry (see the module notes), as is anything not
    a direct sighting; a directly sighted casualty keeps its reported hour
    and is a raid only at night.
    """
    if df.empty:
        return df.assign(**{"Night Of": pd.Series(dtype="datetime64[ns]")})
    from core.analytics import property_damage_mask, stale_mask

    hour = pd.to_numeric(df["Hour"], errors="coerce") if "Hour" in df else \
        pd.Series(np.nan, index=df.index)
    damage = property_damage_mask(df)
    stale = stale_mask(df)
    casualty = (human_injuries(df) > 0) | (human_deaths(df) > 0)
    # A casualty seen directly keeps its hour; one reported indirectly is
    # stale, like damage, and dated to the night before its entry.
    raid = damage | (casualty & (stale | is_night(hour)))
    date = df["Date"].dt.normalize()
    # Entered the same night (after dusk) -> that night; otherwise the night before.
    same_night = hour.notna() & (hour >= NIGHT_START)
    night_of = date.where(same_night, date - pd.Timedelta(days=1))
    return df[raid].assign(**{"Night Of": night_of[raid]})


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
    link_days: int = LINK_DAYS,
    require_raids: bool = True,
    as_of: Optional[pd.Timestamp] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Find day refuges in the register and link each to its night raids.

    Returns ``refuges`` (one row per refuge, most-used first), ``links``
    (one row per linked raid), ``windows`` (land-cover windows by refuge
    id, for thumbnails), ``summary`` and ``caveats``.

    With ``require_raids`` (the default) a refuge no night raid links
    to is left out: a resting place that sends no raids is not a
    management priority, and is counted in the caveats instead.
    """
    from core.analytics import indirect_mask

    located = df[df["Date"].notna() & df["Latitude"].notna() & df["Longitude"].notna()]
    conflict = conflict_mask(located)
    indirect = indirect_mask(located)
    # Only a direct sighting says where the animals were at that hour.
    # Damage, pug marks, dung, broken branches and sounds are stale: they
    # say where the animals were the night before, so they never count as
    # day sightings -- they count as night use of the place instead.
    timed = located[~conflict & ~indirect & located["Hour"].notna()].copy()
    timed["Hour"] = timed["Hour"].astype(int)
    traces = located[~conflict & indirect]
    caveats = []
    if len(traces):
        caveats.append(f"{len(traces):,} indirect sign(s) (pug marks, dung, broken branches, "
                       "sounds) are stale -- traces of the night before, not elephants present "
                       "at the hour entered -- and count as night use of a place, never as day "
                       "rest.")
    untimed = int((~conflict & ~indirect & located["Hour"].isna()).sum())
    if untimed:
        caveats.append(f"{untimed:,} sighting(s) without an hour cannot be placed in day or "
                       "night and are left out.")
    raids = raids_and_nights(located).copy()
    if len(raids):
        raids["_deaths"] = human_deaths(raids)
        raids["_injuries"] = human_injuries(raids)
        if "Nearest Village" not in raids and villages is not None and len(villages):
            from core.spatial import attach_nearest_village

            raids, _ = attach_nearest_village(raids, villages)
    ctx = {"raids": raids, "villages": villages, "land_cover": land_cover,
           "link_km": link_km}

    day_n = int(is_day(timed["Hour"], *day_window).sum())
    if not len(timed) or day_n == 0:
        return _empty(caveats + ["No reports fall in the daytime rest window, so no "
                                 "refuge can be identified."], ctx=ctx)

    as_of = pd.Timestamp(as_of) if as_of is not None else timed["Date"].max()
    base = _candidates(timed, eps_km, day_window, MIN_SIGHTINGS, min_days)

    # Stability: how many settings in the grid find a refuge here too.
    grid = [(e, w) for e in STABILITY_EPS for w in STABILITY_WINDOWS]
    centres_by_setting = []
    for e, w in grid:
        found = _candidates(timed, e, w, MIN_SIGHTINGS, min_days)
        centres_by_setting.append(np.array([_centre(m) for m in found]) if found
                                  else np.empty((0, 2)))

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
        raided_here = int((_km(lat, lon, raids["Latitude"].to_numpy(float),
                               raids["Longitude"].to_numpy(float)) <= radius + 0.25).sum()) \
            if len(raids) else 0
        traced_here = int((_km(lat, lon, traces["Latitude"].to_numpy(float),
                               traces["Longitude"].to_numpy(float)) <= radius + 0.25).sum()) \
            if len(traces) else 0
        local_night = int(is_night(local["Hour"]).sum()) + raided_here + traced_here
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
        caveats.append(f"{len(feeding)} place(s) with repeated day sightings were mostly seen "
                       "at night or raided -- feeding grounds, not refuges -- and are left "
                       "out.")
    if not rows:
        return _empty(caveats + [f"No place had daytime sightings on {min_days} or more "
                                 "separate days and mostly daytime use."], len(timed), day_n,
                      ctx=ctx)

    table = pd.DataFrame(rows)
    known = table[["Latitude", "Longitude"]].copy()   # every day-use place, raids or not
    links = _link_raids(table, use_days, raids, link_km, link_days)
    table = _attach_links(table, links)
    if require_raids:
        quiet = int((table["Linked Raids"] == 0).sum())
        table = table[table["Linked Raids"] > 0].reset_index(drop=True)
        if quiet:
            caveats.append(f"{quiet} place(s) used by day but with no night raid linked "
                           "to them are left out.")
        if table.empty:
            return _empty(caveats + ["No day refuge had a night raid linked to it."],
                          len(timed), day_n, ctx=ctx, known=known)
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
    linked_idx = set(links["_raid"])
    links = links.drop(columns=["_i", "_raid"])
    areas, scattered = raid_areas(raids[~raids.index.isin(linked_idx)], known, villages,
                                  land_cover, link_km)

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
        f"Raids are linked by place ({link_km:g} km) and time (a night within {link_days} "
        "day(s) of a reported day in use), not identity: consistent with coming from the "
        "refuge, not proof. Same-night links are the strongest.",
        "Damage is taken to have happened the night before its entry (the night itself if "
        f"entered after {NIGHT_START}:00), since it is found and recorded the next morning.",
    ]
    summary = {
        "timed": len(timed), "day_sightings": day_n, "refuges": len(table),
        "confirmed": int((table["Confidence"] == CONFIRMED).sum()),
        "probable": int((table["Confidence"] == PROBABLE).sum()),
        "linked_raids": int(len(links)),
        "same_night": int((links["Link"] == SAME_NIGHT).sum()) if len(links) else 0,
        "night_raids": int(len(raids)),
        "villages_to_warn": int(len(set(", ".join(table["Villages Within Reach"].dropna())
                                       .split(", ")) - {""})),
        "areas": int(len(areas)),
        "area_raids": int(areas["Raids"].sum()) if len(areas) else 0,
        "scattered_raids": int(scattered),
    }
    return {"refuges": table, "links": links, "windows": windows, "areas": areas,
            "summary": summary, "caveats": caveats}


def _empty(caveats, timed=0, day=0, ctx=None, known=None) -> Dict[str, object]:
    ctx = ctx or {}
    raids = ctx.get("raids")
    raids = raids if raids is not None else pd.DataFrame()
    areas, scattered = raid_areas(raids, known, ctx.get("villages"), ctx.get("land_cover"),
                                  ctx.get("link_km", LINK_KM))
    return {"refuges": pd.DataFrame(), "links": pd.DataFrame(), "windows": {}, "areas": areas,
            "summary": {"timed": timed, "day_sightings": day, "refuges": 0, "confirmed": 0,
                        "probable": 0, "linked_raids": 0, "same_night": 0,
                        "night_raids": int(len(raids)), "villages_to_warn": 0,
                        "areas": int(len(areas)),
                        "area_raids": int(areas["Raids"].sum()) if len(areas) else 0,
                        "scattered_raids": int(scattered)},
            "caveats": caveats}


# ---------------------------------------------------------------------------
# Raids with no known refuge
# ---------------------------------------------------------------------------
AREA_COLUMNS = ["Area", "Latitude", "Longitude", "Raids", "Nights", "First Raid", "Last Raid",
                "Months", "House Damage", "Casualties", "Villages Raided",
                "To Nearest Known Refuge (km)", "Division", "Range", "Beat"]


def raid_areas(raids: pd.DataFrame, known: Optional[pd.DataFrame], villages, land_cover,
               link_km: float = LINK_KM) -> Tuple[pd.DataFrame, int]:
    """Raids no refuge accounts for, grouped into areas; and the count left scattered.

    ``known`` holds every place with repeated day use (Latitude, Longitude),
    linked or not, for the distance to the nearest known refuge.
    """
    if raids is None or len(raids) < AREA_MIN_RAIDS:
        return pd.DataFrame(columns=AREA_COLUMNS), int(0 if raids is None else len(raids))
    from core.analytics import classify_conflict

    lat = raids["Latitude"].to_numpy(float)
    lon = raids["Longitude"].to_numpy(float)
    labels = _dbscan(_to_km_plane(lat, lon), eps=AREA_EPS_KM, min_samples=AREA_MIN_RAIDS)
    cats = classify_conflict(raids)
    rows = []
    for label in sorted(set(labels) - {-1}):
        part = raids[labels == label]
        clat, clon = float(part["Latitude"].median()), float(part["Longitude"].median())
        nights = part["Night Of"]
        names = part.get("Nearest Village", pd.Series(dtype=object)).replace("", np.nan).dropna()
        row = {
            "Latitude": round(clat, 5), "Longitude": round(clon, 5),
            "Raids": int(len(part)), "Nights": int(nights.nunique()),
            "First Raid": nights.min(), "Last Raid": nights.max(),
            "Months": ", ".join(pd.Timestamp(2000, m, 1).strftime("%b")
                                for m in sorted(set(nights.dt.month))),
            "House Damage": int((cats.loc[part.index] == "House").sum()),
            "Casualties": int((part["_injuries"] + part["_deaths"]).sum())
            if "_injuries" in part else 0,
            "Villages Raided": ", ".join(names.value_counts().index[:4]),
            "To Nearest Known Refuge (km)": (
                round(float(_km(clat, clon, known["Latitude"].to_numpy(float),
                                known["Longitude"].to_numpy(float)).min()), 1)
                if known is not None and len(known) else np.nan),
        }
        if land_cover is not None:
            window = land_cover.window(clat, clon, link_km * 1000.0)
            if window is not None:
                row.update(search_target(window))
        rows.append(row)
    scattered = int((labels == -1).sum())
    if not rows:
        return pd.DataFrame(columns=AREA_COLUMNS), scattered
    table = pd.DataFrame(rows).sort_values(["Raids", "Nights"], ascending=False)
    table = _attach_admin(table.reset_index(drop=True))
    table.insert(0, "Area", [f"A{k + 1:02d}" for k in range(len(table))])
    table["Recommended Action"] = table.apply(_area_action, axis=1, link_km=link_km)
    return table, scattered


def search_target(window) -> Dict[str, object]:
    """The nearest tree patch big enough to hold a herd by day, from a raid area."""
    from scipy import ndimage

    from core import esri_landcover as lc

    trees = window.codes == lc.TREES
    out = {"Search Patch (ha)": np.nan, "Search Distance (km)": np.nan,
           "Search Direction": "", "Search Latitude": np.nan, "Search Longitude": np.nan}
    if not trees.any():
        return out
    labels, n = ndimage.label(trees, structure=np.ones((3, 3)))
    sizes = ndimage.sum(trees, labels, index=np.arange(1, n + 1)) * window.pixel_ha
    big = np.isin(labels, np.nonzero(sizes >= SEARCH_MIN_HA)[0] + 1)
    if not big.any():
        return out
    rows, cols = np.nonzero(big)
    d2 = ((rows - window.row) * window.dy_m) ** 2 + ((cols - window.col) * window.dx_m) ** 2
    k = int(np.argmin(d2))
    r, c = rows[k], cols[k]
    plat = window.lat - (r - window.row) * window.dy_m / 111_320.0
    plon = window.lon + (c - window.col) * window.dx_m / (
        111_320.0 * math.cos(math.radians(window.lat)))
    component = labels == labels[r, c]
    out.update({
        "Search Patch Extends Beyond Window": bool(
            component[0, :].any() or component[-1, :].any()
            or component[:, 0].any() or component[:, -1].any()),
        "Search Patch (ha)": round(float(sizes[labels[r, c] - 1]), 0),
        "Search Distance (km)": round(math.sqrt(float(d2[k])) / 1000.0, 1),
        "Search Direction": compass(float(_bearing(window.lat, window.lon, plat, plon))),
        "Search Latitude": round(plat, 5), "Search Longitude": round(plon, 5),
    })
    return out


def _area_action(row, link_km: float = LINK_KM) -> str:
    villages = row.get("Villages Raided") or "nearby villages"
    acts = []
    patch = row.get("Search Patch (ha)", np.nan)
    known = row.get("To Nearest Known Refuge (km)", np.nan)
    if known == known and known <= link_km:
        acts.append(f"A known day-use site {known} km away was not reported on these nights: "
                    "check it by day after each raid, and the cover around it")
    elif patch == patch:
        more = "+" if row.get("Search Patch Extends Beyond Window") else ""
        acts.append(f"No daytime refuge on record within reach: search the {patch:.0f}{more} ha "
                    f"tree patch {row['Search Distance (km)']} km {row['Search Direction']} "
                    "by day")
    else:
        acts.append("No daytime refuge on record within reach: send trackers out by day to "
                    "find where the herd lies up")
    acts.append(f"Dusk watch and early warning for {villages}")
    if row.get("Casualties"):
        acts.append("People have been hurt here: night patrol with the village response team")
    return "; ".join(acts)


SAME_NIGHT = "Same night"


def _use_gap(use: set, night_of: pd.Timestamp) -> int:
    """Days between a raid night and the nearest reported day in use.

    0 when the refuge was reported the day the night began or the day
    after (lying up before the raid, or after it).
    """
    best = None
    for d in use:
        gap = 0 if 0 <= (d - night_of).days <= 1 else min(abs((d - night_of).days),
                                                           abs((d - night_of).days - 1))
        best = gap if best is None else min(best, gap)
    return best if best is not None else 10 ** 6


def _link_raids(table: pd.DataFrame, use_days: Dict[int, set], raids: pd.DataFrame,
                link_km: float, link_days: int = LINK_DAYS) -> pd.DataFrame:
    """Each raid to a refuge in use within ``link_days`` of its night.

    A refuge reported the same night wins; otherwise the nearer one.
    """
    cols = ["_i", "_raid", "Link", "Days From Use", "Date", "Night Of", "Hour", "Latitude", "Longitude", "Distance (km)", "Bearing",
            "Village", "Deaths", "Injuries", "Category"]
    if raids is None or not len(raids):
        return pd.DataFrame(columns=cols)
    from core.analytics import classify_conflict

    cats = classify_conflict(raids)
    out = []
    for idx, raid in raids.iterrows():
        night_of = raid["Night Of"]
        best = None
        for _, ref in table.iterrows():
            dist = float(_km(ref["Latitude"], ref["Longitude"], raid["Latitude"], raid["Longitude"]))
            if dist > link_km:
                continue
            gap = _use_gap(use_days[ref["_i"]], night_of)
            if gap > link_days:
                continue
            key = (gap > 0, dist)
            if best is None or key < best[0]:
                best = (key, ref, dist, gap)
        if best is None:
            continue
        _, ref, dist, gap = best
        out.append({
            "_i": ref["_i"], "_raid": idx,
            "Link": SAME_NIGHT if gap == 0 else f"{gap} day(s) apart",
            "Days From Use": int(gap), "Date": raid["Date"], "Night Of": night_of,
            "Hour": int(raid["Hour"]) if pd.notna(raid["Hour"]) else None,
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
    table["Same-Night Raids"] = [
        int((links[links["_i"] == i]["Link"] == SAME_NIGHT).sum()) if len(links) else 0
        for i in table["_i"]]
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
def _props(row: Dict[str, object], skip=("Latitude", "Longitude")) -> Dict[str, object]:
    return {k: (v.isoformat() if isinstance(v, pd.Timestamp) else
                (None if isinstance(v, float) and v != v else
                 (v.item() if hasattr(v, "item") else v)))
            for k, v in row.items() if k not in skip}


def to_geojson(refuges: pd.DataFrame, links: pd.DataFrame,
               areas: Optional[pd.DataFrame] = None) -> str:
    """Refuges and raid areas as points, linked raids as lines, for QGIS or a phone map."""
    import json

    features = []
    for row in ([] if areas is None else areas.to_dict("records")):
        features.append({"type": "Feature", "properties": {"kind": "raid area", **_props(row)},
                         "geometry": {"type": "Point",
                                      "coordinates": [row["Longitude"], row["Latitude"]]}})
    for row in refuges.to_dict("records"):
        features.append({"type": "Feature", "properties": {"kind": "refuge", **_props(row)},
                         "geometry": {"type": "Point",
                                      "coordinates": [row["Longitude"], row["Latitude"]]}})
    by_id = refuges.set_index("Refuge")
    for row in links.to_dict("records"):
        ref = by_id.loc[row["Refuge"]]
        features.append({
            "type": "Feature",
            "properties": {"kind": "linked raid", "refuge": row["Refuge"],
                           "date": row["Date"].isoformat(),
                           "night_of": row["Night Of"].isoformat(), "entry_hour": row["Hour"],
                           "category": row["Category"], "distance_km": row["Distance (km)"]},
            "geometry": {"type": "LineString",
                         "coordinates": [[ref["Longitude"], ref["Latitude"]],
                                         [row["Longitude"], row["Latitude"]]]},
        })
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=1)


def to_kml(refuges: pd.DataFrame, areas: Optional[pd.DataFrame] = None) -> str:
    """Refuges, and the tree patches to search near raid areas, as placemarks."""
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
    for r in ([] if areas is None else areas.to_dict("records")):
        lat, lon = r.get("Search Latitude", np.nan), r.get("Search Longitude", np.nan)
        target = "search" if lat == lat else "raids"
        if lat != lat:
            lat, lon = r["Latitude"], r["Longitude"]
        desc = (f"Raid area with no known day refuge: {r['Raids']} raid(s) on {r['Nights']} "
                f"night(s), last {r['Last Raid']:%d %b %Y}. Beat: {r.get('Beat', '')}. "
                f"{r['Recommended Action']}")
        marks.append(
            f"<Placemark><name>{escape(r['Area'])} ({target})</name>"
            f"<description>{escape(desc)}</description><styleUrl>#search</styleUrl>"
            f"<Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>")
    styles += ('<Style id="search"><IconStyle><color>ff6b6b5b</color><scale>1.1</scale>'
               "<Icon><href>http://maps.google.com/mapfiles/kml/shapes/target.png</href>"
               "</Icon></IconStyle></Style>")
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            "<name>Elephant day refuges</name>" + styles + "".join(marks)
            + "</Document></kml>")


def headlines(result: Dict[str, object]) -> List[str]:
    s, table = result["summary"], result["refuges"]
    lines = []
    if s["night_raids"]:
        lines.append(f"Of {s['night_raids']:,} night raids (damage and indirect reports dated "
                     f"to the night before their entry): {s['linked_raids']:,} link to a known day refuge "
                     f"({s['same_night']:,} the same night), {s['area_raids']:,} fall in "
                     f"{s['areas']} raid area(s) with no daytime refuge on record, and "
                     f"{s['scattered_raids']:,} are scattered.")
    areas = result.get("areas")
    if areas is not None and len(areas):
        top = areas.iloc[0]
        where = top["Villages Raided"] or f"{top['Beat']} beat"
        lines.append(f"The biggest raid area with no known refuge, {top['Area']} around "
                     f"{where}, saw {top['Raids']} raids on {top['Nights']} nights: an "
                     "unreported refuge lies within a night's walk of it.")
    lines.append(f"{s['day_sightings']:,} of {s['timed']:,} timed direct sightings fall in "
                 "the daytime rest window; indirect signs are stale and never count as day "
                 "rest.")
    if not s["refuges"]:
        return lines
    lines.append(f"{s['refuges']} day refuge(s) send raids: {s['confirmed']} confirmed, "
                 f"{s['probable']} probable, the rest possible.")
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
