"""Group sightings into plausible movement units.

The export has no individual identification -- no collars, no ear-notch
photographs, no DNA. What it has is a time, a place and a rough
composition. Chaining those into tracks the way a radar tracks aircraft
gives *plausible movement units*: sequences of reports consistent with
one animal or one group moving through the landscape.

That is an inference, not an identification. Two lone bulls working the
same valley in the same week are indistinguishable here, and a unit
breaks in half whenever reporting stops for longer than the gap
tolerance. Everything downstream carries that caveat.

The linking rule is deliberately conservative on two axes:

* Speed. Asian elephants in central India move a few kilometres a day
  and can cover twenty when pushed. Anything faster is two groups.
* Composition. A party of bulls does not become a breeding herd with
  calves overnight. This constraint is what stops single-linkage from
  chaining the whole landscape into one blob.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.features import conflict_target, km_plane

# Sustained daily movement. Twenty is a hard day for a herd; beyond it
# the two reports are far more likely to be different animals.
MAX_SPEED_KM_PER_DAY = 15.0
# Two observers reporting the same herd hours apart, from opposite sides
# of it, still need room to be the same herd.
SAME_DAY_RADIUS_KM = 12.0
# Reporting is patchy. Past this, a track is closed rather than guessed
# across the gap.
MAX_GAP_DAYS = 4
# Counts are eyeball estimates from the field. A herd of nine reported
# as eight is the same herd.
COUNT_TOLERANCE = 2
COUNT_TOLERANCE_FRACTION = 0.4

SOLITARY_MAX = 3


def _class_of(total: float, calves: float, males: float, females: float) -> str:
    """Coarse social class. The one distinction the data can support."""
    if calves > 0 or females > 0:
        return "family herd"
    if total <= 1:
        return "lone bull"
    if total <= SOLITARY_MAX:
        return "bull party"
    return "unsexed group"


def _compatible(a: Dict, b: Dict) -> bool:
    """Could these two reports be the same animals?"""
    if a["social"] != b["social"]:
        return False
    # A herd does not shed or acquire calves between sightings.
    if (a["calves"] > 0) != (b["calves"] > 0):
        return False
    tolerance = max(COUNT_TOLERANCE,
                    COUNT_TOLERANCE_FRACTION * (a["total"] + b["total"]) / 2)
    return abs(a["total"] - b["total"]) <= tolerance


def _reach_km(gap_days: float, max_speed_km_per_day: float) -> float:
    """How far a unit could plausibly have moved in the elapsed time.

    Same-day reports get a fixed radius rather than zero, because two
    observers can log one herd hours apart from opposite sides of it.
    That radius applies only at zero gap: using it as a floor at every
    gap would mask the speed setting below 12 km/day and make the
    sensitivity table understate how much the thresholds matter.
    """
    if gap_days <= 0:
        return SAME_DAY_RADIUS_KM
    return max_speed_km_per_day * gap_days


def assign_units(
    df: pd.DataFrame,
    max_speed_km_per_day: float = MAX_SPEED_KM_PER_DAY,
    max_gap_days: int = MAX_GAP_DAYS,
) -> pd.Series:
    """Greedy nearest-neighbour tracking over time-ordered sightings.

    Each report either extends the best-matching open track or starts a
    new one. Assigning to a single best candidate rather than every
    candidate within reach is what keeps this from degenerating into one
    connected component covering the whole landscape.
    """
    frame = df.sort_values("Date").reset_index()
    ref_lat = float(frame["Latitude"].mean())
    points = km_plane(frame["Latitude"].values, frame["Longitude"].values, ref_lat)
    day = frame["Date"].values.astype("datetime64[D]").astype(int)

    counts = frame.reindex(
        columns=["Male Count", "Female Count", "Calf Count", "Total Count"]
    ).fillna(0)

    records = []
    for i in range(len(frame)):
        total = float(counts["Total Count"].iloc[i])
        records.append({
            "total": total,
            "calves": float(counts["Calf Count"].iloc[i]),
            "males": float(counts["Male Count"].iloc[i]),
            "females": float(counts["Female Count"].iloc[i]),
            "social": _class_of(total,
                                float(counts["Calf Count"].iloc[i]),
                                float(counts["Male Count"].iloc[i]),
                                float(counts["Female Count"].iloc[i])),
        })

    unit_of = np.full(len(frame), -1, dtype=int)
    # Open tracks, each holding the index of its most recent sighting.
    open_tracks: List[int] = []
    next_unit = 0

    for i in range(len(frame)):
        open_tracks = [t for t in open_tracks if day[i] - day[t] <= max_gap_days]

        best, best_cost = None, np.inf
        for t in open_tracks:
            if not _compatible(records[i], records[t]):
                continue
            gap = day[i] - day[t]
            reach = _reach_km(gap, max_speed_km_per_day)
            distance = float(np.hypot(*(points[i] - points[t])))
            if distance > reach:
                continue
            # Prefer close in space, then recent in time.
            cost = distance / reach + 0.1 * gap
            if cost < best_cost:
                best, best_cost = t, cost

        if best is None:
            unit_of[i] = next_unit
            next_unit += 1
        else:
            unit_of[i] = unit_of[best]
            open_tracks.remove(best)
        open_tracks.append(i)

    return pd.Series(unit_of, index=frame["index"]).reindex(df.index)


def _hull_area_km2(xy: np.ndarray) -> float:
    """Minimum convex polygon area, the standard home-range summary."""
    if len(xy) < 3:
        return 0.0
    from scipy.spatial import ConvexHull, QhullError
    try:
        return float(ConvexHull(xy).volume)   # 2-D: 'volume' is area
    except (QhullError, ValueError):
        return 0.0


def _hull_polygon(xy: np.ndarray, lats: np.ndarray, lons: np.ndarray):
    """Hull vertices as (lat, lon), for drawing the demarcation."""
    if len(xy) < 3:
        return list(zip(lats.tolist(), lons.tolist()))
    from scipy.spatial import ConvexHull, QhullError
    try:
        order = ConvexHull(xy).vertices
    except (QhullError, ValueError):
        return list(zip(lats.tolist(), lons.tolist()))
    return [(float(lats[i]), float(lons[i])) for i in order]


def summarise_units(df: pd.DataFrame, units: Optional[pd.Series] = None
                    ) -> pd.DataFrame:
    """One row per movement unit, with its range and its conflict record."""
    if units is None:
        units = assign_units(df)
    work = df.copy()
    work["Unit"] = units.values
    work["conflict"] = conflict_target(work)
    ref_lat = float(work["Latitude"].mean())

    deaths = work.reindex(
        columns=["Male Death Count", "Female Death Count", "Children Death Count"]
    ).fillna(0).sum(axis=1)
    work["deaths"] = np.where(work.get("Death", 0).fillna(0) > 0,
                              np.maximum(deaths, 1), 0)

    rows = []
    for unit, group in work.groupby("Unit"):
        group = group.sort_values("Date")
        xy = km_plane(group["Latitude"].values, group["Longitude"].values, ref_lat)
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.array([0.0])
        span_days = (group["Date"].max() - group["Date"].min()).days

        counts = group.reindex(columns=["Total Count", "Calf Count", "Male Count"]).fillna(0)
        rows.append({
            "Unit": f"U{unit:03d}",
            "Class": group.pipe(lambda g: _class_of(
                float(counts["Total Count"].median()),
                float(counts["Calf Count"].max()),
                float(counts["Male Count"].median()),
                float(group.get("Female Count", pd.Series(0)).fillna(0).max()),
            )),
            "Sightings": len(group),
            "Days Observed": group["Date"].dt.date.nunique(),
            "First Seen": group["Date"].min(),
            "Last Seen": group["Date"].max(),
            "Span (days)": span_days,
            "Typical Size": float(counts["Total Count"].median()),
            "Max Size": float(counts["Total Count"].max()),
            "Calves Seen": int(counts["Calf Count"].max()),
            "Path (km)": float(steps.sum()),
            "Max Step (km)": float(steps.max()),
            "Mean Speed (km/day)": float(steps.sum() / span_days) if span_days else np.nan,
            "Range (km2)": _hull_area_km2(xy),
            "Centre Latitude": float(group["Latitude"].mean()),
            "Centre Longitude": float(group["Longitude"].mean()),
            "Conflict Events": int(group["conflict"].sum()),
            "Human Deaths": float(group["deaths"].sum()),
            "Crop Events": int(group.get("Crop Damage", pd.Series(0)).fillna(0).sum()),
            "House Events": int(group.get("House Damage", pd.Series(0)).fillna(0).sum()),
            "Night Share %": float(
                ((group["Hour"] >= 18) | (group["Hour"] < 6)).mean() * 100
            ) if "Hour" in group else np.nan,
            "Divisions": ", ".join(sorted(group["Division"].astype(str).unique())[:3]),
            "Polygon": _hull_polygon(xy, group["Latitude"].values,
                                     group["Longitude"].values),
        })

    table = pd.DataFrame(rows)
    return table.sort_values(["Conflict Events", "Sightings"], ascending=False
                             ).reset_index(drop=True)


def sensitivity(df: pd.DataFrame,
                speeds: Tuple[float, ...] = (10.0, 15.0, 20.0, 30.0),
                gaps: Tuple[int, ...] = (2, 4, 7)) -> pd.DataFrame:
    """How much does the unit count depend on the thresholds?

    If the answer is 'entirely', the units are an artefact of the
    parameters and should not be reported as animals.
    """
    rows = []
    for speed in speeds:
        for gap in gaps:
            units = assign_units(df, max_speed_km_per_day=speed, max_gap_days=gap)
            sizes = units.value_counts()
            rows.append({
                "speed_km_day": speed,
                "gap_days": gap,
                "units": len(sizes),
                "singletons": int((sizes == 1).sum()),
                "largest": int(sizes.max()),
                "median_size": float(sizes.median()),
            })
    return pd.DataFrame(rows)
