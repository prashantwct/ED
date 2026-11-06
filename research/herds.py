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

from core.movement import (  # noqa: F401 - re-exported for the research scripts
    COUNT_TOLERANCE,
    COUNT_TOLERANCE_FRACTION,
    MAX_GAP_DAYS,
    MAX_SPEED_KM_PER_DAY,
    SAME_DAY_RADIUS_KM,
    SOLITARY_MAX,
    _class_of,
    _compatible,
    _reach_km,
    assign_units,
)
from research.features import conflict_target, km_plane

# The tracker itself moved to core.movement when the app began forecasting
# from it; this module keeps the summaries and the sensitivity table.


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
