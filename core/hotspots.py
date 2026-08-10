"""Movement hotspots and the villages exposed to them.

A beat is an administrative unit; elephants do not move in beats. A herd
working a corridor along a beat boundary reads as moderate pressure in
two beats rather than the one hotspot it is. This clusters the point
data directly, then asks which settlements sit in or beside the result.

Village naming needs centroids (``data/centroids.csv``). Without them
hotspots are still located, sized and tiered.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from core.analytics import (
    classify_conflict,
    conflict_mask,
    human_deaths,
    human_injuries,
)
from core.config import (
    CONFLICT_SHARE_FOR_HIGH,
    CRITICAL_CASUALTY_WINDOW_DAYS,
    DEFAULT_EPS_KM,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_VILLAGE_RADIUS_KM,
    EVENTS_FOR_WATCH,
    KM_PER_DEG_LAT,
    KM_PER_DEG_LON_EQUATOR,
    MAX_SENSIBLE_RADIUS_KM,
)
from core.intelligence import (
    TIER_CRITICAL,
    TIER_HIGH,
    TIER_ORDER,
    TIER_ROUTINE,
    TIER_WATCH,
)

logger = logging.getLogger(__name__)


NOISE_LABEL = -1


def _to_km_plane(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Project lat/lon onto a local plane in kilometres.

    Longitude scaled by cos(latitude): unscaled degrees overstate
    east-west separation by ~8% here, distorting cluster boundaries.
    """
    reference_lat = float(np.nanmean(lat))
    x = lon * KM_PER_DEG_LON_EQUATOR * math.cos(math.radians(reference_lat))
    y = lat * KM_PER_DEG_LAT
    return np.column_stack([x, y])


def _dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Minimal DBSCAN over a KD-tree.

    Hand-rolled to avoid a scikit-learn dependency for one algorithm;
    scipy is already required.

    Returns:
        Integer cluster labels, ``-1`` for points in no cluster.
    """
    n = len(points)
    labels = np.full(n, NOISE_LABEL, dtype=int)
    if n == 0:
        return labels

    tree = cKDTree(points)
    neighbours = tree.query_ball_point(points, r=eps)

    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbours[i]) < min_samples:
            continue  # not a core point; may still be absorbed below

        labels[i] = cluster_id
        queue = list(neighbours[i])
        while queue:
            j = queue.pop()
            if not visited[j]:
                visited[j] = True
                # Only core points extend a cluster; edge points join it
                # but do not pull their own neighbours in.
                if len(neighbours[j]) >= min_samples:
                    queue.extend(neighbours[j])
            if labels[j] == NOISE_LABEL:
                labels[j] = cluster_id
        cluster_id += 1

    return labels


def detect_hotspots(
    df: pd.DataFrame,
    eps_km: float = DEFAULT_EPS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    critical_window_days: int = CRITICAL_CASUALTY_WINDOW_DAYS,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Find density clusters of elephant activity and describe each one.

    Args:
        df: Filtered, enriched dataframe with ``Latitude``/``Longitude``.
        eps_km: Neighbour distance defining cluster density.
        min_samples: Minimum neighbours for a point to anchor a cluster.
        critical_window_days: Casualty recency window for tiering.
        as_of: End of the review period the window is measured from.

    Returns:
        One row per hotspot, worst first, with ``Hotspot``, ``Centre
        Latitude``, ``Centre Longitude``, ``Radius (km)``, ``Sightings``,
        ``Conflict Events``, ``Conflict Share %``, ``Human Deaths``,
        ``Recent Deaths``, ``People Injured``, ``Night Share %``,
        ``Beats``, ``Divisions``, ``First Seen``, ``Last Seen``,
        ``Tier``. Empty input, or data too sparse to cluster, yields an
        empty frame with those columns.
    """
    columns = _hotspot_columns()
    if df.empty or not {"Latitude", "Longitude"}.issubset(df.columns):
        return pd.DataFrame(columns=columns)

    working = df.dropna(subset=["Latitude", "Longitude"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    points = _to_km_plane(
        working["Latitude"].to_numpy(dtype=float),
        working["Longitude"].to_numpy(dtype=float),
    )
    working["_cluster"] = _dbscan(points, eps=eps_km, min_samples=min_samples)
    working["_x"], working["_y"] = points[:, 0], points[:, 1]

    clustered = working[working["_cluster"] != NOISE_LABEL]
    if clustered.empty:
        logger.info("No hotspots found at eps=%.1f km, min_samples=%d.", eps_km, min_samples)
        return pd.DataFrame(columns=columns)

    clustered = clustered.copy()
    clustered["_conflict"] = conflict_mask(clustered)
    clustered["_deaths"] = human_deaths(clustered)
    clustered["_injuries"] = human_injuries(clustered)

    anchor = _resolve_anchor(clustered, as_of)
    if anchor is not None:
        cutoff = anchor - pd.Timedelta(days=critical_window_days)
        clustered["_recent_death"] = clustered["_deaths"].where(
            clustered["Date"] > cutoff, 0.0
        )
    else:
        # No usable dates: count every casualty as recent.
        clustered["_recent_death"] = clustered["_deaths"]

    rows = []
    for cluster_id, group in clustered.groupby("_cluster", observed=True):
        rows.append(_summarise_cluster(cluster_id, group))

    table = pd.DataFrame(rows)
    table["Tier"] = _hotspot_tiers(table)
    table["_rank"] = table["Tier"].map({t: i for i, t in enumerate(TIER_ORDER)})
    table = (
        table.sort_values(
            ["_rank", "Human Deaths", "Conflict Events"], ascending=[True, False, False]
        )
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
    # Name them only after ranking, so Hotspot 1 is the worst one.
    table["Hotspot"] = [f"H{i + 1}" for i in range(len(table))]

    logger.info(
        "Detected %d hotspot(s) covering %d of %d sightings.",
        len(table),
        int(table["Sightings"].sum()),
        len(working),
    )
    return table[columns]


def _hotspot_columns() -> List[str]:
    return [
        "Hotspot",
        "Tier",
        "Sightings",
        "Conflict Events",
        "Conflict Share %",
        "Human Deaths",
        "Recent Deaths",
        "People Injured",
        "Night Share %",
        "Radius (km)",
        "Centre Latitude",
        "Centre Longitude",
        "Beats",
        "Divisions",
        "First Seen",
        "Last Seen",
    ]


def _resolve_anchor(df: pd.DataFrame, as_of: Optional[pd.Timestamp]):
    if "Date" not in df.columns or df["Date"].isna().all():
        return None
    return pd.Timestamp(as_of) if as_of is not None else df["Date"].max()


def _name_list(group: pd.DataFrame, column: str, limit: Optional[int] = None) -> str:
    """Comma-joined unique values, tolerant of nulls and mixed dtypes.

    Arrow-backed string columns keep ``pd.NA`` through ``astype(str)``,
    so a bare ``sorted()`` raises comparing NA to str.
    """
    if column not in group.columns:
        return "N/A"
    names = sorted({str(v) for v in group[column].dropna().tolist()})
    if not names:
        return "N/A"
    if limit is not None and len(names) > limit:
        return ", ".join(names[:limit]) + f", +{len(names) - limit} more"
    return ", ".join(names)


def _summarise_cluster(cluster_id: int, group: pd.DataFrame) -> Dict[str, object]:
    """Describe one cluster: where it is, how big, and what happens in it."""
    centre_lat = float(group["Latitude"].mean())
    centre_lon = float(group["Longitude"].mean())

    # 90th percentile, not the maximum: one outlying report should not
    # inflate the footprint patrol coverage is planned around.
    dx = group["_x"] - group["_x"].mean()
    dy = group["_y"] - group["_y"].mean()
    distances = np.sqrt(dx**2 + dy**2)
    radius = float(np.percentile(distances, 90)) if len(distances) else 0.0

    sightings = int(len(group))
    conflicts = int(group["_conflict"].sum())

    night_share = float("nan")
    if "Is_Night" in group.columns and group["Is_Night"].notna().any():
        night_share = float(group["Is_Night"].dropna().astype(bool).mean() * 100)

    beats = _name_list(group, "Beat", limit=4)
    divisions = _name_list(group, "Division")

    first_seen = last_seen = None
    if "Date" in group.columns and group["Date"].notna().any():
        first_seen = group["Date"].min()
        last_seen = group["Date"].max()

    return {
        "Hotspot": f"H{int(cluster_id) + 1}",
        "Sightings": sightings,
        "Conflict Events": conflicts,
        "Conflict Share %": round(conflicts / sightings * 100, 1) if sightings else 0.0,
        "Human Deaths": float(group["_deaths"].sum()),
        "Recent Deaths": float(group["_recent_death"].sum()),
        "People Injured": float(group["_injuries"].sum()),
        "Night Share %": round(night_share, 1) if not pd.isna(night_share) else float("nan"),
        "Radius (km)": round(radius, 2),
        "Centre Latitude": round(centre_lat, 5),
        "Centre Longitude": round(centre_lon, 5),
        "Beats": beats,
        "Divisions": divisions,
        "First Seen": first_seen,
        "Last Seen": last_seen,
    }


def _hotspot_tiers(table: pd.DataFrame) -> pd.Series:
    """Tier hotspots on the same principle as beats: recent harm first."""
    critical = table["Recent Deaths"] > 0
    high = (table["Human Deaths"] > 0) | (table["People Injured"] > 0) | (
        table["Conflict Share %"] >= CONFLICT_SHARE_FOR_HIGH * 100
    )
    watch = table["Conflict Events"] >= EVENTS_FOR_WATCH

    return pd.Series(
        np.select(
            [critical, high, watch],
            [TIER_CRITICAL, TIER_HIGH, TIER_WATCH],
            default=TIER_ROUTINE,
        ),
        index=table.index,
    )


def villages_at_risk(
    df: pd.DataFrame,
    villages: Optional[pd.DataFrame],
    hotspots: Optional[pd.DataFrame] = None,
    radius_km: float = DEFAULT_VILLAGE_RADIUS_KM,
    critical_window_days: int = CRITICAL_CASUALTY_WINDOW_DAYS,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Rank villages by the conflict recorded around them.

    Unlike :func:`core.intelligence.village_exposure`, which attributes
    each incident to its single nearest village, this counts every
    incident within ``radius_km`` of a village. A settlement between two
    hotspots is exposed to both, and nearest-only attribution hides that.

    Args:
        df: Filtered, enriched dataframe.
        villages: Centroid table with ``Village``, ``Latitude``,
            ``Longitude``. Without it this returns an empty frame -- the
            sighting export contains no village field, so villages cannot
            be named from it alone.
        hotspots: Output of :func:`detect_hotspots`, used to report which
            hotspot each village sits in or beside.
        radius_km: Search radius around each village.
        critical_window_days: Casualty recency window for tiering.
        as_of: End of the review period.

    Returns:
        One row per exposed village, worst first: ``Village``,
        ``Tier``, ``Conflict Events``, ``Human Deaths``,
        ``Recent Deaths``, ``People Injured``, ``House Damage Events``,
        ``Crop Damage Events``, ``Night Share %``, ``Nearest Hotspot``,
        ``Distance to Hotspot (km)``, ``Inside Hotspot``.
    """
    columns = _village_columns()
    if df.empty or villages is None or villages.empty:
        return pd.DataFrame(columns=columns)

    working = df.dropna(subset=["Latitude", "Longitude"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    working["_conflict"] = conflict_mask(working)
    working["_deaths"] = human_deaths(working)
    working["_injuries"] = human_injuries(working)
    working["_category"] = classify_conflict(working)

    anchor = _resolve_anchor(working, as_of)
    if anchor is not None:
        cutoff = anchor - pd.Timedelta(days=critical_window_days)
        working["_recent_death"] = working["_deaths"].where(working["Date"] > cutoff, 0.0)
    else:
        working["_recent_death"] = working["_deaths"]

    incidents = working[working["_conflict"]]
    if incidents.empty:
        return pd.DataFrame(columns=columns)

    # One plane for both sets so the radius query is in real kilometres.
    all_lat = np.concatenate([
        villages["Latitude"].to_numpy(dtype=float),
        incidents["Latitude"].to_numpy(dtype=float),
    ])
    reference_lat = float(np.nanmean(all_lat))
    lon_scale = KM_PER_DEG_LON_EQUATOR * math.cos(math.radians(reference_lat))

    def plane(lat, lon):
        return np.column_stack([lon * lon_scale, lat * KM_PER_DEG_LAT])

    incident_pts = plane(
        incidents["Latitude"].to_numpy(dtype=float),
        incidents["Longitude"].to_numpy(dtype=float),
    )
    village_pts = plane(
        villages["Latitude"].to_numpy(dtype=float),
        villages["Longitude"].to_numpy(dtype=float),
    )

    tree = cKDTree(incident_pts)
    within = tree.query_ball_point(village_pts, r=radius_km)

    rows = []
    for position, indices in enumerate(within):
        if not indices:
            continue
        nearby = incidents.iloc[indices]
        rows.append(_summarise_village(villages.iloc[position], nearby))

    if not rows:
        return pd.DataFrame(columns=columns)

    table = pd.DataFrame(rows)
    table = _attach_hotspot_context(table, villages, hotspots, lon_scale)
    table["Tier"] = _village_tiers(table)
    table["_rank"] = table["Tier"].map({t: i for i, t in enumerate(TIER_ORDER)})
    return (
        table.sort_values(
            ["_rank", "Human Deaths", "Conflict Events"], ascending=[True, False, False]
        )
        .drop(columns="_rank")
        .reset_index(drop=True)[columns]
    )


def _village_columns() -> List[str]:
    return [
        "Village",
        "Tier",
        "Conflict Events",
        "Human Deaths",
        "Recent Deaths",
        "People Injured",
        "House Damage Events",
        "Crop Damage Events",
        "Night Share %",
        "Nearest Hotspot",
        "Distance to Hotspot (km)",
        "Inside Hotspot",
        "Latitude",
        "Longitude",
    ]


def _summarise_village(village: pd.Series, nearby: pd.DataFrame) -> Dict[str, object]:
    night_share = float("nan")
    if "Is_Night" in nearby.columns and nearby["Is_Night"].notna().any():
        night_share = round(
            float(nearby["Is_Night"].dropna().astype(bool).mean() * 100), 1
        )

    return {
        "Village": str(village["Village"]),
        "Latitude": float(village["Latitude"]),
        "Longitude": float(village["Longitude"]),
        "Conflict Events": int(len(nearby)),
        "Human Deaths": float(nearby["_deaths"].sum()),
        "Recent Deaths": float(nearby["_recent_death"].sum()),
        "People Injured": float(nearby["_injuries"].sum()),
        "House Damage Events": int((nearby["_category"] == "House").sum()),
        "Crop Damage Events": int((nearby["_category"] == "Crop").sum()),
        "Night Share %": night_share,
    }


def _attach_hotspot_context(
    table: pd.DataFrame,
    villages: pd.DataFrame,
    hotspots: Optional[pd.DataFrame],
    lon_scale: float,
) -> pd.DataFrame:
    """Record which hotspot each village sits in or nearest to."""
    if hotspots is None or hotspots.empty:
        table["Nearest Hotspot"] = "N/A"
        table["Distance to Hotspot (km)"] = float("nan")
        table["Inside Hotspot"] = False
        return table

    hs_pts = np.column_stack([
        hotspots["Centre Longitude"].to_numpy(dtype=float) * lon_scale,
        hotspots["Centre Latitude"].to_numpy(dtype=float) * KM_PER_DEG_LAT,
    ])
    village_pts = np.column_stack([
        table["Longitude"].to_numpy(dtype=float) * lon_scale,
        table["Latitude"].to_numpy(dtype=float) * KM_PER_DEG_LAT,
    ])

    distances, indices = cKDTree(hs_pts).query(village_pts, k=1)
    table["Nearest Hotspot"] = hotspots.iloc[indices]["Hotspot"].to_numpy()
    table["Distance to Hotspot (km)"] = np.round(distances, 2)
    table["Inside Hotspot"] = (
        distances <= hotspots.iloc[indices]["Radius (km)"].to_numpy()
    )
    return table


def _village_tiers(table: pd.DataFrame) -> pd.Series:
    """Tier villages on recent harm first, consistent with beats."""
    critical = table["Recent Deaths"] > 0
    high = (
        (table["Human Deaths"] > 0)
        | (table["People Injured"] > 0)
        | (table["Inside Hotspot"] & (table["Conflict Events"] >= EVENTS_FOR_WATCH))
    )
    watch = table["Conflict Events"] >= EVENTS_FOR_WATCH

    return pd.Series(
        np.select(
            [critical, high, watch],
            [TIER_CRITICAL, TIER_HIGH, TIER_WATCH],
            default=TIER_ROUTINE,
        ),
        index=table.index,
    )


def hotspot_caveats(
    hotspots: pd.DataFrame, df: pd.DataFrame, villages: Optional[pd.DataFrame]
) -> List[str]:
    """Limits a reader needs before acting on the hotspot table."""
    notes: List[str] = []

    if hotspots.empty:
        notes.append(
            "No density hotspots at these settings. Either sightings are spread "
            "too evenly, or the neighbour distance is too small. Try widening it."
        )
        return notes

    clustered = int(hotspots["Sightings"].sum())
    total = int(len(df))
    if total:
        notes.append(
            f"{clustered:,} of {total:,} sightings ({clustered / total * 100:.0f}%) "
            "fall inside a hotspot. The rest are too dispersed to cluster."
        )

    oversized = hotspots[hotspots["Radius (km)"] > MAX_SENSIBLE_RADIUS_KM]
    if len(oversized):
        names = ", ".join(oversized["Hotspot"].astype(str))
        notes.append(
            f"{len(oversized)} hotspot(s) span over {MAX_SENSIBLE_RADIUS_KM:.0f} km "
            f"({names}) -- chained across an activity belt rather than isolating a "
            "patrol target. Reduce the neighbour distance to split them."
        )

    if villages is None or villages.empty:
        notes.append(
            "No village centroids loaded, so hotspots cannot be named against "
            "settlements."
        )

    return notes


def village_caveats(
    village_risk: pd.DataFrame,
    df: pd.DataFrame,
    radius_km: float = DEFAULT_VILLAGE_RADIUS_KM,
) -> List[str]:
    """Limits a reader needs before acting on the village ranking.

    Chiefly that the casualty columns do not add up: overlapping radii
    credit one fatality to every village near it.
    """
    notes: List[str] = []
    if village_risk.empty:
        return notes

    actual_deaths = float(human_deaths(df).sum())
    counted = float(village_risk["Human Deaths"].sum())
    if counted > actual_deaths:
        notes.append(
            f"Casualty columns do not sum: the {radius_km:g} km radii overlap, so "
            f"one incident counts for every village near it. These rows total "
            f"{counted:.0f} deaths against {actual_deaths:.0f} recorded. A row means "
            "the village is exposed, not that it lost that many people."
        )

    outside = village_risk[~village_risk["Inside Hotspot"].fillna(False).astype(bool)]
    critical_outside = outside[outside["Tier"] == TIER_CRITICAL]
    if len(critical_outside):
        notes.append(
            f"{len(critical_outside)} Critical village(s) sit outside every "
            "hotspot. Hotspot patrols alone will not cover them."
        )

    return notes


def hotspot_membership(
    df: pd.DataFrame,
    eps_km: float = DEFAULT_EPS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> pd.Series:
    """Per-row hotspot label, for colouring the map. ``""`` when unclustered."""
    if df.empty or not {"Latitude", "Longitude"}.issubset(df.columns):
        return pd.Series("", index=df.index, dtype=object)

    usable = df.dropna(subset=["Latitude", "Longitude"])
    labels = pd.Series("", index=df.index, dtype=object)
    if usable.empty:
        return labels

    points = _to_km_plane(
        usable["Latitude"].to_numpy(dtype=float),
        usable["Longitude"].to_numpy(dtype=float),
    )
    clusters = _dbscan(points, eps=eps_km, min_samples=min_samples)
    labels.loc[usable.index] = [
        "" if c == NOISE_LABEL else f"H{c + 1}" for c in clusters
    ]
    return labels
