"""Movement hotspots and the villages exposed to them.

A beat is an administrative unit. Elephants do not move in beats, and a
herd that works a corridor along a beat boundary shows up as moderate
pressure in two beats rather than as the one hotspot it actually is.
This module finds the clusters in the point data directly, then asks
which settlements sit in or beside them.

Two stages:

1. :func:`detect_hotspots` clusters sighting locations by density
   (DBSCAN). Density-based rather than grid-based because a grid imposes
   arbitrary boundaries -- a corridor straddling two cells is split in
   half -- and density clustering also refuses to invent a hotspot where
   the points are simply spread out.
2. :func:`villages_at_risk` ranks villages by the conflict recorded
   around them and their exposure to those hotspots.

Stage 2 needs village centroids, which the sighting export does not
contain. Without them the hotspots are still located, sized and tiered;
they just cannot be named. The functions say so rather than silently
returning nothing.
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
from core.intelligence import (
    CRITICAL_CASUALTY_WINDOW_DAYS,
    TIER_CRITICAL,
    TIER_HIGH,
    TIER_ORDER,
    TIER_ROUTINE,
    TIER_WATCH,
)

logger = logging.getLogger(__name__)

# Kilometres per degree, near enough at these latitudes for clustering.
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON_EQUATOR = 111.32

# Two sightings within this distance are neighbours.
#
# Tuned against a real 1,761-row export rather than picked for
# roundness. DBSCAN chains: if eps reaches the scale at which sightings
# are continuously distributed, clusters merge through the moderate
# density between them and the output stops being hotspots. On that
# export the failure is abrupt -- at eps 2.5 km, 94% of points fell into
# 7 "hotspots", the largest with a 17.7 km radius, which is a region and
# not somewhere a team can be sent. At 1.0 km the radii settle at 1-3 km.
DEFAULT_EPS_KM = 1.0

# Neighbours required before a point can anchor a cluster. Kept high
# relative to eps: this is what stops a chain of moderate density from
# linking two genuine concentrations, and stops a handful of scattered
# reports being promoted into a "hotspot".
DEFAULT_MIN_SAMPLES = 15

# A hotspot wider than this is a sign eps is chaining rather than
# concentrating. Not an error -- some landscapes really do have broad
# activity belts -- but the brief says so instead of presenting it as a
# patrol target.
MAX_SENSIBLE_RADIUS_KM = 5.0

# Radius around a village searched for incidents when ranking exposure.
DEFAULT_VILLAGE_RADIUS_KM = 3.0

# A hotspot needs at least this share of its sightings to be conflict
# before it is treated as a conflict hotspot rather than a movement one.
CONFLICT_SHARE_FOR_HIGH = 0.25
EVENTS_FOR_WATCH = 5

NOISE_LABEL = -1


def _to_km_plane(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Project lat/lon onto a local plane in kilometres.

    Longitude is scaled by cos(latitude) for the same reason as in
    :mod:`core.spatial`: unscaled degrees make east-west separation look
    ~8% larger than it is at this latitude, which distorts which points
    are neighbours and therefore where cluster boundaries fall.
    """
    reference_lat = float(np.nanmean(lat))
    x = lon * KM_PER_DEG_LON_EQUATOR * math.cos(math.radians(reference_lat))
    y = lat * KM_PER_DEG_LAT
    return np.column_stack([x, y])


def _dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Minimal DBSCAN over a KD-tree.

    Implemented here rather than pulled from scikit-learn to avoid adding
    a heavy dependency to a field-deployed app for one algorithm; scipy
    is already required for the KD-tree.

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
        # No usable dates: count every casualty as recent. Over-flagging
        # a hotspot is recoverable; missing a fatal one is not.
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

    Field exports arrive with Arrow-backed string columns that keep
    ``pd.NA`` through ``astype(str)``, which makes a bare ``sorted()``
    raise on comparing NA to str.
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

    # Radius as the 90th percentile of member distances from the centre,
    # not the maximum -- a single outlying report should not inflate the
    # footprint a manager plans patrol coverage around.
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
    ]


def _summarise_village(village: pd.Series, nearby: pd.DataFrame) -> Dict[str, object]:
    night_share = float("nan")
    if "Is_Night" in nearby.columns and nearby["Is_Night"].notna().any():
        night_share = round(
            float(nearby["Is_Night"].dropna().astype(bool).mean() * 100), 1
        )

    return {
        "Village": str(village["Village"]),
        "_lat": float(village["Latitude"]),
        "_lon": float(village["Longitude"]),
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
        return table.drop(columns=["_lat", "_lon"])

    hs_pts = np.column_stack([
        hotspots["Centre Longitude"].to_numpy(dtype=float) * lon_scale,
        hotspots["Centre Latitude"].to_numpy(dtype=float) * KM_PER_DEG_LAT,
    ])
    village_pts = np.column_stack([
        table["_lon"].to_numpy(dtype=float) * lon_scale,
        table["_lat"].to_numpy(dtype=float) * KM_PER_DEG_LAT,
    ])

    distances, indices = cKDTree(hs_pts).query(village_pts, k=1)
    table["Nearest Hotspot"] = hotspots.iloc[indices]["Hotspot"].to_numpy()
    table["Distance to Hotspot (km)"] = np.round(distances, 2)
    table["Inside Hotspot"] = (
        distances <= hotspots.iloc[indices]["Radius (km)"].to_numpy()
    )
    return table.drop(columns=["_lat", "_lon"])


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
            "No density hotspots found at the current settings. Either the "
            "sightings are spread too evenly to concentrate, or the neighbour "
            "distance is too small for this reporting density -- try widening it."
        )
        return notes

    clustered = int(hotspots["Sightings"].sum())
    total = int(len(df))
    if total:
        notes.append(
            f"{clustered:,} of {total:,} sightings ({clustered / total * 100:.0f}%) fall "
            "inside a hotspot. The rest are too dispersed to cluster and are not "
            "represented in this table."
        )

    oversized = hotspots[hotspots["Radius (km)"] > MAX_SENSIBLE_RADIUS_KM]
    if len(oversized):
        names = ", ".join(oversized["Hotspot"].astype(str))
        notes.append(
            f"{len(oversized)} hotspot(s) span more than {MAX_SENSIBLE_RADIUS_KM:.0f} km "
            f"({names}). At that size the cluster has chained across a broad activity "
            "belt rather than isolating a patrol target -- reduce the neighbour "
            "distance to split it."
        )

    if villages is None or villages.empty:
        notes.append(
            "No village centroids loaded, so hotspots cannot be named against "
            "settlements. The sighting export carries no village field; supply a "
            "centroids file (Village, Latitude, Longitude) to rank villages at risk."
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
