"""Spatial enrichment: attach the nearest known village to each sighting.

This is entirely optional. The dashboard works fully without any village
centroid data - enrichment just adds ``Nearest Village``,
``Distance to Village (km)``, and ``Near Village`` columns when a valid
centroids source is available, which in turn feed the risk engine and
map tooltips.

A centroids source can come from three places, tried in this order:
1. An explicit uploaded file object (e.g. from ``st.file_uploader``).
2. An explicit path passed by the caller.
3. The default ``centroids.csv`` in the working directory, if present.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from core.csv_io import read_csv_resilient
from core.exceptions import SpatialEnrichmentError

logger = logging.getLogger(__name__)

DEFAULT_VILLAGE_FILE = "centroids.csv"
REQUIRED_VILLAGE_COLUMNS = {"Latitude", "Longitude", "Village"}
EARTH_RADIUS_KM = 6371.0

VALID_LAT_RANGE = (-90.0, 90.0)
VALID_LON_RANGE = (-180.0, 180.0)

# Distance below which a sighting counts as "at the village" for risk
# scoring. Centroids are single points, but a village is not: its built
# extent plus immediate homestead fields commonly runs several hundred
# metres from the centre, so a tight radius measured from the centroid
# excludes incidents that happened in the village itself. At 0.5 km
# almost nothing qualifies and the village signal quietly contributes
# nothing to the risk score; 2 km is the more honest reading of what
# centroid-only data can actually support.
NEAR_VILLAGE_THRESHOLD_KM = 2.0


def load_village_centroids(
    source: Optional[Union[str, Path, BinaryIO]] = None,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Load and validate a village-centroids table.

    Args:
        source: A path/uploaded-file for the centroids CSV, or ``None``
            to fall back to :data:`DEFAULT_VILLAGE_FILE` if it exists.

    Returns:
        ``(dataframe, warnings)``. The dataframe has ``Latitude``,
        ``Longitude`` and ``Village`` columns, or is ``None`` if no
        source was supplied/found (a normal, silent no-op - the app just
        skips enrichment). ``warnings`` describes non-fatal problems:
        a fallback encoding, rows dropped for bad coordinates, duplicate
        village names.

    Raises:
        SpatialEnrichmentError: If a source WAS supplied (explicitly, not
            the silent default) but could not be read or is missing
            required columns. This lets the caller warn the user instead
            of silently pretending enrichment happened.
    """
    warnings: List[str] = []

    explicit_source = source is not None
    if source is None:
        if not Path(DEFAULT_VILLAGE_FILE).exists():
            return None, warnings
        source = DEFAULT_VILLAGE_FILE

    try:
        villages, read_warnings = read_csv_resilient(
            source, on_error=lambda msg: SpatialEnrichmentError(msg)
        )
    except SpatialEnrichmentError as exc:
        if explicit_source:
            raise
        logger.warning("Default centroids.csv present but unreadable: %s", exc)
        return None, warnings
    warnings.extend(read_warnings)

    missing = REQUIRED_VILLAGE_COLUMNS - set(villages.columns)
    if missing:
        message = (
            "Village centroids file is missing required column(s): "
            f"{', '.join(sorted(missing))}. Expected columns: "
            f"{', '.join(sorted(REQUIRED_VILLAGE_COLUMNS))}."
        )
        if explicit_source:
            raise SpatialEnrichmentError(message)
        logger.warning(message)
        return None, warnings

    villages = villages.copy()
    original_rows = len(villages)

    villages["Latitude"] = pd.to_numeric(villages["Latitude"], errors="coerce")
    villages["Longitude"] = pd.to_numeric(villages["Longitude"], errors="coerce")
    villages["Village"] = villages["Village"].astype("object").where(
        villages["Village"].notna()
    )
    villages = villages.dropna(subset=["Latitude", "Longitude", "Village"])

    out_of_range = ~(
        villages["Latitude"].between(*VALID_LAT_RANGE)
        & villages["Longitude"].between(*VALID_LON_RANGE)
    )
    if out_of_range.any():
        warnings.append(
            f"Dropped {int(out_of_range.sum())} village(s) with coordinates outside "
            "valid latitude/longitude ranges."
        )
        villages = villages[~out_of_range]

    dropped = original_rows - len(villages)
    if dropped and not out_of_range.any():
        warnings.append(
            f"Dropped {dropped} village(s) with a missing name or coordinates."
        )

    if villages.empty:
        message = "Village centroids file has no valid rows after cleaning."
        if explicit_source:
            raise SpatialEnrichmentError(message)
        logger.warning(message)
        return None, warnings

    duplicates = int(villages["Village"].astype(str).duplicated().sum())
    if duplicates:
        warnings.append(
            f"{duplicates} village name(s) appear more than once. They are kept as "
            "separate points, so risk totals are reported per location rather than "
            "merged under one name."
        )

    return villages.reset_index(drop=True), warnings


def attach_nearest_village(
    df: pd.DataFrame,
    villages: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, List[str]]:
    """Attach the nearest village and distance (km) to each sighting row.

    Args:
        df: Sightings dataframe with ``Latitude``/``Longitude`` columns.
        villages: Output of :func:`load_village_centroids`, or ``None``
            to skip enrichment entirely (returns ``df`` unchanged).

    Returns:
        Tuple of ``(dataframe, warnings)``. The dataframe gains
        ``Nearest Village``, ``Distance to Village (km)``, and
        ``Near Village`` columns when enrichment runs.
    """
    warnings: List[str] = []
    if villages is None or villages.empty or df.empty:
        return df, warnings

    village_lat = villages["Latitude"].to_numpy(dtype=float)
    village_lon = villages["Longitude"].to_numpy(dtype=float)
    sighting_lat = df["Latitude"].to_numpy(dtype=float)
    sighting_lon = df["Longitude"].to_numpy(dtype=float)

    # A degree of longitude is shorter than a degree of latitude by
    # cos(latitude) -- about 0.93 at 22 degrees N, where this data sits.
    # Building the tree on unscaled lat/lon (in degrees or radians alike)
    # treats the two axes as equivalent, which overstates east-west
    # separation by ~8% and biases nearest-village selection towards
    # villages that are north or south rather than east or west. Scaling
    # longitude by cos(latitude) first makes the plane locally isotropic,
    # so the neighbour search is comparing like with like.
    reference_lat = float(np.nanmean(np.concatenate([village_lat, sighting_lat])))
    lon_scale = math.cos(math.radians(reference_lat))

    tree = cKDTree(np.column_stack([village_lat, village_lon * lon_scale]))
    _, idx = tree.query(np.column_stack([sighting_lat, sighting_lon * lon_scale]), k=1)

    # The tree gives us the neighbour; report the distance itself with
    # haversine rather than reusing the planar approximation, so the
    # kilometre figure shown to users is the real great-circle distance.
    distances = _haversine_km(
        sighting_lat, sighting_lon, village_lat[idx], village_lon[idx]
    )

    out = df.copy()
    out["Nearest Village"] = villages.iloc[idx]["Village"].to_numpy()
    out["Distance to Village (km)"] = np.round(distances, 2)
    out["Near Village"] = out["Distance to Village (km)"] < NEAR_VILLAGE_THRESHOLD_KM

    logger.info(
        "Spatial enrichment attached to %d rows using %d villages (lon scale %.4f).",
        len(out),
        len(villages),
        lon_scale,
    )
    return out, warnings


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in kilometres between two arrays of points."""
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
