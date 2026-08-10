"""Village-centroid loading and nearest-village enrichment.

Centroids ship with the repo (``data/centroids.csv``) and load by
default; an uploaded file overrides them.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from core.config import (
    DEFAULT_CENTROIDS_PATH,
    EARTH_RADIUS_KM,
    NEAR_VILLAGE_THRESHOLD_KM,
    VALID_LAT_RANGE,
    VALID_LON_RANGE,
)
from core.csv_io import read_csv_resilient
from core.exceptions import SpatialEnrichmentError

logger = logging.getLogger(__name__)

REQUIRED_VILLAGE_COLUMNS = {"Latitude", "Longitude", "Village"}


def load_village_centroids(
    source: Optional[Union[str, Path, BinaryIO]] = None,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Load and validate a village-centroids table.

    Args:
        source: Path or uploaded file. ``None`` uses the bundled
            ``data/centroids.csv``.

    Returns:
        ``(dataframe, warnings)``. The frame is ``None`` only when no
        source was supplied and the bundled file is missing.

    Raises:
        SpatialEnrichmentError: When a source was supplied explicitly but
            is unreadable or missing required columns, so the caller can
            warn rather than silently skip enrichment.
    """
    warnings: List[str] = []

    explicit_source = source is not None
    if source is None:
        if not Path(DEFAULT_CENTROIDS_PATH).exists():
            logger.warning("Bundled centroids missing at %s.", DEFAULT_CENTROIDS_PATH)
            return None, warnings
        source = DEFAULT_CENTROIDS_PATH

    try:
        villages, read_warnings = read_csv_resilient(
            source, on_error=SpatialEnrichmentError
        )
    except SpatialEnrichmentError as exc:
        if explicit_source:
            raise
        logger.warning("Bundled centroids present but unreadable: %s", exc)
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

    # Repeated names are normal -- the same name in two places is two
    # places. Kept separate so unrelated settlements' risk is not pooled.
    duplicates = int(villages["Village"].astype(str).duplicated().sum())
    if duplicates:
        warnings.append(
            f"{duplicates} village name(s) appear more than once. They are kept as "
            "separate points, so risk is reported per location."
        )

    logger.info("Loaded %d village centroids.", len(villages))
    return villages.reset_index(drop=True), warnings


def attach_nearest_village(
    df: pd.DataFrame,
    villages: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, List[str]]:
    """Attach nearest village, distance and a near-village flag.

    Returns ``df`` unchanged when no centroids are available.
    """
    warnings: List[str] = []
    if villages is None or villages.empty or df.empty:
        return df, warnings

    village_lat = villages["Latitude"].to_numpy(dtype=float)
    village_lon = villages["Longitude"].to_numpy(dtype=float)
    sighting_lat = df["Latitude"].to_numpy(dtype=float)
    sighting_lon = df["Longitude"].to_numpy(dtype=float)

    # Scale longitude by cos(latitude) before the tree. Unscaled degrees
    # overstate east-west separation by ~8% at 22 N and bias the
    # neighbour search toward villages due north or south.
    reference_lat = float(np.nanmean(np.concatenate([village_lat, sighting_lat])))
    lon_scale = math.cos(math.radians(reference_lat))

    tree = cKDTree(np.column_stack([village_lat, village_lon * lon_scale]))
    _, idx = tree.query(np.column_stack([sighting_lat, sighting_lon * lon_scale]), k=1)

    # Report the distance with haversine rather than the planar
    # approximation used to find the neighbour.
    distances = _haversine_km(
        sighting_lat, sighting_lon, village_lat[idx], village_lon[idx]
    )

    out = df.copy()
    out["Nearest Village"] = villages.iloc[idx]["Village"].to_numpy()
    out["Distance to Village (km)"] = np.round(distances, 2)
    out["Near Village"] = out["Distance to Village (km)"] < NEAR_VILLAGE_THRESHOLD_KM

    logger.info("Enriched %d rows against %d villages.", len(out), len(villages))
    return out, warnings


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in kilometres between two arrays of points."""
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
