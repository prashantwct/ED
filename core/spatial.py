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
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from core.exceptions import SpatialEnrichmentError

logger = logging.getLogger(__name__)

DEFAULT_VILLAGE_FILE = "centroids.csv"
REQUIRED_VILLAGE_COLUMNS = {"Latitude", "Longitude", "Village"}
NEAR_VILLAGE_THRESHOLD_KM = 0.5
EARTH_RADIUS_KM = 6371.0


def load_village_centroids(
    source: Optional[Union[str, Path, BinaryIO]] = None,
) -> Optional[pd.DataFrame]:
    """Load and validate a village-centroids table.

    Args:
        source: A path/uploaded-file for the centroids CSV, or ``None``
            to fall back to :data:`DEFAULT_VILLAGE_FILE` if it exists.

    Returns:
        A validated DataFrame with ``Latitude``, ``Longitude``, ``Village``
        columns, or ``None`` if no source was supplied/found (this is a
        normal, silent no-op - the app should just skip enrichment).

    Raises:
        SpatialEnrichmentError: If a source WAS supplied (explicitly, not
            the silent default) but could not be read or is missing
            required columns. This lets the caller warn the user instead
            of silently pretending enrichment happened.
    """
    explicit_source = source is not None
    if source is None:
        if not Path(DEFAULT_VILLAGE_FILE).exists():
            return None
        source = DEFAULT_VILLAGE_FILE

    try:
        if hasattr(source, "seek"):
            source.seek(0)
        villages = pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001
        if explicit_source:
            raise SpatialEnrichmentError(
                f"Could not read the village centroids file: {exc}"
            ) from exc
        logger.warning("Default centroids.csv present but unreadable: %s", exc)
        return None

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
        return None

    villages = villages.copy()
    villages["Latitude"] = pd.to_numeric(villages["Latitude"], errors="coerce")
    villages["Longitude"] = pd.to_numeric(villages["Longitude"], errors="coerce")
    villages = villages.dropna(subset=["Latitude", "Longitude", "Village"])

    if villages.empty:
        message = "Village centroids file has no valid rows after cleaning."
        if explicit_source:
            raise SpatialEnrichmentError(message)
        logger.warning(message)
        return None

    return villages.reset_index(drop=True)


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

    village_coords = np.radians(villages[["Latitude", "Longitude"]].to_numpy())
    sighting_coords = np.radians(df[["Latitude", "Longitude"]].to_numpy())

    tree = cKDTree(village_coords)
    dist, idx = tree.query(sighting_coords, k=1)

    out = df.copy()
    out["Nearest Village"] = villages.iloc[idx]["Village"].to_numpy()
    out["Distance to Village (km)"] = (dist * EARTH_RADIUS_KM).round(2)
    out["Near Village"] = out["Distance to Village (km)"] < NEAR_VILLAGE_THRESHOLD_KM

    logger.info("Spatial enrichment attached to %d rows using %d villages.", len(out), len(villages))
    return out, warnings
