"""Land cover and cropping features from ESA WorldCover.

WorldCover v200 is a 10 m global land-cover map for 2021, published by ESA
under CC-BY 4.0 and served as cloud-optimised GeoTIFF from a public S3
bucket. Tiles are 3 degrees square and named by their south-west corner.

This replaces the settlement-density proxy used in the first pass of the
model. That proxy -- counting village centroids within 5 km -- was the
single most important feature, which is exactly why guessing at it was
unsatisfactory.

Cropping season is not observed. WorldCover gives where cropland is, not
what is standing in it, and the Sentinel-2 archive is not reachable from
this environment. The calendar below encodes the Madhya Pradesh cropping
year as domain knowledge and is labelled as such wherever it is used.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
CACHE_DIR = Path(os.getenv("ED_DATA_CACHE", Path(__file__).resolve().parent.parent / ".data_cache"))
TILE_DEGREES = 3

# ESA WorldCover class codes.
CLASSES = {
    10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
    60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove",
    100: "moss",
}
FOREST_CLASS = 10
CROP_CLASS = 40
BUILT_CLASS = 50
WATER_CLASSES = (80, 90)

BUFFERS_KM = (1.0, 2.0, 5.0)
# Raiding happens where a crop field meets cover. One WorldCover pixel is
# 10 m, so three pixels is a 30 m band either side of the boundary.
INTERFACE_PIXELS = 3

METRES_PER_DEG_LAT = 110_574.0

# Madhya Pradesh cropping year. Kharif is sown with the monsoon and
# harvested after it; rabi is sown on residual moisture and harvested in
# spring. Between them the fields are bare. Standing crop is what an
# elephant raids, so the month matters as much as the hectares.
STANDING_CROP = {
    1: 1.0,   # rabi wheat and gram, tillering to grain fill
    2: 1.0,   # rabi at peak, most vulnerable
    3: 0.7,   # rabi harvest under way
    4: 0.2,   # residue and stubble
    5: 0.1,   # bare, pre-monsoon
    6: 0.3,   # kharif sowing with the monsoon onset
    7: 0.6,   # kharif establishing
    8: 0.9,
    9: 1.0,   # kharif paddy and maize at grain fill
    10: 0.9,  # kharif harvest begins
    11: 0.6,  # kharif harvest, rabi sowing
    12: 0.9,  # rabi establishing
}
KHARIF_MONTHS = (6, 7, 8, 9, 10)
RABI_MONTHS = (11, 12, 1, 2, 3)


def tiles_for_bounds(west: float, south: float, east: float, north: float) -> List[str]:
    """WorldCover tile names covering a bounding box."""
    names = []
    lat0 = int(math.floor(south / TILE_DEGREES) * TILE_DEGREES)
    lat1 = int(math.floor(north / TILE_DEGREES) * TILE_DEGREES)
    lon0 = int(math.floor(west / TILE_DEGREES) * TILE_DEGREES)
    lon1 = int(math.floor(east / TILE_DEGREES) * TILE_DEGREES)
    for lat in range(lat0, lat1 + 1, TILE_DEGREES):
        for lon in range(lon0, lon1 + 1, TILE_DEGREES):
            ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            names.append(f"{ns}{ew}")
    return names


def ensure_tiles(tiles: List[str], cache_dir: Path = CACHE_DIR) -> List[Path]:
    """Download any missing tiles. Roughly 120 MB each."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tile in tiles:
        path = cache_dir / f"WC_{tile}.tif"
        if not path.exists() or path.stat().st_size < 1_000_000:
            url = WORLDCOVER_URL.format(tile=tile)
            logger.info("Fetching WorldCover %s", tile)
            with urlopen(url, timeout=300) as response, open(path, "wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
        paths.append(path)
    return paths


def load_mosaic(
    west: float, south: float, east: float, north: float,
    cache_dir: Path = CACHE_DIR,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Read the land-cover window covering a bounding box.

    Returns the raster plus the geotransform pieces needed to map
    lat/lon onto array indices: (array, west, north, lon_step, lat_step).
    """
    import rasterio
    from rasterio.merge import merge

    paths = ensure_tiles(tiles_for_bounds(west, south, east, north), cache_dir)
    sources = [rasterio.open(p) for p in paths]
    try:
        array, transform = merge(sources, bounds=(west, south, east, north))
    finally:
        for src in sources:
            src.close()
    return array[0], transform.c, transform.f, transform.a, transform.e


class LandCover:
    """Land-cover window with buffer statistics around arbitrary points."""

    def __init__(self, array: np.ndarray, west: float, north: float,
                 lon_step: float, lat_step: float):
        self.array = array
        self.west = west
        self.north = north
        self.lon_step = lon_step
        self.lat_step = lat_step          # negative: rows run north to south
        self.forest = array == FOREST_CLASS
        self.crop = array == CROP_CLASS
        # Built once: np.isin over a 270 MB uint8 array costs 0.14 s a
        # call, which dominates everything else at 600 villages.
        self.water = (array == WATER_CLASSES[0]) | (array == WATER_CLASSES[1])
        self._interface = None

    @classmethod
    def for_points(cls, lats, lons, pad_km: float = 6.0, **kw) -> "LandCover":
        pad = pad_km * 1000 / METRES_PER_DEG_LAT
        return cls(*load_mosaic(
            float(np.min(lons)) - pad, float(np.min(lats)) - pad,
            float(np.max(lons)) + pad, float(np.max(lats)) + pad, **kw
        ))

    def _rowcol(self, lat: float, lon: float) -> Tuple[int, int]:
        return (int((lat - self.north) / self.lat_step),
                int((lon - self.west) / self.lon_step))

    @property
    def crop_forest_interface(self) -> np.ndarray:
        """Cropland pixels lying within a short distance of tree cover.

        This is the raiding surface: a field with cover beside it lets a
        bull feed with an escape route, which a field in open country
        does not.
        """
        if self._interface is None:
            from scipy.ndimage import binary_dilation
            near_forest = binary_dilation(
                self.forest, iterations=INTERFACE_PIXELS
            )
            self._interface = self.crop & near_forest
        return self._interface

    def window(self, lat: float, lon: float, radius_km: float):
        """Array slice covering a circular buffer, with a circular mask."""
        row, col = self._rowcol(lat, lon)
        lat_px = int(round(radius_km * 1000 / METRES_PER_DEG_LAT / abs(self.lat_step)))
        lon_px = int(round(radius_km * 1000
                           / (METRES_PER_DEG_LAT * math.cos(math.radians(lat)))
                           / abs(self.lon_step)))
        r0, r1 = max(row - lat_px, 0), min(row + lat_px + 1, self.array.shape[0])
        c0, c1 = max(col - lon_px, 0), min(col + lon_px + 1, self.array.shape[1])
        if r1 <= r0 or c1 <= c0:
            return None, None

        rows = (np.arange(r0, r1) - row) / max(lat_px, 1)
        cols = (np.arange(c0, c1) - col) / max(lon_px, 1)
        mask = (rows[:, None] ** 2 + cols[None, :] ** 2) <= 1.0
        return (slice(r0, r1), slice(c0, c1)), mask

    def stats(self, lat: float, lon: float) -> Dict[str, float]:
        """Composition and configuration around one point."""
        out: Dict[str, float] = {}
        for radius in BUFFERS_KM:
            box, mask = self.window(lat, lon, radius)
            tag = f"{radius:g}km".replace(".", "_")
            if box is None or not mask.any():
                for name in ("tree", "crop", "built", "grass", "shrub", "water"):
                    out[f"{name}_frac_{tag}"] = 0.0
                out[f"crop_edge_frac_{tag}"] = 0.0
                out[f"forest_edge_density_{tag}"] = 0.0
                continue

            patch = self.array[box][mask]
            total = patch.size
            for code, name in ((FOREST_CLASS, "tree"), (CROP_CLASS, "crop"),
                               (BUILT_CLASS, "built"), (30, "grass"), (20, "shrub")):
                out[f"{name}_frac_{tag}"] = float((patch == code).sum() / total)
            out[f"water_frac_{tag}"] = float(self.water[box][mask].sum() / total)
            out[f"crop_edge_frac_{tag}"] = float(
                self.crop_forest_interface[box][mask].sum() / total
            )
            out[f"forest_edge_density_{tag}"] = self._edge_density(box, mask)

        out["dist_forest_km"] = self._distance_to(self.forest, lat, lon)
        out["dist_water_km"] = self._distance_to(self.water, lat, lon)
        return out

    def _edge_density(self, box, mask) -> float:
        """Forest/non-forest boundary length per unit area.

        Two landscapes can hold the same forest fraction and behave
        differently: one solid block, or the same hectares shredded into
        fingers between fields. The second has far more edge, and edge is
        where conflict happens.
        """
        patch = self.forest[box]
        if patch.shape[0] < 2 or patch.shape[1] < 2:
            return 0.0
        horizontal = patch[:, :-1] != patch[:, 1:]
        vertical = patch[:-1, :] != patch[1:, :]
        edges = horizontal[:-1, :].sum() + vertical[:, :-1].sum()
        return float(edges / max(mask.sum(), 1))

    def _distance_to(self, layer: np.ndarray, lat: float, lon: float,
                     limit_km: float = 10.0) -> float:
        """Straight-line distance to the nearest pixel of a class."""
        box, _ = self.window(lat, lon, limit_km)
        if box is None:
            return limit_km
        patch = layer[box]
        if not patch.any():
            return limit_km
        row, col = self._rowcol(lat, lon)
        rows, cols = np.nonzero(patch)
        d_lat = (rows + box[0].start - row) * abs(self.lat_step) * METRES_PER_DEG_LAT
        d_lon = ((cols + box[1].start - col) * abs(self.lon_step)
                 * METRES_PER_DEG_LAT * math.cos(math.radians(lat)))
        return float(np.sqrt(d_lat**2 + d_lon**2).min() / 1000)


def landcover_table(points: pd.DataFrame, cover: Optional[LandCover] = None,
                    **kw) -> pd.DataFrame:
    """Land-cover statistics for a table with Latitude/Longitude columns."""
    if cover is None:
        cover = LandCover.for_points(points["Latitude"].values,
                                     points["Longitude"].values, **kw)
    rows = [cover.stats(float(lat), float(lon))
            for lat, lon in zip(points["Latitude"], points["Longitude"])]
    return pd.DataFrame(rows, index=points.index)


def cropping_features(month: pd.Series, crop_fraction: pd.Series,
                      edge_fraction: pd.Series) -> pd.DataFrame:
    """Combine where cropland is with when a crop is standing in it.

    Hectares alone do not create risk. A harvested field in May is not
    the same place as the same field in February.
    """
    standing = month.map(STANDING_CROP).astype(float)
    return pd.DataFrame({
        "standing_crop_index": standing,
        "is_kharif": month.isin(KHARIF_MONTHS).astype(int),
        "is_rabi": month.isin(RABI_MONTHS).astype(int),
        "exposed_cropland": standing * crop_fraction.astype(float),
        "exposed_crop_edge": standing * edge_fraction.astype(float),
    }, index=month.index)
