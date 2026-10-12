"""Esri 10 m land use / land cover around a point.

Esri's annual Sentinel-2 land use / land cover (Impact Observatory,
CC-BY 4.0) is published as cloud-optimised GeoTIFFs, one per UTM grid
zone and year. This reads a small window around a point from either:

* files a user uploads -- the GeoTIFF the Esri Land Cover Explorer
  downloads, in its UTM projection or reprojected to lat/lon; or
* the published files themselves, read over HTTP with range requests,
  so only the few hundred kilobytes around each point are fetched.

Pixel areas and distances are computed in metres from the raster's own
geometry, so a lat/lon raster gives the same answers as a UTM one.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Class codes in the v02/v003 product.
WATER, TREES, FLOODED, CROPS, BUILT, BARE, SNOW, CLOUDS, RANGELAND = 1, 2, 4, 5, 7, 8, 9, 10, 11
CLASSES: Dict[int, str] = {
    WATER: "Water", TREES: "Trees", FLOODED: "Flooded vegetation", CROPS: "Crops",
    BUILT: "Built area", BARE: "Bare ground", SNOW: "Snow/ice", CLOUDS: "Clouds",
    RANGELAND: "Rangeland",
}
# Esri's published legend colours.
COLORS: Dict[int, str] = {
    WATER: "#419BDF", TREES: "#397D49", FLOODED: "#7A87C6", CROPS: "#E49635",
    BUILT: "#C4281B", BARE: "#A59B8F", SNOW: "#A8EBFF", CLOUDS: "#616161",
    RANGELAND: "#E3E2C3",
}

REMOTE_URL = ("https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/"
              "lc{year}/{tile}_{year}0101-{next}0101.tif")
YEARS = tuple(range(2017, 2025))
DEFAULT_YEAR = 2024

METRES_PER_DEG = 111_320.0
_BANDS = "CDEFGHJKLMNPQRSTUVWX"  # MGRS latitude bands, 8 degrees from 80 S


def utm_tile(lat: float, lon: float) -> str:
    """The UTM grid zone ('44Q') holding a point -- Esri's file naming."""
    zone = int((lon + 180) // 6) + 1
    band = _BANDS[min(max(int((lat + 80) // 8), 0), len(_BANDS) - 1)]
    return f"{zone}{band}"


def remote_url(year: int, lat: float, lon: float) -> str:
    return REMOTE_URL.format(year=year, next=year + 1, tile=utm_tile(lat, lon))


@dataclass
class Window:
    """Land cover codes around a point, with the geometry to measure them."""

    codes: np.ndarray        # 2-D uint8 class codes; 0 is no data
    row: int                 # the point's pixel
    col: int
    dy_m: float              # pixel height and width in metres
    dx_m: float
    source: str
    lat: float = 0.0         # the point the window is centred on
    lon: float = 0.0

    @property
    def pixel_ha(self) -> float:
        return self.dy_m * self.dx_m / 1e4

    def pixel_of(self, lat: float, lon: float) -> Tuple[float, float]:
        """Approximate (row, col) of a nearby point, for marking it."""
        north_m = (lat - self.lat) * METRES_PER_DEG
        east_m = (lon - self.lon) * METRES_PER_DEG * math.cos(math.radians(self.lat))
        return self.row - north_m / self.dy_m, self.col + east_m / self.dx_m

    def disc(self, radius_m: float) -> np.ndarray:
        """Mask of pixels within ``radius_m`` of the point."""
        rows, cols = np.indices(self.codes.shape)
        return (((rows - self.row) * self.dy_m) ** 2
                + ((cols - self.col) * self.dx_m) ** 2) <= radius_m ** 2


class LandCover:
    """Reads windows from a set of GeoTIFFs, local or remote."""

    def __init__(self, sources: Sequence[str] = (), remote_year: Optional[int] = None):
        self.sources = list(sources)
        self.remote_year = remote_year
        self._open: Dict[str, object] = {}
        self.failures: List[str] = []

    @property
    def label(self) -> str:
        if self.remote_year:
            return f"Esri 10 m land cover {self.remote_year} (read online)"
        return f"Esri land cover, {len(self.sources)} uploaded file(s)"

    def _dataset(self, path: str):
        import rasterio

        if path not in self._open:
            try:
                self._open[path] = rasterio.open(path)
            except Exception as exc:  # noqa: BLE001 - a bad tile is reported, not fatal
                self.failures.append(f"{path.rsplit('/', 1)[-1]}: {exc}")
                self._open[path] = None
        return self._open[path]

    def _candidates(self, lat: float, lon: float) -> List[str]:
        if self.remote_year:
            return ["/vsicurl/" + remote_url(self.remote_year, lat, lon)]
        return self.sources

    def window(self, lat: float, lon: float, half_m: float = 3000.0) -> Optional[Window]:
        """The window around (lat, lon) from the first source that covers it."""
        from rasterio.warp import transform as warp
        from rasterio.windows import from_bounds

        for path in self._candidates(lat, lon):
            ds = self._dataset(path)
            if ds is None:
                continue
            try:
                xs, ys = warp("EPSG:4326", ds.crs, [lon], [lat])
                x, y = xs[0], ys[0]
                b = ds.bounds
                if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
                    continue
                if ds.crs.is_geographic:
                    hy = half_m / METRES_PER_DEG
                    hx = half_m / (METRES_PER_DEG * math.cos(math.radians(lat)))
                    dy_m = abs(ds.transform.e) * METRES_PER_DEG
                    dx_m = abs(ds.transform.a) * METRES_PER_DEG * math.cos(math.radians(lat))
                else:
                    hy = hx = half_m
                    dy_m, dx_m = abs(ds.transform.e), abs(ds.transform.a)
                win = from_bounds(x - hx, y - hy, x + hx, y + hy, ds.transform)
                codes = ds.read(1, window=win, boundless=True, fill_value=0).astype(np.uint8)
                # The point's pixel inside the window.
                row = int((ds.transform.f - y) / abs(ds.transform.e) - win.row_off)
                col = int((x - ds.transform.c) / abs(ds.transform.a) - win.col_off)
                row = min(max(row, 0), codes.shape[0] - 1)
                col = min(max(col, 0), codes.shape[1] - 1)
                return Window(codes, row, col, dy_m, dx_m, path.rsplit("/", 1)[-1], lat, lon)
            except Exception as exc:  # noqa: BLE001
                self.failures.append(f"{path.rsplit('/', 1)[-1]}: {exc}")
        return None

    def close(self) -> None:
        for ds in self._open.values():
            if ds is not None:
                ds.close()
        self._open.clear()


def thumbnail_png(window: Window, size: int = 360, marks: Sequence[Tuple[float, float]] = ()
                  ) -> bytes:
    """The window as a PNG in Esri's colours, the point at its centre.

    ``marks`` are (row, col) pixels to ring -- sightings, raids.
    """
    from PIL import Image, ImageDraw

    rgb = np.full(window.codes.shape + (3,), 255, dtype=np.uint8)
    for code, hex_ in COLORS.items():
        rgb[window.codes == code] = [int(hex_[i:i + 2], 16) for i in (1, 3, 5)]
    image = Image.fromarray(rgb).resize((size, size), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    sy, sx = size / window.codes.shape[0], size / window.codes.shape[1]
    for r, c in marks:
        x, y = c * sx, r * sy
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(20, 20, 20), width=1)
    cx, cy = window.col * sx, window.row * sy
    draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(255, 255, 255), width=3)
    draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(20, 20, 20), width=1)
    import io

    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()
