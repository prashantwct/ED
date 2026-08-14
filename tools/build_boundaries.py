"""Turn the MP forest department shapefiles into vendored GeoJSON.

Run once when the shapefiles change; the output is committed so the app
needs no GIS stack at runtime:

    pip install geopandas
    python -m tools.build_boundaries "/path/to/Shp file" -o data/boundaries

The source is a custom Transverse Mercator ("mp") on WGS84, so
everything is reprojected to EPSG:4326 for the web maps.

Three things happen to keep the payload sane. Geometry is simplified per
level, because a beat outline drawn at landscape zoom cannot show detail
finer than the pixel it lands in. Coordinates are rounded to five
decimals, about a metre, which halves the file for no visible loss. And
the whole thing is gzipped.

Simplification is per-feature with topology preserved, so a shared
boundary between two beats can end up with a hairline sliver between
them. That is acceptable for a display layer and would not be for area
calculations; nothing downstream measures these polygons.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Tolerance in degrees. 0.001 is roughly 100 m at this latitude, which is
# a pixel or two at the zoom each level is drawn.
LEVELS: Dict[str, Dict[str, object]] = {
    "division": {
        # Dissolved from Range rather than read from Division.shp, which
        # holds only the seven territorial divisions and leaves the tiger
        # reserves as holes -- 39% coverage of the sightings against 100%
        # from the dissolve.
        "source": "Range.shp",
        "dissolve": "Div_Name",
        "tolerance": 0.0025,
        "fields": {"Div_Name": "name"},
    },
    "range": {
        "source": "Range.shp",
        "tolerance": 0.0010,
        "fields": {"RNG_NM": "name", "Div_Name": "parent"},
    },
    "beat": {
        "source": "Beat.shp",
        "tolerance": 0.0008,
        "fields": {"Beat_Name": "name", "Range_Name": "parent",
                   "Div_Name": "grandparent"},
    },
}

# Reserve outlines are few and small, and a protected-area manager wants
# them on screen whatever else is showing.
RESERVES = [
    ("TR Shp Files/Bandhavgarh TR/Bandhavgarh.shp", "Bandhavgarh TR", None),
    ("TR Shp Files/Sanjay TR/STR/STR_Boundary.shp", "Sanjay TR", None),
    ("TR Shp Files/Sanjay TR/STR/Core_Boundary.shp", "Sanjay TR core", None),
    ("TR Shp Files/Bandhavgarh TR/Core/Core_Range.shp", None, "Range_Name"),
]
RESERVE_TOLERANCE = 0.0010
COORDINATE_DECIMALS = 5

# The shapefiles carry no CRS on some layers; they share the projection
# of the main hierarchy.
FALLBACK_CRS_FROM = "Division.shp"


def _round(value, decimals: int):
    if isinstance(value, list):
        return [_round(v, decimals) for v in value]
    if isinstance(value, float):
        return round(value, decimals)
    return value


def _write(features: List[dict], path: Path) -> int:
    payload = json.dumps(
        {"type": "FeatureCollection", "features": features},
        separators=(",", ":"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as handle:
        handle.write(payload)
    return len(payload)


def _features(frame, fields: Dict[str, str], tolerance: float) -> List[dict]:
    import geopandas as gpd  # noqa: F401 - imported for the side effect of typing

    frame = frame.copy()
    frame["geometry"] = frame.geometry.simplify(tolerance, preserve_topology=True)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]

    features = []
    for feature in json.loads(frame.to_json())["features"]:
        properties = {
            out: feature["properties"].get(src)
            for src, out in fields.items()
        }
        features.append({
            "type": "Feature",
            "properties": {k: v for k, v in properties.items() if v is not None},
            "geometry": {
                "type": feature["geometry"]["type"],
                "coordinates": _round(
                    feature["geometry"]["coordinates"], COORDINATE_DECIMALS
                ),
            },
        })
    return features


def build(source: Path, out: Path) -> None:
    import geopandas as gpd

    reference = gpd.read_file(source / FALLBACK_CRS_FROM)
    fallback_crs = reference.crs

    for level, spec in LEVELS.items():
        frame = gpd.read_file(source / str(spec["source"])).to_crs(4326)
        if spec.get("dissolve"):
            # buffer(0) first: a few source polygons self-intersect, and
            # the union throws on them otherwise.
            frame["geometry"] = frame.geometry.buffer(0)
            frame = frame.dissolve(by=str(spec["dissolve"]), as_index=False)
        features = _features(frame, spec["fields"], float(spec["tolerance"]))
        size = _write(features, out / f"{level}.geojson.gz")
        written = (out / f"{level}.geojson.gz").stat().st_size
        logger.info("%s: %d features, %.2f MB -> %.2f MB gzipped",
                    level, len(features), size / 1e6, written / 1e6)
        print(f"{level:<10} {len(features):>5} features  "
              f"{size / 1e6:5.2f} MB json  {written / 1e6:5.2f} MB gzipped")

    reserves: List[dict] = []
    for relative, label, name_field in RESERVES:
        path = source / relative
        if not path.exists():
            logger.warning("Reserve layer missing: %s", relative)
            continue
        frame = gpd.read_file(path)
        if frame.crs is None:
            frame = frame.set_crs(fallback_crs)
        frame = frame.to_crs(4326)
        frame["name"] = (
            frame[name_field] if name_field and name_field in frame.columns
            else label
        )
        reserves.extend(_features(frame, {"name": "name"}, RESERVE_TOLERANCE))

    size = _write(reserves, out / "reserve.geojson.gz")
    written = (out / "reserve.geojson.gz").stat().st_size
    print(f"{'reserve':<10} {len(reserves):>5} features  "
          f"{size / 1e6:5.2f} MB json  {written / 1e6:5.2f} MB gzipped")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory holding the .shp files")
    parser.add_argument("-o", "--out", type=Path, default=Path("data/boundaries"))
    args = parser.parse_args()
    build(args.source, args.out)


if __name__ == "__main__":
    main()
