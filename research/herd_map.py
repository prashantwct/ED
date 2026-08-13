"""Draw movement-unit ranges over land cover.

The basemap is the WorldCover raster itself rather than fetched tiles.
It needs no network, and for this question it is the better background
anyway: what a manager wants to see is which unit works which habitat.

    python -m research.herd_map path/to/sightings.csv -o herds.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from core.data_loader import load_and_validate_csv
from research.herds import assign_units, summarise_units
from research.landcover import LandCover

WIDTH, HEIGHT = 1400, 900
MARGIN = 40

# Muted so the overlay reads on top of it.
CLASS_COLOURS: Dict[int, Tuple[int, int, int]] = {
    10: (176, 196, 168),   # tree cover
    20: (206, 206, 178),   # shrubland
    30: (222, 222, 194),   # grassland
    40: (238, 230, 205),   # cropland
    50: (206, 198, 196),   # built-up
    60: (226, 220, 210),   # bare
    80: (188, 210, 224),   # water
    90: (198, 214, 214),   # wetland
}
BACKGROUND = (244, 244, 240)

UNIT_COLOURS = {
    "lone bull": (176, 38, 38),
    "bull party": (214, 112, 24),
    "family herd": (38, 106, 152),
    "unsexed group": (120, 128, 120),
}

# Units overlap where they share a corridor, so their labels do too.
LABEL_SEPARATION_PX = 90
FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf")


def _font(size: int):
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _extent(df: pd.DataFrame, pad: float = 0.04):
    return (df["Longitude"].min() - pad, df["Latitude"].min() - pad,
            df["Longitude"].max() + pad, df["Latitude"].max() + pad)


def _projector(west, south, east, north):
    """Equirectangular, scaled so the aspect ratio survives at 23 N."""
    lon_scale = math.cos(math.radians((south + north) / 2))
    span_x = (east - west) * lon_scale
    span_y = north - south
    scale = min((WIDTH - 2 * MARGIN) / span_x, (HEIGHT - 2 * MARGIN) / span_y)
    off_x = (WIDTH - span_x * scale) / 2
    off_y = (HEIGHT - span_y * scale) / 2

    def project(lat, lon):
        return (off_x + (lon - west) * lon_scale * scale,
                off_y + (north - lat) * scale)

    return project


def _basemap(cover: LandCover, west, south, east, north) -> Image.Image:
    """Land-cover raster, downsampled and colour-mapped to the frame."""
    top, left = cover._rowcol(north, west)
    bottom, right = cover._rowcol(south, east)
    top, left = max(top, 0), max(left, 0)
    bottom = min(bottom, cover.array.shape[0])
    right = min(right, cover.array.shape[1])
    patch = cover.array[top:bottom, left:right]
    if patch.size == 0:
        return Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)

    step = max(patch.shape[0] // HEIGHT, patch.shape[1] // WIDTH, 1)
    patch = patch[::step, ::step]

    rgb = np.full(patch.shape + (3,), BACKGROUND, dtype=np.uint8)
    for code, colour in CLASS_COLOURS.items():
        rgb[patch == code] = colour
    return Image.fromarray(rgb).resize((WIDTH, HEIGHT), Image.NEAREST)


def render(df: pd.DataFrame, units: pd.Series, table: pd.DataFrame,
           top: int = 12, out: Path = Path("herds.png")) -> Path:
    west, south, east, north = _extent(df)
    project = _projector(west, south, east, north)

    cover = LandCover.for_points(df["Latitude"].values, df["Longitude"].values)
    image = _basemap(cover, west, south, east, north)
    canvas = Image.new("RGBA", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(canvas, "RGBA")

    shown = table.head(top)
    work = df.copy()
    work["Unit"] = units.values
    lookup = {f"U{u:03d}": u for u in work["Unit"].unique()}

    badges: List[Tuple[float, float, str, Tuple[int, int, int]]] = []

    for rank, (_, row) in enumerate(shown.iterrows(), start=1):
        colour = UNIT_COLOURS.get(row["Class"], (120, 120, 120))
        polygon = [project(lat, lon) for lat, lon in row["Polygon"]]
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=colour + (46,), outline=colour + (220,))
        track = work[work["Unit"] == lookup[row["Unit"]]].sort_values("Date")
        points = [project(lat, lon) for lat, lon
                  in zip(track["Latitude"], track["Longitude"])]
        if len(points) > 1:
            draw.line(points, fill=colour + (150,), width=2)
        for x, y in points:
            draw.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5],
                         fill=colour + (230,))

        cx, cy = project(row["Centre Latitude"], row["Centre Longitude"])
        badges.append((cx, cy, str(rank), colour))

    image = Image.alpha_composite(image.convert("RGBA"), canvas).convert("RGB")
    _draw_badges(image, badges)
    _legend(image, shown)
    image.save(out)
    return out


def _draw_badges(image: Image.Image, badges) -> None:
    """Numbered discs. Units share corridors, so their centroids collide
    and a name at each one is unreadable; the number keys the legend."""
    draw = ImageDraw.Draw(image)
    font = _font(13)
    for x, y, text, colour in badges:
        draw.ellipse([x - 11, y - 11, x + 11, y + 11], fill=colour,
                     outline=(255, 255, 255), width=2)
        box = draw.textbbox((0, 0), text, font=font)
        draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - 1),
                  text, font=font, fill=(255, 255, 255))


def _legend(image: Image.Image, shown: pd.DataFrame) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    title, body, note = _font(17), _font(13), _font(12)

    height = 66 + 20 * len(shown) + 66
    draw.rectangle([0, 0, 430, height], fill=(255, 255, 255, 230))
    draw.text((14, 12), "Movement units ranked by conflict", font=title,
              fill=(20, 20, 20))
    draw.text((14, 36), "  #   unit      conflicts  class           range",
              font=note, fill=(110, 110, 110))

    y = 56
    for rank, (_, row) in enumerate(shown.iterrows(), start=1):
        colour = UNIT_COLOURS.get(row["Class"], (120, 120, 120))
        draw.text((14, y), f"{rank:>2}", font=body, fill=(90, 90, 90))
        draw.ellipse([36, y + 2, 48, y + 14], fill=colour)
        draw.text((56, y), f"{row['Unit']}", font=body, fill=(30, 30, 30))
        draw.text((124, y), f"{int(row['Conflict Events'])}", font=body,
                  fill=(30, 30, 30))
        draw.text((176, y), row["Class"], font=body, fill=(60, 60, 60))
        draw.text((300, y), f"{row['Range (km2)']:,.0f} km²", font=body,
                  fill=(60, 60, 60))
        y += 20

    y += 8
    draw.text((14, y), "Background: ESA WorldCover 2021 — green tree cover, "
              "cream cropland, blue water", font=note, fill=(90, 90, 90))
    draw.text((14, y + 17), "Polygon is the range observed in this window, "
              "not an annual home range", font=note, fill=(90, 90, 90))
    draw.text((14, y + 34), "Units are inferred from time, place and group "
              "composition — not identified animals", font=note,
              fill=(90, 90, 90))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("-o", "--out", default="herds.png")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    df, _ = load_and_validate_csv(args.csv)
    units = assign_units(df)
    table = summarise_units(df, units)
    path = render(df, units, table, top=args.top, out=Path(args.out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
