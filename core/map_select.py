"""What the spatial view shows: layer filters and the row picked in a table.

Everything here filters rather than fades. A manager who picks "crop
damage by lone bulls", or one hotspot, sees only those sightings, and
the hotspots and villages that go with them, so the map can be read
and downloaded without the rest of the landscape in the way.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from core.analytics import classify_conflict, classify_group

EVENT_CLASSES = ("Death", "Injury", "House", "Crop", "Presence")
GROUP_CLASSES = ("Lone bull", "Bull party", "Family herd", "Mixed / unsexed", "Unrecorded")


def _km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def class_counts(frame: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Sightings per event class and per animal group, for the filter labels."""
    if frame.empty:
        return {}, {}
    return (classify_conflict(frame).value_counts().to_dict(),
            classify_group(frame).value_counts().to_dict())


def layer_mask(frame: pd.DataFrame, events: Iterable[str],
               groups: Iterable[str]) -> pd.Series:
    """Sightings of a chosen event class seen with a chosen animal group."""
    if frame.empty:
        return pd.Series(dtype=bool, index=frame.index)
    return (classify_conflict(frame).isin(list(events))
            & classify_group(frame).isin(list(groups)))


def _bounds(lat: float, lon: float, km: float) -> Tuple[float, float, float, float]:
    dlat = km / 111.32
    dlon = km / (111.32 * max(math.cos(math.radians(lat)), 0.2))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def _near(frame: pd.DataFrame, lat: float, lon: float, km: float) -> pd.Series:
    d = _km(lat, lon, frame["Latitude"].to_numpy(float), frame["Longitude"].to_numpy(float))
    return pd.Series(d <= km, index=frame.index)


def focus_for(kind: str, value: object, frame: pd.DataFrame,
              hotspots: pd.DataFrame, villages: pd.DataFrame,
              village_km: float) -> Optional[dict]:
    """The sightings a table pick stands for, and how to frame them.

    Returns ``label``, ``mask`` (sightings to keep), optional ``bounds``,
    and the ``beat``, ``hotspot`` or ``village`` to outline.
    """
    if frame.empty:
        return None
    if kind == "beat" and "Beat" in frame:
        mask = frame["Beat"].astype(str) == str(value)
        return {"label": f"Beat {value}", "mask": mask, "beat": value} if mask.any() else None
    if kind == "hotspot" and hotspots is not None and not hotspots.empty:
        row = hotspots[hotspots["Hotspot"].astype(str) == str(value)]
        if row.empty:
            return None
        r = row.iloc[0]
        lat, lon = float(r["Centre Latitude"]), float(r["Centre Longitude"])
        radius = float(r["Radius (km)"])
        return {"label": f"Hotspot {value}", "hotspot": value,
                "mask": _near(frame, lat, lon, radius),
                "bounds": _bounds(lat, lon, max(radius, 0.5) * 1.6)}
    if kind == "village" and villages is not None and not villages.empty:
        row = villages[villages["Village"].astype(str) == str(value)].dropna(
            subset=["Latitude", "Longitude"])
        if row.empty:
            return None
        lat, lon = float(row.iloc[0]["Latitude"]), float(row.iloc[0]["Longitude"])
        return {"label": f"Village {value}", "village": value,
                "mask": _near(frame, lat, lon, village_km),
                "ring": (lat, lon, village_km),
                "bounds": _bounds(lat, lon, village_km * 1.3)}
    return None


def select(frame: pd.DataFrame, hotspots: Optional[pd.DataFrame],
           villages: Optional[pd.DataFrame], *, events: Iterable[str],
           groups: Iterable[str], tiers: Iterable[str], focus: Optional[dict],
           village_km: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The sightings, hotspots and villages the map should draw.

    Sightings must pass the layer filters and the focus. While anything
    narrows them, hotspots are kept only where a shown sighting falls in
    the footprint, and villages only within ``village_km`` of one, so
    the context layers follow the selection. A picked hotspot or village
    always stays on, even when no sighting passes the filters.
    """
    hotspots = hotspots if hotspots is not None else pd.DataFrame()
    villages = villages if villages is not None else pd.DataFrame()
    keep = layer_mask(frame, events, groups)
    if focus and focus.get("mask") is not None:
        keep &= focus["mask"].reindex(frame.index).fillna(False).astype(bool)
    shown = frame[keep]
    narrowed = len(shown) < len(frame)

    if not villages.empty:
        villages = villages[villages["Tier"].astype(str).isin(list(tiers))]

    points = shown.dropna(subset=["Latitude", "Longitude"])
    lat = points["Latitude"].to_numpy(float)
    lon = points["Longitude"].to_numpy(float)

    if narrowed and not hotspots.empty:
        def holds(r) -> bool:
            if focus and focus.get("hotspot") is not None:
                return str(r["Hotspot"]) == str(focus["hotspot"])
            d = _km(float(r["Centre Latitude"]), float(r["Centre Longitude"]), lat, lon)
            return bool((d <= float(r["Radius (km)"])).any())
        hotspots = hotspots[hotspots.apply(holds, axis=1)] if len(hotspots) else hotspots

    if narrowed and not villages.empty:
        located = villages.dropna(subset=["Latitude", "Longitude"])

        def close(r) -> bool:
            if focus and focus.get("village") is not None:
                return str(r["Village"]) == str(focus["village"])
            d = _km(float(r["Latitude"]), float(r["Longitude"]), lat, lon)
            return bool((d <= village_km).any())
        villages = located[located.apply(close, axis=1)] if len(located) else located

    return shown, hotspots, villages
