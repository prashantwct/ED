"""Synthetic presence reports with a known truth.

Elephants use a landscape according to a few hotspots; people report
them only where people are. Each day a number of elephant positions are
drawn from the true intensity, and each is reported with a probability
that falls with distance from the nearest village -- the observer bias
the presence model must see through. A reported position can be filed
two or three times the same day, as when several people report one
herd, which thinning must collapse.

``drift_km_per_day`` moves the first hotspot steadily, a wave of
movement the dynamic terms should follow; ``seasonal_shift_km`` moves it
between a winter and a monsoon position every year.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

START = pd.Timestamp("2024-01-01")
LAT_C, LON_C = 23.3, 81.0
KM_LAT = 110.574
KM_LON = 111.320 * np.cos(np.radians(LAT_C))

Hotspot = Tuple[float, float, float, float]  # x km, y km, sd km, weight


def to_latlon(x, y):
    return LAT_C + np.asarray(y) / KM_LAT, LON_C + np.asarray(x) / KM_LON


def to_xy(lat, lon):
    return (np.asarray(lon) - LON_C) * KM_LON, (np.asarray(lat) - LAT_C) * KM_LAT


def village_points(seed: int = 1, count: int = 40, box: float = 40.0,
                   cluster: Optional[Tuple[float, float, float]] = (20.0, 0.0, 12.0)
                   ) -> pd.DataFrame:
    """Villages, mostly in one part of the landscape (people farm the
    plains, not the hills)."""
    rng = np.random.default_rng(seed)
    if cluster is None:
        xy = rng.uniform(-box, box, (count, 2))
    else:
        cx, cy, sd = cluster
        xy = np.column_stack([rng.normal(cx, sd, count), rng.normal(cy, sd, count)])
    lat, lon = to_latlon(xy[:, 0], xy[:, 1])
    return pd.DataFrame({"Village": [f"V{i:02d}" for i in range(count)],
                         "Latitude": lat, "Longitude": lon})


def truth(x: np.ndarray, y: np.ndarray, hotspots: Sequence[Hotspot]) -> np.ndarray:
    total = np.zeros(np.broadcast(x, y).shape)
    for hx, hy, sd, weight in hotspots:
        total = total + weight * np.exp(-0.5 * ((x - hx) ** 2 + (y - hy) ** 2) / sd ** 2) \
            / (2 * np.pi * sd ** 2)
    return total


def hotspots_on(day: int, hotspots: Sequence[Hotspot], drift_km_per_day: float,
                seasonal_shift_km: float) -> List[Hotspot]:
    out = list(hotspots)
    hx, hy, sd, weight = out[0]
    hx = hx + drift_km_per_day * day
    if seasonal_shift_km:
        date = START + pd.Timedelta(days=day)
        hy = hy + (seasonal_shift_km if date.month in (6, 7, 8, 9) else 0.0)
    out[0] = (hx, hy, sd, weight)
    return out


def simulate(days: int = 560, seed: int = 0,
             hotspots: Sequence[Hotspot] = ((-20.0, 0.0, 4.0, 1.0), (20.0, 0.0, 4.0, 1.0)),
             villages: Optional[pd.DataFrame] = None, detect_km: Optional[float] = 3.0,
             per_day: float = 8.0, drift_km_per_day: float = 0.0,
             seasonal_shift_km: float = 0.0, wrap_km: Optional[float] = None,
             duplicates: float = 0.5) -> pd.DataFrame:
    """Reports in the register's columns. ``detect_km`` None: everywhere
    is watched alike (no observer bias)."""
    rng = np.random.default_rng(seed)
    if villages is None:
        villages = village_points()
    vx, vy = to_xy(villages["Latitude"].to_numpy(), villages["Longitude"].to_numpy())
    rows = []
    for d in range(days):
        spots = hotspots_on(d, hotspots, drift_km_per_day, seasonal_shift_km)
        if wrap_km:
            hx = spots[0][0]
            spots[0] = ((hx + wrap_km) % (2 * wrap_km) - wrap_km, *spots[0][1:])
        weights = np.array([w for *_, w in spots])
        n = rng.poisson(per_day)
        pick = rng.choice(len(spots), size=n, p=weights / weights.sum())
        for k in pick:
            hx, hy, sd, _ = spots[k]
            x, y = rng.normal(hx, sd), rng.normal(hy, sd)
            if detect_km is not None:
                near = np.sqrt(((vx - x) ** 2 + (vy - y) ** 2).min())
                if rng.random() > np.exp(-near / detect_km):
                    continue
            copies = 1 + rng.binomial(2, duplicates / 2)
            for _ in range(copies):
                jx, jy = x + rng.normal(0, 0.15), y + rng.normal(0, 0.15)
                lat, lon = to_latlon(jx, jy)
                rows.append({"Date": START + pd.Timedelta(days=d, hours=int(rng.integers(0, 24))),
                             "Latitude": float(lat), "Longitude": float(lon),
                             "Total Count": int(rng.integers(1, 20))})
    return pd.DataFrame(rows)


def forest(region_x: Tuple[float, float] = (-40.0, 0.0)):
    """A habitat layer: forest over a band of x."""
    lo, hi = region_x

    def share(lat, lon, cell_km):
        x, _ = to_xy(lat, lon)
        return ((x >= lo) & (x < hi)).astype(float)

    return share
