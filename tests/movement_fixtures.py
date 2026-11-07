"""Synthetic elephant movement for testing the next-move forecast.

No real register is in the repository, so the forecast is tested on
groups whose behaviour is known, one ingredient at a time:

* ``persistence`` -- how much each day's heading follows the last
  (0: a fresh direction every day; 0.9: steady travel);
* ``home`` -- pull back towards the group's range centre;
* ``seasonal`` -- the range centre moves between a wet-season and a
  harvest-season range every year, as groups follow crops;
* class -- lone bulls travel further in a day than family herds.

Positions are reported on a share of days, with a few hundred metres of
error, as field reports are. The output has the register's columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

START = pd.Timestamp("2024-10-01")
KM_PER_DEG = 111.32
CLASSES = (  # (total, males, females, calves, km per day)
    (1, 1, 0, 0, 4.0),     # lone bull
    (3, 3, 0, 0, 3.5),     # bull party
    (12, 2, 7, 3, 2.2),    # family herd
)


def _season(day: pd.Timestamp) -> int:
    """0 for the harvest months (Oct-Feb), 1 for the rest."""
    return 0 if day.month in (10, 11, 12, 1, 2) else 1


def simulate(groups: int = 10, days: int = 420, seed: int = 0, persistence: float = 0.0,
             home: float = 0.0, seasonal: bool = False, report_share: float = 0.6,
             range_km: float = 6.0, spacing_km: float = 35.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ref_lat = 23.3
    km_lon = KM_PER_DEG * np.cos(np.radians(ref_lat))
    rows = []
    side = int(np.ceil(np.sqrt(groups)))
    for g in range(groups):
        total, males, females, calves, speed = CLASSES[g % len(CLASSES)]
        anchor = np.array([(g % side) * spacing_km, (g // side) * spacing_km], dtype=float)
        centres = [anchor, anchor + rng.normal(0, 1, 2) / np.sqrt(2) * 14.0] if seasonal \
            else [anchor, anchor]
        pos = anchor + rng.normal(0, 2, 2)
        heading = rng.uniform(-np.pi, np.pi)
        turn_sd = np.pi * (1 - persistence) + 0.15
        for d in range(days):
            day = START + pd.Timedelta(days=d)
            centre = centres[_season(day)] if seasonal else anchor
            heading = heading + rng.normal(0, turn_sd)
            offset = centre - pos
            distance = float(np.hypot(*offset))
            if home and distance > 0:
                pull = min(1.0, home * distance / range_km)
                to_centre = np.arctan2(offset[1], offset[0])
                # Blend the two directions on the circle.
                heading = np.arctan2((1 - pull) * np.sin(heading) + pull * np.sin(to_centre),
                                     (1 - pull) * np.cos(heading) + pull * np.cos(to_centre))
            step = rng.gamma(2.0, speed / 2.0)
            pos = pos + step * np.array([np.cos(heading), np.sin(heading)])
            if rng.random() < report_share:
                seen = pos + rng.normal(0, 0.3, 2)
                rows.append({
                    "Date": day, "Latitude": ref_lat + seen[1] / KM_PER_DEG,
                    "Longitude": 81.0 + seen[0] / km_lon, "Total Count": total,
                    "Male Count": males, "Female Count": females, "Calf Count": calves,
                    "Division": "D", "Range": f"R{g % 3}", "Beat": f"B{g}",
                    "Nearest Village": f"V{g}",
                })
    return pd.DataFrame(rows)
