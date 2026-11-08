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


def _route(rng: np.random.Generator, start: np.ndarray, length_km: float = 50.0,
           points: int = 400) -> np.ndarray:
    """A winding path from ``start``: the corridor a herd keeps to."""
    heading = rng.uniform(-np.pi, np.pi)
    s = np.linspace(0.0, length_km, points)
    bend = 6.0 * np.sin(s / length_km * 2 * np.pi * rng.uniform(0.6, 1.4))
    along = np.array([np.cos(heading), np.sin(heading)])
    across = np.array([-along[1], along[0]])
    return start + s[:, None] * along[None, :] + bend[:, None] * across[None, :]


def seasonal_routes(herds: int = 6, days: int = 800, seed: int = 0, per_route: int = 2,
                    speed_km: float = 3.0, report_share: float = 0.5,
                    spacing_km: float = 70.0) -> pd.DataFrame:
    """Herds that migrate along the same corridor every year.

    Each route runs from a monsoon range A to a winter range B. In October
    the herds on it walk A to B along the corridor; in June, back. In
    between they range around the end they reached. So the same cells on
    the corridor see herds heading one way in winter and the other way in
    the monsoon: where they went from a cell depends on the season, which
    the usage map (where they have been) cannot say and a group's last
    move says only once it is already travelling.
    """
    rng = np.random.default_rng(seed)
    ref_lat = 23.3
    km_lon = KM_PER_DEG * np.cos(np.radians(ref_lat))
    routes = []
    side = int(np.ceil(np.sqrt(max(herds // per_route, 1))))
    for r in range(int(np.ceil(herds / per_route))):
        start = np.array([(r % side) * spacing_km, (r // side) * spacing_km])
        routes.append(_route(rng, start))
    rows = []
    for h in range(herds):
        path = routes[h // per_route]
        steps = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
        total = 8 + 4 * h
        delay = rng.integers(0, 10)
        pos = path[0] + rng.normal(0, 1.5, 2)
        for d in range(days):
            day = START + pd.Timedelta(days=d)
            out_start = pd.Timestamp(day.year if day.month >= 10 else day.year - 1, 10, 5) \
                + pd.Timedelta(days=int(delay))
            back_start = pd.Timestamp(day.year if day.month >= 6 else day.year - 1, 6, 5) \
                + pd.Timedelta(days=int(delay))
            since_out = (day - out_start).days
            since_back = (day - back_start).days
            # Which phase: the most recent departure decides the direction.
            if since_out >= 0 and (since_back < 0 or since_out < since_back):
                travelled, forward = since_out * speed_km, True
            else:
                travelled, forward = since_back * speed_km, False
            if travelled < steps[-1]:
                s = travelled if forward else steps[-1] - travelled
                k = int(np.searchsorted(steps, s))
                pos = path[min(k, len(path) - 1)] + rng.normal(0, 0.5, 2)
            else:
                home = path[-1] if forward else path[0]
                pos = pos + rng.normal(0, 1.5, 2) + 0.3 * (home - pos)
            if rng.random() < report_share:
                seen = pos + rng.normal(0, 0.3, 2)
                rows.append({
                    "Date": day, "Latitude": ref_lat + seen[1] / KM_PER_DEG,
                    "Longitude": 81.0 + seen[0] / km_lon, "Total Count": total,
                    "Male Count": 2, "Female Count": total - 5, "Calf Count": 3,
                    "Division": "D", "Range": "R", "Beat": f"B{h}",
                    "Nearest Village": f"V{h}",
                })
    return pd.DataFrame(rows)


def circuits(herds: int = 6, days: int = 500, seed: int = 0, report_share: float = 0.4,
             radius_km: float = 7.0, speed_km: float = 3.0) -> pd.DataFrame:
    """Herds walking a loop, always the same way round, stopping and starting
    at random. Which way a herd goes from a place is fixed by the place, not
    by the date, and with reports this patchy a herd's last move is often
    unknown: what the grid memory is for."""
    rng = np.random.default_rng(seed)
    km_lon = KM_PER_DEG * np.cos(np.radians(23.3))
    rows = []
    for h in range(herds):
        centre = np.array([(h % 3) * 60.0, (h // 3) * 60.0])
        theta = rng.uniform(0, 2 * np.pi)
        moving = True
        for d in range(days):
            if rng.random() < (0.15 if moving else 0.3):
                moving = not moving
            if moving:
                theta += speed_km / radius_km
            pos = centre + radius_km * np.array([np.cos(theta), np.sin(theta)]) \
                + rng.normal(0, 0.6, 2)
            if rng.random() < report_share:
                rows.append({"Date": START + pd.Timedelta(days=d),
                             "Latitude": 23.3 + pos[1] / KM_PER_DEG,
                             "Longitude": 81.0 + pos[0] / km_lon, "Total Count": 10 + h,
                             "Male Count": 2, "Female Count": 5 + h, "Calf Count": 3,
                             "Division": "D", "Range": "R", "Beat": f"B{h}",
                             "Nearest Village": "V"})
    return pd.DataFrame(rows)
