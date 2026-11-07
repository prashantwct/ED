"""Where groups seen in the last few days may be next, and how often that has been right.

The register has no collars and no identified animals, only a time, a
place and a rough count per report. So the forecast is built from parts,
each of them inspectable:

1. **Movement units.** Reports are chained into tracks the way radar
   tracks aircraft (``assign_units``): a report joins the best-matching
   open track if it is close enough in space for the time elapsed and its
   group composition is compatible. A unit is a plausible sequence of
   reports of one group, not an identified animal.
2. **How far** (``StepModel``). Every consecutive pair of reports in a
   unit, a day or more apart, is one observed step. Their lengths, scaled
   to one day, give the distance a group covers -- per class once a class
   has enough steps of its own, since a lone bull covers more ground than
   a family herd.
3. **Which way** (``StepModel.turns``). The direction of each move relative
   to the move before it: steady travel shows as turns near zero, a group
   shuttling between a day refuge and the fields it raids as turns near a
   half circle. Learned from the register, so it can be either.
4. **Where** (``usage_prior``, ``season_prior``). Elephants keep returning
   to the same forest blocks, nalas and fields, so the places reported
   before -- recent ones weighted more, and optionally the same weeks of
   earlier years -- form a usage map.
5. **Its own range.** A group's own earlier reports.

The forecast for a group is the reach centred on its last report, times
whichever of the other parts help: within reach, in the direction groups
tend to go, where elephants have been and where this group has been. Its
*likely zone* is the smallest set of 1 km cells holding half the
forecast, its *possible zone* 80%.

**Every forecast comes with its own track record** (``backtest``). Each
month of the register is forecast from the months before it only and
scored against where the group was actually reported next. Which parts to
switch on is chosen the same way -- the combination with the best log
score on the earlier months -- and so is the size of the zones: if zones
of half the forecast caught the group only 40% of the time, the likely
zone is widened until it would have caught half (a conformal
calibration). The comparison throughout is a circle around where the
group was last seen -- "they stay near where they were". If the forecast
does not beat that, the page says so.

What this cannot do: predict a first visit to a place nobody has reported
elephants in, follow a group through a reporting gap longer than
``MAX_GAP_DAYS``, or tell two bulls in the same valley apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

# ---------------------------------------------------------------------------
# Movement units (moved here from research/herds.py, which re-exports them)
# ---------------------------------------------------------------------------
# Sustained daily movement. Twenty is a hard day for a herd; beyond it
# the two reports are far more likely to be different animals.
MAX_SPEED_KM_PER_DAY = 15.0
# Two observers reporting the same herd hours apart, from opposite sides
# of it, still need room to be the same herd.
SAME_DAY_RADIUS_KM = 12.0
# Reporting is patchy. Past this, a track is closed rather than guessed
# across the gap.
MAX_GAP_DAYS = 4
# Counts are eyeball estimates from the field. A herd of nine reported
# as eight is the same herd.
COUNT_TOLERANCE = 2
COUNT_TOLERANCE_FRACTION = 0.4

SOLITARY_MAX = 3


def km_plane(lat: np.ndarray, lon: np.ndarray, ref_lat: float) -> np.ndarray:
    """Project to a local plane where one unit is one kilometre.

    Longitude degrees shrink with latitude; leaving them unscaled
    stretches east-west distances by 8% in this landscape.
    """
    x = np.radians(lon) * np.cos(np.radians(ref_lat)) * EARTH_RADIUS_KM
    y = np.radians(lat) * EARTH_RADIUS_KM
    return np.column_stack([x, y])


def pair_km(lat_a, lon_a, lat_b, lon_b):
    """East and north offsets in km from a to b, measured at the pair's own latitude.

    Distances between two reports must not depend on which other reports
    are in the register -- a global reference latitude would move them
    whenever a new area joined it -- so each pair is projected about its
    own midpoint.
    """
    lat_a, lon_a = np.asarray(lat_a, dtype=float), np.asarray(lon_a, dtype=float)
    lat_b, lon_b = np.asarray(lat_b, dtype=float), np.asarray(lon_b, dtype=float)
    mid = np.radians((lat_a + lat_b) / 2)
    dx = np.radians(lon_b - lon_a) * np.cos(mid) * EARTH_RADIUS_KM
    dy = np.radians(lat_b - lat_a) * EARTH_RADIUS_KM
    return dx, dy


def from_plane(xy: np.ndarray, ref_lat: float) -> np.ndarray:
    """Back from ``km_plane`` to (lat, lon) columns."""
    lat = np.degrees(xy[:, 1] / EARTH_RADIUS_KM)
    lon = np.degrees(xy[:, 0] / (EARTH_RADIUS_KM * np.cos(np.radians(ref_lat))))
    return np.column_stack([lat, lon])


def _class_of(total: float, calves: float, males: float, females: float) -> str:
    """Coarse social class. The one distinction the data can support."""
    if calves > 0 or females > 0:
        return "family herd"
    if total <= 1:
        return "lone bull"
    if total <= SOLITARY_MAX:
        return "bull party"
    return "unsexed group"


def _compatible(a: Dict, b: Dict) -> bool:
    """Could these two reports be the same animals?"""
    if a["social"] != b["social"]:
        return False
    # A herd does not shed or acquire calves between sightings.
    if (a["calves"] > 0) != (b["calves"] > 0):
        return False
    tolerance = max(COUNT_TOLERANCE,
                    COUNT_TOLERANCE_FRACTION * (a["total"] + b["total"]) / 2)
    return abs(a["total"] - b["total"]) <= tolerance


def _reach_km(gap_days: float, max_speed_km_per_day: float) -> float:
    """How far a unit could plausibly have moved in the elapsed time.

    Same-day reports get a fixed radius rather than zero, because two
    observers can log one herd hours apart from opposite sides of it.
    That radius applies only at zero gap: using it as a floor at every
    gap would mask the speed setting below 12 km/day and make the
    sensitivity table understate how much the thresholds matter.
    """
    if gap_days <= 0:
        return SAME_DAY_RADIUS_KM
    return max_speed_km_per_day * gap_days


def _records(frame: pd.DataFrame) -> List[Dict]:
    counts = frame.reindex(
        columns=["Male Count", "Female Count", "Calf Count", "Total Count"]
    ).apply(pd.to_numeric, errors="coerce").fillna(0)
    out = []
    for total, calves, males, females in zip(
            counts["Total Count"], counts["Calf Count"], counts["Male Count"],
            counts["Female Count"]):
        out.append({"total": float(total), "calves": float(calves),
                    "males": float(males), "females": float(females),
                    "social": _class_of(float(total), float(calves), float(males),
                                        float(females))})
    return out


def assign_units(
    df: pd.DataFrame,
    max_speed_km_per_day: float = MAX_SPEED_KM_PER_DAY,
    max_gap_days: int = MAX_GAP_DAYS,
) -> pd.Series:
    """Greedy nearest-neighbour tracking over time-ordered sightings.

    Each report either extends the best-matching open track or starts a
    new one. Assigning to a single best candidate rather than every
    candidate within reach is what keeps this from degenerating into one
    connected component covering the whole landscape.
    """
    if df.empty:
        return pd.Series(dtype=int, index=df.index)
    frame = df.sort_values("Date", kind="stable").reset_index()
    lat = frame["Latitude"].to_numpy(dtype=float)
    lon = frame["Longitude"].to_numpy(dtype=float)
    day = frame["Date"].values.astype("datetime64[D]").astype(int)
    records = _records(frame)

    unit_of = np.full(len(frame), -1, dtype=int)
    # Open tracks, each holding the index of its most recent sighting.
    open_tracks: List[int] = []
    next_unit = 0

    for i in range(len(frame)):
        open_tracks = [t for t in open_tracks if day[i] - day[t] <= max_gap_days]

        best, best_cost = None, np.inf
        for t in open_tracks:
            if not _compatible(records[i], records[t]):
                continue
            gap = day[i] - day[t]
            reach = _reach_km(gap, max_speed_km_per_day)
            distance = float(np.hypot(*pair_km(lat[t], lon[t], lat[i], lon[i])))
            if distance > reach:
                continue
            # Prefer close in space, then recent in time.
            cost = distance / reach + 0.1 * gap
            if cost < best_cost:
                best, best_cost = t, cost

        if best is None:
            unit_of[i] = next_unit
            next_unit += 1
        else:
            unit_of[i] = unit_of[best]
            open_tracks.remove(best)
        open_tracks.append(i)

    return pd.Series(unit_of, index=frame["index"]).reindex(df.index)




# ---------------------------------------------------------------------------
# The forecast
# ---------------------------------------------------------------------------
CELL_KM = 1.0
# Usage map: each past report spread over ~1.5 km, recent ones counting more.
USAGE_BANDWIDTH_KM = 1.5
USAGE_HALF_LIFE_DAYS = 120.0
# A place with no reports is unlikely, not impossible: this share of the
# mean is added everywhere, to the usage map and to a group's own range.
USAGE_FLOOR = 0.05
# A group's own range: its earlier reports, each spread over ~2 km.
OWN_BANDWIDTH_KM = 2.0
OWN_MAX_REPORTS = 60
# The same season in earlier years: reports within this many days of the
# same calendar date, at least ``SEASON_MIN_AGE_DAYS`` old.
SEASON_HALF_WIDTH_DAYS = 30
SEASON_MIN_AGE_DAYS = 300
# Steps used to learn how far groups move; fewer than this and a typical
# figure for elephants in central India stands in (3.8 km/day median).
MIN_STEPS = 30
# A class (lone bull, bull party, family herd) gets its own step lengths
# once it has this many steps of its own.
MIN_CLASS_STEPS = 60
# Turns used to learn direction of travel, and the shortest move whose
# direction means anything given a few hundred metres of report error.
MIN_TURNS = 30
MIN_MOVE_KM = 0.5
MAX_KAPPA = 20.0
DEFAULT_KM_PER_DAY = 3.8
# Groups seen this recently are forecast; older ones may be anywhere.
ACTIVE_DAYS = 3
MAX_HORIZON_DAYS = 7
LIKELY_MASS = 0.5
POSSIBLE_MASS = 0.8
GRID_MARGIN_KM = 15.0
# Past forecasts needed before the zones are calibrated on them.
MIN_CALIBRATION = 40
_TINY = 1e-12


# Cells sit on one lattice for the whole country (1 km squares counted from
# the projection's origin), and the projection's reference latitude is the
# middle of the degree the reports fall in. So adding reports somewhere
# else -- a new month, a wider area -- never moves a cell, and the months
# already scored stay valid (see ``train``).
_KEY_STRIDE = 10 ** 8
# A place with no reports keeps this much usage per 100 x 100 km.
FLOOR_AREA_KM2 = 10_000.0


@dataclass
class Local:
    """The cells within reach of one group: lattice keys, grid index, centres, distances."""

    keys: np.ndarray       # ascending; comparable across grids
    flat: np.ndarray       # index into the grid's arrays, -1 outside it
    centres: np.ndarray    # km plane
    distance: np.ndarray   # from the group's last report


@dataclass
class Grid:
    """Square cells over the landscape, in the ``km_plane`` projection."""

    ref_lat: float
    x0: float
    y0: float
    nx: int
    ny: int
    cell_km: float = CELL_KM
    centres: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell_km
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.cell_km
        gx, gy = np.meshgrid(xs, ys)
        self.centres = np.column_stack([gx.ravel(), gy.ravel()])
        self._ix0 = int(np.floor(self.x0 / self.cell_km + 1e-9))
        self._iy0 = int(np.floor(self.y0 / self.cell_km + 1e-9))

    @classmethod
    def around(cls, df: pd.DataFrame, cell_km: float = CELL_KM,
               margin_km: float = GRID_MARGIN_KM) -> "Grid":
        """The reports' extent plus a margin, snapped to the national lattice."""
        ref_lat = float(np.floor(df["Latitude"].mean())) + 0.5
        xy = km_plane(df["Latitude"].values, df["Longitude"].values, ref_lat)
        x0, y0 = np.floor((xy.min(axis=0) - margin_km) / cell_km) * cell_km
        x1, y1 = xy.max(axis=0) + margin_km
        return cls(ref_lat, float(x0), float(y0), int(np.ceil((x1 - x0) / cell_km)),
                   int(np.ceil((y1 - y0) / cell_km)), cell_km)

    def project(self, lat, lon) -> np.ndarray:
        return km_plane(np.atleast_1d(lat), np.atleast_1d(lon), self.ref_lat)

    def cell_of(self, xy: np.ndarray) -> np.ndarray:
        """Flat cell index of each point; -1 outside the grid."""
        ix = np.floor((xy[:, 0] - self.x0) / self.cell_km).astype(int)
        iy = np.floor((xy[:, 1] - self.y0) / self.cell_km).astype(int)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.where(inside, iy * self.nx + ix, -1)

    def key_of(self, xy: np.ndarray) -> np.ndarray:
        """Lattice key of each point: the same cell gets the same key on any grid."""
        ix = np.floor(xy[:, 0] / self.cell_km + 1e-9).astype(np.int64)
        iy = np.floor(xy[:, 1] / self.cell_km + 1e-9).astype(np.int64)
        return iy * _KEY_STRIDE + ix

    def near(self, xy: np.ndarray, radius_km: float) -> Local:
        """Cells within ``radius_km`` of ``xy``, whether or not the grid covers them.

        A forecast only needs the cells a group could reach, which is a few
        thousand rather than the landscape's tens of thousands; and because
        they are taken from the lattice rather than the grid, the forecast
        does not change when the grid grows.
        """
        c = self.cell_km
        ix = np.arange(int(np.floor((xy[0] - radius_km) / c)),
                       int(np.floor((xy[0] + radius_km) / c)) + 1)
        iy = np.arange(int(np.floor((xy[1] - radius_km) / c)),
                       int(np.floor((xy[1] + radius_km) / c)) + 1)
        gx, gy = np.meshgrid(ix, iy)
        gx, gy = gx.ravel(), gy.ravel()
        centres = np.column_stack([(gx + 0.5) * c, (gy + 0.5) * c])
        offset = centres - xy
        distance = np.hypot(offset[:, 0], offset[:, 1])
        keep = distance <= radius_km
        gx, gy, centres, distance = gx[keep], gy[keep], centres[keep], distance[keep]
        lx, ly = gx - self._ix0, gy - self._iy0
        inside = (lx >= 0) & (lx < self.nx) & (ly >= 0) & (ly < self.ny)
        flat = np.where(inside, ly * self.nx + lx, -1)
        return Local(gy.astype(np.int64) * _KEY_STRIDE + gx, flat, centres, distance)

    def floor(self) -> float:
        """Usage left on a cell nobody has reported elephants near."""
        return USAGE_FLOOR * self.cell_km ** 2 / FLOOR_AREA_KM2

    def at(self, values: np.ndarray, local: Local) -> np.ndarray:
        """A grid-wide map's values on ``local``'s cells; the floor off the grid."""
        out = np.full(len(local.flat), self.floor())
        inside = local.flat >= 0
        out[inside] = values[local.flat[inside]]
        return out

    def latlon(self, index: Optional[np.ndarray] = None) -> np.ndarray:
        centres = self.centres if index is None else self.centres[index]
        return from_plane(centres, self.ref_lat)


def _density(points_xy: np.ndarray, weight: np.ndarray, grid: Grid,
             bandwidth_km: float) -> np.ndarray:
    """Share of the weighted reports per cell, smoothed, plus the floor.

    Zero padding at the edges and an absolute floor keep a cell's value
    independent of how far the grid happens to reach.
    """
    from scipy.ndimage import gaussian_filter

    density = np.zeros(grid.nx * grid.ny)
    if len(points_xy) and weight.sum() > 0:
        cells = grid.cell_of(points_xy)
        keep = cells >= 0
        np.add.at(density, cells[keep], weight[keep])
        density = gaussian_filter(density.reshape(grid.ny, grid.nx),
                                  sigma=bandwidth_km / grid.cell_km, mode="constant").ravel()
        density /= weight.sum()
    return density + grid.floor()


def usage_prior(df: pd.DataFrame, grid: Grid, as_of: pd.Timestamp,
                half_life_days: float = USAGE_HALF_LIFE_DAYS,
                bandwidth_km: float = USAGE_BANDWIDTH_KM) -> np.ndarray:
    """Where elephants have been reported before ``as_of``, recent reports weighted more.

    Only reports strictly before ``as_of`` count, so a forecast never sees
    the report it is about to be scored on.
    """
    past = df[df["Date"] < as_of]
    age = (as_of - past["Date"]).dt.total_seconds().to_numpy() / 86400.0
    return _density(grid.project(past["Latitude"].values, past["Longitude"].values),
                    0.5 ** (age / half_life_days), grid, bandwidth_km)


def season_prior(df: pd.DataFrame, grid: Grid, as_of: pd.Timestamp,
                 recent: np.ndarray) -> Optional[np.ndarray]:
    """The usage map with the same weeks of earlier years counted as well.

    Raiding follows the harvest, so where groups were this time last year
    can say more than where they were last month. Half recent, half the
    same season; None while the register holds no earlier year.
    """
    past = df[df["Date"] < as_of - pd.Timedelta(days=SEASON_MIN_AGE_DAYS)]
    if past.empty:
        return None
    gap = np.abs(past["Date"].dt.dayofyear.to_numpy() - as_of.dayofyear)
    gap = np.minimum(gap, 365 - gap)
    same = past[gap <= SEASON_HALF_WIDTH_DAYS]
    if len(same) < MIN_STEPS:
        return None
    seasonal = _density(grid.project(same["Latitude"].values, same["Longitude"].values),
                        np.ones(len(same)), grid, USAGE_BANDWIDTH_KM)
    return 0.5 * recent + 0.5 * seasonal


def _angle(dx, dy):
    return np.arctan2(dy, dx)


def steps(df: pd.DataFrame, units: pd.Series) -> pd.DataFrame:
    """Every move between consecutive reports of a unit, a day or more apart.

    Each row also carries the group's class and the direction of the move
    before it, which is what the direction-of-travel model learns from.
    """
    columns = ["Unit", "Class", "From Date", "To Date", "From Latitude", "From Longitude",
               "To Latitude", "To Longitude", "Gap (days)", "Distance (km)", "Angle",
               "Prev Angle", "Prev Distance (km)"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    work = df.assign(_unit=units.values).sort_values(["_unit", "Date"], kind="stable")
    lat = work["Latitude"].to_numpy(dtype=float)
    lon = work["Longitude"].to_numpy(dtype=float)
    same = work["_unit"].values[1:] == work["_unit"].values[:-1]
    gap = (work["Date"].values[1:] - work["Date"].values[:-1]).astype("timedelta64[D]")
    gap = gap.astype(int)
    keep = same & (gap >= 1)
    delta = np.column_stack(pair_km(lat[:-1], lon[:-1], lat[1:], lon[1:]))
    origin = work.iloc[:-1][keep]
    target = work.iloc[1:][keep]
    social = [r["social"] for r in _records(origin)]
    table = pd.DataFrame({
        "Unit": origin["_unit"].to_numpy(),
        "Class": social,
        "From Date": origin["Date"].to_numpy(),
        "To Date": target["Date"].to_numpy(),
        "From Latitude": origin["Latitude"].to_numpy(),
        "From Longitude": origin["Longitude"].to_numpy(),
        "To Latitude": target["Latitude"].to_numpy(),
        "To Longitude": target["Longitude"].to_numpy(),
        "Gap (days)": gap[keep],
        "Distance (km)": np.hypot(delta[keep, 0], delta[keep, 1]),
        "Angle": _angle(delta[keep, 0], delta[keep, 1]),
    })
    # The move that brought the group to this step's start, if it was one.
    previous = table.groupby("Unit")
    table["Prev Angle"] = previous["Angle"].shift(1)
    table["Prev Distance (km)"] = previous["Distance (km)"].shift(1)
    joined = previous["To Date"].shift(1) == table["From Date"]
    table.loc[~joined, ["Prev Angle", "Prev Distance (km)"]] = np.nan
    return table[columns]


def step_lengths(step_table: pd.DataFrame) -> np.ndarray:
    """Observed distances scaled to one day (distance / sqrt(days)).

    Square-root scaling treats a multi-day gap as several daily moves in
    loosely related directions, which is closer to what tracked elephants
    do than straight-line travel at a constant speed.
    """
    if step_table.empty:
        return np.array([])
    return (step_table["Distance (km)"] / np.sqrt(step_table["Gap (days)"])).to_numpy()


def radial_profile(per_day: np.ndarray, days: float, cell_km: float = CELL_KM):
    """Density per square km of being ``r`` km from the start after ``days``, as a lookup table.

    Built once per horizon and interpolated for every cell, rather than
    smoothing every observed step against every cell.
    """
    days = max(days, 1.0)
    if len(per_day) >= MIN_STEPS:
        sample = per_day * np.sqrt(days)
        bandwidth = max(1.06 * sample.std() * len(sample) ** -0.2, cell_km)
        r = np.linspace(0, sample.max() + 4 * bandwidth, 400)
        # Kernel density of the step length, reflected at zero.
        z = np.exp(-0.5 * ((r[:, None] - sample[None, :]) / bandwidth) ** 2) + \
            np.exp(-0.5 * ((r[:, None] + sample[None, :]) / bandwidth) ** 2)
        radial = z.mean(axis=1)
    else:
        # Too few observed steps: a half-normal with the typical daily move.
        scale = DEFAULT_KM_PER_DAY * np.sqrt(days) / 0.6745
        r = np.linspace(0, 4 * scale, 400)
        radial = np.exp(-0.5 * (r / scale) ** 2)
    # Per unit area: a ring at distance r has area 2*pi*r*dr.
    return r, radial / (2 * np.pi * np.maximum(r, cell_km / 2))


def _kappa(resultant: float) -> float:
    """Von Mises concentration from the mean resultant length (Fisher's approximation)."""
    r = float(resultant)
    if r < 0.53:
        k = 2 * r + r ** 3 + 5 * r ** 5 / 6
    elif r < 0.85:
        k = -0.4 + 1.39 * r + 0.43 / (1 - r)
    else:
        k = 1 / max(r ** 3 - 4 * r ** 2 + 3 * r, 1e-6)
    return float(min(k, MAX_KAPPA))


@dataclass
class StepModel:
    """How far groups go in a day, by class, and which way after a known move."""

    per_day: Dict[str, np.ndarray]
    # Direction of the next move relative to the last one, by gap (1, 2, 3+
    # days): mean turn and concentration. A turn near 0 is steady travel,
    # near pi a return the way it came -- refuge to field and back.
    turns: Dict[int, tuple]
    _profiles: Dict[tuple, tuple] = field(default_factory=dict, repr=False)

    @classmethod
    def learn(cls, step_table: pd.DataFrame) -> "StepModel":
        per_day = {"all": step_lengths(step_table)}
        if not step_table.empty:
            for social, group in step_table.groupby("Class"):
                if len(group) >= MIN_CLASS_STEPS:
                    per_day[social] = step_lengths(group)
        turns: Dict[int, tuple] = {}
        usable = step_table.dropna(subset=["Prev Angle"]) if len(step_table) else step_table
        if len(usable):
            usable = usable[(usable["Distance (km)"] >= MIN_MOVE_KM)
                            & (usable["Prev Distance (km)"] >= MIN_MOVE_KM)]
        for bucket in (1, 2, 3):
            part = usable[usable["Gap (days)"].clip(upper=3) == bucket] if len(usable) else usable
            if len(part) < MIN_TURNS:
                part = usable
            if len(part) < MIN_TURNS:
                continue
            turn = part["Angle"].to_numpy() - part["Prev Angle"].to_numpy()
            c, s = np.cos(turn).mean(), np.sin(turn).mean()
            turns[bucket] = (float(np.arctan2(s, c)), _kappa(np.hypot(c, s)))
        return cls(per_day, turns)

    def lengths(self, social: str, use_class: bool = True) -> np.ndarray:
        if use_class and social in self.per_day:
            return self.per_day[social]
        return self.per_day["all"]

    def profile(self, social: str, days: float, cell_km: float):
        key = (social if social in self.per_day else "all", int(days), cell_km)
        if key not in self._profiles:
            self._profiles[key] = radial_profile(self.lengths(social), days, cell_km)
        return self._profiles[key]

    def turn(self, days: float) -> Optional[tuple]:
        return self.turns.get(int(min(max(days, 1), 3)))


@dataclass(frozen=True)
class Config:
    """Which parts of the forecast are switched on, and how strongly."""

    places: float = 0.0      # the usage map: 0 ignored, 1 taken at face value
    season: bool = False     # the usage map also counts the same weeks of earlier years
    heading: bool = False    # direction relative to the group's last move
    own: bool = False        # the group's own earlier reports

    def parts(self) -> List[str]:
        out = []
        if self.places:
            out.append("places elephants used before"
                       + (" (same season counted)" if self.season else ""))
        if self.heading:
            out.append("the group's direction of travel")
        if self.own:
            out.append("the group's own range")
        return out


BASELINE = Config()
DEFAULT_CONFIG = Config(places=0.5)


def configs(with_season: bool) -> List[Config]:
    """Every combination the backtest chooses from."""
    out = []
    for places in (0.0, 0.5, 1.0):
        for season in ((False, True) if places and with_season else (False,)):
            for heading in (False, True):
                for own in (False, True):
                    out.append(Config(places, season, heading, own))
    return out


@dataclass
class Situation:
    """What is known about a group at the moment of forecasting."""

    xy: np.ndarray
    days: float
    social: str
    prev_angle: Optional[float] = None
    own_xy: Optional[np.ndarray] = None   # its earlier reports, km plane


def _components(s: Situation, grid: Grid, model: StepModel, priors: Dict[str, np.ndarray]):
    """Each part of the forecast as a log-density over the cells within reach."""
    r, density = model.profile(s.social, s.days, grid.cell_km)
    local = grid.near(s.xy, float(r[-1]))
    parts = {"reach": np.log(np.interp(local.distance, r, density) + _TINY)}
    for name, prior in priors.items():
        parts[name] = np.log(grid.at(prior, local))
    turn = model.turn(s.days)
    if turn is not None and s.prev_angle is not None and np.isfinite(s.prev_angle):
        mu, kappa = turn
        offset = local.centres - s.xy
        theta = np.arctan2(offset[:, 1], offset[:, 0])
        heading = kappa * np.cos(theta - s.prev_angle - mu)
        # The cell the group stands in has no direction: give it the average.
        heading[local.distance < grid.cell_km / 2] = np.log(np.i0(kappa))
        parts["heading"] = heading
    if s.own_xy is not None and len(s.own_xy):
        offset = local.centres[:, None, :] - s.own_xy[None, :, :]
        d2 = (offset ** 2).sum(axis=2)
        own = np.exp(-0.5 * d2 / OWN_BANDWIDTH_KM ** 2).sum(axis=1)
        own = own + USAGE_FLOOR * (own.mean() if own.any() else 1.0)
        parts["own"] = np.log(own)
    return local, parts


def _combine(parts: Dict[str, np.ndarray], config: Config) -> np.ndarray:
    log_p = parts["reach"].copy()
    if config.places:
        key = "season" if config.season and "season" in parts else "places"
        log_p += config.places * parts[key]
    if config.heading and "heading" in parts:
        log_p += parts["heading"]
    if config.own and "own" in parts:
        log_p += parts["own"]
    log_p -= log_p.max()
    p = np.exp(log_p)
    return p / p.sum()


def zone(prob: np.ndarray, mass: float = LIKELY_MASS) -> np.ndarray:
    """The fewest cells that together hold ``mass`` of the probability."""
    order = np.argsort(prob)[::-1]
    upto = np.searchsorted(np.cumsum(prob[order]), min(mass, 1.0)) + 1
    mask = np.zeros(len(prob), dtype=bool)
    mask[order[:upto]] = True
    return mask


def _mass_to(prob: np.ndarray, at: int) -> float:
    """How much probability a zone must hold to include cell ``at`` (1 when outside)."""
    if at < 0:
        return 1.0
    return float(prob[prob >= prob[at]].sum())


def calibrated_mass(scores: List[float], target: float) -> float:
    """The zone mass that held the next report ``target`` of the time in past forecasts.

    A split-conformal quantile: if past zones of half the forecast caught
    the group only 40% of the time, the likely zone is widened until it
    would have caught half. Nominal until enough forecasts have been
    scored.
    """
    if len(scores) < MIN_CALIBRATION:
        return target
    n = len(scores)
    return float(np.quantile(scores, min(1.0, target * (n + 1) / n)))


# ---------------------------------------------------------------------------
# Track record
# ---------------------------------------------------------------------------
@dataclass
class TrackRecord:
    """How the forecast has done on this register, forecasting each month from the past."""

    tested: int
    forecast_hits: int          # next report inside the likely zone
    stay_hits: int              # ... inside a same-size circle around the last report
    mean_area_km2: float        # the likely zone's average area
    months: int
    horizon_days: int
    discordant: int = 0
    # A circle around the last report, sized from past months to catch half
    # the next reports: how often it did, and how big it had to be.
    circle_hits: int = 0
    circle_area_km2: float = float("nan")
    config: Config = DEFAULT_CONFIG
    masses: tuple = (LIKELY_MASS, POSSIBLE_MASS)
    # Average log score per forecast for each configuration (higher is better).
    log_scores: Dict[Config, float] = field(default_factory=dict)

    @property
    def forecast_rate(self) -> float:
        return self.forecast_hits / self.tested if self.tested else float("nan")

    @property
    def stay_rate(self) -> float:
        return self.stay_hits / self.tested if self.tested else float("nan")

    @property
    def circle_rate(self) -> float:
        return self.circle_hits / self.tested if self.tested else float("nan")

    @property
    def beats_staying_put(self) -> bool:
        """Ahead of the same-size circle by more than chance, at about 90% one-sided."""
        if self.tested < 20:
            return False
        diff = self.forecast_hits - self.stay_hits
        # Paired binomial: only discordant pairs carry information.
        return diff > 1.28 * np.sqrt(max(self.discordant, 1))

    @property
    def gain_bits(self) -> float:
        """How much better the chosen forecast scores than reach alone, in bits per forecast."""
        if self.config not in self.log_scores or BASELINE not in self.log_scores:
            return float("nan")
        return (self.log_scores[self.config] - self.log_scores[BASELINE]) / np.log(2)


class _History:
    """Each unit's reports in date order, to look up its range and last move."""

    def __init__(self, df: pd.DataFrame, units: pd.Series, grid: Grid):
        work = df.assign(_unit=units.values).sort_values("Date", kind="stable")
        xy = grid.project(work["Latitude"].values, work["Longitude"].values)
        self._by_unit: Dict[int, tuple] = {}
        for unit, rows in work.groupby("_unit").indices.items():
            self._by_unit[int(unit)] = (work["Date"].values[rows], xy[rows])

    def own(self, unit: int, before) -> Optional[np.ndarray]:
        """The unit's reports before ``before``, the most recent ``OWN_MAX_REPORTS``."""
        dates, xy = self._by_unit.get(int(unit), (None, None))
        if dates is None:
            return None
        return xy[dates < np.datetime64(before)][-OWN_MAX_REPORTS:]


def _situations(table: pd.DataFrame, grid: Grid, history: _History) -> List[Situation]:
    origin = grid.project(table["From Latitude"].values, table["From Longitude"].values)
    prev = table["Prev Angle"].to_numpy(dtype=float)
    prev_km = table["Prev Distance (km)"].to_numpy(dtype=float)
    out = []
    for i, (unit, social, start, days) in enumerate(zip(
            table["Unit"].to_numpy(), table["Class"].to_numpy(),
            table["From Date"].to_numpy(), table["Gap (days)"].to_numpy())):
        usable = np.isfinite(prev[i]) and prev_km[i] >= MIN_MOVE_KM
        out.append(Situation(origin[i], float(days), str(social),
                             float(prev[i]) if usable else None, history.own(unit, start)))
    return out


# ---------------------------------------------------------------------------
# Training: month by month, and only the months that changed
# ---------------------------------------------------------------------------
# Bump when the method changes, so a training made by older code is never
# reused by newer code.
TRAINING_VERSION = 1
MIN_HISTORY_DAYS = 60


@dataclass
class MonthResult:
    """One month forecast from the months before it, kept so it need not be redone."""

    month: str
    fingerprint: str
    scores: Dict[Config, float]           # summed log score per configuration
    masses: Dict[Config, List[float]]     # mass needed to reach each next report
    reach: List[float]                    # next-report distance / sqrt(days)
    tested: int = 0
    hits: int = 0
    stay_hits: int = 0
    discordant: int = 0
    area_sum: float = 0.0
    circle_hits: int = 0
    circle_tested: int = 0
    circle_area_sum: float = 0.0


@dataclass
class Training:
    """The forecast trained on a register: its months, what it chose, how it did.

    Hand a training back to ``train`` with a newer register and only the
    months whose reports changed -- new entries, or edits to old ones --
    are forecast again; the rest are carried over unchanged.
    """

    horizon_days: int
    version: int
    with_season: bool
    months: List[MonthResult]
    record: "TrackRecord"
    reports: int = 0
    first_report: Optional[pd.Timestamp] = None
    last_report: Optional[pd.Timestamp] = None
    moves: int = 0
    reused_months: int = 0
    data_key: str = ""

    @property
    def retrained_months(self) -> int:
        return len(self.months) - self.reused_months

    # Saved as JSON, never pickle: a saved training is data, and loading one
    # must not be able to run anything. It holds scores and settings only.
    def to_dict(self) -> dict:
        def key(c: Config) -> str:
            return f"{c.places:g}|{int(c.season)}|{int(c.heading)}|{int(c.own)}"
        return {
            "horizon_days": int(self.horizon_days), "version": int(self.version),
            "with_season": bool(self.with_season), "reports": int(self.reports),
            "first_report": str(self.first_report) if self.first_report is not None else None,
            "last_report": str(self.last_report) if self.last_report is not None else None,
            "moves": int(self.moves), "data_key": self.data_key,
            "months": [{
                "month": m.month, "fingerprint": m.fingerprint,
                "scores": {key(c): float(v) for c, v in m.scores.items()},
                "masses": {key(c): [float(x) for x in v] for c, v in m.masses.items()},
                "reach": [float(x) for x in m.reach], "tested": int(m.tested),
                "hits": int(m.hits), "stay_hits": int(m.stay_hits),
                "discordant": int(m.discordant), "area_sum": float(m.area_sum),
                "circle_hits": int(m.circle_hits), "circle_tested": int(m.circle_tested),
                "circle_area_sum": float(m.circle_area_sum),
            } for m in self.months],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Training":
        def config(text: str) -> Config:
            places, season, heading, own = text.split("|")
            return Config(float(places), bool(int(season)), bool(int(heading)),
                          bool(int(own)))
        months = [MonthResult(
            str(m["month"]), str(m["fingerprint"]),
            {config(k): float(v) for k, v in m["scores"].items()},
            {config(k): [float(x) for x in v] for k, v in m["masses"].items()},
            [float(x) for x in m["reach"]], int(m["tested"]), int(m["hits"]),
            int(m["stay_hits"]), int(m["discordant"]), float(m["area_sum"]),
            int(m["circle_hits"]), int(m["circle_tested"]), float(m["circle_area_sum"]),
        ) for m in data["months"]]
        with_season = bool(data["with_season"])
        horizon = int(data["horizon_days"])
        stamp = (lambda v: pd.Timestamp(v) if v else None)
        return cls(horizon, int(data["version"]), with_season, months,
                   _summarise(months, configs(with_season), horizon),
                   reports=int(data["reports"]), first_report=stamp(data["first_report"]),
                   last_report=stamp(data["last_report"]), moves=int(data["moves"]),
                   reused_months=len(months), data_key=str(data["data_key"]))


def _fingerprints(df: pd.DataFrame, cutoffs: List[pd.Timestamp], salt: str = "") -> List[str]:
    """A digest of every report up to each cutoff, in the order the tracker reads them.

    Two registers share a month's fingerprint only if every report the
    month's forecasts and scores depend on is identical, including the
    order of same-day reports, which the tracker is sensitive to.
    """
    import hashlib

    order = df.sort_values("Date", kind="stable")
    counts = order.reindex(columns=["Total Count", "Male Count", "Female Count",
                                    "Calf Count"]).apply(pd.to_numeric, errors="coerce")
    table = np.column_stack([
        order["Date"].values.astype("datetime64[ns]").astype(np.int64).astype(np.float64),
        order["Latitude"].to_numpy(dtype=float), order["Longitude"].to_numpy(dtype=float),
        counts.fillna(-1).to_numpy(dtype=float)])
    dates = order["Date"].values
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"v{TRAINING_VERSION}|{salt}".encode())
    out, done = [], 0
    for cutoff in cutoffs:
        upto = int(np.searchsorted(dates, np.datetime64(cutoff), side="right"))
        digest.update(table[done:upto].tobytes())
        done = upto
        out.append(digest.copy().hexdigest())
    return out


def _months(df: pd.DataFrame, min_history_days: int) -> pd.PeriodIndex:
    start = (df["Date"].min() + pd.Timedelta(days=min_history_days)).to_period("M").start_time
    return pd.period_range(start, df["Date"].max(), freq="M")


def _reusable(previous, months, prints, horizon_days, with_season) -> List[MonthResult]:
    """The longest run of leading months a previous training already did on identical data."""
    best: List[MonthResult] = []
    for candidate in previous or []:
        if (candidate is None or candidate.version != TRAINING_VERSION
                or candidate.horizon_days != horizon_days
                or candidate.with_season != with_season):
            continue
        run: List[MonthResult] = []
        for done, month, digest in zip(candidate.months, months, prints):
            if done.month != str(month) or done.fingerprint != digest:
                break
            run.append(done)
        if len(run) > len(best):
            best = run
    return best


def _summarise(months: List[MonthResult], candidates: List[Config],
               horizon_days: int) -> "TrackRecord":
    tested = sum(m.tested for m in months)
    if not months or not tested:
        return TrackRecord(0, 0, 0, float("nan"), len(months), horizon_days)
    scores = {c: sum(m.scores.get(c, 0.0) for m in months) for c in candidates}
    masses: Dict[Config, List[float]] = {c: [] for c in candidates}
    for m in months:
        for c in candidates:
            masses[c].extend(m.masses.get(c, []))
    best = max(scores, key=scores.get)
    circle_tested = sum(m.circle_tested for m in months)
    circle_hits = sum(m.circle_hits for m in months)
    return TrackRecord(
        tested, sum(m.hits for m in months), sum(m.stay_hits for m in months),
        sum(m.area_sum for m in months) / tested, sum(1 for m in months if m.tested),
        horizon_days,
        sum(m.discordant for m in months),
        # The calibrated circle's hit count over the forecasts it covered,
        # scaled to all of them so the two rates share a denominator.
        int(round(circle_hits * tested / circle_tested)) if circle_tested else 0,
        (sum(m.circle_area_sum for m in months) / circle_tested) if circle_tested
        else float("nan"),
        best,
        (calibrated_mass(masses[best], LIKELY_MASS),
         calibrated_mass(masses[best], POSSIBLE_MASS)),
        {c: v / tested for c, v in scores.items()},
    )


def train(df: pd.DataFrame, horizon_days: int = 1, previous=None,
          min_history_days: int = MIN_HISTORY_DAYS) -> Training:
    """Train the forecast on the register, month by month.

    Each month is forecast from the months before it only: units, step
    lengths, turns, the usage maps, which parts to switch on (the best log
    score over the earlier months) and the zone sizes (calibrated on the
    earlier months) are all built from reports before the month began,
    and the forecasts are scored against where each group was reported
    next. Nothing from the month being scored leaks in.

    ``previous`` is an earlier ``Training`` (or a list of them). Months it
    already scored on identical reports are carried over, so a register
    that gained a day of entries is trained again in the time it takes to
    score that day's month; an edit to an old entry retrains from that
    entry's month on. The result is the same as training from scratch.
    """
    if isinstance(previous, Training):
        previous = [previous]
    if df.empty or (df["Date"].max() - df["Date"].min()).days < min_history_days + 14:
        return Training(horizon_days, TRAINING_VERSION, False, [],
                        TrackRecord(0, 0, 0, float("nan"), 0, horizon_days),
                        reports=len(df),
                        first_report=df["Date"].min() if len(df) else None,
                        last_report=df["Date"].max() if len(df) else None,
                        data_key=_fingerprints(df, [df["Date"].max()])[0] if len(df) else "")
    with_season = (df["Date"].max() - df["Date"].min()).days > SEASON_MIN_AGE_DAYS + 60
    candidates = configs(with_season)
    months = _months(df, min_history_days)
    # A month's results depend on every report up to its last day plus the
    # horizon: the next report that scores a forecast made on its last day.
    cutoffs = [m.end_time + pd.Timedelta(days=horizon_days) for m in months]
    grid = Grid.around(df)
    # The projection's reference latitude shapes every forecast; a register
    # whose reports moved to another degree of latitude starts afresh.
    prints = _fingerprints(df, cutoffs + [df["Date"].max()],
                           salt=f"{grid.ref_lat:g}|{grid.cell_km:g}")
    data_key, prints = prints[-1], prints[:-1]
    # The very same register, already trained: nothing to do.
    for candidate in previous or []:
        if (candidate is not None and candidate.data_key == data_key
                and candidate.version == TRAINING_VERSION
                and candidate.horizon_days == horizon_days
                and candidate.with_season == with_season):
            return replace(candidate, reused_months=len(candidate.months))
    kept = _reusable(previous, months, prints, horizon_days, with_season)

    units_all = assign_units(df)
    all_steps = steps(df, units_all)
    history = _History(df, units_all, grid)

    scores = {c: 0.0 for c in candidates}
    masses: Dict[Config, List[float]] = {c: [] for c in candidates}
    reach_scores: List[float] = []
    results: List[MonthResult] = []
    for done in kept:
        results.append(done)
        for c in candidates:
            scores[c] += done.scores.get(c, 0.0)
            masses[c].extend(done.masses.get(c, []))
        reach_scores.extend(done.reach)

    for month, digest in zip(months[len(kept):], prints[len(kept):]):
        begin, end = month.start_time, month.end_time
        result = MonthResult(str(month), digest, {c: 0.0 for c in candidates},
                             {c: [] for c in candidates}, [])
        tests = all_steps[(all_steps["From Date"] >= begin) & (all_steps["From Date"] <= end)
                          & (all_steps["Gap (days)"] <= horizon_days)]
        scored_before = any(m.tested for m in results)
        if not tests.empty:
            # The tracker only ever looks back, so the units it gives the
            # reports before ``begin`` are the same whether or not later
            # reports exist: the steps finished before the month are the
            # ones a forecast made then could have learned from.
            model = StepModel.learn(all_steps[all_steps["To Date"] < begin])
            recent = usage_prior(df, grid, begin)
            priors = {"places": recent}
            seasonal = season_prior(df, grid, begin + pd.Timedelta(days=14), recent) \
                if with_season else None
            if seasonal is not None:
                priors["season"] = seasonal
            chosen = max(scores, key=scores.get) if scored_before else DEFAULT_CONFIG
            likely_mass = calibrated_mass(masses[chosen], LIKELY_MASS)
            circle_r = (float(np.quantile(reach_scores, 0.5)) if len(reach_scores)
                        >= MIN_CALIBRATION else None)
            targets = grid.project(tests["To Latitude"].values, tests["To Longitude"].values)
            target_keys = grid.key_of(targets)
            for situation, target, key in zip(_situations(tests, grid, history), targets,
                                              target_keys):
                local, parts = _components(situation, grid, model, priors)
                at = int(np.searchsorted(local.keys, key))
                at = at if at < len(local.keys) and local.keys[at] == key else -1
                probs = {}
                for c in candidates:
                    p = _combine(parts, c)
                    probs[c] = p
                    result.scores[c] += float(np.log(p[at] + _TINY)) if at >= 0 \
                        else float(np.log(_TINY))
                    result.masses[c].append(_mass_to(p, at))
                moved = float(np.hypot(*(target - situation.xy)))
                result.reach.append(moved / np.sqrt(situation.days))

                likely = zone(probs[chosen], likely_mass)
                area = likely.sum() * grid.cell_km ** 2
                hit = at >= 0 and bool(likely[at])
                # The same area as a disc around where the group was last seen.
                stay = moved <= np.sqrt(area / np.pi)
                result.tested += 1
                result.hits += hit
                result.stay_hits += stay
                result.discordant += hit != stay
                result.area_sum += area
                if circle_r is not None:
                    radius = circle_r * np.sqrt(situation.days)
                    result.circle_hits += moved <= radius
                    result.circle_tested += 1
                    result.circle_area_sum += np.pi * radius ** 2
        results.append(result)
        for c in candidates:
            scores[c] += result.scores[c]
            masses[c].extend(result.masses[c])
        reach_scores.extend(result.reach)

    return Training(horizon_days, TRAINING_VERSION, with_season, results,
                    _summarise(results, candidates, horizon_days),
                    reports=len(df), first_report=df["Date"].min(),
                    last_report=df["Date"].max(), moves=len(all_steps),
                    reused_months=len(kept), data_key=data_key)


def backtest(df: pd.DataFrame, horizon_days: int = 1,
             min_history_days: int = MIN_HISTORY_DAYS) -> "TrackRecord":
    """The track record of a forecast trained from scratch on ``df``."""
    return train(df, horizon_days, min_history_days=min_history_days).record


# ---------------------------------------------------------------------------
# The groups to forecast now
# ---------------------------------------------------------------------------
SOCIAL_LABELS = {"lone bull": "Lone bull", "bull party": "Bull party",
                 "family herd": "Family herd", "unsexed group": "Group, sex not recorded"}
COMPASS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def compass(angle: float) -> str:
    """A plane angle (counter-clockwise from east) as one of eight compass points."""
    return COMPASS[int(np.round(angle / (np.pi / 4))) % 8]


def next_moves(df: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int = 1,
               villages: Optional[pd.DataFrame] = None,
               active_days: int = ACTIVE_DAYS,
               record: Optional[TrackRecord] = None) -> Dict[str, object]:
    """Groups seen in the ``active_days`` before ``as_of``, and where they may be next.

    Uses the configuration and calibrated zone masses from ``record`` (the
    register's own backtest); without one, the default and nominal masses.

    Returns ``groups`` (one row per unit: heading, likely zone, most likely
    spot, villages in it), ``cells`` (the zones with the combined chance,
    for the map), ``villages`` (villages in any likely or possible zone,
    likely first) and ``trails`` (each group's recent reports).
    """
    config = record.config if record is not None else DEFAULT_CONFIG
    likely_mass, possible_mass = record.masses if record is not None else (
        LIKELY_MASS, POSSIBLE_MASS)
    out = {"groups": pd.DataFrame(), "cells": pd.DataFrame(), "villages": pd.DataFrame(),
           "trails": pd.DataFrame(), "steps_learned": 0, "config": config,
           "masses": (likely_mass, possible_mass), "cell_km": CELL_KM}
    history = df[df["Date"] < as_of.normalize() + pd.Timedelta(days=1)]
    if history.empty:
        return out
    units = assign_units(history)
    learned = steps(history, units)
    model = StepModel.learn(learned)
    out["steps_learned"] = len(learned)
    out["turns"] = model.turns
    work = history.assign(_unit=units.values)
    last = work.sort_values("Date", kind="stable").groupby("_unit").tail(1)
    active = last[last["Date"] >= as_of.normalize() - pd.Timedelta(days=active_days - 1)]
    if active.empty:
        return out

    grid = Grid.around(history)
    tracks = _History(history, units, grid)
    recent = usage_prior(history, grid, as_of.normalize() + pd.Timedelta(days=1))
    priors = {"places": recent}
    seasonal = season_prior(history, grid, as_of, recent) if config.season else None
    if seasonal is not None:
        priors["season"] = seasonal
    located = village_xy = None
    if villages is not None and len(villages):
        located = villages.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
        village_xy = grid.project(located["Latitude"].values, located["Longitude"].values)

    village_keys = grid.key_of(village_xy) if village_xy is not None else None
    size = grid.nx * grid.ny
    combined_miss = np.ones(size)
    likely_any = np.zeros(size, dtype=bool)
    possible_any = np.zeros(size, dtype=bool)
    village_hits: Dict[int, Dict[str, object]] = {}
    rows, trails = [], []
    for row, rec in zip(active.to_dict("records"), _records(active)):
        since = max((as_of.normalize() - pd.Timestamp(row["Date"]).normalize()).days, 0)
        days = min(since + horizon_days, MAX_HORIZON_DAYS)
        unit = int(row["_unit"])
        origin = grid.project(row["Latitude"], row["Longitude"])[0]
        unit_steps = learned[learned["Unit"] == unit]
        prev = None
        if len(unit_steps):
            final = unit_steps.iloc[-1]
            if final["To Date"] == pd.Timestamp(row["Date"]) and \
                    final["Distance (km)"] >= MIN_MOVE_KM:
                prev = float(final["Angle"])
        situation = Situation(origin, days, rec["social"], prev,
                              tracks.own(unit, pd.Timestamp(row["Date"])))
        local, parts = _components(situation, grid, model, priors)
        prob = _combine(parts, config)
        likely, possible = zone(prob, likely_mass), zone(prob, possible_mass)
        # The map covers the grid; a zone cell beyond its margin is not drawn.
        on_grid = local.flat >= 0
        likely_any[local.flat[on_grid & likely]] = True
        possible_any[local.flat[on_grid & possible]] = True
        combined_miss[local.flat[on_grid]] *= 1 - prob[on_grid]
        best_xy = local.centres[int(np.argmax(prob))]
        best_latlon = from_plane(best_xy[None, :], grid.ref_lat)[0]
        offset = best_xy - origin
        unit_rows = work[work["_unit"] == unit].sort_values("Date").tail(6)
        label = f"G{unit + 1:03d}"
        trails.append(unit_rows.assign(Group=label)[["Group", "Date", "Latitude", "Longitude"]])

        near: List[str] = []
        most_likely_near = ""
        if village_xy is not None:
            spot = np.searchsorted(local.keys, village_keys)
            spot = np.clip(spot, 0, max(len(local.keys) - 1, 0))
            inside = (len(local.keys) > 0) & (local.keys[spot] == village_keys)
            in_likely = inside & likely[spot]
            in_possible = inside & possible[spot]
            for vi in np.flatnonzero(in_possible):
                name = str(located.iloc[vi]["Village"])
                band = "Likely" if in_likely[vi] else "Possible"
                current = village_hits.get(vi)
                if current is None or (band == "Likely" and current["Band"] != "Likely"):
                    village_hits[vi] = {
                        "Village": name, "Band": band, "Group": label,
                        "Latitude": float(located.iloc[vi]["Latitude"]),
                        "Longitude": float(located.iloc[vi]["Longitude"]),
                        "From last report (km)": round(float(
                            np.hypot(*(village_xy[vi] - origin))), 1)}
                if in_likely[vi]:
                    near.append(name)
            nearest = int(np.argmin(np.hypot(*(village_xy - best_xy).T)))
            most_likely_near = str(located.iloc[nearest]["Village"])

        moved = float(np.hypot(*offset))
        rows.append({
            "Group": label,
            "Class": SOCIAL_LABELS.get(rec["social"], rec["social"]),
            "Size": int(rec["total"]) if rec["total"] else None,
            "Last Seen": pd.Timestamp(row["Date"]),
            "Days Ago": since,
            "Division": row.get("Division", ""),
            "Range": row.get("Range", ""),
            "Beat": row.get("Beat", ""),
            "Near": row.get("Nearest Village", ""),
            "Heading": compass(prev) if prev is not None else "",
            "Most Likely": ("near its last report" if moved < 1.0
                            else f"{moved:.0f} km {compass(np.arctan2(offset[1], offset[0]))}")
            + (f", by {most_likely_near}" if most_likely_near else ""),
            "Reports": int((work["_unit"] == unit).sum()),
            "Likely Zone (km2)": float(likely.sum() * grid.cell_km ** 2),
            "Villages in Likely Zone": ", ".join(sorted(set(near))[:6])
            + (f" and {len(set(near)) - 6} more" if len(set(near)) > 6 else ""),
            "Latitude": float(row["Latitude"]),
            "Longitude": float(row["Longitude"]),
            "Spot Latitude": float(best_latlon[0]),
            "Spot Longitude": float(best_latlon[1]),
        })

    shown = np.flatnonzero(possible_any)
    latlon = grid.latlon(shown)
    out["cells"] = pd.DataFrame({
        "Latitude": latlon[:, 0], "Longitude": latlon[:, 1],
        "Chance": 1 - combined_miss[shown], "Likely": likely_any[shown],
    })
    out["groups"] = pd.DataFrame(rows).sort_values(["Days Ago", "Group"]).reset_index(drop=True)
    out["trails"] = pd.concat(trails, ignore_index=True) if trails else pd.DataFrame()
    if village_hits:
        table = pd.DataFrame(village_hits.values())
        table["_order"] = (table["Band"] != "Likely").astype(int)
        out["villages"] = table.sort_values(["_order", "From last report (km)"]).drop(
            columns="_order").reset_index(drop=True)
    return out


def recent_groups(df: pd.DataFrame, as_of: pd.Timestamp,
                  active_days: int = ACTIVE_DAYS) -> int:
    """How many movement units were last reported in the ``active_days`` up to ``as_of``.

    The tracker alone, without a forecast: cheap enough for the Overview.
    """
    history = df[df["Date"] < as_of.normalize() + pd.Timedelta(days=1)]
    if history.empty:
        return 0
    last = history.assign(_unit=assign_units(history).values).groupby("_unit")["Date"].max()
    return int((last >= as_of.normalize() - pd.Timedelta(days=active_days - 1)).sum())
