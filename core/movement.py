"""Where groups seen in the last few days may be next, and how often that has been right.

The register has no collars and no identified animals, only a time, a
place and a rough count per report. So the forecast is built from parts,
each of them inspectable (``docs/MOVEMENT_REVIEW.md`` reviews them):

1. **Movement units** (``assign_units``). Day by day, the day's reports are
   grouped into sightings (several people reporting one herd) and matched
   to the open tracks all at once, the cheapest set of links: within reach
   for the time elapsed, a compatible count, and the breakdown compared
   only where both reports recorded it. A unit is a plausible sequence of
   reports of one group, not an identified animal.
2. **How far** (``StepModel``). Every consecutive pair of reports in a
   unit, a day or more apart, is one observed step, per class once a class
   has enough steps of its own, since a lone bull covers more ground than
   a family herd. How distance grows with days is learned too
   (``scaling``): a random walk goes twice as far in four days, a group
   holding its heading nearly four times.
3. **Which way** (``StepModel.turn_density``). The turn from each move to
   the next, smoothed, so the density can hold both steady travel (turns
   near zero) and a group shuttling between a day refuge and the fields it
   raids (turns near a half circle). A group's own recent turns say which
   it is likely to do.
4. **Where** (``usage_prior``, ``season_prior``). Elephants keep returning
   to the same forest blocks, nalas and fields, so the places reported
   before -- recent ones weighted more, and optionally the same weeks of
   earlier years -- form a usage map.
5. **Its own range.** A group's own earlier reports.

The forecast for a group covers every day of the horizon -- "the next
three days" is where it may be on any of them -- as the reach from its last
report, times whichever of the other parts help. Its *likely zone* is the
smallest set of 1 km cells holding half the forecast, its *possible zone*
80%.

**Every forecast comes with its own track record** (``train``). The
backtest makes the forecasts the page would have made: each morning, for
every group seen in the three days before, from what was known that
morning, scored against where the group was reported next (``_cases``).
Which parts to switch on is chosen the same way -- the combination with
the best log score on the earlier months -- and so is the size of the
zones: if zones of half the forecast caught the group only 40% of the
time, the likely zone is widened until it would have caught half (a
conformal calibration on recent forecasts). The comparison throughout is a
circle around where the group was last seen -- "they stay near where they
were". Each morning's whole map is also scored against every report of the
days after, linked or not (``_score_landscape``). If the forecast does not
beat staying put, the page says so.

What this cannot do: predict a first visit to a place nobody has reported
elephants in, follow a group through a reporting gap longer than
``MAX_GAP_DAYS``, or tell two bulls in the same valley apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

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
# How much a count difference (as a share of the tolerance) weighs against
# distance (as a share of reach) when two open tracks could take a report.
COUNT_COST = 0.3
# Reports the same day this close, and compatible, are one sighting:
# several people reporting one herd -- its spread, where each stood, and
# GPS error.
DUPLICATE_KM = 3.0
# Costs in the daily matching: no link possible, and a new track (dearer
# than any feasible link, so a report continues a track whenever one can
# take it).
_NO_LINK = 1e6
_NEW_TRACK = 1e3


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


def _tolerance(a: Dict, b: Dict) -> float:
    return max(COUNT_TOLERANCE, COUNT_TOLERANCE_FRACTION * (a["total"] + b["total"]) / 2)


def _compatible(a: Dict, b: Dict) -> bool:
    """Could these two reports be the same animals?

    Composition is compared only where both reports recorded it: a herd
    reported once with its calves counted and once as a bare total is
    still one herd. Treating "not recorded" as a kind of group of its own
    split every track at each report that left the breakdown blank.
    """
    if a["recorded"] and b["recorded"]:
        if a["social"] != b["social"]:
            return False
        # A herd does not shed or acquire calves between sightings.
        if (a["calves"] > 0) != (b["calves"] > 0):
            return False
    return abs(a["total"] - b["total"]) <= _tolerance(a, b)


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
                    # A breakdown that adds up to nothing was not recorded.
                    "recorded": bool(males + females + calves > 0),
                    "social": _class_of(float(total), float(calves), float(males),
                                        float(females))})
    return out


def _same_day_reach_km(hours: float, max_speed_km_per_day: float) -> float:
    """How far apart two reports the same day can be and still be one herd:
    the width of a herd seen by two people, plus what it can cover in the
    hours between them. Without times (both at midnight), the width alone;
    two herds 6 km apart on the same morning are two herds."""
    return min(DUPLICATE_KM + max_speed_km_per_day * hours / 24.0, SAME_DAY_RADIUS_KM)


def _link_costs(rows: np.ndarray, tracks: List[int], day: np.ndarray, lat: np.ndarray,
                lon: np.ndarray, records: List[Dict], max_speed_km_per_day: float,
                hour: Optional[np.ndarray] = None) -> np.ndarray:
    """Cost of each report in ``rows`` continuing each open track; inf where it cannot.

    Close in space (as a share of the reach for the time elapsed) first,
    then a similar count, then recent in time.
    """
    costs = np.full((len(rows), len(tracks)), np.inf)
    for a, i in enumerate(rows):
        for b, t in enumerate(tracks):
            if not _compatible(records[i], records[t]):
                continue
            gap = day[i] - day[t]
            if gap <= 0:
                hours = abs(hour[i] - hour[t]) if hour is not None else 0.0
                reach = _same_day_reach_km(hours, max_speed_km_per_day)
            else:
                reach = _reach_km(gap, max_speed_km_per_day)
            distance = float(np.hypot(*pair_km(lat[t], lon[t], lat[i], lon[i])))
            if distance > reach:
                continue
            count = abs(records[i]["total"] - records[t]["total"]) / _tolerance(
                records[i], records[t])
            costs[a, b] = distance / reach + COUNT_COST * count + 0.1 * gap
    return costs


def _sightings(rows: np.ndarray, lat: np.ndarray, lon: np.ndarray,
               records: List[Dict]) -> np.ndarray:
    """Group a day's reports into sightings: compatible reports within
    ``DUPLICATE_KM`` of each other are one herd seen by several people."""
    from scipy.sparse.csgraph import connected_components

    if len(rows) < 2:
        return np.zeros(len(rows), dtype=int)
    dx, dy = pair_km(lat[rows][:, None], lon[rows][:, None], lat[rows][None, :],
                     lon[rows][None, :])
    close = np.hypot(dx, dy) <= DUPLICATE_KM
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            if close[a, b] and not _compatible(records[rows[a]], records[rows[b]]):
                close[a, b] = close[b, a] = False
    return connected_components(close, directed=False)[1]


def _hours(frame: pd.DataFrame) -> np.ndarray:
    """Hour of day of each report: the register's Hour column where filled
    in, else the time in Date (midnight when it has none)."""
    clock = (frame["Date"].values.astype("datetime64[m]").astype(np.int64) / 60.0) % 24.0
    if "Hour" in frame.columns:
        given = pd.to_numeric(frame["Hour"], errors="coerce").to_numpy(dtype=float)
        clock = np.where(np.isfinite(given), given, clock)
    return clock


def assign_units(
    df: pd.DataFrame,
    max_speed_km_per_day: float = MAX_SPEED_KM_PER_DAY,
    max_gap_days: int = MAX_GAP_DAYS,
) -> pd.Series:
    """Tracking, one day at a time.

    Each day's reports are first grouped into sightings (several people
    reporting one herd). The day's sightings and the open tracks are then
    matched all at once, the set of links with the lowest total cost
    (the Hungarian method), each track taking at most one sighting. A
    sighting no track can take joins one already placed that day if the
    hours between them allow it (``_same_day_reach_km``) -- the herd moved
    on during the day -- or starts a new track.

    Matching the whole day at once matters where herds share a landscape:
    taking reports one by one in the order they were filed let an early
    report claim the track a later, closer one belonged to. On synthetic
    herds with overlapping ranges, it raised the share of linked moves
    joining one herd to itself from 0.70 to 0.78 (identical herds) and
    0.82 to 0.86 (herds of different sizes).
    """
    from scipy.optimize import linear_sum_assignment

    if df.empty:
        return pd.Series(dtype=int, index=df.index)
    frame = df.sort_values("Date", kind="stable").reset_index()
    lat = frame["Latitude"].to_numpy(dtype=float)
    lon = frame["Longitude"].to_numpy(dtype=float)
    day = frame["Date"].values.astype("datetime64[D]").astype(int)
    hour = _hours(frame)
    records = _records(frame)

    unit_of = np.full(len(frame), -1, dtype=int)
    # Open tracks, each held by the index of its most recent report.
    open_tracks: List[int] = []
    next_unit = 0
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    for begin, end in zip(starts, np.r_[starts[1:], len(frame)]):
        rows = np.arange(begin, end)
        open_tracks = [t for t in open_tracks if day[begin] - day[t] <= max_gap_days]
        label = _sightings(rows, lat, lon, records)
        groups = [rows[label == k] for k in np.unique(label)]
        heads = np.array([g[0] for g in groups])
        linked: Dict[int, int] = {}
        if open_tracks:
            costs = _link_costs(heads, open_tracks, day, lat, lon, records,
                                max_speed_km_per_day)
            # Any feasible link is taken before a new track is started; among
            # the ways to link as many as possible, the cheapest.
            new = np.full((len(heads), len(heads)), _NO_LINK)
            np.fill_diagonal(new, _NEW_TRACK)
            matrix = np.hstack([np.where(np.isfinite(costs), costs, _NO_LINK), new])
            for a, b in zip(*linear_sum_assignment(matrix)):
                if b < len(open_tracks) and np.isfinite(costs[a, b]):
                    linked[a] = open_tracks[b]
        placed: List[int] = []
        for a, members in enumerate(groups):
            if a in linked:
                unit = unit_of[linked[a]]
            else:
                same_day = _link_costs(heads[a:a + 1], placed, day, lat, lon, records,
                                       max_speed_km_per_day, hour)[0] if placed \
                    else np.array([])
                if len(same_day) and np.isfinite(same_day).any():
                    unit = unit_of[placed[int(np.argmin(same_day))]]
                else:
                    unit, next_unit = next_unit, next_unit + 1
            unit_of[members] = unit
            placed.extend(int(m) for m in members)
        latest = {int(unit_of[t]): t for t in open_tracks}
        for i in rows:
            latest[int(unit_of[i])] = int(i)
        open_tracks = list(latest.values())

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
# The detection grid: 2 km squares on the same national lattice, the unit
# the route memory is kept in and the page names next grids by.
GRID_KM = 2.0
# Route memory (``Routes``): moves that began this close to a group's
# position count as moves from the same place (Gaussian sd), each one's
# destination is spread this much, older moves count for less (half-life),
# and with the season on, moves made near this date of any year count most.
ROUTE_ORIGIN_KM = 1.5
ROUTE_DEST_KM = 1.0
ROUTE_HALF_LIFE_DAYS = 730.0
ROUTE_SEASON_DAYS = 30.0
# Weighted moves from a place needed before its memory says anything, and
# the trust levels the backtest chooses between: the memory gets weight
# n / (n + trust) for n remembered moves.
ROUTE_MIN_WEIGHT = 1.0
ROUTE_TRUST = (2.0, 8.0)
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
# The turn density: bins round the circle, the smoothing kernel's
# concentration (about 20 degrees), a group's own turns weighed against
# the register's (``TURN_POOL`` turns of its own count as much as the
# register), and the most recent of them used.
TURN_BINS = 144
TURN_KAPPA = 8.0
TURN_POOL = 5.0
OWN_MAX_TURNS = 10
DEFAULT_KM_PER_DAY = 3.8
# Distance grows with time as days ** alpha, alpha learned between these
# (``scaling``), from gaps with at least this many moves.
DIFFUSIVE, BALLISTIC = 0.5, 1.0
# Share of moves the forecast's window must reach.
REACH_MASS = 0.999
MIN_GAP_STEPS = 15
_MIN_MEDIAN_KM = 0.05
# Groups seen this recently are forecast; older ones may be anywhere.
ACTIVE_DAYS = 3
MAX_HORIZON_DAYS = 7
LIKELY_MASS = 0.5
POSSIBLE_MASS = 0.8
GRID_MARGIN_KM = 15.0
# Past forecasts needed before the zones are calibrated on them, and the
# most recent this many are what they are calibrated on: the forecast gets
# better as the register grows, and early forecasts made from a few
# dozen moves would otherwise keep every later zone too wide.
MIN_CALIBRATION = 40
CALIBRATION_WINDOW = 300
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


def scaling(step_table: pd.DataFrame) -> float:
    """How distance grows with time on this register: distance ~ days ** alpha.

    One half is a random walk (four days, twice as far as one); one is
    travel in a straight line (four days, four times as far). Groups that
    keep their heading for several days sit between, and assuming one half
    for them made every multi-day zone too small. Fitted to the median
    distance at each gap with enough moves, by weighted least squares on
    the log scale; one half until two gaps have enough moves.
    """
    if step_table.empty:
        return DIFFUSIVE
    gaps, medians, weights = [], [], []
    for gap, group in step_table.groupby("Gap (days)"):
        if len(group) >= MIN_GAP_STEPS:
            gaps.append(float(gap))
            medians.append(max(float(group["Distance (km)"].median()), _MIN_MEDIAN_KM))
            weights.append(float(len(group)))
    if len(gaps) < 2:
        return DIFFUSIVE
    x, y, w = np.log(gaps), np.log(medians), np.asarray(weights)
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    spread = float(np.sum(w * (x - xm) ** 2))
    if spread <= 0:
        return DIFFUSIVE
    slope = float(np.sum(w * (x - xm) * (y - ym)) / spread)
    return float(np.clip(slope, DIFFUSIVE, BALLISTIC))


def step_lengths(step_table: pd.DataFrame, alpha: float = 0.5) -> np.ndarray:
    """Observed distances scaled to one day: distance / days ** alpha (see ``scaling``)."""
    if step_table.empty:
        return np.array([])
    return (step_table["Distance (km)"]
            / step_table["Gap (days)"].astype(float) ** alpha).to_numpy()


def radial_profile(per_day: np.ndarray, days: float, cell_km: float = CELL_KM,
                   alpha: float = 0.5):
    """Density per square km of being ``r`` km from the start after ``days``, as a lookup table.

    Built once per horizon and interpolated for every cell, rather than
    smoothing every observed step against every cell.
    """
    days = max(days, 1.0)
    if len(per_day) >= MIN_STEPS:
        sample = per_day * days ** alpha
        bandwidth = max(1.06 * sample.std() * len(sample) ** -0.2, cell_km)
        r = np.linspace(0, sample.max() + 4 * bandwidth, 400)
        # Kernel density of the step length, reflected at zero.
        z = np.exp(-0.5 * ((r[:, None] - sample[None, :]) / bandwidth) ** 2) + \
            np.exp(-0.5 * ((r[:, None] + sample[None, :]) / bandwidth) ** 2)
        radial = z.mean(axis=1)
    else:
        # Too few observed steps: a half-normal with the typical daily move.
        scale = DEFAULT_KM_PER_DAY * days ** alpha / 0.6745
        r = np.linspace(0, 4 * scale, 400)
        radial = np.exp(-0.5 * (r / scale) ** 2)
    # The forecast's window stops where the reach holds REACH_MASS of the
    # moves: the longest move ever seen plus a margin made windows of
    # thousands of cells that held almost nothing.
    held = np.cumsum(radial)
    cut = min(int(np.searchsorted(held / held[-1], REACH_MASS)) + 1, len(r) - 1)
    cut = max(cut, int(np.searchsorted(r, cell_km * 2)) + 1)
    r, radial = r[:cut + 1], radial[:cut + 1]
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
    # How distance grows with time (``scaling``): 0.5 a random walk, 1 travel.
    alpha: float = 0.5
    # The turns themselves, by gap bucket: their smoothed density can have
    # two peaks -- some groups keep going, others turn back -- where one
    # mean direction averages the two away (``turn_density``).
    turn_samples: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _profiles: Dict[tuple, tuple] = field(default_factory=dict, repr=False)
    _turn_grids: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def learn(cls, step_table: pd.DataFrame) -> "StepModel":
        alpha = scaling(step_table)
        per_day = {"all": step_lengths(step_table, alpha)}
        if not step_table.empty:
            for social, group in step_table.groupby("Class"):
                if len(group) >= MIN_CLASS_STEPS:
                    per_day[social] = step_lengths(group, alpha)
        turns: Dict[int, tuple] = {}
        samples: Dict[int, np.ndarray] = {}
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
            samples[bucket] = np.mod(turn, 2 * np.pi)
        return cls(per_day, turns, alpha, samples)

    def lengths(self, social: str, use_class: bool = True) -> np.ndarray:
        if use_class and social in self.per_day:
            return self.per_day[social]
        return self.per_day["all"]

    def profile(self, social: str, days: float, cell_km: float):
        key = (social if social in self.per_day else "all", int(days), cell_km)
        if key not in self._profiles:
            self._profiles[key] = radial_profile(self.lengths(social), days, cell_km,
                                                 self.alpha)
        return self._profiles[key]

    def turn(self, days: float) -> Optional[tuple]:
        return self.turns.get(_bucket(days))

    def turn_density(self, days: float, own: Optional[np.ndarray] = None
                     ) -> Optional[np.ndarray]:
        """Density of the turn (radians, on ``TURN_BINS`` bins round the circle).

        The register's turns, smoothed. For a one-day forecast, a group's
        own earlier turns are mixed in, counting for more the more of them
        there are: a group that has been shuttling is expected to shuttle,
        one that has been travelling to keep going.
        """
        bucket = _bucket(days)
        if bucket not in self.turn_samples:
            return None
        if bucket not in self._turn_grids:
            self._turn_grids[bucket] = _circular_kde(self.turn_samples[bucket])
        density = self._turn_grids[bucket]
        if own is not None and len(own) and bucket == 1:
            weight = len(own) / (len(own) + TURN_POOL)
            density = (1 - weight) * density + weight * _circular_kde(np.mod(own, 2 * np.pi))
        return density

    def turn_shares(self) -> Optional[Tuple[float, float]]:
        """Shares of one-day moves that keep within 45 degrees of the last
        heading, and that turn back within 45 degrees of the way they came."""
        turn = self.turn_samples.get(1)
        if turn is None or not len(turn):
            return None
        off = np.abs(np.angle(np.exp(1j * turn)))
        return float(np.mean(off <= np.pi / 4)), float(np.mean(off >= 3 * np.pi / 4))


def _bucket(days: float) -> int:
    return int(min(max(days, 1), 3))


def _circular_kde(angles: np.ndarray) -> np.ndarray:
    """Von Mises kernel density of ``angles`` on ``TURN_BINS`` bins."""
    grid = (np.arange(TURN_BINS) + 0.5) * 2 * np.pi / TURN_BINS
    weights = np.exp(TURN_KAPPA * (np.cos(grid[:, None] - angles[None, :]) - 1)).mean(axis=1)
    norm = 2 * np.pi * np.i0(TURN_KAPPA) * np.exp(-TURN_KAPPA)
    return weights / norm


@dataclass(frozen=True)
class Config:
    """Which parts of the forecast are switched on, and how strongly."""

    places: float = 0.0      # the usage map: 0 ignored, 1 taken at face value
    season: bool = False     # the usage map also counts the same weeks of earlier years
    heading: bool = False    # direction relative to the group's last move
    own: bool = False        # the group's own earlier reports
    # Where groups went next from the same grid, mixed in with weight
    # n / (n + routes) for n remembered moves (weighted): 0 leaves it out; the
    # smaller the number, the fewer moves it takes to trust the memory.
    routes: float = 0.0
    route_season: bool = False  # ... weighted to moves made at this time of year

    def parts(self) -> List[str]:
        out = []
        if self.places:
            out.append("places elephants used before"
                       + (" (same season counted)" if self.season else ""))
        if self.heading:
            out.append("the group's direction of travel")
        if self.own:
            out.append("the group's own range")
        if self.routes:
            out.append("where groups went next from the same grid"
                       + (" at this time of year" if self.route_season else ""))
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
                    for routes in (0.0, *ROUTE_TRUST):
                        for route_season in ((False, True) if routes and with_season
                                             else (False,)):
                            out.append(Config(places, season, heading, own, routes,
                                              route_season))
    return out


@dataclass
class Situation:
    """What is known about a group at the moment of forecasting.

    The forecast covers the ``span`` days ending ``days`` after the group's
    last report: "the next three days" for a group seen yesterday is days
    2, 3 and 4 after that report, and the group may be anywhere it could
    be on any of them.
    """

    xy: np.ndarray
    days: float
    social: str
    prev_angle: Optional[float] = None
    own_xy: Optional[np.ndarray] = None   # its earlier reports, km plane
    span: int = 1
    own_turns: Optional[np.ndarray] = None  # its earlier one-day turns
    as_of: Optional[pd.Timestamp] = None    # the morning the forecast is made

    def leads(self) -> List[float]:
        last = float(self.days)
        first = max(1.0, last - max(int(self.span), 1) + 1)
        return [float(d) for d in np.arange(first, last + 0.5)]


def _components(s: Situation, grid: Grid, model: StepModel, priors: Dict[str, np.ndarray],
                routes: Optional["Routes"] = None, only: Optional[Config] = None):
    """Each part of the forecast as a log-density over the cells within reach.

    Reach and heading are one per day of the forecast's span (lists);
    places, the group's own range and the route memory are one each.
    ``only`` builds just the parts that configuration uses.
    """
    if only is not None:
        if not only.places:
            priors = {}
        elif only.season and "season" in priors:
            priors = {"season": priors["season"]}
        else:
            priors = {"places": priors["places"]}
    leads = s.leads()
    profiles = [model.profile(s.social, d, grid.cell_km) for d in leads]
    local = grid.near(s.xy, max(float(r[-1]) for r, _ in profiles))
    parts = {"reach": [np.log(np.interp(local.distance, r, density, right=0.0) + _TINY)
                       for r, density in profiles]}
    for name, prior in priors.items():
        parts[name] = np.log(grid.at(prior, local))
    if s.prev_angle is not None and np.isfinite(s.prev_angle) and \
            (only is None or only.heading):
        offset = local.centres - s.xy
        turn = np.mod(np.arctan2(offset[:, 1], offset[:, 0]) - s.prev_angle, 2 * np.pi)
        bins = np.minimum((turn / (2 * np.pi) * TURN_BINS).astype(int), TURN_BINS - 1)
        still = local.distance < grid.cell_km / 2
        headings = []
        for d in leads:
            density = model.turn_density(d, s.own_turns)
            if density is None:
                headings = []
                break
            # Relative to the uniform density, so the cell the group stands
            # in -- which has no direction -- is neutral.
            heading = np.log(density[bins] * 2 * np.pi + _TINY)
            heading[still] = 0.0
            headings.append(heading)
        if headings:
            parts["heading"] = headings
    if s.own_xy is not None and len(s.own_xy) and (only is None or only.own):
        own = _own_range(local, s.own_xy, grid.cell_km)
        own = own + USAGE_FLOOR * (own.mean() if own.any() else 1.0)
        parts["own"] = np.log(own)
    if routes is not None and s.as_of is not None and (only is None or only.routes):
        wanted = (("routes", False), ("routes_season", True)) if only is None else \
            ((("routes_season", True),) if only.route_season else (("routes", False),))
        for name, seasonal in wanted:
            memory = routes.prior(local, s.xy, s.as_of, leads, seasonal, grid.cell_km)
            if memory is not None:
                parts[name], parts[name + "_n"] = memory
    return local, parts


def _own_range(local: Local, own_xy: np.ndarray, cell_km: float) -> np.ndarray:
    """A group's earlier reports, each spread over ``OWN_BANDWIDTH_KM``."""
    return _spread(local, own_xy, cell_km, OWN_BANDWIDTH_KM)


def _spread(local: Local, points: np.ndarray, cell_km: float, bandwidth_km: float,
            weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted points, each spread over ``bandwidth_km``, on the cells within
    reach: counted on a raster of the window and blurred, rather than a
    distance from every cell to every point."""
    from scipy.ndimage import gaussian_filter

    ix = local.keys % _KEY_STRIDE
    iy = local.keys // _KEY_STRIDE
    x0, y0 = int(ix.min()), int(iy.min())
    image = np.zeros((int(iy.max()) - y0 + 1, int(ix.max()) - x0 + 1))
    px = np.floor(points[:, 0] / cell_km + 1e-9).astype(np.int64) - x0
    py = np.floor(points[:, 1] / cell_km + 1e-9).astype(np.int64) - y0
    inside = (px >= 0) & (px < image.shape[1]) & (py >= 0) & (py < image.shape[0])
    if not inside.any():
        return np.zeros(len(local.keys))
    w = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=float)
    np.add.at(image, (py[inside], px[inside]), w[inside])
    image = gaussian_filter(image, sigma=bandwidth_km / cell_km, mode="constant")
    return image[iy - y0, ix - x0]


# ---------------------------------------------------------------------------
# Route memory: which grids groups went to from each grid
# ---------------------------------------------------------------------------
def grid_key(xy: np.ndarray) -> np.ndarray:
    """The detection grid (``GRID_KM`` squares on the national lattice) of each point."""
    xy = np.atleast_2d(xy)
    ix = np.floor(xy[:, 0] / GRID_KM + 1e-9).astype(np.int64)
    iy = np.floor(xy[:, 1] / GRID_KM + 1e-9).astype(np.int64)
    return iy * _KEY_STRIDE + ix


def grid_label(key: int) -> str:
    """A grid's name: its column and row on the national lattice."""
    return f"{int(key) % _KEY_STRIDE}-{int(key) // _KEY_STRIDE}"


def grid_centre(key: int) -> np.ndarray:
    return np.array([(int(key) % _KEY_STRIDE + 0.5) * GRID_KM,
                     (int(key) // _KEY_STRIDE + 0.5) * GRID_KM])


def _day_gap(a: np.ndarray, b: float) -> np.ndarray:
    """Days between days of the year, round the calendar."""
    gap = np.abs(np.asarray(a, dtype=float) - b)
    return np.minimum(gap, 365.25 - gap)


@dataclass
class Routes:
    """Every move a group made, from the grid it was in to the one it was
    next reported in, kept so a forecast can ask: from here, where did
    groups go, and where did they go at this time of year?

    Each remembered move is weighted by how close its start is to the
    group's position, how long ago it was, how well its length in days
    matches the forecast's, and, with the season on, how near its date is
    to today's in any year. A move only counts once its second report is in
    the register, so a forecast never remembers a move it could not have
    known about.
    """

    origin: np.ndarray       # km plane
    dest: np.ndarray
    start: np.ndarray        # datetime64[D] of the first report
    known: np.ndarray        # datetime64[D] of the second: known from that day
    gap: np.ndarray          # days
    doy: np.ndarray          # day of the year the move began
    _tree: object = field(default=None, repr=False)

    @classmethod
    def from_steps(cls, step_table: pd.DataFrame, grid: "Grid") -> "Routes":
        from scipy.spatial import cKDTree

        if step_table.empty:
            empty = np.empty((0, 2))
            return cls(empty, empty, np.array([], dtype="datetime64[D]"),
                       np.array([], dtype="datetime64[D]"), np.array([]), np.array([]))
        origin = grid.project(step_table["From Latitude"].values,
                              step_table["From Longitude"].values)
        dest = grid.project(step_table["To Latitude"].values, step_table["To Longitude"].values)
        start = pd.to_datetime(step_table["From Date"])
        return cls(origin, dest, start.values.astype("datetime64[D]"),
                   pd.to_datetime(step_table["To Date"]).values.astype("datetime64[D]"),
                   step_table["Gap (days)"].to_numpy(dtype=float),
                   start.dt.dayofyear.to_numpy(dtype=float), cKDTree(origin))

    def __len__(self) -> int:
        return len(self.gap)

    def weights(self, xy: np.ndarray, as_of: pd.Timestamp, leads: List[float],
                seasonal: bool, radius_km: float = 3 * ROUTE_ORIGIN_KM
                ) -> Tuple[np.ndarray, np.ndarray]:
        """The moves remembered from near ``xy`` on the morning of ``as_of``, and
        their weights."""
        if not len(self):
            return np.array([], dtype=int), np.array([])
        near = np.asarray(self._tree.query_ball_point(xy, radius_km), dtype=int)
        if not len(near):
            return near, np.array([])
        today = np.datetime64(pd.Timestamp(as_of).normalize(), "D")
        near = near[self.known[near] <= today]
        if not len(near):
            return near, np.array([])
        distance = np.hypot(*(self.origin[near] - xy).T)
        weight = np.exp(-0.5 * (distance / ROUTE_ORIGIN_KM) ** 2)
        age = (today - self.known[near]).astype(float)
        weight *= 0.5 ** (age / ROUTE_HALF_LIFE_DAYS)
        # A move of 2 days says less about tomorrow than a move of 1.
        off = np.abs(self.gap[near] - np.clip(self.gap[near], min(leads), max(leads)))
        weight *= 0.5 ** off
        if seasonal:
            gap = _day_gap(self.doy[near], pd.Timestamp(as_of).dayofyear)
            weight *= np.exp(-0.5 * (gap / ROUTE_SEASON_DAYS) ** 2)
        return near, weight

    def prior(self, local: Local, xy: np.ndarray, as_of: pd.Timestamp, leads: List[float],
              seasonal: bool, cell_km: float) -> Optional[Tuple[np.ndarray, float]]:
        """Where the remembered moves from here went, as a probability over the
        cells within reach, and how many moves (weighted) it rests on; None
        while too few are remembered from here.

        It is a forecast of its own -- remembered destinations already say
        how far groups go -- so it is mixed with the rest, not multiplied
        into it: multiplying counted the distance twice.
        """
        near, weight = self.weights(xy, as_of, leads, seasonal)
        if weight.sum() < ROUTE_MIN_WEIGHT:
            return None
        density = _spread(local, self.dest[near], cell_km, ROUTE_DEST_KM, weight)
        if density.sum() <= 0:
            return None
        return density / density.sum(), float(weight.sum())

    def from_grid(self, key: int, as_of: pd.Timestamp, season_days: float = ROUTE_SEASON_DAYS
                  ) -> pd.DataFrame:
        """Every remembered move that began in grid ``key`` before ``as_of``: where
        it went, and whether it was made at this time of year."""
        columns = ["To Grid", "Moves", "This Season", "Last"]
        if not len(self):
            return pd.DataFrame(columns=columns)
        today = np.datetime64(pd.Timestamp(as_of).normalize(), "D")
        mask = (grid_key(self.origin) == key) & (self.known <= today)
        if not mask.any():
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame({
            "To Grid": grid_key(self.dest[mask]),
            "This Season": _day_gap(self.doy[mask], pd.Timestamp(as_of).dayofyear)
            <= season_days,
            "Last": self.known[mask],
        })
        out = frame.groupby("To Grid").agg(Moves=("Last", "size"),
                                           **{"This Season": ("This Season", "sum"),
                                              "Last": ("Last", "max")})
        return out.sort_values(["This Season", "Moves"], ascending=False).reset_index()


def _movement(parts: Dict[str, object], heading: bool) -> np.ndarray:
    """Where the group may be on any day of the span: the days' movement
    kernels (reach, and heading if used), each normalised, averaged."""
    out = None
    for k, reach in enumerate(parts["reach"]):
        log_k = reach + parts["heading"][k] if heading and "heading" in parts else reach.copy()
        log_k = log_k - log_k.max()
        p = np.exp(log_k)
        p /= p.sum()
        out = p if out is None else out + p
    return out / len(parts["reach"])


def _combine(parts: Dict[str, object], config: Config,
             cache: Optional[Dict[object, np.ndarray]] = None) -> np.ndarray:
    """The forecast for ``config``: movement times the place parts, then the
    route memory mixed in by how much of it there is."""
    base_key = ("base", config.places, config.season, config.heading, config.own)
    if cache is not None and base_key in cache:
        p = cache[base_key]
    else:
        heading = bool(config.heading and "heading" in parts)
        if cache is not None and heading in cache:
            move = cache[heading]
        else:
            move = _movement(parts, heading)
            if cache is not None:
                cache[heading] = move
        log_p = np.log(move + _TINY)
        if config.places:
            key = "season" if config.season and "season" in parts else "places"
            log_p = log_p + config.places * parts[key]
        if config.own and "own" in parts:
            log_p = log_p + parts["own"]
        log_p -= log_p.max()
        p = np.exp(log_p)
        p /= p.sum()
        if cache is not None:
            cache[base_key] = p
    if config.routes:
        key = "routes_season" if config.route_season else "routes"
        if key in parts:
            n = float(parts[key + "_n"])
            weight = n / (n + config.routes)
            p = (1 - weight) * p + weight * parts[key]
    return p


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
    scores = scores[-CALIBRATION_WINDOW:]
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
    # Across the landscape, whatever the tracker linked: of every report in
    # the days after each morning, how many fell in some group's likely
    # zone, and in same-size circles round each group's last report; and
    # the zones' area on an average morning.
    field_reports: int = 0
    field_hits: int = 0
    field_circle_hits: int = 0
    field_area_km2: float = float("nan")

    @property
    def field_rate(self) -> float:
        return self.field_hits / self.field_reports if self.field_reports else float("nan")

    @property
    def field_circle_rate(self) -> float:
        return (self.field_circle_hits / self.field_reports if self.field_reports
                else float("nan"))

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
    """Each unit's reports in date order, to look up its range and its turns."""

    def __init__(self, df: pd.DataFrame, units: pd.Series, grid: Grid,
                 step_table: Optional[pd.DataFrame] = None):
        work = df.assign(_unit=units.values).sort_values("Date", kind="stable")
        xy = grid.project(work["Latitude"].values, work["Longitude"].values)
        self._by_unit: Dict[int, tuple] = {}
        for unit, rows in work.groupby("_unit").indices.items():
            self._by_unit[int(unit)] = (work["Date"].values[rows], xy[rows])
        self._turns: Dict[int, tuple] = {}
        if step_table is not None and len(step_table):
            turns = step_table.dropna(subset=["Prev Angle"])
            turns = turns[(turns["Gap (days)"] == 1) & (turns["Distance (km)"] >= MIN_MOVE_KM)
                          & (turns["Prev Distance (km)"] >= MIN_MOVE_KM)]
            for unit, rows in turns.groupby("Unit"):
                self._turns[int(unit)] = (rows["To Date"].to_numpy(),
                                          (rows["Angle"] - rows["Prev Angle"]).to_numpy())

    def turns(self, unit: int, upto) -> Optional[np.ndarray]:
        """The unit's one-day turns known at ``upto`` (the time of its last report)."""
        dates, turn = self._turns.get(int(unit), (None, None))
        if dates is None:
            return None
        known = turn[dates <= np.datetime64(upto)][-OWN_MAX_TURNS:]
        return known if len(known) else None

    def own(self, unit: int, before) -> Optional[np.ndarray]:
        """The unit's reports before ``before``, the most recent ``OWN_MAX_REPORTS``."""
        dates, xy = self._by_unit.get(int(unit), (None, None))
        if dates is None:
            return None
        return xy[dates < np.datetime64(before)][-OWN_MAX_REPORTS:]


def _situations(table: pd.DataFrame, grid: Grid, history: _History) -> List[Situation]:
    """The forecasts in ``table`` (from ``_cases``), as the page would make them."""
    origin = grid.project(table["From Latitude"].values, table["From Longitude"].values)
    prev = table["Prev Angle"].to_numpy(dtype=float)
    prev_km = table["Prev Distance (km)"].to_numpy(dtype=float)
    out = []
    mornings = table["As Of"].to_numpy() if "As Of" in table else [None] * len(table)
    for i, (unit, social, start, days, span, as_of) in enumerate(zip(
            table["Unit"].to_numpy(), table["Class"].to_numpy(),
            table["From Date"].to_numpy(), table["Days"].to_numpy(),
            table["Span"].to_numpy(), mornings)):
        usable = np.isfinite(prev[i]) and prev_km[i] >= MIN_MOVE_KM
        out.append(Situation(origin[i], float(days), str(social),
                             float(prev[i]) if usable else None, history.own(unit, start),
                             int(span), history.turns(unit, start),
                             pd.Timestamp(as_of) if as_of is not None else None))
    return out


def lead(since: int, horizon_days: int) -> Tuple[int, int]:
    """The forecast's last day after the group's report, and how many days
    it covers: a group seen ``since`` days ago, forecast ``horizon_days``
    ahead."""
    days = min(since + horizon_days, MAX_HORIZON_DAYS)
    return days, max(days - since, 1)


def _cases(step_table: pd.DataFrame, horizon_days: int, begin: pd.Timestamp,
           end: pd.Timestamp) -> pd.DataFrame:
    """The forecasts the page would have shown on each day from ``begin`` to
    ``end``, for which the group's next report says how they did.

    On day ``t`` a group is forecast if its last report is at most
    ``ACTIVE_DAYS - 1`` days old, for the ``horizon_days`` after ``t``. A
    move from report a to report b is the outcome of every such forecast
    made from a with b inside its window: up to ``ACTIVE_DAYS`` of them,
    one per morning the page showed the group. Scoring each move once,
    with the time to the next report known, as this did before, tested a
    forecast the page never makes.
    """
    columns = [*step_table.columns, "As Of", "Since", "Days", "Span"]
    if step_table.empty:
        return pd.DataFrame(columns=columns)
    a = step_table["From Date"].dt.normalize()
    b = step_table["To Date"].dt.normalize()
    one = pd.Timedelta(days=1)
    first = np.maximum(a, b - pd.Timedelta(days=horizon_days))
    last = np.minimum(b - one, a + pd.Timedelta(days=ACTIVE_DAYS - 1))
    first = np.maximum(first, begin.normalize())
    last = np.minimum(last, end.normalize())
    keep = first <= last
    frames = []
    for offset in range(ACTIVE_DAYS):
        at = first + pd.Timedelta(days=offset)
        ok = keep & (at <= last)
        if not ok.any():
            continue
        part = step_table[ok].copy()
        part["As Of"] = at[ok].to_numpy()
        part["Since"] = (part["As Of"] - a[ok]).dt.days.to_numpy()
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    leads = [lead(int(since), horizon_days) for since in out["Since"]]
    out["Days"] = [d for d, _ in leads]
    out["Span"] = [n for _, n in leads]
    return out.sort_values(["As Of", "Unit"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training: month by month, and only the months that changed
# ---------------------------------------------------------------------------
# Bump when the method changes, so a training made by older code is never
# reused by newer code.
TRAINING_VERSION = 3
MIN_HISTORY_DAYS = 60


@dataclass
class MonthResult:
    """One month forecast from the months before it, kept so it need not be redone."""

    month: str
    fingerprint: str
    scores: Dict[Config, float]           # summed log score per configuration
    masses: Dict[Config, List[float]]     # mass needed to reach each next report
    reach: List[List[float]]              # [distance to the next report, days forecast]
    tested: int = 0
    hits: int = 0
    stay_hits: int = 0
    discordant: int = 0
    area_sum: float = 0.0
    circle_hits: int = 0
    circle_tested: int = 0
    circle_area_sum: float = 0.0
    field_reports: int = 0
    field_hits: int = 0
    field_circle_hits: int = 0
    field_mornings: int = 0
    field_area_sum: float = 0.0


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
            return (f"{c.places:g}|{int(c.season)}|{int(c.heading)}|{int(c.own)}|"
                    f"{c.routes:g}|{int(c.route_season)}")
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
                "reach": [[float(x), float(d)] for x, d in m.reach], "tested": int(m.tested),
                "hits": int(m.hits), "stay_hits": int(m.stay_hits),
                "discordant": int(m.discordant), "area_sum": float(m.area_sum),
                "circle_hits": int(m.circle_hits), "circle_tested": int(m.circle_tested),
                "circle_area_sum": float(m.circle_area_sum),
                "field": [int(m.field_reports), int(m.field_hits), int(m.field_circle_hits),
                          int(m.field_mornings), float(m.field_area_sum)],
            } for m in self.months],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Training":
        def config(text: str) -> Config:
            places, season, heading, own, routes, route_season = text.split("|")
            return Config(float(places), bool(int(season)), bool(int(heading)),
                          bool(int(own)), float(routes), bool(int(route_season)))
        months = [MonthResult(
            str(m["month"]), str(m["fingerprint"]),
            {config(k): float(v) for k, v in m["scores"].items()},
            {config(k): [float(x) for x in v] for k, v in m["masses"].items()},
            [[float(x), float(d)] for x, d in m["reach"]], int(m["tested"]), int(m["hits"]),
            int(m["stay_hits"]), int(m["discordant"]), float(m["area_sum"]),
            int(m["circle_hits"]), int(m["circle_tested"]), float(m["circle_area_sum"]),
            *[int(x) for x in m["field"][:4]], float(m["field"][4]),
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
    mornings = sum(m.field_mornings for m in months)
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
        sum(m.field_reports for m in months), sum(m.field_hits for m in months),
        sum(m.field_circle_hits for m in months),
        sum(m.field_area_sum for m in months) / mornings if mornings else float("nan"),
    )


def _origins(df: pd.DataFrame, units: pd.Series, step_table: pd.DataFrame) -> pd.DataFrame:
    """Each unit's last report of each day: where the page would forecast
    it from, with the move that brought it there."""
    work = df.assign(Unit=units.values).sort_values("Date", kind="stable")
    work = work.assign(Day=work["Date"].dt.normalize())
    last = work.groupby(["Unit", "Day"], sort=False).tail(1)
    out = pd.DataFrame({
        "Unit": last["Unit"].to_numpy(), "Date": last["Date"].to_numpy(),
        "Day": last["Day"].to_numpy(), "Latitude": last["Latitude"].to_numpy(),
        "Longitude": last["Longitude"].to_numpy(),
        "Class": [r["social"] for r in _records(last)]})
    moves = step_table[step_table["Distance (km)"] >= MIN_MOVE_KM]
    into = dict(zip(zip(moves["Unit"].to_numpy(), moves["To Date"].to_numpy()),
                    moves["Angle"].to_numpy()))
    out["Prev"] = [into.get((u, d), np.nan) for u, d in zip(out["Unit"], out["Date"])]
    return out.sort_values("Date", kind="stable").reset_index(drop=True)


def _score_cases(rows: pd.DataFrame, grid: Grid, history: "_History", model: StepModel,
                 priors: Dict[str, np.ndarray], candidates: List[Config], chosen: Config,
                 likely_mass: float, circle_r: Optional[float], result: "MonthResult",
                 memo: Dict[tuple, tuple], routes: Optional[Routes] = None) -> None:
    """Score one morning's forecasts of groups whose next report came in time."""
    targets = grid.project(rows["To Latitude"].values, rows["To Longitude"].values)
    for situation, target, key, unit, start in zip(
            _situations(rows, grid, history), targets, grid.key_of(targets),
            rows["Unit"].to_numpy(), rows["From Date"].to_numpy()):
        local, parts = _components(situation, grid, model, priors, routes)
        memo[(int(unit), np.datetime64(start, "ns"))] = (local, parts)
        at = int(np.searchsorted(local.keys, key))
        at = at if at < len(local.keys) and local.keys[at] == key else -1
        probs, cache = {}, {}
        for c in candidates:
            p = _combine(parts, c, cache)
            probs[c] = p
            result.scores[c] += float(np.log(p[at] + _TINY)) if at >= 0 \
                else float(np.log(_TINY))
            result.masses[c].append(_mass_to(p, at))
        moved = float(np.hypot(*(target - situation.xy)))
        result.reach.append([moved, float(situation.days)])

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
            radius = circle_r * situation.days ** model.alpha
            result.circle_hits += moved <= radius
            result.circle_tested += 1
            result.circle_area_sum += np.pi * radius ** 2


def _score_landscape(mornings: pd.DataFrame, as_of: pd.Timestamp, df_days: np.ndarray,
                     df_keys: np.ndarray, horizon_days: int, grid: Grid, history: "_History",
                     model: StepModel, priors: Dict[str, np.ndarray], chosen: Config,
                     likely_mass: float, result: "MonthResult",
                     memo: Dict[tuple, tuple], routes: Optional[Routes] = None) -> None:
    """The morning's map as a whole, against every report of the days after.

    Every group the page would have shown gets its likely zone, whether or
    not it was reported again; every report in the window counts, whether
    or not the tracker linked it, including groups nobody had seen for
    days. This is what the map promises a field team, and it does not
    depend on the tracker's choices.
    """
    after = (df_days > np.datetime64(as_of)) & \
        (df_days <= np.datetime64(as_of + pd.Timedelta(days=horizon_days)))
    if not after.any():
        return
    latest = mornings.groupby("Unit").tail(1)
    zone_keys = set()
    centres, radii = [], []
    for row in latest.itertuples(index=False):
        since = int((as_of - pd.Timestamp(row.Day)).days)
        days, span = lead(since, horizon_days)
        prev = float(row.Prev) if np.isfinite(row.Prev) else None
        situation = Situation(grid.project(row.Latitude, row.Longitude)[0], days, row.Class,
                              prev, history.own(row.Unit, row.Date), span,
                              history.turns(row.Unit, row.Date), as_of)
        known = memo.get((int(row.Unit), np.datetime64(row.Date, "ns")))
        local, parts = known if known is not None else _components(
            situation, grid, model, priors, routes, only=chosen)
        likely = zone(_combine(parts, chosen), likely_mass)
        zone_keys.update(local.keys[likely].tolist())
        centres.append(situation.xy)
        radii.append(float(np.sqrt(likely.sum() * grid.cell_km ** 2 / np.pi)))
    targets = df_keys[after]
    result.field_reports += int(len(targets))
    result.field_hits += int(sum(k in zone_keys for k in targets.tolist()))
    # Same-size circles round each group's last report: a report is caught
    # if its cell's centre lies in any of them.
    cell = grid.cell_km
    target_xy = np.column_stack([(targets % _KEY_STRIDE + 0.5) * cell,
                                 (targets // _KEY_STRIDE + 0.5) * cell])
    gap = np.hypot(target_xy[:, None, 0] - np.array(centres)[None, :, 0],
                   target_xy[:, None, 1] - np.array(centres)[None, :, 1])
    result.field_circle_hits += int((gap <= np.array(radii)[None, :]).any(axis=1).sum())
    result.field_mornings += 1
    result.field_area_sum += len(zone_keys) * grid.cell_km ** 2


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
    history = _History(df, units_all, grid, all_steps)
    origins = _origins(df, units_all, all_steps)
    # Every move, remembered from the day its second report came in.
    routes = Routes.from_steps(all_steps, grid)
    df_days = df["Date"].dt.normalize().to_numpy()
    df_keys = grid.key_of(grid.project(df["Latitude"].values, df["Longitude"].values))

    scores = {c: 0.0 for c in candidates}
    masses: Dict[Config, List[float]] = {c: [] for c in candidates}
    reach_scores: List[List[float]] = []
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
        cases = _cases(all_steps, horizon_days, begin, end)
        scored_before = any(m.tested for m in results)
        active = origins[(origins["Day"] >= begin.normalize() - pd.Timedelta(days=ACTIVE_DAYS))
                         & (origins["Day"] <= end)]
        if len(cases) or len(active):
            # The tracker only ever looks back, so the units it gives the
            # reports before ``begin`` are the same whether or not later
            # reports exist: the steps finished before the month are the
            # ones a forecast made then could have learned from.
            model = StepModel.learn(all_steps[all_steps["To Date"] < begin])
            chosen = max(scores, key=scores.get) if scored_before else DEFAULT_CONFIG
            likely_mass = calibrated_mass(masses[chosen], LIKELY_MASS)
            per_day = [moved / days ** model.alpha for moved, days in reach_scores]
            circle_r = (float(np.quantile(per_day, 0.5)) if len(per_day) >= MIN_CALIBRATION
                        else None)
            by_day = dict(tuple(cases.groupby("As Of", sort=True))) if len(cases) else {}
            for as_of in pd.date_range(begin.normalize(), end.normalize()):
                mornings = origins[(origins["Day"] <= as_of)
                                   & (origins["Day"] > as_of - pd.Timedelta(days=ACTIVE_DAYS))]
                rows = by_day.get(as_of)
                if rows is None and mornings.empty:
                    continue
                # What the page would have known that morning: every report
                # up to the end of the day before ``as_of + 1``.
                recent = usage_prior(df, grid, as_of + pd.Timedelta(days=1))
                priors = {"places": recent}
                seasonal = season_prior(df, grid, as_of, recent) if with_season else None
                if seasonal is not None:
                    priors["season"] = seasonal
                memo: Dict[tuple, tuple] = {}
                if rows is not None:
                    _score_cases(rows, grid, history, model, priors, candidates, chosen,
                                 likely_mass, circle_r, result, memo, routes)
                if not mornings.empty:
                    _score_landscape(mornings, as_of, df_days, df_keys, horizon_days, grid,
                                     history, model, priors, chosen, likely_mass, result,
                                     memo, routes)
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


def _neighbourhood(prob: np.ndarray, local: Local) -> np.ndarray:
    """Each cell's probability with its eight neighbours': the likeliest
    3 x 3 km block. A single-cell peak always sits on the group's own cell,
    because moves spread over rings around it, so the likeliest *cell*
    says "where it was" even for a group plainly on the move."""
    total = np.zeros_like(prob)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = local.keys + dy * _KEY_STRIDE + dx
            at = np.clip(np.searchsorted(local.keys, shifted), 0, len(local.keys) - 1)
            found = local.keys[at] == shifted
            total[found] += prob[at[found]]
    return total


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
           "trails": pd.DataFrame(), "next_grids": pd.DataFrame(),
           "grid_memory": pd.DataFrame(), "grid_moves": pd.DataFrame(),
           "steps_learned": 0, "config": config,
           "masses": (likely_mass, possible_mass), "cell_km": CELL_KM}
    history = df[df["Date"] < as_of.normalize() + pd.Timedelta(days=1)]
    if history.empty:
        return out
    units = assign_units(history)
    learned = steps(history, units)
    model = StepModel.learn(learned)
    out["steps_learned"] = len(learned)
    out["turns"] = model.turns
    out["turn_shares"] = model.turn_shares()
    out["alpha"] = model.alpha
    work = history.assign(_unit=units.values)
    last = work.sort_values("Date", kind="stable").groupby("_unit").tail(1)
    active = last[last["Date"] >= as_of.normalize() - pd.Timedelta(days=active_days - 1)]
    if active.empty:
        return out

    grid = Grid.around(history)
    tracks = _History(history, units, grid, learned)
    routes = Routes.from_steps(learned, grid)
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
    rows, trails, next_rows, memory_rows = [], [], [], []

    def place(key: int) -> str:
        if village_xy is None:
            return ""
        return _village_names(grid_centre(key)[None, :], village_xy, located)[0]
    for row, rec in zip(active.to_dict("records"), _records(active)):
        since = max((as_of.normalize() - pd.Timestamp(row["Date"]).normalize()).days, 0)
        days, span = lead(since, horizon_days)
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
                              tracks.own(unit, pd.Timestamp(row["Date"])), span,
                              tracks.turns(unit, pd.Timestamp(row["Date"])),
                              as_of.normalize())
        local, parts = _components(situation, grid, model, priors, routes, only=config)
        prob = _combine(parts, config)
        likely, possible = zone(prob, likely_mass), zone(prob, possible_mass)
        # The map covers the grid; a zone cell beyond its margin is not drawn.
        on_grid = local.flat >= 0
        likely_any[local.flat[on_grid & likely]] = True
        possible_any[local.flat[on_grid & possible]] = True
        combined_miss[local.flat[on_grid]] *= 1 - prob[on_grid]
        best_xy = local.centres[int(np.argmax(_neighbourhood(prob, local)))]
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
        here = int(grid_key(origin)[0])
        chances = pd.Series(prob).groupby(grid_key(local.centres)).sum()
        chances = chances.sort_values(ascending=False, kind="stable")
        for rank, (key, chance) in enumerate(chances.head(NEXT_GRIDS).items(), start=1):
            centre = from_plane(grid_centre(key)[None, :], grid.ref_lat)[0]
            next_rows.append({"Group": label, "Rank": rank, "Grid": grid_label(key),
                              "Near": place(key), "Chance": float(chance),
                              "Stay": int(key) == here,
                              "Latitude": float(centre[0]), "Longitude": float(centre[1])})
        memory = routes.from_grid(here, as_of)
        for item in memory.head(NEXT_GRIDS * 2).itertuples(index=False):
            memory_rows.append({"Group": label, "From Grid": grid_label(here),
                                "To Grid": grid_label(item[0]), "Near": place(item[0]),
                                "Moves": int(item.Moves),
                                "This Time of Year": int(item[2]),
                                "Last": pd.Timestamp(item.Last)})
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
            "Grid": grid_label(here),
            "Next Grids": "; ".join(
                f"{grid_label(key)}{' (stays)' if int(key) == here else ''} "
                f"{chance:.0%}" for key, chance in chances.head(NEXT_GRIDS).items()),
            "Moves From Grid": int(memory["Moves"].sum()) if len(memory) else 0,
            "This Time of Year": int(memory["This Season"].sum()) if len(memory) else 0,
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
    out["next_grids"] = pd.DataFrame(next_rows)
    out["grid_memory"] = pd.DataFrame(memory_rows)
    out["grid_moves"] = grid_moves(routes, grid, villages)
    if village_hits:
        table = pd.DataFrame(village_hits.values())
        table["_order"] = (table["Band"] != "Likely").astype(int)
        out["villages"] = table.sort_values(["_order", "From last report (km)"]).drop(
            columns="_order").reset_index(drop=True)
    return out


NEXT_GRIDS = 3
SEASON_MONTHS = {"Winter (Oct-Feb)": (10, 11, 12, 1, 2), "Summer (Mar-May)": (3, 4, 5),
                 "Monsoon (Jun-Sep)": (6, 7, 8, 9)}


def _village_names(xy: np.ndarray, village_xy: np.ndarray, villages: pd.DataFrame,
                   within_km: float = 3.0) -> List[str]:
    """The nearest village to each point, or "" beyond ``within_km``."""
    from scipy.spatial import cKDTree

    if not len(xy):
        return []
    distance, index = cKDTree(village_xy).query(np.atleast_2d(xy))
    names = villages["Village"].astype(str).to_numpy()
    return [str(names[k]) if d <= within_km else "" for d, k in zip(distance, index)]


def grid_moves(routes: Routes, grid: Grid, villages: Optional[pd.DataFrame] = None
               ) -> pd.DataFrame:
    """The whole route memory: every pair of grids groups moved between,
    how often, in which season, and when last."""
    columns = ["From Grid", "From Near", "To Grid", "To Near", "Moves", *SEASON_MONTHS,
               "Last", "From Latitude", "From Longitude", "To Latitude", "To Longitude"]
    if not len(routes):
        return pd.DataFrame(columns=columns)
    month = pd.DatetimeIndex(routes.start).month
    frame = pd.DataFrame({"From": grid_key(routes.origin), "To": grid_key(routes.dest),
                          "Last": routes.known})
    for name, months in SEASON_MONTHS.items():
        frame[name] = np.isin(month, months)
    table = frame.groupby(["From", "To"]).agg(
        Moves=("Last", "size"), Last=("Last", "max"),
        **{name: (name, "sum") for name in SEASON_MONTHS}).reset_index()
    located = village_xy = None
    if villages is not None and len(villages):
        located = villages.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
        village_xy = grid.project(located["Latitude"].values, located["Longitude"].values)
    out = pd.DataFrame({"From Grid": [grid_label(k) for k in table["From"]],
                        "To Grid": [grid_label(k) for k in table["To"]]})
    for side, keys in (("From", table["From"]), ("To", table["To"])):
        centres = np.array([grid_centre(k) for k in keys])
        latlon = from_plane(centres, grid.ref_lat)
        out[f"{side} Latitude"] = latlon[:, 0].round(5)
        out[f"{side} Longitude"] = latlon[:, 1].round(5)
        out[f"{side} Near"] = (_village_names(centres, village_xy, located)
                               if village_xy is not None else "")
    out["Moves"] = table["Moves"].to_numpy()
    for name in SEASON_MONTHS:
        out[name] = table[name].astype(int).to_numpy()
    out["Last"] = pd.to_datetime(table["Last"].to_numpy())
    return out[columns].sort_values(["Moves", "Last"], ascending=False).reset_index(drop=True)


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
