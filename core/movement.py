"""Where groups seen in the last few days may be next, and how often that has been right.

The register has no collars and no identified animals, only a time, a
place and a rough count per report. So the forecast is built in three
steps, each of them inspectable:

1. **Movement units.** Reports are chained into tracks the way radar
   tracks aircraft (``assign_units``): a report joins the best-matching
   open track if it is close enough in space for the time elapsed and its
   group composition is compatible. A unit is a plausible sequence of
   reports of one group, not an identified animal.
2. **How far they go.** Every consecutive pair of reports in a unit, a day
   or more apart, is one observed step. Their lengths, scaled to one day,
   give the distance a group typically covers (``step_lengths``).
3. **Where they go.** Elephants keep returning to the same forest blocks,
   nalas and fields, so the places reported before -- recent ones weighted
   more -- form a usage map (``usage_prior``).

The forecast for a group is the step distribution centred on where it was
last seen, multiplied by the usage map: within reach, and where elephants
have been before. Its *likely zone* is the smallest set of 1 km cells
holding half the probability.

**Every forecast comes with its own track record** (``backtest``). Each
month of the register is forecast from the months before it only, and the
likely zone is scored against where the group was actually reported next.
The comparison is a circle of the same area around where the group was
last seen -- "they stay near where they were". If the forecast does not
beat that, the page says so, and the zone should be read as nothing more
than that circle.

What this cannot do: predict a first visit to a place nobody has reported
elephants in, follow a group through a reporting gap longer than
``MAX_GAP_DAYS``, or tell two bulls in the same valley apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    ref_lat = float(frame["Latitude"].mean())
    points = km_plane(frame["Latitude"].values, frame["Longitude"].values, ref_lat)
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
            distance = float(np.hypot(*(points[i] - points[t])))
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
# mean usage is added everywhere.
USAGE_FLOOR = 0.05
# Steps used to learn how far groups move; fewer than this and a typical
# figure for elephants in central India stands in (3.8 km/day median).
MIN_STEPS = 30
DEFAULT_KM_PER_DAY = 3.8
# Groups seen this recently are forecast; older ones may be anywhere.
ACTIVE_DAYS = 3
MAX_HORIZON_DAYS = 7
LIKELY_MASS = 0.5
POSSIBLE_MASS = 0.8
GRID_MARGIN_KM = 15.0
# How much the usage map counts, chosen month by month from the months
# before: groups that keep to the same places reward it, groups that
# wander do not, and on those it would drag the zone back along the trail.
PRIOR_WEIGHTS = (0.0, 0.5, 1.0)
DEFAULT_PRIOR_WEIGHT = 0.5


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

    @classmethod
    def around(cls, df: pd.DataFrame, cell_km: float = CELL_KM,
               margin_km: float = GRID_MARGIN_KM) -> "Grid":
        ref_lat = float(df["Latitude"].mean())
        xy = km_plane(df["Latitude"].values, df["Longitude"].values, ref_lat)
        x0, y0 = xy.min(axis=0) - margin_km
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

    def latlon(self) -> np.ndarray:
        return from_plane(self.centres, self.ref_lat)


def usage_prior(df: pd.DataFrame, grid: Grid, as_of: pd.Timestamp,
                half_life_days: float = USAGE_HALF_LIFE_DAYS,
                bandwidth_km: float = USAGE_BANDWIDTH_KM) -> np.ndarray:
    """Where elephants have been reported before ``as_of``, recent reports weighted more.

    Only reports strictly before ``as_of`` count, so a forecast never sees
    the report it is about to be scored on.
    """
    from scipy.ndimage import gaussian_filter

    past = df[df["Date"] < as_of]
    density = np.zeros(grid.nx * grid.ny)
    if len(past):
        age = (as_of - past["Date"]).dt.total_seconds().to_numpy() / 86400.0
        weight = 0.5 ** (age / half_life_days)
        cells = grid.cell_of(grid.project(past["Latitude"].values, past["Longitude"].values))
        keep = cells >= 0
        np.add.at(density, cells[keep], weight[keep])
        density = gaussian_filter(density.reshape(grid.ny, grid.nx),
                                  sigma=bandwidth_km / grid.cell_km).ravel()
    mean = density.mean() if density.any() else 1.0
    density = density + USAGE_FLOOR * mean
    return density / density.sum()


def steps(df: pd.DataFrame, units: pd.Series) -> pd.DataFrame:
    """Every move between consecutive reports of a unit, a day or more apart."""
    work = df.assign(_unit=units.values).sort_values(["_unit", "Date"], kind="stable")
    ref_lat = float(df["Latitude"].mean()) if len(df) else 0.0
    xy = km_plane(work["Latitude"].values, work["Longitude"].values, ref_lat)
    same = work["_unit"].values[1:] == work["_unit"].values[:-1]
    gap = (work["Date"].values[1:] - work["Date"].values[:-1]).astype("timedelta64[D]")
    gap = gap.astype(int)
    keep = same & (gap >= 1)
    moved = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    origin = work.iloc[:-1][keep]
    target = work.iloc[1:][keep]
    return pd.DataFrame({
        "Unit": origin["_unit"].to_numpy(),
        "From Date": origin["Date"].to_numpy(),
        "To Date": target["Date"].to_numpy(),
        "From Latitude": origin["Latitude"].to_numpy(),
        "From Longitude": origin["Longitude"].to_numpy(),
        "To Latitude": target["Latitude"].to_numpy(),
        "To Longitude": target["Longitude"].to_numpy(),
        "Gap (days)": gap[keep],
        "Distance (km)": moved[keep],
    })


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


def forecast(origin_xy: np.ndarray, days: float, grid: Grid, prior: np.ndarray,
             per_day: np.ndarray, prior_weight: float = 1.0, profile=None) -> np.ndarray:
    """Probability of each cell holding the group ``days`` after it was at ``origin_xy``.

    ``prior_weight`` is how much the usage map counts: 0 ignores it (the
    zone is then a disc around the last report), 1 takes it at face value.
    """
    r, density = profile if profile is not None else radial_profile(per_day, days,
                                                                     grid.cell_km)
    distances = np.linalg.norm(grid.centres - origin_xy, axis=1)
    reach = np.interp(distances, r, density, right=0.0)
    p = reach * prior ** prior_weight if prior_weight else reach
    total = p.sum()
    return p / total if total > 0 else np.full_like(p, 1.0 / len(p))


def zone(prob: np.ndarray, mass: float = LIKELY_MASS) -> np.ndarray:
    """The fewest cells that together hold ``mass`` of the probability."""
    order = np.argsort(prob)[::-1]
    upto = np.searchsorted(np.cumsum(prob[order]), mass) + 1
    mask = np.zeros(len(prob), dtype=bool)
    mask[order[:upto]] = True
    return mask


# ---------------------------------------------------------------------------
# Track record
# ---------------------------------------------------------------------------
@dataclass
class TrackRecord:
    """How often the likely zone caught the next report, against staying put."""

    tested: int
    forecast_hits: int
    stay_hits: int
    mean_area_km2: float
    months: int
    horizon_days: int

    @property
    def forecast_rate(self) -> float:
        return self.forecast_hits / self.tested if self.tested else float("nan")

    @property
    def stay_rate(self) -> float:
        return self.stay_hits / self.tested if self.tested else float("nan")

    @property
    def beats_staying_put(self) -> bool:
        """Ahead by more than chance on the paired tests, at about 90% one-sided."""
        if self.tested < 20:
            return False
        diff = self.forecast_hits - self.stay_hits
        # Paired binomial: only discordant pairs carry information.
        return diff > 1.28 * np.sqrt(max(self.discordant, 1))

    discordant: int = 0
    # The weight the live forecast uses: the best log score over every
    # month scored, and that score per weight (higher is better).
    prior_weight: float = DEFAULT_PRIOR_WEIGHT
    log_scores: Dict[float, float] = field(default_factory=dict)


def backtest(df: pd.DataFrame, horizon_days: int = 1, min_history_days: int = 60,
             grid: Optional[Grid] = None) -> TrackRecord:
    """Forecast each month from the months before it, and score the likely zone.

    A test is a step whose origin falls in the month and whose next report
    came within ``horizon_days`` (1 means the next day). Units, step
    lengths and the usage map are all rebuilt from reports before the
    month began, so nothing from the month being scored leaks in.
    """
    empty = TrackRecord(0, 0, 0, float("nan"), 0, horizon_days)
    if df.empty or (df["Date"].max() - df["Date"].min()).days < min_history_days + 14:
        return empty
    grid = grid or Grid.around(df)
    units_all = assign_units(df)
    all_steps = steps(df, units_all)
    if all_steps.empty:
        return empty
    start = (df["Date"].min() + pd.Timedelta(days=min_history_days)).to_period("M").start_time
    months = pd.period_range(start, df["Date"].max(), freq="M")
    tested = forecast_hits = stay_hits = discordant = 0
    areas: List[float] = []
    used_months = 0
    # Log score per prior weight, summed over the months already scored.
    scores = {w: 0.0 for w in PRIOR_WEIGHTS}
    for month in months:
        begin, end = month.start_time, month.end_time
        tests = all_steps[(all_steps["From Date"] >= begin) & (all_steps["From Date"] <= end)
                          & (all_steps["Gap (days)"] <= horizon_days)]
        if tests.empty:
            continue
        # The tracker only ever looks back, so the units it gives the
        # reports before ``begin`` are the same whether or not later
        # reports exist: the steps finished before the month are the
        # ones a forecast made then could have learned from.
        per_day = step_lengths(all_steps[all_steps["To Date"] < begin])
        prior = usage_prior(df, grid, begin)
        profiles: Dict[int, tuple] = {}
        # The weight chosen for this month is the one that scored best on
        # the months before it, so the choice never sees the month.
        chosen = (max(scores, key=scores.get) if used_months else DEFAULT_PRIOR_WEIGHT)
        used_months += 1
        origin = grid.project(tests["From Latitude"].values, tests["From Longitude"].values)
        target_cells = grid.cell_of(grid.project(tests["To Latitude"].values,
                                                 tests["To Longitude"].values))
        month_scores = {w: 0.0 for w in PRIOR_WEIGHTS}
        for xy, days, target in zip(origin, tests["Gap (days)"].values, target_cells):
            if target < 0:
                continue
            if days not in profiles:
                profiles[days] = radial_profile(per_day, days, grid.cell_km)
            probs = {w: forecast(xy, days, grid, prior, per_day, w, profiles[days])
                     for w in PRIOR_WEIGHTS}
            for w, p in probs.items():
                month_scores[w] += float(np.log(p[target] + 1e-12))
            likely = zone(probs[chosen], LIKELY_MASS)
            area = likely.sum() * grid.cell_km ** 2
            # The same area as a disc around where the group was last seen.
            radius = np.sqrt(area / np.pi)
            stay = bool(np.linalg.norm(grid.centres[target] - xy) <= radius)
            hit = bool(likely[target])
            tested += 1
            forecast_hits += hit
            stay_hits += stay
            discordant += hit != stay
            areas.append(area)
        for w in PRIOR_WEIGHTS:
            scores[w] += month_scores[w]
    best = max(scores, key=scores.get) if used_months else DEFAULT_PRIOR_WEIGHT
    return TrackRecord(tested, forecast_hits, stay_hits,
                       float(np.mean(areas)) if areas else float("nan"), used_months,
                       horizon_days, discordant, best,
                       {w: v / max(tested, 1) for w, v in scores.items()})


# ---------------------------------------------------------------------------
# The groups to forecast now
# ---------------------------------------------------------------------------
SOCIAL_LABELS = {"lone bull": "Lone bull", "bull party": "Bull party",
                 "family herd": "Family herd", "unsexed group": "Group, sex not recorded"}


def next_moves(df: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int = 1,
               villages: Optional[pd.DataFrame] = None,
               active_days: int = ACTIVE_DAYS,
               prior_weight: float = DEFAULT_PRIOR_WEIGHT) -> Dict[str, object]:
    """Groups seen in the ``active_days`` before ``as_of``, and where they may be next.

    Returns ``groups`` (one row per unit, with its likely zone in square km
    and the villages in it), ``cells`` (cell centres with the combined
    chance, for the map), ``villages`` (villages in any likely or possible
    zone, worst first) and ``trails`` (each group's recent reports).
    """
    out = {"groups": pd.DataFrame(), "cells": pd.DataFrame(), "villages": pd.DataFrame(),
           "trails": pd.DataFrame(), "steps_learned": 0}
    history = df[df["Date"] <= as_of + pd.Timedelta(days=1)]
    if history.empty:
        return out
    units = assign_units(history)
    learned = steps(history, units)
    per_day = step_lengths(learned)
    out["steps_learned"] = len(per_day)
    work = history.assign(_unit=units.values)
    last = work.sort_values("Date").groupby("_unit").tail(1)
    active = last[last["Date"] >= as_of - pd.Timedelta(days=active_days - 1)]
    if active.empty:
        return out

    grid = Grid.around(history)
    prior = usage_prior(history, grid, as_of + pd.Timedelta(days=1))
    village_xy = village_cells = None
    if villages is not None and len(villages):
        located = villages.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
        village_xy = grid.project(located["Latitude"].values, located["Longitude"].values)
        village_cells = grid.cell_of(village_xy)

    combined_miss = np.ones(grid.nx * grid.ny)
    likely_any = np.zeros(grid.nx * grid.ny, dtype=bool)
    possible_any = np.zeros(grid.nx * grid.ny, dtype=bool)
    village_hits: Dict[int, Dict[str, object]] = {}
    rows, trails = [], []
    records = _records(active)
    for (row, rec) in zip(active.to_dict("records"), records):
        since = max((as_of.normalize() - pd.Timestamp(row["Date"]).normalize()).days, 0)
        days = min(since + horizon_days, MAX_HORIZON_DAYS)
        origin = grid.project(row["Latitude"], row["Longitude"])[0]
        prob = forecast(origin, days, grid, prior, per_day, prior_weight)
        likely, possible = zone(prob, LIKELY_MASS), zone(prob, POSSIBLE_MASS)
        likely_any |= likely
        possible_any |= possible
        combined_miss *= 1 - prob
        unit_rows = work[work["_unit"] == row["_unit"]].sort_values("Date").tail(6)
        label = f"G{int(row['_unit']) + 1:03d}"
        trails.append(unit_rows.assign(Group=label)[["Group", "Date", "Latitude", "Longitude"]])
        near: List[str] = []
        if village_xy is not None:
            in_likely = likely[np.clip(village_cells, 0, None)] & (village_cells >= 0)
            in_possible = possible[np.clip(village_cells, 0, None)] & (village_cells >= 0)
            for vi in np.flatnonzero(in_possible):
                name = str(located.iloc[vi]["Village"])
                d = float(np.linalg.norm(village_xy[vi] - origin))
                band = "Likely" if in_likely[vi] else "Possible"
                current = village_hits.get(vi)
                if current is None or (band == "Likely" and current["Band"] != "Likely"):
                    village_hits[vi] = {"Village": name, "Band": band, "Group": label,
                                        "Latitude": float(located.iloc[vi]["Latitude"]),
                                        "Longitude": float(located.iloc[vi]["Longitude"]),
                                        "From last report (km)": round(d, 1)}
                if in_likely[vi]:
                    near.append(name)
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
            "Reports": int((work["_unit"] == row["_unit"]).sum()),
            "Likely Zone (km2)": float(likely.sum() * grid.cell_km ** 2),
            "Villages in Likely Zone": ", ".join(sorted(set(near))[:6])
            + (f" and {len(set(near)) - 6} more" if len(set(near)) > 6 else ""),
            "Latitude": float(row["Latitude"]),
            "Longitude": float(row["Longitude"]),
        })

    latlon = grid.latlon()
    chance = 1 - combined_miss
    shown = possible_any
    out["cells"] = pd.DataFrame({
        "Latitude": latlon[shown, 0], "Longitude": latlon[shown, 1],
        "Chance": chance[shown], "Likely": likely_any[shown],
    })
    out["groups"] = pd.DataFrame(rows).sort_values(["Days Ago", "Group"]).reset_index(drop=True)
    out["trails"] = pd.concat(trails, ignore_index=True) if trails else pd.DataFrame()
    if village_hits:
        table = pd.DataFrame(village_hits.values())
        table["_order"] = (table["Band"] != "Likely").astype(int)
        out["villages"] = table.sort_values(["_order", "From last report (km)"]).drop(
            columns="_order").reset_index(drop=True)
    out["cell_km"] = grid.cell_km
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
