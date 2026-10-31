"""Emerging hotspots: is each place new, growing, steady or fading?

``core.hotspots`` answers *where* activity clusters over the whole
period. It cannot say whether a cluster appeared last month or has been
there for three years, and that is what decides the response: a new
hotspot wants investigation and early warning, a persistent one wants
structural measures, a diminishing one shows something worked.

This bins the reports into a space-time cube -- hexagonal cells on the
ground, fixed time steps on the third axis -- and runs the emerging
hotspot analysis on it:

1. Getis-Ord Gi* for every cell and step, over a space-time
   neighbourhood (the cell and its six neighbours, this step and the
   ``time_window`` before). Each bin is compared with the whole cube.
2. Optionally, a Benjamini-Hochberg false discovery rate correction
   across all bins, since a cube holds thousands of tests. A bin only
   counts as hot if the cell itself had a report in its time window:
   Gi* borrows from neighbours, and a manager shown an empty cell
   coloured as a hotspot stops trusting the map.
3. A Mann-Kendall test on each cell's Gi* z-score series for a trend,
   leaving out the first ``time_window`` steps, whose window is cut
   short by the start of the data and would read as a rise.
4. Each cell is labelled from its run of significant hot steps and that
   trend.

Labels follow the published emerging hotspot categories, hot side only:
with counts this sparse a cold spot is a quiet place, not a finding.
One is deliberately relaxed -- see :data:`HISTORICAL`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.analytics import conflict_mask
from core.config import (
    DEFAULT_CUBE_CELL_KM,
    DEFAULT_CUBE_STEP,
    DEFAULT_CUBE_TIME_WINDOW,
    KM_PER_DEG_LAT,
    KM_PER_DEG_LON_EQUATOR,
    MIN_CUBE_STEPS,
    PERSISTENT_SHARE,
    SIGNIFICANCE_ALPHA,
)

logger = logging.getLogger(__name__)

STEPS: Dict[str, int] = {"Week": 7, "Fortnight": 14, "Month": 0}  # 0: calendar month
METRICS = ("Conflict events", "All reports")

NEW = "New"
CONSECUTIVE = "Consecutive"
INTENSIFYING = "Intensifying"
PERSISTENT = "Persistent"
DIMINISHING = "Diminishing"
SPORADIC = "Sporadic"
# The published definition needs 90% of steps hot but not the latest.
# Here it is any place that was a significant hotspot earlier and is not
# now: "it used to be bad here" is worth a line to a manager even when
# it was bad for a season rather than for years.
HISTORICAL = "Historical"
NO_PATTERN = "No pattern"

# Worst first: the order the table, legend and brief use.
PATTERN_ORDER = [NEW, INTENSIFYING, PERSISTENT, CONSECUTIVE, SPORADIC, DIMINISHING,
                 HISTORICAL, NO_PATTERN]

PATTERN_MEANING: Dict[str, str] = {
    NEW: "Hot in the latest step for the first time.",
    INTENSIFYING: "Hot almost throughout, and getting hotter.",
    PERSISTENT: "Hot almost throughout, with no clear trend.",
    CONSECUTIVE: "An unbroken run of hot steps up to now, after none before.",
    SPORADIC: "Hot now, and on and off before.",
    DIMINISHING: "Hot almost throughout, but cooling.",
    HISTORICAL: "Hot earlier in the period, not in the latest step.",
    NO_PATTERN: "Never a significant hotspot.",
}

# Okabe-Ito based. New stands apart in hue because it is the call to
# act; the rest step from dark (worse) to grey (over).
PATTERN_COLORS: Dict[str, str] = {
    NEW: "#CC79A7",
    INTENSIFYING: "#5C1406",
    PERSISTENT: "#C2410C",
    CONSECUTIVE: "#E69F00",
    SPORADIC: "#F0C55A",
    DIMINISHING: "#56B4E9",
    HISTORICAL: "#8C9A93",
    NO_PATTERN: "#D9DED9",
}

_NEIGHBOURS = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


@dataclass
class Cube:
    """Reports binned by hex cell and time step.

    ``counts`` is cells x steps; ``cells`` carries each cell's axial
    coordinates, centre and the beats inside it, on the same index.
    """

    counts: pd.DataFrame
    cells: pd.DataFrame
    cell_km: float
    step: str
    metric: str
    reference_lat: float
    notes: List[str] = field(default_factory=list)

    @property
    def steps(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.counts.columns)


# ---------------------------------------------------------------------------
# The cube
# ---------------------------------------------------------------------------
def _hex_size(cell_km: float) -> float:
    """Centre-to-corner size of a pointy hex whose centres sit ``cell_km`` apart."""
    return cell_km / math.sqrt(3)


def _axial(x: np.ndarray, y: np.ndarray, size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Plane kilometres to the axial coordinates of the containing hex."""
    qf = (math.sqrt(3) / 3 * x - y / 3) / size
    rf = (2 / 3 * y) / size
    sf = -qf - rf
    q, r, s = np.round(qf), np.round(rf), np.round(sf)
    # Cube rounding: fix whichever coordinate rounded furthest.
    dq, dr, ds = np.abs(q - qf), np.abs(r - rf), np.abs(s - sf)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    q = np.where(fix_q, -r - s, q)
    r = np.where(fix_r, -q - s, r)
    return q.astype(int), r.astype(int)


def _centre_km(q, r, size: float) -> Tuple[np.ndarray, np.ndarray]:
    q, r = np.asarray(q, dtype=float), np.asarray(r, dtype=float)
    return size * math.sqrt(3) * (q + r / 2), size * 1.5 * r


def _lon_km(reference_lat: float) -> float:
    return KM_PER_DEG_LON_EQUATOR * math.cos(math.radians(reference_lat))


def _step_starts(dates: pd.Series, step: str) -> pd.Series:
    """The start of the time step each date falls in."""
    if STEPS[step] == 0:
        return dates.dt.to_period("M").dt.start_time
    monday = dates.dt.normalize() - pd.to_timedelta(dates.dt.dayofweek, unit="D")
    origin = monday.min()
    offset = ((monday - origin).dt.days // STEPS[step]) * STEPS[step]
    return origin + pd.to_timedelta(offset, unit="D")


def _step_end(start: pd.Timestamp, step: str) -> pd.Timestamp:
    """The last day of the step beginning at ``start``."""
    if STEPS[step] == 0:
        return start + pd.offsets.MonthEnd(0)
    return start + pd.Timedelta(days=STEPS[step] - 1)


def _all_steps(first: pd.Timestamp, last: pd.Timestamp, step: str) -> pd.DatetimeIndex:
    if STEPS[step] == 0:
        return pd.date_range(first, last, freq="MS")
    return pd.date_range(first, last, freq=f"{STEPS[step]}D")


def build_cube(
    df: pd.DataFrame,
    cell_km: float = DEFAULT_CUBE_CELL_KM,
    step: str = DEFAULT_CUBE_STEP,
    metric: str = METRICS[0],
    as_of: Optional[pd.Timestamp] = None,
) -> Cube:
    """Bin reports into hex cells by time step.

    Only cells that hold at least one counted report in some step are
    kept: a cell nothing ever happened in is not a location, and keeping
    the empty landscape would pull the cube's mean towards zero and make
    everything look hot.

    A trailing step less than half covered by the data (the period ends
    on the 5th of a month) is dropped and noted. Left in, its low count
    would read as every place cooling off at once.

    Args:
        df: Filtered sightings with ``Date``, ``Latitude``, ``Longitude``.
        cell_km: Distance between neighbouring cell centres.
        step: One of :data:`STEPS`.
        metric: ``"Conflict events"`` counts conflict reports only;
            ``"All reports"`` counts every sighting.
        as_of: End of the review period; defaults to the latest date.
    """
    if step not in STEPS:
        raise ValueError(f"step must be one of {list(STEPS)}, not {step!r}")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, not {metric!r}")

    empty = Cube(pd.DataFrame(), pd.DataFrame(), cell_km, step, metric, 0.0)
    needed = {"Date", "Latitude", "Longitude"}
    if df.empty or not needed.issubset(df.columns):
        return empty
    working = df.dropna(subset=list(needed))
    if metric == METRICS[0]:
        working = working[conflict_mask(working).reindex(working.index, fill_value=False)]
    if working.empty:
        return empty

    reference_lat = float(working["Latitude"].mean())
    size = _hex_size(cell_km)
    x = working["Longitude"].to_numpy(dtype=float) * _lon_km(reference_lat)
    y = working["Latitude"].to_numpy(dtype=float) * KM_PER_DEG_LAT
    q, r = _axial(x, y, size)

    binned = pd.DataFrame({
        "q": q, "r": r,
        "Step": _step_starts(pd.to_datetime(working["Date"]), step).to_numpy(),
        "Beat": working["Beat"].astype(str).to_numpy() if "Beat" in working else "",
    })
    binned["Cell"] = binned["q"].astype(str) + "," + binned["r"].astype(str)

    steps = _all_steps(binned["Step"].min(), binned["Step"].max(), step)
    counts = (
        binned.groupby(["Cell", "Step"]).size().unstack(fill_value=0)
        .reindex(columns=steps, fill_value=0)
    )

    notes: List[str] = []
    end = pd.Timestamp(as_of) if as_of is not None and not pd.isna(as_of) \
        else pd.Timestamp(df["Date"].max())
    last = steps[-1]
    span = (_step_end(last, step) - last).days + 1
    covered = (min(end.normalize(), _step_end(last, step)) - last).days + 1
    if len(steps) > 1 and covered / span < 0.5:
        counts = counts.iloc[:, :-1]
        counts = counts[counts.sum(axis=1) > 0]
        notes.append(
            f"The last {step.lower()} ({last:%d %b %Y}) is only {covered} of {span} days "
            "into the data, so it is left out rather than read as a drop."
        )

    beats = (
        binned[binned["Cell"].isin(counts.index)]
        .groupby("Cell")["Beat"]
        .agg(lambda b: ", ".join(b.value_counts().index[:3]))
    )
    cells = (
        binned.drop_duplicates("Cell").set_index("Cell")[["q", "r"]]
        .reindex(counts.index)
    )
    cx, cy = _centre_km(cells["q"], cells["r"], size)
    cells["Centre Latitude"] = cy / KM_PER_DEG_LAT
    cells["Centre Longitude"] = cx / _lon_km(reference_lat)
    cells["Beats"] = beats.reindex(cells.index).fillna("")

    return Cube(counts, cells, cell_km, step, metric, reference_lat, notes)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def gi_star(cube: Cube, time_window: int = DEFAULT_CUBE_TIME_WINDOW) -> pd.DataFrame:
    """Getis-Ord Gi* z-score for every cell and step of the cube.

    Neighbours are the cell itself and the adjacent hexes that are in
    the cube, at this step and the ``time_window`` steps before it, all
    weighted one. The mean and spread are the whole cube's.
    """
    x = cube.counts.to_numpy(dtype=float)
    n_cells, n_steps = x.shape
    if n_cells == 0 or n_steps == 0:
        return pd.DataFrame(index=cube.counts.index, columns=cube.counts.columns, dtype=float)

    position = {(int(q), int(r)): i for i, (q, r) in
                enumerate(cube.cells[["q", "r"]].itertuples(index=False))}
    spatial = np.zeros_like(x)
    spatial_w = np.zeros(n_cells)
    for (q, r), i in position.items():
        for dq, dr in _NEIGHBOURS:
            j = position.get((q + dq, r + dr))
            if j is not None:
                spatial[i] += x[j]
                spatial_w[i] += 1

    # Running sum over the time window, truncated at the first step.
    cumulative = np.cumsum(spatial, axis=1)
    lagged = np.zeros_like(cumulative)
    if time_window + 1 < n_steps:
        lagged[:, time_window + 1:] = cumulative[:, :-(time_window + 1)]
    local = cumulative - lagged
    window_len = np.minimum(np.arange(n_steps) + 1, time_window + 1)
    weight = spatial_w[:, None] * window_len[None, :]

    n = x.size
    mean = x.mean()
    spread = math.sqrt(max((x ** 2).mean() - mean ** 2, 0.0))
    if spread == 0 or n < 2:
        z = np.zeros_like(x)
    else:
        denominator = spread * np.sqrt((n * weight - weight ** 2) / (n - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(denominator > 0, (local - mean * weight) / denominator, 0.0)
    return pd.DataFrame(z, index=cube.counts.index, columns=cube.counts.columns)


def _hot_mask(z: np.ndarray, fdr: bool, alpha: float = SIGNIFICANCE_ALPHA) -> np.ndarray:
    """Significant positive Gi* bins, optionally under a Benjamini-Hochberg FDR."""
    p = 2 * norm.sf(np.abs(z))
    if not fdr:
        return (z > 0) & (p <= alpha)
    flat = np.sort(p.ravel())
    ranks = np.arange(1, flat.size + 1)
    passing = flat <= alpha * ranks / flat.size
    cutoff = flat[passing].max() if passing.any() else -1.0
    return (z > 0) & (p <= cutoff)


def hot_bins(cube: Cube, time_window: int = DEFAULT_CUBE_TIME_WINDOW, fdr: bool = True,
             z: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Which cell-steps are significant hotspots, on the cube's index and columns.

    A bin counts only if the cell itself had a report in its time window,
    so Gi* borrowing from busy neighbours never marks an empty cell.
    """
    if cube.counts.empty:
        return pd.DataFrame(index=cube.counts.index, columns=cube.counts.columns, dtype=bool)
    if z is None:
        z = gi_star(cube, time_window=time_window)
    own = cube.counts.T.rolling(time_window + 1, min_periods=1).sum().T.to_numpy()
    return pd.DataFrame(_hot_mask(z.to_numpy(), fdr=fdr) & (own > 0),
                        index=cube.counts.index, columns=cube.counts.columns)


def mann_kendall(series: np.ndarray) -> Tuple[float, float]:
    """Mann-Kendall trend test with the ties correction: (z, two-sided p)."""
    values = np.asarray(series, dtype=float)
    n = len(values)
    if n < 3:
        return 0.0, 1.0
    diffs = np.sign(values[None, :] - values[:, None])
    s = float(np.triu(diffs, k=1).sum())
    _, ties = np.unique(values, return_counts=True)
    variance = (n * (n - 1) * (2 * n + 5)
                - float((ties * (ties - 1) * (2 * ties + 5)).sum())) / 18.0
    if variance <= 0:
        return 0.0, 1.0
    z = (s - np.sign(s)) / math.sqrt(variance)
    return float(z), float(2 * norm.sf(abs(z)))


def classify_pattern(hot: np.ndarray, trend_z: float, trend_p: float,
                     alpha: float = SIGNIFICANCE_ALPHA) -> str:
    """One cell's emerging hotspot label from its hot steps and Gi* trend."""
    hot = np.asarray(hot, dtype=bool)
    n = len(hot)
    if n == 0 or not hot.any():
        return NO_PATTERN
    share = hot.sum() / n
    rising = trend_p <= alpha and trend_z > 0
    falling = trend_p <= alpha and trend_z < 0

    if not hot[-1]:
        return HISTORICAL
    if hot.sum() == 1:
        return NEW
    if share >= PERSISTENT_SHARE:
        if rising:
            return INTENSIFYING
        if falling:
            return DIMINISHING
        return PERSISTENT
    # The run of hot steps ending now, and whether anything was hot before it.
    run = n - np.flatnonzero(~hot).max() - 1 if (~hot).any() else n
    if run >= 2 and not hot[:n - run].any():
        return CONSECUTIVE
    return SPORADIC


def _pattern_columns() -> List[str]:
    return [
        "Place", "Pattern", "Nearest Village", "Beats", "Events", "Recent Events",
        "Hot Steps", "Steps", "Trend", "First Hot", "Last Hot",
        "Centre Latitude", "Centre Longitude", "Cell",
    ]


def emerging_hotspots(
    cube: Cube,
    time_window: int = DEFAULT_CUBE_TIME_WINDOW,
    fdr: bool = True,
    villages: Optional[pd.DataFrame] = None,
    recent_steps: int = 3,
) -> pd.DataFrame:
    """Label every cell of the cube with its emerging hotspot pattern.

    Returns:
        One row per cell, worst pattern first, then most recent events.
        ``Place`` names cells with a pattern ``E1``, ``E2``... in that
        order; cells with no pattern keep their row so the counts add up
        but are not named. ``Events`` and ``Recent Events`` (the last
        ``recent_steps`` steps) count whatever the cube's metric is.
        ``Trend`` is ``Rising``, ``Falling`` or ``Flat`` by Mann-Kendall
        on the Gi* series. Fewer than :data:`MIN_CUBE_STEPS` steps
        returns an empty frame: there is no "emerging" in three months.
    """
    columns = _pattern_columns()
    if cube.counts.empty or cube.counts.shape[1] < MIN_CUBE_STEPS:
        return pd.DataFrame(columns=columns)

    z = gi_star(cube, time_window=time_window)
    hot = hot_bins(cube, time_window=time_window, fdr=fdr, z=z).to_numpy()
    steps = cube.steps

    rows = []
    for i, cell in enumerate(z.index):
        trend_z, trend_p = mann_kendall(z.iloc[i].to_numpy()[time_window:])
        pattern = classify_pattern(hot[i], trend_z, trend_p)
        hot_at = np.flatnonzero(hot[i])
        counts = cube.counts.iloc[i]
        rows.append({
            "Cell": cell,
            "Pattern": pattern,
            "Beats": cube.cells.at[cell, "Beats"],
            "Events": int(counts.sum()),
            "Recent Events": int(counts.iloc[-recent_steps:].sum()),
            "Hot Steps": int(hot[i].sum()),
            "Steps": len(steps),
            "Trend": ("Rising" if trend_p <= SIGNIFICANCE_ALPHA and trend_z > 0
                      else "Falling" if trend_p <= SIGNIFICANCE_ALPHA and trend_z < 0
                      else "Flat"),
            "First Hot": steps[hot_at[0]] if len(hot_at) else pd.NaT,
            "Last Hot": steps[hot_at[-1]] if len(hot_at) else pd.NaT,
            "Centre Latitude": float(cube.cells.at[cell, "Centre Latitude"]),
            "Centre Longitude": float(cube.cells.at[cell, "Centre Longitude"]),
        })

    table = pd.DataFrame(rows)
    table["_rank"] = table["Pattern"].map({p: k for k, p in enumerate(PATTERN_ORDER)})
    table = (
        table.sort_values(["_rank", "Recent Events", "Events"], ascending=[True, False, False])
        .drop(columns="_rank").reset_index(drop=True)
    )
    named = table["Pattern"] != NO_PATTERN
    table["Place"] = ""
    table.loc[named, "Place"] = [f"E{k + 1}" for k in range(int(named.sum()))]
    table["Nearest Village"] = _nearest_villages(table, villages)

    logger.info("Emerging hotspots: %s", table["Pattern"].value_counts().to_dict())
    return table[columns]


def _nearest_villages(table: pd.DataFrame, villages: Optional[pd.DataFrame]) -> pd.Series:
    if villages is None or villages.empty or table.empty:
        return pd.Series("", index=table.index)
    from core.spatial import attach_nearest_village

    centres = table[["Centre Latitude", "Centre Longitude"]].rename(
        columns={"Centre Latitude": "Latitude", "Centre Longitude": "Longitude"})
    near, _ = attach_nearest_village(centres, villages)
    return near["Nearest Village"].astype(str) + " (" + \
        near["Distance to Village (km)"].map("{:.1f} km".format) + ")"


# ---------------------------------------------------------------------------
# Views of the result
# ---------------------------------------------------------------------------
def pattern_counts(table: pd.DataFrame) -> Dict[str, int]:
    """Cells per pattern, worst first, patterns with no cells left out."""
    if table.empty:
        return {}
    counts = table["Pattern"].value_counts()
    return {p: int(counts[p]) for p in PATTERN_ORDER if counts.get(p, 0)}


def location_time_matrix(cube: Cube, table: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Events per step for the worst named places, rows in table order.

    Row labels carry the place, its pattern and village, so the heatmap
    reads without the table beside it.
    """
    if table.empty or cube.counts.empty:
        return pd.DataFrame()
    picked = table[table["Place"] != ""].head(limit)
    if picked.empty:
        return pd.DataFrame()
    matrix = cube.counts.loc[picked["Cell"]].copy()
    labels = []
    for row in picked.to_dict("records"):
        where = str(row["Nearest Village"]).split(" (")[0] or str(row["Beats"]).split(",")[0]
        labels.append(f"{row['Place']} {row['Pattern']}" + (f" - {where}" if where else ""))
    matrix.index = labels
    return matrix


def hex_polygon(lat: float, lon: float, cell_km: float, reference_lat: float) -> List[List[float]]:
    """Corners of one pointy hex as ``[lon, lat]`` pairs, for the map."""
    size = _hex_size(cell_km)
    corners = []
    for k in range(6):
        angle = math.radians(60 * k - 30)
        corners.append([
            lon + size * math.cos(angle) / _lon_km(reference_lat),
            lat + size * math.sin(angle) / KM_PER_DEG_LAT,
        ])
    return corners


def headline(table: pd.DataFrame) -> Optional[str]:
    """One sentence for the assessment: what is new or getting worse."""
    counts = pattern_counts(table)
    acting = [(p, counts[p]) for p in (NEW, INTENSIFYING, CONSECUTIVE) if counts.get(p)]
    if not acting:
        return None
    parts = [f"{n} {p.lower()}" for p, n in acting]
    worst = table[table["Pattern"].isin([p for p, _ in acting])].head(3)
    where = [str(v).split(" (")[0] for v in worst["Nearest Village"] if v]
    text = (", ".join(parts[:-1]) + " and " + parts[-1]) if len(parts) > 1 else parts[0]
    noun = "place" if sum(n for _, n in acting) == 1 else "places"
    return (f"Space-time trend: {text} hotspot {noun}"
            + (f", worst near {', '.join(where)}." if where else "."))


def pattern_caveats(cube: Cube, table: pd.DataFrame, fdr: bool) -> List[str]:
    """What a reader must know before acting on a pattern."""
    notes = list(cube.notes)
    if cube.counts.empty:
        return notes
    n_steps = cube.counts.shape[1]
    if n_steps < MIN_CUBE_STEPS:
        notes.append(
            f"Only {n_steps} {cube.step.lower()} step(s) in the period; at least "
            f"{MIN_CUBE_STEPS} are needed to call anything emerging. Shorten the step "
            "or widen the date range."
        )
        return notes
    if n_steps < 10:
        notes.append(
            f"{n_steps} steps is short for a trend test, so Intensifying and Diminishing "
            "need a strong, steady change to show up."
        )
    sparse = float((cube.counts.to_numpy() > 0).mean())
    if sparse < 0.05:
        notes.append(
            f"Only {sparse:.0%} of cell-steps hold any event. A larger cell or a longer "
            "step gives the statistics more to work with."
        )
    notes.append(
        "Each place is compared with the whole cube, so a pattern means unusual for "
        "this landscape and period, not unusual in absolute terms. It also inherits "
        "reporting effort: a beat that starts reporting diligently can look New."
    )
    if not fdr:
        notes.append(
            "Without the false discovery rate correction about 1 in 40 quiet cell-steps "
            "comes up hot by chance."
        )
    return notes
