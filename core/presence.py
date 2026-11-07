"""Where elephants range, from presence-only reports.

The register holds presence reports only. Nobody files "looked here,
saw nothing", so a quiet place may be empty or just unwatched. Reports
pile up where people are, so a raw map of them is partly a map of
people. This module models the reports so that the two can be
separated, as far as the register allows:

* **Thinning.** One record per grid cell per day. Five reports of the
  same herd in the same square on the same day are one piece of
  evidence, not five.
* **Fortnight blocks.** Records are counted per cell per fortnight,
  ending on the period's last day, which evens out days with no reports.
* **Effort-corrected point process.** The counts follow a Poisson point
  process. Its log intensity adds habitat (notified forest), reporting
  effort (distance from the nearest village, and where registered
  villagers and app users are, when those files are loaded), a smooth
  spatial surface (an intrinsic CAR penalty on neighbouring cells),
  seasonal shifts of that surface, and optionally the recent past
  (last fortnight's records nearby, and long-term use). Fitting it
  conditional on each fortnight's total -- a multinomial over cells --
  makes it immune to the register growing as more people use the app.
  With the effort terms held level everywhere, what is left is where
  elephants are, not where they are seen from: the effort-corrected
  surface (Warton, Renner & Ramp 2013). MaxEnt fits the same model
  (Renner & Warton 2013), so nothing separate is needed for it.
* **Tested forecasts.** Every fortnight among the last few is forecast
  only from the fortnights before it, and scored against simple maps
  (last fortnight's reports, all past reports). Which version of the
  model forecasts is chosen the same way, so the track record shown
  is the record of the model as used.

What the register cannot support is stated rather than modelled around.
Dynamic occupancy models (colonisation and extinction rates) need
repeat visits that record non-detections, which the register does not
have. The dynamic terms here are their presence-only analogue: how much
a cell's chance of reports follows its own and its neighbours' last
fortnight. Graph neural networks need far more data than a district
register produces. Distance from villages is both reporting effort and
a real pull (crops), so correcting for it also removes some real use;
app-open pings or patrol tracks would be cleaner effort layers, and
any CSV of such points can be added.

Corridors between the most-used places are worked out in
``core.corridors`` from the effort-corrected surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial import QhullError

try:  # BLAS threads spinning between the optimiser's small steps cost 5x
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - the fit is slower, not wrong
    from contextlib import nullcontext

    def threadpool_limits(*_args, **_kwargs):
        return nullcontext()

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320

CELL_KM = 2.0
BLOCK_DAYS = 14
# The landscape the register speaks for: the convex hull of the reports
# and this far beyond. The hull keeps the land between two clusters of
# reports in, so a corridor across it can be traced.
DOMAIN_KM = 10.0
# Above this many cells the grid coarsens, to keep a fit to seconds.
MAX_CELLS = 9_000
# Fortnights at the start whose recent-past terms are still filling up.
BURN_IN = 2
# Fortnights of history before the first one tested.
MIN_FIT_BLOCKS = 6
MIN_RECORDS = 40
TEST_BLOCKS = 12
MIN_TESTED = 4
# The fit reads at most this many fortnights (two years).
MAX_FIT_BLOCKS = 52
# In testing, refit every fortnight while a fit is cheap; on a large
# landscape every month or six weeks. A forecast only ever uses a fit on
# the fortnights before it, whatever the cadence.
REFIT_WORK = 150_000  # cells x fortnights per fit before the cadence stretches
# History needed before seasonal shifts are fitted.
SEASONAL_DAYS = 300

NEAR_KM = 2.5
WIDE_KM = 7.0
MEMORY_HALF_LIFE = 6  # fortnights
LAG_FLOOR = 0.1
FOREST_CONTEXT_KM = 5.0

AREA_SHARES = (0.05, 0.10, 0.25)
HEADLINE_SHARE = 0.10
UNIFORM_FLOOR = 0.01

# Penalties, in units of the log-likelihood.
SMOOTH = 2.0          # neighbouring cells' spatial effects
SEASON_SMOOTH = 2.0   # and their seasonal shifts
RIDGE = 0.02          # every cell effect towards the covariates alone
SEASON_RIDGE = 0.5    # seasonal shifts towards none
COEF_RIDGE = 0.01

SEASONS: Dict[str, Tuple[int, ...]] = {
    "Winter (Oct-Feb)": (10, 11, 12, 1, 2),
    "Summer (Mar-May)": (3, 4, 5),
    "Monsoon (Jun-Sep)": (6, 7, 8, 9),
}
ALL_YEAR = "All year"

VILLAGE_KM = "village_km"
FOREST = "forest"
FOREST_NEAR = "forest_near"
LABELS = {
    VILLAGE_KM: "Distance from the nearest village",
    FOREST: "Notified forest in the cell",
    FOREST_NEAR: f"Notified forest within {FOREST_CONTEXT_KM:g} km",
}
RECENT, NEARBY, MEMORY, HISTORY = "recent", "nearby", "memory", "history"
DYNAMIC = (RECENT, NEARBY, MEMORY, HISTORY)
DYNAMIC_LABELS = {
    RECENT: "Records here last fortnight",
    NEARBY: f"Records within about {WIDE_KM:g} km last fortnight",
    MEMORY: "Use over the last few months (half-life of about 3 months)",
    HISTORY: "Every earlier record here",
}


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
@dataclass
class Lattice:
    """Square cells over the landscape the register covers.

    A local equirectangular projection about the reports' centre: over
    a district the scale error is a fraction of a percent.
    """

    cell_km: float
    lat_c: float
    lon_c: float
    x0: float
    y0: float
    ix: np.ndarray
    iy: np.ndarray
    lookup: np.ndarray  # (nx, ny) -> cell index or -1

    @property
    def size(self) -> int:
        return len(self.ix)

    @property
    def area_km2(self) -> float:
        return self.cell_km ** 2

    def _k_lon(self) -> float:
        return KM_PER_DEG_LON * math.cos(math.radians(self.lat_c))

    def xy(self, lat, lon) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        return (lon - self.lon_c) * self._k_lon(), (lat - self.lat_c) * KM_PER_DEG_LAT

    @property
    def x(self) -> np.ndarray:
        return self.x0 + (self.ix + 0.5) * self.cell_km

    @property
    def y(self) -> np.ndarray:
        return self.y0 + (self.iy + 0.5) * self.cell_km

    @property
    def lat(self) -> np.ndarray:
        return self.lat_c + self.y / KM_PER_DEG_LAT

    @property
    def lon(self) -> np.ndarray:
        return self.lon_c + self.x / self._k_lon()

    def locate(self, lat, lon) -> np.ndarray:
        """The cell holding each point, or -1 outside the landscape."""
        x, y = self.xy(lat, lon)
        out = np.full(len(x), -1, dtype=int)
        ok = np.isfinite(x) & np.isfinite(y)
        gx = np.floor((x[ok] - self.x0) / self.cell_km).astype(int)
        gy = np.floor((y[ok] - self.y0) / self.cell_km).astype(int)
        nx, ny = self.lookup.shape
        inside = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
        found = np.full(len(gx), -1, dtype=int)
        found[inside] = self.lookup[gx[inside], gy[inside]]
        out[np.flatnonzero(ok)] = found
        return out

    def neighbours(self, diagonal: bool = False) -> sparse.csr_matrix:
        """Which cells touch: edges (rook), or edges and corners (queen)."""
        steps = [(1, 0), (0, 1)] + ([(1, 1), (1, -1)] if diagonal else [])
        nx, ny = self.lookup.shape
        rows, cols = [], []
        for dx, dy in steps:
            jx, jy = self.ix + dx, self.iy + dy
            ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
            j = np.full(self.size, -1)
            j[ok] = self.lookup[jx[ok], jy[ok]]
            keep = j >= 0
            rows.append(np.flatnonzero(keep))
            cols.append(j[keep])
        r = np.concatenate(rows) if rows else np.empty(0, int)
        c = np.concatenate(cols) if cols else np.empty(0, int)
        data = np.ones(len(r))
        upper = sparse.coo_matrix((data, (r, c)), shape=(self.size, self.size))
        return (upper + upper.T).tocsr()

    def kernel(self, sd_km: float, reach: float = 2.5) -> sparse.csr_matrix:
        """Gaussian weights between cells, 1 at the cell itself."""
        points = np.column_stack([self.x, self.y])
        tree = cKDTree(points)
        pairs = tree.sparse_distance_matrix(tree, sd_km * reach, output_type="coo_matrix")
        weights = np.exp(-0.5 * (pairs.data / sd_km) ** 2)
        k = sparse.coo_matrix((weights, (pairs.row, pairs.col)), shape=(self.size, self.size))
        return (k + sparse.identity(self.size, format="coo")).tocsr()


def _inside_hull(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    try:
        return Delaunay(points).find_simplex(centres) >= 0
    except (QhullError, ValueError):  # fewer than three points, or all in a line
        return np.zeros(len(centres), dtype=bool)


def lattice_for(lat: Sequence[float], lon: Sequence[float], cell_km: float = CELL_KM,
                domain_km: float = DOMAIN_KM, max_cells: int = MAX_CELLS) -> Lattice:
    """The cells inside the reports' convex hull or within ``domain_km`` of
    a report, coarsened if too many."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[ok], lon[ok]
    if not len(lat):
        raise ValueError("No report has coordinates.")
    lat_c = float((lat.min() + lat.max()) / 2)
    lon_c = float((lon.min() + lon.max()) / 2)
    k_lon = KM_PER_DEG_LON * math.cos(math.radians(lat_c))
    px, py = (lon - lon_c) * k_lon, (lat - lat_c) * KM_PER_DEG_LAT
    tree = cKDTree(np.column_stack([px, py]))
    while True:
        x0 = math.floor((px.min() - domain_km) / cell_km) * cell_km
        y0 = math.floor((py.min() - domain_km) / cell_km) * cell_km
        nx = int(math.ceil((px.max() + domain_km - x0) / cell_km))
        ny = int(math.ceil((py.max() + domain_km - y0) / cell_km))
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        gx, gy = gx.ravel(), gy.ravel()
        centres = np.column_stack([x0 + (gx + 0.5) * cell_km, y0 + (gy + 0.5) * cell_km])
        distance, _ = tree.query(centres, distance_upper_bound=domain_km + cell_km / 2)
        keep = np.isfinite(distance) | _inside_hull(np.column_stack([px, py]), centres)
        if keep.sum() <= max_cells:
            break
        cell_km = math.ceil(cell_km * math.sqrt(keep.sum() / max_cells) * 2) / 2
    lookup = np.full((nx, ny), -1, dtype=int)
    lookup[gx[keep], gy[keep]] = np.arange(int(keep.sum()))
    return Lattice(cell_km=float(cell_km), lat_c=lat_c, lon_c=lon_c, x0=float(x0),
                   y0=float(y0), ix=gx[keep], iy=gy[keep], lookup=lookup)


# ---------------------------------------------------------------------------
# Fortnights
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Blocks:
    """Consecutive blocks of ``days`` ending the day after ``as_of``."""

    end: pd.Timestamp  # exclusive
    count: int
    days: int = BLOCK_DAYS

    def start(self, t: int) -> pd.Timestamp:
        return self.end - pd.Timedelta(days=self.days * (self.count - t))

    def index(self, dates: pd.Series) -> np.ndarray:
        age = (self.end - pd.to_datetime(dates).dt.normalize()).dt.days.to_numpy()
        return self.count - 1 - (age - 1) // self.days

    def middles(self, extra: int = 0) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([self.start(t) + pd.Timedelta(days=self.days / 2)
                                 for t in range(self.count + extra)])


def blocks_for(first: pd.Timestamp, as_of: pd.Timestamp, days: int = BLOCK_DAYS) -> Blocks:
    end = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
    span = (end - pd.Timestamp(first).normalize()).days
    return Blocks(end=end, count=max(1, int(math.ceil(span / days))), days=days)


def _angle(dates: pd.DatetimeIndex) -> np.ndarray:
    return 2 * np.pi * (dates.dayofyear.to_numpy() - 1) / 365.25


# ---------------------------------------------------------------------------
# Inputs: thinned counts and covariates
# ---------------------------------------------------------------------------
@dataclass
class Inputs:
    lattice: Lattice
    blocks: Blocks
    counts: np.ndarray            # (T, C) thinned records
    reports: int                  # reports with coordinates read
    records: int                  # after thinning
    outside: int                  # reports beyond the grid (none, by construction)
    names: List[str]              # static covariates
    labels: Dict[str, str]
    effort: List[str]             # the static covariates that are reporting effort
    raw: np.ndarray               # (C, p) as measured
    static: np.ndarray            # (C, p) standardised
    centre: np.ndarray
    scale: np.ndarray
    dynamic: Dict[str, np.ndarray]  # name -> (T + 1, C): row t from blocks before t
    cos: np.ndarray               # (T + 1,)
    sin: np.ndarray
    laplacian: sparse.csr_matrix
    near: sparse.csr_matrix
    seasonal: bool
    first_report: pd.Timestamp
    last_report: pd.Timestamp


def thin(df: pd.DataFrame, lattice: Lattice) -> pd.DataFrame:
    """One record per cell per day: the same herd reported five times in
    one square on one day is one piece of evidence."""
    cell = lattice.locate(df["Latitude"].to_numpy(float), df["Longitude"].to_numpy(float))
    frame = pd.DataFrame({"Cell": cell, "Day": pd.to_datetime(df["Date"]).dt.normalize()})
    return frame[frame["Cell"] >= 0].drop_duplicates().reset_index(drop=True)


def forest_share(lat: np.ndarray, lon: np.ndarray, cell_km: float) -> np.ndarray:
    """Share of each cell inside notified forest (vendored beat polygons),
    from a 3 x 3 sample of points per cell."""
    from core import boundaries

    if not boundaries.available():
        return np.zeros(len(lat))
    offsets = (np.arange(3) - 1) * cell_km / 3
    dlat = offsets / KM_PER_DEG_LAT
    shares = np.zeros(len(lat))
    for oy in dlat:
        for ox in offsets:
            dlon = ox / (KM_PER_DEG_LON * np.cos(np.radians(lat)))
            inside = boundaries.locate(boundaries.BEAT, lon + dlon, lat + oy) != ""
            shares += inside
    return shares / 9.0


Habitat = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def _point_density(lattice: Lattice, points: pd.DataFrame, kernel: sparse.csr_matrix
                   ) -> Optional[np.ndarray]:
    if points is None or not {"Latitude", "Longitude"} <= set(points.columns):
        return None
    cell = lattice.locate(points["Latitude"].to_numpy(float),
                          points["Longitude"].to_numpy(float))
    cell = cell[cell >= 0]
    if len(cell) < 10:
        return None
    per_cell = np.bincount(cell, minlength=lattice.size).astype(float)
    return np.log1p(kernel @ per_cell)


def _dynamic(counts: np.ndarray, near: sparse.csr_matrix, wide: sparse.csr_matrix
             ) -> Dict[str, np.ndarray]:
    """Row t of each holds what was known before block t (row T: the next)."""
    t, c = counts.shape
    previous = np.vstack([np.zeros((1, c)), counts])           # (T + 1, C)
    decay = 0.5 ** (1 / MEMORY_HALF_LIFE)
    memory = np.zeros_like(previous)
    for k in range(1, t + 1):
        memory[k] = decay * memory[k - 1] + counts[k - 1]
    history = np.cumsum(previous, axis=0)
    # log(floor + weighted records): a cell with nothing nearby sits well
    # below one with a single record, so the terms can say "nothing here"
    # as firmly as last fortnight's map does.
    return {
        RECENT: np.log(LAG_FLOOR + (near @ previous.T).T),
        NEARBY: np.log(LAG_FLOOR + (wide @ previous.T).T),
        MEMORY: np.log(LAG_FLOOR + (near @ memory.T).T),
        HISTORY: np.log(LAG_FLOOR + (near @ history.T).T),
    }


def prepare(df: pd.DataFrame, as_of, villages: Optional[pd.DataFrame] = None,
            effort_layers: Optional[Mapping[str, pd.DataFrame]] = None,
            habitat: Optional[Habitat] = forest_share, cell_km: float = CELL_KM,
            block_days: int = BLOCK_DAYS, domain_km: float = DOMAIN_KM,
            village_effort: bool = True) -> Inputs:
    """Thinned fortnightly counts and the covariates for every cell.

    ``effort_layers`` maps a name ("App users", "Registered villagers",
    "Uploaded effort") to points where people who report are; each is a
    reporting-effort covariate. ``village_effort`` decides whether the
    distance from the nearest village counts as effort (corrected for)
    or as habitat (kept).
    """
    as_of = pd.Timestamp(as_of).normalize()
    dates = pd.to_datetime(df["Date"])
    frame = df[(dates < as_of + pd.Timedelta(days=1))
               & df["Latitude"].notna() & df["Longitude"].notna()]
    if frame.empty:
        raise ValueError("No report with coordinates up to this date.")
    lattice = lattice_for(frame["Latitude"], frame["Longitude"], cell_km, domain_km)
    thinned = thin(frame, lattice)
    blocks = blocks_for(pd.to_datetime(frame["Date"]).min(), as_of, block_days)
    block = blocks.index(thinned["Day"])
    counts = np.zeros((blocks.count, lattice.size))
    np.add.at(counts, (block, thinned["Cell"].to_numpy()), 1.0)

    near = lattice.kernel(NEAR_KM)
    wide = lattice.kernel(WIDE_KM)
    names: List[str] = []
    labels: Dict[str, str] = {}
    effort: List[str] = []
    columns: List[np.ndarray] = []

    if villages is not None and len(villages):
        vx, vy = lattice.xy(villages["Latitude"].to_numpy(float),
                            villages["Longitude"].to_numpy(float))
        ok = np.isfinite(vx) & np.isfinite(vy)
        if ok.sum():
            distance, _ = cKDTree(np.column_stack([vx[ok], vy[ok]])).query(
                np.column_stack([lattice.x, lattice.y]))
            names.append(VILLAGE_KM)
            columns.append(np.log(0.5 + distance))
            if village_effort:
                effort.append(VILLAGE_KM)
    if habitat is not None:
        share = np.asarray(habitat(lattice.lat, lattice.lon, lattice.cell_km), dtype=float)
        if share.std() > 1e-6 and (share > 0).mean() >= 0.02:
            names.append(FOREST)
            columns.append(share)
            context = lattice.kernel(FOREST_CONTEXT_KM)
            weight = np.asarray(context.sum(axis=1)).ravel()
            names.append(FOREST_NEAR)
            columns.append((context @ share) / weight)
    for name, points in (effort_layers or {}).items():
        density = _point_density(lattice, points, near)
        if density is not None and density.std() > 1e-6:
            names.append(name)
            columns.append(density)
            effort.append(name)
    for name in names:
        labels[name] = LABELS.get(name, f"{name} nearby")

    raw = np.column_stack(columns) if columns else np.zeros((lattice.size, 0))
    centre = raw.mean(axis=0) if columns else np.zeros(0)
    scale = raw.std(axis=0) if columns else np.zeros(0)
    scale = np.where(scale > 0, scale, 1.0)
    static = (raw - centre) / scale

    rook = lattice.neighbours()
    degree = np.asarray(rook.sum(axis=1)).ravel()
    laplacian = (sparse.diags(degree) - rook).tocsr()
    angle = _angle(blocks.middles(extra=1))
    first = pd.to_datetime(frame["Date"]).min()
    last = pd.to_datetime(frame["Date"]).max()
    return Inputs(
        lattice=lattice, blocks=blocks, counts=counts, reports=len(frame),
        records=len(thinned), outside=0, names=names, labels=labels, effort=effort,
        raw=raw, static=static, centre=centre, scale=scale,
        dynamic=_dynamic(counts, near, wide), cos=np.cos(angle), sin=np.sin(angle),
        laplacian=laplacian, near=near, seasonal=(last - first).days >= SEASONAL_DAYS,
        first_report=first, last_report=last,
    )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Spec:
    """One version of the model."""

    dynamic: bool = True
    seasonal: bool = True
    smooth: float = SMOOTH
    # A smooth surface of use of its own, beyond what the covariates and
    # the recent past say. Without it the model follows a moving herd
    # rather than where herds used to be.
    spatial: bool = True

    def parts(self) -> List[str]:
        out = ["habitat and effort"]
        if self.spatial:
            out.append("a smooth map of use")
        if self.seasonal:
            out.append("seasonal shifts")
        if self.dynamic:
            out.append("the recent past")
        return out

    @property
    def name(self) -> str:
        return ("dynamic" if self.dynamic else "static") + (" seasonal" if self.seasonal else "") \
            + ("" if self.spatial else " no-map")


@dataclass
class Fit:
    spec: Spec
    beta: np.ndarray
    gamma: np.ndarray
    s: np.ndarray
    u: np.ndarray
    v: np.ndarray
    blocks: Tuple[int, int]  # fitted on [first, last)
    iterations: int = 0
    converged: bool = True

    def theta(self) -> np.ndarray:
        parts = [self.beta, self.gamma]
        if self.spec.spatial:
            parts.append(self.s)
            if self.spec.seasonal:
                parts += [self.u, self.v]
        return np.concatenate(parts)


def _layout(inputs: Inputs, spec: Spec) -> Tuple[int, int, int]:
    p = inputs.static.shape[1]
    q = len(DYNAMIC) if spec.dynamic else 0
    return p, q, inputs.lattice.size if spec.spatial else 0


def _split(theta: np.ndarray, p: int, q: int, c: int, seasonal: bool, cells: int):
    beta = theta[:p]
    gamma = theta[p:p + q]
    s = theta[p + q:p + q + c] if c else np.zeros(cells)
    if seasonal and c:
        u = theta[p + q + c:p + q + 2 * c]
        v = theta[p + q + 2 * c:p + q + 3 * c]
    else:
        u = v = np.zeros(cells)
    return beta, gamma, s, u, v


def _eta(inputs: Inputs, spec: Spec, rows: np.ndarray, beta, gamma, s, u, v,
         dyn: Optional[np.ndarray] = None) -> np.ndarray:
    """Log intensity for blocks ``rows``, up to each block's constant."""
    base = inputs.static @ beta + s if len(beta) else s
    if spec.dynamic:
        if dyn is None:
            dyn = np.stack([inputs.dynamic[name][rows] for name in DYNAMIC])
        eta = np.tensordot(gamma, dyn, axes=1)
        eta += base[None, :]
    else:
        eta = np.repeat(base[None, :], len(rows), axis=0)
    if spec.seasonal:
        eta += np.outer(inputs.cos[rows], u)
        eta += np.outer(inputs.sin[rows], v)
    return eta


def _softmax(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    top = eta.max(axis=-1, keepdims=True)
    e = np.exp(eta - top)
    total = e.sum(axis=-1, keepdims=True)
    return e / total, (np.log(total) + top)[..., 0]


def fit(inputs: Inputs, spec: Spec, first: int, last: int,
        start: Optional[Fit] = None, max_iter: int = 400) -> Fit:
    """Penalised maximum likelihood on blocks ``[first, last)``.

    The Poisson likelihood with a free intercept per fortnight, profiled
    out: each fortnight's records are a multinomial draw over the cells.
    """
    p, q, c = _layout(inputs, spec)
    rows = np.arange(first, last)
    y = inputs.counts[rows]
    n = y.sum(axis=1)
    keep = n > 0
    rows, y, n = rows[keep], y[keep], n[keep]
    lap = inputs.laplacian
    tau = spec.smooth
    tau_season = SEASON_SMOOTH * spec.smooth / SMOOTH
    seasonal = spec.seasonal and inputs.seasonal and spec.spatial
    spec = Spec(spec.dynamic, seasonal, spec.smooth, spec.spatial)
    cells = inputs.lattice.size
    dyn = np.stack([inputs.dynamic[name][rows] for name in DYNAMIC]) if spec.dynamic else None
    cos, sin = inputs.cos[rows], inputs.sin[rows]

    def objective(theta: np.ndarray):
        beta, gamma, s, u, v = _split(theta, p, q, c, seasonal, cells)
        eta = _eta(inputs, spec, rows, beta, gamma, s, u, v, dyn)
        loglik = float(np.vdot(y, eta))
        top = eta.max(axis=1)
        eta -= top[:, None]
        np.exp(eta, out=eta)
        z = eta.sum(axis=1)
        loglik -= float(n @ (np.log(z) + top))
        resid = eta  # reused in place: y - n * prob
        resid *= (-(n / z))[:, None]
        resid += y
        total = resid.sum(axis=0)
        value = -loglik + COEF_RIDGE * beta @ beta
        grads = [-(inputs.static.T @ total) + 2 * COEF_RIDGE * beta]
        if q:
            grads.append(-np.tensordot(dyn, resid, axes=([1, 2], [0, 1])) + 2 * COEF_RIDGE * gamma)
            value += COEF_RIDGE * gamma @ gamma
        if c:
            ls = lap @ s
            value += tau * s @ ls + RIDGE * s @ s
            grads.append(-total + 2 * tau * ls + 2 * RIDGE * s)
        if seasonal:
            lu, lv = lap @ u, lap @ v
            value += tau_season * (u @ lu + v @ lv) + SEASON_RIDGE * (u @ u + v @ v)
            grads.append(-(cos @ resid) + 2 * tau_season * lu + 2 * SEASON_RIDGE * u)
            grads.append(-(sin @ resid) + 2 * tau_season * lv + 2 * SEASON_RIDGE * v)
        return value, np.concatenate(grads)

    size = p + q + c * (3 if seasonal else 1)
    theta0 = np.zeros(size)
    if start is not None and start.spec == spec and len(start.theta()) == size:
        theta0 = start.theta()
    with threadpool_limits(limits=1, user_api="blas"):
        result = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                          options={"maxiter": max_iter, "maxcor": 20, "ftol": 1e-10,
                                   "gtol": 1e-6})
    beta, gamma, s, u, v = _split(result.x, p, q, c, seasonal, cells)
    return Fit(spec=spec, beta=beta.copy(), gamma=gamma.copy(), s=s.copy(), u=u.copy(),
               v=v.copy(), blocks=(first, last), iterations=int(result.nit),
               converged=bool(result.success))


def surface(inputs: Inputs, model: Fit, row: int, corrected: bool = False) -> np.ndarray:
    """Share of the records in block ``row`` each cell is expected to hold.

    ``corrected`` holds every reporting-effort covariate level: the share
    of elephant use, not of reports.
    """
    beta = model.beta.copy()
    if corrected:
        for k, name in enumerate(inputs.names):
            if name in inputs.effort:
                beta[k] = 0.0
    eta = _eta(inputs, model.spec, np.array([row]), beta, model.gamma, model.s, model.u,
               model.v)
    return _softmax(eta)[0][0]


def seasonal_use(inputs: Inputs, model: Fit, months: Sequence[int],
                 corrected: bool = True) -> np.ndarray:
    """Average share over the 15th of each of ``months``, without the
    recent past: the standing pattern for that time of year."""
    spec = Spec(False, model.spec.seasonal, model.spec.smooth)
    beta = model.beta.copy()
    if corrected:
        for k, name in enumerate(inputs.names):
            if name in inputs.effort:
                beta[k] = 0.0
    base = (inputs.static @ beta if len(beta) else 0.0) + model.s
    total = np.zeros(inputs.lattice.size)
    for month in months:
        eta = base
        if spec.seasonal:
            angle = 2 * np.pi * (pd.Timestamp(2001, month, 15).dayofyear - 1) / 365.25
            eta = base + math.cos(angle) * model.u + math.sin(angle) * model.v
        total += _softmax(eta[None, :])[0][0]
    return total / len(months)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def capture(share: np.ndarray, counts: np.ndarray, area: float) -> float:
    """Share of ``counts`` in the top ``area`` of cells by ``share``; cells
    tied at the cut count in proportion, so ties favour no one."""
    total = counts.sum()
    if total <= 0:
        return float("nan")
    k = max(1, int(round(area * len(share))))
    order = np.sort(share)[::-1]
    cut = order[k - 1]
    above = share > cut
    tied = share == cut
    room = k - above.sum()
    caught = counts[above].sum() + counts[tied].sum() * room / max(tied.sum(), 1)
    return float(caught / total)


def area_for(share: np.ndarray, counts: np.ndarray, target: float = 0.5) -> float:
    """Cells needed, best first by ``share``, to hold ``target`` of the
    records. Records in cells tied on share are spread evenly over them."""
    total = counts.sum()
    if total <= 0:
        return float("nan")
    values, group = np.unique(-share, return_inverse=True)
    held = np.bincount(group, weights=counts, minlength=len(values))
    size = np.bincount(group, minlength=len(values)).astype(float)
    need = target * total
    before = np.concatenate([[0.0], np.cumsum(held)[:-1]])
    cells_before = np.concatenate([[0.0], np.cumsum(size)[:-1]])
    k = int(np.searchsorted(np.cumsum(held), need - 1e-9))
    k = min(k, len(values) - 1)
    part = (need - before[k]) / held[k] if held[k] > 0 else 0.0
    return float(cells_before[k] + part * size[k])


def log_score(share: np.ndarray, counts: np.ndarray) -> float:
    """Mean log probability per record (nats)."""
    total = counts.sum()
    if total <= 0:
        return float("nan")
    return float((counts * np.log(np.maximum(share, 1e-300))).sum() / total)


def _floor(share: np.ndarray) -> np.ndarray:
    share = share / share.sum() if share.sum() > 0 else np.full(len(share), 1 / len(share))
    return (1 - UNIFORM_FLOOR) * share + UNIFORM_FLOOR / len(share)


def baseline_last(inputs: Inputs, row: int) -> np.ndarray:
    """Last fortnight's records, smoothed; all past records break ties."""
    recent = inputs.near @ inputs.counts[row - 1] if row > 0 else np.zeros(inputs.lattice.size)
    past = inputs.near @ inputs.counts[:row].sum(axis=0)
    mix = np.zeros(inputs.lattice.size)
    if recent.sum() > 0:
        mix += 0.9 * recent / recent.sum()
    if past.sum() > 0:
        mix += (0.1 if recent.sum() > 0 else 1.0) * past / past.sum()
    return _floor(mix)


def baseline_past(inputs: Inputs, row: int) -> np.ndarray:
    """Every record before the block, smoothed."""
    return _floor(inputs.near @ inputs.counts[:row].sum(axis=0))


BASELINES = {"Last fortnight's reports": baseline_last, "All past reports": baseline_past}
MODEL = "Model"


@dataclass
class Tested:
    """One fortnight forecast from the fortnights before it."""

    row: int
    start: pd.Timestamp
    records: int
    spec: Spec
    capture: Dict[str, float]
    log_score: Dict[str, float]
    half_km2: Dict[str, float]   # area holding half the records, best cells first
    curve: Dict[float, float]  # area share -> model capture


@dataclass
class Record:
    tests: List[Tested] = field(default_factory=list)
    spec: Spec = field(default_factory=Spec)
    scores: Dict[str, float] = field(default_factory=dict)  # spec name -> total log score

    @property
    def tested(self) -> int:
        return len(self.tests)

    @property
    def records(self) -> int:
        return sum(t.records for t in self.tests)

    def rate(self, method: str = MODEL) -> float:
        """Records caught in the top tenth of the area, pooled over tests."""
        if not self.tests:
            return float("nan")
        caught = sum(t.capture[method] * t.records for t in self.tests)
        return caught / self.records

    def half_km2(self, method: str = MODEL) -> float:
        """Area that held half of each fortnight's records, on average."""
        if not self.tests:
            return float("nan")
        return sum(t.half_km2[method] * t.records for t in self.tests) / self.records

    def curve(self) -> Dict[float, float]:
        if not self.tests:
            return {}
        return {a: sum(t.curve[a] * t.records for t in self.tests) / self.records
                for a in AREA_SHARES}

    def bits(self, method: str = MODEL) -> float:
        """Mean log score per record, in bits."""
        if not self.tests:
            return float("nan")
        return sum(t.log_score[method] * t.records for t in self.tests) / self.records \
            / math.log(2)

    def best_baseline(self) -> str:
        return max(BASELINES, key=self.bits)

    def wins(self) -> int:
        """Fortnights the model needed less area than the better simple map."""
        rival = self.best_baseline()
        return sum(t.half_km2[MODEL] < t.half_km2[rival] for t in self.tests)

    def gain_bits(self) -> float:
        """Bits per record over the better simple map (positive: sharper)."""
        if not self.tests:
            return float("nan")
        return self.bits(MODEL) - self.bits(self.best_baseline())

    @property
    def beats_baselines(self) -> bool:
        if self.tested < MIN_TESTED:
            return False
        rival = self.best_baseline()
        return self.gain_bits() > 0 and self.half_km2() <= self.half_km2(rival)


def specs(inputs: Inputs) -> List[Spec]:
    """The versions tried; the first is used until the tests pick."""
    return [Spec(dynamic=True, seasonal=inputs.seasonal),
            Spec(dynamic=False, seasonal=inputs.seasonal),
            Spec(dynamic=True, seasonal=False, spatial=False)]


def _window(inputs: Inputs, row: int) -> int:
    return max(BURN_IN, row - MAX_FIT_BLOCKS)


def validate(inputs: Inputs, tests: int = TEST_BLOCKS) -> Tuple[Record, Dict[str, Fit]]:
    """Forecast each of the last ``tests`` fortnights from those before it.

    Every version of the model is fitted at every test; the one used at
    a test is whichever scored best on the tests before it, so the
    record is that of the model as it would have been used.
    """
    candidates = specs(inputs)
    record = Record(spec=candidates[0], scores={s.name: 0.0 for s in candidates})
    latest: Dict[str, Fit] = {}
    total = inputs.blocks.count
    first_test = max(BURN_IN + MIN_FIT_BLOCKS, total - tests)
    work = inputs.lattice.size * min(total, MAX_FIT_BLOCKS)
    every = max(1, min(3, int(math.ceil(work / REFIT_WORK))))
    since = every
    for row in range(first_test, total):
        y = inputs.counts[row]
        if y.sum() == 0:
            continue
        fitted = inputs.counts[_window(inputs, row):row].sum()
        if fitted < MIN_RECORDS:
            continue
        chosen = max(candidates, key=lambda s: record.scores[s.name])
        refit = since >= every or not latest
        since = 1 if refit else since + 1
        shares: Dict[str, np.ndarray] = {}
        for spec in candidates:
            if refit:
                latest[spec.name] = fit(inputs, spec, _window(inputs, row), row,
                                        start=latest.get(spec.name))
            shares[spec.name] = surface(inputs, latest[spec.name], row)
        model_share = shares[chosen.name]
        cell_area = inputs.lattice.area_km2
        cap = {MODEL: capture(model_share, y, HEADLINE_SHARE)}
        score = {MODEL: log_score(model_share, y)}
        half = {MODEL: area_for(model_share, y) * cell_area}
        for name, make in BASELINES.items():
            share = make(inputs, row)
            cap[name] = capture(share, y, HEADLINE_SHARE)
            score[name] = log_score(share, y)
            half[name] = area_for(share, y) * cell_area
        record.tests.append(Tested(
            row=row, start=inputs.blocks.start(row), records=int(y.sum()), spec=chosen,
            capture=cap, log_score=score, half_km2=half,
            curve={a: capture(model_share, y, a) for a in AREA_SHARES}))
        for spec in candidates:
            record.scores[spec.name] += log_score(shares[spec.name], y) * y.sum()
    record.spec = max(candidates, key=lambda s: record.scores[s.name])
    return record, latest


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------
@dataclass
class Result:
    inputs: Inputs
    record: Record
    forecast: Optional[Fit]
    standing: Optional[Fit]

    @property
    def fitted(self) -> bool:
        return self.forecast is not None

    @property
    def next_start(self) -> pd.Timestamp:
        return self.inputs.blocks.end

    @property
    def next_end(self) -> pd.Timestamp:
        return self.inputs.blocks.end + pd.Timedelta(days=self.inputs.blocks.days - 1)

    def next_fortnight(self, corrected: bool = False) -> np.ndarray:
        """Expected share of next fortnight's records per cell."""
        if self.forecast is None:
            return baseline_past(self.inputs, self.inputs.blocks.count)
        return surface(self.inputs, self.forecast, self.inputs.blocks.count, corrected)

    def use(self, season: str = ALL_YEAR, corrected: bool = True) -> np.ndarray:
        """The standing pattern of use for a season, without the recent past."""
        if self.standing is None:
            return baseline_past(self.inputs, self.inputs.blocks.count)
        months = tuple(range(1, 13)) if season == ALL_YEAR else SEASONS[season]
        return seasonal_use(self.inputs, self.standing, months, corrected)

    def seasons(self) -> List[str]:
        if self.standing is not None and self.standing.spec.seasonal:
            return [ALL_YEAR, *SEASONS]
        return [ALL_YEAR]

    def effects(self) -> pd.DataFrame:
        """How much each term moves the expected records, in words and ratios."""
        rows = []
        model = self.forecast
        if model is None:
            return pd.DataFrame(columns=["Term", "Kind", "Effect", "Ratio"])
        for k, name in enumerate(self.inputs.names):
            per_unit = model.beta[k] / self.inputs.scale[k]
            if name == VILLAGE_KM:
                ratio, what = 2.0 ** per_unit, "twice as far from a village"
            elif name in (FOREST, FOREST_NEAR):
                ratio, what = math.exp(per_unit * 0.5), "half the area forest rather than none"
            else:
                ratio, what = 2.0 ** per_unit, "twice as many nearby"
            kind = "Reporting effort" if name in self.inputs.effort else "Habitat"
            rows.append({"Term": self.inputs.labels[name], "Kind": kind,
                         "Effect": f"{what}: x{ratio:.2f} records", "Ratio": ratio})
        if model.spec.dynamic:
            for k, name in enumerate(DYNAMIC):
                ratio = 2.0 ** model.gamma[k]
                rows.append({"Term": DYNAMIC_LABELS[name], "Kind": "Recent past",
                             "Effect": f"twice as many: x{ratio:.2f} records", "Ratio": ratio})
        return pd.DataFrame(rows)

    def model_table(self) -> pd.DataFrame:
        """Cell x fortnight counts and covariates, for refitting elsewhere
        (R-INLA, inlabru, mgcv): one row per cell per fortnight."""
        inputs = self.inputs
        t, c = inputs.counts.shape
        lat, lon = inputs.lattice.lat, inputs.lattice.lon
        frame = pd.DataFrame({
            "cell": np.tile(np.arange(c), t),
            "latitude": np.tile(np.round(lat, 5), t),
            "longitude": np.tile(np.round(lon, 5), t),
            "x_km": np.tile(np.round(inputs.lattice.x, 3), t),
            "y_km": np.tile(np.round(inputs.lattice.y, 3), t),
            "fortnight": np.repeat(np.arange(t), c),
            "fortnight_start": np.repeat([inputs.blocks.start(i).date() for i in range(t)], c),
            "records": inputs.counts.ravel().astype(int),
            "area_km2": inputs.lattice.area_km2,
        })
        for k, name in enumerate(inputs.names):
            frame[name] = np.tile(np.round(inputs.raw[:, k], 5), t)
            frame[f"{name}_is_effort"] = name in inputs.effort
        for name in DYNAMIC:
            frame[f"lag_{name}"] = np.round(inputs.dynamic[name][:t].ravel(), 5)
        return frame


def run(inputs: Inputs, tests: int = TEST_BLOCKS) -> Result:
    """Test the model on past fortnights, then fit it for the next one."""
    total = inputs.blocks.count
    enough = (total - BURN_IN >= MIN_FIT_BLOCKS
              and inputs.counts[BURN_IN:].sum() >= MIN_RECORDS)
    if not enough:
        return Result(inputs=inputs, record=Record(), forecast=None, standing=None)
    record, latest = validate(inputs, tests)
    first = _window(inputs, total)
    chosen = record.spec
    forecast = fit(inputs, chosen, first, total, start=latest.get(chosen.name))
    static = Spec(False, inputs.seasonal, chosen.smooth)
    standing = forecast if chosen == static else fit(
        inputs, static, first, total, start=latest.get(static.name))
    return Result(inputs=inputs, record=record, forecast=forecast, standing=standing)


# ---------------------------------------------------------------------------
# For the page
# ---------------------------------------------------------------------------
BANDS = (("Top 5%", 0.05), ("Next 5%", 0.10), ("Next 15%", 0.25))
PLACE_KM = 5.0


def bands(share: np.ndarray) -> np.ndarray:
    """Each cell's band: the top 5% of the landscape by share, the next
    5%, the next 15%, or "" for the rest."""
    order = np.argsort(-share, kind="stable")
    rank = np.empty(len(share), dtype=int)
    rank[order] = np.arange(len(share))
    out = np.full(len(share), "", dtype=object)
    lower = 0
    for label, top in BANDS:
        upper = int(round(top * len(share)))
        out[(rank >= lower) & (rank < upper)] = label
        lower = upper
    return out


def nearest_village(lattice: Lattice, villages: Optional[pd.DataFrame],
                    within_km: float = PLACE_KM) -> Tuple[np.ndarray, np.ndarray]:
    """Each cell's nearest village within ``within_km`` ("" beyond), and
    the distance."""
    names = np.full(lattice.size, "", dtype=object)
    km = np.full(lattice.size, np.nan)
    if villages is None or not len(villages):
        return names, km
    vx, vy = lattice.xy(villages["Latitude"].to_numpy(float),
                        villages["Longitude"].to_numpy(float))
    ok = np.isfinite(vx) & np.isfinite(vy)
    if not ok.any():
        return names, km
    distance, index = cKDTree(np.column_stack([vx[ok], vy[ok]])).query(
        np.column_stack([lattice.x, lattice.y]))
    labels = villages["Village"].astype(str).to_numpy()[ok]
    near = distance <= within_km
    names[near] = labels[index[near]]
    km[near] = distance[near]
    return names, km


def places(lattice: Lattice, share: np.ndarray, villages: Optional[pd.DataFrame],
           limit: int = 15) -> pd.DataFrame:
    """The villages around the most expected, summed over their cells."""
    names, _ = nearest_village(lattice, villages)
    band = bands(share)
    frame = pd.DataFrame({"Village": names, "Share": share, "Band": band})
    frame = frame[(frame["Village"] != "") & (frame["Band"] != "")]
    if frame.empty:
        return pd.DataFrame(columns=["Village", "Share", "Band", "Latitude", "Longitude"])
    order = {label: k for k, (label, _) in enumerate(BANDS)}
    grouped = frame.groupby("Village").agg(
        Share=("Share", "sum"), Band=("Band", lambda b: min(b, key=order.get)))
    grouped = grouped.sort_values("Share", ascending=False).head(limit).reset_index()
    where = villages.drop_duplicates("Village").set_index(
        villages.drop_duplicates("Village")["Village"].astype(str))
    grouped["Latitude"] = where.loc[grouped["Village"], "Latitude"].to_numpy(float)
    grouped["Longitude"] = where.loc[grouped["Village"], "Longitude"].to_numpy(float)
    return grouped


_LAT_NAMES = ("latitude", "lat", "y")
_LON_NAMES = ("longitude", "lon", "long", "lng", "x")


def read_effort(source) -> pd.DataFrame:
    """Points where people who report are: app-open pings, patrol tracks,
    user homes. Only the coordinates are kept, rounded to about 100 m,
    as the registry is; every other column is dropped as it is read."""
    frame = pd.read_csv(source)
    lower = {str(c).strip().lower(): c for c in frame.columns}
    lat = next((lower[k] for k in _LAT_NAMES if k in lower), None)
    lon = next((lower[k] for k in _LON_NAMES if k in lower), None)
    if lat is None or lon is None:
        raise ValueError("The file needs Latitude and Longitude columns.")
    out = pd.DataFrame({
        "Latitude": pd.to_numeric(frame[lat], errors="coerce").round(3),
        "Longitude": pd.to_numeric(frame[lon], errors="coerce").round(3),
    }).dropna()
    out = out[out["Latitude"].between(-90, 90) & out["Longitude"].between(-180, 180)]
    return out.reset_index(drop=True)


def result_key(result: Result) -> str:
    """Stands for a result in a cache key: its data and fitted surface."""
    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    digest.update(result.inputs.counts.tobytes())
    digest.update(str(result.inputs.names).encode())
    if result.standing is not None:
        digest.update(result.standing.theta().tobytes())
    return digest.hexdigest()
