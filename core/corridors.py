"""Corridors between the places elephants use most.

Worked out from an effort-corrected surface of use (``core.presence``),
season by season if the register is long enough to have seasons:

1. **Resistance.** Each cell's use becomes a percentile (0 for the least
   used, 1 for the most) and then a cost of crossing it, from 1 (most
   used) to 100 (least used), with the negative-exponential transform of
   Keeley, Beier & Gagnon (2016, c = 8): a cell has to be used well
   below the typical cell before it is truly hard to cross.
2. **Core areas.** Touching cells in the top ``CORE_SHARE`` of use, the
   largest few by the use they hold.
3. **Links.** The cheapest network joining the cores: a minimum spanning
   tree over their cost distances (Dijkstra on the grid, corners
   included).
4. **Least-cost path and corridor.** For each link, the cheapest route,
   and the corridor around it: every cell through which a route costs
   at most ``SLACK`` more than the cheapest. That is the normalised
   least-cost corridor of Linkage Mapper.
5. **Current flow.** Circuitscape's pairwise mode, without Circuitscape.
   Each link's two cores are held at 1 V and 0 V. Each cell conducts in
   inverse proportion to its resistance, and the current through it shows
   where movement concentrates. A narrow band of high current is a
   pinch point.

These are hypotheses for the landscape, not tracks. They are as good as
the surface they come from, and that surface is relative use, not
abundance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import spsolve

CORE_SHARE = 0.05
MAX_CORES = 8
MIN_CORE_CELLS = 2
SLACK = 0.05
KEELEY_C = 8.0


@dataclass
class Core:
    id: int
    cells: np.ndarray
    use: float          # share of all use held
    latitude: float
    longitude: float
    area_km2: float


@dataclass
class Link:
    a: int
    b: int
    cost: float          # cost-weighted km
    length_km: float     # along the least-cost path
    straight_km: float   # between the cores' nearest cells
    path: np.ndarray     # cells, from core a to core b
    corridor: np.ndarray  # cells within SLACK of the cheapest route

    @property
    def mean_resistance(self) -> float:
        return self.cost / self.length_km if self.length_km else float("nan")


@dataclass
class Network:
    resistance: np.ndarray
    cores: List[Core] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    current: Optional[np.ndarray] = None  # summed over links, each carrying 1 A

    def corridor_cells(self) -> np.ndarray:
        """Link number (from 0) of each cell's corridor, -1 if none."""
        out = np.full(len(self.resistance), -1)
        for k, link in enumerate(self.links):
            free = out[link.corridor] < 0
            out[link.corridor[free]] = k
        return out


def resistance(use: np.ndarray, c: float = KEELEY_C) -> np.ndarray:
    """From 1 (most used) to 100 (least used)."""
    order = np.argsort(np.argsort(use, kind="stable"), kind="stable")
    h = order / max(len(use) - 1, 1)
    return 100.0 - 99.0 * (1.0 - np.exp(-c * h)) / (1.0 - math.exp(-c))


def _graph(lattice, cost: np.ndarray) -> sparse.csr_matrix:
    """Cells joined to their eight neighbours, each edge costing its
    length times the mean resistance of its two cells."""
    adjacent = lattice.neighbours(diagonal=True).tocoo()
    diagonal = (lattice.ix[adjacent.row] != lattice.ix[adjacent.col]) & \
        (lattice.iy[adjacent.row] != lattice.iy[adjacent.col])
    length = lattice.cell_km * np.where(diagonal, math.sqrt(2), 1.0)
    weight = length * (cost[adjacent.row] + cost[adjacent.col]) / 2
    return sparse.csr_matrix((weight, (adjacent.row, adjacent.col)), shape=adjacent.shape)


def find_cores(lattice, use: np.ndarray, share: float = CORE_SHARE,
               max_cores: int = MAX_CORES, min_cells: int = MIN_CORE_CELLS) -> List[Core]:
    """Touching cells in the top ``share`` of use, the largest first."""
    k = max(1, int(round(share * len(use))))
    top = np.zeros(len(use), dtype=bool)
    top[np.argsort(use)[::-1][:k]] = True
    touching = lattice.neighbours(diagonal=True)
    picked = np.flatnonzero(top)
    sub = touching[picked][:, picked]
    count, label = csgraph.connected_components(sub, directed=False)
    total = use.sum()
    found = []
    for g in range(count):
        cells = picked[label == g]
        if len(cells) < min_cells:
            continue
        weight = use[cells]
        found.append((weight.sum(), cells))
    found.sort(key=lambda item: -item[0])
    cores = []
    lat, lon = lattice.lat, lattice.lon
    for i, (mass, cells) in enumerate(found[:max_cores]):
        w = use[cells] / use[cells].sum()
        cores.append(Core(id=i, cells=cells, use=float(mass / total),
                          latitude=float(w @ lat[cells]), longitude=float(w @ lon[cells]),
                          area_km2=float(len(cells) * lattice.area_km2)))
    return cores


def _path(predecessors: np.ndarray, end: int) -> np.ndarray:
    cells = [end]
    while predecessors[cells[-1]] >= 0:
        cells.append(int(predecessors[cells[-1]]))
    return np.asarray(cells[::-1])


def _current(graph: sparse.csr_matrix, source: np.ndarray, ground: np.ndarray
             ) -> Optional[np.ndarray]:
    """Current through each cell with ``source`` at 1 V and ``ground`` at 0 V,
    scaled to 1 A in total. Edge conductance is 1 / edge cost."""
    n = graph.shape[0]
    conductance = graph.copy()
    conductance.data = 1.0 / conductance.data
    _, label = csgraph.connected_components(conductance, directed=False)
    part = label == label[source[0]]
    if not part[ground].any():
        return None
    nodes = np.flatnonzero(part)
    local = -np.ones(n, dtype=int)
    local[nodes] = np.arange(len(nodes))
    g = conductance[nodes][:, nodes].tocsr()
    degree = np.asarray(g.sum(axis=1)).ravel()
    lap = (sparse.diags(degree) - g).tocsr()
    fixed = np.zeros(len(nodes), dtype=bool)
    volts = np.zeros(len(nodes))
    fixed[local[source]] = True
    volts[local[source]] = 1.0
    fixed[local[ground[part[ground]]]] = True
    free = np.flatnonzero(~fixed)
    if len(free):
        rhs = -lap[free][:, np.flatnonzero(fixed)] @ volts[fixed]
        volts[free] = spsolve(lap[free][:, free].tocsc(), rhs)
    edges = g.tocoo()
    flow = edges.data * np.abs(volts[edges.row] - volts[edges.col])
    through = np.zeros(len(nodes))
    np.add.at(through, edges.row, flow)
    # Each edge is counted from both ends; a cell's current is half the
    # flow on its edges.
    through /= 2.0
    injected = float((lap[local[source]] @ volts).sum())
    if injected <= 0:
        return None
    out = np.zeros(n)
    out[nodes] = through / injected
    return out


def network(lattice, use: np.ndarray, flow: bool = True, share: float = CORE_SHARE,
            slack: float = SLACK) -> Network:
    """Cores, the cheapest links between them, their corridors and current."""
    cost = resistance(use)
    net = Network(resistance=cost, cores=find_cores(lattice, use, share))
    if len(net.cores) < 2:
        return net
    graph = _graph(lattice, cost)
    fields = []
    for core in net.cores:
        dist, pred, _ = csgraph.dijkstra(graph, directed=False, indices=core.cells,
                                         min_only=True, return_predecessors=True)
        fields.append((dist, pred))
    m = len(net.cores)
    between = np.full((m, m), np.inf)
    for i in range(m):
        for j in range(m):
            if i != j:
                between[i, j] = fields[i][0][net.cores[j].cells].min()
    finite = np.where(np.isfinite(between), between, 0.0)
    tree = csgraph.minimum_spanning_tree(sparse.csr_matrix(np.triu(finite))).tocoo()
    x, y = lattice.x, lattice.y
    current = np.zeros(len(use)) if flow else None
    for i, j in sorted(zip(tree.row, tree.col), key=lambda e: between[e[0], e[1]]):
        a, b = net.cores[i], net.cores[j]
        dist_a, pred_a = fields[i]
        dist_b = fields[j][0]
        end = int(b.cells[np.argmin(dist_a[b.cells])])
        path = _path(pred_a, end)
        steps = np.hypot(np.diff(x[path]), np.diff(y[path]))
        start = path[0]
        corridor = np.flatnonzero(dist_a + dist_b <= between[i, j] * (1 + slack) + 1e-9)
        net.links.append(Link(
            a=a.id, b=b.id, cost=float(between[i, j]), length_km=float(steps.sum()),
            straight_km=float(np.hypot(x[end] - x[start], y[end] - y[start])),
            path=path, corridor=corridor))
        if flow:
            through = _current(graph, a.cells, b.cells)
            if through is not None:
                current += through
    net.current = current
    return net


def villages_on(lattice, link: Link, villages: Optional[pd.DataFrame],
                within_km: Optional[float] = None) -> pd.DataFrame:
    """Villages within about a cell of a link's least-cost path."""
    if villages is None or not len(villages):
        return pd.DataFrame(columns=["Village", "Latitude", "Longitude", "From path (km)"])
    within = lattice.cell_km if within_km is None else within_km
    vx, vy = lattice.xy(villages["Latitude"].to_numpy(float),
                        villages["Longitude"].to_numpy(float))
    px, py = lattice.x[link.path], lattice.y[link.path]
    d = np.sqrt((vx[:, None] - px[None, :]) ** 2 + (vy[:, None] - py[None, :]) ** 2).min(axis=1)
    near = d <= within
    out = villages.loc[near, ["Village", "Latitude", "Longitude"]].copy()
    out["From path (km)"] = np.round(d[near], 1)
    return out.sort_values("From path (km)").reset_index(drop=True)
