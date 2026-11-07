"""Corridors on surfaces whose best routes are known."""

import numpy as np
import pandas as pd
import pytest

from core import corridors as cr
from core import presence as pr
from presence_fixtures import to_latlon


def _box(width_km=60, height_km=40):
    xs, ys = np.meshgrid(np.linspace(-width_km / 2, width_km / 2, 31),
                         np.linspace(-height_km / 2, height_km / 2, 21))
    lat, lon = to_latlon(xs.ravel(), ys.ravel())
    return pr.lattice_for(lat, lon, cell_km=2.0, domain_km=2.0)


def _two_cores_and_a_bent_strip(lattice, strip=0.05):
    """Use peaks at both ends, joined by a band of moderate use that
    bows north, with almost nothing on the straight line between."""
    x, y = lattice.x, lattice.y
    use = np.exp(-((x + 25) ** 2 + y ** 2) / 8) + np.exp(-((x - 25) ** 2 + y ** 2) / 8)
    bow = 12 * np.cos(x / 25 * np.pi / 2)
    use += np.where(np.abs(x) < 25, strip * np.exp(-((y - bow) ** 2) / 4), 0)
    return use + 1e-4


def test_resistance_runs_from_one_for_the_most_used_to_a_hundred():
    cost = cr.resistance(np.array([0.0, 0.1, 0.5, 2.0]))
    assert cost[-1] == pytest.approx(1.0)
    assert cost[0] == pytest.approx(100.0)
    assert np.all(np.diff(cost) < 0)


def test_the_cheapest_route_follows_the_band_of_use_not_the_straight_line():
    lattice = _box()
    net = cr.network(lattice, _two_cores_and_a_bent_strip(lattice))
    assert len(net.cores) == 2 and len(net.links) == 1
    link = net.links[0]
    middle = link.path[np.abs(lattice.x[link.path]) < 5]
    assert lattice.y[middle].min() > 8  # up on the bow, not across the middle
    assert link.length_km > link.straight_km * 1.1
    assert set(link.path) <= set(link.corridor)


def test_current_concentrates_on_the_band():
    lattice = _box()
    net = cr.network(lattice, _two_cores_and_a_bent_strip(lattice))
    x, y = lattice.x, lattice.y
    on_band = net.current[np.argmin(x ** 2 + (y - 12) ** 2)]
    straight = net.current[np.argmin(x ** 2 + y ** 2)]
    assert on_band > 5 * straight


def test_a_chain_carries_the_whole_current_through_every_cell():
    """Series resistors: one ampere through each cell between the ends."""
    n = 12
    lookup = np.arange(n).reshape(n, 1)
    lattice = pr.Lattice(cell_km=1.0, lat_c=23.0, lon_c=81.0, x0=0.0, y0=0.0,
                         ix=np.arange(n), iy=np.zeros(n, dtype=int), lookup=lookup)
    graph = cr._graph(lattice, np.ones(n))
    current = cr._current(graph, np.array([0]), np.array([n - 1]))
    assert current[1:-1] == pytest.approx(np.ones(n - 2))


def test_one_block_of_use_has_no_corridor():
    lattice = _box()
    x, y = lattice.x, lattice.y
    net = cr.network(lattice, np.exp(-(x ** 2 + y ** 2) / 30) + 1e-4)
    assert len(net.cores) == 1 and net.links == []


def test_cores_are_named_by_the_use_they_hold():
    lattice = _box()
    use = _two_cores_and_a_bent_strip(lattice)
    use[np.argmin((lattice.x + 25) ** 2 + lattice.y ** 2)] *= 3  # the west peak is bigger
    cores = cr.find_cores(lattice, use)
    assert cores[0].use >= cores[1].use
    assert cores[0].longitude < cores[1].longitude


def test_villages_on_a_route_are_listed_nearest_first():
    lattice = _box()
    net = cr.network(lattice, _two_cores_and_a_bent_strip(lattice))
    link = net.links[0]
    on = link.path[len(link.path) // 2]
    lat, lon = lattice.lat[on], lattice.lon[on]
    villages = pd.DataFrame({"Village": ["On it", "Beside it", "Far"],
                             "Latitude": [lat, lat + 0.009, lat - 0.3],
                             "Longitude": [lon, lon, lon]})
    listed = cr.villages_on(lattice, link, villages)
    assert list(listed["Village"]) == ["On it", "Beside it"]
