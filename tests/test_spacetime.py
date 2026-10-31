"""Space-time cube and emerging hotspot labels."""

import numpy as np
import pandas as pd
import pytest

from core import charts, spacetime
from core.report import _emerging_section
from core.spacetime import (
    CONSECUTIVE,
    DIMINISHING,
    HISTORICAL,
    INTENSIFYING,
    NEW,
    NO_PATTERN,
    PERSISTENT,
    SPORADIC,
    build_cube,
    classify_pattern,
    emerging_hotspots,
    location_time_matrix,
    mann_kendall,
)

STEADY = (23.30, 81.40)
FRESH = (23.10, 81.70)


def _frame(months=18, fresh_from=17, seed=0):
    """A steady hotspot throughout, one appearing late, scattered noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(months):
        start = pd.Timestamp("2024-01-01") + pd.DateOffset(months=m)

        def add(n, lat, lon, spread):
            for _ in range(n):
                rows.append({
                    "Date": start + pd.Timedelta(days=int(rng.integers(0, 27))),
                    "Latitude": lat + rng.normal(0, spread),
                    "Longitude": lon + rng.normal(0, spread),
                })

        add(8, *STEADY, 0.002)
        if m >= fresh_from:
            add(12, *FRESH, 0.002)
        for _ in range(20):
            rows.append({
                "Date": start + pd.Timedelta(days=int(rng.integers(0, 27))),
                "Latitude": 23.0 + rng.random() * 0.5,
                "Longitude": 81.3 + rng.random() * 0.5,
            })
    df = pd.DataFrame(rows)
    df["Crop Damage"] = 1
    for col in ["Grain Damage", "House Damage", "Injury", "Death"]:
        df[col] = 0
    df["Beat"] = "B1"
    return df


def _place(table, lat, lon):
    d = (table["Centre Latitude"] - lat) ** 2 + (table["Centre Longitude"] - lon) ** 2
    return table.loc[d.idxmin()]


def test_cube_keeps_only_cells_with_events_and_fills_quiet_steps():
    cube = build_cube(_frame(), as_of=pd.Timestamp("2025-06-28"))
    assert cube.counts.shape[1] == 18
    assert (cube.counts.sum(axis=1) > 0).all()
    assert int(cube.counts.to_numpy().sum()) == 18 * 28 + 12


def test_cube_drops_a_barely_started_last_step():
    df = _frame()
    df = pd.concat([df, df.head(1).assign(Date=pd.Timestamp("2025-07-02"))])
    cube = build_cube(df, as_of=pd.Timestamp("2025-07-02"))
    assert cube.steps[-1] == pd.Timestamp("2025-06-01")
    assert any("left out" in n for n in cube.notes)


def test_cube_counts_conflict_only_by_default():
    df = _frame()
    df.loc[df.index[:10], "Crop Damage"] = 0
    conflict = build_cube(df, as_of=df["Date"].max())
    everything = build_cube(df, metric="All reports", as_of=df["Date"].max())
    assert everything.counts.to_numpy().sum() - conflict.counts.to_numpy().sum() == 10


def test_steady_place_is_persistent_and_late_one_is_new():
    df = _frame()
    table = emerging_hotspots(build_cube(df, as_of=pd.Timestamp("2025-06-28")))
    assert _place(table, *STEADY)["Pattern"] == PERSISTENT
    assert _place(table, *FRESH)["Pattern"] == NEW
    # Worst first, and only patterned places are named.
    assert table.iloc[0]["Pattern"] == NEW and table.iloc[0]["Place"] == "E1"
    assert (table.loc[table["Pattern"] == NO_PATTERN, "Place"] == "").all()


def test_no_empty_cell_is_called_hot():
    """Gi* borrows from neighbours; an empty cell must not inherit the label."""
    cube = build_cube(_frame(), as_of=pd.Timestamp("2025-06-28"))
    table = emerging_hotspots(cube)
    for row in table[table["Pattern"].isin([NEW, PERSISTENT])].to_dict("records"):
        assert cube.counts.loc[row["Cell"]].iloc[-2:].sum() > 0


def test_too_few_steps_returns_nothing_and_says_why():
    df = _frame(months=4, fresh_from=99)
    cube = build_cube(df, as_of=df["Date"].max())
    table = emerging_hotspots(cube)
    assert table.empty
    assert any("at least" in n for n in spacetime.pattern_caveats(cube, table, True))


@pytest.mark.parametrize("hot, z, p, expected", [
    ([0] * 9 + [1], 0, 1, NEW),
    ([1] * 10, 3, 0.001, INTENSIFYING),
    ([1] * 10, 0.2, 0.8, PERSISTENT),
    ([1] * 10, -3, 0.001, DIMINISHING),
    ([0] * 7 + [1] * 3, 2, 0.01, CONSECUTIVE),
    ([1, 0, 1, 0, 0, 1, 0, 0, 1, 1], 0, 1, SPORADIC),
    ([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], -2, 0.01, HISTORICAL),
    ([0] * 10, 0, 1, NO_PATTERN),
])
def test_classification_rules(hot, z, p, expected):
    assert classify_pattern(np.array(hot, dtype=bool), z, p) == expected


def test_mann_kendall_sees_a_rise_and_not_noise():
    z, p = mann_kendall(np.arange(12, dtype=float))
    assert z > 0 and p < 0.01
    z, p = mann_kendall(np.ones(12))
    assert p == 1.0


def test_heatmap_and_brief_follow_the_table():
    df = _frame()
    cube = build_cube(df, as_of=pd.Timestamp("2025-06-28"))
    table = emerging_hotspots(cube)
    matrix = location_time_matrix(cube, table)
    assert matrix.index[0].startswith("E1 New")
    fig = charts.place_time_heatmap(matrix, cube.step)
    assert list(fig.data[0].y) == list(matrix.index)

    html = _emerging_section(table, cube.step)
    assert "Emerging Hotspots" in html and "<b>E1</b>" in html
    assert spacetime.headline(table).startswith("Space-time trend:")
    assert _emerging_section(None, "Month") == ""


def test_hex_polygon_is_closed_around_its_centre():
    corners = np.array(spacetime.hex_polygon(23.3, 81.4, 2.0, 23.3))
    assert corners.shape == (6, 2)
    assert np.allclose(corners.mean(axis=0), [81.4, 23.3], atol=1e-6)


def _cube_and_table():
    cube = build_cube(_frame(), as_of=pd.Timestamp("2025-06-28"))
    return cube, emerging_hotspots(cube)


def test_hot_bins_match_the_labels():
    cube, table = _cube_and_table()
    hot = spacetime.hot_bins(cube)
    assert hot.shape == cube.counts.shape
    for row in table.to_dict("records"):
        assert int(hot.loc[row["Cell"]].sum()) == row["Hot Steps"]
    # Never hot without a report of its own in the window.
    own = cube.counts.T.rolling(2, min_periods=1).sum().T
    assert not (hot & (own == 0)).to_numpy().any()


def _named_bins(cube, table, hot):
    """Hot and not-hot bins the cube should draw for the named places."""
    named = table[table["Place"] != ""]
    counts = cube.counts.loc[named["Cell"]]
    is_hot = hot.loc[named["Cell"]]
    return int(is_hot.to_numpy().sum()), int(((counts > 0) & ~is_hot).to_numpy().sum())


def test_cube_chart_draws_a_prism_per_bin_coloured_where_hot():
    cube, table = _cube_and_table()
    hot = spacetime.hot_bins(cube)
    fig = charts.space_time_cube(cube, table, hot, floor_boundaries=False)
    n_hot, n_pale = _named_bins(cube, table, hot)
    meshes = {t.name: t for t in fig.data if t.type == "mesh3d" and t.name}
    pattern_meshes = [m for name, m in meshes.items() if name in spacetime.PATTERN_COLORS]
    # Twelve vertices and twenty faces a prism.
    assert sum(len(m.x) for m in pattern_meshes) == 12 * n_hot
    assert sum(len(m.i) for m in pattern_meshes) == 20 * n_hot
    assert {m.color for m in pattern_meshes} <= set(spacetime.PATTERN_COLORS.values())
    assert len(meshes["Events, not hot that step"].x) == 12 * n_pale
    # Every vertex of a prism carries that bin's tip, for the hover.
    tips = meshes["Events, not hot that step"].text
    assert all(len(set(tips[i:i + 12])) == 1 for i in range(0, len(tips), 12))


def test_cube_chart_time_axis_ends_on_the_latest_step_at_true_ground_ratio():
    cube, table = _cube_and_table()
    fig = charts.space_time_cube(cube, table, spacetime.hot_bins(cube), floor_boundaries=False)
    scene = fig.layout.scene
    assert scene.zaxis.tickvals[-1] == len(cube.steps) - 1
    assert scene.zaxis.ticktext[-1] == f"{cube.steps[-1]:%b %Y}"
    assert list(scene.zaxis.tickvals) == sorted(scene.zaxis.tickvals)
    assert scene.aspectmode == "manual"
    span_x = scene.xaxis.range[1] - scene.xaxis.range[0]
    span_y = scene.yaxis.range[1] - scene.yaxis.range[0]
    assert scene.aspectratio.x / scene.aspectratio.y == pytest.approx(span_x / span_y)


def test_prisms_face_outward_and_size_by_area():
    vx, vy, vz, i, j, k = charts._prisms([0, 10], [0, 0], [0, 3], [1.0, 2.0], 0.4)
    points = np.c_[vx, vy, vz]
    for prism in range(2):
        faces = slice(20 * prism, 20 * prism + 20)
        a, b, c = points[i[faces]], points[j[faces]], points[k[faces]]
        centre = points[12 * prism:12 * prism + 12].mean(axis=0)
        normals = np.cross(b - a, c - a)
        assert (np.einsum("ij,ij->i", normals, (a + b + c) / 3 - centre) > 0).all()
    assert vz[:12].min() == pytest.approx(-0.4) and vz[12:].max() == pytest.approx(3.4)


def test_cube_floor_carries_the_forest_boundaries_inside_the_scene():
    from core import boundaries

    if not boundaries.available():
        pytest.skip("boundary layers not vendored")
    cube, table = _cube_and_table()
    hot = spacetime.hot_bins(cube)
    fig = charts.space_time_cube(cube, table, hot)
    lines = [t for t in fig.data if (t.name or "").endswith("boundaries")]
    assert len(lines) == 1
    xs = [v for v in lines[0].x if v is not None]
    ys = [v for v in lines[0].y if v is not None]
    scene = fig.layout.scene
    assert xs and scene.xaxis.range[0] <= min(xs) and max(xs) <= scene.xaxis.range[1]
    assert scene.yaxis.range[0] <= min(ys) and max(ys) <= scene.yaxis.range[1]
    assert len({z for z in lines[0].z if z is not None}) == 1  # flat on the floor
    off = charts.space_time_cube(cube, table, hot, floor_boundaries=False)
    assert not [t for t in off.data if (t.name or "").endswith("boundaries")]


def test_boundary_outlines_are_limited_to_the_box():
    from core import boundaries

    if not boundaries.available():
        pytest.skip("boundary layers not vendored")
    rings = boundaries.outlines(boundaries.RANGE, 81.3, 23.0, 81.8, 23.5)
    everything = boundaries.load(boundaries.RANGE)["features"]
    assert rings and len({name for name, _ in rings}) < len(everything)
    assert all(ring.ndim == 2 and ring.shape[1] == 2 for _, ring in rings)


def test_cube_chart_draws_only_the_villages_places_are_nearest_to():
    cube, _ = _cube_and_table()
    villages = pd.DataFrame({
        "Village": ["Steadypur", "Freshgaon", "Faraway"] + [f"V{i}" for i in range(50)],
        "Latitude": [STEADY[0] + 0.004, FRESH[0] + 0.004, 23.40]
        + list(np.linspace(23.0, 23.5, 50)),
        "Longitude": [STEADY[1], FRESH[1], 81.50] + list(np.linspace(81.3, 81.8, 50)),
    })
    table = emerging_hotspots(cube, villages=villages)
    fig = charts.space_time_cube(cube, table, spacetime.hot_bins(cube), villages,
                                 floor_boundaries=False)
    shown = [t for t in fig.data if t.name == "Villages"][0]
    assert {"Steadypur", "Freshgaon"} <= set(shown.text)
    assert len(shown.text) < 10


def test_cube_chart_cameras_and_slice():
    cube, table = _cube_and_table()
    hot = spacetime.hot_bins(cube)
    for camera, preset in charts.CUBE_CAMERAS.items():
        fig = charts.space_time_cube(cube, table, hot, camera=camera, floor_boundaries=False)
        assert fig.layout.scene.camera.eye.x == preset["eye"]["x"]
        assert fig.layout.uirevision == camera
    sliced = charts.space_time_cube(cube, table, hot, highlight_step=cube.steps[5],
                                    floor_boundaries=False)
    plane = [t for t in sliced.data if t.name == f"{cube.steps[5]:%b %Y}"]
    assert len(plane) == 1 and set(plane[0].z) == {5}


def test_cube_chart_with_nothing_to_show_is_empty():
    cube, table = _cube_and_table()
    quiet = table.assign(Pattern=NO_PATTERN, Place="")
    assert len(charts.space_time_cube(cube, quiet, spacetime.hot_bins(cube)).data) == 0


@pytest.fixture
def deck(monkeypatch):
    from core import map_engine

    captured = {}
    monkeypatch.setattr(map_engine, "_render_deck",
                        lambda layers, view, style: captured.update(layers=layers, view=view))
    return captured


def _layers(captured, kind):
    return [layer for layer in captured["layers"] if layer.type == kind]


def test_pattern_map_flat_and_columns(deck):
    from core import map_engine

    cube, table = _cube_and_table()
    map_engine.render_pattern_map(table, cube.cell_km, cube.reference_lat, cube.step,
                                  show_boundaries=False, mode=map_engine.MAP_COLUMNS)
    hexes = _layers(deck, "PolygonLayer")[0]
    assert hexes.extruded and deck["view"].pitch > 0
    data = pd.DataFrame(hexes.data)
    assert data["_elev"].max() == pytest.approx(3 * cube.cell_km * 1000)

    map_engine.render_pattern_map(table, cube.cell_km, cube.reference_lat, cube.step,
                                  show_boundaries=False)
    flat = _layers(deck, "PolygonLayer")[0]
    assert not flat.extruded and deck["view"].pitch == 0


def test_pattern_map_stacked_builds_the_cube_on_the_map(deck):
    from core import map_engine

    cube, table = _cube_and_table()
    hot = spacetime.hot_bins(cube)
    map_engine.render_pattern_map(table, cube.cell_km, cube.reference_lat, cube.step,
                                  show_boundaries=False, mode=map_engine.MAP_STACKED,
                                  counts=cube.counts, hot=hot)
    ground, slabs = _layers(deck, "PolygonLayer")
    assert not ground.extruded and slabs.extruded
    data = pd.DataFrame(slabs.data)
    n_hot, n_pale = _named_bins(cube, table, hot)
    assert len(data) == n_hot + n_pale
    # Each slab's base is its step's height: oldest on the ground, latest on top.
    tower = 3 * cube.cell_km * 1000
    slab = tower / len(cube.steps)
    bases = sorted({round(poly[0][2], 6) for poly in data["_polygon"]})
    assert bases[0] >= 0 and bases[-1] <= tower - slab + 1e-6
    assert all(round(b / slab, 6).is_integer() for b in bases)
    assert (data["_elev"] < slab).all()
    assert _layers(deck, "LineLayer") and deck["view"].pitch > 0


def test_pattern_map_stacked_without_the_cube_draws_flat(deck):
    from core import map_engine

    cube, table = _cube_and_table()
    map_engine.render_pattern_map(table, cube.cell_km, cube.reference_lat, cube.step,
                                  show_boundaries=False, mode=map_engine.MAP_STACKED)
    assert len(_layers(deck, "PolygonLayer")) == 1 and deck["view"].pitch == 0
