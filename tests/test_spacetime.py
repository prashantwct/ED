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


def test_cube_chart_colours_hot_bins_and_greys_the_rest():
    cube, table = _cube_and_table()
    hot = spacetime.hot_bins(cube)
    fig = charts.space_time_cube(cube, table, hot)
    traces = {t.name: t for t in fig.data if t.name}
    hot_trace = traces["Hot that step"]
    pale_trace = traces["Events, not hot that step"]
    named = table[table["Place"] != ""]
    assert len(hot_trace.x) == int(hot.loc[named["Cell"]].to_numpy().sum())
    assert set(hot_trace.marker.color) <= set(spacetime.PATTERN_COLORS.values())
    assert len(pale_trace.x) > 0
    # Time is the vertical axis, labelled with dates, and ground at true ratio.
    scene = fig.layout.scene
    assert scene.zaxis.ticktext[0] == "Jan 2024"
    assert scene.aspectmode == "manual"
    span_x = scene.xaxis.range[1] - scene.xaxis.range[0]
    span_y = scene.yaxis.range[1] - scene.yaxis.range[0]
    assert scene.aspectratio.x / scene.aspectratio.y == pytest.approx(span_x / span_y)


def test_cube_chart_draws_only_the_villages_places_are_nearest_to():
    cube, _ = _cube_and_table()
    villages = pd.DataFrame({
        "Village": ["Steadypur", "Freshgaon", "Faraway"] + [f"V{i}" for i in range(50)],
        "Latitude": [STEADY[0] + 0.004, FRESH[0] + 0.004, 23.40]
        + list(np.linspace(23.0, 23.5, 50)),
        "Longitude": [STEADY[1], FRESH[1], 81.50] + list(np.linspace(81.3, 81.8, 50)),
    })
    table = emerging_hotspots(cube, villages=villages)
    fig = charts.space_time_cube(cube, table, spacetime.hot_bins(cube), villages)
    shown = [t for t in fig.data if t.name == "Villages"][0]
    assert {"Steadypur", "Freshgaon"} <= set(shown.text)
    assert len(shown.text) < 10


def test_cube_chart_cameras_and_slice():
    cube, table = _cube_and_table()
    hot = spacetime.hot_bins(cube)
    for camera, preset in charts.CUBE_CAMERAS.items():
        fig = charts.space_time_cube(cube, table, hot, camera=camera)
        assert fig.layout.scene.camera.eye.x == preset["eye"]["x"]
        assert fig.layout.uirevision == camera
    sliced = charts.space_time_cube(cube, table, hot, highlight_step=cube.steps[5])
    plane = [t for t in sliced.data if t.type == "mesh3d"]
    assert len(plane) == 1 and set(plane[0].z) == {5}


def test_cube_chart_with_nothing_to_show_is_empty():
    cube, table = _cube_and_table()
    quiet = table.assign(Pattern=NO_PATTERN, Place="")
    assert len(charts.space_time_cube(cube, quiet, spacetime.hot_bins(cube)).data) == 0


def test_pattern_map_extrudes_and_tilts(monkeypatch):
    from core import map_engine

    captured = {}
    monkeypatch.setattr(map_engine, "_render_deck",
                        lambda layers, view, style: captured.update(layers=layers, view=view))
    cube, table = _cube_and_table()
    map_engine.render_pattern_map(table, cube.cell_km, cube.reference_lat, cube.step,
                                  show_boundaries=False, extruded=True)
    hexes = [l for l in captured["layers"] if l.type == "PolygonLayer"][0]
    assert hexes.extruded and captured["view"].pitch > 0
    data = pd.DataFrame(hexes.data)
    tallest = data.loc[data["_elev"].idxmax()]
    assert tallest["_elev"] == pytest.approx(3 * cube.cell_km * 1000)

    map_engine.render_pattern_map(table, cube.cell_km, cube.reference_lat, cube.step,
                                  show_boundaries=False)
    flat = [l for l in captured["layers"] if l.type == "PolygonLayer"][0]
    assert not flat.extruded and captured["view"].pitch == 0
