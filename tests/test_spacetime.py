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
