"""Tests for the conflict-model features.

The one that matters is leakage. A feature that can see the outcome it
predicts produces a number that looks excellent and means nothing, and
it fails silently -- there is no error, just an optimistic score that
the field never reproduces.
"""

import numpy as np
import pandas as pd
import pytest

from research.features import (
    conflict_target,
    km_plane,
    panel_feature_columns,
    sighting_features,
    village_month_panel,
)


def _centroids():
    return pd.DataFrame([
        {"Village": "Near", "Latitude": 23.000, "Longitude": 81.000},
        {"Village": "Far", "Latitude": 23.500, "Longitude": 81.500},
    ])


def _sightings(rows):
    base = {
        "Division": "D", "Range": "R", "Beat": "B", "Sighting Type": "Direct",
        "Sighting Type Detail": "", "Hour": 20, "Male Count": 1,
        "Female Count": 0, "Calf Count": 0, "Unknown Count": 0, "Total Count": 1,
        "Crop Damage": 0, "Grain Damage": 0, "House Damage": 0,
        "Injury": 0, "Death": 0,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def test_longitude_is_scaled_by_latitude():
    """A degree of longitude is shorter than a degree of latitude here."""
    at_23 = km_plane(np.array([23.0, 23.0]), np.array([81.0, 82.0]), 23.0)
    east_west = abs(at_23[1, 0] - at_23[0, 0])
    north_south = abs(
        km_plane(np.array([23.0, 24.0]), np.array([81.0, 81.0]), 23.0)[1, 1]
        - km_plane(np.array([23.0, 24.0]), np.array([81.0, 81.0]), 23.0)[0, 1]
    )
    assert east_west == pytest.approx(north_south * np.cos(np.radians(23.0)), rel=1e-6)


def test_conflict_target_counts_any_damage_type():
    df = _sightings([{"Crop Damage": 0}, {"House Damage": 1}, {"Death": 1}])
    assert conflict_target(df).tolist() == [0, 1, 1]


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------
def test_history_cannot_see_the_present_row():
    """A report's own damage must not appear in its own history."""
    df = _sightings([
        {"Date": pd.Timestamp("2026-03-01"), "Latitude": 23.0, "Longitude": 81.0,
         "Crop Damage": 1},
    ])
    _, features = sighting_features(df, _centroids())
    assert features["prior_conflicts_2km_90d"].iloc[0] == 0
    assert features["prior_conflicts_2km_all"].iloc[0] == 0


def test_history_cannot_see_later_rows():
    """A quiet January must not be told about a damaging March."""
    df = _sightings([
        {"Date": pd.Timestamp("2026-01-01"), "Latitude": 23.0, "Longitude": 81.0},
        {"Date": pd.Timestamp("2026-03-01"), "Latitude": 23.0, "Longitude": 81.0,
         "Crop Damage": 1},
        {"Date": pd.Timestamp("2026-04-01"), "Latitude": 23.0, "Longitude": 81.0},
    ])
    _, features = sighting_features(df, _centroids())
    assert features["prior_conflicts_2km_all"].tolist() == [0, 0, 1]


def test_history_is_local_not_landscape_wide():
    """Damage 60 km away is not this village's history."""
    df = _sightings([
        {"Date": pd.Timestamp("2026-01-01"), "Latitude": 23.5, "Longitude": 81.5,
         "Crop Damage": 1},
        {"Date": pd.Timestamp("2026-02-01"), "Latitude": 23.0, "Longitude": 81.0},
    ])
    _, features = sighting_features(df, _centroids())
    assert features["prior_conflicts_2km_all"].iloc[1] == 0


def test_history_window_expires():
    """The 90-day window drops what is older; the all-time count keeps it."""
    df = _sightings([
        {"Date": pd.Timestamp("2026-01-01"), "Latitude": 23.0, "Longitude": 81.0,
         "Crop Damage": 1},
        {"Date": pd.Timestamp("2026-07-01"), "Latitude": 23.0, "Longitude": 81.0},
    ])
    _, features = sighting_features(df, _centroids())
    assert features["prior_conflicts_2km_90d"].iloc[1] == 0
    assert features["prior_conflicts_2km_all"].iloc[1] == 1
    assert features["days_since_conflict_2km"].iloc[1] == 181


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
def _panel_fixture():
    rows = []
    for month, damage in [(1, 0), (2, 1), (3, 0), (4, 1)]:
        for day in (5, 15):
            rows.append({
                "Date": pd.Timestamp(f"2026-{month:02d}-{day:02d}"),
                "Latitude": 23.0, "Longitude": 81.0, "Crop Damage": damage,
            })
    return _sightings(rows)


def test_panel_target_month_is_excluded_from_its_own_features():
    panel = village_month_panel(_panel_fixture(), _centroids())
    february = panel[(panel["village"] == "Near") & (panel["period"] == "2026-02")]
    assert february["y"].iloc[0] == 1, "February had damage"
    assert february["conf_prev_month"].iloc[0] == 0, "January was quiet"
    assert february["conf_all"].iloc[0] == 0, "nothing preceded February"


def test_panel_history_accumulates_over_months():
    panel = village_month_panel(_panel_fixture(), _centroids())
    near = panel[panel["village"] == "Near"].set_index("period")
    assert near.loc["2026-03", "conf_all"] == 2, "two damaging reports in February"
    assert near.loc["2026-04", "conf_all"] == 2


def test_panel_skips_the_first_month():
    """There is no history to score it against."""
    panel = village_month_panel(_panel_fixture(), _centroids())
    assert "2026-01" not in set(panel["period"])


def test_panel_only_includes_villages_elephants_came_near():
    panel = village_month_panel(_panel_fixture(), _centroids())
    assert set(panel["village"]) == {"Near"}


def test_panel_feature_columns_exclude_identifiers_and_target():
    panel = village_month_panel(_panel_fixture(), _centroids())
    columns = panel_feature_columns(panel)
    for excluded in ("village", "period", "y", "month_index"):
        assert excluded not in columns
    assert "conf_prev_quarter" in columns


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def test_lone_bull_and_family_herd_are_distinguished():
    df = _sightings([
        {"Date": pd.Timestamp("2026-01-01"), "Latitude": 23.0, "Longitude": 81.0,
         "Male Count": 1, "Total Count": 1},
        {"Date": pd.Timestamp("2026-01-02"), "Latitude": 23.0, "Longitude": 81.0,
         "Male Count": 0, "Female Count": 4, "Calf Count": 2, "Total Count": 6},
    ])
    _, features = sighting_features(df, _centroids())
    assert features["lone_male"].tolist() == [1, 0]
    assert features["calf_present"].tolist() == [0, 1]
    assert features["male_fraction"].tolist() == [1.0, 0.0]


def test_male_fraction_survives_a_miscounted_row():
    """Male Count above Total Count occurs in the real export."""
    df = _sightings([
        {"Date": pd.Timestamp("2026-01-01"), "Latitude": 23.0, "Longitude": 81.0,
         "Male Count": 3, "Total Count": 1},
    ])
    _, features = sighting_features(df, _centroids())
    assert features["male_fraction"].iloc[0] == 1.0
