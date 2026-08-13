"""Tests for movement-unit tracking.

The tracker's job is to link reports that could be the same animals and
refuse to link ones that could not. Its characteristic failure is
chaining -- every constraint relaxed slightly, until one unit covers the
whole landscape and the output means nothing. Most of these tests are
about the refusals.
"""

import numpy as np
import pandas as pd
import pytest

from research.herds import assign_units, sensitivity, summarise_units

BASE = {
    "Division": "D", "Range": "R", "Beat": "B", "Sighting Type": "Direct",
    "Hour": 20, "Male Count": 1, "Female Count": 0, "Calf Count": 0,
    "Unknown Count": 0, "Total Count": 1, "Crop Damage": 0, "Grain Damage": 0,
    "House Damage": 0, "Injury": 0, "Death": 0,
}


def _df(rows):
    return pd.DataFrame([{**BASE, **row} for row in rows])


def _bull(day, lat=23.0, lon=81.0, **extra):
    return {"Date": pd.Timestamp(f"2026-01-{day:02d}"),
            "Latitude": lat, "Longitude": lon, **extra}


def _herd(day, lat=23.0, lon=81.0, size=8, calves=2, **extra):
    return {"Date": pd.Timestamp(f"2026-01-{day:02d}"), "Latitude": lat,
            "Longitude": lon, "Male Count": 0, "Female Count": size - calves,
            "Calf Count": calves, "Total Count": size, **extra}


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------
def test_same_animal_on_consecutive_days_is_one_unit():
    units = assign_units(_df([_bull(1), _bull(2, lat=23.03)]))
    assert units.nunique() == 1


def test_repeat_reports_on_one_day_do_not_become_separate_animals():
    """Trackers log the same bull several times a day."""
    units = assign_units(_df([_bull(1), _bull(1, lat=23.01), _bull(1, lon=81.02)]))
    assert units.nunique() == 1


def test_two_animals_far_apart_on_the_same_day_stay_separate():
    """Nothing walks 200 km between two reports on one day."""
    units = assign_units(_df([_bull(1, lat=23.0), _bull(1, lat=24.8)]))
    assert units.nunique() == 2


def test_a_long_reporting_gap_closes_the_track():
    units = assign_units(_df([_bull(1), _bull(20)]))
    assert units.nunique() == 2


def test_movement_faster_than_an_elephant_is_two_animals():
    """0.9 degrees of latitude is about 100 km, in one day."""
    units = assign_units(_df([_bull(1, lat=23.0), _bull(2, lat=23.9)]))
    assert units.nunique() == 2


# ---------------------------------------------------------------------------
# Composition -- what stops the chaining
# ---------------------------------------------------------------------------
def test_a_lone_bull_and_a_family_herd_are_never_the_same_unit():
    units = assign_units(_df([_bull(1), _herd(1, lat=23.001)]))
    assert units.nunique() == 2


def test_a_herd_does_not_acquire_calves_between_sightings():
    with_calves = _herd(1, size=8, calves=2)
    without = _herd(2, size=8, calves=0, lat=23.01)
    assert assign_units(_df([with_calves, without])).nunique() == 2


def test_a_miscounted_herd_is_still_the_same_herd():
    """Nine reported as eight must not split the track."""
    units = assign_units(_df([_herd(1, size=9, calves=2),
                              _herd(2, size=8, calves=2, lat=23.02)]))
    assert units.nunique() == 1


def test_a_big_size_change_splits_the_track():
    units = assign_units(_df([_herd(1, size=3, calves=1),
                              _herd(2, size=20, calves=4, lat=23.01)]))
    assert units.nunique() == 2


def test_two_bulls_in_the_same_valley_chain_into_one_unit():
    """A known limit, asserted so it cannot be forgotten.

    With no individual identification, two bulls working the same ground
    on alternating days are one track. The output is a movement unit,
    not an animal.
    """
    rows = [_bull(day, lat=23.0 + 0.005 * (day % 2)) for day in range(1, 9)]
    assert assign_units(_df(rows)).nunique() == 1


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def test_a_single_sighting_has_no_range_and_no_path():
    table = summarise_units(_df([_bull(1)]))
    assert len(table) == 1
    assert table["Range (km2)"].iloc[0] == 0.0
    assert table["Path (km)"].iloc[0] == 0.0


def test_path_length_follows_the_track_in_order():
    """Three points 0.01 degrees apart: roughly 1.1 km a step."""
    table = summarise_units(_df([_bull(1, lat=23.00), _bull(2, lat=23.01),
                                 _bull(3, lat=23.02)]))
    assert table["Path (km)"].iloc[0] == pytest.approx(2.21, abs=0.1)
    assert table["Days Observed"].iloc[0] == 3
    assert table["Span (days)"].iloc[0] == 2


def test_range_area_matches_a_known_square():
    """Corners of roughly a 1.1 km square enclose about 1.2 km²."""
    rows = [_bull(1, lat=23.00, lon=81.00), _bull(1, lat=23.01, lon=81.00),
            _bull(2, lat=23.01, lon=81.0109), _bull(2, lat=23.00, lon=81.0109)]
    table = summarise_units(_df(rows))
    assert table["Range (km2)"].iloc[0] == pytest.approx(1.23, rel=0.15)


def test_social_class_is_assigned_from_composition():
    table = summarise_units(_df([_bull(1), _herd(1, lat=23.5)]))
    assert set(table["Class"]) == {"lone bull", "family herd"}


def test_conflict_is_attributed_to_the_unit_that_caused_it():
    rows = [_bull(1, **{"Crop Damage": 1}), _bull(2, lat=23.01),
            _herd(1, lat=23.6)]
    table = summarise_units(_df(rows)).set_index("Class")
    assert table.loc["lone bull", "Conflict Events"] == 1
    assert table.loc["family herd", "Conflict Events"] == 0


def test_polygon_is_returned_for_drawing():
    rows = [_bull(1, lat=23.00, lon=81.00), _bull(1, lat=23.01, lon=81.00),
            _bull(2, lat=23.01, lon=81.01)]
    polygon = summarise_units(_df(rows))["Polygon"].iloc[0]
    assert len(polygon) >= 3
    assert all(len(point) == 2 for point in polygon)


# ---------------------------------------------------------------------------
# Honesty about the thresholds
# ---------------------------------------------------------------------------
def test_relaxing_the_thresholds_merges_units():
    """The unit count is a function of the parameters, and the
    sensitivity table has to show that rather than hide it."""
    rows = [_bull(day, lat=23.0 + 0.02 * day) for day in range(1, 10)]
    tight = assign_units(_df(rows), max_speed_km_per_day=1.0, max_gap_days=1)
    loose = assign_units(_df(rows), max_speed_km_per_day=30.0, max_gap_days=7)
    assert tight.nunique() > loose.nunique()


def test_sensitivity_reports_one_row_per_setting():
    rows = [_bull(day) for day in range(1, 6)]
    table = sensitivity(_df(rows), speeds=(10.0, 20.0), gaps=(2, 4))
    assert len(table) == 4
    assert {"units", "singletons", "largest"}.issubset(table.columns)
