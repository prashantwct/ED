"""Tests for group composition in the live pipeline.

The bull/herd split is the most robust finding in the research work --
it held at every clustering setting tried -- so it is the one that got
wired into the app. These tests pin down what it must and must not do.

The load-bearing constraint is that composition changes the *ordering
score and the recommended action*, never the tier. Tiers record harm
that has already happened; composition predicts harm that has not. If a
tier could move on composition alone, "Critical" would stop meaning the
same thing month to month.
"""

import pandas as pd
import pytest

from core.analytics import classify_group, composition_summary, is_bull_type
from core.intelligence import beat_intelligence, management_brief

BASE = {
    "Division": "D", "Range": "R", "Beat": "B", "Latitude": 23.0,
    "Longitude": 81.0, "Hour": 20, "Male Count": 0, "Female Count": 0,
    "Calf Count": 0, "Unknown Count": 0, "Total Count": 0,
    "Crop Damage": 0, "Grain Damage": 0, "House Damage": 0,
    "Injury": 0, "Death": 0,
}


def _df(rows):
    return pd.DataFrame([{**BASE, **row} for row in rows])


def _bull(day=1, **extra):
    return {"Date": pd.Timestamp(f"2026-06-{day:02d}"), "Male Count": 1,
            "Total Count": 1, **extra}


def _herd(day=1, **extra):
    return {"Date": pd.Timestamp(f"2026-06-{day:02d}"), "Female Count": 5,
            "Calf Count": 2, "Total Count": 7, **extra}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def test_group_types_are_read_from_composition():
    df = _df([
        {"Male Count": 1, "Total Count": 1},
        {"Male Count": 3, "Total Count": 3},
        {"Female Count": 5, "Calf Count": 2, "Total Count": 7},
        {"Unknown Count": 9, "Total Count": 9},
        {},
    ])
    assert classify_group(df).tolist() == [
        "Lone bull", "Bull party", "Family herd", "Mixed / unsexed", "Unrecorded",
    ]


def test_a_female_without_calves_is_still_a_breeding_herd():
    """Cows do not range alone the way bulls do."""
    df = _df([{"Female Count": 3, "Total Count": 3}])
    assert classify_group(df).iloc[0] == "Family herd"


def test_a_large_all_male_group_is_not_called_a_bull_party():
    """Above the solitary threshold the label stops being useful."""
    df = _df([{"Male Count": 9, "Total Count": 9}])
    assert classify_group(df).iloc[0] == "Mixed / unsexed"


def test_unrecorded_composition_is_not_folded_into_a_class():
    """Guessing here would put weight on the app's strongest claim."""
    df = _df([{"Total Count": 4}])
    assert classify_group(df).iloc[0] == "Unrecorded"
    assert not is_bull_type(df).iloc[0]


def test_bull_type_covers_lone_bulls_and_small_parties_only():
    df = _df([
        {"Male Count": 1, "Total Count": 1},
        {"Male Count": 2, "Total Count": 2},
        {"Female Count": 5, "Calf Count": 1, "Total Count": 6},
    ])
    assert is_bull_type(df).tolist() == [True, True, False]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def test_composition_summary_reports_damage_rate_per_group():
    df = _df([
        _bull(1, **{"Crop Damage": 1}), _bull(2), _bull(3), _bull(4),
        _herd(5), _herd(6),
    ])
    summary = composition_summary(df).set_index("Group Type")
    assert summary.loc["Lone bull", "Sightings"] == 4
    assert summary.loc["Lone bull", "Damage Rate %"] == 25.0
    assert summary.loc["Family herd", "Damage Rate %"] == 0.0


def test_composition_summary_is_empty_for_an_empty_frame():
    assert composition_summary(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# Beat table
# ---------------------------------------------------------------------------
def test_bull_share_measures_conflict_not_sightings():
    """A beat full of quiet bulls is not a bull-driven conflict beat."""
    rows = [_bull(day) for day in range(1, 9)]          # 8 quiet bulls
    rows += [_herd(day, **{"Crop Damage": 1}) for day in range(9, 13)]
    table = beat_intelligence(_df(rows))
    assert table["Bull-Type Conflict %"].iloc[0] == 0.0


def test_bull_share_is_reported_when_bulls_cause_the_conflict():
    rows = [_bull(day, **{"Crop Damage": 1}) for day in range(1, 5)]
    rows += [_herd(day) for day in range(5, 9)]
    table = beat_intelligence(_df(rows))
    assert table["Bull-Type Conflict %"].iloc[0] == 100.0


def test_composition_never_moves_a_beat_between_tiers():
    """The load-bearing invariant. Same incidents, different animals."""
    damage = {"Crop Damage": 1, "House Damage": 1}
    by_bulls = _df([_bull(day, **damage) for day in range(1, 13)])
    by_herds = _df([_herd(day, **damage) for day in range(1, 13)])

    bull_tier = beat_intelligence(by_bulls)["Priority Tier"].iloc[0]
    herd_tier = beat_intelligence(by_herds)["Priority Tier"].iloc[0]
    assert bull_tier == herd_tier


def test_composition_does_move_the_ordering_score():
    """It is allowed to change the order within a tier, and should."""
    damage = {"Crop Damage": 1, "House Damage": 1}
    bull_score = beat_intelligence(
        _df([_bull(day, **damage) for day in range(1, 13)])
    )["Priority Score"].iloc[0]
    herd_score = beat_intelligence(
        _df([_herd(day, **damage) for day in range(1, 13)])
    )["Priority Score"].iloc[0]
    assert bull_score > herd_score


def test_a_beat_with_too_little_conflict_gets_no_composition_claim():
    """One raid is not evidence about which animal is responsible."""
    table = beat_intelligence(_df([_bull(1, **{"Crop Damage": 1}), _bull(2)]))
    assert "Bull-driven" not in table["Recommended Action"].iloc[0]
    assert "Herd movement" not in table["Recommended Action"].iloc[0]


# ---------------------------------------------------------------------------
# Recommended action
# ---------------------------------------------------------------------------
def test_bull_driven_beats_are_told_to_identify_the_animal():
    rows = [_bull(day, **{"Crop Damage": 1}) for day in range(1, 7)]
    action = beat_intelligence(_df(rows))["Recommended Action"].iloc[0]
    assert "Bull-driven" in action
    assert "identify the animal" in action


def test_herd_driven_beats_are_told_not_to_drive_the_herd():
    """Getting this backwards is how people get killed."""
    rows = [_herd(day, **{"Crop Damage": 1}) for day in range(1, 7)]
    action = beat_intelligence(_df(rows))["Recommended Action"].iloc[0]
    assert "do not drive the herd" in action
    assert "Bull-driven" not in action


# ---------------------------------------------------------------------------
# Brief
# ---------------------------------------------------------------------------
def test_the_brief_states_the_split():
    rows = [_bull(day, **{"Crop Damage": 1}) for day in range(1, 7)]
    rows += [_herd(day) for day in range(7, 13)]
    brief = management_brief(_df(rows))
    assert any("bull" in line.lower() for line in brief["headlines"])
    assert not brief["composition"].empty


def test_the_brief_stays_quiet_when_there_is_nothing_to_say():
    """Two conflicts cannot support a claim about which animal."""
    brief = management_brief(_df([_bull(1, **{"Crop Damage": 1}), _herd(2)]))
    assert not any("came from lone bulls" in line for line in brief["headlines"])


def test_composition_survives_a_frame_with_no_count_columns():
    """Older exports lack the composition fields entirely."""
    rows = [{"Date": pd.Timestamp("2026-06-01"), "Division": "D", "Range": "R",
             "Beat": "B", "Latitude": 23.0, "Longitude": 81.0,
             "Crop Damage": 1}]
    table = beat_intelligence(pd.DataFrame(rows))
    assert len(table) == 1
    assert pd.isna(table["Bull-Type Conflict %"].iloc[0]) or \
        table["Bull-Type Conflict %"].iloc[0] == 0.0
