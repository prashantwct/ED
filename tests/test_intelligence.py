"""Tests for the conservation intelligence layer.

These lock in the properties that make the beat ranking usable for a
posting decision, as opposed to merely producing a number:

* a beat cannot reach the top on one lucky report (rate shrinkage),
* a fatality outranks any volume of crop damage (tiering),
* tiers do not move when the user changes a filter (stability),
* "escalating" means a real change against a like-for-like window,
* flat data does not get labelled as having a peak.

Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from core.analytics import compute_is_night, compute_severity
from core.intelligence import (
    CONFIDENCE_LOW,
    TIER_CRITICAL,
    TIER_ROUTINE,
    beat_intelligence,
    format_window,
    management_brief,
    shrink_rates,
    temporal_risk_windows,
    village_exposure,
    _peak_hour_window,
    _peak_months,
)

START = pd.Timestamp("2026-01-01")


def _rows(beat, n, conflict_n=0, deaths=0, injuries=0, hour=12, day_step=1,
          division="D1", range_name="R1", start=START, house=False):
    """Build ``n`` reports for one beat, ``conflict_n`` of them conflicts."""
    out = []
    for i in range(n):
        is_conflict = i < conflict_n
        out.append({
            "Date": start + pd.Timedelta(days=i * day_step),
            "Division": division, "Range": range_name, "Beat": beat,
            "Latitude": 22.3, "Longitude": 80.6, "Hour": hour,
            "Total Count": 1,
            "Crop Damage": int(is_conflict and not house),
            "Grain Damage": 0,
            "House Damage": int(is_conflict and house),
            "Injury": 1 if (injuries and i < injuries) else 0,
            "Death": 1 if (deaths and i < deaths) else 0,
            "Male Death Count": 1 if (deaths and i < deaths) else 0,
            "Female Death Count": 0, "Children Death Count": 0,
        })
    return out


def _prepare(rows):
    df = pd.DataFrame(rows)
    df["Severity Score"] = compute_severity(df)
    df["Is_Night"] = compute_is_night(df)
    return df


# ---------------------------------------------------------------------------
# Rate shrinkage
# ---------------------------------------------------------------------------
def test_shrinkage_pulls_a_one_report_beat_off_the_top():
    """A beat with 1 report and 1 conflict shows a 100% raw rate. It must
    not outrank a beat with a large, consistently high sample."""
    result = shrink_rates(successes=[1, 90, 20], trials=[1, 200, 100])
    adjusted = result["adjusted"]
    assert adjusted[0] < 1.0, "a 1-of-1 beat must be pulled below 100%"
    assert adjusted[0] < adjusted[1], "1-of-1 must not outrank 90-of-200"


def test_shrinkage_leaves_well_evidenced_rates_roughly_intact():
    result = shrink_rates(successes=[90, 20], trials=[200, 100])
    assert result["adjusted"][0] == pytest.approx(0.45, abs=0.08)


def test_shrinkage_handles_zero_trials_without_dividing_by_zero():
    result = shrink_rates(successes=[], trials=[])
    assert result["prior_mean"] == 0.0
    assert len(result["adjusted"]) == 0


def test_prior_strength_is_higher_when_beats_look_alike():
    """Homogeneous groups should be shrunk harder than heterogeneous ones."""
    alike = shrink_rates([25, 25, 25, 25], [100, 100, 100, 100])
    differing = shrink_rates([5, 95, 10, 90], [100, 100, 100, 100])
    assert alike["prior_strength"] > differing["prior_strength"]


# ---------------------------------------------------------------------------
# Tiering
# ---------------------------------------------------------------------------
def test_a_fatality_beat_outranks_a_high_volume_crop_damage_beat():
    """The whole point of the ranking: no amount of crop damage should
    displace the beat where someone was killed."""
    df = _prepare(
        _rows("CropHeavy", 200, conflict_n=150)
        + _rows("Fatal", 20, conflict_n=5, deaths=1, division="D2", range_name="R2")
    )
    table = beat_intelligence(df)
    assert table.iloc[0]["Beat"] == "Fatal"
    assert table.iloc[0]["Priority Tier"] == TIER_CRITICAL


def test_quiet_beat_is_routine():
    df = _prepare(_rows("Quiet", 40, conflict_n=1) + _rows("Busy", 40, conflict_n=30,
                                                           division="D2", range_name="R2"))
    table = beat_intelligence(df).set_index("Beat")
    assert table.loc["Quiet", "Priority Tier"] == TIER_ROUTINE


def test_tier_is_stable_when_other_beats_are_filtered_out():
    """A tier must describe the beat, not its rank on the current screen.
    Filtering away unrelated beats must not change the survivor's tier."""
    full = _prepare(
        _rows("Fatal", 20, conflict_n=5, deaths=1)
        + _rows("Other", 200, conflict_n=150, division="D2", range_name="R2")
    )
    tier_together = (
        beat_intelligence(full).set_index("Beat").loc["Fatal", "Priority Tier"]
    )

    alone = _prepare(_rows("Fatal", 20, conflict_n=5, deaths=1))
    tier_alone = beat_intelligence(alone).set_index("Beat").loc["Fatal", "Priority Tier"]

    assert tier_together == tier_alone == TIER_CRITICAL


def test_a_landscape_with_no_conflict_is_all_routine():
    """With no conflict anywhere the landscape rate is zero, so a naive
    'rate >= threshold' test is true for every beat and would promote a
    completely quiet landscape wholesale."""
    df = _prepare(
        _rows("A", 40, conflict_n=0)
        + _rows("B", 40, conflict_n=0, division="D2", range_name="R2")
    )
    table = beat_intelligence(df)
    assert (table["Priority Tier"] == TIER_ROUTINE).all()


def test_thin_beat_is_marked_low_confidence():
    df = _prepare(_rows("Thin", 3, conflict_n=3) + _rows("Thick", 60, conflict_n=20,
                                                         division="D2", range_name="R2"))
    table = beat_intelligence(df).set_index("Beat")
    assert table.loc["Thin", "Confidence"] == CONFIDENCE_LOW
    assert table.loc["Thin", "Adj. Conflict Rate %"] < table.loc["Thin", "Conflict Rate %"]


def test_same_beat_name_in_two_ranges_is_not_merged():
    """Beat names repeat across ranges; merging them would attribute one
    range's fatality to the other's staff."""
    df = _prepare(
        _rows("Kisli", 20, conflict_n=5, division="D1", range_name="R1")
        + _rows("Kisli", 20, conflict_n=5, division="D1", range_name="R2", deaths=1)
    )
    table = beat_intelligence(df)
    assert len(table) == 2
    assert set(table["Range"]) == {"R1", "R2"}


def test_every_beat_gets_a_recommended_action():
    df = _prepare(_rows("A", 30, conflict_n=10) + _rows("B", 30, conflict_n=1,
                                                        division="D2", range_name="R2"))
    table = beat_intelligence(df)
    assert table["Recommended Action"].str.len().gt(0).all()


def test_empty_input_returns_empty_table_with_columns():
    table = beat_intelligence(pd.DataFrame())
    assert table.empty
    assert "Priority Tier" in table.columns


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------
def test_escalating_beat_is_detected():
    """Few conflicts in the first window, many in the second."""
    rows = []
    for i in range(40):  # prior window: 4 conflicts
        rows += _rows("Esc", 1, conflict_n=1 if i < 4 else 0, start=START + pd.Timedelta(days=i))
    for i in range(40):  # recent window: 25 conflicts
        rows += _rows("Esc", 1, conflict_n=1 if i < 25 else 0,
                      start=START + pd.Timedelta(days=40 + i))
    table = beat_intelligence(_prepare(rows), recent_days=40).set_index("Beat")
    assert table.loc["Esc", "Trend"] == "Escalating"


def test_steady_beat_is_not_called_escalating():
    rows = []
    for i in range(80):
        rows += _rows("Steady", 1, conflict_n=i % 2, start=START + pd.Timedelta(days=i))
    table = beat_intelligence(_prepare(rows), recent_days=40).set_index("Beat")
    assert table.loc["Steady", "Trend"] == "Stable"


def test_no_trend_claimed_on_too_few_events():
    rows = []
    for i in range(60):
        rows += _rows("Sparse", 1, conflict_n=1 if i in (0, 55) else 0,
                      start=START + pd.Timedelta(days=i))
    table = beat_intelligence(_prepare(rows), recent_days=30).set_index("Beat")
    assert table.loc["Sparse", "Trend"] == "Insufficient data"


def test_no_trend_claimed_when_period_is_too_short():
    df = _prepare(_rows("Short", 10, conflict_n=8))  # 10 days total
    table = beat_intelligence(df, recent_days=90).set_index("Beat")
    assert table.loc["Short", "Trend"] == "Insufficient data"


# ---------------------------------------------------------------------------
# Temporal windows
# ---------------------------------------------------------------------------
def test_peak_window_wraps_around_midnight():
    """Elephant activity straddles midnight; a non-circular scan would
    split the night in two and understate it."""
    hourly = pd.Series([0] * 24)
    for h in [21, 22, 23, 0, 1, 2]:
        hourly[h] = 20

    # Demanding full coverage forces the window to span the whole block,
    # which it can only do by wrapping past midnight.
    window = _peak_hour_window(hourly, coverage_target=1.0)
    assert window["start"] == 21
    assert window["hours"] == 6
    assert window["end"] == 3
    assert window["share"] == pytest.approx(100.0)

    # At a lower target it should still start at 21 rather than splitting
    # the night into a pre- and post-midnight fragment.
    shorter = _peak_hour_window(hourly, coverage_target=0.6)
    assert shorter["start"] == 21
    assert shorter["share"] == pytest.approx(66.7, abs=0.1)


def test_flat_hours_report_a_lift_of_about_one():
    """A window covering 60% of a flat day is not a finding, and the lift
    figure is what says so."""
    window = _peak_hour_window(pd.Series([10] * 24), coverage_target=0.6)
    assert window["lift"] == pytest.approx(1.0, abs=0.05)


def test_concentrated_hours_report_a_high_lift():
    hourly = pd.Series([1] * 24)
    for h in [19, 20, 21]:
        hourly[h] = 60
    window = _peak_hour_window(hourly, coverage_target=0.6)
    assert window["lift"] > 4


def test_flat_months_produce_no_seasonal_claim():
    months = pd.Series([20] * 12, index=[f"M{i}" for i in range(12)])
    assert _peak_months(months) == []


def test_seasonal_months_are_reported():
    months = pd.Series([2] * 12, index=[f"M{i}" for i in range(12)])
    months.iloc[8] = 60
    months.iloc[9] = 50
    assert _peak_months(months) == ["M8", "M9"]


def test_format_window_handles_none():
    assert "not determinable" in format_window(None)


def test_temporal_windows_uses_conflict_rows_only():
    """A profile of all sightings mostly shows when patrols go out."""
    rows = _rows("A", 20, conflict_n=0, hour=9) + _rows("A", 5, conflict_n=5, hour=22)
    result = temporal_risk_windows(_prepare(rows))
    assert result["hourly"][22] == 5
    assert result["hourly"][9] == 0


# ---------------------------------------------------------------------------
# Village exposure
# ---------------------------------------------------------------------------
def test_village_exposure_is_empty_without_centroids():
    assert village_exposure(_prepare(_rows("A", 10, conflict_n=5))).empty


def test_village_exposure_ranks_by_casualties_first():
    df = _prepare(_rows("A", 10, conflict_n=10) + _rows("B", 10, conflict_n=2, deaths=1,
                                                        division="D2", range_name="R2"))
    df["Nearest Village"] = ["Manypest"] * 10 + ["Fatalville"] * 10
    df["Distance to Village (km)"] = 0.5
    df["Near Village"] = True

    out = village_exposure(df)
    assert out.iloc[0]["Village"] == "Fatalville"


def test_village_exposure_excludes_distant_incidents():
    df = _prepare(_rows("A", 10, conflict_n=10))
    df["Nearest Village"] = "Faraway"
    df["Distance to Village (km)"] = 40.0
    df["Near Village"] = False
    assert village_exposure(df).empty


# ---------------------------------------------------------------------------
# The brief
# ---------------------------------------------------------------------------
def test_brief_has_every_expected_section():
    brief = management_brief(_prepare(_rows("A", 60, conflict_n=20, deaths=1)))
    for key in ["period", "coverage", "kpis", "beats", "priority_beats",
                "escalating", "temporal", "villages", "headlines", "caveats"]:
        assert key in brief


def test_brief_always_states_the_reporting_effort_caveat():
    """Managers must never read the volumes as pure elephant activity."""
    brief = management_brief(_prepare(_rows("A", 60, conflict_n=20)))
    assert any("effort" in c for c in brief["caveats"])


def test_brief_survives_an_empty_dataframe():
    brief = management_brief(pd.DataFrame())
    assert brief["kpis"]["entries"] == 0
    assert brief["beats"].empty
    assert brief["headlines"]


def test_brief_flags_missing_village_data():
    brief = management_brief(_prepare(_rows("A", 30, conflict_n=10)))
    assert any("village" in c.lower() for c in brief["caveats"])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
