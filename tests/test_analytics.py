"""
Smoke + regression tests for the analytics/risk-engine core.

These exist mainly to lock in two specific regressions found during the
July 2026 review: human deaths scoring below routine crop damage, and
the conflict KPI silently excluding pure-fatality reports. If either
comes back, one of these should fail loudly instead of shipping quietly.

Run with: pytest tests/ -v
"""
import pandas as pd
import pytest

from core.analytics import (
    classify_conflict,
    compute_kpis,
    compute_severity,
    division_conflict_rate,
    hourly_conflict_profile,
    monthly_trend,
)


def _row(**overrides):
    base = {
        "Total Count": 1, "Crop Damage": 0, "Grain Damage": 0, "House Damage": 0,
        "Injury": 0, "Death": 0, "Male Death Count": 0, "Female Death Count": 0,
        "Children Death Count": 0, "Division": "Test Division", "Range": "Test Range",
        "Beat": "Test Beat", "Is_Night": False,
    }
    base.update(overrides)
    return base


def make_df(rows):
    return pd.DataFrame(rows)


def test_death_outranks_ordinary_crop_damage():
    """The bug this whole review started with: a fatality must never
    score at or below a routine crop-damage report."""
    df = make_df([
        _row(Death=1, **{"Male Death Count": 1}),
        _row(**{"Crop Damage": 1}),
    ])
    df["Severity Score"] = compute_severity(df)
    death_score, crop_score = df["Severity Score"]
    assert death_score > crop_score, "a human death must outscore plain crop damage"


def test_two_person_fatality_scores_higher_than_one():
    df = make_df([
        _row(Death=1, **{"Male Death Count": 1, "Female Death Count": 1}),
        _row(Death=1, **{"Male Death Count": 1}),
    ])
    df["Severity Score"] = compute_severity(df)
    two_person, one_person = df["Severity Score"]
    assert two_person > one_person


def test_death_count_without_flag_is_not_counted_as_a_fatality():
    """A death-count field populated without the Death flag set is
    treated as a duplicate/follow-up report, not a new fatality --
    otherwise the same real death gets counted twice."""
    df = make_df([
        _row(Death=0, **{"Male Death Count": 1}),  # flag not set -> not counted
    ])
    kpis = compute_kpis(df)
    assert kpis["human_deaths"] == 0
    assert kpis["human_death_incidents"] == 0


def test_conflict_kpi_includes_pure_fatality_reports():
    """A report that is ONLY a death (no crop/house/injury flag) must
    still count as a conflict event -- this is exactly what the
    original app's KPI silently dropped."""
    df = make_df([
        _row(Death=1, **{"Male Death Count": 1}),  # death only, nothing else
    ])
    kpis = compute_kpis(df)
    assert kpis["conflicts"] == 1


def test_classify_conflict_prioritizes_death_over_other_damage():
    df = make_df([
        _row(Death=1, **{"Male Death Count": 1}, **{"House Damage": 1, "Crop Damage": 1}),
    ])
    cat = classify_conflict(df)
    assert cat.iloc[0] == "Death"


def test_division_conflict_rate_normalizes_not_just_counts():
    df = make_df([
        _row(Division="Small Division", **{"Crop Damage": 1}),
        _row(Division="Small Division"),
        _row(Division="Big Division", **{"Crop Damage": 1}),
        *([_row(Division="Big Division")] * 9),
    ])
    rates = division_conflict_rate(df)
    assert rates.loc["Small Division", "Conflict Rate %"] == 50.0
    assert rates.loc["Big Division", "Conflict Rate %"] == 10.0
    # Small Division has fewer sightings but a higher rate -- rank must follow rate, not volume.
    assert rates.index[0] == "Small Division"


def test_monthly_trend_and_hourly_profile_run_without_error():
    df = make_df([_row(**{"Crop Damage": 1}, **{"Hour": 21})])
    df["Date"] = pd.to_datetime(["2026-02-01"])
    df["Hour"] = [21]
    trend = monthly_trend(df)
    hourly = hourly_conflict_profile(df)
    assert len(trend) == 1
    assert len(hourly) == 24  # every hour represented, zero-filled where empty


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
