"""The Overview's words, from the analysis's own tables."""

import pandas as pd

from core import overview


def _beats(rows):
    base = {"Division": "D", "Range": "R", "Priority Score": 50.0, "Recent Deaths": 0,
            "Recent Injuries": 0, "Human Deaths": 0, "Crop Damage Events": 1,
            "House Damage Events": 0, "Bull-Type Conflict %": float("nan"),
            "Group-Known Conflicts": 0, "Trend": "Stable", "Recent Conflicts": 1,
            "Prior Conflicts": 1, "Recommended Action": "Routine monitoring"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_the_change_sentence_prefers_the_same_days_last_year():
    said = overview.change_sentence(
        {"conflicts_recent": 81, "conflicts_prior": 60, "conflicts_year": 65}, 90)
    assert said == "Conflict is up 25% on the same 90 days last year (81 events against 65)."


def test_the_change_sentence_falls_back_to_the_days_before():
    said = overview.change_sentence(
        {"conflicts_recent": 30, "conflicts_prior": 60, "conflicts_year": float("nan")}, 90)
    assert said.startswith("Conflict is down 50% on the 90 days before")


def test_a_small_change_reads_as_level():
    said = overview.change_sentence(
        {"conflicts_recent": 100, "conflicts_prior": float("nan"), "conflicts_year": 98}, 30)
    assert said.startswith("Conflict is level with the same 30 days last year")


def test_no_comparison_says_nothing_rather_than_guessing():
    assert overview.change_sentence({"conflicts_recent": 5}, 90) is None


def test_the_headline_counts_critical_beats_and_new_ones():
    beats = _beats([{"Beat": "A", "Priority Tier": "Critical"},
                    {"Beat": "B", "Priority Tier": "Critical"},
                    {"Beat": "C", "Priority Tier": "High"}])
    said = overview.headline({}, 90, beats, ["B"], None)
    assert said == "2 beats are Critical, 1 of them new in the last 30 days."


def test_the_headline_names_critical_villages_outside_hotspots():
    villages = pd.DataFrame({"Village": ["V1", "V2"], "Tier": ["Critical", "Critical"],
                             "Inside Hotspot": [False, True]})
    said = overview.headline({}, 90, _beats([{"Beat": "A", "Priority Tier": "High"}]),
                             [], villages)
    assert "No beat is Critical." in said
    assert "1 Critical village lies outside every hotspot" in said


def test_act_now_lists_critical_then_high_by_score_and_nothing_lower():
    beats = _beats([
        {"Beat": "Watchful", "Priority Tier": "Watch", "Priority Score": 99},
        {"Beat": "High low", "Priority Tier": "High", "Priority Score": 10},
        {"Beat": "High top", "Priority Tier": "High", "Priority Score": 90},
        {"Beat": "Crit", "Priority Tier": "Critical", "Priority Score": 5},
    ])
    rows = overview.act_now(beats)
    assert [r["beat"] for r in rows] == ["Crit", "High top", "High low"]
    assert rows[0]["do"] == "Routine monitoring"


def test_new_critical_compares_with_the_tiers_a_month_earlier():
    now = _beats([{"Beat": "A", "Priority Tier": "Critical"},
                  {"Beat": "B", "Priority Tier": "Critical"}])
    before = _beats([{"Beat": "A", "Priority Tier": "Critical"},
                     {"Beat": "B", "Priority Tier": "High"}])
    assert overview.new_critical(now, before) == ["B"]


def test_what_changed_lists_new_critical_rising_and_new_places():
    now = _beats([{"Beat": "A", "Priority Tier": "Critical"}])
    escalating = _beats([
        {"Beat": "Slow", "Priority Tier": "Watch", "Recent Conflicts": 3, "Prior Conflicts": 2},
        {"Beat": "Fast", "Priority Tier": "Watch", "Recent Conflicts": 9, "Prior Conflicts": 1},
    ])
    emerging = pd.DataFrame({"Place": ["E1", "E2"], "Pattern": ["New", "Historical"],
                             "Nearest Village": ["Khati (1.3 km)", "X"]})
    changes = overview.what_changed(now, None, escalating, emerging)
    assert [c.label for c in changes] == [
        "Now Critical: A", "Rising: Fast (9 vs 1)", "Rising: Slow (3 vs 2)",
        "New hotspot near Khati (1.3 km)"]
    assert changes[0].target == "A"


def test_villages_to_warn_counts_critical_and_high():
    villages = pd.DataFrame({"Village": list("abcd"),
                             "Tier": ["Critical", "High", "Watch", "Routine"]})
    assert overview.villages_to_warn(villages) == 2
    assert overview.villages_to_warn(None) == 0
