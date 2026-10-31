"""The brief's findings, action plan and ecology, and the brief they make.

The brief used to repeat each beat's recommendation down every table the
beat appeared in, wrote its counts as "fatality(ies)", told a beat of
unsexed raids to give a breeding herd passage, and said nothing about the
elephants themselves. These pin what replaced that.
"""

import re

import numpy as np
import pandas as pd
import pytest

from core import brief_insights as bi
from core import report_charts
from core.analytics import window_comparison
from core.intelligence import beat_intelligence, management_brief
from core.report import generate_html_report

BASE = {
    "Division": "D", "Range": "R", "Latitude": 23.0, "Longitude": 81.0,
    "Male Count": 0, "Female Count": 0, "Calf Count": 0, "Unknown Count": 0,
    "Total Count": 0, "Crop Damage": 0, "Grain Damage": 0, "House Damage": 0,
    "Injury": 0, "Death": 0, "Sighting Type": "direct",
}


def _row(day, beat="Bholgarh", **extra):
    return {**BASE, "Date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
            "Beat": beat, "Hour": 19, **extra}


def _bull(day, beat="Bholgarh", **extra):
    return _row(day, beat, **{"Male Count": 1, "Total Count": 1, **extra})


def _herd(day, beat="Goraiya", **extra):
    return _row(day, beat, **{"Female Count": 6, "Calf Count": 3, "Total Count": 10, **extra})


def _register():
    """A bull beat raiding and rising, a herd beat passing, a quiet beat."""
    rows = []
    for day in range(0, 300, 3):
        rows.append(_bull(day, **{"Crop Damage": int(day % 6 == 0 or day > 200)}))
        rows.append(_herd(day + 1, **{"Crop Damage": int(day % 30 == 0)}))
        rows.append(_row(day + 2, "Quiet", **{"Male Count": 1, "Total Count": 1,
                                              "Hour": 7}))
    rows.append(_bull(290, **{"Death": 1, "Male Death Count": 1, "Crop Damage": 1}))
    df = pd.DataFrame(rows)
    df["Is_Night"] = df["Hour"].between(18, 23)
    return df


def _brief(df, days=90):
    end = df["Date"].max()
    return management_brief(df, recent_days=days, as_of=end), \
        window_comparison(df, days, as_of=end)


# --- wording ----------------------------------------------------------------


@pytest.mark.parametrize("count, one, many, expected", [
    (1, "person", "people", "1 person"),
    (6, "person", "people", "6 people"),
    (0, "incident", None, "0 incidents"),
    (1234, "report", None, "1,234 reports"),
])
def test_plural_agrees_with_its_count(count, one, many, expected):
    assert bi.plural(count, one, many) == expected


def test_names_join_like_a_sentence_and_count_the_rest():
    assert bi.names(["A"]) == "A"
    assert bi.names(["A", "B"]) == "A and B"
    assert bi.names(["A", "B", "C", "A", ""]) == "A, B and C"
    assert bi.names(list("ABCDEFGH"), limit=3) == "A, B, C and 5 more"


def test_harm_leaves_out_a_zero():
    assert bi.harm(2, 0) == "2 people killed"
    assert bi.harm(0, 3) == "3 injured"
    assert bi.harm(1, 1) == "1 person killed and 1 injured"


# --- the bull share --------------------------------------------------------


def test_unsexed_raids_do_not_make_a_beat_herd_driven():
    """The bug behind 'give the herd passage' on bull beats: conflict with
    no group recorded was counted as not-bulls."""
    rows = [_row(d, **{"Crop Damage": 1, "Sighting Type": "damage"}) for d in range(12)]
    rows += [_bull(d + 20, **{"Crop Damage": 1}) for d in range(4)]
    table = beat_intelligence(pd.DataFrame(rows))
    row = table.iloc[0]
    assert row["Bull-Type Conflict %"] == 100.0
    assert row["Group-Known Conflicts"] == 4
    assert "do not drive the herd" not in row["Recommended Action"]


def test_too_few_recorded_groups_make_no_claim_either_way():
    rows = [_row(d, **{"Crop Damage": 1}) for d in range(12)] + [_bull(30, **{"Crop Damage": 1})]
    action = beat_intelligence(pd.DataFrame(rows))["Recommended Action"].iloc[0]
    assert "Bull-driven" not in action and "do not drive" not in action


# --- findings ---------------------------------------------------------------


def test_findings_carry_their_numbers_in_plain_words():
    df = _register()
    brief, change = _brief(df)
    lines = bi.key_findings(df, brief, change, 90)
    text = " ".join(lines)
    assert lines[0].startswith("1 person killed in 1 incident")
    assert "Conflict is up" in text and "Bholgarh" in text
    assert "Crop loss is" in text
    assert "came from lone bulls" not in text  # the old headline's phrasing
    assert "Lone bulls and bull parties cause" in text
    assert not re.search(r"\(s\)|\(ies\)| -- ", text)


def test_a_quiet_period_says_so():
    df = pd.DataFrame([_row(d) for d in range(10)])
    brief, change = _brief(df)
    assert bi.key_findings(df, brief, change, 90)[0].startswith("No one was killed")


# --- action plan -------------------------------------------------------------


def test_each_action_appears_once_naming_every_beat_it_covers():
    df = _register()
    brief, _ = _brief(df)
    plan = bi.action_plan(brief, 90)
    kinds = [item.kind for item in plan]
    # Only "warn" may recur: new hotspots and exposed villages are two lists.
    assert all(kinds.count(kind) == 1 for kind in set(kinds) - {"warn"})
    bulls = next(item for item in plan if item.kind == "bulls")
    assert "Bholgarh" in bulls.where and "Goraiya" not in bulls.where
    assert bulls.when == "18:00-06:00"
    respond = next(item for item in plan if item.kind == "respond")
    assert respond.where == "Bholgarh" and "1 person killed" in respond.why


def test_herd_beats_are_told_to_give_passage_not_to_track_bulls():
    rows = [_herd(d, **{"Crop Damage": 1}) for d in range(0, 120, 2)]
    rows += [_herd(d, beat="Other") for d in range(1, 120, 2)]
    df = pd.DataFrame(rows)
    brief, _ = _brief(df)
    assert brief["beats"].set_index("Beat").loc["Goraiya", "Priority Tier"] == "High"
    plan = {item.kind: item for item in bi.action_plan(brief, 90)}
    assert plan["herds"].where == "Goraiya"
    assert "do not drive" in plan["herds"].action
    assert "bulls" not in plan


def test_village_counts_are_quoted_not_summed():
    """Neighbouring villages share incidents; a total would double count."""
    df = _register()
    brief, _ = _brief(df)
    risk = pd.DataFrame({"Village": ["Choi", "Koda", "Pipra"], "Tier": ["Critical", "High", "Watch"],
                         "Conflict Events": [40, 38, 3], "Human Deaths": [1, 1, 0]})
    warn = next(item for item in bi.action_plan(brief, 90, village_risk=risk)
                if item.action.startswith("Early-warning"))
    assert warn.where == "Choi and Koda"
    assert "Choi 40, Koda 38" in warn.why and "78" not in warn.why


def test_refuges_become_dusk_watch_and_search_actions():
    df = _register()
    brief, _ = _brief(df)
    refuges = {
        "refuges": pd.DataFrame({"Refuge": ["R01", "R02"], "Beat": ["Mahainwah", "Dhangawa"],
                                 "Linked Raids": [47, 45],
                                 "Villages Raided": ["Raipur, Bahannara", "Choi"]}),
        "areas": pd.DataFrame({"Area": ["A01"], "Beat": ["Cholna"], "Raids": [23]}),
    }
    plan = {item.kind: item for item in bi.action_plan(brief, 90, refuges=refuges)}
    assert plan["refuge"].where == "R01 Mahainwah and R02 Dhangawa"
    assert "92 raids" in plan["refuge"].why and "Raipur" in plan["refuge"].why
    assert plan["search"].where == "A01 Cholna"
    assert any("forest day refuges send night raids" in line
               for line in bi.key_findings(df, brief, {}, 90, refuges=refuges))


def test_beat_driver_says_who_what_and_which_way():
    row = {"Bull-Type Conflict %": 82.0, "Group-Known Conflicts": 9, "Crop Damage Events": 6,
           "House Damage Events": 1, "Recent Deaths": 0, "Recent Injuries": 0,
           "Human Deaths": 1, "Trend": "Escalating", "Recent Conflicts": 8, "Prior Conflicts": 3}
    assert bi.beat_driver(row) == "Bulls 82%, crops, past death, up, 8 vs 3"
    assert bi.beat_driver({"Crop Damage Events": 0, "House Damage Events": 0}) == "Presence only"


# --- ecology ----------------------------------------------------------------


def test_ecology_reads_direct_sightings_not_signs_or_damage():
    df = _register()
    df.loc[df.index[:5], "Sighting Type"] = "dung"
    hours = bi.diel_activity(df)
    assert len(hours) == 24 and hours.sum() < len(df)
    sizes = bi.herd_sizes(df)
    assert list(sizes.index) == ["1", "2-3", "4-9", "10-19", "20+"]
    assert sizes["10-19"] > 0 and sizes["1"] > 0
    notes = " ".join(bi.ecology_notes(df))
    assert "breeding herds" in notes and "median group" in notes


def test_range_use_does_not_call_the_first_month_new():
    df = _register()
    used = bi.range_by_month(df)
    assert used["First used"].iloc[0] == 0
    assert (used["Beats used"] >= used["First used"]).all()


def test_presence_grid_and_damage_months_have_their_shape():
    df = _register()
    grid = bi.presence_grid(df)
    assert list(grid.index) == ["D"] and grid.shape[1] >= 9
    months = bi.damage_by_month(df)
    assert list(months.index[:3]) == ["Jan", "Feb", "Mar"] and set(months) == {"Crop", "House"}


def test_peak_hours_wrap_round_midnight():
    hours = pd.Series(0, index=range(24))
    hours[23], hours[0] = 5, 5
    assert bi.peak_hours(hours) == "23:00-01:00"


@pytest.mark.parametrize("build", [
    lambda df: report_charts.diel_clock_svg(bi.diel_activity(df), 18, 6),
    lambda df: report_charts.range_use_svg(bi.range_by_month(df)),
    lambda df: report_charts.damage_season_svg(bi.damage_by_month(df)),
    lambda df: report_charts.herd_composition_svg(_brief(df)[0]["composition"]),
    lambda df: report_charts.seasonal_heatmap_svg(bi.presence_grid(df), left=120,
                                                  steps=report_charts.PRESENCE_STEPS),
])
def test_ecology_charts_are_well_formed_svg(build):
    svg = build(_register())
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("<svg") == 1 and "nan" not in svg.lower()


# --- the brief ----------------------------------------------------------------


def test_the_brief_leads_with_findings_and_the_plan():
    df = _register()
    html = generate_html_report(df, df["Date"].min(), df["Date"].max())
    order = [html.index(h) for h in ("Key Findings", "Action Plan", "Priority Beats",
                                     "Daily and Seasonal Rhythm", "The Elephants", "Notes")]
    assert order == sorted(order)
    for gone in ("Severity Distribution", "Night vs. Day", "fatality(ies)", "beat(s)",
                 "carry tusks", "Recommended action"):
        assert gone not in html
    assert not re.search(r"\b\w+\((s|ies)\)", html)


def test_the_plan_names_a_beat_once_per_action_not_once_per_table():
    df = _register()
    html = generate_html_report(df, df["Date"].min(), df["Date"].max())
    assert html.count("identify the animal") == 0
    assert html.count("Identify the raiding bulls") == 1
