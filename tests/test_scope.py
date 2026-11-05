"""The scope: a focus window to count and draw, a history to analyse."""

from datetime import date

import pandas as pd
import pytest

from core import scope
from core.analytics import compute_severity, window_comparison
from core.intelligence import management_brief

START, END = date(2024, 10, 1), date(2026, 10, 31)


def test_presets_end_on_the_registers_last_report():
    w = scope.focus_window(scope.LAST_90, START, END)
    assert w.end == END and w.days == 90


def test_a_preset_longer_than_the_register_is_the_register():
    w = scope.focus_window(scope.LAST_12M, date(2026, 9, 1), END)
    assert w.start == date(2026, 9, 1)


def test_whole_register_is_every_day():
    assert scope.focus_window(scope.WHOLE, START, END) == scope.Window(START, END)


def test_a_custom_range_is_clipped_to_the_register():
    w = scope.focus_window(scope.CUSTOM, START, END, (date(2020, 1, 1), date(2025, 1, 31)))
    assert w == scope.Window(START, date(2025, 1, 31))


def test_an_unusable_custom_range_falls_back_to_the_default():
    w = scope.focus_window(scope.CUSTOM, START, END, (date(2030, 1, 1), date(2030, 2, 1)))
    assert w.days == scope.PERIODS[scope.DEFAULT_PERIOD]


def _register():
    rows = []
    for i in range(0, 760, 4):  # two years of reports, every fourth day
        day = pd.Timestamp(START) + pd.Timedelta(days=i)
        rows.append({"Date": day, "Division": "D", "Range": "R", "Beat": f"B{i % 3}",
                     "Latitude": 23.0 + (i % 7) * 0.01, "Longitude": 81.0, "Hour": 21,
                     "Total Count": 1, "Crop Damage": int(i % 8 == 0), "Grain Damage": 0,
                     "House Damage": 0, "Injury": 0, "Death": int(i == 600),
                     "Male Death Count": int(i == 600), "Female Death Count": 0,
                     "Children Death Count": 0})
    df = pd.DataFrame(rows)
    df["Severity Score"] = compute_severity(df)
    return df


def test_the_focus_window_does_not_change_tiers():
    """The point of the split: a 90-day view ranks beats exactly as the
    whole register does, as of the same day."""
    df = _register()
    end = df["Date"].max().date()
    whole = scope.history(df, scope.focus_window(scope.WHOLE, START, end))
    short = scope.history(df, scope.focus_window(scope.LAST_90, START, end))
    tiers = [management_brief(h, 90, as_of=pd.Timestamp(end))["beats"]
             .set_index("Beat")["Priority Tier"].sort_index() for h in (whole, short)]
    pd.testing.assert_series_equal(*tiers)


def test_the_focus_is_the_windows_rows_with_the_historys_index():
    df = _register()
    w = scope.focus_window(scope.LAST_30, START, df["Date"].max().date())
    h = scope.history(df, w)
    f = scope.focus(h, w)
    assert f["Date"].min() >= pd.Timestamp(w.start)
    assert set(f.index) <= set(h.index)


def test_the_history_ends_with_the_window():
    df = _register()
    w = scope.focus_window(scope.CUSTOM, START, END, (date(2025, 1, 1), date(2025, 3, 31)))
    h = scope.history(df, w)
    assert h["Date"].max() <= pd.Timestamp(w.end) + pd.Timedelta(days=1)
    assert h["Date"].min() == df["Date"].min()


def test_a_short_window_compares_itself_with_a_year_earlier():
    df = _register()
    end = df["Date"].max().date()
    w = scope.focus_window(scope.LAST_90, START, end)
    days = scope.comparison_days(w, START, recent_days=60)
    assert days == 90
    change = window_comparison(scope.history(df, w), days, as_of=pd.Timestamp(end))
    assert change["conflicts_year"] == change["conflicts_year"]  # not NaN


def test_the_whole_register_compares_the_recent_window():
    w = scope.focus_window(scope.WHOLE, START, END)
    assert scope.comparison_days(w, START, recent_days=60) == 60


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------
TYPES = ["All reports", "Conflict only", "Human harm"]


def test_the_default_scope_makes_an_empty_query():
    assert scope.to_query([], [], [], scope.DEFAULT_PERIOD, None, TYPES[0], TYPES) == {}


def test_a_scope_survives_the_round_trip():
    query = scope.to_query(["Anuppur"], ["Kotma"], [], scope.CUSTOM,
                           (date(2025, 1, 1), date(2025, 3, 31)), TYPES[2], TYPES)
    params = {k: (v if isinstance(v, list) else [v]) for k, v in query.items()}
    back = scope.from_query(params, ["Anuppur", "Umaria"], ["Kotma"], ["B1"], TYPES)
    assert back == {"divisions": ["Anuppur"], "ranges": ["Kotma"], "beats": [],
                    "period": scope.CUSTOM, "custom": (date(2025, 1, 1), date(2025, 3, 31)),
                    "incident": TYPES[2]}


@pytest.mark.parametrize("params", [
    {"div": ["Nowhere"]},
    {"period": ["fortnight"]},
    {"period": ["custom"], "from": ["not a date"]},
    {"type": ["99"]},
    {"type": ["x"]},
])
def test_a_link_from_another_register_opens_on_what_still_makes_sense(params):
    back = scope.from_query(params, ["Anuppur"], [], [], TYPES)
    assert back["divisions"] == []
    assert back["period"] in scope.PERIODS
    assert back["incident"] in TYPES


def test_the_area_label_names_the_narrowest_level():
    assert scope.area_label([], [], []) == "All divisions"
    assert scope.area_label(["Anuppur"], [], []) == "Anuppur"
    assert scope.area_label(["Anuppur"], ["Kotma", "Jaithari"], []) == "2 ranges"
    assert scope.area_label(["Anuppur"], ["Kotma"], ["Jamudi"]) == "Jamudi"
