"""Tests for the day-refuge plate: the key, and what it drives.

The view borrows the field-atlas layout -- a map sheet with a register
panel beside it -- where the key is not a legend but the controls. That
only works if three things hold.

Switching a mark off has to take it off the map *and* out of everything
downstream, or the plate and the file a reader downloads from it
disagree. A raid line has to end at a refuge that is still shown, since
a line from nowhere is worse than no line. And the legend has to list
what is actually drawn, not the whole palette: a key offering four
confidence levels on a map showing one teaches the reader something
untrue about the data.
"""

import pandas as pd
import pytest

from core import refuge_view as rv
from core import refuges as rf
from core.charts import raid_start_hours, raids_by_month


def _refuges(*rows):
    """(name, confidence, group, days since last use)."""
    return pd.DataFrame([
        {
            "Refuge": name, "Confidence": conf, "Group": group,
            "Days Since Last Use": since, "Latitude": 23.1 + i * 0.01,
            "Longitude": 81.7 + i * 0.01, "Beat": "B", "Raids Head": "SW",
            "Villages Within Reach": "Kusumhai, Koda",
        }
        for i, (name, conf, group, since) in enumerate(rows)
    ])


def _links(*rows):
    """(refuge, nights ago, category)."""
    end = pd.Timestamp("2026-03-01")
    return pd.DataFrame([
        {
            "Refuge": refuge, "Night Of": end - pd.Timedelta(days=ago),
            "Category": category, "Hour": 19, "Latitude": 23.2, "Longitude": 81.8,
            "Distance (km)": 2.0, "Link": rf.SAME_NIGHT,
        }
        for refuge, ago, category in rows
    ])


ALL_CONF = [rf.CONFIRMED, rf.PROBABLE, rf.POSSIBLE]


def _pick(**over):
    base = {
        "mode": rv.BY_CONFIDENCE, "confidence": ALL_CONF,
        "groups": list(rv.GROUP_COLORS), "within": None, "period": None,
        "links": True, "areas": True,
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
def test_each_mode_colours_by_the_thing_it_names():
    row = {"Confidence": rf.CONFIRMED, "Group": "Lone bull", "Days Since Last Use": 2}
    assert rv.refuge_color(row, rv.BY_GROUP) == rv.GROUP_COLORS["Lone bull"]
    assert rv.refuge_color(row, rv.BY_RECENCY) == rv.RECENCY_BANDS[0][2]
    assert rv.refuge_color(row, rv.BY_CONFIDENCE) != rv.GROUP_COLORS["Lone bull"]


@pytest.mark.parametrize(
    "days, label",
    [(0, "Last 7 days"), (7, "Last 7 days"), (8, "8-30 days"), (30, "8-30 days"),
     (31, "31-60 days"), (60, "31-60 days"), (61, "Over 60 days")],
)
def test_the_recency_bands_meet_at_their_edges(days, label):
    assert rv.recency_band(days)[0] == label


def test_a_refuge_with_no_known_last_use_is_not_called_fresh():
    assert rv.recency_band(None)[0] == "Over 60 days"
    assert rv.recency_band(float("nan"))[0] == "Over 60 days"


def test_an_unknown_group_still_gets_a_colour():
    colour = rv.refuge_color({"Group": "Something new"}, rv.BY_GROUP)
    assert colour == rv.GROUP_COLORS["Unrecorded"]


def test_the_recency_ramp_runs_one_way():
    """Dark is recent. An ordinal question gets an ordinal ramp."""
    limits = [limit for limit, _, _ in rv.RECENCY_BANDS]
    assert limits == sorted(limits)


# ---------------------------------------------------------------------------
# The legend lists what is drawn
# ---------------------------------------------------------------------------
def test_the_legend_names_only_what_is_on_the_map():
    table = _refuges(("R01", rf.PROBABLE, "Lone bull", 3))
    assert rv.legend_entries(table, rv.BY_CONFIDENCE) == [
        (rf.PROBABLE, rv.CONFIDENCE_COLORS[rf.PROBABLE])
    ]
    assert rv.legend_entries(table, rv.BY_GROUP) == [
        ("Lone bull", rv.GROUP_COLORS["Lone bull"])
    ]
    assert rv.legend_entries(table, rv.BY_RECENCY) == [
        ("Last 7 days", rv.RECENCY_BANDS[0][2])
    ]


def test_an_empty_map_gets_an_empty_legend():
    empty = _refuges()
    for mode in rv.COLOUR_MODES:
        assert rv.legend_entries(empty, mode) == []


def test_the_legend_keeps_the_palette_order_not_the_data_order():
    table = _refuges(("R01", rf.POSSIBLE, "Family herd", 90),
                     ("R02", rf.CONFIRMED, "Lone bull", 1))
    names = [n for n, _ in rv.legend_entries(table, rv.BY_CONFIDENCE)]
    assert names == [rf.CONFIRMED, rf.POSSIBLE]


# ---------------------------------------------------------------------------
# The key drives the map
# ---------------------------------------------------------------------------
def test_switching_a_confidence_off_removes_those_refuges():
    table = _refuges(("R01", rf.CONFIRMED, "Lone bull", 1),
                     ("R02", rf.POSSIBLE, "Family herd", 1))
    shown, _, _ = rv._apply(table, _links(), pd.DataFrame(),
                            _pick(confidence=[rf.CONFIRMED]))
    assert shown["Refuge"].tolist() == ["R01"]


def test_switching_a_group_off_removes_those_refuges():
    table = _refuges(("R01", rf.CONFIRMED, "Lone bull", 1),
                     ("R02", rf.CONFIRMED, "Family herd", 1))
    shown, _, _ = rv._apply(table, _links(), pd.DataFrame(),
                            _pick(groups=["Family herd"]))
    assert shown["Refuge"].tolist() == ["R02"]


def test_last_used_within_keeps_only_refuges_that_recent():
    table = _refuges(("R01", rf.CONFIRMED, "Lone bull", 3),
                     ("R02", rf.CONFIRMED, "Lone bull", 45))
    shown, _, _ = rv._apply(table, _links(), pd.DataFrame(), _pick(within=7))
    assert shown["Refuge"].tolist() == ["R01"]


def test_a_raid_line_never_ends_at_a_refuge_that_is_hidden():
    """A line from nowhere is worse than no line."""
    table = _refuges(("R01", rf.CONFIRMED, "Lone bull", 1),
                     ("R02", rf.POSSIBLE, "Lone bull", 1))
    links = _links(("R01", 1, "Crop"), ("R02", 1, "House"))
    shown, shown_links, _ = rv._apply(table, links, pd.DataFrame(),
                                      _pick(confidence=[rf.CONFIRMED]))
    assert set(shown_links["Refuge"]) <= set(shown["Refuge"])
    assert shown_links["Refuge"].tolist() == ["R01"]


def test_the_raid_period_cuts_older_nights():
    table = _refuges(("R01", rf.CONFIRMED, "Lone bull", 1))
    links = _links(("R01", 5, "Crop"), ("R01", 200, "Crop"))
    _, shown_links, _ = rv._apply(table, links, pd.DataFrame(), _pick(period=30))
    assert len(shown_links) == 1


def test_turning_the_lines_off_leaves_the_refuges_alone():
    table = _refuges(("R01", rf.CONFIRMED, "Lone bull", 1))
    links = _links(("R01", 1, "Crop"))
    shown, shown_links, _ = rv._apply(table, links, pd.DataFrame(), _pick(links=False))
    assert len(shown) == 1 and shown_links.empty


def test_turning_raid_areas_off_hides_them():
    areas = pd.DataFrame([{"Area": "A01", "Latitude": 23.0, "Longitude": 81.0}])
    _, _, shown_areas = rv._apply(_refuges(), _links(), areas, _pick(areas=False))
    assert shown_areas.empty
    _, _, kept = rv._apply(_refuges(), _links(), areas, _pick(areas=True))
    assert len(kept) == 1


def test_every_entry_point_survives_empty_frames():
    empty = pd.DataFrame()
    shown, links, areas = rv._apply(_refuges(), empty, empty, _pick())
    assert shown.empty and links.empty and areas.empty


def test_filtering_everything_out_is_allowed_and_says_nothing_false():
    table = _refuges(("R01", rf.CONFIRMED, "Lone bull", 90))
    links = _links(("R01", 1, "Crop"))
    shown, shown_links, _ = rv._apply(table, links, pd.DataFrame(), _pick(within=7))
    assert shown.empty and shown_links.empty
    assert rv.legend_entries(shown, rv.BY_CONFIDENCE) == []


# ---------------------------------------------------------------------------
# The panel charts
# ---------------------------------------------------------------------------
def test_the_hour_chart_splits_night_from_day():
    links = _links(("R01", 1, "Crop"))
    fig = raid_start_hours(links, 18, 6)
    assert {trace.name for trace in fig.data} == {"Day", "Night"}
    night = next(t for t in fig.data if t.name == "Night")
    assert sum(night.y) == 1  # the 19:00 raid


def test_the_hour_chart_marks_the_peak():
    links = _links(*[("R01", i, "Crop") for i in range(4)])
    fig = raid_start_hours(links, 18, 6)
    assert any("Peak 19:00" in str(a.text) for a in fig.layout.annotations)


def test_the_month_chart_stacks_by_what_the_raid_cost():
    links = _links(("R01", 1, "Crop"), ("R01", 40, "House"))
    fig = raids_by_month(links)
    assert fig.layout.barmode == "stack"
    assert {t.name for t in fig.data} == {"Crop or grain loss", "House / store damage"}


def test_the_month_chart_shows_a_month_with_no_raids_as_a_gap():
    """Dropping the empty month would compress the year and hide a lull."""
    links = _links(("R01", 0, "Crop"), ("R01", 70, "Crop"))
    fig = raids_by_month(links)
    assert len(fig.data[0].x) >= 3


def test_both_charts_render_with_nothing_to_show():
    for fig in (raid_start_hours(pd.DataFrame(), 18, 6), raids_by_month(pd.DataFrame())):
        assert fig is not None


def test_the_hour_chart_ignores_an_hour_outside_the_clock():
    links = _links(("R01", 1, "Crop"))
    links.loc[0, "Hour"] = 99
    fig = raid_start_hours(links, 18, 6)
    assert sum(sum(t.y) for t in fig.data) == 0
