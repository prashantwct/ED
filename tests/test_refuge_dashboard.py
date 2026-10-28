"""Tests for the embedded day-refuge dashboard.

The plate is a vendored design with its data lifted out. Filling it back
in is mostly a pass-through, because the sheet reads ``core.refuges``'
own column names -- so what these guard is the handful of places where
the two sides can quietly stop agreeing.

The one that matters most is the script block. The register is an
uploaded CSV, so a village or beat name is attacker-controllable text,
and it ends up inside a ``<script>`` in a frame that has same-origin
access to the app. ``json.dumps`` does not escape ``<``.

After that: every placeholder has to be filled, or the reader sees
``__ED_DATA__`` on the page; a raid line has to start at its own refuge;
dates have to arrive in the shape the sheet's own parser slices; and
``DIV_LABEL`` has to name every division present, because the sheet
filters its map labels through it and a division missing from it is a
division with no label.
"""

import json
import re

import pandas as pd
import pytest

from core import refuge_dashboard as rd


def _result(refuges=None, areas=None, links=None):
    return {
        "refuges": refuges if refuges is not None else pd.DataFrame(),
        "areas": areas if areas is not None else pd.DataFrame(),
        "links": links if links is not None else pd.DataFrame(),
    }


def _refuges(*rows):
    """(name, division, last used, lat, lon)."""
    return pd.DataFrame([
        {
            "Refuge": name, "Division": division, "Range": "Rng", "Beat": "Bt",
            "Last Used": pd.Timestamp(last), "First Used": pd.Timestamp("2025-10-01"),
            "Days Since Last Use": 3, "Days Used": 5, "Confidence": "Confirmed",
            "Group": "Lone bull", "Linked Raids": 2, "Radius (km)": 1.1,
            "Cover Share %": float("nan"), "Recommended Action": "Dusk watch.",
            "Latitude": lat, "Longitude": lon,
        }
        for name, division, last, lat, lon in rows
    ])


def _areas(*rows):
    return pd.DataFrame([
        {"Area": name, "Division": division, "Raids": 4, "Nights": 3,
         "First Raid": pd.Timestamp("2026-01-01"),
         "Last Raid": pd.Timestamp(last), "Latitude": lat, "Longitude": lon}
        for name, division, last, lat, lon in rows
    ])


def _links(*rows):
    """(refuge, lat, lon)."""
    return pd.DataFrame([
        {"Refuge": refuge, "Date": pd.Timestamp("2026-01-05"),
         "Night Of": pd.Timestamp("2026-01-04"), "Hour": 19, "Category": "Crop",
         "Distance (km)": 2.5, "Latitude": lat, "Longitude": lon}
        for refuge, lat, lon in rows
    ])


def _data_of(page):
    return json.loads(re.search(r"^const DATA = (\{.*\});$", page, re.M).group(1))


def _kind(page, kind):
    return [f for f in _data_of(page)["features"] if f["properties"]["kind"] == kind]


# ---------------------------------------------------------------------------
# The script block
# ---------------------------------------------------------------------------
def test_a_village_name_cannot_close_the_script_block():
    """The frame has same-origin access to the app, and the register is
    an uploaded file: this is the path from a CSV cell to running code."""
    hostile = "</script><script>alert(1)</script>"
    page = rd.build(_result(_refuges(("R01", hostile, "2026-01-28", 23.1, 81.7))))
    script = page[page.index("const DATA = "):page.index("const CTX = ")]
    assert "</script>" not in script
    assert "<script>" not in script


def test_the_escaping_does_not_change_the_value_a_reader_sees():
    hostile = "</script> & <b>Amdari</b>"
    encoded = rd._script_json({"Village": hostile})
    assert json.loads(encoded)["Village"] == hostile
    for char in "<>&":
        assert char not in encoded


def test_the_line_separators_json_allows_but_javascript_does_not_are_escaped():
    """U+2028 is legal in a JSON string and a line break in a script."""
    encoded = rd._script_json({"x": "a b c"})
    assert " " not in encoded and " " not in encoded
    assert json.loads(encoded)["x"] == "a b c"


def test_the_division_labels_are_escaped_the_same_way():
    """They are JSON in a script block too, built from the same CSV."""
    page = rd.build(_result(_refuges(("R01", "</script>x", "2026-01-28", 23.1, 81.7))))
    labels = re.search(r"^const DIV_LABEL = (.*);$", page, re.M).group(1)
    assert "</script>" not in labels
    assert "</script>x" in json.loads(labels)


def test_a_division_name_cannot_inject_markup_into_the_header():
    page = rd.build(_result(_refuges(("R01", "<img src=x onerror=alert(1)>",
                                      "2026-01-28", 23.1, 81.7))))
    head = page[:page.index("const DATA = ")]
    assert "<img src=x" not in head
    assert "&lt;img" in head


# ---------------------------------------------------------------------------
# Filling the plate
# ---------------------------------------------------------------------------
def test_the_plate_is_vendored_and_carries_every_placeholder():
    assert rd.available()
    template = rd.TEMPLATE_PATH.read_text(encoding="utf-8")
    for token in rd.TOKENS:
        assert template.count(token) == 1, token


def test_no_placeholder_survives_into_the_page():
    """A leftover token is the reader seeing __ED_DATA__ on the plate."""
    page = rd.build(_result(_refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7))))
    for token in rd.TOKENS:
        assert token not in page


def test_every_row_becomes_a_feature_of_the_right_kind():
    result = _result(
        _refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7),
                 ("R02", "Umaria", "2026-01-20", 23.2, 81.8)),
        _areas(("A01", "Anuppur", "2026-01-25", 23.3, 81.9)),
        _links(("R01", 23.15, 81.75), ("R02", 23.25, 81.85)),
    )
    page = rd.build(result)
    assert len(_kind(page, "refuge")) == 2
    assert len(_kind(page, "raid area")) == 1
    assert len(_kind(page, "linked raid")) == 2


def test_a_raid_line_starts_at_its_own_refuge():
    result = _result(_refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7)),
                     links=_links(("R01", 23.15, 81.75)))
    line = _kind(rd.build(result), "linked raid")[0]
    assert line["geometry"]["type"] == "LineString"
    assert line["geometry"]["coordinates"] == [[81.7, 23.1], [81.75, 23.15]]


def test_a_raid_whose_refuge_is_not_drawn_is_dropped():
    """A line from nowhere is worse than no line."""
    result = _result(_refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7)),
                     links=_links(("R99", 23.15, 81.75)))
    assert _kind(rd.build(result), "linked raid") == []


def test_dates_arrive_in_the_shape_the_sheet_slices():
    """Its parser does s.slice(0, 10) + 'T00:00:00Z'."""
    page = rd.build(_result(_refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7))))
    last = _kind(page, "refuge")[0]["properties"]["Last Used"]
    assert last[:10] == "2026-01-28"
    assert pd.Timestamp(last[:10]) == pd.Timestamp("2026-01-28")


def test_a_missing_measurement_arrives_as_null_not_nan():
    """NaN is not JSON, and the sheet already draws a blank as a dash."""
    page = rd.build(_result(_refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7))))
    assert _kind(page, "refuge")[0]["properties"]["Cover Share %"] is None
    assert "NaN" not in page[page.index("const DATA = "):page.index("const CTX = ")]


def test_the_geometry_columns_do_not_also_ride_along_as_properties():
    page = rd.build(_result(_refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7))))
    properties = _kind(page, "refuge")[0]["properties"]
    assert "Latitude" not in properties and "Longitude" not in properties
    assert _kind(page, "refuge")[0]["geometry"]["coordinates"] == [81.7, 23.1]


def test_a_row_with_no_position_is_left_off_rather_than_placed_at_zero():
    frame = _refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7))
    frame.loc[0, "Latitude"] = float("nan")
    assert _kind(rd.build(_result(frame)), "refuge") == []


# ---------------------------------------------------------------------------
# The header follows the data
# ---------------------------------------------------------------------------
def test_recency_is_measured_from_the_register_not_the_clock():
    """A file that ends in March must not open with everything stale."""
    page = rd.build(_result(_refuges(("R01", "Anuppur", "2026-03-21", 23.1, 81.7))))
    assert "Date.UTC(2026, 2, 21)" in page


def test_the_anchor_can_be_pinned():
    page = rd.build(_result(_refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7))),
                    today=pd.Timestamp("2026-06-05"))
    assert "Date.UTC(2026, 5, 5)" in page


def test_a_raid_area_later_than_any_refuge_still_moves_the_anchor():
    result = _result(_refuges(("R01", "Anuppur", "2026-01-10", 23.1, 81.7)),
                     _areas(("A01", "Anuppur", "2026-02-02", 23.3, 81.9)))
    assert "Date.UTC(2026, 1, 2)" in rd.build(result)


def test_every_division_present_gets_a_label():
    """The sheet filters its map labels through DIV_LABEL, so one missing
    from it is a division drawn with no name against it."""
    result = _result(_refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7)),
                     _areas(("A01", "Umaria", "2026-01-25", 23.3, 81.9)))
    page = rd.build(result)
    labels = json.loads(re.search(r"^const DIV_LABEL = (.*);$", page, re.M).group(1))
    assert set(labels) == {"Anuppur", "Umaria"}


def test_the_go_to_buttons_follow_the_divisions_in_the_data():
    result = _result(_refuges(("R01", "Bandhavgarh National Park",
                               "2026-01-28", 23.1, 81.7)))
    page = rd.build(result)
    jump = re.search(r'id="jumpSeg".*?</div>', page, re.S).group(0)
    assert re.findall(r'data-v="([^"]*)"', jump) == ["all", "Bandhavgarh National Park"]
    # trimmed for a crowded row, but the value it filters on is the full name
    assert ">Bandhavgarh<" in jump


def test_the_standfirst_names_what_is_covered():
    result = _result(_refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7),
                              ("R02", "Umaria", "2026-01-20", 23.2, 81.8)))
    sub = re.search(r'id="sub">([^<]*)<', rd.build(result)).group(1)
    assert "Anuppur and Umaria" in sub


def test_three_divisions_read_as_a_sentence():
    assert rd._sentence(["A", "B", "C"]) == "A, B and C"
    assert rd._sentence(["A"]) == "A"
    assert rd._sentence([]) == ""


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------
def test_an_empty_analysis_still_produces_a_page():
    page = rd.build(_result())
    assert page.lstrip().startswith("<!doctype html")
    assert _data_of(page)["features"] == []
    for token in rd.TOKENS:
        assert token not in page


def test_a_result_missing_a_frame_entirely_does_not_raise():
    page = rd.build({"refuges": _refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7))})
    assert len(_kind(page, "refuge")) == 1


def test_the_page_is_one_self_contained_file():
    """It is handed to a range office as a download; it must not need
    anything of ours alongside it."""
    page = rd.build(_result(_refuges(("R01", "Anuppur", "2026-01-28", 23.1, 81.7))))
    assert "const CTX = " in page          # its own context geography
    assert "const DATA = " in page         # its own register
    assert 'src="/' not in page            # nothing served from the app
