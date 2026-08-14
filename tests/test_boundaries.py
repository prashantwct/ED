"""Tests for forest boundaries and map tooltips.

Two things here are load-bearing. Names in the shapefiles carry a
reserve-zone suffix the sighting export does not, so any join has to
normalise it away. And tooltips are written into the page with
innerHTML, so every value taken from an uploaded CSV has to be escaped
on the way there -- a beat name is field-entered text.
"""

import pandas as pd
import pytest

from core import boundaries
from core.map_engine import _boundary_tooltip, _card, _clean, _view_bounds


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------
def test_reserve_zone_suffixes_are_stripped():
    """The export says "Kallwah"; the shapefile says "Kallwah Core"."""
    assert boundaries.normalise("Kallwah Core") == "kallwah"
    assert boundaries.normalise("Manpur Buffer") == "manpur"
    assert boundaries.normalise("Panpatha Core Zone") == "panpatha"


def test_normalisation_leaves_ordinary_names_alone():
    assert boundaries.normalise("Jaithari") == "jaithari"
    assert boundaries.normalise("  Rajendra   Gram ") == "rajendra gram"


def test_a_name_ending_in_core_as_a_word_is_not_mangled():
    """Only a trailing zone word goes; "Corehat" keeps its letters."""
    assert boundaries.normalise("Corehat") == "corehat"


def test_normalisation_survives_blanks():
    assert boundaries.normalise(None) == ""
    assert boundaries.normalise(float("nan")) == "nan"


# ---------------------------------------------------------------------------
# Level selection
# ---------------------------------------------------------------------------
def test_zoom_picks_a_level_that_can_actually_be_read():
    assert boundaries.level_for_zoom(6.0) == boundaries.DIVISION
    assert boundaries.level_for_zoom(9.5) == boundaries.RANGE
    assert boundaries.level_for_zoom(13.0) == boundaries.BEAT


def test_level_thresholds_are_ordered():
    """A wider view never draws a finer level than a tighter one."""
    seen = [boundaries.level_for_zoom(z) for z in (5, 7, 9, 10, 11, 13, 15)]
    order = {boundaries.DIVISION: 0, boundaries.RANGE: 1, boundaries.BEAT: 2}
    ranks = [order[s] for s in seen]
    assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# View filtering
# ---------------------------------------------------------------------------
def _feature(west, south, east, north, name="X"):
    return {
        "type": "Feature",
        "properties": {"name": name, "_key": name.lower()},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north],
                             [west, north], [west, south]]],
        },
    }


def test_only_features_overlapping_the_view_are_kept():
    near = _feature(81.0, 23.0, 81.2, 23.2, "Near")
    far = _feature(90.0, 30.0, 90.2, 30.2, "Far")
    kept = boundaries.in_view([near, far], 80.9, 22.9, 81.5, 23.5)
    assert [f["properties"]["name"] for f in kept] == ["Near"]


def test_a_feature_straddling_the_edge_is_kept():
    """Clipping at the frame edge would leave a boundary stopping dead."""
    straddling = _feature(80.5, 22.5, 81.1, 23.1, "Edge")
    assert boundaries.in_view([straddling], 81.0, 23.0, 81.5, 23.5)


def test_annotate_does_not_mutate_the_cached_layer():
    """The cache is shared across reruns with different filters."""
    feature = _feature(81.0, 23.0, 81.2, 23.2, "Cholna")
    out = boundaries.annotate([feature], {"cholna": {"Conflict Events": 45}})
    assert out[0]["properties"]["Conflict Events"] == 45
    assert "Conflict Events" not in feature["properties"]


def test_stats_join_uses_the_normalised_name():
    table = pd.DataFrame([{"Beat": "Kallwah", "Conflict Events": 12}])
    stats = boundaries.stats_by_name(table, "Beat", ["Conflict Events"])
    assert stats["kallwah"]["Conflict Events"] == 12


def test_stats_join_tolerates_an_empty_table():
    assert boundaries.stats_by_name(pd.DataFrame(), "Beat", ["x"]) == {}
    assert boundaries.stats_by_name(None, "Beat", ["x"]) == {}


# ---------------------------------------------------------------------------
# The vendored files
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not boundaries.available(), reason="boundaries not vendored")
def test_every_level_loads_with_named_features():
    for level in boundaries.LEVELS + (boundaries.RESERVE,):
        features = boundaries.load(level)["features"]
        assert features, f"{level} is empty"
        assert all(f["properties"].get("name") for f in features)


@pytest.mark.skipif(not boundaries.available(), reason="boundaries not vendored")
def test_administrative_levels_are_areas():
    for level in boundaries.LEVELS:
        assert all(f["geometry"]["type"] in ("Polygon", "MultiPolygon")
                   for f in boundaries.load(level)["features"]), level


@pytest.mark.skipif(not boundaries.available(), reason="boundaries not vendored")
def test_a_reserve_may_be_a_line_rather_than_an_area():
    """Sanjay TR ships its boundary as a LineString.

    An outline layer draws it either way, so the build does not force it
    into a polygon it was never digitised as.
    """
    kinds = {f["geometry"]["type"]
             for f in boundaries.load(boundaries.RESERVE)["features"]}
    assert kinds <= {"Polygon", "MultiPolygon", "LineString", "MultiLineString"}


@pytest.mark.skipif(not boundaries.available(), reason="boundaries not vendored")
def test_boundaries_are_in_lon_lat_over_the_landscape():
    """A projection slip would put central India in the ocean."""
    for feature in boundaries.load(boundaries.DIVISION)["features"]:
        west, south, east, north = boundaries._bounds(
            feature["geometry"]["coordinates"]
        )
        assert 78 < west < 85 and 78 < east < 85
        assert 20 < south < 27 and 20 < north < 27


# ---------------------------------------------------------------------------
# Tooltips
# ---------------------------------------------------------------------------
def test_tooltip_escapes_names_from_the_uploaded_csv():
    """deck.gl writes the tooltip with innerHTML."""
    html = _card("<img src=x onerror=alert(1)>", (0, 0, 0),
                 [("Beat", "<script>alert(2)</script>")],
                 subtitle="<b>x</b>", footer="<i>y</i>")
    for tag in ("<img", "<script", "<b>x", "<i>y"):
        assert tag not in html
    assert "&lt;script&gt;" in html


def test_tooltip_keeps_its_own_markup():
    html = _card("Crop damage", (200, 100, 50), [("Killed", 2)])
    assert html.startswith("<div")
    assert "rgb(200,100,50)" in html
    assert "Killed" in html and ">2<" in html


def test_tooltip_drops_rows_with_nothing_in_them():
    html = _card("T", (0, 0, 0), [("Shown", 1), ("Hidden", None), ("Blank", "")])
    assert "Shown" in html
    assert "Hidden" not in html and "Blank" not in html


def test_blank_values_render_as_a_dash_not_the_word_nan():
    assert _clean(float("nan")) == "—"
    assert _clean(None) == "—"
    assert _clean("Unknown") == "—"
    assert _clean("Cholna") == "Cholna"


def test_boundary_tooltip_names_the_level_and_carries_stats():
    html = _boundary_tooltip(boundaries.BEAT, {
        "name": "Cholna", "parent": "Jaithari", "grandparent": "Anuppur",
        "Reports": 87, "Conflict Events": 45, "Human Deaths": 1,
        "Priority Tier": "High",
    })
    for expected in ("Cholna", "Beat", "Jaithari", "Anuppur", "87", "45", "High"):
        assert expected in html


def test_boundary_tooltip_works_with_no_statistics():
    html = _boundary_tooltip(boundaries.DIVISION, {"name": "Anuppur"})
    assert "Anuppur" in html and "Division" in html


# ---------------------------------------------------------------------------
# Viewport maths
# ---------------------------------------------------------------------------
def test_view_bounds_widen_as_the_view_zooms_out():
    import pydeck as pdk

    tight = _view_bounds(pdk.ViewState(latitude=23.0, longitude=81.0, zoom=13))
    wide = _view_bounds(pdk.ViewState(latitude=23.0, longitude=81.0, zoom=7))
    assert (wide[2] - wide[0]) > (tight[2] - tight[0])


def test_view_bounds_are_centred_on_the_view():
    import pydeck as pdk

    west, south, east, north = _view_bounds(
        pdk.ViewState(latitude=23.0, longitude=81.0, zoom=10)
    )
    assert (west + east) / 2 == pytest.approx(81.0)
    assert (south + north) / 2 == pytest.approx(23.0)


# ---------------------------------------------------------------------------
# A failing map must not take the page with it
# ---------------------------------------------------------------------------
def test_a_map_failure_is_contained_to_its_own_section(monkeypatch):
    """The beat table and the brief are what the manager came for."""
    import streamlit as st

    from core import map_engine

    shown = []
    monkeypatch.setattr(st, "error", lambda msg: shown.append(msg))

    @map_engine._never_breaks_the_page
    def boom():
        raise TypeError("unexpected keyword argument")

    assert boom() is None
    assert shown and "TypeError" in shown[0]


def test_the_guard_returns_the_map_when_nothing_goes_wrong():
    from core import map_engine

    @map_engine._never_breaks_the_page
    def fine(value):
        return value * 2

    assert fine(21) == 42
    assert fine.__name__ == "fine"


# ---------------------------------------------------------------------------
# Payload size
# ---------------------------------------------------------------------------
def test_the_deck_spec_is_not_pretty_printed():
    """pydeck serialises with indent=2 and Streamlit sends that string.

    On this landscape the whitespace was two thirds of the payload, and
    an oversized spec is what stopped the maps loading once already.
    """
    import pydeck as pdk

    from core.map_engine import _compact

    deck = pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer",
                          data=[{"lon": 81.0, "lat": 23.0} for _ in range(200)])],
        initial_view_state=pdk.ViewState(latitude=23.0, longitude=81.0, zoom=8),
    )
    pretty = len(deck.to_json())
    compact = len(_compact(deck).to_json())
    assert compact < pretty * 0.6
    assert "\n" not in _compact(deck).to_json()


def test_compacting_preserves_the_spec():
    import json

    import pydeck as pdk

    from core.map_engine import _compact

    deck = pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", data=[{"lon": 81.0, "lat": 23.0}])],
        initial_view_state=pdk.ViewState(latitude=23.0, longitude=81.0, zoom=8),
    )
    before = json.loads(deck.to_json())
    assert json.loads(_compact(deck).to_json()) == before


def test_only_reserves_carry_a_casing_layer():
    """The casing doubles the geometry payload, which 1,274 beats cannot
    afford; a dozen reserve outlines can."""
    from core.map_engine import _outline_layers

    feature = _feature(81.0, 23.0, 81.2, 23.2, "X")
    admin = _outline_layers(boundaries.BEAT, boundaries.annotate([feature]))
    reserve = _outline_layers(boundaries.RESERVE, boundaries.annotate([feature]))
    assert len(admin) == 1
    assert len(reserve) == 2
