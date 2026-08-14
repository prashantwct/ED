"""Tests for what a hover actually says.

A tooltip is read in about a second, standing at a desk, often on a
laptop in a range office. Two things kept breaking that.

The first is length. Cards listed every metric whether or not it had
happened, so a village that lost a crop and nothing else still showed
five rows of zeros to get there.

The second is punctuation. Em dashes and middots were used as separators
throughout. They are fine in a document and wrong in a hover: they read
as stray marks at 11px, and the only non-ASCII characters that earn
their place here are the tier glyphs, which exist precisely so the
ranking survives greyscale and colour vision deficiency.
"""

import ast
import pathlib

import pandas as pd
import pytest

from core.map_engine import (
    _OFF,
    _TOOLTIP_HTML,
    _TOOLTIP_ROWS,
    TOOLTIP_FIELDS,
    _clean,
    _count,
    _pct,
    _slots,
    _trail,
)

REPO = pathlib.Path(__file__).resolve().parent.parent

# Everything else must be ASCII. These are the exceptions and why.
ALLOWED_NON_ASCII = {
    "▲": "Critical tier glyph -- colour is never the only signal",
    "◆": "High tier glyph",
    "●": "Watch tier glyph",
    "○": "Routine tier glyph",
    "©": "basemap attribution, required by the tile licence",
    "\U0001f418": "browser tab icon; in-page icons are SVG",
}

# A help bubble is a hint, not documentation. Past this it is a paragraph
# nobody reads, and the explanation belongs in the section prose.
MAX_HELP_CHARS = 110

SOURCE_FILES = [REPO / "app.py"] + sorted((REPO / "core").glob("*.py"))


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_stray_punctuation_reaches_the_screen(path):
    offenders = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        for char in line:
            if ord(char) > 127 and char not in ALLOWED_NON_ASCII:
                offenders.setdefault(char, []).append(number)
    assert not offenders, (
        f"{path.name} carries non-ASCII with no stated reason: "
        + ", ".join(f"{c!r} on line(s) {v[:3]}" for c, v in offenders.items())
    )


def test_help_bubbles_stay_hints():
    tree = ast.parse((REPO / "app.py").read_text())
    too_long = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords or []:
            if keyword.arg != "help":
                continue
            try:
                value = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                continue  # an f-string; its length is checked by eye
            if isinstance(value, str) and len(value) > MAX_HELP_CHARS:
                too_long.append((node.lineno, len(value), value[:60]))
    assert not too_long, f"help text past {MAX_HELP_CHARS} chars: {too_long}"


# ---------------------------------------------------------------------------
# Card content
# ---------------------------------------------------------------------------
def test_a_zero_count_drops_its_row_entirely():
    """Absence says the same thing as "0" and takes no space saying it."""
    assert _count(0) is None
    assert _count(0.0) is None
    assert _count(None) is None
    assert _count(float("nan")) is None
    assert _count(3) == "3"
    assert _count(1200) == "1,200"


def test_a_village_that_lost_nobody_shows_no_casualty_rows():
    slots = _slots(
        "Kua",
        (107, 133, 120),
        [
            ("Conflicts", "1"),
            ("Killed", _count(0)),
            ("Injured", _count(0)),
            ("House", _count(0)),
            ("Crop", _count(1)),
            ("Night", _pct(0.0)),
        ],
        subtitle="Routine tier",
    )
    filled = [slots[f"_l{i}"] for i in range(1, _TOOLTIP_ROWS + 1) if not slots[f"_h{i}"]]
    assert filled == ["Conflicts", "Crop", "Night"]
    # The unused rows are switched off rather than left blank on screen.
    assert all(slots[f"_h{i}"] == _OFF for i in (4, 5))


def test_a_village_that_lost_someone_still_says_so():
    slots = _slots("Tirchuli", (179, 38, 30), [("Killed", _count(2))])
    assert slots["_l1"] == "Killed" and slots["_v1"] == "2"


def test_a_night_share_of_zero_is_kept_because_it_was_measured():
    """Unlike a casualty count, 0% night is a finding, not an absence."""
    assert _pct(0.0) == "0%"
    assert _pct(None) is None


def test_separators_are_plain_ascii():
    assert _trail("Beat", "Range", "Division") == "Beat / Range / Division"
    assert _clean(None) == "-"
    for text in (_trail("A", "B"), _clean(None), _clean(float("nan"))):
        text.encode("ascii")  # raises if not


def test_the_breadcrumb_skips_what_is_missing_rather_than_padding_it():
    assert _trail("Beat", None, "Division") == "Beat / Division"
    assert _trail(None, None) == ""


# ---------------------------------------------------------------------------
# The template contract
# ---------------------------------------------------------------------------
def test_values_are_left_unescaped_for_streamlit_to_escape():
    """Escaping twice renders "&amp;amp;" in a village name.

    Streamlit HTML-escapes each interpolated value on the way into the
    template, so this side must hand over the raw text.
    """
    slots = _slots("Bhamhani & Co", (0, 0, 0), [("Beat", "<Ram>")])
    assert slots["_t"] == "Bhamhani & Co"
    assert slots["_v1"] == "<Ram>"


def test_every_field_the_template_names_is_one_a_layer_fills():
    """A field the hovered object lacks is left as "{_v3}" on screen."""
    import re

    named = set(re.findall(r"{(_[a-z0-9]+)}", _TOOLTIP_HTML))
    assert named == set(TOOLTIP_FIELDS)
    assert set(_slots("t", (0, 0, 0), []).keys()) == named


def test_the_accent_colour_survives_the_escaper():
    """It lands in a style attribute, so it must carry no & < > " '."""
    accent = _slots("t", (179, 38, 30), [])["_a"]
    assert accent == "rgb(179,38,30)"
    assert not set(accent) & set("&<>\"'")


def test_a_card_never_exceeds_the_rows_the_template_has():
    slots = _slots("t", (0, 0, 0), [(f"l{i}", i) for i in range(1, 12)])
    assert len([k for k in slots if k.startswith("_l")]) == _TOOLTIP_ROWS
    assert all(not slots[f"_h{i}"] for i in range(1, _TOOLTIP_ROWS + 1))


def test_every_pickable_layer_fills_every_slot():
    """The template is deck-level; the layers are not.

    Streamlit leaves a field the hovered object does not carry exactly as
    it found it, so one layer missing one slot puts a literal "{_v3}" on
    screen for whatever the reader happens to hover. This walks the real
    layers a full render builds and checks each against the template.
    """
    import numpy as np

    from core import map_engine as me

    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Latitude": 23.10 + np.linspace(0, 0.02, 30),
        "Longitude": 81.75 + np.linspace(0, 0.02, 30),
        "Division": "Anuppur", "Range": "Jaithari", "Beat": "Dhangava",
        "Hour": 20, "Total Count": 2,
        "Crop Damage": 0, "Grain Damage": 0, "House Damage": 0,
        "Injury": 0, "Death": 0,
        "Nearest Village": "Takhuli", "Distance to Village (km)": 0.8,
    })
    df["Severity Score"] = 0.5
    df["Is_Night"] = True

    from core.hotspots import detect_hotspots, villages_at_risk

    centroids = pd.DataFrame({
        "Village": ["Takhuli"], "Latitude": [23.11], "Longitude": [81.76],
    })
    hotspots = detect_hotspots(df, eps_km=2.0, min_samples=5)
    villages = villages_at_risk(df, centroids, hotspots)

    layers = (
        [me._sighting_layer(me._prepare_sightings(df))]
        + me._hotspot_layers(hotspots)
        + me._village_layers(villages, emphasise=True)
        + me.boundary_layers(me._adaptive_view_state(df))
    )

    checked = 0
    for layer in layers:
        if not getattr(layer, "pickable", False):
            continue
        records = layer.data
        if isinstance(records, pd.DataFrame):
            records = records.to_dict("records")
        elif isinstance(records, dict):  # a GeoJSON FeatureCollection
            records = records.get("features", [])
        if not records:
            continue
        checked += 1
        first = records[0]
        available = set(first) | set(first.get("properties", {}) or {})
        missing = set(TOOLTIP_FIELDS) - available
        assert not missing, f"{layer.type} would render literal braces for: {sorted(missing)}"

    assert checked >= 3, f"only {checked} pickable layers exercised"
