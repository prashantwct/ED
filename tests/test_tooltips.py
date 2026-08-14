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

from core.map_engine import _card, _clean, _count, _pct, _trail

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
    card = _card(
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
    assert "Killed" not in card and "Injured" not in card and "House" not in card
    assert "Crop" in card
    assert card.count('class="mt-r"') == 3


def test_a_village_that_lost_someone_still_says_so():
    card = _card("Tirchuli", (179, 38, 30), [("Killed", _count(2))])
    assert "Killed" in card and ">2<" in card


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


def test_card_values_from_the_csv_are_escaped():
    card = _card("<script>x</script>", (0, 0, 0), [("<b>k</b>", "<i>v</i>")])
    assert "<script>" not in card and "<b>k</b>" not in card
    assert "&lt;script&gt;" in card
