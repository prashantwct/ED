"""Tests for the pre-upload screen.

The one that matters is the partner mark. An official emblem must never
be approximated, so a partner with no supplied logo file has to degrade
to typography rather than to a placeholder box or a broken image.
"""

import base64

import pytest

from core import landing


def test_a_partner_without_a_logo_file_gets_a_wordmark(monkeypatch, tmp_path):
    """No file, no invented emblem."""
    monkeypatch.setattr(landing, "ASSET_DIR", tmp_path)
    marks = landing._partner_marks()
    assert "<img" not in marks
    assert "MPFD" in marks and "Madhya Pradesh Forest Department" in marks
    assert "Wildlife Conservation Trust" in marks


def test_a_supplied_logo_is_embedded(monkeypatch, tmp_path):
    (tmp_path / "logo-wct.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    monkeypatch.setattr(landing, "ASSET_DIR", tmp_path)
    marks = landing._partner_marks()
    assert 'src="data:image/png;base64,' in marks
    # The partner with no file still gets typography, not a gap.
    assert "MPFD" in marks


def test_logos_carry_alt_text(monkeypatch, tmp_path):
    (tmp_path / "logo-wct.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    monkeypatch.setattr(landing, "ASSET_DIR", tmp_path)
    assert 'alt="Wildlife Conservation Trust"' in landing._partner_marks()


def test_a_missing_asset_returns_nothing_rather_than_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(landing, "ASSET_DIR", tmp_path)
    assert landing._data_uri("nope.jpg") is None


def test_jpeg_and_png_get_the_right_mime(monkeypatch, tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8ff")
    (tmp_path / "b.png").write_bytes(b"\x89PNG")
    monkeypatch.setattr(landing, "ASSET_DIR", tmp_path)
    assert landing._data_uri("a.jpg").startswith("data:image/jpeg;base64,")
    assert landing._data_uri("b.png").startswith("data:image/png;base64,")


def test_the_data_uri_round_trips(monkeypatch, tmp_path):
    payload = b"\x89PNG\r\n\x1a\n" + bytes(range(200))
    (tmp_path / "x.png").write_bytes(payload)
    monkeypatch.setattr(landing, "ASSET_DIR", tmp_path)
    encoded = landing._data_uri("x.png").split(",", 1)[1]
    assert base64.b64decode(encoded) == payload


def test_partner_names_are_escaped(monkeypatch, tmp_path):
    monkeypatch.setattr(landing, "ASSET_DIR", tmp_path)
    monkeypatch.setattr(
        landing, "PARTNERS", (("<script>x</script>", "<b>S</b>", "none.png"),)
    )
    marks = landing._partner_marks()
    assert "<script>" not in marks and "<b>S</b>" not in marks
    assert "&lt;script&gt;" in marks


def test_capability_cards_cover_every_entry():
    cards = landing._capability_cards()
    for _, title, _ in landing.CAPABILITIES:
        assert title in cards
    assert cards.count('class="lp-cap"') == len(landing.CAPABILITIES)


def test_capability_icons_exist_in_the_icon_set():
    """A missing name silently falls back to the generic glyph."""
    from core.ui import _ICON_PATHS

    for name, _, _ in landing.CAPABILITIES:
        assert name in _ICON_PATHS, name


def test_required_columns_match_what_the_loader_demands():
    from core.data_loader import REQUIRED_COLUMNS as loader_required

    assert set(landing.REQUIRED_COLUMNS) == set(loader_required)


def test_the_hero_is_described_for_a_screen_reader():
    """It is a CSS background, so it needs a label to exist at all."""
    assert len(landing.HERO_ALT) > 30
    assert "elephant" in landing.HERO_ALT.lower()


def test_the_access_warning_survives_in_the_copy():
    assert "no access control" in landing.ACCESS_NOTE
