"""How the loaded page is laid out, driven through the app itself.

The register is fetched from the local replica (the same route as
``test_app_fetch``), because AppTest cannot drive a file upload.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import core.fetch
from core import analysis_settings as cfg
from replica import EMAIL, PASSWORD

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120


@pytest.fixture
def loaded(site, monkeypatch):
    """The app with a register fetched from the replica."""
    real = core.fetch.fetch_sightings

    def against_replica(**kwargs):
        kwargs.setdefault("base_url", site)
        kwargs["delay"] = 0
        return real(**kwargs)

    monkeypatch.setattr(core.fetch, "fetch_sightings", against_replica)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    fields = {w.label: w for w in at.text_input}
    fields["Email"].set_value(EMAIL)
    fields["Password"].set_value(PASSWORD)
    at.button[0].click().run()
    assert not at.exception
    return at


def _badges(at):
    """st.badge is a markdown element whose body is ``:color-badge[...]``."""
    return [m.value.split("]", 1)[0].split(" ", 1)[-1] for m in at.markdown
            if "-badge[" in m.value]


def test_where_to_act_is_the_tab_that_opens(loaded):
    assert loaded.tabs[0].label.endswith("Where to act")


def test_standard_settings_carry_no_badge(loaded):
    assert not [b for b in _badges(loaded) if b.startswith("Custom analysis")]


def test_a_changed_setting_is_named_at_the_top_of_the_page(loaded):
    loaded.slider(key=cfg.EPS_KM.key).set_value(1.5).run()
    assert not loaded.exception
    assert any("Hotspot merge distance 1.5 km (standard 1)" in b for b in _badges(loaded))
    # AppTest files an expander that has an icon under status containers.
    sections = list(loaded.expander) + list(loaded.status)
    assert any(e.label == "Advanced analysis (changed)" for e in sections)


def test_reset_to_standard_puts_every_setting_back(loaded):
    loaded.slider(key=cfg.EPS_KM.key).set_value(1.5).run()
    loaded.slider(key=cfg.RECENT_DAYS.key).set_value(60).run()
    next(b for b in loaded.button if b.label == "Reset to standard").click().run()
    assert not loaded.exception
    assert loaded.slider(key=cfg.EPS_KM.key).value == cfg.EPS_KM.default
    assert loaded.slider(key=cfg.RECENT_DAYS.key).value == cfg.RECENT_DAYS.default
    assert not [b for b in _badges(loaded) if b.startswith("Custom analysis")]


def test_no_developer_note_sits_under_the_maps(loaded):
    """The MapTiler note lives on the basemap selector, once."""
    captions = " ".join(c.value for c in loaded.caption)
    assert "MAPTILER_KEY" not in captions
