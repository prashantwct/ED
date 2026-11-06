"""How the loaded app is laid out, driven through the app itself.

The register is fetched from the local replica (the same route as
``test_app_fetch``), because AppTest cannot drive a file upload.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import core.fetch
from core import analysis_settings as cfg
from core import scope
from pages import open_page
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


def _texts(at):
    return " ".join(m.value for m in at.markdown)


def test_the_overview_opens_first_with_the_four_numbers(loaded):
    labels = [m.label for m in loaded.metric]
    assert labels[:4] == ["Conflict events", "Killed / injured", "Beats Critical / High",
                          "Villages to warn"]
    assert "Act now" in _texts(loaded)
    assert any(b.label == "Generate brief" for b in loaded.button)


@pytest.mark.parametrize("page", ["where", "next", "villages", "patterns", "data",
                                  "analyst"])
def test_every_page_draws_under_the_same_scope(loaded, page):
    open_page(loaded, page).run()
    assert not loaded.exception, loaded.exception
    assert loaded.segmented_control(key="flt_period").value == scope.DEFAULT_PERIOD


def test_the_period_survives_a_trip_to_another_page(loaded):
    """Streamlit forgets a widget not drawn on a run; the scope bar is drawn
    on every conflict page, and the refuge page is not one of them."""
    loaded.segmented_control(key="flt_period").set_value(scope.WHOLE).run()
    open_page(loaded, "refuges").run()
    assert not loaded.exception, loaded.exception
    open_page(loaded, "where").run()
    assert loaded.segmented_control(key="flt_period").value == scope.WHOLE


def test_the_scope_goes_into_the_link(loaded):
    loaded.segmented_control(key="flt_period").set_value(scope.LAST_30).run()
    assert loaded.query_params.get("period") == ["30d"]


def test_a_link_opens_on_its_scope(site, monkeypatch):
    real = core.fetch.fetch_sightings
    monkeypatch.setattr(core.fetch, "fetch_sightings",
                        lambda **kw: real(**{**kw, "base_url": site, "delay": 0}))
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["period"] = "all"
    at.run()
    fields = {w.label: w for w in at.text_input}
    fields["Email"].set_value(EMAIL)
    fields["Password"].set_value(PASSWORD)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert at.segmented_control(key="flt_period").value == scope.WHOLE


def test_standard_settings_carry_no_badge(loaded):
    assert not [b for b in _badges(loaded) if b.startswith("Custom analysis")]


def test_a_setting_changed_on_the_analyst_page_is_named_everywhere(loaded):
    open_page(loaded, "analyst").run()
    loaded.slider(key=cfg.EPS_KM.key).set_value(1.5).run()
    assert not loaded.exception
    open_page(loaded, "where").run()
    assert any("Hotspot merge distance 1.5 km (standard 1)" in b for b in _badges(loaded))


def test_reset_to_standard_puts_every_setting_back(loaded):
    open_page(loaded, "analyst").run()
    loaded.slider(key=cfg.EPS_KM.key).set_value(1.5).run()
    loaded.slider(key=cfg.RECENT_DAYS.key).set_value(60).run()
    next(b for b in loaded.button if b.label == "Reset to standard").click().run()
    assert not loaded.exception
    assert loaded.slider(key=cfg.EPS_KM.key).value == cfg.EPS_KM.default
    assert loaded.slider(key=cfg.RECENT_DAYS.key).value == cfg.RECENT_DAYS.default
    assert not [b for b in _badges(loaded) if b.startswith("Custom analysis")]


def test_no_developer_note_sits_under_the_maps(loaded):
    """The MapTiler note lives on the basemap selector, once."""
    for page in ("overview", "where", "villages"):
        open_page(loaded, page).run()
        assert "MAPTILER_KEY" not in " ".join(c.value for c in loaded.caption)


def test_a_focus_link_opens_the_map_on_that_beat(loaded):
    """The Overview's Map links carry ``focus=beat:<name>``: real links keep
    the uploaded register in the browser, where st.switch_page drops it."""
    loaded.query_params["focus"] = "beat:Beat 01"
    open_page(loaded, "where").run()
    assert not loaded.exception, loaded.exception
    assert any("Showing Beat Beat 01" in i.value for i in loaded.info)
    assert "focus" not in loaded.query_params


def test_the_overview_links_carry_the_scope(loaded):
    loaded.segmented_control(key="flt_period").set_value(scope.LAST_30).run()
    links = loaded.get("page_link")
    assert links, "the Overview should link onward"
    assert all("period=30d" in link.proto.query_string for link in links)
    maps = [link for link in links if link.proto.label == "Map"]
    assert maps and all("focus=beat" in link.proto.query_string for link in maps)


def test_next_moves_says_when_there_is_too_little_history_to_test_it(loaded):
    """Nine reports over nine days cannot test a forecast: the page must say
    so rather than present zones as if they had a track record."""
    open_page(loaded, "next").run()
    assert not loaded.exception, loaded.exception
    assert any("Not enough history to test the forecast" in i.value for i in loaded.info)


def test_the_overview_points_to_next_moves_when_groups_are_on_the_move(loaded):
    links = [link for link in loaded.get("page_link") if link.proto.page == "next"]
    assert links and "On the move" in _texts(loaded)
