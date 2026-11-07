"""The Range & corridors page with a fitted model behind it.

AppTest cannot upload a register, and the replica's nine reports are too
few to fit, so the page is driven directly with a synthetic register in
the context it reads.
"""

from streamlit.testing.v1 import AppTest

TIMEOUT = 180


def _page():
    from types import SimpleNamespace

    from app_pages import range as page
    from core.map_engine import DEFAULT_BASEMAP
    from presence_fixtures import simulate, village_points

    villages = village_points()
    df = simulate(days=730, seed=4, villages=villages, per_day=5, seasonal_shift_km=18,
                  hotspots=((-20.0, -9.0, 3.0, 1.0), (20.0, 0.0, 4.0, 0.6)))
    page.render(SimpleNamespace(
        df=df, period_end=df["Date"].max().normalize(), villages=villages, registry=None,
        roster=None, basemap=DEFAULT_BASEMAP, layer_boundaries=False, boundary_level=None))


def _run():
    return AppTest.from_function(_page, default_timeout=TIMEOUT).run()


def _texts(at):
    return " ".join(m.value for m in at.markdown)


def test_the_page_opens_on_the_tested_forecast():
    at = _run()
    assert not at.exception, at.exception
    record = " ".join([_texts(at)] + [w.value for w in at.warning])
    assert "Tested on" in record and "past fortnights" in record
    assert "Next fortnight (" in _texts(at)
    assert at.dataframe, "the villages to watch are listed"
    assert any("held" in m.value and "of records in tests" in m.value for m in at.markdown)


def test_every_view_draws():
    at = _run()
    for view in ("Where they range", "Corridors"):
        at.segmented_control(key="range_view").set_value(view).run()
        assert not at.exception, at.exception
    at.segmented_control(key="range_season").set_value("Monsoon (Jun-Sep)").run()
    assert not at.exception, at.exception


def test_the_range_view_says_what_the_correction_did():
    at = _run()
    at.segmented_control(key="range_view").set_value("Where they range").run()
    assert any("half of the landscape farthest from villages" in c.value for c in at.caption)


def test_turning_off_village_effort_refits_without_it():
    at = _run()
    at.checkbox(key="range_village_effort").uncheck().run()
    assert not at.exception, at.exception
    assert "Corrected for: nothing yet." in _texts(at)
