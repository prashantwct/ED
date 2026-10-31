"""The sign-in on the landing page, driven through the app itself.

These run ``app.py`` under Streamlit's AppTest against the local replica,
because the wiring is where this feature can break in ways neither the
client's tests nor the bridge's would catch: a form that collects but
never submits, a fetch that succeeds into a session key nothing reads, a
password left sitting in session state once the page has moved on.

The form asks for an email and a password and nothing else. The client
signs in the way the site's own login page does, so there is no endpoint
to supply and no cookie to paste, and a field offering either would be a
field that does nothing.
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

import core.fetch
from replica import EMAIL, PASSWORD, TOTAL_ROWS

# Relative paths resolve against this file, which is not where the app
# lives.
APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120

FIELDS = {"email": "Email", "password": "Password"}


@pytest.fixture
def app(site, monkeypatch):
    """The app, with its fetch pointed at the replica."""
    real = core.fetch.fetch_sightings

    def against_replica(**kwargs):
        kwargs.setdefault("base_url", site)
        kwargs["delay"] = 0
        return real(**kwargs)

    monkeypatch.setattr(core.fetch, "fetch_sightings", against_replica)
    return AppTest.from_file(APP, default_timeout=TIMEOUT)


def _field(at, key):
    """Address the form by label rather than by index: an index that
    silently means a different field is worse than no test."""
    return next(w for w in at.text_input if w.label == FIELDS[key])


def _sign_in(at, email=EMAIL, password=PASSWORD):
    _field(at, "email").set_value(email)
    _field(at, "password").set_value(password)
    at.button[0].click().run()
    return at


def test_the_landing_page_offers_a_sign_in_beside_the_uploader(app):
    at = app.run()
    assert not at.exception
    assert [w.label for w in at.text_input] == list(FIELDS.values())
    assert at.button[0].label == "Fetch the register"


def test_the_password_field_is_masked(app):
    at = app.run()
    assert _field(at, "password").proto.type == 1   # a credential
    assert _field(at, "email").proto.type == 0


def test_the_page_offers_no_field_the_fetch_would_ignore(app):
    """The cookie and endpoint boxes went when the client stopped being
    able to use them. A form that collects what nothing reads teaches a
    reader that the fetch needs setting up, and it does not."""
    at = app.run()
    labels = " ".join(w.label.lower() for w in at.text_input)
    for gone in ("cookie", "endpoint", "token"):
        assert gone not in labels


def test_the_form_asks_for_dates_from_the_register_start_to_today(app):
    """The export is asked for a window: by default the whole register,
    1 October 2025 through the viewer's today."""
    from core.landing import REGISTER_START, local_today

    at = app.run()
    dates = {w.label: w.value for w in at.date_input}
    assert dates == {"From": REGISTER_START, "To": local_today()}
    markdown = " ".join(m.value for m in at.markdown)
    assert "Export button" in markdown


def test_signing_in_loads_the_dashboard(app):
    at = _sign_in(app.run())
    assert not at.exception
    assert any(f"Loaded {TOTAL_ROWS:,} valid rows" in s.value for s in at.success), \
        [s.value for s in at.success]


def test_the_loaded_page_says_where_the_data_came_from(app):
    at = _sign_in(app.run())
    markdown = " ".join(m.value for m in at.markdown)
    assert "Forest Alerts" in markdown
    assert f"{TOTAL_ROWS:,} rows from the export" in markdown


def test_the_loaded_page_reports_the_span_that_actually_arrived(app):
    at = _sign_in(app.run())
    markdown = " ".join(m.value for m in at.markdown)
    assert "2025-10-01 to 2025-10-09" in markdown


def test_the_password_does_not_outlive_the_fetch(app):
    """A credential that survives into session state survives into
    everything keyed off it. The form clears on submit; this is the test
    that says so after a real round trip."""
    at = _sign_in(app.run())
    leftovers = [
        key for key, value in at.session_state.filtered_state.items()
        if isinstance(value, str) and value and value in (PASSWORD, EMAIL)
    ]
    assert not leftovers, leftovers


def test_an_email_without_a_password_is_refused_without_a_request(app):
    at = _sign_in(app.run(), password="")
    assert not at.exception
    assert any("password" in w.value for w in at.warning), [w.value for w in at.warning]
    assert not at.success


def test_the_refusal_points_at_no_field_the_form_lacks(app):
    at = _sign_in(app.run(), password="")
    said = " ".join(w.value.lower() for w in at.warning)
    assert "cookie" not in said and "endpoint" not in said


def test_a_rejected_sign_in_shows_the_reason_and_stays_on_the_landing_page(app):
    at = _sign_in(app.run(), password="wrong")
    assert not at.exception
    assert any("rejected" in e.value.lower() for e in at.error), \
        [e.value for e in at.error]
    assert at.button[0].label == "Fetch the register"


def test_an_unreachable_site_is_reported_on_the_page_not_raised(site, monkeypatch):
    """The dashboard must survive the network being down."""
    real = core.fetch.fetch_sightings
    monkeypatch.setattr(
        core.fetch, "fetch_sightings",
        lambda **kw: real(**{**kw, "base_url": "http://127.0.0.1:1", "delay": 0}),
    )
    at = _sign_in(AppTest.from_file(APP, default_timeout=TIMEOUT).run())
    assert not at.exception
    assert any("could not reach" in e.value.lower() for e in at.error), \
        [e.value for e in at.error]
    assert at.button[0].label == "Fetch the register"


def test_clearing_the_fetch_returns_to_the_landing_page(app):
    at = _sign_in(app.run())
    clear = next(b for b in at.button if b.label == "Clear")
    clear.click().run()
    assert not at.exception
    assert not at.success
    assert at.button[0].label == "Fetch the register"
