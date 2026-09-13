"""The sign-in on the landing page, driven through the app itself.

These run ``app.py`` under Streamlit's AppTest against the local replica,
because the wiring is where this feature can break in ways neither the
scraper's tests nor the bridge's would catch: a form that collects but
never submits, a fetch that succeeds into a session key nothing reads, a
password left sitting in session state once the page has moved on.
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

import core.fetch
from core.fetch import window
from replica import EMAIL, PASSWORD, TOTAL_ROWS

# Relative paths resolve against this file, which is not where the app
# lives.
APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120


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


FIELDS = {
    "email": "Email",
    "password": "Password",
    "cookie": "Session cookie (only if your sign-in needs a one-time code)",
    "api_path": "Rows endpoint",
    "api_login_path": "Sign-in endpoint",
}


def _field(at, key):
    """Address the form by label: it has grown fields, and an index that
    silently means something else is worse than no test."""
    label = FIELDS[key]
    return next(w for w in at.text_input if w.label == label)


def _sign_in(at, email=EMAIL, password=PASSWORD, cookie="",
             api_path="", api_login_path=""):
    for key, value in (
        ("email", email), ("password", password), ("cookie", cookie),
        ("api_path", api_path), ("api_login_path", api_login_path),
    ):
        _field(at, key).set_value(value)
    at.button[0].click().run()
    return at


def test_the_landing_page_offers_a_sign_in_beside_the_uploader(app):
    at = app.run()
    assert not at.exception
    assert [w.label for w in at.text_input] == list(FIELDS.values())
    assert at.button[0].label == "Fetch the register"
    # proto.type 1 is PASSWORD: both secrets are masked on screen, and
    # the cookie is a credential exactly as much as the password is.
    assert _field(at, "password").proto.type == 1
    assert _field(at, "cookie").proto.type == 1
    # The endpoints are configuration, not secrets.
    assert _field(at, "api_path").proto.type == 0


def test_the_landing_page_says_the_site_renders_in_the_browser(app):
    """The first real run failed because the site is a script shell. The
    page has to say so before someone types a password into it."""
    at = app.run()
    markdown = " ".join(m.value for m in at.markdown)
    assert "renders in the browser" in markdown
    assert "API" in markdown


def test_the_rows_endpoint_can_be_given_on_the_page(app):
    at = _sign_in(app.run(), api_path="/api/sightings", api_login_path="/api/login")
    assert not at.exception
    assert any(f"Loaded {TOTAL_ROWS:,} valid rows" in s.value for s in at.success), \
        [s.value for s in at.success]


def test_the_landing_page_states_the_window_it_will_pull(app):
    at = app.run()
    start, end = window()
    markdown = " ".join(m.value for m in at.markdown)
    assert f"from {start} to {end}" in markdown


def test_signing_in_loads_the_dashboard(app):
    at = _sign_in(app.run())
    assert not at.exception
    assert any(f"Loaded {TOTAL_ROWS:,} valid rows" in s.value for s in at.success), \
        [s.value for s in at.success]


def test_the_loaded_page_says_where_the_data_came_from(app):
    at = _sign_in(app.run())
    markdown = " ".join(m.value for m in at.markdown)
    assert "Forest Alerts" in markdown
    assert f"{TOTAL_ROWS:,} rows over 3 page(s)" in markdown


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


def test_a_rejected_sign_in_shows_the_reason_and_stays_on_the_landing_page(app):
    at = _sign_in(app.run(), password="wrong")
    assert not at.exception
    assert any("rejected" in e.value for e in at.error), [e.value for e in at.error]
    assert at.button[0].label == "Fetch the register"


def test_a_login_the_app_cannot_fill_shows_the_page_report(site, monkeypatch):
    """The diagnosis is several lines long. Markdown would run them into
    one another, so the detail has to reach the screen as a block."""
    from replica import APP_SHELL_PAGE, Handler
    import core.fetch

    Handler.login_html = APP_SHELL_PAGE
    real = core.fetch.fetch_sightings
    monkeypatch.setattr(
        core.fetch, "fetch_sightings",
        lambda **kw: real(**{**kw, "base_url": site, "delay": 0}),
    )
    at = _sign_in(AppTest.from_file(APP, default_timeout=TIMEOUT).run())

    assert not at.exception
    assert any("No login form" in e.value for e in at.error), [e.value for e in at.error]
    blocks = " ".join(c.value for c in at.code)
    assert "application shell" in blocks
    assert "--cookie" in blocks


def test_clearing_the_fetch_returns_to_the_landing_page(app):
    at = _sign_in(app.run())
    clear = next(b for b in at.button if b.label == "Clear")
    clear.click().run()
    assert not at.exception
    assert not at.success
    assert at.button[0].label == "Fetch the register"
