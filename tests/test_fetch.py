"""Tests for the app's Forest Alerts fetch, driven against the replica.

The bridge has two jobs beyond calling the scraper, and both are here:
it has to hand back something ``core.data_loader`` accepts, so the fetch
and upload paths cannot diverge, and it has to turn every failure into a
sentence a forest officer can act on rather than a traceback.

The third job is a rule rather than a behaviour -- credentials are used
once and kept nowhere -- and the tests that hold it assert on what comes
back out, because a password that reaches the result is a password that
reaches a cache.
"""

import pytest

from core.data_loader import load_and_validate_csv
from core.fetch import FetchError, FetchResult, fetch_sightings, window
from core.landing import FetchRequest
from replica import EMAIL, PASSWORD, SESSION_COOKIE, TOTAL_ROWS


def _fetch(site, **overrides):
    kwargs = dict(email=EMAIL, password=PASSWORD, base_url=site, delay=0)
    kwargs.update(overrides)
    return fetch_sightings(**kwargs)


def test_a_fetch_returns_a_csv_the_dashboard_loads(site, tmp_path):
    """The assertion that keeps the two entry paths honest: a fetch and
    an upload have to arrive at the same loader in the same shape."""
    result = _fetch(site)
    assert isinstance(result, FetchResult)
    assert result.rows == TOTAL_ROWS

    path = tmp_path / result.name
    path.write_bytes(result.data)
    frame, warnings = load_and_validate_csv(str(path))
    assert len(frame) == TOTAL_ROWS
    assert not warnings, warnings
    assert frame["Hour"].eq(19).all()


def test_the_window_is_reported_with_the_result(site):
    result = _fetch(site, end_date="2026-09-12")
    assert (result.start_date, result.end_date) == ("2025-10-01", "2026-09-12")
    assert result.name == "forestalerts-2025-10-01-to-2026-09-12.csv"
    assert result.pages == 3


def test_progress_is_reported_page_by_page(site):
    seen = []
    _fetch(site, progress=lambda page, rows: seen.append((page, rows)))
    assert seen == [(1, 4), (2, 8), (3, 9)]


def test_a_rejected_sign_in_is_a_sentence_not_a_traceback(site):
    with pytest.raises(FetchError) as caught:
        _fetch(site, password="wrong")
    message = str(caught.value)
    assert "rejected" in message
    assert "Traceback" not in message


def test_a_cookie_works_without_a_password(site):
    result = _fetch(site, email="", password="", cookie=SESSION_COOKIE)
    assert result.rows == TOTAL_ROWS


def test_no_credentials_at_all_is_refused_with_advice(site):
    with pytest.raises(FetchError, match="No credentials"):
        _fetch(site, email="", password="")


def test_an_unreachable_site_fails_with_a_message(monkeypatch):
    """A refused connection has to read as a failed fetch, not crash the
    page it was drawn on. The backoff is cut to nothing here: the retry
    itself has its own coverage, and waiting 14 seconds for it in this
    test buys nothing."""
    from tools import scrape_sightings

    monkeypatch.setattr(scrape_sightings, "RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(scrape_sightings, "RETRY_BACKOFF", [])
    with pytest.raises(FetchError) as caught:
        fetch_sightings(email=EMAIL, password=PASSWORD, base_url="http://127.0.0.1:1")
    assert "127.0.0.1:1" in str(caught.value)


def test_the_credentials_are_nowhere_in_the_result(site):
    """A password that survives into the result is a password that
    survives into session state and the load cache behind it."""
    result = _fetch(site)
    haystack = repr(result.__dict__).lower() + result.data.decode("utf-8").lower()
    assert PASSWORD.lower() not in haystack
    assert EMAIL.lower() not in haystack


def test_the_advertised_window_comes_from_the_scraper_itself(site):
    """The sentence on the landing page and the dates the fetch sends have
    to be one fact, or the screen ends up describing a pull nobody made."""
    from tools.scrape_sightings import START_DATE, today

    assert window() == (START_DATE, today())
    assert window("2026-01-31") == (START_DATE, "2026-01-31")

    advertised_start, advertised_end = window()
    result = _fetch(site)
    assert (result.start_date, result.end_date) == (advertised_start, advertised_end)


# --- the form's own validation -------------------------------------------


def test_email_without_a_password_is_refused_before_any_request():
    refusal = FetchRequest(email="a@b.c", password="", cookie="").is_usable()
    assert refusal and "password" in refusal


def test_a_cookie_alone_is_enough():
    assert FetchRequest(email="", password="", cookie="x=1").is_usable() is None


def test_a_complete_sign_in_is_accepted():
    assert FetchRequest(email="a@b.c", password="p", cookie="").is_usable() is None


def test_an_endpoint_without_credentials_is_still_refused():
    assert FetchRequest(email="", password="", cookie="",
                        api_path="/api/sightings").is_usable()


def test_an_empty_form_is_refused():
    assert FetchRequest(email="", password="", cookie="").is_usable()
