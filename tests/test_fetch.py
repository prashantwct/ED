"""Tests for the app's Forest Alerts fetch, driven against the replica.

The bridge has two jobs beyond calling the client, and both are here:
it has to hand back something ``core.data_loader`` accepts, so the fetch
and upload paths cannot diverge, and it has to turn every failure into a
sentence a forest officer can act on rather than a traceback.

The third job is a rule rather than a behaviour -- credentials are used
once and kept nowhere -- and the tests that hold it assert on what comes
back out, because a password that reaches the result is a password that
reaches a cache.

The site is a Laravel + Inertia app, so the replica is driven over that
protocol: a page load carrying the asset version, a login POST checked
against the XSRF cookie, and the listing as JSON page by page. Nothing
here configures an endpoint or pastes a cookie, because the client no
longer has anywhere to put either.
"""

import pytest

from core.data_loader import load_and_validate_csv
from core.fetch import FetchError, FetchResult, fetch_sightings, observed_window
from core.landing import FetchRequest
from replica import EMAIL, PASSWORD, TOTAL_ROWS, Handler


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


def test_the_rows_keep_the_dashboard_column_names(site):
    """The client flattens the API's nested lookups on the way out, so
    nothing downstream has to rename a column."""
    result = _fetch(site)
    for column in ("Date", "Latitude", "Longitude", "Division", "Range", "Beat"):
        assert column in result.columns


def test_progress_is_reported_page_by_page(site):
    seen = []
    _fetch(site, progress=lambda page, rows: seen.append((page, rows)))
    assert seen == [(1, 4), (2, 8), (3, 9)]
    assert Handler.inertia_pages == [1, 2, 3]


def test_the_result_counts_the_pages_it_read(site):
    assert _fetch(site).pages == 3


def test_a_rejected_sign_in_is_a_sentence_not_a_traceback(site):
    with pytest.raises(FetchError) as caught:
        _fetch(site, password="wrong")
    message = str(caught.value)
    assert "rejected" in message.lower()
    assert "Traceback" not in message


def test_the_sign_in_carries_the_xsrf_token_the_site_issued(site):
    """Laravel rejects a POST whose header does not match the cookie, so
    a client that skipped it would fail against the real site only."""
    result = _fetch(site)
    assert result.rows == TOTAL_ROWS  # 419 from the replica if it had not


def test_a_stale_asset_version_is_retried_rather_than_surfaced(site):
    """The site redeploys and the version moves; a 409 is its way of
    saying so, and it is the client's job to re-read and carry on."""
    Handler.inertia_stale_once = True
    try:
        assert _fetch(site).rows == TOTAL_ROWS
    finally:
        Handler.inertia_stale_once = False


def test_no_credentials_at_all_is_refused_before_any_request(site):
    with pytest.raises(FetchError, match="email and password"):
        _fetch(site, email="", password="")
    assert Handler.inertia_pages == []


def test_an_email_without_a_password_is_refused_too(site):
    with pytest.raises(FetchError, match="email and password"):
        _fetch(site, password="")


def test_an_unreachable_site_fails_with_a_message():
    """A refused connection has to read as a failed fetch, not crash the
    page it was drawn on -- and not as a dump of requests' internals."""
    with pytest.raises(FetchError) as caught:
        fetch_sightings(email=EMAIL, password=PASSWORD,
                        base_url="http://127.0.0.1:1", delay=0)
    message = str(caught.value)
    assert "http://127.0.0.1:1" in message
    assert "upload" in message.lower()
    for leak in ("Traceback", "HTTPConnectionPool", "Max retries",
                 "NewConnectionError"):
        assert leak not in message


def test_a_failure_that_is_not_the_network_still_says_what_it_was():
    """Guessing "check the network" at an unrelated bug sends the reader
    the wrong way; the type and message are the most that is honest."""
    from core.fetch import _unreachable

    said = _unreachable("http://x", ValueError("column missing"))
    assert "ValueError" in said and "column missing" in said
    assert "network" not in said.lower()


def test_the_credentials_are_nowhere_in_the_result(site):
    """A password that survives into the result is a password that
    survives into session state and the load cache behind it."""
    result = _fetch(site)
    haystack = repr(result.__dict__).lower() + result.data.decode("utf-8").lower()
    assert PASSWORD.lower() not in haystack
    assert EMAIL.lower() not in haystack


# --- the window is read off the rows, not promised in advance ------------


def test_the_reported_window_is_the_span_of_what_arrived(site):
    """The listing is not filtered by date, so a window named before the
    call would be a guess the screen then reports as fact."""
    result = _fetch(site)
    assert (result.start_date, result.end_date) == ("2025-10-01", "2025-10-09")
    assert result.name == "forestalerts-2025-10-01-to-2025-10-09.csv"


def test_the_span_reads_day_first_dates_the_way_the_register_writes_them():
    import pandas as pd

    dates = pd.Series(["01-10-2025", "09-10-2025", "05-10-2025"])
    assert observed_window(dates) == ("2025-10-01", "2025-10-09")


def test_an_unparseable_date_does_not_invent_a_span():
    import pandas as pd

    assert observed_window(pd.Series(["not a date"])) == ("", "")
    assert observed_window(pd.Series([], dtype=str)) == ("", "")


def test_a_register_with_no_usable_dates_falls_back_to_the_fetch_date(site,
                                                                     monkeypatch):
    """The file still needs a name, and the day it was pulled is one."""
    import core.fetch as fetch_module

    monkeypatch.setattr(fetch_module, "observed_window", lambda dates: ("", ""))
    result = _fetch(site)
    assert result.name == f"forestalerts-{result.fetched_at}.csv"
    assert (result.start_date, result.end_date) == ("", "")


# --- the form's own validation -------------------------------------------


def test_email_without_a_password_is_refused_before_any_request():
    refusal = FetchRequest(email="a@b.c", password="").is_usable()
    assert refusal and "password" in refusal


def test_a_complete_sign_in_is_accepted():
    assert FetchRequest(email="a@b.c", password="p").is_usable() is None


def test_an_empty_form_is_refused():
    assert FetchRequest(email="", password="").is_usable()


def test_the_refusal_does_not_offer_a_field_the_form_no_longer_has():
    """It used to say "or paste a session cookie instead". There is no
    cookie field now, and advice pointing at one wastes a reader's time.
    """
    refusal = FetchRequest(email="", password="").is_usable()
    assert "cookie" not in refusal.lower()
    assert "endpoint" not in refusal.lower()
