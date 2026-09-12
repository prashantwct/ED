"""Tests for the Forest Alerts scraper.

The site itself cannot be reached from CI, and a scraper that is only
ever exercised against the real thing is a scraper nobody can change. So
the end-to-end tests run the real code -- session, login, CSRF,
pagination, parse, CSV -- against a local replica of the listing served
by :mod:`http.server`, and finish by loading the result through
``core.data_loader``. That last step is the assertion that matters: the
scrape is correct when the dashboard accepts it.

The replica deliberately does not serve tidy HTML. Its header spellings
differ from the dashboard's, the date cell carries a 12-hour time, there
is a decorative summary table above the listing, a cell uses an entity
and a ``<br>``, and asking for a page past the end re-serves the last one
-- each of which is a way a real admin panel has of breaking a parser
that assumed its own output.
"""

import csv

import pytest

from replica import (
    CSRF,
    EMAIL,
    Handler,
    LOGIN_PAGE,
    PASSWORD,
    ROWS,
    SESSION_COOKIE,
    TOTAL_ROWS,
    _listing,
)

from core.data_loader import load_and_validate_csv
from tools.scrape_sightings import (
    ForestAlertsScraper,
    _has_listing,
    diagnose,
    ScrapeError,
    Table,
    canonical_header,
    choose_table,
    extract_forms,
    extract_tables,
    fingerprint,
    last_page_number,
    looks_like_login,
    map_record,
    normalise_time,
    order_columns,
    scrape,
    table_records,
    today,
    write_csv,
)


def _scrape(site, **overrides):
    kwargs = dict(
        base_url=site,
        email=EMAIL,
        password=PASSWORD,
        start_date="2025-10-01",
        delay=0,
    )
    kwargs.update(overrides)
    return scrape(**kwargs)


# --- end to end ----------------------------------------------------------


def test_scrape_walks_every_page_and_stops_at_the_end(site):
    records = _scrape(site)
    assert len(records) == TOTAL_ROWS
    assert [r["ID"] for r in records] == [str(row["id"]) for row in ROWS]


def test_a_re_served_page_stops_the_walk(site):
    """Page 4 comes back holding rows already seen. Without that check the
    walk spins to --max-pages and duplicates the tail."""
    records = _scrape(site, max_pages=50)
    pages = [int(q["page"][0]) for q in Handler.requests]
    assert pages == [1, 2, 3, 4], f"kept paging past the end: {pages}"
    assert len(records) == TOTAL_ROWS, "the clamped page was scraped twice"


def test_a_walk_with_no_pagination_links_at_all_still_ends(site):
    Handler.advertise = None
    records = _scrape(site, max_pages=50)
    assert len(records) == TOTAL_ROWS
    assert [int(q["page"][0]) for q in Handler.requests] == [1, 2, 3, 4]


def test_a_windowed_paginator_does_not_truncate_the_pull(site):
    """Plenty of paginators link only to nearby pages, so the advertised
    last page is a floor and never a stop condition. Believing it here
    would silently drop the final third of the register."""
    Handler.advertise = (1, 2)
    records = _scrape(site, max_pages=50)
    assert len(records) == TOTAL_ROWS, "stopped at the advertised last page"


def test_the_window_is_pinned_and_every_filter_is_sent(site):
    _scrape(site, end_date="2026-09-12")
    for query in Handler.requests:
        assert query["start_date"] == ["2025-10-01"]
        assert query["end_date"] == ["2026-09-12"]
        for name in ("beat", "damage_type", "division", "elephant_visible",
                     "range", "search"):
            assert query[name] == [""], f"{name} was not sent empty"


def test_end_date_defaults_to_today(site):
    _scrape(site)
    assert Handler.requests[0]["end_date"] == [today()]


def test_scraped_csv_loads_in_the_dashboard(site, tmp_path):
    """The end-to-end assertion: the app's own loader accepts the file."""
    records = _scrape(site)
    out = tmp_path / "sightings.csv"
    columns = write_csv(records, out)

    for required in ("Date", "Latitude", "Longitude", "Division", "Range", "Beat"):
        assert required in columns

    frame, warnings = load_and_validate_csv(str(out))
    assert len(frame) == TOTAL_ROWS
    assert not warnings, warnings
    assert frame["Date"].dt.year.eq(2025).all()
    assert frame["Division"].isin({"Shahdol", "Anuppur"}).all()
    assert frame["Latitude"].between(23.0, 23.3).all()
    assert frame["Longitude"].between(81.0, 81.5).all()
    # "07:03 PM" in the date cell has to survive as an evening hour, or
    # the night-share calculation quietly reads every sighting as 00:00.
    assert frame["Hour"].eq(19).all()


def test_header_synonyms_and_extra_columns_both_survive(site, tmp_path):
    records = _scrape(site)
    out = tmp_path / "sightings.csv"
    write_csv(records, out)
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["Total Count"] == "0"       # from "Total Elephants"
    assert rows[1]["Male Count"] == "1"        # from "Tuskers"
    assert rows[3]["Death"] == "1"             # from "Human Death"
    assert rows[0]["Beat"] == "Beat 00 Sub"    # entity and <br> flattened
    assert rows[0]["Action"] == "View"         # unmapped column kept as-is


def test_dashboard_columns_come_before_the_sites_own(site):
    records = _scrape(site)
    columns = order_columns(records)
    assert columns[:8] == ["ID", "Date", "Time", "Latitude", "Longitude",
                           "Division", "Range", "Beat"]
    assert columns[-1] == "Action"


# --- auth ----------------------------------------------------------------


def test_bad_password_is_reported_not_scraped_around(site):
    with pytest.raises(ScrapeError, match="rejected"):
        _scrape(site, password="wrong")


def test_missing_credentials_fail_before_any_request(site):
    with pytest.raises(ScrapeError, match="No credentials"):
        _scrape(site, email=None, password=None)
    assert Handler.requests == []


def test_a_session_that_lapses_mid_walk_is_re_established(site):
    Handler.expire_on_page = 2
    records = _scrape(site)
    assert len(records) == TOTAL_ROWS, "rows were lost to the re-login"


def test_a_page_with_no_table_says_so_rather_than_writing_nothing(site):
    with pytest.raises(ScrapeError, match="No data table"):
        _scrape(site, sightings_path="/admin/no-table")


def test_a_pasted_cookie_is_accepted_instead_of_a_login(site):
    records = _scrape(site, email=None, password=None, cookie=SESSION_COOKIE)
    assert len(records) == TOTAL_ROWS


# --- parsing units -------------------------------------------------------


def test_the_listing_wins_over_a_decorative_table():
    table = choose_table(extract_tables(_listing(1, ROWS[:2])))
    assert table is not None
    assert len(table.data_rows) == 2
    assert table.header[0] == "S.No"


def test_script_text_that_looks_like_a_table_is_not_parsed():
    tables = extract_tables(_listing(1, ROWS[:1]))
    assert not any("not a table" in cell for t in tables for r in t.rows for cell in r)


def test_colspan_does_not_shift_the_columns():
    html = """<table><tr><th>A</th><th>B</th><th>C</th></tr>
              <tr><td colspan="2">wide</td><td>c</td></tr></table>"""
    table = extract_tables(html)[0]
    record = table_records(table)[0]
    assert record["A"] == "wide"
    assert record["C"] == "c"


def test_a_row_longer_than_its_header_keeps_the_overflow():
    html = "<table><tr><th>A</th></tr><tr><td>1</td><td>2</td></tr></table>"
    record = table_records(extract_tables(html)[0])[0]
    assert record == {"A": "1", "Column 2": "2"}


def test_headerless_tables_are_read_as_data():
    html = "<table><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>"
    records = table_records(extract_tables(html)[0])
    assert records == [{"Column 1": "1", "Column 2": "2"},
                       {"Column 1": "3", "Column 2": "4"}]


def test_an_unclosed_table_still_yields_its_rows():
    html = "<table><tr><th>Date</th></tr><tr><td>01-10-2025</td></tr>"
    assert extract_tables(html)[0].data_rows == [["01-10-2025"]]


@pytest.mark.parametrize("raw,expected", [
    ("Sighting Date", "Date"),
    ("division_name", "Division"),
    ("LAT", "Latitude"),
    ("No. of Elephants", "Total Count"),
    ("Tuskers", "Male Count"),
    ("Beat", "Beat"),
    ("Something Else", None),
])
def test_header_synonyms(raw, expected):
    assert canonical_header(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("07:05 PM", "19:05"),
    ("12:30 AM", "00:30"),
    ("12:30 PM", "12:30"),
    ("7:05 p.m.", "19:05"),
    ("19:05:30", "19:05:30"),
    ("23:59", "23:59"),
    ("25:00", "25:00"),     # not a clock: left for the loader to warn about
    ("morning", "morning"),
])
def test_time_normalisation(raw, expected):
    assert normalise_time(raw) == expected


def test_a_time_in_the_date_cell_moves_to_its_own_column():
    record = map_record({"Sighting Date": "01-10-2025, 07:05 PM"})
    assert record == {"Date": "01-10-2025", "Time": "19:05"}


def test_an_existing_time_column_is_not_overwritten_by_the_date_cell():
    record = map_record({"Date": "01-10-2025 07:05 PM", "Time": "06:00"})
    assert record["Date"] == "01-10-2025"
    assert record["Time"] == "06:00"


def test_an_iso_stamp_splits_on_its_t_separator():
    record = map_record({"Date": "2025-10-01T19:05:00"})
    assert record["Date"] == "2025-10-01"
    assert record["Time"] == "19:05:00"


def test_a_whole_number_pair_is_not_mistaken_for_coordinates():
    """"12, 34" in some unnamed count column is two counts, and inventing
    a location from it would put a sighting on a map that never happened."""
    record = map_record({"Date": "01-10-2025", "Casualties": "12, 34"})
    assert "Latitude" not in record


def test_a_column_that_says_it_holds_coordinates_is_taken_at_its_word():
    record = map_record({"Date": "01-10-2025", "Lat Long": "23, 81"})
    assert record["Latitude"] == "23"
    assert record["Longitude"] == "81"


def test_a_combined_location_cell_becomes_latitude_and_longitude():
    record = map_record({"Date": "01-10-2025", "Location": "23.1234, 81.5678"})
    assert record["Latitude"] == "23.1234"
    assert record["Longitude"] == "81.5678"


def test_a_reversed_coordinate_pair_is_put_back_in_order():
    """Longitude here runs past 90 and latitude cannot, so the order is
    recoverable without trusting the site's column name."""
    record = map_record({"Coordinates": "95.5, 23.1"})
    assert record["Latitude"] == "23.1"
    assert record["Longitude"] == "95.5"


def test_two_headers_claiming_one_column_keep_both_values():
    record = map_record({"Beat": "Kachhar", "Beat Name": "Kachhar East"})
    assert record["Beat"] == "Kachhar"
    assert record["Beat Name"] == "Kachhar East"


def test_login_page_detection():
    assert looks_like_login(LOGIN_PAGE)
    assert not looks_like_login(_listing(1, ROWS[:1]))


def test_the_login_forms_hidden_token_is_read_back():
    form = next(f for f in extract_forms(LOGIN_PAGE) if f.password_field)
    assert form.fields["_token"] == CSRF
    assert form.password_field == "password"
    assert ForestAlertsScraper._user_field(form) == "email"


def test_pagination_links_give_the_last_page():
    assert last_page_number(_listing(1, ROWS[:1])) == 3
    assert last_page_number("<p>no links</p>") is None


def test_identical_rows_share_a_fingerprint():
    assert fingerprint({"a": "1", "b": "2"}) == fingerprint({"b": "2", "a": "1"})
    assert fingerprint({"a": "1"}) != fingerprint({"a": "2"})


def test_writing_nothing_is_an_error_not_an_empty_file(tmp_path):
    with pytest.raises(ScrapeError, match="nothing to write"):
        write_csv([], tmp_path / "out.csv")


def test_a_backwards_window_is_refused(site):
    with pytest.raises(ScrapeError, match="before start_date"):
        _scrape(site, end_date="2025-09-01")


def test_a_malformed_date_argument_is_refused(site):
    with pytest.raises(ScrapeError, match="YYYY-MM-DD"):
        _scrape(site, end_date="13/09/2026")


# --- when the login form is not what was expected -------------------------
# The live site answered the first real run with "no password field found
# at /login". Three ordinary things produce that message, and only one of
# them is the site being a JavaScript app: a password box with no name
# attribute and a password box outside any <form> were both invisible to
# the parser, and a form at another path was never looked for.


def test_a_password_box_with_no_name_is_seen_even_though_it_cannot_be_posted():
    """Invisible to the old parser, which skipped any input without a
    name -- so a perfectly ordinary login screen read as 'no password
    field found'."""
    from replica import NAMELESS_LOGIN_PAGE

    assert looks_like_login(NAMELESS_LOGIN_PAGE)
    report = diagnose(NAMELESS_LOGIN_PAGE)
    assert report.unsubmittable_password
    assert not report.submittable_password
    assert "no name attribute" in report.verdict()


def test_a_password_box_outside_any_form_is_seen_too():
    from replica import LOOSE_LOGIN_PAGE

    assert looks_like_login(LOOSE_LOGIN_PAGE)
    report = diagnose(LOOSE_LOGIN_PAGE)
    assert report.unsubmittable_password
    assert "outside any <form>" in report.verdict()


def test_an_app_shell_is_named_as_one():
    from replica import APP_SHELL_PAGE

    report = diagnose(APP_SHELL_PAGE)
    assert report.looks_like_an_app_shell
    assert report.mounts == ["#app"]
    assert "not in its HTML" in report.verdict()


def test_a_real_login_form_is_still_reported_as_submittable():
    report = diagnose(LOGIN_PAGE)
    assert report.submittable_password
    assert "fill and post" in report.verdict()


def test_a_login_form_at_another_path_is_found(site):
    """A form at /admin/login used to fail as 'no password field at
    /login'. The candidates are tried in turn instead."""
    Handler.login_path = "/admin/login"
    records = _scrape(site)
    assert len(records) == TOTAL_ROWS


def test_the_failure_names_what_each_candidate_page_held(site):
    Handler.login_path = "/nowhere"
    Handler.login_html = None
    with pytest.raises(ScrapeError) as caught:
        _scrape(site)
    message = str(caught.value)
    assert "/login" in message and "/admin/login" in message
    assert "HTTP 404" in message
    assert "--cookie" in message


def test_a_javascript_login_says_to_use_a_cookie(site):
    from replica import APP_SHELL_PAGE

    Handler.login_html = APP_SHELL_PAGE
    with pytest.raises(ScrapeError) as caught:
        _scrape(site)
    message = str(caught.value)
    assert "application shell" in message
    assert "--cookie" in message
    assert "developer tools" in message


def test_the_probe_reports_every_candidate_and_the_listing(site):
    """The point of --probe: say what is actually being served, without
    needing credentials to find out."""
    Handler.login_html = None
    scraper = ForestAlertsScraper(site, delay=0)
    reports = scraper.probe()
    assert [r.url.rsplit("/", 1)[-1] or "root" for r in reports][:1] == ["login"]
    assert reports[0].submittable_password
    # The listing, fetched without a session, is the login page again.
    assert reports[-1].url.endswith("/admin/sightings")


def test_a_listing_that_mentions_a_password_is_not_an_expired_session(site):
    """looks_like_login is now satisfied by any password box, so a
    listing carrying a profile widget must not be read as a logout."""
    listing = _listing(1, ROWS[:2]).replace(
        "</body>", '<input type="password" name="new_password"></body>'
    )
    assert looks_like_login(listing)
    assert _has_listing(listing)
