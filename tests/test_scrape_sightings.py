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
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from core.data_loader import load_and_validate_csv
from tools.scrape_sightings import (
    ForestAlertsScraper,
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

EMAIL = "ranger@example.org"
PASSWORD = "correct horse"
CSRF = "tok-abc123"
SESSION_COOKIE = "fa_session=live"
PAGE_SIZE = 4
TOTAL_ROWS = 9  # three pages, the last one short


def _row(index):
    """One sighting, in the shapes the listing actually prints."""
    return {
        "id": 100 + index,
        "date": f"{(index % 28) + 1:02d}-10-2025 07:{index % 60:02d} PM",
        "lat": round(23.10 + index * 0.01, 4),
        "lng": round(81.20 + index * 0.01, 4),
        "division": "Shahdol" if index % 2 else "Anuppur",
        "range": "Jaisinghnagar" if index % 2 else "Kotma",
        "beat": f"Beat {index:02d}",
        "total": index % 5,
        "male": index % 2,
        "crop": index % 3,
        "death": 1 if index == 3 else 0,
    }


ROWS = [_row(index) for index in range(TOTAL_ROWS)]

LOGIN_PAGE = f"""<!doctype html><html><head><title>Sign in</title></head><body>
<form method="post" action="/login">
  <input type="hidden" name="_token" value="{CSRF}">
  <input type="email" name="email" value="">
  <input type="password" name="password">
  <button type="submit">Login</button>
</form></body></html>"""


def _listing(page, rows, advertise=(1, 2, 3)):
    """A page of the listing, wrapped in the noise a real one carries."""
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['id']}</td>"
            f"<td>{row['date']}</td>"
            f"<td>{row['lat']}</td><td>{row['lng']}</td>"
            f"<td>{row['division']}</td><td>{row['range']}</td>"
            # An entity and a line break inside a cell.
            f"<td>{row['beat']}&nbsp;<br>Sub</td>"
            f"<td>{row['total']}</td><td>{row['male']}</td>"
            f"<td>{row['crop']}</td><td>{row['death']}</td>"
            "<td><a href='/admin/sightings/" f"{row['id']}" "'>View</a></td>"
            "</tr>"
        )
    nav = "".join(
        f"<a href='?page={n}&start_date=2025-10-01'>{n}</a>" for n in (advertise or ())
    )
    return f"""<!doctype html><html><head><title>Sightings</title>
<script>var t = "<table><tr><td>not a table</td></tr></table>";</script></head><body>
<table class="summary"><tr><th>Filter</th><th>Value</th></tr>
  <tr><td>Division</td><td>All</td></tr></table>
<table class="listing">
  <thead><tr>
    <th>S.No</th><th>Sighting Date</th><th>Latitude</th><th>Longitude</th>
    <th>Division Name</th><th>Range Name</th><th>Beat Name</th>
    <th>Total Elephants</th><th>Tuskers</th><th>Crop Damage</th>
    <th>Human Death</th><th>Action</th>
  </tr></thead>
  <tbody>{''.join(body)}</tbody>
</table>
<nav>{nav}</nav>
<p>Showing page {page}</p></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    """A small, deliberately awkward stand-in for the admin site."""

    requests = []          # every listing query the scraper sent
    expire_on_page = None  # serve the login page once for this page number
    expired = set()
    advertise = (1, 2, 3)  # what the paginator links to

    def log_message(self, *args):  # keep pytest output clean
        pass

    def _send(self, body, status=200, headers=()):
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _authenticated(self):
        return SESSION_COOKIE in (self.headers.get("Cookie") or "")

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if parsed.path == "/login":
            return self._send(LOGIN_PAGE)
        if parsed.path == "/admin/no-table":
            # A 200 with nothing to parse, which is what a listing that
            # renders its rows in JavaScript looks like from here.
            return self._send("<html><body><p>Loading...</p></body></html>")
        if parsed.path != "/admin/sightings":
            return self._send("<h1>Not found</h1>", status=404)
        if not self._authenticated():
            return self._send(LOGIN_PAGE)

        page = int(query.get("page", ["1"])[0])
        if "page" in query:  # the redirect after login is not a listing fetch
            type(self).requests.append(query)
        if self.expire_on_page == page and page not in self.expired:
            # A session that lapses mid-walk: a 200 carrying the login form.
            type(self).expired.add(page)
            return self._send(LOGIN_PAGE)

        start = (page - 1) * PAGE_SIZE
        rows = ROWS[start:start + PAGE_SIZE]
        if not rows:
            # Out of range: this framework clamps back to the last page
            # rather than returning an empty table.
            rows = ROWS[-PAGE_SIZE:]
        return self._send(_listing(page, rows, advertise=self.advertise))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
        if form.get("_token", [""])[0] != CSRF:
            return self._send("<h1>419 Page Expired</h1>", status=419)
        if form.get("email", [""])[0] != EMAIL or form.get("password", [""])[0] != PASSWORD:
            return self._send(LOGIN_PAGE, status=200)
        self._send(
            "<html><body>Redirecting</body></html>",
            status=302,
            headers=[("Set-Cookie", f"{SESSION_COOKIE}; Path=/"),
                     ("Location", "/admin/sightings")],
        )


@pytest.fixture
def site():
    """A running replica; yields its base URL."""
    _Handler.requests = []
    _Handler.expire_on_page = None
    _Handler.expired = set()
    _Handler.advertise = (1, 2, 3)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


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
    pages = [int(q["page"][0]) for q in _Handler.requests]
    assert pages == [1, 2, 3, 4], f"kept paging past the end: {pages}"
    assert len(records) == TOTAL_ROWS, "the clamped page was scraped twice"


def test_a_walk_with_no_pagination_links_at_all_still_ends(site):
    _Handler.advertise = None
    records = _scrape(site, max_pages=50)
    assert len(records) == TOTAL_ROWS
    assert [int(q["page"][0]) for q in _Handler.requests] == [1, 2, 3, 4]


def test_a_windowed_paginator_does_not_truncate_the_pull(site):
    """Plenty of paginators link only to nearby pages, so the advertised
    last page is a floor and never a stop condition. Believing it here
    would silently drop the final third of the register."""
    _Handler.advertise = (1, 2)
    records = _scrape(site, max_pages=50)
    assert len(records) == TOTAL_ROWS, "stopped at the advertised last page"


def test_the_window_is_pinned_and_every_filter_is_sent(site):
    _scrape(site, end_date="2026-09-12")
    for query in _Handler.requests:
        assert query["start_date"] == ["2025-10-01"]
        assert query["end_date"] == ["2026-09-12"]
        for name in ("beat", "damage_type", "division", "elephant_visible",
                     "range", "search"):
            assert query[name] == [""], f"{name} was not sent empty"


def test_end_date_defaults_to_today(site):
    _scrape(site)
    assert _Handler.requests[0]["end_date"] == [today()]


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
    assert _Handler.requests == []


def test_a_session_that_lapses_mid_walk_is_re_established(site):
    _Handler.expire_on_page = 2
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
