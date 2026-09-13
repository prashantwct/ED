"""Tests for downloading the site's own export.

The sightings screen has an Export button, which makes it the best of
the three sources here: one request against the register as the
department publishes it, with no pagination to walk and no row shape to
infer. What it hands back is not known in advance, so the tests are
mostly about reading a response whose format the server may not declare
honestly -- a .xlsx served as application/octet-stream, a CSV in cp1252,
an HTML page where a file was expected.
"""

import io

import pytest

from core.data_loader import load_and_validate_csv
from tools.export_source import (
    ExportFormatError,
    decode,
    detect_format,
    records_from_csv_text,
    suggested_filename,
    to_csv_text,
)
from tools.scrape_sightings import records_to_csv, scrape
from replica import API_TOKEN, EMAIL, Handler, PASSWORD, SESSION_COOKIE, TOTAL_ROWS, export_csv_text


def _scrape(site, **overrides):
    kwargs = dict(
        base_url=site,
        export_path="/admin/sightings/export",
        cookie=SESSION_COOKIE,
        start_date="2025-10-01",
        delay=0,
    )
    kwargs.update(overrides)
    return scrape(**kwargs)


# --- end to end -----------------------------------------------------------


def test_the_export_loads_in_the_dashboard(site, tmp_path):
    records = _scrape(site)
    assert len(records) == TOTAL_ROWS

    text, columns = records_to_csv(records)
    path = tmp_path / "export.csv"
    path.write_text(text, encoding="utf-8")
    frame, warnings = load_and_validate_csv(str(path))
    assert len(frame) == TOTAL_ROWS
    assert not warnings, warnings
    # The export's own spellings -- "Division Name", "Total Elephants" --
    # go through the same synonym table as every other source.
    assert frame["Division"].isin({"Shahdol", "Anuppur"}).all()
    assert "Total Count" in columns and "Male Count" in columns
    # A 12-hour clock in its own column, converted so the night share
    # does not silently read every sighting as midnight.
    assert frame["Hour"].eq(19).all()


def test_a_spreadsheet_export_is_read_as_one(site, tmp_path):
    """Served as application/octet-stream with an .xlsx name, which is
    how an admin panel's Export button usually arrives."""
    Handler.export_as = "xlsx"
    records = _scrape(site)
    assert len(records) == TOTAL_ROWS

    text, _ = records_to_csv(records)
    path = tmp_path / "export.csv"
    path.write_text(text, encoding="utf-8")
    frame, warnings = load_and_validate_csv(str(path))
    assert len(frame) == TOTAL_ROWS
    assert not warnings, warnings


def test_the_export_is_one_request_over_the_asked_for_window(site):
    _scrape(site, end_date="2026-09-13")
    assert len(Handler.export_requests) == 1, Handler.export_requests
    query = Handler.export_requests[0]
    assert query["start_date"] == ["2025-10-01"]
    assert query["end_date"] == ["2026-09-13"]
    assert "page" not in query, "an export is not paginated"
    for name in ("beat", "damage_type", "division", "elephant_visible",
                 "range", "search"):
        assert query[name] == [""]


def test_a_bearer_token_authorises_the_export_too(site):
    records = _scrape(site, cookie=None,
                      headers={"Authorization": f"Bearer {API_TOKEN}"})
    assert len(records) == TOTAL_ROWS


def test_an_unauthorised_export_says_what_to_refresh(site):
    with pytest.raises(Exception) as caught:
        _scrape(site, cookie="stale=1")
    message = str(caught.value)
    assert "401" in message and "--cookie" in message


def test_an_html_page_where_a_file_was_expected_says_so(site):
    """A session that has lapsed gets the app shell back with a 200. That
    has to read as a failed download, not as an export of no rows."""
    Handler.export_as = "html"
    with pytest.raises(Exception, match="HTML page"):
        _scrape(site)


def test_the_export_route_is_ranked_above_the_listing_route(site):
    """One request against a published file beats walking an API, so the
    discovery output has to put it first."""
    from tools.scrape_sightings import ForestAlertsScraper, diagnose
    import requests as _requests

    scraper = ForestAlertsScraper(site, delay=0)
    shell = diagnose(_requests.get(f"{site}/spa").text, url=f"{site}/spa")
    routes = scraper.discover_api(shell)
    assert routes  # the bundle names some
    bundle = "var a='/api/sightings/export', b='/api/sightings', c='/api/divisions';"
    from tools.api_source import find_api_paths
    from tools.scrape_sightings import _looks_like_export

    found = find_api_paths(bundle)
    assert _looks_like_export("/api/sightings/export")
    assert not _looks_like_export("/api/sightings")
    assert set(found) == {"/api/sightings/export", "/api/sightings", "/api/divisions"}


# --- reading a response the server did not describe honestly -------------


@pytest.mark.parametrize("content,content_type,filename,expected", [
    (b"a,b\n1,2\n", "text/csv", None, "csv"),
    (b"a,b\n1,2\n", "application/octet-stream", "x.csv", "csv"),
    (b"a\tb\n1\t2\n", "application/octet-stream", "x.tsv", "tsv"),
    (b"PK\x03\x04rest", "application/octet-stream", None, "xlsx"),
    (b"PK\x03\x04rest", "", "sightings.xlsx", "xlsx"),
    (b"\xd0\xcf\x11\xe0rest", "", None, "xls"),
    (b'{"data": []}', "", None, "json"),
    (b"<html><body>hi", "", None, "html"),
    (b"a,b\n1,2\n", "", None, "csv"),
    (b"a\tb\tc\n1\t2\t3\n", "", None, "tsv"),
])
def test_the_format_is_detected_from_whatever_the_response_offers(
    content, content_type, filename, expected
):
    assert detect_format(content, content_type, filename) == expected


def test_the_filename_beats_a_lying_content_type():
    """A CSV served as application/vnd.ms-excel is common, and reading it
    as a workbook would fail on a file that is plainly text."""
    assert detect_format(b"a,b\n1,2\n", "application/vnd.ms-excel", "out.csv") == "csv"


def test_an_unreadable_response_is_an_error_not_a_guess():
    with pytest.raises(ExportFormatError, match="does not recognise"):
        detect_format(b"\x00\x01\x02\x03", "application/octet-stream", None)


@pytest.mark.parametrize("header,expected", [
    ('attachment; filename="sightings.xlsx"', "sightings.xlsx"),
    ("attachment; filename=sightings.csv", "sightings.csv"),
    ("attachment; filename*=UTF-8''report.csv", "report.csv"),
    ("inline", None),
    ("", None),
])
def test_the_suggested_filename_is_read(header, expected):
    assert suggested_filename(header) == expected


def test_a_windows_export_decodes_rather_than_mangling():
    """Excel on Windows emits cp1252, and a name with a dash in it is
    where a naive utf-8 decode falls over."""
    assert decode("Beat – North".encode("cp1252")) == "Beat – North"


def test_a_tab_separated_export_becomes_comma_separated():
    assert to_csv_text(b"a\tb\n1\t2\n", "tsv") == "a,b\r\n1,2\r\n"


def test_a_json_export_is_read_through_the_same_shapes():
    text = to_csv_text(b'{"data": [{"a": 1, "b": 2}]}', "json")
    assert records_from_csv_text(text) == [{"a": "1", "b": "2"}]


def test_an_html_body_is_refused_with_a_reason():
    with pytest.raises(ExportFormatError, match="not authorised"):
        to_csv_text(b"<html>nope</html>", "html")


# --- reading the rows -----------------------------------------------------


def test_export_rows_keep_the_sites_own_headers():
    records = records_from_csv_text(export_csv_text())
    assert len(records) == TOTAL_ROWS
    assert records[0]["Division Name"] == "Anuppur"
    assert records[0]["S.No"] == "100"
    assert records[0]["Sighting Time"] == "07:00 PM"


def test_blank_rows_at_the_end_of_an_export_are_dropped():
    """Spreadsheet exports routinely carry trailing empties, and a row of
    nothing becomes a dropped-row warning in the loader."""
    records = records_from_csv_text("a,b\n1,2\n,\n\n")
    assert records == [{"a": "1", "b": "2"}]


def test_whitespace_around_values_is_trimmed():
    assert records_from_csv_text("a , b \n 1 , 2 \n") == [{"a": "1", "b": "2"}]
