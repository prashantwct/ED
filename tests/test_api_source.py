"""Tests for reading the register out of a JSON API.

The live site turned out to be a single-page application: `/` and
`/login` serve a four-script shell around `<div id="app">` with no text
and no form in the HTML. There is no table to parse, so the rows can
only come from the API the page calls after it boots -- and the shape of
that payload is still unknown.

That unknown is what these tests are about. The wrapper the rows arrive
in is guessed at from the shapes APIs actually use, so each of those
shapes is here; the vocabulary inside a row is not guessed at, because
the synonym table already handles it.
"""

import json

import pytest

from core.data_loader import load_and_validate_csv
from tools.api_source import (
    ApiShapeError,
    find_api_paths,
    find_rows,
    flatten,
    has_next_page,
    last_page_from_json,
    looks_like_json,
    records_from_json,
)
from tools.scrape_sightings import (
    ForestAlertsScraper,
    ScrapeError,
    map_record,
    parse_headers,
    records_to_csv,
    scrape,
)
from replica import API_TOKEN, APP_BUNDLE, EMAIL, PASSWORD, Handler, TOTAL_ROWS


def _scrape(site, **overrides):
    kwargs = dict(
        base_url=site,
        api_path="/api/sightings",
        api_login_path="/api/login",
        email=EMAIL,
        password=PASSWORD,
        start_date="2025-10-01",
        delay=0,
    )
    kwargs.update(overrides)
    return scrape(**kwargs)


# --- end to end against the API ------------------------------------------


def test_the_api_walk_produces_a_csv_the_dashboard_loads(site, tmp_path):
    """The same assertion the HTML path has to meet: correct is what the
    dashboard accepts, not what parsed."""
    records = _scrape(site)
    assert len(records) == TOTAL_ROWS

    text, columns = records_to_csv(records)
    for required in ("Date", "Latitude", "Longitude", "Division", "Range", "Beat"):
        assert required in columns, columns

    path = tmp_path / "api.csv"
    path.write_text(text, encoding="utf-8")
    frame, warnings = load_and_validate_csv(str(path))
    assert len(frame) == TOTAL_ROWS
    assert not warnings, warnings
    assert frame["Division"].isin({"Shahdol", "Anuppur"}).all()
    # The nested {"id": 1, "name": "Shahdol"} has to arrive as the name.
    assert not frame["Division"].astype(str).str.isdigit().any()


def test_a_token_returned_by_the_sign_in_authorises_the_walk(site):
    records = _scrape(site)
    assert len(records) == TOTAL_ROWS
    assert [int(q["page"][0]) for q in Handler.api_requests] == [1, 2, 3]


def test_a_rejected_api_sign_in_says_so(site):
    with pytest.raises(ScrapeError, match="HTTP 422"):
        _scrape(site, password="wrong")


def test_an_unauthorised_api_says_what_to_refresh(site):
    with pytest.raises(ScrapeError) as caught:
        _scrape(site, email=None, password=None, api_login_path=None,
                headers={"Authorization": "Bearer stale"})
    message = str(caught.value)
    assert "401" in message
    assert "--cookie" in message and "--header" in message


def test_a_browser_cookie_authorises_the_api_too(site):
    from replica import SESSION_COOKIE

    records = _scrape(site, email=None, password=None, api_login_path=None,
                      cookie=SESSION_COOKIE)
    assert len(records) == TOTAL_ROWS


def test_the_api_is_believed_when_it_says_there_is_no_next_page(site):
    """next_page_url of null ends the walk, so the clamp-detection never
    has to spend a request finding out."""
    _scrape(site)
    assert len(Handler.api_requests) == 3, Handler.api_requests


def test_routes_are_read_out_of_the_sites_own_bundles(site):
    from tools.scrape_sightings import diagnose
    import requests as _requests

    scraper = ForestAlertsScraper(site, delay=0)
    shell = diagnose(_requests.get(f"{site}/spa").text, url=f"{site}/spa")
    assert shell.looks_like_an_app_shell
    routes = scraper.discover_api(shell)
    assert "/api/sightings" in routes
    # The route naming what we are after is ranked to the top.
    assert routes[0] == "/api/sightings"
    assert "/api/login" in routes
    # Assets are not routes.
    assert not any(route.endswith((".css", ".png")) for route in routes)


# --- payload shapes -------------------------------------------------------


@pytest.mark.parametrize("payload", [
    [{"a": 1}, {"a": 2}],
    {"data": [{"a": 1}, {"a": 2}]},
    {"results": [{"a": 1}, {"a": 2}]},
    {"data": {"data": [{"a": 1}, {"a": 2}], "current_page": 1}},
    {"meta": {"total": 2}, "payload": {"items": [{"a": 1}, {"a": 2}]}},
])
def test_rows_are_found_whatever_they_are_wrapped_in(payload):
    assert find_rows(payload) == [{"a": 1}, {"a": 2}]


def test_a_named_wrapper_wins_over_some_other_list():
    """A payload can hold more than one list, and the one called data is
    the one meant -- picking the first found would be a coin toss."""
    payload = {"filters": [{"id": 1}], "data": [{"a": 1}]}
    assert find_rows(payload) == [{"a": 1}]


def test_a_payload_with_no_rows_is_an_error_not_an_empty_scrape():
    with pytest.raises(ApiShapeError, match="No list of records"):
        records_from_json({"message": "Unauthenticated."})


def test_a_nested_object_contributes_its_label_and_its_keys():
    flat = flatten({"division": {"id": 3, "name": "Shahdol"}})
    assert flat["division"] == "Shahdol"
    assert flat["division.id"] == "3"


def test_a_nested_object_with_no_label_keeps_only_its_keys():
    flat = flatten({"reporter": {"id": 7}})
    assert "reporter" not in flat
    assert flat["reporter.id"] == "7"


def test_booleans_become_numbers_so_damage_flags_stay_countable():
    """crop_damage: true has to reach the dashboard as 1, or every
    conflict rate computed from it reads zero."""
    assert flatten({"crop_damage": True, "house_damage": False}) == {
        "crop_damage": "1", "house_damage": "0"
    }


def test_nulls_become_empty_rather_than_the_text_none():
    assert flatten({"beat": None}) == {"beat": ""}


def test_whole_floats_lose_their_decimal_point():
    """An id of 101.0 in the CSV is an id nobody can join on."""
    assert flatten({"id": 101.0, "lat": 23.15}) == {"id": "101", "lat": "23.15"}


def test_a_list_of_scalars_is_joined_and_a_list_of_objects_skipped():
    flat = flatten({"photos": ["a.jpg", "b.jpg"], "history": [{"at": "x"}]})
    assert flat["photos"] == "a.jpg; b.jpg"
    assert "history" not in flat


def test_an_api_row_maps_onto_the_dashboards_columns():
    from replica import ROWS, _api_row

    record = map_record(flatten(_api_row(ROWS[1])))
    assert record["Division"] == "Shahdol"
    assert record["Beat"] == "Beat 01"
    assert record["Total Count"] == "1"
    assert record["Male Count"] == "1"
    assert record["Crop Damage"] == "1"
    # The ISO stamp splits on its T, or the night/day answer is lost.
    assert record["Date"] == "2025-10-02"
    assert record["Time"] == "19:01:00"


@pytest.mark.parametrize("payload,expected", [
    ({"last_page": 4}, 4),
    ({"meta": {"last_page": 7}}, 7),
    ({"totalPages": 2}, 2),
    ({"total": 90}, None),
    ({}, None),
])
def test_the_page_count_is_read_where_apis_put_it(payload, expected):
    assert last_page_from_json(payload) == expected


@pytest.mark.parametrize("payload,expected", [
    ({"next_page_url": "/api/x?page=2"}, True),
    ({"next_page_url": None}, False),
    ({"links": {"next": None}}, False),
    ({}, None),
])
def test_whether_another_page_exists(payload, expected):
    assert has_next_page(payload) is expected


def test_json_is_recognised_before_it_is_parsed():
    assert looks_like_json('  {"a": 1}')
    assert looks_like_json("[1]")
    assert not looks_like_json("<!doctype html>")


# --- bundle reading -------------------------------------------------------


def test_api_routes_are_picked_out_of_a_bundle():
    routes = find_api_paths(APP_BUNDLE)
    assert "/api/sightings" in routes
    assert "/api/login" in routes
    assert "/api/beats/list" in routes
    assert "/build/app.css" not in routes
    assert "/img/logo.png" not in routes
    assert not any(route.startswith("https://") for route in routes)


def test_the_route_worth_trying_first_is_first():
    assert find_api_paths(APP_BUNDLE)[0] == "/api/sightings"


def test_a_bundle_with_no_routes_yields_nothing():
    assert find_api_paths("var x = 1; console.log('hello');") == []


# --- headers --------------------------------------------------------------


def test_headers_are_parsed_from_the_command_line():
    assert parse_headers(["Authorization: Bearer abc", "X-Tenant:  mp "]) == {
        "Authorization": "Bearer abc", "X-Tenant": "mp",
    }


def test_a_malformed_header_is_refused():
    with pytest.raises(ScrapeError, match="Name: Value"):
        parse_headers(["Authorization Bearer abc"])
