"""Tests for the static maps embedded in the downloadable brief.

The brief is a single self-contained HTML file that gets emailed and
printed, so the maps must render without any external asset and must
never be able to fail the whole document.
"""

import io

import pandas as pd
import pytest
from PIL import Image

from core import map_export
from core.map_export import (
    MAP_HEIGHT,
    MAP_WIDTH,
    _fit_view,
    _projector,
    _world_xy,
    filter_summary,
    sightings_map_svg,
    village_map_svg,
)

# Captured before the autouse fixture stubs it out, for the one test that
# needs the real fetch path.
_REAL_FETCH_TILE = map_export._fetch_tile


def _png(colour=(198, 214, 199), size=(256, 256)):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _sightings(n=12):
    rows = []
    for i in range(n):
        rows.append({
            "Date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "Division": "Anuppur", "Range": "Jaithari", "Beat": f"B{i}",
            "Latitude": 23.0 + i * 0.01, "Longitude": 81.0 + i * 0.01,
            "Total Count": 1, "Crop Damage": int(i % 3 == 0), "Grain Damage": 0,
            "House Damage": int(i == 1), "Injury": int(i == 2),
            "Death": int(i == 3), "Male Death Count": int(i == 3),
            "Female Death Count": 0, "Children Death Count": 0,
        })
    return pd.DataFrame(rows)


def _hotspots():
    return pd.DataFrame([{
        "Hotspot": "H1", "Tier": "Critical", "Sightings": 40,
        "Conflict Events": 12, "Conflict Share %": 30.0,
        "Human Deaths": 1.0, "Recent Deaths": 1.0, "People Injured": 0.0,
        "Night Share %": 80.0, "Radius (km)": 2.2,
        "Centre Latitude": 23.05, "Centre Longitude": 81.05,
        "Beats": "B1", "Divisions": "Anuppur",
        "First Seen": pd.Timestamp("2026-01-01"),
        "Last Seen": pd.Timestamp("2026-01-10"),
    }])


def _villages():
    return pd.DataFrame([
        {"Village": "Kusumhai", "Tier": "Critical", "Conflict Events": 10,
         "Human Deaths": 1.0, "Recent Deaths": 1.0, "People Injured": 0.0,
         "House Damage Events": 0, "Crop Damage Events": 9, "Night Share %": 100.0,
         "Nearest Hotspot": "H1", "Distance to Hotspot (km)": 3.7,
         "Inside Hotspot": False, "Latitude": 23.10, "Longitude": 81.72},
        {"Village": "Kansa", "Tier": "Routine", "Conflict Events": 2,
         "Human Deaths": 0.0, "Recent Deaths": 0.0, "People Injured": 0.0,
         "House Damage Events": 1, "Crop Damage Events": 1, "Night Share %": 50.0,
         "Nearest Hotspot": "H1", "Distance to Hotspot (km)": 4.7,
         "Inside Hotspot": False, "Latitude": 23.08, "Longitude": 81.74},
    ])


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Never reach for a basemap in tests.

    The keyless fallback means clearing the key is no longer enough to
    keep the suite offline; the fetch itself has to be stubbed.
    """
    monkeypatch.setattr(map_export, "maptiler_key", lambda: None)
    monkeypatch.setattr(map_export, "_fetch_tile", lambda url: None)
    map_export._TILE_CACHE.clear()
    map_export._SOURCE_MEMO.clear()


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def test_projection_places_the_centre_at_the_frame_centre():
    project = _projector(23.0, 81.0, 8, MAP_WIDTH, MAP_HEIGHT)
    x, y = project(23.0, 81.0)
    assert x == pytest.approx(MAP_WIDTH / 2)
    assert y == pytest.approx(MAP_HEIGHT / 2)


def test_north_is_up_and_east_is_right():
    project = _projector(23.0, 81.0, 8, MAP_WIDTH, MAP_HEIGHT)
    cx, cy = project(23.0, 81.0)
    _, north_y = project(23.1, 81.0)
    east_x, _ = project(23.0, 81.1)
    assert north_y < cy
    assert east_x > cx


def test_mercator_matches_the_known_tile_origin():
    """Zoom 0 puts (0, 0) at the centre of a single 256 px tile."""
    x, y = _world_xy(0.0, 0.0, 0)
    assert x == pytest.approx(128.0)
    assert y == pytest.approx(128.0)


def test_fit_view_keeps_every_point_inside_the_frame():
    df = _sightings(30)
    lats, lons = df["Latitude"].tolist(), df["Longitude"].tolist()
    centre_lat, centre_lon, zoom = _fit_view(lats, lons, MAP_WIDTH, MAP_HEIGHT)
    project = _projector(centre_lat, centre_lon, zoom, MAP_WIDTH, MAP_HEIGHT)

    for lat, lon in zip(lats, lons):
        x, y = project(lat, lon)
        assert 0 <= x <= MAP_WIDTH
        assert 0 <= y <= MAP_HEIGHT


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_sightings_map_is_self_contained_svg():
    import re

    svg = sightings_map_svg(_sightings(), _hotspots())
    assert "<svg" in svg
    assert "<circle" in svg
    assert "<script" not in svg

    # Nothing may be fetched at open time. The xmlns is a namespace
    # identifier, not a request, so check actual asset references: every
    # href/src must be an inline data URI or not present at all.
    for value in re.findall(r'(?:href|src)="([^"]*)"', svg):
        assert value.startswith("data:"), f"external asset reference: {value}"


def test_hotspot_ring_is_drawn_and_labelled():
    svg = sightings_map_svg(_sightings(), _hotspots())
    assert "H1" in svg


def test_village_map_renders_and_labels_the_critical_village():
    svg = village_map_svg(_villages(), _hotspots())
    assert "<svg" in svg
    assert "Kusumhai" in svg
    # Routine villages are plotted but not labelled.
    assert svg.count("Kansa") == 0


def test_village_names_are_escaped():
    villages = _villages()
    villages.loc[0, "Village"] = "<script>alert(1)</script>"
    svg = village_map_svg(villages)
    assert "<script>alert(1)</script>" not in svg
    assert "&lt;script&gt;" in svg


def test_maps_note_the_missing_basemap_rather_than_pretending():
    svg = sightings_map_svg(_sightings())
    assert "Basemap unavailable" in svg


# ---------------------------------------------------------------------------
# Never break the brief
# ---------------------------------------------------------------------------
def test_empty_dataframe_yields_a_note_not_an_exception():
    """An empty frame has no coordinate columns at all."""
    assert "No mappable sightings" in sightings_map_svg(pd.DataFrame())
    assert "No village ranking" in village_map_svg(pd.DataFrame())


def test_missing_coordinate_columns_are_handled():
    assert "No mappable" in sightings_map_svg(pd.DataFrame({"Beat": ["A"]}))


def test_none_inputs_are_handled():
    assert "No village ranking" in village_map_svg(None)
    assert "No mappable" in sightings_map_svg(None)


def test_hotspots_without_geometry_columns_are_skipped():
    svg = sightings_map_svg(_sightings(), pd.DataFrame({"Hotspot": ["H1"]}))
    assert "<svg" in svg


def test_basemap_failure_does_not_raise(monkeypatch):
    """A network error while fetching tiles must not fail the brief."""
    monkeypatch.setattr(map_export, "maptiler_key", lambda: "KEY")
    monkeypatch.setattr(map_export, "_fetch_tile", _REAL_FETCH_TILE)

    def boom(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(map_export.urllib.request, "urlopen", boom)
    svg = sightings_map_svg(_sightings())
    assert "<svg" in svg
    assert "Basemap unavailable" in svg


def test_a_non_image_response_is_not_treated_as_a_tile(monkeypatch):
    """A rejected key can come back as JSON with a 200."""
    monkeypatch.setattr(map_export, "_fetch_tile", _REAL_FETCH_TILE)

    class _Response:
        status = 200

        def read(self):
            return b'{"message":"Key is invalid"}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(map_export.urllib.request, "urlopen", lambda *a, **k: _Response())
    assert map_export._fetch_tile("https://example.invalid/1/2/3.png") is None


# ---------------------------------------------------------------------------
# Basemap tiles
# ---------------------------------------------------------------------------
def test_tile_placement_matches_the_point_projection():
    """The invariant the whole mosaic rests on.

    Tiles are stitched at integer zoom and scaled to the frame, while
    points are projected at fractional zoom. If the two disagree, every
    sighting sits off its own terrain.
    """
    centre_lat, centre_lon, zoom = 23.05, 81.05, 8.5
    tile_zoom, scale, left, top, _ = map_export._tile_grid(
        centre_lat, centre_lon, zoom, MAP_WIDTH, MAP_HEIGHT
    )
    project = _projector(centre_lat, centre_lon, zoom, MAP_WIDTH, MAP_HEIGHT)

    for lat, lon in [(23.05, 81.05), (23.0, 81.0), (23.2, 81.3)]:
        world_x, world_y = _world_xy(lat, lon, tile_zoom)
        mosaic = ((world_x - left) * scale, (world_y - top) * scale)
        assert mosaic == pytest.approx(project(lat, lon))


def test_tile_grid_spans_the_whole_frame():
    tile_zoom, scale, left, top, tiles = map_export._tile_grid(
        23.0, 81.0, 8.0, MAP_WIDTH, MAP_HEIGHT
    )
    assert tile_zoom == 8 and scale == pytest.approx(1.0)

    xs = [x for x, _ in tiles]
    ys = [y for _, y in tiles]
    assert min(xs) * 256 <= left
    assert (max(xs) + 1) * 256 >= left + MAP_WIDTH
    assert min(ys) * 256 <= top
    assert (max(ys) + 1) * 256 >= top + MAP_HEIGHT
    assert len(tiles) <= map_export.MAX_TILES


def test_basemap_is_embedded_when_tiles_are_reachable(monkeypatch):
    monkeypatch.setattr(map_export, "_fetch_tile", lambda url: _png())
    svg = sightings_map_svg(_sightings())
    assert 'href="data:image/jpeg;base64,' in svg
    assert "Basemap unavailable" not in svg


def test_a_keyless_deployment_still_gets_a_basemap(monkeypatch):
    """The whole point of the fallback: no key, still terrain."""
    seen = []
    monkeypatch.setattr(
        map_export, "_fetch_tile", lambda url: seen.append(url) or _png()
    )
    svg = village_map_svg(_villages())

    assert seen
    assert all("basemaps.cartocdn.com" in url for url in seen)
    assert not any("api.maptiler.com" in url for url in seen)
    assert 'href="data:image/jpeg;base64,' in svg


def test_maptiler_tiles_are_preferred_when_a_key_is_set(monkeypatch):
    monkeypatch.setattr(map_export, "maptiler_key", lambda: "KEY")
    seen = []
    monkeypatch.setattr(
        map_export, "_fetch_tile", lambda url: seen.append(url) or _png()
    )
    sightings_map_svg(_sightings())

    assert seen
    assert all("api.maptiler.com" in url for url in seen)
    assert all("key=KEY" in url for url in seen)


def test_an_unusable_key_falls_through_to_the_keyless_source(monkeypatch):
    monkeypatch.setattr(map_export, "maptiler_key", lambda: "REVOKED")
    monkeypatch.setattr(
        map_export,
        "_fetch_tile",
        lambda url: None if "maptiler" in url else _png(),
    )
    svg = sightings_map_svg(_sightings())

    assert 'href="data:image/jpeg;base64,' in svg
    assert "Basemap unavailable" not in svg


def test_a_missing_tile_does_not_lose_the_whole_basemap(monkeypatch):
    state = {"probe": None, "dropped": 0}

    def fetch(url):
        if state["probe"] is None:
            state["probe"] = url
            return _png()
        if url != state["probe"] and state["dropped"] == 0:
            state["dropped"] = 1
            return None
        return _png()

    monkeypatch.setattr(map_export, "_fetch_tile", fetch)
    svg = sightings_map_svg(_sightings())

    assert state["dropped"] == 1
    assert 'href="data:image/jpeg;base64,' in svg


def test_oversized_tiles_are_normalised(monkeypatch):
    """Some styles serve 512 px tiles for the same z/x/y address."""
    monkeypatch.setattr(map_export, "_fetch_tile", lambda url: _png(size=(512, 512)))
    svg = sightings_map_svg(_sightings())
    assert 'href="data:image/jpeg;base64,' in svg


def test_tiles_are_credited(monkeypatch):
    """MapTiler, Carto and OpenStreetMap all require attribution."""
    monkeypatch.setattr(map_export, "_fetch_tile", lambda url: _png())
    svg = sightings_map_svg(_sightings())
    assert "OpenStreetMap contributors" in svg


def test_an_unreachable_network_is_probed_once_not_once_per_map(monkeypatch):
    """A brief renders two maps. A dead network must not cost two rounds
    of connection timeouts before either of them gives up."""
    monkeypatch.setattr(map_export, "maptiler_key", lambda: "KEY")
    calls = []
    monkeypatch.setattr(
        map_export, "_fetch_tile", lambda url: calls.append(url) or None
    )

    sightings_map_svg(_sightings())
    assert len(calls) == 2, "expected one probe per source, no grid fetch"

    village_map_svg(_villages())
    assert len(calls) == 2, "second map re-probed a source already known dead"


def test_tiles_are_fetched_once_across_maps(monkeypatch):
    """A brief renders two maps over one landscape, often the same tiles."""
    calls = []
    monkeypatch.setattr(
        map_export, "_fetch_tile", lambda url: calls.append(url) or _png()
    )
    sightings_map_svg(_sightings())
    first = len(calls)
    sightings_map_svg(_sightings())
    assert len(calls) == first, "second render refetched tiles instead of caching"


# ---------------------------------------------------------------------------
# Filter summary
# ---------------------------------------------------------------------------
def test_filter_summary_names_the_selected_division():
    summary = filter_summary({"divisions": ["Anuppur"], "ranges": [], "beats": []})
    assert "Division: Anuppur" in summary


def test_filter_summary_covers_every_level():
    summary = filter_summary({
        "divisions": ["Anuppur"], "ranges": ["Jaithari"], "beats": ["Cholna"],
        "severity": "Fatalities only",
    })
    for expected in ["Anuppur", "Jaithari", "Cholna", "Fatalities only"]:
        assert expected in summary


def test_filter_summary_truncates_long_selections():
    summary = filter_summary({"divisions": [f"D{i}" for i in range(9)]})
    assert "+5 more" in summary


def test_filter_summary_defaults_to_all_divisions():
    assert filter_summary(None) == "All divisions"
    assert filter_summary({"divisions": [], "ranges": [], "beats": []}) == "All divisions"


def test_filter_summary_omits_the_default_severity():
    summary = filter_summary({"divisions": ["Anuppur"], "severity": "All reports"})
    assert "All reports" not in summary


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


def test_filter_summary_escapes_csv_derived_names():
    """Division, Range and Beat names come from the uploaded CSV and land
    in a file that gets forwarded and opened elsewhere. The separator is
    markup, so escaping happens inside filter_summary."""
    summary = filter_summary({
        "divisions": ["<script>alert(1)</script>"],
        "ranges": ["<img src=x onerror=alert(2)>"],
        "beats": [],
        "severity": "<b>x</b>",
    })

    assert "<script" not in summary
    assert "<img" not in summary
    assert "<b>" not in summary
    assert "&lt;script&gt;" in summary
    # The separator must survive as markup.
    assert "&nbsp;|&nbsp;" in summary


def test_report_header_does_not_carry_raw_html_from_filters():
    """End to end: a hostile division name must not reach the document."""
    from core.report import generate_html_report

    html = generate_html_report(
        pd.DataFrame(), None, None,
        filters={"divisions": ["<script>alert(1)</script>"]},
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
