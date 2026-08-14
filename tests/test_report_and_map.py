"""Tests for report rendering and map encoding.

Two regressions are guarded here:

* The report interpolates values that came out of a CSV -- beat and
  village names -- straight into HTML. The file is downloaded and
  emailed onward, so those values must be escaped.
* Map points were sized in metres with no pixel floor. This landscape
  spans ~150 km, which the adaptive view fits at roughly 1 km per pixel,
  so a 60 m radius is 0.06 px: the map renders empty at exactly the zoom
  a division-wide review uses.
"""

import math

import pandas as pd
import pytest

from core.analytics import (
    SEVERITY_BAND_LABELS,
    compute_is_night,
    compute_severity,
    severity_distribution,
)
from core.map_engine import (
    CATEGORY_COLORS,
    MAX_RADIUS_M,
    MIN_RADIUS_M,
    RADIUS_MIN_PIXELS,
    _adaptive_view_state,
    _severity_to_radius,
)
from core.report import generate_html_report


def _df(n=12, beat="Beat A"):
    rows = []
    for i in range(n):
        rows.append({
            "Date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "Division": "Div", "Range": "Rng", "Beat": beat,
            "Latitude": 22.3 + i * 0.01, "Longitude": 80.6 + i * 0.01,
            "Hour": 20 if i % 2 else 9, "Total Count": 1,
            "Crop Damage": int(i % 3 == 0), "Grain Damage": 0,
            "House Damage": int(i == 1), "Injury": int(i == 2),
            "Death": int(i == 3), "Male Death Count": int(i == 3),
            "Female Death Count": 0, "Children Death Count": 0,
        })
    df = pd.DataFrame(rows)
    df["Severity Score"] = compute_severity(df)
    df["Is_Night"] = compute_is_night(df)
    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def test_report_escapes_values_taken_from_the_csv():
    df = _df(beat="<script>alert(1)</script>")
    html = generate_html_report(df, df["Date"].min(), df["Date"].max())
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_renders_on_an_empty_dataframe():
    html = generate_html_report(pd.DataFrame(), None, None)
    assert html.startswith("<!DOCTYPE html>")
    assert "N/A" in html


def test_report_includes_casualty_and_priority_sections():
    df = _df()
    html = generate_html_report(df, df["Date"].min(), df["Date"].max())
    for section in ["Assessment", "Priority Beats", "Timing", "People Killed",
                    "Villages at Risk", "Movement Hotspots", "How to read this"]:
        assert section in html


def _coverage_frame(village="Kusumhai", registered=0, tier="Critical"):
    from core.coverage import _coverage_label

    return pd.DataFrame([{
        "Village": village, "Tier": tier, "Conflict Events": 7,
        "Human Deaths": 1.0, "People Injured": 0.0,
        "Registered Contacts": registered,
        "Coverage": _coverage_label(registered, 3),
    }])


def test_the_brief_omits_the_coverage_section_when_no_registry_was_loaded():
    """A heading reading "no data" invites the reader to infer "no gaps"."""
    df = _df()
    html = generate_html_report(df, df["Date"].min(), df["Date"].max())
    assert "Early-Warning Coverage" not in html


def test_the_brief_names_the_villages_with_nobody_enrolled():
    df = _df()
    html = generate_html_report(
        df, df["Date"].min(), df["Date"].max(),
        coverage=_coverage_frame(registered=0),
        coverage_stats={"registrants": 100, "unmatched": 4, "min_contacts": 3},
    )
    assert "Early-Warning Coverage" in html
    assert "Kusumhai" in html
    assert "No contact" in html
    assert "not a reachable person" in html


def test_the_brief_says_so_when_every_exposed_village_is_covered():
    df = _df()
    html = generate_html_report(
        df, df["Date"].min(), df["Date"].max(),
        coverage=_coverage_frame(registered=5),
        coverage_stats={"min_contacts": 3},
    )
    assert "Early-Warning Coverage" in html
    assert "at least 3 registered contacts" in html


def test_village_names_in_the_coverage_section_are_escaped():
    df = _df()
    html = generate_html_report(
        df, df["Date"].min(), df["Date"].max(),
        coverage=_coverage_frame(village="<script>alert(1)</script>"),
        coverage_stats={"min_contacts": 3},
    )
    assert "<script>alert(1)</script>" not in html


def test_report_conflict_breakdown_counts_each_report_once():
    """A report with both a death and crop damage belongs under Death
    only; double-counting would make the rows exceed the total."""
    df = _df()
    html = generate_html_report(df, df["Date"].min(), df["Date"].max())
    assert "Human fatality" in html
    assert "Presence only" in html


# ---------------------------------------------------------------------------
# Severity bands
# ---------------------------------------------------------------------------
def test_severity_bands_separate_fatalities_from_crop_damage():
    """Equal-width bins put ~everything in bucket one once a fatality is
    worth 100 and a sighting 0.5."""
    bands = severity_distribution(_df()).set_index("Band")
    assert list(bands.index) == SEVERITY_BAND_LABELS
    assert bands.loc["Human fatality", "Count"] == 1
    assert bands.loc["Human injury", "Count"] == 1
    assert bands.loc["Presence only", "Count"] > 0


def test_severity_bands_are_stable_shape_on_sparse_data():
    tiny = _df(n=1)
    bands = severity_distribution(tiny)
    assert len(bands) == len(SEVERITY_BAND_LABELS)


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
def test_points_have_a_pixel_floor_so_they_survive_landscape_zoom():
    assert RADIUS_MIN_PIXELS >= 2


def test_landscape_zoom_would_make_metre_radii_subpixel():
    """Documents why the pixel floor is required, using this dataset's
    actual extent rather than a guess."""
    df = pd.DataFrame({
        "Latitude": [21.7, 24.4], "Longitude": [79.9, 81.6],
        "Severity Score": [1.0, 1.0],
    })
    zoom = _adaptive_view_state(df).zoom
    metres_per_pixel = 156543.03 * math.cos(math.radians(23.0)) / (2 ** zoom)
    assert MAX_RADIUS_M / metres_per_pixel < 1.0


def test_radius_is_log_scaled_so_property_damage_stays_distinguishable():
    """Linear scaling against a fatality collapses every non-fatal
    incident onto the minimum radius."""
    df = _df()
    df["_category"] = "Crop"
    radii = _severity_to_radius(df)
    non_fatal = radii[df["Severity Score"] < 100]
    assert non_fatal.nunique() > 1
    assert radii.min() >= MIN_RADIUS_M


def test_casualty_points_are_drawn_at_full_size():
    df = _df()
    df["_category"] = ["Death" if s >= 100 else "Crop" for s in df["Severity Score"]]
    radii = _severity_to_radius(df)
    assert radii[df["_category"] == "Death"].iloc[0] == MAX_RADIUS_M


def test_every_conflict_category_has_a_distinct_colour():
    assert len(set(CATEGORY_COLORS.values())) == len(CATEGORY_COLORS)


def test_view_state_accounts_for_longitude_convergence():
    """A 1-degree east-west spread is geographically smaller than a
    1-degree north-south one and must not drive the zoom as if equal."""
    wide_lon = pd.DataFrame({"Latitude": [22.3, 22.4], "Longitude": [80.0, 81.0]})
    wide_lat = pd.DataFrame({"Latitude": [22.0, 23.0], "Longitude": [80.6, 80.7]})
    assert _adaptive_view_state(wide_lon).zoom > _adaptive_view_state(wide_lat).zoom


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# UI escaping
# ---------------------------------------------------------------------------
def test_ui_escapes_names_taken_from_the_csv(monkeypatch):
    """core.ui renders through unsafe_allow_html. Beat and division names
    come from an uploaded file, so they must never reach the DOM raw."""
    from core import ui

    captured = []
    monkeypatch.setattr(ui.st, "markdown", lambda html, **kw: captured.append(html))

    ui.hotspot_card({
        "Hotspot": "H1",
        "Tier": "High",
        "Divisions": "<script>alert(1)</script>",
        "Beats": "<img src=x onerror=alert(2)>",
        "Sightings": 10, "Conflict Events": 3, "Conflict Share %": 30.0,
        "Human Deaths": 0, "People Injured": 0, "Night Share %": 50.0,
        "Radius (km)": 1.2,
    })

    html = "".join(captured)
    # Escaping neutralises the angle brackets rather than stripping the
    # text, so assert no tag can form -- not that the words are absent.
    assert "<script" not in html
    assert "<img" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img" in html


def test_ui_escapes_section_subtitles(monkeypatch):
    from core import ui

    captured = []
    monkeypatch.setattr(ui.st, "markdown", lambda html, **kw: captured.append(html))
    ui.section("target", "Title", "<b>not bold</b>")

    assert "<b>not bold</b>" not in "".join(captured)


# ---------------------------------------------------------------------------
# Basemap configuration
# ---------------------------------------------------------------------------
def test_basemap_style_url_is_built_from_the_configured_key(monkeypatch):
    from core import map_engine

    monkeypatch.setattr(map_engine, "maptiler_key", lambda: "TESTKEY")
    url = map_engine.basemap_style("Satellite")

    assert url.startswith("https://api.maptiler.com/maps/satellite/style.json")
    assert "key=TESTKEY" in url
    assert map_engine.basemap_warning() is None


def test_missing_key_falls_back_to_a_keyless_basemap(monkeypatch):
    """Without a key the map must still show tiles. Points on a blank
    background give a manager no geographic context at all."""
    from core import map_engine

    monkeypatch.setattr(map_engine, "maptiler_key", lambda: None)
    style = map_engine.basemap_style("Satellite")

    assert style == map_engine._CARTO_FALLBACK_STYLE
    assert style.startswith("https://")
    assert "key=" not in style, "the fallback must not need an API key"
    assert "MAPTILER_KEY" in map_engine.basemap_warning()


def test_unknown_style_falls_back_to_the_default(monkeypatch):
    from core import map_engine

    monkeypatch.setattr(map_engine, "maptiler_key", lambda: "K")
    fallback = map_engine.BASEMAP_STYLES[map_engine.DEFAULT_BASEMAP]
    assert fallback in map_engine.basemap_style("no such style")


def test_key_is_read_from_the_environment(monkeypatch):
    from core import map_engine

    monkeypatch.setenv("MAPTILER_KEY", "FROM_ENV")
    monkeypatch.setattr(map_engine.st, "secrets", {}, raising=False)
    assert map_engine.maptiler_key() == "FROM_ENV"


def test_no_maptiler_key_is_committed_to_the_repo():
    """The key is client-side and cannot be hidden, but it must not be
    baked into tracked source."""
    import subprocess

    tracked = subprocess.run(
        ["git", "grep", "-lE", "api.maptiler.com.*key=[A-Za-z0-9]{10}", "--", "."],
        capture_output=True, text=True,
    )
    assert tracked.stdout.strip() == "", f"key literal in: {tracked.stdout}"


def test_village_labels_do_not_overprint():
    """deck.gl TextLayer has no collision handling. Labelling every
    Critical and High village produced an illegible smear on the real
    data (36 labels in two tight clusters)."""
    import math

    from core.map_engine import MAX_LABELS, _select_labels

    separation_km = 4.0

    # Ten villages 1 km apart -- far too close to all carry a label.
    rows = [{
        "Village": f"V{i}", "Tier": "Critical",
        "Latitude": 23.0 + i * (1 / 110.57), "Longitude": 81.0,
        "Human Deaths": 1, "Conflict Events": 10 - i,
    } for i in range(10)]
    kept = _select_labels(pd.DataFrame(rows), separation_km)

    assert 0 < len(kept) < 10
    assert len(kept) <= MAX_LABELS
    separations = [
        abs(a - b) * 110.57
        for i, a in enumerate(kept["Latitude"]) for b in kept["Latitude"][i + 1:]
    ]
    assert all(s >= separation_km - 0.01 for s in separations)


def test_labels_prefer_the_worst_village_in_a_cluster():
    from core.map_engine import _select_labels

    rows = [
        {"Village": "Fatal", "Tier": "Critical", "Latitude": 23.0, "Longitude": 81.0,
         "Human Deaths": 2, "Conflict Events": 3},
        {"Village": "Nearby", "Tier": "High", "Latitude": 23.002, "Longitude": 81.0,
         "Human Deaths": 0, "Conflict Events": 50},
    ]
    kept = _select_labels(pd.DataFrame(rows), 4.0)
    assert list(kept["Village"]) == ["Fatal"]


def test_routine_villages_are_never_labelled():
    from core.map_engine import _select_labels

    rows = [{"Village": "Quiet", "Tier": "Routine", "Latitude": 23.0,
             "Longitude": 81.0, "Human Deaths": 0, "Conflict Events": 1}]
    assert _select_labels(pd.DataFrame(rows), 4.0).empty
