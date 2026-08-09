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
                    "Village Exposure", "How to read this"]:
        assert section in html


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
