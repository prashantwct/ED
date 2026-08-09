"""Tests for movement hotspot detection and village risk ranking.

The properties that matter for a table someone deploys staff from:

* a hotspot is a place, not a region -- clustering must not chain
  through moderate density until it spans a whole landscape,
* scattered reports are not promoted into hotspots,
* distances are real kilometres, not degrees,
* recent casualties drive the tier, consistent with beat tiering,
* villages cannot be invented when no centroid data exists.
"""

import numpy as np
import pandas as pd
import pytest

from core.analytics import compute_is_night, compute_severity
from core.hotspots import (
    DEFAULT_EPS_KM,
    MAX_SENSIBLE_RADIUS_KM,
    _dbscan,
    _to_km_plane,
    detect_hotspots,
    hotspot_caveats,
    hotspot_membership,
    villages_at_risk,
)

START = pd.Timestamp("2026-01-01")
LAT, LON = 23.10, 81.75


def _blob(lat, lon, n, spread_deg=0.004, beat="B", division="D", start=START,
          deaths=0, injuries=0, conflicts=0, day_step=1, seed=0):
    """A tight cluster of ``n`` sightings around one point."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "Date": start + pd.Timedelta(days=i * day_step),
            "Division": division, "Range": "R", "Beat": beat,
            "Latitude": lat + rng.normal(0, spread_deg),
            "Longitude": lon + rng.normal(0, spread_deg),
            "Hour": 21, "Total Count": 2,
            "Crop Damage": int(i < conflicts), "Grain Damage": 0, "House Damage": 0,
            "Injury": int(i >= n - injuries) if injuries else 0,
            "Death": int(i >= n - deaths) if deaths else 0,
            "Male Death Count": int(i >= n - deaths) if deaths else 0,
            "Female Death Count": 0, "Children Death Count": 0,
        })
    return rows


def _prepare(rows):
    df = pd.DataFrame(rows)
    df["Severity Score"] = compute_severity(df)
    df["Is_Night"] = compute_is_night(df)
    return df


# ---------------------------------------------------------------------------
# Clustering mechanics
# ---------------------------------------------------------------------------
def test_projection_produces_real_kilometres():
    """One degree of latitude is ~110.6 km; one of longitude is shorter."""
    lat = np.array([23.0, 24.0])
    lon = np.array([81.0, 81.0])
    plane = _to_km_plane(lat, lon)
    assert abs((plane[1, 1] - plane[0, 1]) - 110.57) < 0.1

    plane_lon = _to_km_plane(np.array([23.0, 23.0]), np.array([81.0, 82.0]))
    east_west = plane_lon[1, 0] - plane_lon[0, 0]
    assert 100 < east_west < 104, "longitude must be shortened by cos(latitude)"


def test_dbscan_separates_two_distinct_blobs():
    a = _to_km_plane(*np.array([[23.0] * 20, [81.0] * 20]))
    b = _to_km_plane(*np.array([[23.5] * 20, [81.5] * 20]))
    labels = _dbscan(np.vstack([a, b]), eps=2.0, min_samples=5)
    assert len(set(labels[labels >= 0])) == 2


def test_dbscan_leaves_sparse_points_unclustered():
    points = _to_km_plane(
        np.array([23.0, 23.4, 23.8, 24.2]), np.array([81.0, 81.4, 81.8, 82.2])
    )
    labels = _dbscan(points, eps=1.0, min_samples=5)
    assert (labels == -1).all()


# ---------------------------------------------------------------------------
# Hotspot detection
# ---------------------------------------------------------------------------
def test_two_separated_concentrations_stay_separate():
    df = _prepare(
        _blob(LAT, LON, 30, seed=1)
        + _blob(LAT + 0.30, LON + 0.30, 30, beat="B2", seed=2)
    )
    hotspots = detect_hotspots(df, eps_km=1.0, min_samples=10)
    assert len(hotspots) == 2


def test_hotspots_are_places_not_regions():
    """A hotspot a manager can act on has a small footprint. This is the
    chaining guard: at a large neighbour distance DBSCAN links separate
    concentrations through the moderate density between them."""
    df = _prepare(
        _blob(LAT, LON, 40, seed=3)
        + _blob(LAT + 0.25, LON + 0.25, 40, beat="B2", seed=4)
    )
    hotspots = detect_hotspots(df, eps_km=DEFAULT_EPS_KM, min_samples=10)
    assert not hotspots.empty
    assert hotspots["Radius (km)"].max() < MAX_SENSIBLE_RADIUS_KM


def test_scattered_data_yields_no_hotspots():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(60):
        rows.append({
            "Date": START + pd.Timedelta(days=i),
            "Division": "D", "Range": "R", "Beat": "B",
            "Latitude": 23.0 + rng.uniform(0, 1.2),
            "Longitude": 81.0 + rng.uniform(0, 1.2),
            "Hour": 12, "Total Count": 1, "Crop Damage": 0, "Grain Damage": 0,
            "House Damage": 0, "Injury": 0, "Death": 0,
            "Male Death Count": 0, "Female Death Count": 0, "Children Death Count": 0,
        })
    hotspots = detect_hotspots(_prepare(rows), eps_km=0.5, min_samples=10)
    assert hotspots.empty


def test_hotspot_with_a_recent_fatality_is_critical():
    df = _prepare(_blob(LAT, LON, 30, deaths=1, conflicts=10, seed=5))
    hotspots = detect_hotspots(df, eps_km=1.0, min_samples=10)
    assert hotspots.iloc[0]["Tier"] == "Critical"
    assert hotspots.iloc[0]["Human Deaths"] == 1


def test_hotspot_with_an_old_fatality_is_not_critical():
    """Consistent with beat tiering: Critical is the deploy-now tier."""
    rows = _blob(LAT, LON, 40, deaths=0, conflicts=5, seed=6)
    rows[0]["Death"] = 1
    rows[0]["Male Death Count"] = 1
    rows[0]["Date"] = START
    df = _prepare(rows)
    period_end = df["Date"].max() + pd.Timedelta(days=300)
    hotspots = detect_hotspots(df, eps_km=1.0, min_samples=10, as_of=period_end)
    assert hotspots.iloc[0]["Recent Deaths"] == 0
    assert hotspots.iloc[0]["Tier"] != "Critical"


def test_hotspots_are_numbered_worst_first():
    df = _prepare(
        _blob(LAT, LON, 30, conflicts=1, seed=8)
        + _blob(LAT + 0.3, LON + 0.3, 30, beat="B2", deaths=1, conflicts=10, seed=9)
    )
    hotspots = detect_hotspots(df, eps_km=1.0, min_samples=10)
    assert hotspots.iloc[0]["Hotspot"] == "H1"
    assert hotspots.iloc[0]["Tier"] == "Critical"


def test_radius_ignores_a_single_outlying_report():
    """Radius is a 90th percentile, so one stray report does not inflate
    the footprint a patrol is planned around."""
    rows = _blob(LAT, LON, 40, seed=10)
    rows[0]["Latitude"] = LAT + 0.5
    hotspots = detect_hotspots(_prepare(rows), eps_km=1.0, min_samples=10)
    assert hotspots.iloc[0]["Radius (km)"] < 3.0


def test_empty_input_returns_empty_frame_with_columns():
    hotspots = detect_hotspots(pd.DataFrame())
    assert hotspots.empty
    assert "Tier" in hotspots.columns


def test_membership_labels_align_with_the_source_frame():
    df = _prepare(_blob(LAT, LON, 30, seed=11))
    labels = hotspot_membership(df, eps_km=1.0, min_samples=10)
    assert len(labels) == len(df)
    assert (labels == "H1").sum() > 0


def test_caveats_report_uncovered_sightings_and_missing_villages():
    df = _prepare(_blob(LAT, LON, 30, seed=12))
    hotspots = detect_hotspots(df, eps_km=1.0, min_samples=10)
    notes = " ".join(hotspot_caveats(hotspots, df, None))
    assert "fall inside a hotspot" in notes
    assert "centroid" in notes.lower()


def test_caveats_flag_an_oversized_hotspot():
    df = _prepare(_blob(LAT, LON, 200, spread_deg=0.05, seed=13))
    hotspots = detect_hotspots(df, eps_km=3.0, min_samples=10)
    if not hotspots.empty and hotspots["Radius (km)"].max() > MAX_SENSIBLE_RADIUS_KM:
        assert any("chained" in n for n in hotspot_caveats(hotspots, df, None))


# ---------------------------------------------------------------------------
# Villages at risk
# ---------------------------------------------------------------------------
def _villages(**named):
    return pd.DataFrame(
        [{"Village": n, "Latitude": la, "Longitude": lo} for n, (la, lo) in named.items()]
    )


def test_no_villages_without_centroids():
    """The export has no village field. Villages must not be invented."""
    df = _prepare(_blob(LAT, LON, 30, conflicts=20, seed=14))
    assert villages_at_risk(df, None).empty
    assert villages_at_risk(df, pd.DataFrame()).empty


def test_village_beside_a_hotspot_is_ranked():
    df = _prepare(_blob(LAT, LON, 30, conflicts=20, seed=15))
    hotspots = detect_hotspots(df, eps_km=1.0, min_samples=10)
    out = villages_at_risk(df, _villages(Near=(LAT, LON)), hotspots, radius_km=3.0)
    assert len(out) == 1
    assert out.iloc[0]["Village"] == "Near"
    assert out.iloc[0]["Conflict Events"] > 0


def test_distant_village_is_excluded():
    df = _prepare(_blob(LAT, LON, 30, conflicts=20, seed=16))
    out = villages_at_risk(df, _villages(Far=(LAT + 1.0, LON + 1.0)), None, radius_km=3.0)
    assert out.empty


def test_a_village_between_two_hotspots_counts_both():
    """Nearest-village attribution would credit only one of them."""
    left = _blob(LAT, LON - 0.02, 30, conflicts=30, seed=17)
    right = _blob(LAT, LON + 0.02, 30, beat="B2", conflicts=30, seed=18)
    df = _prepare(left + right)
    out = villages_at_risk(df, _villages(Middle=(LAT, LON)), None, radius_km=5.0)
    assert out.iloc[0]["Conflict Events"] == 60


def test_village_with_a_recent_fatality_is_critical():
    df = _prepare(_blob(LAT, LON, 30, deaths=1, conflicts=20, seed=19))
    out = villages_at_risk(df, _villages(V=(LAT, LON)), None, radius_km=3.0)
    assert out.iloc[0]["Tier"] == "Critical"


def test_village_inside_hotspot_is_flagged():
    df = _prepare(_blob(LAT, LON, 40, conflicts=30, seed=20))
    hotspots = detect_hotspots(df, eps_km=1.0, min_samples=10)
    out = villages_at_risk(df, _villages(V=(LAT, LON)), hotspots, radius_km=3.0)
    assert bool(out.iloc[0]["Inside Hotspot"]) is True
    assert out.iloc[0]["Nearest Hotspot"] == "H1"


def test_village_radius_is_kilometres_not_degrees():
    """A village 4 km away must fall outside a 3 km radius. Treating the
    radius as degrees would sweep in most of the district."""
    df = _prepare(_blob(LAT, LON, 30, conflicts=20, seed=21))
    four_km_north = LAT + 4 / 110.57
    assert villages_at_risk(
        df, _villages(V=(four_km_north, LON)), None, radius_km=3.0
    ).empty
    assert not villages_at_risk(
        df, _villages(V=(four_km_north, LON)), None, radius_km=5.0
    ).empty


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
