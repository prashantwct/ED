"""Tests for village-centroid enrichment.

The regression guarded here: building the neighbour tree on unscaled
lat/lon treats a degree of longitude as equal to a degree of latitude.
At the ~22 N latitude of this data a longitude degree is only 0.93 of a
latitude degree, so east-west separation is overstated by about 8% and
the nearest-village search is biased toward villages lying north or
south of a sighting.
"""

import numpy as np
import pandas as pd
import pytest

from core.spatial import (
    NEAR_VILLAGE_THRESHOLD_KM,
    _haversine_km,
    attach_nearest_village,
    load_village_centroids,
)

LAT, LON = 22.30, 80.60
# At 22.3 N: 1 deg latitude ~ 110.9 km, 1 deg longitude ~ 103.1 km.
ONE_KM_LAT = 0.00901
ONE_KM_LON = 0.00970


def _sighting():
    return pd.DataFrame({"Latitude": [LAT], "Longitude": [LON]})


def _villages(**named):
    return pd.DataFrame(
        [{"Village": n, "Latitude": la, "Longitude": lo} for n, (la, lo) in named.items()]
    )


def test_east_west_distance_is_not_overstated():
    """A pure east-west offset of 0.1 degrees is 10.29 km here, not 11.12."""
    villages = _villages(East=(LAT, LON + 0.10))
    out, _ = attach_nearest_village(_sighting(), villages)
    assert out["Distance to Village (km)"].iloc[0] == pytest.approx(10.29, abs=0.05)


def test_north_south_distance_is_unchanged():
    villages = _villages(North=(LAT + 0.10, LON))
    out, _ = attach_nearest_village(_sighting(), villages)
    assert out["Distance to Village (km)"].iloc[0] == pytest.approx(11.12, abs=0.05)


def test_nearest_village_is_not_biased_toward_north_south():
    """An eastern village 0.8 km away must beat a northern one at 1.0 km.
    Unscaled longitude inflates the eastern distance and picks North."""
    villages = _villages(
        North=(LAT + ONE_KM_LAT, LON),
        East=(LAT, LON + ONE_KM_LON * 0.8),
    )
    out, _ = attach_nearest_village(_sighting(), villages)
    assert out["Nearest Village"].iloc[0] == "East"


def test_reported_distance_matches_haversine():
    villages = _villages(V=(LAT + 0.05, LON + 0.07))
    out, _ = attach_nearest_village(_sighting(), villages)
    expected = float(
        _haversine_km(
            np.array([LAT]), np.array([LON]),
            np.array([LAT + 0.05]), np.array([LON + 0.07]),
        )[0]
    )
    assert out["Distance to Village (km)"].iloc[0] == pytest.approx(expected, abs=0.01)


def test_near_village_threshold_admits_the_village_surroundings():
    """A centroid is a point but a village is not. An incident 1 km from
    the centre is at the village; a 0.5 km radius would exclude it and
    quietly zero out the proximity signal."""
    villages = _villages(V=(LAT + ONE_KM_LAT, LON))
    out, _ = attach_nearest_village(_sighting(), villages)
    assert NEAR_VILLAGE_THRESHOLD_KM >= 1.0
    assert bool(out["Near Village"].iloc[0]) is True


def test_far_incident_is_not_near_a_village():
    villages = _villages(V=(LAT + 0.30, LON))
    out, _ = attach_nearest_village(_sighting(), villages)
    assert bool(out["Near Village"].iloc[0]) is False


def test_enrichment_is_a_no_op_without_centroids():
    df = _sighting()
    out, warnings = attach_nearest_village(df, None)
    assert "Nearest Village" not in out.columns
    assert warnings == []


def test_missing_default_centroids_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_village_centroids(None) is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
