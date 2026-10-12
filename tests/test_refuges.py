"""Day refuges, against a landscape where the truth is planted.

refuge_fixtures builds a 10 m Esri-coded raster and a register with two
real refuges and three decoys, each decoy breaking one of the tests a
refuge has to pass. These check every decoy is caught for the right
reason, not just that "some refuges come out".
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

import refuge_fixtures as fx
from core import esri_landcover as lc
from core import refuges as rf


@pytest.fixture(scope="module")
def tif(tmp_path_factory):
    return fx.landcover_tif(tmp_path_factory.mktemp("lc") / "esri.tif")


@pytest.fixture(scope="module")
def result(tif):
    return rf.find_refuges(fx.register(), fx.villages(), lc.LandCover([tif]))


def _at(table, xy, km=0.6):
    lat, lon = fx.latlon(xy)
    d = rf._km(lat, lon, table["Latitude"].to_numpy(float), table["Longitude"].to_numpy(float))
    hits = table[d <= km]
    return None if hits.empty else hits.iloc[0]


def test_the_planted_refuges_are_found_with_the_right_confidence(result):
    table = result["refuges"]
    a, b = _at(table, fx.REFUGE_A), _at(table, fx.REFUGE_B)
    assert a is not None and a["Confidence"] == rf.CONFIRMED
    assert a["Days Used"] == 10 and a["Cover Share %"] >= 90
    assert a["Tree Patch (ha)"] > 1000
    # Four days is a refuge, but not enough to confirm one.
    assert b is not None and b["Confidence"] == rf.PROBABLE
    assert "Small isolated patch" in b["Flags"]
    assert 5 <= b["Tree Patch (ha)"] <= 15


def test_a_night_feeding_ground_is_not_a_refuge(result):
    assert _at(result["refuges"], fx.FEEDING, km=0.4) is None
    assert any("feeding grounds" in c for c in result["caveats"])


def test_one_afternoon_of_sightings_is_a_passage_not_a_refuge(result):
    assert _at(result["refuges"], fx.PASSAGE) is None


def test_day_use_without_cover_is_flagged_and_capped(result):
    open_day = _at(result["refuges"], fx.OPEN_DAY)
    assert open_day is not None
    assert open_day["Confidence"] == rf.POSSIBLE
    assert "Little cover" in open_day["Flags"]
    assert "Verify on the ground" in open_day["Recommended Action"]


def test_dawn_and_dusk_sightings_are_not_rest(result):
    a = _at(result["refuges"], fx.REFUGE_A)
    assert a["Day Sightings"] == 30  # 10 days x 3 rest-hour reports; 06 and 17 h left out


def test_raids_link_to_the_refuge_in_use_and_point_the_right_way(result):
    a = _at(result["refuges"], fx.REFUGE_A)
    assert a["Linked Raids"] >= 10 and a["Raids Head"] == "E"
    assert 2.0 <= a["Raid Distance (km, median)"] <= 3.5
    b = _at(result["refuges"], fx.REFUGE_B)
    assert b["Linked Raids"] >= 4 and "Patchgaon" in b["Villages Raided"]
    assert "Warn Patchgaon" in b["Recommended Action"]


def test_a_raid_is_linked_once_to_the_nearest_refuge(result):
    links = result["links"]
    assert not links.duplicated(["Date", "Hour", "Latitude", "Longitude"]).any()


def test_a_raid_on_a_night_the_refuge_was_empty_is_not_linked(tif):
    df = fx.register()
    lat, lon = fx.latlon((fx.REFUGE_A[0] + 2500, fx.REFUGE_A[1]))
    far_night = {"Date": pd.Timestamp("2026-06-15"), "Hour": 22, "Latitude": lat,
                 "Longitude": lon, "Division": "Test", "Range": "Test", "Beat": "Test",
                 "Total Count": 3, "Crop Damage": 1, "House Damage": 0, "Injury": 0, "Death": 0}
    out = rf.find_refuges(pd.concat([df, pd.DataFrame([far_night])], ignore_index=True),
                          fx.villages(), lc.LandCover([tif]))
    assert not (out["links"]["Date"] == pd.Timestamp("2026-06-15")).any()


def test_without_land_cover_nothing_is_confirmed():
    out = rf.find_refuges(fx.register(), fx.villages(), None)
    assert len(out["refuges"]) >= 2
    assert (out["refuges"]["Confidence"] != rf.CONFIRMED).all()
    assert any("No land cover" in c for c in out["caveats"])


def test_no_hours_means_no_refuges_and_says_why():
    df = fx.register().assign(Hour=pd.NA)
    out = rf.find_refuges(df, None, None)
    assert out["refuges"].empty
    assert any("without an hour" in c for c in out["caveats"])


def test_a_lat_lon_raster_measures_the_same_as_utm(tif, tmp_path):
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    geo = tmp_path / "esri_wgs84.tif"
    with rasterio.open(tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        meta = src.meta | {"crs": "EPSG:4326", "transform": transform,
                           "width": width, "height": height}
        with rasterio.open(geo, "w", **meta) as dst:
            reproject(rasterio.band(src, 1), rasterio.band(dst, 1),
                      resampling=Resampling.nearest)
    lat, lon = fx.latlon(fx.REFUGE_B)
    utm = rf.cover_metrics(lc.LandCover([tif]).window(lat, lon))
    wgs = rf.cover_metrics(lc.LandCover([str(geo)]).window(lat, lon))
    assert abs(utm["Tree Patch (ha)"] - wgs["Tree Patch (ha)"]) <= 2
    assert abs(utm["Cover Share %"] - wgs["Cover Share %"]) <= 10


def test_utm_tile_names_match_esri_files():
    assert lc.utm_tile(23.8, 81.0) == "44Q"
    assert lc.utm_tile(24.5, 81.0) == "44R"
    assert lc.remote_url(2024, 23.8, 81.0).endswith("/lc2024/44Q_20240101-20250101.tif")


def test_field_files(result):
    gj = json.loads(rf.to_geojson(result["refuges"], result["links"]))
    kinds = {f["properties"]["kind"] for f in gj["features"]}
    assert kinds == {"refuge", "linked raid"}
    root = ET.fromstring(rf.to_kml(result["refuges"]))
    marks = root.findall(".//{http://www.opengis.net/kml/2.2}Placemark")
    assert len(marks) == len(result["refuges"])


def test_thumbnail_is_a_png(result):
    window = next(iter(result["windows"].values()))
    assert lc.thumbnail_png(window, 120).startswith(b"\x89PNG")


def test_view_reads_the_conflict_views_register(tmp_path):
    from streamlit.testing.v1 import AppTest

    df = fx.register()
    df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")
    data = df.to_csv(index=False).encode()
    app = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app, default_timeout=240)
    at.session_state["loaded_register"] = (data, "register.csv")
    at.session_state["view"] = "Day refuges"
    at.session_state["refuge_lc_source"] = "None"
    at.run()
    assert not at.exception, at.exception
    assert any(m.label == "Day refuges" for m in at.metric)
    labels = {b.label for b in at.get("download_button")}
    assert {"Refuges (CSV)", "Refuges + raids (GeoJSON)", "Refuges (KML, Google Earth)"} <= labels


def test_view_without_a_register_says_where_to_load_it():
    from streamlit.testing.v1 import AppTest

    app = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app, default_timeout=120)
    at.session_state["view"] = "Day refuges"
    at.run()
    assert not at.exception, at.exception
    assert any("Conflict dashboard view first" in i.value for i in at.info)
