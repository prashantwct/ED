"""Day refuges, against a landscape where the truth is planted.

refuge_fixtures builds a 10 m Esri-coded raster and a register with two
real refuges and three decoys, each decoy breaking one of the tests a
refuge has to pass. These check every decoy is caught for the right
reason, not just that "some refuges come out".
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import refuge_fixtures as fx
from core import esri_landcover as lc
from core import refuges as rf


@pytest.fixture(scope="module")
def tif(tmp_path_factory):
    return fx.landcover_tif(tmp_path_factory.mktemp("lc") / "esri.tif")


def run(tif, df=None, **kwargs):
    """Refuges over the synthetic landscape, forest taken from its land cover."""
    kwargs.setdefault("forest", "landcover")
    return rf.find_refuges(fx.register() if df is None else df, fx.villages(),
                           lc.LandCover([tif]), **kwargs)


@pytest.fixture(scope="module")
def result(tif):
    return run(tif)


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


def test_a_feeding_ground_in_fields_is_not_a_refuge(result):
    assert _at(result["refuges"], fx.FEEDING, km=0.4) is None
    assert any("outside forest" in c for c in result["caveats"])


def test_one_afternoon_of_sightings_is_a_passage_not_a_refuge(result):
    assert _at(result["refuges"], fx.PASSAGE) is None


def test_day_use_in_open_fields_is_not_a_refuge(result):
    assert _at(result["refuges"], fx.OPEN_DAY) is None


def test_a_forest_block_with_no_cover_on_the_ground_is_capped_and_flagged():
    # Notified forest can be bare on the ground: the cover check still caps it.
    row = {"Cover Share %": 10.0, "Days Used": 9, "Stability %": 100, "Direct Sightings": 6,
           "To Nearest Village (km)": 3.0, "Tree Patch (ha)": 0.0, "Days Since Last Use": 2,
           "Linked Raids": 0, "Group": "Mixed", "Bull-Type Share %": 0, "Nearest Village": "X"}
    assert rf._confidence(row, land_cover=True) == rf.POSSIBLE
    assert "Little cover" in rf._flags(row)
    assert "Verify on the ground" in rf._actions(row)


def test_damage_entered_next_morning_is_a_raid_not_a_refuge(result):
    assert _at(result["refuges"], fx.DAMAGE_FIELD) is None
    links = result["links"]
    # A's raids are entered at 08:00 the day after and still link to A.
    a = _at(result["refuges"], fx.REFUGE_A)
    mine = links[(links["Refuge"] == a["Refuge"]) & (links["Hour"] == 8)]
    assert len(mine) >= 10
    assert (mine["Night Of"] == mine["Date"].dt.normalize() - pd.Timedelta(days=1)).all()


def test_the_night_a_report_belongs_to():
    d = pd.Timestamp("2026-03-10")
    df = pd.DataFrame({
        "Date": [d] * 5, "Hour": [8, 21, None, 14, 2],
        "Latitude": [23.8] * 5, "Longitude": [81.0] * 5,
        "Crop Damage": [1, 1, 1, 0, 0], "House Damage": [0] * 5,
        "Injury": [0, 0, 0, 1, 1], "Death": [0] * 5,
    })
    out = rf.raids_and_nights(df)
    nights = dict(zip(out.index, out["Night Of"]))
    day_before = d - pd.Timedelta(days=1)
    assert nights == {0: day_before, 1: d, 2: day_before, 4: day_before}  # 3: a day injury


def test_signs_in_fields_are_not_a_refuge_but_signs_in_forest_are(tif, result):
    out = run(tif, require_raids=False)
    assert _at(out["refuges"], fx.PUGMARKS) is None
    signs = _at(result["refuges"], fx.FOREST_SIGNS)
    assert signs is not None and signs["Direct Sightings"] == 0 and signs["Indirect Signs"] == 8
    assert signs["Confidence"] == rf.PROBABLE        # never Confirmed on signs alone
    assert "Signs only" in signs["Flags"] and "Confirm by day" in signs["Recommended Action"]
    assert signs["Same-Night Raids"] == 4


def test_links_beat_the_chance_baseline(result):
    s = result["summary"]
    assert 0 <= s["chance_links"] < s["linked_raids"] / 2
    assert any("by coincidence" in line for line in rf.headlines(result))


def test_an_indirectly_reported_injury_is_a_raid_of_the_night_before():
    d = pd.Timestamp("2026-03-10")
    df = pd.DataFrame({
        "Date": [d, d], "Hour": [14, 14], "Latitude": [23.8] * 2, "Longitude": [81.0] * 2,
        "Sighting Type": ["Indirect", "Direct"], "Crop Damage": [0, 0], "House Damage": [0, 0],
        "Injury": [1, 1], "Death": [0, 0],
    })
    out = rf.raids_and_nights(df)
    assert list(out.index) == [0] and out["Night Of"].iloc[0] == d - pd.Timedelta(days=1)


def test_every_raid_is_accounted_for(result):
    s = result["summary"]
    assert s["linked_raids"] + s["area_raids"] + s["scattered_raids"] == s["night_raids"]


def test_raids_with_no_known_refuge_become_an_area_to_search(result):
    areas = result["areas"]
    lat, lon = fx.latlon(fx.UNKNOWN_RAIDS)
    d = rf._km(lat, lon, areas["Latitude"].to_numpy(float), areas["Longitude"].to_numpy(float))
    area = areas[d <= 1.0].iloc[0]
    assert area["Raids"] == 5 and area["Nights"] == 5
    assert area["To Nearest Known Refuge (km)"] > rf.LINK_KM
    # The forest block is the nearest patch big enough to hide a herd.
    assert area["Search Patch (ha)"] > 300 and area["Search Patch Extends Beyond Window"]
    assert area["Search Direction"] in ("W", "SW", "NW")
    assert 3.5 <= area["Search Distance (km)"] <= 4.6
    assert "search the" in area["Recommended Action"]
    assert any("no known refuge" in line for line in rf.headlines(result))


def test_a_raid_days_after_a_reported_day_still_links(tif):
    df = fx.register()
    lat, lon = fx.latlon((fx.REFUGE_A[0] + 2500, fx.REFUGE_A[1]))
    # Entered 2 Feb, so the night of 1 Feb; A was last reported on 28 Jan.
    late = {"Date": pd.Timestamp("2026-02-02"), "Hour": 8, "Latitude": lat, "Longitude": lon,
            "Division": "Test", "Range": "Test", "Beat": "Test", "Total Count": 3,
            "Crop Damage": 1, "House Damage": 0, "Injury": 0, "Death": 0}
    reg = pd.concat([df, pd.DataFrame([late])], ignore_index=True)
    wide = run(tif, reg, link_days=7)["links"]
    hit = wide[wide["Night Of"] == pd.Timestamp("2026-02-01")]
    assert len(hit) == 1 and hit.iloc[0]["Link"] == "4 day(s) apart"
    strict = run(tif, reg)["links"]           # the default: the night before or after
    assert not (strict["Night Of"] == pd.Timestamp("2026-02-01")).any()


def test_a_refuge_with_no_raids_linked_is_left_out(result):
    assert _at(result["refuges"], fx.QUIET) is None
    assert any("no raid linked" in c for c in result["caveats"])
    assert (result["refuges"]["Linked Raids"] > 0).all()


def test_quiet_refuges_can_still_be_listed_on_request(tif):
    out = run(tif, require_raids=False)
    quiet = _at(out["refuges"], fx.QUIET)
    assert quiet is not None and quiet["Linked Raids"] == 0


def test_every_report_inside_forest_counts_whatever_its_hour(result):
    a = _at(result["refuges"], fx.REFUGE_A)
    # 10 days x 3 rest-hour sightings plus the 06:00 and 17:00 ones.
    assert a["Reports"] == 50 and a["Direct Day Sightings"] == 30


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


def test_a_raid_long_after_the_refuge_was_used_is_not_linked(tif):
    df = fx.register()
    lat, lon = fx.latlon((fx.REFUGE_A[0] + 2500, fx.REFUGE_A[1]))
    far_night = {"Date": pd.Timestamp("2026-06-15"), "Hour": 22, "Latitude": lat,
                 "Longitude": lon, "Division": "Test", "Range": "Test", "Beat": "Test",
                 "Total Count": 3, "Crop Damage": 1, "House Damage": 0, "Injury": 0, "Death": 0}
    out = run(tif, pd.concat([df, pd.DataFrame([far_night])], ignore_index=True))
    assert not (out["links"]["Date"] == pd.Timestamp("2026-06-15")).any()


def test_without_land_cover_nothing_is_confirmed():
    out = rf.find_refuges(fx.register(), fx.villages(), None, forest="none")
    assert len(out["refuges"]) >= 2
    assert (out["refuges"]["Confidence"] != rf.CONFIRMED).all()
    assert any("No land cover" in c for c in out["caveats"])


def test_refuges_do_not_need_the_hour(tif):
    # Forest evidence counts at any hour, so a register without times still
    # finds the refuges; only the daytime-sighting count is lost.
    out = run(tif, fx.register().assign(Hour=pd.NA))
    a = _at(out["refuges"], fx.REFUGE_A)
    assert a is not None and a["Direct Day Sightings"] == 0


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


def test_an_unpublished_year_falls_back_to_the_latest_published(tif, tmp_path, monkeypatch):
    import shutil

    lat, lon = fx.latlon(fx.REFUGE_A)
    tile = lc.utm_tile(lat, lon)
    (tmp_path / "lc2022").mkdir()
    shutil.copy(tif, tmp_path / "lc2022" / f"{tile}_20220101-20230101.tif")
    monkeypatch.setenv(lc.URL_ENV, str(tmp_path / "lc{year}" / "{tile}_{year}0101-{next}0101.tif"))
    cover = lc.LandCover(remote_year=2024)
    window = cover.window(lat, lon)
    assert window is not None and window.source.startswith(f"{tile}_2022")
    assert cover.fallbacks == {tile: 2022} and not cover.failures


def test_no_published_year_is_one_clear_failure(tmp_path, monkeypatch):
    monkeypatch.setenv(lc.URL_ENV, str(tmp_path / "{year}" / "{tile}.tif"))
    cover = lc.LandCover(remote_year=2024)
    lat, lon = fx.latlon(fx.REFUGE_A)
    assert cover.window(lat, lon) is None and cover.window(lat, lon) is None
    assert len(cover.failures) == 1 and "no year from 2024 back to 2017" in cover.failures[0]


def test_the_brief_carries_the_raiding_refuges_only(result):
    from core.report import generate_html_report

    df = fx.register()
    html = generate_html_report(df, df["Date"].min(), df["Date"].max(), refuges=result)
    assert "<h2>Day Refuges" in html
    for ref in result["refuges"]["Refuge"]:
        assert f"<td>{ref}</td>" in html
    assert "no raid linked" in html
    assert "Elephant day refuges, linked raids and raid areas" in html  # the map
    assert "Day Refuges" not in generate_html_report(df, df["Date"].min(), df["Date"].max())


def test_utm_tile_names_match_esri_files():
    assert lc.utm_tile(23.8, 81.0) == "44Q"
    assert lc.utm_tile(24.5, 81.0) == "44R"
    assert lc.remote_url(2024, 23.8, 81.0).endswith("/lc2024/44Q_20240101-20250101.tif")


def test_field_files(result):
    gj = json.loads(rf.to_geojson(result["refuges"], result["links"]))
    kinds = {f["properties"]["kind"] for f in gj["features"]}
    assert kinds == {"refuge", "linked raid"}
    gj = json.loads(rf.to_geojson(result["refuges"], result["links"], result["areas"]))
    assert "raid area" in {f["properties"]["kind"] for f in gj["features"]}
    root = ET.fromstring(rf.to_kml(result["refuges"], result["areas"]))
    names = [n.text for n in root.findall(".//{http://www.opengis.net/kml/2.2}name")]
    assert any(n.endswith("(search)") for n in names if n)
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
    # The synthetic register lies outside the vendored forest boundaries.
    at.session_state["refuge_forest"] = "No forest test"
    at.run()
    assert not at.exception, at.exception
    assert any(m.label == "Forest refuges" for m in at.metric)
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


def test_generate_brief_in_the_app_includes_refuges():
    from streamlit.testing.v1 import AppTest

    from core import refuge_view

    df = fx.register()
    df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")
    app = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app, default_timeout=240)
    from types import SimpleNamespace

    at.session_state["fetched_register"] = SimpleNamespace(
        data=df.to_csv(index=False).encode(), name="register.csv", rows=len(df), pages=1,
        start_date="01/01/2026", end_date="01/03/2026", fetched_at="today")
    at.session_state[refuge_view.LAST_USED_KEY] = (
        ("none", None), ("none", rf.EPS_KM, rf.MIN_DAYS, rf.LINK_KM, rf.LINK_DAYS))
    at.run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Generate brief").click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert any(b.label == "Download brief (HTML)" for b in at.get("download_button"))


def test_a_raid_area_far_from_any_known_refuge_names_the_patch_to_search():
    row = {"Villages Raided": "Kukargoda", "To Nearest Known Refuge (km)": 40.0,
           "Search Patch (ha)": 850.0, "Search Patch Extends Beyond Window": True,
           "Search Distance (km)": 2.4, "Search Direction": "NE", "Casualties": 1}
    action = rf._area_action(row)
    assert "search the 850+ ha tree patch 2.4 km NE by day" in action
    assert "Kukargoda" in action and "night patrol" in action


def test_downloaded_map_has_the_basemap_and_few_enough_labels(result, monkeypatch):
    import re

    from core import map_export
    from core.refuge_report import MAP_LABELS, map_svg

    monkeypatch.setattr(map_export, "_basemap_data_uri",
                        lambda *a, **k: ("data:image/jpeg;base64,AAAA", "(c) tiles"))
    # Forty refuges crowded into a few kilometres must not print forty names.
    many = pd.concat([result["refuges"]] * 20, ignore_index=True)
    many["Refuge"] = [f"R{k:02d}" for k in range(len(many))]
    many["Latitude"] += np.linspace(0, 0.05, len(many))
    svg = map_svg(many, result["links"].iloc[0:0], areas=result["areas"], style_name="x")
    assert '<image href="data:image/jpeg;base64,AAAA"' in svg and "(c) tiles" in svg
    names = re.findall(r'paint-order="stroke"[^>]*>(R\d+|A\d+)<', svg)
    assert 0 < len(names) <= MAP_LABELS
    assert "Raid area, no known refuge" in svg


def test_downloaded_map_renders_to_png_and_pdf(result):
    from core import exports
    from core.refuge_report import map_svg

    svg = map_svg(result["refuges"], result["links"], areas=result["areas"])
    assert exports.to_png(svg, scale=1).startswith(b"\x89PNG")
    assert exports.to_pdf(svg).startswith(b"%PDF")
