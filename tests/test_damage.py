"""The household damage surveys and their view.

Run against invented surveys in the live forms' shape (damage_fixtures),
read the two ways they arrive: the CSV/ZIP export from the project page,
and entries from the API.
"""

from pathlib import Path

import pandas as pd
import pytest

from core import damage, damage_charts
from damage_fixtures import (
    INSIDE, NOWHERE, crop_entry, export_csv, export_zip, house_entry,
)


def _load(*files):
    return damage.load(files)


@pytest.fixture
def surveys():
    crops = [
        crop_entry(1, value=8000, remark="Not found in Gaj Rakshak"),
        crop_entry(2, value=12000, status="In process",
                   remark="Gaj rakshak me entry nahi tha"),
        crop_entry(3, value=3000, status="No", remark="Location incorrect in Gaj Rakshak."),
        # The same household as 1, hit again a month later.
        crop_entry(4, owner="Owner 1", mobile="9000000001", damage="2026-08-10",
                   survey="2026-08-12", value=4000, crops="Rice, Kodo",
                   fencing="Solar fence, Bamboo fence"),
        # Survey dated before the damage: the damage date is a typo.
        crop_entry(5, damage="2026-10-21", survey="2026-10-01", location=NOWHERE),
    ]
    houses = [house_entry(1), house_entry(2, elephants=12, light="Yes", value=4000,
                                         status="Not Applied")]
    return _load((export_zip("herd-crop-damage", crops), "crop.zip"),
                 (export_csv(houses), "form-1__herd-house-damage.csv"))


def test_reads_zip_and_csv_exports_into_one_frame(surveys):
    frame, notes = surveys
    assert len(frame) == 7
    assert frame["Kind"].value_counts().to_dict() == {"Crop": 5, "House": 2}
    crop = frame[frame["Entry"] == "crop-1"].iloc[0]
    assert crop["Damage Date"] == pd.Timestamp("2026-09-10")
    assert crop["Hour"] == 22 and crop["Days to Survey"] == 4
    assert crop["Estimated Loss"] == 8000 and crop["Elephants"] == 3
    assert crop["Village"] == "Testgaon"
    house = frame[frame["Entry"] == "house-1"].iloc[0]
    assert house["House Type"] == "Semi-Pucca"  # the form's "Semi- Pucca"
    assert house["Village"] == "Testgaon"  # title-cased to match the crop form
    assert house["Light On"] == False  # noqa: E712 - nullable boolean


def test_personal_data_never_leaves_the_reader(surveys):
    frame, _ = surveys
    text = frame.to_csv()
    for leak in ("Owner 1", "House Owner", "9000000001", "8000000001",
                 "surveyor@example.org", ".jpg", "Gaj rakshak me entry"):
        assert leak not in text
    assert not {"Remarks", "Owner", "Mobile"} & set(frame.columns)


def test_compensation_status_is_read_as_recorded(surveys):
    frame, _ = surveys
    status = frame.set_index("Entry")["Compensation"]
    assert status["crop-1"] == damage.NOT_APPLIED
    assert status["crop-2"] == damage.IN_PROCESS
    assert status["crop-3"] == damage.NOT_RECEIVED
    assert status["house-2"] == damage.NOT_APPLIED


def test_remarks_reduce_to_what_they_say_about_gaj_rakshak(surveys):
    frame, _ = surveys
    remark = frame.set_index("Entry")["Register Remark"]
    assert remark["crop-1"] == damage.REG_MISSING
    assert remark["crop-2"] == damage.REG_MISSING  # Hindi: "entry nahi tha"
    assert remark["crop-3"] == damage.REG_WRONG_PLACE
    assert remark["crop-4"] == ""
    assert damage.register_remark("2nd time gaj rakshak pe entry huea hai") == damage.REG_MENTIONED


def test_a_household_hit_twice_is_recognised_without_keeping_its_name(surveys):
    frame, _ = surveys
    repeat = frame.set_index("Entry")["Repeat Household"]
    assert repeat["crop-1"] and repeat["crop-4"]
    assert not repeat["crop-2"]
    assert damage.summary(frame)["repeat_households"] == 1


def test_a_damage_date_after_the_survey_is_blanked_and_noted(surveys):
    frame, notes = surveys
    typo = frame.set_index("Entry").loc["crop-5"]
    assert pd.isna(typo["Damage Date"]) and pd.isna(typo["Days to Survey"])
    assert any("dated before the damage" in n for n in notes)


def test_points_get_their_beat_or_the_nearest_within_reach(surveys):
    frame, _ = surveys
    placed = frame.set_index("Entry")
    assert placed.loc["crop-1", "Beat"] == "Kaseru"
    assert placed.loc["crop-1", "Beat Distance (km)"] == 0
    assert placed.loc["crop-5", "Beat"] == "Unknown"  # hundreds of km from any beat


def test_the_same_entry_twice_counts_once():
    entries = [crop_entry(1)]
    frame, notes = _load((export_csv(entries), "a.csv"), (export_csv(entries), "b.csv"))
    assert len(frame) == 1
    assert any("twice" in n for n in notes)


def test_a_sighting_register_is_refused_with_a_reason():
    register = b"Date,Latitude,Longitude,Division,Range,Beat\n01/01/2026,23.1,81.1,A,B,C\n"
    with pytest.raises(damage.DamageDataError, match="does not look like"):
        _load((register, "register.csv"))


def test_summary_and_headlines(surveys):
    frame, _ = surveys
    s = damage.summary(frame)
    assert s["loss"] == frame["Estimated Loss"].sum()
    assert s["not_applied"] == 4  # crop-1, crop-4, crop-5, house-2
    assert s["register_missing"] == 2 and s["register_wrong"] == 1
    assert s["house_lone"] == 1
    text = " ".join(damage.headlines(frame))
    assert "No compensation claim made in 3 of 5 crop cases and 1 of 2 house cases" in text


def test_follow_up_queue_is_the_unclaimed_cases_largest_first(surveys):
    frame, _ = surveys
    queue = damage.follow_up(frame, as_of=pd.Timestamp("2026-10-05"))
    assert set(queue["Entry"]) == {"crop-1", "crop-4", "crop-5", "house-2"}
    assert queue["Estimated Loss"].is_monotonic_decreasing
    assert queue.set_index("Entry").loc["crop-1", "Days Since Damage"] == 25
    assert "Household" not in queue.columns


def test_multi_answer_questions_count_under_each_answer(surveys):
    frame, _ = surveys
    crops = damage.crop_table(frame).set_index("Crop")["Cases"]
    assert crops["Rice"] == 5 and crops["Kodo"] == 1
    fences = damage.exploded_counts(frame[frame["Kind"] == "Crop"], "Fencing")
    assert fences.set_index("Fencing").loc["Solar fence", "Cases"] == 1


def test_register_match_needs_both_place_and_time(surveys):
    frame, _ = surveys
    register = pd.DataFrame({
        "Date": pd.to_datetime(["2026-09-11", "2026-08-01"]),
        # crop-1's field, the day after; and the same field, far off in time.
        "Latitude": [INSIDE["latitude"] + 0.001, INSIDE["latitude"]],
        "Longitude": [INSIDE["longitude"], INSIDE["longitude"]],
    })
    match = damage.match_register(frame, register, radius_km=1.0, days=3)
    by_entry = pd.Series(match.to_numpy(), index=frame["Entry"])
    assert by_entry["crop-1"]  # 0.1 km, 1 day
    assert not by_entry["crop-4"]  # same place, 9 days from the 1 Aug report
    assert not by_entry["crop-5"]  # no damage date


def test_charts_build(surveys):
    frame, _ = surveys
    pipeline = damage.compensation_pipeline(frame)
    damage_charts.compensation_bars(pipeline, "Cases")
    loss = damage_charts.compensation_bars(pipeline, "Loss")
    assert max(max(t.x) for t in loss.data) < 1  # in lakh
    damage_charts.monthly_loss(damage.monthly(frame))
    damage_charts.by_kind_bars(damage.herd_sizes(frame), "Herd", damage.HERD_LABELS)
    damage_charts.by_kind_bars(damage.survey_lag(frame), "Delay", damage.LAG_LABELS)
    damage_charts.hour_profile(damage.hourly(frame))
    damage_charts.area_comparison(damage.area_check(frame))
    damage_charts.ranked_bars(damage.crop_table(frame), "Crop")


def test_rupee_formats():
    assert damage.rupees(3743431) == "Rs 37.4 lakh"
    assert damage.rupees(2600) == "Rs 2,600"
    assert damage.lakh(3743431) == "37.4 lakh"
    assert damage.lakh(2600) == "2,600"


# --- The app ---------------------------------------------------------------
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_damage_view_is_separate_from_the_conflict_dashboard(surveys, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from core import damage_view

    frame, notes = surveys
    monkeypatch.setattr(damage_view, "sources", lambda: (frame, notes))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at = next(b for b in at.button if "damage surveys" in b.label).click().run()

    assert not at.exception, at.exception
    titles = " ".join(m.value for m in at.markdown)
    assert "Compensation" in titles and "Gaj Rakshak gap" in titles
    # None of the conflict dashboard's sections.
    assert "Beat priorities" not in titles and "Movement hotspots" not in titles
    assert any("No claim made" == m.label for m in at.metric)


def test_damage_view_waits_for_an_export(monkeypatch):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["view"] = "Damage & compensation"
    at.run()
    assert not at.exception, at.exception
    assert any("Upload a herd-crop-damage" in i.value for i in at.info)


# --- The downloadable study -----------------------------------------------
def test_html_study_is_self_contained_vector_and_de_identified(surveys):
    import re
    import xml.etree.ElementTree as ET

    from core import damage_report

    frame, notes = surveys
    html = damage_report.build_html(frame, notes)
    svgs = re.findall(r"<svg.*?</svg>", html, re.S)
    assert len(svgs) >= 8  # every chart and the map
    for svg in svgs:
        ET.fromstring(svg)  # well-formed: names reach SVG text escaped
    # Nothing fetched when it opens: no scripts, images or linked files.
    assert "<script" not in html and "<img" not in html
    assert "src=" not in html and "<link" not in html
    for leak in ("Owner 1", "9000000001", "surveyor@example.org", ".jpg"):
        assert leak not in html
    assert "Follow-up queue" in html and "Gaj Rakshak gap" in html


def test_pdf_study_draws_charts_as_vectors(surveys):
    from core import damage_report

    frame, notes = surveys
    pdf = damage_report.build_pdf(frame, notes)
    assert pdf.startswith(b"%PDF")
    # Charts and map are drawing operators, not embedded pictures.
    assert b"/Subtype /Image" not in pdf and b"/Subtype/Image" not in pdf
    assert b"Owner 1" not in pdf and b"9000000001" not in pdf


def test_map_outlines_are_clipped_to_the_frame():
    from core.damage_report import _clip

    square = [(-50, -50), (150, -50), (150, 150), (-50, 150)]
    clipped = _clip(square, 100, 100)
    assert all(0 <= x <= 100 and 0 <= y <= 100 for x, y in clipped)
    assert sorted(set(clipped)) == [(0, 0), (0, 100), (100, 0), (100, 100)]


def test_long_tables_move_to_an_appendix(surveys, monkeypatch):
    from core import damage_report

    monkeypatch.setattr(damage_report, "TOP_ROWS", 2)
    frame, notes = surveys
    html = damage_report.build_html(frame, notes)
    assert "Appendix" in html and "the 2 largest of 4 case(s)" in html


def test_view_offers_html_and_pdf_downloads(surveys, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from core import damage_view

    frame, notes = surveys
    monkeypatch.setattr(damage_view, "sources", lambda: (frame, notes))
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["view"] = "Damage & compensation"
    at.run()
    at = next(b for b in at.button if b.label == "Prepare HTML and PDF").click().run()
    assert not at.exception, at.exception
    labels = [b.label for b in at.get("download_button")]
    assert "Download HTML" in labels and "Download PDF (A4)" in labels
