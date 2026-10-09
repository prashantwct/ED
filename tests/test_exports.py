"""Chart and map downloads: SVG, high-resolution PNG and vector PDF.

Every figure in both views exports from an SVG twin. These check the
conversions themselves, that each twin is a drawable SVG, and that each
view puts a Download menu under every chart and map.
"""

import io
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from core import exports, report_charts

APP = str(Path(__file__).resolve().parent.parent / "app.py")

BAR = report_charts.bars_svg(["Day", "Night"], [40, 120], "Night and day")


def test_standalone_unwraps_and_sizes_a_brief_map():
    wrapped = ('<div class="map-figure"><svg xmlns="http://www.w3.org/2000/svg" '
               'viewBox="0 0 900 548" width="100%" role="img"><rect width="9" height="9"/>'
               "</svg></div>")
    svg = exports.standalone(wrapped)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    root = ET.fromstring(svg)
    assert (root.get("width"), root.get("height")) == ("900", "548")


def test_png_is_four_times_the_drawn_size():
    png = exports.to_png(BAR)
    image = Image.open(io.BytesIO(png))
    assert image.format == "PNG"
    assert image.size == (report_charts.WIDTH * exports.PNG_SCALE, 260 * exports.PNG_SCALE)
    # Text and bars drew: the image is not a blank white canvas.
    assert len(set(image.convert("L").get_flattened_data())) > 10


def test_png_text_renders_from_the_shipped_fonts_alone():
    """Text naming a font the host lacks still draws: the host may have none."""
    assert list(exports.FONT_DIR.glob("*.ttf")), "assets/fonts is empty"
    label_only = ('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40">'
                  '<text x="10" y="28" font-size="20" font-family="Helvetica">Rice 166</text></svg>')
    pixels = Image.open(io.BytesIO(exports.to_png(label_only, scale=1))).convert("L")
    assert min(pixels.get_flattened_data()) < 100  # glyphs were drawn


def test_pdf_is_vector():
    pdf = exports.to_pdf(BAR)
    assert pdf.startswith(b"%PDF")
    assert b"/Subtype /Image" not in pdf and b"/Subtype/Image" not in pdf


def _register(rows=600, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 400, rows), "D")
    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Hour": rng.choice(list(range(18, 24)) + list(range(0, 6)) + [12], rows),
        "Latitude": 23.8 + rng.normal(0, 0.05, rows),
        "Longitude": 81.0 + rng.normal(0, 0.05, rows),
        "Division": rng.choice(["Anuppur", "Shahdol"], rows),
        "Range": rng.choice(["Kotma", "Beohari"], rows),
        "Beat": [f"Beat {i}" for i in rng.integers(1, 6, rows)],
        "Total Count": rng.integers(1, 6, rows),
        "Crop Damage": (rng.random(rows) < 0.4).astype(int),
        "House Damage": (rng.random(rows) < 0.08).astype(int),
        "Injury": (rng.random(rows) < 0.02).astype(int),
        "Death": (rng.random(rows) < 0.01).astype(int),
    })


def test_every_conflict_view_figure_has_a_drawable_twin():
    from core import charts
    from core.analytics import (compute_is_night, compute_kpis, compute_severity,
                                division_conflict_rate, division_monthly_conflict,
                                monthly_conflict_breakdown, monthly_trend,
                                night_day_comparison, seasonal_matrix,
                                severity_distribution)
    from core.data_loader import load_and_validate_csv
    from core.intelligence import management_brief

    df, _ = load_and_validate_csv(io.BytesIO(_register().to_csv(index=False).encode()))
    df["Severity Score"] = compute_severity(df)
    df["Is_Night"] = compute_is_night(df)
    monthly = monthly_conflict_breakdown(df)
    brief = management_brief(df)
    rate = compute_kpis(df)["conflict_rate"]
    colors = charts.division_colors(df["Division"].unique())
    bands, nd = severity_distribution(df), night_day_comparison(df)
    svgs = [
        report_charts.conflict_type_trend_svg(monthly, 90, None),
        report_charts.conflict_rate_svg(monthly, rate),
        report_charts.casualty_svg(monthly),
        report_charts.seasonal_heatmap_svg(seasonal_matrix(df)),
        report_charts.division_trend_svg(division_monthly_conflict(df), colors),
        report_charts.hourly_svg(brief["temporal"]["hourly"], 18, 6),
        report_charts.sightings_vs_conflict_svg(monthly_trend(df)),
        report_charts.division_rate_svg(division_conflict_rate(df), rate, colors),
        report_charts.bars_svg(bands["Band"].tolist(), bands["Count"].tolist(), "Severity"),
        report_charts.bars_svg(nd["Period"].tolist(), nd["Entries"].tolist(), "Night/day"),
    ]
    for svg in svgs:
        ET.fromstring(exports.standalone(svg))
        assert exports.to_pdf(svg).startswith(b"%PDF")


def _menus(at):
    return [b for b in at.get("download_button") if b.label == "SVG (vector)"]


def test_conflict_view_offers_a_download_under_every_figure(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from core.fetch import FetchResult

    data = _register().to_csv(index=False).encode()
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["fetched_register"] = FetchResult(
        data=data, name="register.csv", rows=600, pages=1, columns=[],
        start_date="2025-01-01", end_date="2026-02-05", fetched_at="2026-02-05")
    at.run()
    assert not at.exception, at.exception
    # Trends (5), escalation, hour, spatial map, monthly trend, division
    # rates, severity, night/day; the village map needs centroids, which
    # ship with the repo, so it is there too.
    assert len(_menus(at)) >= 12
    assert {b.label for b in at.get("download_button")} >= {
        "SVG (vector)", "PNG (4x resolution)", "PDF (vector)"}


def test_damage_view_offers_a_download_under_every_figure(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from core import damage, damage_view
    from damage_fixtures import crop_entry, export_csv, house_entry

    frame, notes = damage.load([
        (export_csv([crop_entry(i, value=1000 * i) for i in range(1, 6)]), "crop.csv"),
        (export_csv([house_entry(1), house_entry(2)]), "house.csv"),
    ])
    monkeypatch.setattr(damage_view, "sources", lambda: (frame, notes))
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["view"] = "Damage & compensation"
    at.run()
    assert not at.exception, at.exception
    # 12 charts and the map.
    assert len(_menus(at)) == 13
