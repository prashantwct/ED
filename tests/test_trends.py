"""The conflict-trend views: monthly breakdown, seasonal grid, windows, charts."""

import numpy as np
import pandas as pd

from core import charts
from core.analytics import (
    TREND_CATEGORIES,
    division_monthly_conflict,
    monthly_conflict_breakdown,
    seasonal_matrix,
    window_comparison,
)


def _frame(rows):
    base = {
        "Total Count": 1, "Crop Damage": 0, "Grain Damage": 0, "House Damage": 0,
        "Injury": 0, "Death": 0, "Division": "North",
    }
    out = pd.DataFrame([{**base, **row} for row in rows])
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def test_monthly_breakdown_zero_fills_quiet_months():
    df = _frame([
        {"Date": "2025-01-05", "Crop Damage": 1},
        {"Date": "2025-03-10", "Death": 1},
        {"Date": "2025-03-11"},
    ])
    out = monthly_conflict_breakdown(df)

    assert list(out.index.strftime("%Y-%m")) == ["2025-01", "2025-02", "2025-03"]
    assert out.loc["2025-02-01", "Conflict Events"] == 0
    assert out.loc["2025-03-01", "Death"] == 1
    assert out.loc["2025-03-01", "Conflict Rate %"] == 50.0
    # A month with no reports has no rate, not a zero rate.
    assert np.isnan(out.loc["2025-02-01", "Conflict Rate %"])


def test_monthly_breakdown_counts_each_incident_once():
    """A death with crop damage is one Death event, not one of each."""
    df = _frame([{"Date": "2025-01-05", "Death": 1, "Crop Damage": 1}])
    row = monthly_conflict_breakdown(df).iloc[0]
    assert row[TREND_CATEGORIES].sum() == 1 == row["Conflict Events"]
    assert row["Death"] == 1


def test_seasonal_matrix_separates_no_data_from_no_conflict():
    df = _frame([
        {"Date": "2024-11-02", "Crop Damage": 1},
        {"Date": "2025-02-02"},
    ])
    grid = seasonal_matrix(df)

    assert list(grid.index) == [2024, 2025]
    assert grid.loc[2024, "Nov"] == 1
    assert grid.loc[2025, "Jan"] == 0  # covered, quiet
    assert np.isnan(grid.loc[2024, "Jan"])  # before the data starts


def test_division_monthly_is_long_and_zero_filled():
    df = _frame([
        {"Date": "2025-01-05", "Crop Damage": 1, "Division": "North"},
        {"Date": "2025-02-05", "Crop Damage": 1, "Division": "South"},
    ])
    out = division_monthly_conflict(df)
    assert len(out) == 4
    assert out["Conflict Events"].sum() == 2


def test_window_comparison_needs_a_full_prior_window():
    df = _frame([{"Date": "2025-03-01", "Crop Damage": 1}])
    out = window_comparison(df, 30)
    assert out["conflicts_recent"] == 1
    assert np.isnan(out["conflicts_prior"])


def test_window_comparison_counts_both_windows():
    df = _frame([
        {"Date": "2025-01-01"},
        {"Date": "2025-01-15", "Crop Damage": 1},
        {"Date": "2025-02-20", "Crop Damage": 1},
        {"Date": "2025-02-25", "Injury": 1},
    ])
    out = window_comparison(df, 30, as_of=pd.Timestamp("2025-03-01"))
    assert out["conflicts_recent"] == 2
    assert out["conflicts_prior"] == 1
    assert out["casualties_recent"] == 1


def test_charts_build_from_real_shapes():
    df = _frame([
        {"Date": f"2025-{m:02d}-10", "Crop Damage": m % 2, "Death": int(m == 4),
         "Division": "North" if m % 3 else "South", "Hour": m}
        for m in range(1, 13)
    ])
    monthly = monthly_conflict_breakdown(df)
    colors = charts.division_colors(df["Division"].unique())

    fig = charts.conflict_type_trend(monthly, recent_days=90,
                                     as_of=pd.Timestamp("2025-12-31"))
    # Most severe first, so it sits at the base of the stack.
    assert fig.data[0].name == charts.SEVERITY_LABELS["Death"]
    charts.conflict_rate_trend(monthly, 50.0)
    charts.casualty_trend(monthly)
    charts.seasonal_heatmap(seasonal_matrix(df))
    charts.division_trend(division_monthly_conflict(df), colors)

    beats = pd.DataFrame({"Beat": ["A", "B"], "Division": ["North", "South"],
                          "Recent Conflicts": [9, 4], "Prior Conflicts": [2, 3]})
    dumbbell = charts.escalation_dumbbell(beats)
    assert list(dumbbell.data[1].text) == ["+1", "+7"]  # biggest rise on top


def test_division_colours_are_distinct_and_order_independent():
    """The app passes every division in the file, not the filtered ones,
    so a division keeps its colour when the filter changes."""
    forward = charts.division_colors(["Umaria", "Anuppur", "Shahdol"])
    backward = charts.division_colors(["Shahdol", "Anuppur", "Umaria"])
    assert forward == backward
    assert len(set(forward.values())) == 3


def _year_of_reports():
    return _frame([
        {"Date": f"2025-{m:02d}-10", "Crop Damage": m % 2, "Death": int(m == 4),
         "Injury": int(m == 9), "Division": "North" if m % 3 else "South<script>",
         "Hour": m, "Range": "R", "Beat": f"B{m % 4}", "Latitude": 23.0,
         "Longitude": 81.0}
        for m in range(1, 13)
    ])


def test_brief_draws_the_trend_charts_inline():
    from core.analytics import compute_is_night, compute_severity
    from core.report import generate_html_report

    df = _year_of_reports()
    df["Severity Score"] = compute_severity(df)
    df["Is_Night"] = compute_is_night(df)
    html = generate_html_report(df, df["Date"].min(), df["Date"].max())

    assert "Conflict Trends" in html
    assert "Conflict events per month by type" in html  # the stacked chart
    assert "class='heatmap'" in html
    assert "could not be drawn" not in html
    # Division names come from the CSV and reach SVG text.
    assert "<script>" not in html
    # Self-contained: no script tags and no remote chart library.
    assert "plotly" not in html.lower()


def test_brief_svg_charts_are_well_formed():
    import xml.etree.ElementTree as ET

    from core import report_charts

    df = _year_of_reports()
    monthly = monthly_conflict_breakdown(df)
    colors = charts.division_colors(df["Division"].unique())
    beats = pd.DataFrame({"Beat": ["A&B", "C"], "Division": ["N", "S"],
                          "Recent Conflicts": [9, 4], "Prior Conflicts": [2, 3]})
    rates = pd.DataFrame({"Sightings": [10, 5], "Conflict Events": [5, 1],
                          "Human Deaths": [0, 0], "Conflict Rate %": [50.0, 20.0]},
                         index=["North", "South"])
    hourly = pd.Series(range(24), index=range(24))
    for svg in (
        report_charts.conflict_type_trend_svg(monthly, 90, pd.Timestamp("2025-12-31")),
        report_charts.conflict_rate_svg(monthly, 40.0),
        report_charts.casualty_svg(monthly),
        report_charts.division_trend_svg(division_monthly_conflict(df), colors),
        report_charts.escalation_svg(beats),
        report_charts.hourly_svg(hourly, 18, 6),
        report_charts.division_rate_svg(rates, 40.0, colors),
    ):
        ET.fromstring(svg)  # raises on unescaped names or broken markup


def test_heatmap_leaves_months_without_data_blank():
    from core import report_charts

    df = _frame([{"Date": "2024-11-02", "Crop Damage": 1}, {"Date": "2025-02-02"}])
    html = report_charts.seasonal_heatmap_html(seasonal_matrix(df))
    assert html.count("heat-na") == 24 - 4  # Nov 2024 to Feb 2025 are covered
