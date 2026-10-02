"""Plotly figures for the conflict-trend views.

Pure functions from DataFrames to figures, so the dashboard only places
them and the tests can check what they say without a browser.

Colour rules, checked with the dataviz palette validator:

* Conflict type is ordered by severity, so it gets one warm hue stepped
  light (crop) to dark (death), not four unrelated hues. The map keeps
  its own Okabe-Ito colours, where points overlap and need hue contrast.
* Divisions are identities, so they get the Okabe-Ito sequence in a
  fixed order by name: filtering a division out does not repaint the
  rest. Lines are labelled at their end as well, since two of the hues
  sit close together under deuteranopia.
* Grid and axes are hairlines; text stays in ink, never a series colour.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.analytics import TREND_CATEGORIES

INK = "#16221c"
INK_MUTED = "#5b6b62"
GRID = "rgba(22,34,28,0.08)"
SURFACE = "#ffffff"
BRAND = "#1f5f3f"
BRAND_SOFT = "#a9c8b6"

# One hue, dark = worse. Validated as an ordinal ramp on white.
SEVERITY_RAMP: Dict[str, str] = {
    "Death": "#5C1406",
    "Injury": "#9A2A0C",
    "House": "#D04E14",
    "Crop": "#EE8A3C",
}
SEVERITY_LABELS: Dict[str, str] = {
    "Death": "Human fatality",
    "Injury": "Human injury",
    "House": "House / store damage",
    "Crop": "Crop or grain loss",
}

# Okabe-Ito, colourblind-safe. Six named slots, then "Other".
DIVISION_COLORS = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#D55E00"]
OTHER_COLOR = "#9AA5A0"

# Sequential heat for the seasonal grid, same hue as the severity ramp.
HEAT_SCALE = [[0.0, "#FBF1EA"], [0.35, "#F2B27E"], [0.7, "#C2410C"], [1.0, "#5C1406"]]

NIGHT_COLOR = "#1E3A5F"
DAY_COLOR = "#C9D6E3"


def _base_layout(fig: go.Figure, height: int, **extra) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(t=8, b=8, l=8, r=8),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=INK, size=12),
        hoverlabel=dict(bgcolor="#12211a", font_color="#eaf2ec", bordercolor="#12211a"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            title_text="", font=dict(color=INK_MUTED),
        ),
    )
    fig.update_layout(bargap=0.18)
    fig.update_layout(**extra)
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(
        gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK_MUTED),
        title_font=dict(color=INK_MUTED),
    )
    return fig


def _shade_recent(fig: go.Figure, months: pd.DatetimeIndex, recent_days: Optional[int],
                  as_of: Optional[pd.Timestamp]) -> None:
    """Shade the escalation window so the chart and the beat table agree."""
    if not recent_days or len(months) < 2:
        return
    end = pd.Timestamp(as_of) if as_of is not None else months.max() + pd.offsets.MonthEnd(0)
    start = end - pd.Timedelta(days=recent_days)
    if start <= months.min():
        return
    fig.add_vrect(
        x0=start, x1=end,
        fillcolor="#C2410C", opacity=0.06, line_width=0, layer="below",
    )
    fig.add_annotation(
        # Above the plot, so it cannot collide with the peak label.
        x=end, y=1, xref="x", yref="paper", xanchor="right", yanchor="bottom",
        text=f"Shaded: last {recent_days} days", showarrow=False,
        font=dict(size=11, color=INK_MUTED),
    )


def conflict_type_trend(
    monthly: pd.DataFrame,
    recent_days: Optional[int] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> go.Figure:
    """Stacked monthly conflict by type, with a three-month average line.

    Stacked rather than grouped: the total is the headline, and the
    darker segments at the bottom show whether the worst kinds of
    conflict are growing with it. The line shares the bars' axis and
    unit, so it is one scale, not two.
    """
    fig = go.Figure()
    months = monthly.index
    # Most severe at the base, where its height is read off the axis.
    for cat in TREND_CATEGORIES:
        fig.add_bar(
            x=months, y=monthly[cat], name=SEVERITY_LABELS[cat],
            marker=dict(color=SEVERITY_RAMP[cat], line=dict(color=SURFACE, width=1)),
            hovertemplate="%{x|%b %Y}<br>" + SEVERITY_LABELS[cat]
            + ": <b>%{y}</b><extra></extra>",
        )
    fig.add_scatter(
        x=months, y=monthly["Rolling Conflicts"], name="3-month average",
        mode="lines", line=dict(color=INK, width=2, shape="spline", smoothing=0.6),
        hovertemplate="%{x|%b %Y}<br>3-month average: <b>%{y:.1f}</b><extra></extra>",
    )

    if len(monthly):
        peak_month = monthly["Conflict Events"].idxmax()
        peak = int(monthly["Conflict Events"].max())
        if peak > 0:
            fig.add_annotation(
                x=peak_month, y=peak, text=f"Peak {peak}", showarrow=True,
                arrowhead=0, arrowcolor=INK_MUTED, ax=0, ay=-22,
                font=dict(size=11, color=INK),
            )
    _shade_recent(fig, months, recent_days, as_of)
    _base_layout(fig, 360, barmode="stack", hovermode="x unified",
                 legend_traceorder="normal")
    fig.update_yaxes(title_text="Conflict events")
    fig.update_xaxes(dtick="M2" if len(months) > 18 else "M1", tickformat="%b\n%Y")
    return fig


def conflict_rate_trend(monthly: pd.DataFrame, landscape_rate: float) -> go.Figure:
    """Monthly share of reports that were conflict, against the period average.

    The rate is what separates more conflict from more reporting: a
    month with twice the patrols has twice the sightings.
    """
    fig = go.Figure()
    fig.add_scatter(
        x=monthly.index, y=monthly["Conflict Rate %"], mode="lines+markers",
        name="Conflict rate", line=dict(color=BRAND, width=2),
        marker=dict(size=7, color=BRAND, line=dict(color=SURFACE, width=2)),
        fill="tozeroy", fillcolor="rgba(31,95,63,0.08)",
        customdata=np.stack([monthly["Conflict Events"], monthly["Sightings"]], axis=-1)
        if len(monthly) else None,
        hovertemplate="%{x|%b %Y}<br><b>%{y:.1f}%</b> of reports"
        "<br>%{customdata[0]} of %{customdata[1]} reports<extra></extra>",
        connectgaps=False,
    )
    if landscape_rate == landscape_rate:  # not NaN
        fig.add_hline(
            y=landscape_rate, line=dict(color=INK_MUTED, width=1, dash="dot"),
            annotation_text=f"Period average {landscape_rate:.0f}%",
            annotation_position="top left",
            annotation_font=dict(size=11, color=INK_MUTED),
        )
    _base_layout(fig, 280, showlegend=False)
    fig.update_yaxes(title_text="% of reports", rangemode="tozero", ticksuffix="%")
    return fig


def casualty_trend(monthly: pd.DataFrame) -> go.Figure:
    """People killed and injured per month."""
    fig = go.Figure()
    fig.add_bar(
        x=monthly.index, y=monthly["Human Deaths"], name="Killed",
        marker=dict(color=SEVERITY_RAMP["Death"], line=dict(color=SURFACE, width=1)),
        hovertemplate="%{x|%b %Y}<br>Killed: <b>%{y:.0f}</b><extra></extra>",
    )
    fig.add_bar(
        x=monthly.index, y=monthly["People Injured"], name="Injured",
        marker=dict(color=SEVERITY_RAMP["House"], line=dict(color=SURFACE, width=1)),
        hovertemplate="%{x|%b %Y}<br>Injured: <b>%{y:.0f}</b><extra></extra>",
    )
    _base_layout(fig, 280, barmode="stack", hovermode="x unified")
    fig.update_yaxes(title_text="People", rangemode="tozero", dtick=1
                     if float((monthly["Human Deaths"] + monthly["People Injured"]).max()
                              if len(monthly) else 0) <= 6 else None)
    return fig


def seasonal_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Year by calendar month, so a season reads down a column."""
    z = matrix.to_numpy(dtype=float)
    text = np.where(np.isnan(z), "", np.nan_to_num(z).astype(int).astype(str))
    fig = go.Figure(go.Heatmap(
        z=z, x=list(matrix.columns), y=[str(y) for y in matrix.index],
        colorscale=HEAT_SCALE, zmin=0, xgap=2, ygap=2,
        text=text, texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title=dict(text="Conflicts", font=dict(color=INK_MUTED)),
                      thickness=10, outlinewidth=0, tickfont=dict(color=INK_MUTED)),
        hovertemplate="%{x} %{y}<br>Conflict events: <b>%{z:.0f}</b><extra></extra>",
        hoverongaps=False,
    ))
    _base_layout(fig, 90 + 46 * max(len(matrix), 1))
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(side="top")
    return fig


def division_colors(divisions) -> Dict[str, str]:
    """Fixed colour per division name, independent of the current filter."""
    names = sorted(str(d) for d in divisions)
    return {
        name: DIVISION_COLORS[i] if i < len(DIVISION_COLORS) else OTHER_COLOR
        for i, name in enumerate(names)
    }


def division_trend(long: pd.DataFrame, colors: Dict[str, str]) -> go.Figure:
    """Monthly conflict per division, each line labelled at its end.

    Past six divisions the smallest are summed into "Other" rather than
    given a seventh hue no one can tell from the first.
    """
    totals = long.groupby("Division")["Conflict Events"].sum().sort_values(ascending=False)
    keep = list(totals.index[: len(DIVISION_COLORS)])
    if len(totals) > len(DIVISION_COLORS):
        keep = list(totals.index[: len(DIVISION_COLORS) - 1])
        long = long.assign(
            Division=long["Division"].where(long["Division"].isin(keep), "Other")
        ).groupby(["Month", "Division"], as_index=False)["Conflict Events"].sum()
        keep.append("Other")

    fig = go.Figure()
    for name in keep:
        series = long[long["Division"] == name].sort_values("Month")
        color = colors.get(name, OTHER_COLOR)
        smooth = series["Conflict Events"].rolling(3, min_periods=1).mean()
        fig.add_scatter(
            x=series["Month"], y=smooth, name=name, mode="lines",
            line=dict(color=color, width=2),
            customdata=series["Conflict Events"],
            hovertemplate=f"<b>{name}</b> %{{x|%b %Y}}<br>"
            "Conflicts: %{customdata} (3-mo avg %{y:.1f})<extra></extra>",
        )
        if len(series):
            fig.add_annotation(
                x=series["Month"].iloc[-1], y=float(smooth.iloc[-1]), text=name,
                showarrow=False, xanchor="left", xshift=6,
                font=dict(size=11, color=INK),
            )
    _base_layout(fig, 300, hovermode="x unified")
    fig.update_layout(margin=dict(t=8, b=8, l=8, r=90))
    fig.update_yaxes(title_text="Conflicts / month (3-mo avg)", rangemode="tozero")
    return fig


def escalation_dumbbell(beats: pd.DataFrame, limit: int = 12) -> go.Figure:
    """Prior window to recent window for the beats with the biggest rise.

    A dumbbell shows both counts and the gap between them, which is the
    claim being made; a bar of the ratio would hide that 3 vs 1 and 30
    vs 10 are not the same problem.
    """
    data = beats.assign(_change=beats["Recent Conflicts"] - beats["Prior Conflicts"])
    data = data.sort_values(["_change", "Recent Conflicts"], ascending=False).head(limit)
    data = data.iloc[::-1]  # biggest rise at the top
    labels = [f"{b} ({d})" for b, d in zip(data["Beat"], data["Division"])]

    fig = go.Figure()
    for label, prior, recent in zip(labels, data["Prior Conflicts"], data["Recent Conflicts"]):
        fig.add_shape(
            type="line", x0=prior, x1=recent, y0=label, y1=label,
            line=dict(color="#E8B9A0", width=4), layer="below",
        )
    fig.add_scatter(
        x=data["Prior Conflicts"], y=labels, mode="markers", name="Previous window",
        marker=dict(size=11, color="#B7C2BC", line=dict(color=SURFACE, width=2)),
        hovertemplate="%{y}<br>Previous window: <b>%{x}</b><extra></extra>",
    )
    fig.add_scatter(
        x=data["Recent Conflicts"], y=labels, mode="markers+text", name="Recent window",
        marker=dict(size=13, color=SEVERITY_RAMP["Injury"], line=dict(color=SURFACE, width=2)),
        text=[f"+{c}" for c in data["_change"]], textposition="middle right",
        textfont=dict(size=11, color=INK),
        hovertemplate="%{y}<br>Recent window: <b>%{x}</b><extra></extra>",
    )
    _base_layout(fig, 70 + 30 * max(len(data), 1))
    fig.update_xaxes(title_text="Conflict events", rangemode="tozero", showgrid=True,
                     gridcolor=GRID, title_font=dict(color=INK_MUTED))
    fig.update_yaxes(showgrid=False, type="category")
    return fig


def hourly_profile(hourly: pd.Series, night_start: int, night_end: int) -> go.Figure:
    """Conflict by hour, night hours in the darker colour."""
    hours = np.asarray(hourly.index, dtype=int)
    night = (hours >= night_start) | (hours < night_end)
    fig = go.Figure()
    for is_night, name, color in ((False, "Day", DAY_COLOR), (True, "Night", NIGHT_COLOR)):
        mask = night == is_night
        fig.add_bar(
            x=hours[mask], y=hourly.to_numpy()[mask], name=name,
            marker=dict(color=color, line=dict(color=SURFACE, width=1)),
            hovertemplate="%{x}:00<br>Conflict events: <b>%{y}</b><extra></extra>",
        )
    _base_layout(fig, 260, barmode="overlay", bargap=0.12)
    fig.update_xaxes(title_text="Hour of day (24h)", dtick=2, range=[-0.6, 23.6],
                     title_font=dict(color=INK_MUTED))
    fig.update_yaxes(title_text="Conflict events")
    return fig


def division_rate_bars(rates: pd.DataFrame, landscape_rate: float,
                       colors: Dict[str, str]) -> go.Figure:
    """Conflict rate per division as ranked horizontal bars."""
    data = rates.sort_values("Conflict Rate %")
    names = [str(n) for n in data.index]
    fig = go.Figure(go.Bar(
        x=data["Conflict Rate %"], y=names, orientation="h",
        marker=dict(color=[colors.get(n, OTHER_COLOR) for n in names]),
        text=[f"{v:.0f}%" for v in data["Conflict Rate %"]], textposition="outside",
        textfont=dict(color=INK, size=11), cliponaxis=False,
        customdata=np.stack([data["Conflict Events"], data["Sightings"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% of reports"
        "<br>%{customdata[0]} conflicts in %{customdata[1]} reports<extra></extra>",
    ))
    if landscape_rate == landscape_rate:
        fig.add_vline(
            x=landscape_rate, line=dict(color=INK_MUTED, width=1, dash="dot"),
            annotation_text=f"All divisions {landscape_rate:.0f}%",
            annotation_position="top", annotation_font=dict(size=11, color=INK_MUTED),
        )
    _base_layout(fig, 70 + 34 * max(len(data), 1), showlegend=False, bargap=0.45)
    fig.update_layout(margin=dict(t=24, b=8, l=8, r=40))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, ticksuffix="%", rangemode="tozero")
    fig.update_yaxes(showgrid=False)
    return fig
