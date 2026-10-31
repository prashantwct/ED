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

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.analytics import TREND_CATEGORIES
from core.config import KM_PER_DEG_LAT, KM_PER_DEG_LON_EQUATOR

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
    """Injuries and deaths by hour, night hours in the darker colour."""
    hours = np.asarray(hourly.index, dtype=int)
    night = (hours >= night_start) | (hours < night_end)
    fig = go.Figure()
    for is_night, name, color in ((False, "Day", DAY_COLOR), (True, "Night", NIGHT_COLOR)):
        mask = night == is_night
        fig.add_bar(
            x=hours[mask], y=hourly.to_numpy()[mask], name=name,
            marker=dict(color=color, line=dict(color=SURFACE, width=1)),
            hovertemplate="%{x}:00<br>Injuries and deaths: <b>%{y}</b><extra></extra>",
        )
    _base_layout(fig, 260, barmode="overlay", bargap=0.12)
    fig.update_xaxes(title_text="Hour of day (24h)", dtick=2, range=[-0.6, 23.6],
                     title_font=dict(color=INK_MUTED))
    fig.update_yaxes(title_text="Injuries and deaths")
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


# ---------------------------------------------------------------------------
# Day refuges
# ---------------------------------------------------------------------------
def raid_start_hours(links: pd.DataFrame, night_start: int, night_end: int) -> go.Figure:
    """When linked raids are entered, hour by hour.

    The question behind it is what time to put people out, so the night
    hours carry the darker colour and the busiest hour is called out
    rather than left for the reader to find.
    """
    counts = pd.Series(0, index=range(24), dtype=int)
    if len(links) and "Hour" in links:
        hours = pd.to_numeric(links["Hour"], errors="coerce").dropna().astype(int)
        counts = counts.add(hours[hours.between(0, 23)].value_counts(), fill_value=0).astype(int)

    index = np.asarray(counts.index, dtype=int)
    night = (index >= night_start) | (index < night_end)
    fig = go.Figure()
    for is_night, name, color in ((False, "Day", DAY_COLOR), (True, "Night", NIGHT_COLOR)):
        mask = night == is_night
        fig.add_bar(
            x=index[mask], y=counts.to_numpy()[mask], name=name,
            marker=dict(color=color, line=dict(color=SURFACE, width=1)),
            hovertemplate="%{x}:00<br>Raids: <b>%{y}</b><extra></extra>",
        )
    if int(counts.sum()):
        peak = int(counts.idxmax())
        fig.add_vline(
            x=peak, line=dict(color=INK, width=1, dash="dot"),
            annotation_text=f"Peak {peak:02d}:00", annotation_position="top",
            annotation_font=dict(size=11, color=INK),
        )
    _base_layout(fig, 210, barmode="overlay", bargap=0.12)
    fig.update_layout(margin=dict(t=26, b=8, l=8, r=8))
    fig.update_xaxes(title_text="Hour entered (24h)", dtick=4, range=[-0.6, 23.6],
                     title_font=dict(color=INK_MUTED))
    fig.update_yaxes(title_text="Raids", rangemode="tozero")
    return fig


def raids_by_month(links: pd.DataFrame) -> go.Figure:
    """Linked raids per month, stacked by what the raid cost.

    Same severity ramp as the trend charts, so a reader who has seen
    those reads this one without relearning the colours.
    """
    fig = go.Figure()
    if len(links) and "Night Of" in links:
        nights = pd.to_datetime(links["Night Of"], errors="coerce")
        frame = pd.DataFrame({
            "Month": nights.dt.to_period("M").dt.to_timestamp(),
            "Category": links.get("Category", pd.Series("Crop", index=links.index)),
        }).dropna(subset=["Month"])
        if len(frame):
            counts = frame.pivot_table(index="Month", columns="Category",
                                       aggfunc="size", fill_value=0)
            months = pd.date_range(counts.index.min(), counts.index.max(), freq="MS")
            counts = counts.reindex(months, fill_value=0)
            # Worst on top, so the eye lands on casualties first.
            for key in reversed(list(SEVERITY_RAMP)):
                if key not in counts:
                    continue
                fig.add_bar(
                    x=counts.index, y=counts[key].to_numpy(),
                    name=SEVERITY_LABELS[key], marker=dict(color=SEVERITY_RAMP[key]),
                    hovertemplate="%{x|%b %Y}<br>" + SEVERITY_LABELS[key]
                    + ": <b>%{y}</b><extra></extra>",
                )
    _base_layout(fig, 210, barmode="stack", bargap=0.2)
    fig.update_layout(margin=dict(t=8, b=8, l=8, r=8),
                      legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=10)))
    fig.update_xaxes(tickformat="%b<br>%Y", showgrid=False)
    fig.update_yaxes(title_text="Raids", rangemode="tozero")
    return fig


def place_time_heatmap(matrix: pd.DataFrame, step: str) -> go.Figure:
    """Places down the side, time steps across: when each place lit up.

    Rows come in the emerging hotspot table's order, worst first, so the
    eye starts on the place that needs a decision.
    """
    z = matrix.to_numpy(dtype=float)
    fmt = "%b %Y" if step == "Month" else "%d %b %Y"
    x = [pd.Timestamp(c).strftime(fmt) for c in matrix.columns]
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=list(matrix.index),
        colorscale=HEAT_SCALE, zmin=0, xgap=1, ygap=2,
        colorbar=dict(title=dict(text="Events", font=dict(color=INK_MUTED)),
                      thickness=10, outlinewidth=0, tickfont=dict(color=INK_MUTED)),
        hovertemplate=f"%{{y}}<br>{step} of %{{x}}<br>Events: <b>%{{z:.0f}}</b><extra></extra>",
    ))
    _base_layout(fig, 90 + 26 * max(len(matrix), 1))
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(type="category", tickangle=-45, nticks=24)
    return fig


CUBE_CAMERAS: Dict[str, dict] = {
    "Oblique": dict(eye=dict(x=1.55, y=-1.55, z=0.95), up=dict(x=0, y=0, z=1)),
    "From above": dict(eye=dict(x=0, y=-0.01, z=2.6), up=dict(x=0, y=1, z=0)),
    "Time vs east-west": dict(eye=dict(x=0, y=-2.6, z=0.05), up=dict(x=0, y=0, z=1)),
    "Time vs north-south": dict(eye=dict(x=2.6, y=0, z=0.05), up=dict(x=0, y=0, z=1)),
}
CONTEXT_COLOR = "#B8C2BC"


def space_time_cube(
    cube,
    table: pd.DataFrame,
    hot: pd.DataFrame,
    villages: Optional[pd.DataFrame] = None,
    camera: str = "Oblique",
    highlight_step: Optional[pd.Timestamp] = None,
    limit: int = 30,
    context: bool = True,
) -> go.Figure:
    """The cube itself: the ground on two axes, time rising on the third.

    Each place with a pattern is a column of bins, one per step with
    events, sized by count. A bin is drawn in its place's pattern colour
    in the steps it was a significant hotspot and pale grey otherwise, so
    a column reads bottom to top as the place's history: a New place is
    grey with a coloured cap, a Diminishing one coloured with a grey cap.
    A thin stalk runs the full height of every column so empty steps are
    visible as gaps rather than read as a shorter history.

    The floor carries hexagon outlines and the nearest villages, since a
    3D scene has no basemap to orient by. Horizontal axes are kilometres
    from the centre of the data at true ratio; time is stretched to fit.
    """
    from core.spacetime import NO_PATTERN, PATTERN_COLORS, PATTERN_ORDER, hex_polygon

    fig = go.Figure()
    named = table[table["Pattern"] != NO_PATTERN].head(limit)
    if cube.counts.empty or named.empty:
        return _base_layout(fig, 200)

    steps = cube.steps
    fmt = "%b %Y" if cube.step == "Month" else "%d %b %Y"
    lat0 = float(cube.cells["Centre Latitude"].mean())
    lon0 = float(cube.cells["Centre Longitude"].mean())
    lon_km = KM_PER_DEG_LON_EQUATOR * np.cos(np.radians(cube.reference_lat))

    def km(lat, lon):
        return (np.asarray(lon) - lon0) * lon_km, (np.asarray(lat) - lat0) * KM_PER_DEG_LAT

    def size(n):
        return np.clip(5 + 3.2 * np.sqrt(n), 5, 26)

    floor = -0.6
    tx, ty = km(named["Centre Latitude"], named["Centre Longitude"])
    # The scene is fitted to the named places, two cells round, at true
    # ground ratio. Fitting it to every event instead shrinks the columns
    # that matter to a sliver in the middle of the landscape.
    pad = cube.cell_km * 2
    x0, x1 = float(tx.min()) - pad, float(tx.max()) + pad
    y0, y1 = float(ty.min()) - pad, float(ty.max()) + pad

    def inside(x, y):
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

    # Context: every other cell with events, faint, so the named columns
    # are seen against the landscape's activity and not in a void.
    if context:
        others = cube.counts.drop(index=named["Cell"], errors="ignore")
        long = others.stack()
        long = long[long > 0]
        cells = cube.cells.loc[long.index.get_level_values(0)]
        x, y = km(cells["Centre Latitude"], cells["Centre Longitude"])
        keep = inside(x, y)
        if keep.any():
            fig.add_scatter3d(
                x=x[keep], y=y[keep],
                z=steps.get_indexer(long.index.get_level_values(1))[keep],
                mode="markers", name="Other places",
                marker=dict(size=2.5, color=CONTEXT_COLOR, opacity=0.35),
                hoverinfo="skip",
            )

    hot_x, hot_y, hot_z, hot_s, hot_c, hot_t = [], [], [], [], [], []
    pale_x, pale_y, pale_z, pale_s, pale_t = [], [], [], [], []
    for row in named.to_dict("records"):
        counts = cube.counts.loc[row["Cell"]]
        is_hot = hot.loc[row["Cell"]]
        x, y = km(row["Centre Latitude"], row["Centre Longitude"])
        where = str(row["Nearest Village"]).split(" (")[0] or str(row["Beats"]).split(",")[0]
        label = f"<b>{row['Place']} {row['Pattern']}</b>" + (f" - {where}" if where else "")
        for k, (step, n) in enumerate(counts.items()):
            # A step can be hot with no events of its own: the time window
            # carries the step before. Drawn at the smallest size, so a hot
            # run is not broken by a quiet week inside it.
            if n <= 0 and not is_hot.iloc[k]:
                continue
            tip = f"{label}<br>{step:{fmt}}: {int(n)} event{'s' if n != 1 else ''}"
            if n <= 0:
                tip += " (hot from the step before)"
            if is_hot.iloc[k]:
                hot_x.append(x); hot_y.append(y); hot_z.append(k); hot_s.append(n)
                hot_c.append(PATTERN_COLORS[row["Pattern"]])
                hot_t.append(tip + "<br>Significant hotspot this step")
            else:
                pale_x.append(x); pale_y.append(y); pale_z.append(k); pale_s.append(n)
                pale_t.append(tip + "<br>Not a hotspot this step")

    # Stalks and floor hexes, one trace per pattern so the legend
    # doubles as the colour key and clicking it hides a pattern.
    for pattern in PATTERN_ORDER:
        group = named[named["Pattern"] == pattern]
        if group.empty:
            continue
        color = PATTERN_COLORS[pattern]
        sx, sy, sz, hx, hy, hz = [], [], [], [], [], []
        for row in group.to_dict("records"):
            x, y = km(row["Centre Latitude"], row["Centre Longitude"])
            sx += [x, x, None]; sy += [y, y, None]; sz += [floor, len(steps) - 1, None]
            corners = hex_polygon(row["Centre Latitude"], row["Centre Longitude"],
                                  cube.cell_km, cube.reference_lat)
            corners.append(corners[0])
            cx, cy = km([c[1] for c in corners], [c[0] for c in corners])
            hx += list(cx) + [None]; hy += list(cy) + [None]; hz += [floor] * 7 + [None]
        fig.add_scatter3d(x=sx, y=sy, z=sz, mode="lines", name=pattern, legendgroup=pattern,
                          line=dict(color=color, width=2), hoverinfo="skip")
        fig.add_scatter3d(x=hx, y=hy, z=hz, mode="lines", legendgroup=pattern,
                          showlegend=False, line=dict(color=color, width=4), hoverinfo="skip")

    fig.add_scatter3d(
        x=pale_x, y=pale_y, z=pale_z, mode="markers", name="Events, not hot that step",
        marker=dict(size=size(np.array(pale_s, dtype=float)), color="#D9DED9", symbol="square",
                    opacity=0.75, line=dict(color="#8C9A93", width=1)),
        text=pale_t, hovertemplate="%{text}<extra></extra>",
    )
    fig.add_scatter3d(
        x=hot_x, y=hot_y, z=hot_z, mode="markers", name="Hot that step",
        marker=dict(size=size(np.array(hot_s, dtype=float)), color=hot_c, symbol="square",
                    opacity=1.0, line=dict(color=INK, width=1)),
        text=hot_t, hovertemplate="%{text}<extra></extra>",
    )

    # Place IDs on top of each column, where the eye lands first. Columns
    # in neighbouring cells would print their IDs over each other, so
    # each label steps up past any already placed within 1.5 cells.
    label_z: List[float] = []
    for i in range(len(named)):
        level = 0
        while any(abs(label_z[j] - (len(steps) - 0.3 + level * 1.1)) < 0.5
                  and np.hypot(tx[i] - tx[j], ty[i] - ty[j]) < 1.5 * cube.cell_km
                  for j in range(i)):
            level += 1
        label_z.append(len(steps) - 0.3 + level * 1.1)
    # Up to three labels per stack; past that a column of IDs reads as
    # noise and the hover names each bin anyway.
    keep_label = [round((z - (len(steps) - 0.3)) / 1.1) < 3 for z in label_z]
    fig.add_scatter3d(
        x=tx, y=ty, z=label_z, mode="text",
        text=[p if k else "" for p, k in zip(named["Place"], keep_label)], textfont=dict(color=INK, size=11), showlegend=False,
        hoverinfo="skip",
    )

    # Villages on the floor: only the one each place is nearest to. A
    # landscape holds thousands of centroids, and drawing them all buries
    # the hexes under a carpet of markers. Names repeat across districts,
    # so a village must also sit close to the place that named it.
    if villages is not None and not villages.empty:
        nearest = {str(v).split(" (")[0] for v in named["Nearest Village"] if v}
        pool = villages[villages["Village"].astype(str).isin(nearest)]
        vx, vy = km(pool["Latitude"], pool["Longitude"])
        reach = 3 * cube.cell_km
        close = np.zeros(len(pool), dtype=bool)
        for x, y in zip(tx, ty):
            close |= np.hypot(vx - x, vy - y) <= reach
        close &= inside(vx, vy)
        if close.any():
            names = pool["Village"].astype(str).to_numpy()[close]
            fig.add_scatter3d(
                x=vx[close], y=vy[close], z=[floor] * int(close.sum()),
                mode="markers+text", name="Villages",
                marker=dict(size=4, color=INK_MUTED, symbol="diamond"),
                text=names, textposition="bottom center",
                textfont=dict(color=INK_MUTED, size=10),
                hovertemplate="%{text}<extra>Village</extra>",
            )

    if highlight_step is not None and pd.Timestamp(highlight_step) in steps:
        k = steps.get_loc(pd.Timestamp(highlight_step))
        fig.add_mesh3d(
            x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[k] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
            color=BRAND, opacity=0.12, name=f"{steps[k]:{fmt}}", showlegend=True,
            hoverinfo="skip",
        )

    # True ground ratio; time gets about as much height as the wider side.
    span_x, span_y = x1 - x0, y1 - y0
    widest = max(span_x, span_y)
    ticks = list(range(0, len(steps), max(1, int(np.ceil(len(steps) / 8)))))
    axis = dict(backgroundcolor=SURFACE, gridcolor=GRID, zerolinecolor=GRID,
                showspikes=False, tickfont=dict(color=INK_MUTED, size=10),
                title_font=dict(color=INK_MUTED, size=11))
    fig.update_layout(
        height=640,
        margin=dict(t=8, b=8, l=8, r=8),
        paper_bgcolor=SURFACE,
        font=dict(color=INK, size=12),
        hoverlabel=dict(bgcolor="#12211a", font_color="#eaf2ec", bordercolor="#12211a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(color=INK_MUTED), itemsizing="constant"),
        # A new camera preset resets the view; anything else keeps the
        # angle the user has rotated to across reruns.
        uirevision=camera,
        scene=dict(
            xaxis=dict(axis, title="km east", range=[x0, x1]),
            yaxis=dict(axis, title="km north", range=[y0, y1]),
            zaxis=dict(axis, title="", tickvals=ticks,
                       ticktext=[f"{steps[t]:{fmt}}" for t in ticks],
                       range=[floor - 0.2, max(z for z, k in zip(label_z, keep_label) if k) + 0.4]),
            aspectmode="manual",
            aspectratio=dict(x=span_x / widest, y=span_y / widest, z=0.9),
            camera=CUBE_CAMERAS.get(camera, CUBE_CAMERAS["Oblique"]),
            dragmode="turntable",
        ),
    )
    return fig
