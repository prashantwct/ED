"""Static SVG trend charts for the HTML brief.

The brief is emailed and printed, so it cannot lean on Plotly's
JavaScript: these draw the same views as ``core.charts`` as inline SVG,
in the same colours, with a ``<title>`` on each mark so a browser still
shows the figure on hover. Every chart has a text fallback in the brief's
tables, so nothing here is the only place a number appears.

Anything drawn from the CSV (division and beat names) is escaped.
"""

from __future__ import annotations

import math
from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.analytics import TREND_CATEGORIES
from core.charts import (
    BRAND,
    DAY_COLOR,
    INK,
    INK_MUTED,
    NIGHT_COLOR,
    OTHER_COLOR,
    SEVERITY_LABELS,
    SEVERITY_RAMP,
)

WIDTH = 880
# Charts that sit two to a row are drawn at half width, so their text
# prints at the same size as the full-width ones rather than shrinking.
HALF_WIDTH = 430
GRID = "#e8ece9"
FONT = "font-family=\"Segoe UI, Arial, sans-serif\""

# Heat steps for the seasonal grid, light to dark, same hue as the ramp.
HEAT_STEPS = ["#FBF1EA", "#F6D2B5", "#F2B27E", "#E07B3C", "#C2410C", "#8A2A0A", "#5C1406"]
# The same steps in the brand green, for presence rather than conflict:
# where elephants were is not, by itself, bad news.
PRESENCE_STEPS = ["#EEF5F0", "#D3E6DA", "#A9C8B6", "#7FAE93", "#4F8C6A", "#2E6B4A", "#1A4A32"]
# Daylight hours on the activity clock; night keeps the brief's navy.
DAYLIGHT = "#C9952B"
# Group types: bulls in an earth brown, breeding herds in the brand green,
# groups nobody could sex in greys.
GROUP_COLORS = {
    "Lone bull": "#7A4A2A", "Bull party": "#A8683C", "Family herd": BRAND,
    "Mixed / unsexed": "#8C9A93", "Unrecorded": "#C5CDC8",
}


def _nice_max(value: float) -> Tuple[float, float]:
    """A round axis maximum and tick step covering ``value``."""
    if value <= 0 or not math.isfinite(value):
        return 1.0, 1.0
    raw = value / 4
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    if value <= 6 and step < 1:
        step = 1.0
    top = math.ceil(value / step) * step
    return float(top), float(step)


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


class _Plot:
    """Plot area geometry and the axis furniture around it."""

    def __init__(self, height: int, y_max: float, left: int = 44, right: int = 16,
                 top: int = 34, bottom: int = 34, width: int = WIDTH,
                 suffix: str = "") -> None:
        self.width, self.height = width, height
        self.left, self.right, self.top, self.bottom = left, right, top, bottom
        self.y_top, self.y_step = _nice_max(y_max)
        self.suffix = suffix
        self.parts: List[str] = []

    @property
    def inner_w(self) -> float:
        return self.width - self.left - self.right

    @property
    def inner_h(self) -> float:
        return self.height - self.top - self.bottom

    def y(self, value: float) -> float:
        return self.top + self.inner_h * (1 - value / self.y_top)

    def grid(self, label: str = "") -> None:
        ticks = np.arange(0, self.y_top + self.y_step / 2, self.y_step)
        for tick in ticks:
            y = self.y(tick)
            self.parts.append(
                f'<line x1="{self.left}" x2="{self.width - self.right}" y1="{y:.1f}" '
                f'y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
                f'<text x="{self.left - 6}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-size="11" fill="{INK_MUTED}">{_fmt(tick)}{self.suffix}</text>'
            )
        if label:
            self.parts.append(
                f'<text x="{self.left}" y="{self.top - 22}" font-size="11" '
                f'fill="{INK_MUTED}">{escape(label)}</text>'
            )

    def legend(self, items: Sequence[Tuple[str, str, str]]) -> None:
        """Legend row at the top right: (kind, colour, label)."""
        x = self.width - self.right
        chunks = []
        for kind, color, label in reversed(items):
            text_w = 6.2 * len(label)
            x -= text_w
            chunks.append(
                f'<text x="{x:.1f}" y="{self.top - 18}" font-size="11" '
                f'fill="{INK_MUTED}">{escape(label)}</text>'
            )
            x -= 16
            if kind == "line":
                chunks.append(
                    f'<line x1="{x:.1f}" x2="{x + 12:.1f}" y1="{self.top - 22}" '
                    f'y2="{self.top - 22}" stroke="{color}" stroke-width="2"/>'
                )
            else:
                chunks.append(
                    f'<rect x="{x + 1:.1f}" y="{self.top - 27}" width="10" height="10" '
                    f'rx="2" fill="{color}"/>'
                )
            x -= 12
        self.parts.extend(chunks)

    def svg(self, label: str) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} '
            f'{self.height}" role="img" aria-label="{escape(label)}" {FONT}>'
            + "".join(self.parts)
            + "</svg>"
        )


def _month_ticks(plot: _Plot, months: pd.DatetimeIndex, x_of) -> None:
    slots = max(int(plot.inner_w // 46), 1)  # room for one label per ~46px
    every = max(1, math.ceil(len(months) / slots))
    every = next(step for step in (1, 2, 3, 4, 6, 12, 24) if step >= every) \
        if every <= 24 else every
    for i, month in enumerate(months):
        if i % every:
            continue
        label = month.strftime("%b")
        if month.month == 1 or i == 0:
            label += f" {month:%y}"
        plot.parts.append(
            f'<text x="{x_of(i):.1f}" y="{plot.height - plot.bottom + 16}" '
            f'text-anchor="middle" font-size="10.5" fill="{INK_MUTED}">{label}</text>'
        )


def _band_x(plot: _Plot, n: int):
    """Centre of the i-th of n equal bands across the plot."""
    band = plot.inner_w / max(n, 1)
    return band, (lambda i: plot.left + band * (i + 0.5))


def _recent_start_index(months: pd.DatetimeIndex, recent_days: Optional[int],
                        as_of: Optional[pd.Timestamp]) -> Optional[int]:
    if not recent_days or len(months) < 2:
        return None
    end = pd.Timestamp(as_of) if as_of is not None else months[-1] + pd.offsets.MonthEnd(0)
    start = end - pd.Timedelta(days=recent_days)
    if start <= months[0]:
        return None
    # The first month whose end falls inside the window.
    month_ends = months + pd.offsets.MonthEnd(0)
    inside = np.flatnonzero(month_ends > start)
    return int(inside[0]) if len(inside) else None


def conflict_type_trend_svg(monthly: pd.DataFrame, recent_days: Optional[int] = None,
                            as_of: Optional[pd.Timestamp] = None) -> str:
    """Stacked monthly conflict by type, three-month average over it."""
    months = pd.DatetimeIndex(monthly.index)
    plot = _Plot(300, float(monthly["Conflict Events"].max()), top=44)
    band, x_of = _band_x(plot, len(months))
    bar_w = max(band * 0.72, 1.5)

    shade_from = _recent_start_index(months, recent_days, as_of)
    if shade_from is not None:
        x0 = plot.left + band * shade_from
        plot.parts.append(
            f'<rect x="{x0:.1f}" y="{plot.top}" width="{plot.width - plot.right - x0:.1f}" '
            f'height="{plot.inner_h:.1f}" fill="#C2410C" fill-opacity="0.07"/>'
            f'<text x="{plot.width - plot.right - 4}" y="{plot.top + 12}" text-anchor="end" '
            f'font-size="10.5" fill="{INK_MUTED}">last {recent_days} days</text>'
        )
    plot.grid("Conflict events per month")

    for i, month in enumerate(months):
        base = 0.0
        for cat in TREND_CATEGORIES:
            value = float(monthly[cat].iloc[i])
            if value <= 0:
                continue
            y_hi, y_lo = plot.y(base + value), plot.y(base)
            plot.parts.append(
                f'<rect x="{x_of(i) - bar_w / 2:.1f}" y="{y_hi:.1f}" width="{bar_w:.1f}" '
                f'height="{max(y_lo - y_hi - 1, 0.8):.1f}" fill="{SEVERITY_RAMP[cat]}">'
                f"<title>{month:%b %Y}: {SEVERITY_LABELS[cat]} {value:.0f}</title></rect>"
            )
            base += value

    points = " ".join(
        f"{x_of(i):.1f},{plot.y(float(v)):.1f}"
        for i, v in enumerate(monthly["Rolling Conflicts"])
    )
    plot.parts.append(
        f'<polyline points="{points}" fill="none" stroke="{INK}" stroke-width="2" '
        f'stroke-linejoin="round"/>'
    )
    peak_i = int(np.argmax(monthly["Conflict Events"].to_numpy()))
    peak = float(monthly["Conflict Events"].iloc[peak_i])
    if peak > 0:
        plot.parts.append(
            f'<text x="{x_of(peak_i):.1f}" y="{plot.y(peak) - 5:.1f}" text-anchor="middle" '
            f'font-size="10.5" font-weight="600" fill="{INK}">{peak:.0f}</text>'
        )
    _month_ticks(plot, months, x_of)
    plot.legend(
        [("box", SEVERITY_RAMP[c], SEVERITY_LABELS[c]) for c in TREND_CATEGORIES]
        + [("line", INK, "3-month avg")]
    )
    return plot.svg("Conflict events per month by type")


def _line_chart(months: pd.DatetimeIndex, series: Sequence[Tuple[str, str, np.ndarray]],
                height: int, label: str, suffix: str = "",
                reference: Optional[Tuple[float, str]] = None,
                end_labels: bool = False, width: int = WIDTH) -> str:
    values = [v for _, _, s in series for v in s if v == v]
    y_max = max(values + ([reference[0]] if reference else []) + [0.0])
    plot = _Plot(height, y_max, width=width, suffix=suffix,
                 right=96 if end_labels else 16)
    _, x_of = _band_x(plot, len(months))
    plot.grid(label)
    if reference and reference[0] == reference[0]:
        y = plot.y(reference[0])
        plot.parts.append(
            f'<line x1="{plot.left}" x2="{plot.width - plot.right}" y1="{y:.1f}" '
            f'y2="{y:.1f}" stroke="{INK_MUTED}" stroke-width="1" stroke-dasharray="2 3"/>'
            f'<text x="{plot.width - plot.right}" y="{y - 4:.1f}" text-anchor="end" '
            f'font-size="10.5" fill="{INK_MUTED}">{escape(reference[1])}</text>'
        )
    for name, color, data in series:
        # Break the line at a missing month rather than bridge it.
        runs, current = [], []
        for i, v in enumerate(data):
            if v == v:
                current.append(f"{x_of(i):.1f},{plot.y(float(v)):.1f}")
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        for run in runs:
            plot.parts.append(
                f'<polyline points="{" ".join(run)}" fill="none" stroke="{color}" '
                f'stroke-width="2" stroke-linejoin="round"><title>{escape(name)}</title>'
                "</polyline>"
            )
    if end_labels:
        _end_labels(plot, x_of(len(months) - 1) + 6, [
            (name, plot.y(float(data[-1])))
            for name, _, data in series if len(data) and data[-1] == data[-1]
        ])
    _month_ticks(plot, months, x_of)
    return plot.svg(label)


def _end_labels(plot: _Plot, x: float, labels: List[Tuple[str, float]],
                gap: float = 13.0) -> None:
    """Line-end labels, nudged apart so close lines stay readable."""
    placed: List[Tuple[str, float]] = []
    for name, y in sorted(labels, key=lambda item: item[1]):
        if placed and y - placed[-1][1] < gap:
            y = placed[-1][1] + gap
        placed.append((name, y))
    for name, y in placed:
        plot.parts.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" font-size="10.5" fill="{INK}">'
            f"{escape(name)}</text>"
        )


def conflict_rate_svg(monthly: pd.DataFrame, period_rate: float) -> str:
    return _line_chart(
        pd.DatetimeIndex(monthly.index),
        [("Conflict rate", BRAND, monthly["Conflict Rate %"].to_numpy(dtype=float))],
        220, "Share of reports that were conflict", suffix="%",
        reference=(period_rate, f"period average {period_rate:.0f}%"),
        width=HALF_WIDTH,
    )


def casualty_svg(monthly: pd.DataFrame) -> str:
    months = pd.DatetimeIndex(monthly.index)
    totals = (monthly["Human Deaths"] + monthly["People Injured"]).to_numpy(dtype=float)
    plot = _Plot(220, float(totals.max()), width=HALF_WIDTH)
    band, x_of = _band_x(plot, len(months))
    bar_w = max(band * 0.72, 1.5)
    plot.grid("People per month")
    for i, month in enumerate(months):
        base = 0.0
        for col, color, name in (("Human Deaths", SEVERITY_RAMP["Death"], "killed"),
                                 ("People Injured", SEVERITY_RAMP["House"], "injured")):
            value = float(monthly[col].iloc[i])
            if value <= 0:
                continue
            y_hi, y_lo = plot.y(base + value), plot.y(base)
            plot.parts.append(
                f'<rect x="{x_of(i) - bar_w / 2:.1f}" y="{y_hi:.1f}" width="{bar_w:.1f}" '
                f'height="{max(y_lo - y_hi - 1, 0.8):.1f}" fill="{color}">'
                f"<title>{month:%b %Y}: {value:.0f} {name}</title></rect>"
            )
            base += value
    _month_ticks(plot, months, x_of)
    plot.legend([("box", SEVERITY_RAMP["Death"], "Killed"),
                 ("box", SEVERITY_RAMP["House"], "Injured")])
    return plot.svg("People killed and injured per month")


def seasonal_heatmap_html(matrix: pd.DataFrame) -> str:
    """Year by month as a shaded table: prints cleanly and copies as text."""
    values = matrix.to_numpy(dtype=float)
    top = np.nanmax(values) if np.isfinite(values).any() else 0.0
    head = "".join(f"<th class='num'>{escape(str(c))}</th>" for c in matrix.columns)
    rows = []
    for year, row in matrix.iterrows():
        cells = []
        for value in row:
            if value != value:
                cells.append("<td class='heat heat-na'></td>")
                continue
            step = 0 if top <= 0 else min(int(value / top * (len(HEAT_STEPS) - 1) + 0.5),
                                         len(HEAT_STEPS) - 1)
            ink = "#fff" if step >= 4 else INK
            cells.append(
                f"<td class='heat num' style='background:{HEAT_STEPS[step]};color:{ink};'>"
                f"{value:.0f}</td>"
            )
        rows.append(f"<tr><th>{escape(str(year))}</th>{''.join(cells)}</tr>")
    return (
        f"<table class='heatmap'><thead><tr><th></th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def division_trend_svg(long: pd.DataFrame, colors: Dict[str, str]) -> str:
    """Three-month average conflict per division, labelled at the line end."""
    months = pd.DatetimeIndex(sorted(long["Month"].unique()))
    totals = long.groupby("Division")["Conflict Events"].sum().sort_values(ascending=False)
    names = list(totals.index[:6])
    if len(totals) > 6:
        names = list(totals.index[:5])
        long = long.assign(
            Division=long["Division"].where(long["Division"].isin(names), "Other")
        ).groupby(["Month", "Division"], as_index=False)["Conflict Events"].sum()
        names.append("Other")
    series = []
    for name in names:
        data = (long[long["Division"] == name].set_index("Month")["Conflict Events"]
                .reindex(months, fill_value=0).rolling(3, min_periods=1).mean())
        series.append((str(name), colors.get(str(name), OTHER_COLOR), data.to_numpy(float)))
    return _line_chart(months, series, 240, "Conflict events per month, 3-month average",
                       end_labels=True)


def escalation_svg(beats: pd.DataFrame, limit: int = 12) -> str:
    """Prior window to recent window per escalating beat, biggest rise first."""
    data = beats.assign(_change=beats["Recent Conflicts"] - beats["Prior Conflicts"])
    data = data.sort_values(["_change", "Recent Conflicts"], ascending=False).head(limit)
    row_h, left = 24, 190
    height = 40 + row_h * len(data) + 26
    plot = _Plot(height, float(data["Recent Conflicts"].max()), left=left, top=40,
                 bottom=26)
    x_top = plot.y_top

    def x(value: float) -> float:
        return left + plot.inner_w * value / x_top

    ticks = np.arange(0, x_top + plot.y_step / 2, plot.y_step)
    for tick in ticks:
        plot.parts.append(
            f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{plot.top - 6}" '
            f'y2="{height - plot.bottom}" stroke="{GRID}"/>'
            f'<text x="{x(tick):.1f}" y="{height - plot.bottom + 15}" text-anchor="middle" '
            f'font-size="10.5" fill="{INK_MUTED}">{_fmt(tick)}</text>'
        )
    for i, row in enumerate(data.to_dict("records")):
        y = plot.top + row_h * i + row_h / 2
        prior, recent = float(row["Prior Conflicts"]), float(row["Recent Conflicts"])
        label = f"{row['Beat']} ({row['Division']})"
        plot.parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="{INK}">{escape(label)}</text>'
            f'<line x1="{x(prior):.1f}" x2="{x(recent):.1f}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="#E8B9A0" stroke-width="4" stroke-linecap="round"/>'
            f'<circle cx="{x(prior):.1f}" cy="{y:.1f}" r="5" fill="#B7C2BC" stroke="#fff" '
            f'stroke-width="1.5"><title>Previous window: {prior:.0f}</title></circle>'
            f'<circle cx="{x(recent):.1f}" cy="{y:.1f}" r="6" fill="{SEVERITY_RAMP["Injury"]}" '
            f'stroke="#fff" stroke-width="1.5"><title>Recent window: {recent:.0f}</title>'
            "</circle>"
            f'<text x="{x(recent) + 10:.1f}" y="{y + 4:.1f}" font-size="10.5" '
            f'fill="{INK}">+{int(row["_change"])}</text>'
        )
    plot.legend([("box", "#B7C2BC", "Previous window"),
                 ("box", SEVERITY_RAMP["Injury"], "Recent window")])
    return plot.svg("Escalating beats, previous window against recent window")


def hourly_svg(hourly: pd.Series, night_start: int, night_end: int) -> str:
    hours = np.asarray(hourly.index, dtype=int)
    counts = hourly.to_numpy(dtype=float)
    plot = _Plot(200, float(counts.max()))
    band, x_of = _band_x(plot, 24)
    bar_w = band * 0.78
    plot.grid("Injuries and deaths by hour of day")
    for i, (hour, value) in enumerate(zip(hours, counts)):
        night = hour >= night_start or hour < night_end
        if value > 0:
            plot.parts.append(
                f'<rect x="{x_of(i) - bar_w / 2:.1f}" y="{plot.y(value):.1f}" '
                f'width="{bar_w:.1f}" height="{plot.y(0) - plot.y(value):.1f}" rx="1.5" '
                f'fill="{NIGHT_COLOR if night else DAY_COLOR}">'
                f"<title>{hour:02d}:00 - {value:.0f} injuries and deaths</title></rect>"
            )
        if i % 2 == 0:
            plot.parts.append(
                f'<text x="{x_of(i):.1f}" y="{plot.height - plot.bottom + 16}" '
                f'text-anchor="middle" font-size="10.5" fill="{INK_MUTED}">{hour:02d}</text>'
            )
    plot.legend([("box", DAY_COLOR, "Day"), ("box", NIGHT_COLOR, "Night")])
    return plot.svg("Injuries and deaths by hour of day")


def division_rate_svg(rates: pd.DataFrame, overall: float, colors: Dict[str, str]) -> str:
    data = rates.sort_values("Conflict Rate %", ascending=False)
    row_h, left = 26, 150
    height = 30 + row_h * len(data) + 26
    plot = _Plot(height, float(max(data["Conflict Rate %"].max(), overall
                                   if overall == overall else 0)),
                 left=left, top=30, bottom=26, right=48)

    def x(value: float) -> float:
        return left + plot.inner_w * value / plot.y_top

    for tick in np.arange(0, plot.y_top + plot.y_step / 2, plot.y_step):
        plot.parts.append(
            f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{plot.top - 4}" '
            f'y2="{height - plot.bottom}" stroke="{GRID}"/>'
            f'<text x="{x(tick):.1f}" y="{height - plot.bottom + 15}" text-anchor="middle" '
            f'font-size="10.5" fill="{INK_MUTED}">{_fmt(tick)}%</text>'
        )
    for i, (name, row) in enumerate(data.iterrows()):
        y = plot.top + row_h * i
        rate = float(row["Conflict Rate %"])
        plot.parts.append(
            f'<text x="{left - 10}" y="{y + row_h / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="{INK}">{escape(str(name))}</text>'
            f'<rect x="{left}" y="{y + 6:.1f}" width="{max(x(rate) - left, 1):.1f}" '
            f'height="{row_h - 12}" rx="3" fill="{colors.get(str(name), OTHER_COLOR)}">'
            f"<title>{escape(str(name))}: {rate:.1f}% of {int(row['Sightings'])} reports"
            "</title></rect>"
            f'<text x="{x(rate) + 6:.1f}" y="{y + row_h / 2 + 4:.1f}" font-size="10.5" '
            f'fill="{INK}">{rate:.0f}%</text>'
        )
    if overall == overall:
        plot.parts.append(
            f'<line x1="{x(overall):.1f}" x2="{x(overall):.1f}" y1="{plot.top - 4}" '
            f'y2="{height - plot.bottom}" stroke="{INK_MUTED}" stroke-dasharray="2 3"/>'
            f'<text x="{x(overall):.1f}" y="{plot.top - 10}" text-anchor="middle" '
            f'font-size="10.5" fill="{INK_MUTED}">all divisions {overall:.0f}%</text>'
        )
    return plot.svg("Conflict rate by division")


def bars_svg(categories: Sequence[str], values: Sequence[float], label: str,
             y_title: str = "Reports", color: str = BRAND, width: int = WIDTH) -> str:
    """Plain vertical bars, each labelled with its value."""
    plot = _Plot(260, float(max(values) if len(values) else 0), width=width)
    band, x_of = _band_x(plot, len(categories))
    bar_w = band * 0.6
    plot.grid(y_title)
    for i, (cat, value) in enumerate(zip(categories, values)):
        value = float(value)
        if value > 0:
            plot.parts.append(
                f'<rect x="{x_of(i) - bar_w / 2:.1f}" y="{plot.y(value):.1f}" '
                f'width="{bar_w:.1f}" height="{plot.y(0) - plot.y(value):.1f}" rx="2" '
                f'fill="{color}"><title>{escape(str(cat))}: {value:,.0f}</title></rect>'
                f'<text x="{x_of(i):.1f}" y="{plot.y(value) - 5:.1f}" text-anchor="middle" '
                f'font-size="10.5" fill="{INK}">{value:,.0f}</text>'
            )
        plot.parts.append(
            f'<text x="{x_of(i):.1f}" y="{plot.height - plot.bottom + 16}" '
            f'text-anchor="middle" font-size="10.5" fill="{INK_MUTED}">{escape(str(cat))}</text>'
        )
    return plot.svg(label)


def sightings_vs_conflict_svg(trend: pd.DataFrame) -> str:
    """Monthly reports and conflict events: two lines on one count axis."""
    months = pd.DatetimeIndex(pd.to_datetime(trend.index.astype(str)))
    return _line_chart(
        months,
        [("Sightings", BRAND, trend["Sightings"].to_numpy(float)),
         ("Conflict events", SEVERITY_RAMP["House"], trend["Conflict Events"].to_numpy(float))],
        260, "Reports per month", end_labels=True,
    )


def seasonal_heatmap_svg(matrix: pd.DataFrame, width: int = WIDTH, left: int = 56,
                         steps: Sequence[str] = HEAT_STEPS,
                         label: str = "Conflict events by month and year") -> str:
    """Rows by columns as shaded cells, the count in each.

    Drawn for year by calendar month (conflict), and for division by
    month (presence, in ``PRESENCE_STEPS``), with ``left`` room for the
    longer row names.
    """
    HEAT_STEPS = list(steps)  # noqa: N806 - the ramp this grid is drawn in
    values = matrix.to_numpy(dtype=float)
    top = np.nanmax(values) if np.isfinite(values).any() else 0.0
    head, row_h = 26, 34
    cell_w = (width - left - 10) / max(len(matrix.columns), 1)
    height = head + row_h * len(matrix) + 10
    parts = []
    for j, month in enumerate(matrix.columns):
        parts.append(f'<text x="{left + cell_w * (j + 0.5):.1f}" y="18" text-anchor="middle" '
                     f'font-size="11" fill="{INK_MUTED}">{escape(str(month))}</text>')
    for i, (year, row) in enumerate(matrix.iterrows()):
        y = head + i * row_h
        parts.append(f'<text x="{left - 10}" y="{y + row_h / 2 + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="{INK}">{escape(str(year))}</text>')
        for j, value in enumerate(row):
            x = left + j * cell_w
            if value != value:
                parts.append(f'<rect x="{x + 1:.1f}" y="{y + 1}" width="{cell_w - 2:.1f}" '
                             f'height="{row_h - 2}" fill="#fafbfa"/>')
                continue
            step = 0 if top <= 0 else min(int(value / top * (len(HEAT_STEPS) - 1) + 0.5),
                                         len(HEAT_STEPS) - 1)
            ink = "#ffffff" if step >= 4 else INK
            parts.append(
                f'<rect x="{x + 1:.1f}" y="{y + 1}" width="{cell_w - 2:.1f}" height="{row_h - 2}" '
                f'fill="{HEAT_STEPS[step]}"><title>{escape(str(matrix.columns[j]))} '
                f'{escape(str(year))}: {value:.0f}</title></rect>'
                f'<text x="{x + cell_w / 2:.1f}" y="{y + row_h / 2 + 4:.1f}" text-anchor="middle" '
                f'font-size="11" fill="{ink}">{value:.0f}</text>'
            )
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{escape(label)}" {FONT}>'
            + "".join(parts) + "</svg>")


# ---------------------------------------------------------------------------
# The elephants
# ---------------------------------------------------------------------------
def _polar(cx: float, cy: float, r: float, angle: float) -> Tuple[float, float]:
    return cx + r * math.cos(angle), cy + r * math.sin(angle)


def diel_clock_svg(hourly: pd.Series, night_start: int, night_end: int,
                   size: int = HALF_WIDTH) -> str:
    """Direct sightings round a 24-hour clock, midnight at the top.

    A clock rather than a row of bars because the day wraps: the dusk to
    dawn run that matters reads as one arc here, where a bar chart cuts
    it in two at midnight.
    """
    counts = hourly.reindex(range(24), fill_value=0).to_numpy(dtype=float)
    top = counts.max() if counts.max() > 0 else 1.0
    height = size
    cx, cy = size / 2, height / 2 + 6
    r0, r1 = size * 0.11, size * 0.40
    step = 2 * math.pi / 24

    def angle(hour: float) -> float:
        return hour * step - math.pi / 2

    parts = []
    # The night as a shaded sector behind the bars.
    span = (night_end - night_start) % 24
    a0, a1 = angle(night_start), angle(night_start + span)
    xo0, yo0 = _polar(cx, cy, r1 + 8, a0)
    xo1, yo1 = _polar(cx, cy, r1 + 8, a1)
    large = 1 if span > 12 else 0
    parts.append(f'<path d="M{cx:.1f},{cy:.1f} L{xo0:.1f},{yo0:.1f} A{r1 + 8:.1f},{r1 + 8:.1f} '
                 f'0 {large} 1 {xo1:.1f},{yo1:.1f} Z" fill="{NIGHT_COLOR}" opacity="0.07"/>')
    for share in (0.5, 1.0):
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r0 + (r1 - r0) * share:.1f}" '
                     f'fill="none" stroke="{GRID}"/>')
    for hour, value in enumerate(counts):
        if value <= 0:
            continue
        r = r0 + (r1 - r0) * value / top
        b0, b1 = angle(hour + 0.1), angle(hour + 0.9)
        x0, y0 = _polar(cx, cy, r0, b0)
        x1, y1 = _polar(cx, cy, r, b0)
        x2, y2 = _polar(cx, cy, r, b1)
        x3, y3 = _polar(cx, cy, r0, b1)
        night = hour >= night_start or hour < night_end
        parts.append(
            f'<path d="M{x0:.1f},{y0:.1f} L{x1:.1f},{y1:.1f} A{r:.1f},{r:.1f} 0 0 1 '
            f'{x2:.1f},{y2:.1f} L{x3:.1f},{y3:.1f} A{r0:.1f},{r0:.1f} 0 0 0 {x0:.1f},{y0:.1f} Z" '
            f'fill="{NIGHT_COLOR if night else DAYLIGHT}"><title>{hour:02d}:00-'
            f'{(hour + 1) % 24:02d}:00: {value:.0f} sightings</title></path>')
    for hour in range(0, 24, 3):
        x, y = _polar(cx, cy, r1 + 20, angle(hour))
        parts.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="11" '
                     f'fill="{INK_MUTED}">{hour:02d}</text>')
    parts.append(f'<text x="{cx:.1f}" y="{cy - 2:.1f}" text-anchor="middle" font-size="15" '
                 f'font-weight="600" fill="{INK}">{counts.sum():,.0f}</text>'
                 f'<text x="{cx:.1f}" y="{cy + 13:.1f}" text-anchor="middle" font-size="10" '
                 f'fill="{INK_MUTED}">sightings</text>')
    parts.append(f'<rect x="8" y="8" width="10" height="10" rx="2" fill="{DAYLIGHT}"/>'
                 f'<text x="22" y="17" font-size="11" fill="{INK_MUTED}">Day</text>'
                 f'<rect x="56" y="8" width="10" height="10" rx="2" fill="{NIGHT_COLOR}"/>'
                 f'<text x="70" y="17" font-size="11" fill="{INK_MUTED}">Night '
                 f'{night_start:02d}:00-{night_end:02d}:00</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {height + 12}" '
            f'role="img" aria-label="Direct sightings by hour of day" {FONT}>'
            + "".join(parts) + "</svg>")


def herd_composition_svg(composition: pd.DataFrame, width: int = HALF_WIDTH) -> str:
    """Sightings by kind of group, the share that raided filled in darker."""
    data = composition[composition["Sightings"] > 0]
    row_h, left = 34, 112
    height = 30 + row_h * len(data) + 26
    plot = _Plot(height, float(data["Sightings"].max()) if len(data) else 1.0,
                 left=left, top=30, bottom=26, right=86, width=width)

    def x(value: float) -> float:
        return left + plot.inner_w * value / plot.y_top

    for tick in np.arange(0, plot.y_top + plot.y_step / 2, plot.y_step):
        plot.parts.append(
            f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{plot.top - 4}" '
            f'y2="{height - plot.bottom}" stroke="{GRID}"/>'
            f'<text x="{x(tick):.1f}" y="{height - plot.bottom + 15}" text-anchor="middle" '
            f'font-size="10.5" fill="{INK_MUTED}">{_fmt(tick)}</text>')
    for i, row in enumerate(data.to_dict("records")):
        y = plot.top + row_h * i
        name = str(row["Group Type"])
        seen, raids = float(row["Sightings"]), float(row["Conflict Events"])
        color = GROUP_COLORS.get(name, OTHER_COLOR)
        plot.parts.append(
            f'<text x="{left - 10}" y="{y + row_h / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="{INK}">{escape(name)}</text>'
            f'<rect x="{left}" y="{y + 7:.1f}" width="{max(x(seen) - left, 1):.1f}" '
            f'height="{row_h - 14}" rx="3" fill="{color}" opacity="0.28">'
            f'<title>{escape(name)}: {seen:,.0f} reports</title></rect>'
            f'<rect x="{left}" y="{y + 7:.1f}" width="{max(x(raids) - left, 0):.1f}" '
            f'height="{row_h - 14}" rx="3" fill="{color}">'
            f'<title>{escape(name)}: {raids:,.0f} conflict events</title></rect>'
            f'<text x="{x(seen) + 6:.1f}" y="{y + row_h / 2 + 4:.1f}" font-size="10.5" '
            f'fill="{INK}">{float(row["Damage Rate %"]):.0f}% raided</text>')
    plot.parts.append(
        f'<text x="{width - plot.right}" y="{plot.top - 14}" text-anchor="end" font-size="11" '
        f'fill="{INK_MUTED}">solid: conflict events, pale: all reports</text>')
    return plot.svg("Reports and conflict by kind of group")


def range_use_svg(used: pd.DataFrame, width: int = WIDTH) -> str:
    """Beats with elephants each month, those used for the first time darker."""
    months = pd.DatetimeIndex(used.index)
    plot = _Plot(230, float(used["Beats used"].max()) if len(used) else 1.0, width=width)
    band, x_of = _band_x(plot, len(months))
    bar_w = band * 0.66
    plot.grid("Beats with elephant reports")
    for i, row in enumerate(used.to_dict("records")):
        total, first = float(row["Beats used"]), float(row["First used"])
        if total > 0:
            plot.parts.append(
                f'<rect x="{x_of(i) - bar_w / 2:.1f}" y="{plot.y(total):.1f}" width="{bar_w:.1f}" '
                f'height="{plot.y(0) - plot.y(total):.1f}" rx="2" fill="#A9C8B6">'
                f'<title>{months[i]:%b %Y}: {total:.0f} beats, {first:.0f} for the first time'
                '</title></rect>')
        if first > 0:
            plot.parts.append(
                f'<rect x="{x_of(i) - bar_w / 2:.1f}" y="{plot.y(first):.1f}" width="{bar_w:.1f}" '
                f'height="{plot.y(0) - plot.y(first):.1f}" rx="2" fill="{BRAND}"/>')
    _month_ticks(plot, months, x_of)
    plot.legend([("box", "#A9C8B6", "Beats with reports"), ("box", BRAND, "First time")])
    return plot.svg("Beats with elephant reports each month")


def damage_season_svg(months: pd.DataFrame, width: int = HALF_WIDTH) -> str:
    """Crop and house or store damage by calendar month, stacked."""
    totals = months.sum(axis=1).to_numpy(dtype=float)
    plot = _Plot(240, float(totals.max()) if len(totals) else 1.0, width=width)
    band, x_of = _band_x(plot, len(months))
    bar_w = band * 0.68
    plot.grid("Damage reports")
    for i, (month, row) in enumerate(months.iterrows()):
        base = 0.0
        for column, color in (("Crop", SEVERITY_RAMP["Crop"]), ("House", SEVERITY_RAMP["House"])):
            value = float(row[column])
            if value > 0:
                plot.parts.append(
                    f'<rect x="{x_of(i) - bar_w / 2:.1f}" y="{plot.y(base + value):.1f}" '
                    f'width="{bar_w:.1f}" height="{plot.y(base) - plot.y(base + value):.1f}" '
                    f'fill="{color}"><title>{escape(str(month))}: {value:.0f} '
                    f'{column.lower()} damage</title></rect>')
            base += value
        plot.parts.append(
            f'<text x="{x_of(i):.1f}" y="{plot.height - plot.bottom + 16}" text-anchor="middle" '
            f'font-size="10.5" fill="{INK_MUTED}">{escape(str(month))[:1]}</text>')
    plot.legend([("box", SEVERITY_RAMP["Crop"], "Crop"),
                 ("box", SEVERITY_RAMP["House"], "House / store")])
    return plot.svg("Damage by calendar month")
