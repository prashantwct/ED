"""The damage study as a downloadable report: one HTML file and one PDF.

Both are vector throughout. Every chart and the map is drawn here as SVG:
the HTML carries it inline, and the PDF embeds the same SVG as vector
drawing operators (svglib -> ReportLab), not as a screenshot. So both
stay sharp at any zoom and print cleanly at A4 or A3, and the figures in
the two files cannot drift apart. There is deliberately no raster
basemap behind the map: forest boundary outlines give the place, and a
tile image would be the one blurry thing on the page.

The report carries what the damage view shows and nothing personal: it
is built from the frame ``core.damage`` returns, which has already
dropped names, phone numbers, photos, emails and remark text.
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from html import escape
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core import damage
from core.charts import BRAND, INK, INK_MUTED
from core.damage_charts import KIND_COLORS, STATUS_COLORS

GRID = "#e3e9e5"
FONT = 'font-family="Helvetica, Arial, sans-serif"'
TITLE = "Elephant Damage and Compensation Study"
NIGHT_BAND = "#F2C94C"


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------
def _nice(value: float, ticks: int = 4) -> Tuple[float, float]:
    """A round axis top and step covering ``value``."""
    if value <= 0 or not math.isfinite(value):
        return 1.0, 1.0
    raw = value / ticks
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    if value <= ticks and step < 1:
        step = 1.0
    return math.ceil(value / step) * step, step


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def _text(x, y, text, size=11, anchor="start", color=INK, weight="normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{color}">{escape(str(text))}</text>')


def _svg(width: int, height: int, parts: Sequence[str], label: str) -> str:
    # Explicit width and height as well as the viewBox: svglib sizes the
    # PDF drawing from them, and browsers scale it with CSS either way.
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)}" {FONT}>'
            + "".join(parts) + "</svg>")


def _legend(items: Sequence[Tuple[str, str]], x: float, y: float) -> List[str]:
    """Swatch-and-word legend, left to right from (x, y)."""
    parts = []
    for label, color in items:
        parts.append(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" '
                     f'fill="{color}"/>')
        parts.append(_text(x + 15, y, label, 10.5, color=INK_MUTED))
        x += 15 + 6.0 * len(label) + 18
    return parts


def stacked_hbars(rows: Sequence[str], segments: Sequence[Tuple[str, str, Sequence[float]]],
                  axis_title: str, width: int = 860, value_fmt: Callable[[float], str] = _fmt,
                  label: str = "") -> str:
    """One bar per row, split into labelled segments, share printed inside."""
    left, right, top, row_h = 70, 20, 34, 34
    height = top + row_h * len(rows) + 40
    totals = np.sum([np.asarray(v, float) for _, _, v in segments], axis=0)
    x_top, step = _nice(float(totals.max()) if len(totals) else 0)
    inner = width - left - right
    xs = lambda v: left + inner * v / x_top  # noqa: E731
    parts = _legend([(n, c) for n, c, v in segments if np.any(np.asarray(v) > 0)], left, 16)
    for tick in np.arange(0, x_top + step / 2, step):
        parts.append(f'<line x1="{xs(tick):.1f}" x2="{xs(tick):.1f}" y1="{top - 4}" '
                     f'y2="{height - 36}" stroke="{GRID}"/>')
        parts.append(_text(xs(tick), height - 22, _fmt(tick), 10, "middle", INK_MUTED))
    parts.append(_text(left + inner / 2, height - 6, axis_title, 10.5, "middle", INK_MUTED))
    for i, row in enumerate(rows):
        y = top + i * row_h + 6
        parts.append(_text(left - 8, y + 15, row, 11, "end"))
        start = 0.0
        for name, color, values in segments:
            value = float(values[i])
            if value <= 0:
                continue
            x0, x1 = xs(start), xs(start + value)
            parts.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0 - 1.5, 0.8):.1f}" '
                         f'height="22" fill="{color}"><title>{escape(row)} - {escape(name)}: '
                         f'{escape(value_fmt(value))}</title></rect>')
            share = value / totals[i] * 100 if totals[i] else 0
            if x1 - x0 > 34:
                parts.append(_text((x0 + x1) / 2, y + 15, f"{share:.0f}%", 10.5, "middle",
                                   "#ffffff", "bold"))
            start += value
        parts.append(_text(xs(start) + 6, y + 15, value_fmt(totals[i]), 10.5))
    return _svg(width, height, parts, label or axis_title)


def vbars(categories: Sequence[str], series: Sequence[Tuple[str, str, Sequence[float]]],
          y_title: str, mode: str = "group", width: int = 860, height: int = 260,
          band: Optional[Tuple[int, int, str]] = None, label: str = "",
          every: int = 1) -> str:
    """Vertical bars per category, grouped or stacked by series."""
    left, right, top, bottom = 52, 16, 34, 40
    values = np.array([np.asarray(v, float) for _, _, v in series])
    peak = values.sum(axis=0).max() if mode == "stack" else values.max()
    y_top, step = _nice(float(peak) if values.size else 0)
    inner_w, inner_h = width - left - right, height - top - bottom
    ys = lambda v: top + inner_h * (1 - v / y_top)  # noqa: E731
    slot = inner_w / max(len(categories), 1)
    parts = _legend([(n, c) for n, c, _ in series], left, 16) if len(series) > 1 else []
    if band:
        b0, b1, text = band
        parts.append(f'<rect x="{left + slot * b0:.1f}" y="{top}" width="{slot * (b1 - b0):.1f}" '
                     f'height="{inner_h}" fill="{NIGHT_BAND}" fill-opacity="0.12"/>')
        parts.append(_text(left + slot * (b0 + b1) / 2, top + 12, text, 10, "middle", INK_MUTED))
    for tick in np.arange(0, y_top + step / 2, step):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{ys(tick):.1f}" '
                     f'y2="{ys(tick):.1f}" stroke="{GRID}"/>')
        parts.append(_text(left - 6, ys(tick) + 4, _fmt(tick), 10, "end", INK_MUTED))
    parts.append(f'<text x="14" y="{top + inner_h / 2:.1f}" font-size="10.5" fill="{INK_MUTED}" '
                 f'text-anchor="middle" transform="rotate(-90 14 {top + inner_h / 2:.1f})">'
                 f'{escape(y_title)}</text>')
    n = len(series)
    for i, cat in enumerate(categories):
        x_mid = left + slot * (i + 0.5)
        if i % every == 0:
            parts.append(_text(x_mid, height - bottom + 16, cat, 10, "middle", INK_MUTED))
        base = 0.0
        for j, (name, color, vals) in enumerate(series):
            v = float(vals[i])
            if v <= 0:
                continue
            if mode == "stack":
                bw = slot * 0.7
                x = x_mid - bw / 2
                y0, y1 = ys(base + v), ys(base)
                base += v
            else:
                bw = slot * 0.78 / n
                x = x_mid - slot * 0.39 + j * bw
                y0, y1 = ys(v), ys(0)
            parts.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{max(bw - 1.5, 0.8):.1f}" '
                         f'height="{max(y1 - y0 - (1 if mode == "stack" else 0), 0.6):.1f}" '
                         f'fill="{color}"><title>{escape(str(cat))} - {escape(name)}: '
                         f'{_fmt(v)}</title></rect>')
    return _svg(width, height, parts, label or y_title)


def ranked_hbars(labels: Sequence[str], values: Sequence[float], color: str,
                 width: int = 420, label: str = "") -> str:
    """Ranked horizontal bars, each labelled with its value."""
    left = int(min(max(max((len(str(l)) for l in labels), default=4) * 6.2 + 16, 70), 170))
    right, top, row_h = 40, 8, 24
    height = top + row_h * len(labels) + 10
    vmax = max(values) if len(values) else 1
    inner = width - left - right
    parts = []
    for i, (name, v) in enumerate(zip(labels, values)):
        y = top + i * row_h
        w = inner * v / vmax if vmax else 0
        parts.append(_text(left - 8, y + 14, name, 10.5, "end"))
        parts.append(f'<rect x="{left}" y="{y + 3}" width="{max(w, 1):.1f}" height="16" rx="2" '
                     f'fill="{color}"/>')
        parts.append(_text(left + w + 5, y + 15, _fmt(v), 10.5))
    return _svg(width, height, parts, label)


def area_bars(check: Dict[str, float], width: int = 420) -> str:
    return vbars(["Owner's claim", "Surveyor's estimate", "Calculated"],
                 [("Damaged area", BRAND, [check["owner"], check["surveyor"],
                                          check["calculated"]])],
                 "Total damaged area", width=width, height=230,
                 label="Damaged area: claimed against measured")


# ---------------------------------------------------------------------------
# The map, vector only
# ---------------------------------------------------------------------------
def _clip(points: List[Tuple[float, float]], width: float, height: float
          ) -> List[Tuple[float, float]]:
    """A polygon cut to the map frame (Sutherland-Hodgman).

    Done in geometry rather than with an SVG clipPath, which PDF
    converters honour unevenly: an outline left unclipped runs off the
    map and across the text around it.
    """
    edges = (
        (lambda p: p[0] >= 0, lambda a, b: (0.0, a[1] + (b[1] - a[1]) * (0 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= width, lambda a, b: (width, a[1] + (b[1] - a[1]) * (width - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= 0, lambda a, b: (a[0] + (b[0] - a[0]) * (0 - a[1]) / (b[1] - a[1]), 0.0)),
        (lambda p: p[1] <= height, lambda a, b: (a[0] + (b[0] - a[0]) * (height - a[1]) / (b[1] - a[1]), height)),
    )
    out = points
    for inside, cross in edges:
        if not out:
            break
        src, out = out, []
        prev = src[-1]
        for cur in src:
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
            prev = cur
    return out


def damage_map_svg(df: pd.DataFrame, width: int = 900, height: int = 560) -> str:
    """Surveys over forest outlines: area is loss, colour the claim, ring a house."""
    from core import boundaries
    from core.map_export import _fit_view, _projector

    points = df.dropna(subset=["Latitude", "Longitude"])
    if points.empty:
        return ""
    lat, lon, zoom = _fit_view(points["Latitude"].tolist(), points["Longitude"].tolist(),
                               width, height)
    project = _projector(lat, lon, zoom, width, height)
    parts = [f'<rect width="{width}" height="{height}" fill="#f4f6f3"/>']

    if boundaries.available():
        for level, color, stroke in ((boundaries.BEAT, "#d7c7a6", 0.6),
                                     (boundaries.RANGE, "#c4a265", 0.9),
                                     (boundaries.DIVISION, "#a9762c", 1.6)):
            for feature in boundaries.load(level)["features"]:
                for polygon in boundaries._rings(feature["geometry"]):
                    ring = polygon[0]
                    xy = [project(la, lo) for lo, la in ring]
                    xs_, ys_ = [p[0] for p in xy], [p[1] for p in xy]
                    if max(xs_) < 0 or min(xs_) > width or max(ys_) < 0 or min(ys_) > height:
                        continue
                    if min(xs_) < 0 or max(xs_) > width or min(ys_) < 0 or max(ys_) > height:
                        xy = _clip(xy, width, height)
                        if len(xy) < 3:
                            continue
                    path = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in xy) + "Z"
                    parts.append(f'<path d="{path}" fill="none" stroke="{color}" '
                                 f'stroke-width="{stroke}"/>')

    loss = points["Estimated Loss"].fillna(0).clip(lower=0)
    radius = 2.5 + np.sqrt(loss) / np.sqrt(max(float(loss.max()), 1.0)) * 13
    order = np.argsort(-radius.to_numpy())  # big first, so small ones stay visible
    for k in order:
        row = points.iloc[k]
        x, y = project(row["Latitude"], row["Longitude"])
        color = STATUS_COLORS.get(row["Compensation"], "#9AA5A0")
        r = float(radius.iloc[k])
        title = (f"{row['Village']} - {row['Kind']}, {damage.rupees(row['Estimated Loss'])}, "
                 f"{row['Compensation']}")
        if row["Kind"] == damage.HOUSE:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#ffffff" '
                         f'fill-opacity="0.85" stroke="{color}" stroke-width="2.2">'
                         f'<title>{escape(title)}</title></circle>')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" '
                         f'fill-opacity="0.82" stroke="#ffffff" stroke-width="0.8">'
                         f'<title>{escape(title)}</title></circle>')

    # Scale bar: Web Mercator metres per pixel at this latitude and zoom.
    m_per_px = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    km = next(k for k in (1, 2, 5, 10, 20, 50, 100) if k * 1000 / m_per_px >= 60)
    px = km * 1000 / m_per_px
    parts.append(f'<rect x="14" y="{height - 34}" width="{px + 16:.1f}" height="24" rx="3" '
                 f'fill="#ffffff" fill-opacity="0.85"/>')
    parts.append(f'<line x1="22" x2="{22 + px:.1f}" y1="{height - 16}" y2="{height - 16}" '
                 f'stroke="{INK}" stroke-width="2"/>')
    parts.append(_text(22 + px / 2, height - 21, f"{km} km", 10, "middle"))
    parts.append(f'<rect width="{width}" height="{height}" fill="none" stroke="{GRID}"/>')
    return _svg(width, height, parts, "Damage surveys by claim status")


# ---------------------------------------------------------------------------
# What goes in the report: one list of sections, rendered twice
# ---------------------------------------------------------------------------
def figure(df: pd.DataFrame, key: str) -> Optional[str]:
    """One chart or the map, as SVG -- what a single download needs."""
    if key == "map":
        return damage_map_svg(df) or None
    return _figures(df, with_map=False).get(key)


def _figures(df: pd.DataFrame, with_map: bool = True) -> Dict[str, str]:
    """Every chart, as SVG, keyed by name. The map is the slow one."""
    figs: Dict[str, str] = {}
    pipe = damage.compensation_pipeline(df)
    kinds = [k for k in (damage.CROP, damage.HOUSE) if k in set(df["Kind"])]

    def by_status(measure):
        segs = []
        for status in damage.STATUS_ORDER:
            vals = [float(pipe[(pipe["Kind"] == k) & (pipe["Compensation"] == status)]
                          [measure].sum()) for k in kinds]
            if measure == "Loss":
                vals = [v / 1e5 for v in vals]
            segs.append((status, STATUS_COLORS[status], vals))
        return segs

    figs["claims_cases"] = stacked_hbars(kinds, by_status("Cases"), "Cases",
                                         label="Cases by claim status")
    figs["claims_loss"] = stacked_hbars(kinds, by_status("Loss"), "Estimated loss (Rs lakh)",
                                        value_fmt=lambda v: f"{v:,.1f}",
                                        label="Estimated loss by claim status")

    m = damage.monthly(df)
    if not m.empty:
        months = sorted(m["Month"].unique())
        names = [pd.Timestamp(x).strftime("%b %y") for x in months]
        series = []
        for k in kinds:
            part = m[m["Kind"] == k].set_index("Month")["Loss"].reindex(months, fill_value=0)
            series.append((f"{k} damage", KIND_COLORS[k], (part / 1e5).tolist()))
        figs["monthly"] = vbars(names, series, "Estimated loss (Rs lakh)", mode="stack",
                                height=280, every=1 if len(names) <= 14 else 2,
                                label="Estimated loss by month of damage")

    def kind_series(table, col, order):
        out = []
        for k in kinds:
            part = table[table["Kind"] == k].set_index(col)["Cases"].reindex(order).fillna(0)
            out.append((f"{k} damage", KIND_COLORS[k], part.tolist()))
        return out

    figs["herd"] = vbars(damage.HERD_LABELS, kind_series(damage.herd_sizes(df), "Herd",
                                                         damage.HERD_LABELS),
                         "Cases", width=430, label="Elephants present")
    hours = damage.hourly(df)
    figs["hours"] = vbars([f"{h:02d}" for h in range(24)],
                          kind_series(hours, "Hour", list(range(24))), "Cases",
                          mode="stack", width=430, band=(6, 18, "daylight"), every=3,
                          label="Hour of damage")
    figs["lag"] = vbars(damage.LAG_LABELS, kind_series(damage.survey_lag(df), "Delay",
                                                       damage.LAG_LABELS),
                        "Cases", height=240, label="Days from damage to survey")

    crop = df[df["Kind"] == damage.CROP]
    if not crop.empty:
        ct = damage.crop_table(df).head(12)
        figs["crops"] = ranked_hbars(ct["Crop"].tolist(), ct["Cases"].tolist(),
                                     KIND_COLORS[damage.CROP], label="Crops damaged")
        st_ = damage.exploded_counts(crop, "Crop Stage", damage.CROP_STAGES)
        figs["stages"] = ranked_hbars(st_["Crop Stage"].tolist(), st_["Cases"].tolist(),
                                      KIND_COLORS[damage.CROP], label="Stage of the crop")
        fe = damage.exploded_counts(crop, "Fencing").head(10)
        figs["fencing"] = ranked_hbars(fe["Fencing"].tolist(), fe["Cases"].tolist(),
                                       "#6B8578", label="Fencing at the damaged field")
        check = damage.area_check(df)
        if check:
            figs["area"] = area_bars(check)
    house = df[df["Kind"] == damage.HOUSE]
    if not house.empty:
        ht = damage.exploded_counts(house, "House Type")
        figs["house_type"] = ranked_hbars(ht["House Type"].tolist(), ht["Cases"].tolist(),
                                          KIND_COLORS[damage.HOUSE], label="Type of house")
        rm = damage.exploded_counts(house, "Rooms Damaged")
        figs["rooms"] = ranked_hbars(rm["Rooms Damaged"].tolist(), rm["Cases"].tolist(),
                                     KIND_COLORS[damage.HOUSE], label="Part of the house broken")
    if with_map:
        figs["map"] = damage_map_svg(df)
    return figs


def _kpis(df: pd.DataFrame, match: Optional[pd.Series]) -> List[Tuple[str, str, str]]:
    s = damage.summary(df)
    rows = [
        ("Households surveyed", f"{s['reports']:,}", f"{s['crop']} crop, {s['house']} house"),
        ("Estimated loss (Rs)", damage.lakh(s["loss"]), f"{s['villages']} villages"),
        ("No claim made", f"{s['not_applied']:,}",
         f"{s['not_applied'] / s['reports']:.0%} of cases, {damage.rupees(s['not_applied_loss'])}"
         if s["reports"] else ""),
        ("Compensation paid (Rs)", damage.lakh(s["paid"]), f"{s['paid_cases']} case(s)"),
        ("Median days to survey", "-" if s["median_days"] != s["median_days"]
         else f"{s['median_days']:.0f}", "from damage to visit"),
    ]
    if match is not None and len(match):
        rows.append(("No Gaj Rakshak report nearby", f"{int((~match).sum()):,}",
                     f"{(~match).mean():.0%} of surveys"))
    else:
        rows.append(("Remarks: not in Gaj Rakshak", f"{s['register_missing']:,}",
                     f"of {s['reports']} surveys"))
    return rows


def _period(df: pd.DataFrame) -> str:
    dates = df["Damage Date"].dropna()
    if dates.empty:
        return "dates not recorded"
    return f"{dates.min():%d %b %Y} to {dates.max():%d %b %Y}"


# Rows of a long table shown in the body; the full table is the appendix.
TOP_ROWS = 20

FOLLOW_UP_COLUMNS = ["Entry", "Village", "Beat", "Kind", "Damage Date", "Estimated Loss",
                     "Days Since Damage"]
VILLAGE_COLUMNS = ["Village", "Division", "Beat", "Cases", "Crop", "House", "Estimated Loss",
                   "Not Applied", "Repeat Households"]


QNUM = ["Estimated Loss", "Days Since Damage"]
VNUM = ["Cases", "Crop", "House", "Estimated Loss", "Not Applied", "Repeat Households",
        "No Register Match"]


def _cell(value) -> str:
    if isinstance(value, pd.Timestamp):
        return "" if pd.isna(value) else f"{value:%d %b %Y}"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else ""
    if isinstance(value, (float, np.floating)):
        return "" if value != value else f"{value:,.0f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
_CSS = """
@page { size: A4; margin: 14mm 12mm; }
* { box-sizing: border-box; }
body { font-family: Helvetica, Arial, sans-serif; color: #16221c; margin: 0;
       background: #eef2ef; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.page { max-width: 1040px; margin: 0 auto; padding: 28px; }
header { background: #1f5f3f; color: #fff; padding: 26px 30px; border-radius: 10px 10px 0 0; }
header h1 { margin: 0 0 6px; font-size: 26px; letter-spacing: -0.01em; }
header p { margin: 0; opacity: .9; font-size: 13.5px; }
main { background: #fff; padding: 26px 30px 30px; border-radius: 0 0 10px 10px; }
h2 { font-size: 18px; color: #1f5f3f; border-bottom: 2px solid #e3e9e5; padding-bottom: 6px;
     margin: 30px 0 10px; break-after: avoid; }
h3 { font-size: 13.5px; margin: 16px 0 4px; break-after: avoid; }
.kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0; }
.kpi { border: 1px solid #dce5df; border-radius: 8px; padding: 12px 14px; background: #f7faf8; }
.kpi .l { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #5b6b62; }
.kpi .v { font-size: 24px; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }
.kpi .d { font-size: 11.5px; color: #5b6b62; margin-top: 2px; }
.findings { background: #eef4f0; border-left: 4px solid #1f5f3f; padding: 12px 18px 12px 30px;
            border-radius: 0 8px 8px 0; }
.findings li { margin: 5px 0; font-size: 13.5px; line-height: 1.45; }
figure { margin: 8px 0 14px; break-inside: avoid; }
figure svg { width: 100%; height: auto; display: block; }
figcaption { font-size: 11.5px; color: #5b6b62; margin-top: 4px; line-height: 1.45; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
table { width: 100%; border-collapse: collapse; font-size: 11.5px; margin: 8px 0 4px; }
th, td { padding: 5px 7px; border-bottom: 1px solid #e8ece9; text-align: left; }
th { background: #f0f6f2; color: #1f5f3f; font-weight: 600; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 11.5px; color: #5b6b62; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
            margin-right: 5px; vertical-align: -1px; }
.notes, .privacy { font-size: 11.5px; color: #5b6b62; }
.privacy { background: #fbfaf2; border: 1px solid #ece7d5; padding: 10px 14px; border-radius: 6px;
           margin-top: 22px; }
@media print { body { background: #fff; } .page { padding: 0; max-width: none; }
               header, main { border-radius: 0; } .two { gap: 14px; } }
@media (max-width: 720px) { .kpis { grid-template-columns: 1fr 1fr; } .two { grid-template-columns: 1fr; } }
"""


def _html_table(frame: pd.DataFrame, numeric: Sequence[str]) -> str:
    head = "".join(f"<th class='{'n' if c in numeric else ''}'>{escape(c)}</th>"
                   for c in frame.columns)
    body = "".join(
        "<tr>" + "".join(f"<td class='{'n' if c in numeric else ''}'>{escape(_cell(v))}</td>"
                         for c, v in zip(frame.columns, row)) + "</tr>"
        for row in frame.itertuples(index=False)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_html(df: pd.DataFrame, notes: Sequence[str] = (),
               match: Optional[pd.Series] = None, scope: str = "") -> str:
    """The study as one self-contained, vector HTML file."""
    figs = _figures(df)
    fig = lambda key, cap="": (  # noqa: E731
        f"<figure>{figs[key]}" + (f"<figcaption>{escape(cap)}</figcaption>" if cap else "")
        + "</figure>" if figs.get(key) else "")
    kpis = "".join(f"<div class='kpi'><div class='l'>{escape(l)}</div><div class='v'>"
                   f"{escape(v)}</div><div class='d'>{escape(d)}</div></div>"
                   for l, v, d in _kpis(df, match))
    findings = "".join(f"<li>{escape(line)}</li>" for line in damage.headlines(df, match))
    queue = damage.follow_up(df)[FOLLOW_UP_COLUMNS]
    villages = damage.villages(df, match)
    vcols = VILLAGE_COLUMNS + (["No Register Match"] if "No Register Match" in villages else [])
    legend = "".join(f"<span><i style='background:{c}'></i>{escape(s)}</span>"
                     for s, c in STATUS_COLORS.items() if s in set(df["Compensation"]))
    s = damage.summary(df)
    check = damage.area_check(df)
    note_list = "".join(f"<li>{escape(n)}</li>" for n in notes)
    appendix = ""
    if len(queue) > TOP_ROWS or len(villages) > TOP_ROWS:
        appendix = "<h2 style='break-before: page'>Appendix</h2>"
        if len(queue) > TOP_ROWS:
            appendix += (f"<h3>A. Follow-up queue: all {len(queue)} case(s) with no claim</h3>"
                         + _html_table(queue, QNUM))
        if len(villages) > TOP_ROWS:
            appendix += (f"<h3>B. All {len(villages)} villages</h3>"
                         + _html_table(villages[vcols], VNUM))

    area_note = (f"Over {check['cases']} fields with both figures the calculated area is "
                 f"{check['ratio']:.0%} of what owners reported, in the units the form "
                 "records." if check else "")
    crop_block = ""
    if "crops" in figs:
        crop_block = (
            "<h2>Crop damage</h2><div class='two'>"
            f"<div><h3>Crops damaged</h3>{fig('crops', 'A field with two crops counts under both.')}</div>"
            f"<div><h3>Stage of the crop</h3>{fig('stages')}</div></div><div class='two'>"
            f"<div><h3>Fencing at the damaged field</h3>{fig('fencing', 'Only damaged fields are surveyed: this is what was in place where damage happened, not evidence of which fence works.')}</div>"
            + (f"<div><h3>Damaged area: claimed against measured</h3>{fig('area', area_note)}</div>"
               if check else "<div></div>")
            + "</div>")
    house_block = ""
    if "house_type" in figs:
        house = df[df["Kind"] == damage.HOUSE]
        lit = house["Light On"].dropna()
        lit_line = (f"A light was on at {int(lit.sum())} of {len(lit)} houses "
                    f"({lit.mean():.0%}) when the elephant came." if len(lit) else "")
        house_block = (
            "<h2>House damage</h2><div class='two'>"
            f"<div><h3>Type of house</h3>{fig('house_type')}</div>"
            f"<div><h3>Part of the house broken</h3>{fig('rooms', 'Kitchens and bakhari (grain stores) are where food is kept.')}</div>"
            f"</div><p class='notes'>{escape(lit_line)}</p>")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{TITLE}</title><style>{_CSS}</style></head>
<body><div class="page">
<header><h1>{TITLE}</h1>
<p>Household crop and house damage surveys (Epicollect5) &nbsp;|&nbsp; Damage {escape(_period(df))}
&nbsp;|&nbsp; Generated {datetime.now():%d %b %Y, %H:%M}</p>
{f"<p>{escape(scope)}</p>" if scope else ""}</header>
<main>
<div class="kpis">{kpis}</div>
<h2>Assessment</h2><ul class="findings">{findings}</ul>

<h2>Compensation</h2>
<h3>Cases by claim status</h3>{fig('claims_cases')}
<h3>Estimated loss by claim status</h3>{fig('claims_loss')}
<h3>Follow-up queue: the {min(TOP_ROWS, len(queue))} largest of {len(queue)} case(s) with no claim</h3>
{_html_table(queue.head(TOP_ROWS), QNUM) if len(queue) else "<p class='notes'>Every surveyed household has a claim made or recorded.</p>"}
<p class="notes">{"The full list is in the appendix. " if len(queue) > TOP_ROWS else ""}The Epicollect5 entry opens the household's record for anyone with access to the project.</p>

<h2>Losses over time</h2>{fig('monthly')}

<h2>Gaj Rakshak gap</h2>
<p>Surveyors noted <b>{s['register_missing']}</b> incident(s) missing from Gaj Rakshak and
<b>{s['register_wrong']}</b> with the wrong location there.
{f"Checked against the register by place and date, <b>{int((~match).sum())}</b> of {len(match)} surveys have no report nearby." if match is not None and len(match) else ""}
Damage that never reached the register is invisible to the conflict dashboard and the beat ranking.</p>

<h2>Where</h2>
{fig('map')}
<div class="legend">{legend}</div>
<p class="notes">Circle area is estimated loss; colour is the claim's status. Fields are solid, houses ringed.
Outlines are forest divisions, ranges and beats.</p>
<h3>Villages: the {min(TOP_ROWS, len(villages))} with the largest loss of {len(villages)}</h3>
{_html_table(villages[vcols].head(TOP_ROWS), VNUM)}
<p class="notes">Beat is the one the point falls in, or the nearest within 10 km when it lies on farmland outside every beat.</p>

<h2>When and by how many</h2>
<div class="two"><div><h3>Elephants present</h3>{fig('herd')}</div>
<div><h3>Hour of damage</h3>{fig('hours')}</div></div>
{crop_block}
{house_block}
<h2>Survey timeliness</h2>{fig('lag', 'Days from the damage to the survey visit.')}

{f"<h2>Data notes</h2><ul class='notes'>{note_list}</ul>" if note_list else ""}
{appendix}
<p class="privacy">Owner names, phone numbers, photos, surveyor emails and remark text were dropped
when the export was read. This report identifies households only by their Epicollect5 entry.</p>
</main></div></body></html>"""


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def build_pdf(df: pd.DataFrame, notes: Sequence[str] = (),
              match: Optional[pd.Series] = None, scope: str = "") -> bytes:
    """The study as an A4 PDF with every chart as vector drawing."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (CondPageBreak, KeepTogether, ListFlowable, ListItem,
                                    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)
    from svglib.svglib import svg2rlg

    figs = _figures(df)
    page_w, page_h = A4
    margin = 13 * mm
    frame_w = page_w - 2 * margin
    green, muted = colors.HexColor(BRAND), colors.HexColor(INK_MUTED)

    base = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
                          leading=13, textColor=colors.HexColor(INK))
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10.5, textColor=muted)
    h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=19, leading=23,
                        textColor=colors.white)
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=13, leading=16,
                        textColor=green, spaceBefore=12, spaceAfter=5)
    h3 = ParagraphStyle("h3", parent=body, fontName="Helvetica-Bold", fontSize=9.5,
                        spaceBefore=6, spaceAfter=2)
    cell = ParagraphStyle("c", parent=body, fontSize=7.6, leading=9.4)
    cell_r = ParagraphStyle("cr", parent=cell, alignment=TA_RIGHT)

    def esc(text):
        return escape(str(text))

    def drawing(key, width):
        svg = figs.get(key)
        if not svg:
            return Spacer(1, 1)
        d = svg2rlg(io.BytesIO(svg.encode("utf-8")))
        scale = width / d.width
        d.width, d.height = d.width * scale, d.height * scale
        d.scale(scale, scale)
        return d

    def pair(left, right):
        gap = 6 * mm
        w = (frame_w - gap) / 2
        t = Table([[left(w), right(w)]], colWidths=[w + gap / 2, w + gap / 2])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        return t

    def titled(title, key, caption=""):
        def make(w):
            items = [Paragraph(esc(title), h3), drawing(key, w)]
            if caption:
                items.append(Paragraph(esc(caption), small))
            return items
        return make

    def table(frame, numeric, widths):
        data = [[Paragraph(f"<b>{esc(c)}</b>", cell_r if c in numeric else cell)
                 for c in frame.columns]]
        for row in frame.itertuples(index=False):
            data.append([Paragraph(esc(_cell(v)), cell_r if c in numeric else cell)
                         for c, v in zip(frame.columns, row)])
        t = Table(data, colWidths=[frame_w * w for w in widths], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f6f2")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#e3e9e5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        return t

    story = []
    head = Table([[Paragraph(TITLE, h1)],
                  [Paragraph(esc(f"Household crop and house damage surveys (Epicollect5) | "
                                 f"Damage {_period(df)} | Generated "
                                 f"{datetime.now():%d %b %Y, %H:%M}"
                                 + (f" | {scope}" if scope else "")),
                             ParagraphStyle("hs", parent=small, textColor=colors.white))]],
                 colWidths=[frame_w])
    head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), green),
                              ("LEFTPADDING", (0, 0), (-1, -1), 12),
                              ("TOPPADDING", (0, 0), (0, 0), 12),
                              ("BOTTOMPADDING", (0, -1), (-1, -1), 12)]))
    story += [head, Spacer(1, 10)]

    k = _kpis(df, match)
    kpi_style = ParagraphStyle("k", parent=body, leading=19)
    cells = [[Paragraph(f"<font size=7.5 color='{INK_MUTED}'>{esc(l.upper())}</font><br/>"
                        f"<font size=16><b>{esc(v)}</b></font><br/>"
                        f"<font size=7.5 color='{INK_MUTED}'>{esc(d)}</font>", kpi_style)
              for l, v, d in k[i:i + 3]] for i in range(0, len(k), 3)]
    kt = Table(cells, colWidths=[frame_w / 3] * 3)
    kt.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dce5df")),
                            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dce5df")),
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7faf8")),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story += [kt, Paragraph("Assessment", h2),
              ListFlowable([ListItem(Paragraph(esc(t), body), leftIndent=10)
                            for t in damage.headlines(df, match)],
                           bulletType="bullet", start="\u2022", leftIndent=10)]

    story += [Paragraph("Compensation", h2),
              Paragraph("Cases by claim status", h3), drawing("claims_cases", frame_w * 0.92),
              Paragraph("Estimated loss by claim status", h3),
              drawing("claims_loss", frame_w * 0.92)]
    queue = damage.follow_up(df)[FOLLOW_UP_COLUMNS]
    queue_widths = [0.30, 0.17, 0.15, 0.07, 0.11, 0.10, 0.10]
    story.append(Paragraph(f"Follow-up queue: the {min(TOP_ROWS, len(queue))} largest of "
                           f"{len(queue)} case(s) with no claim", h3))
    if len(queue):
        story.append(table(queue.head(TOP_ROWS), ["Estimated Loss", "Days Since Damage"],
                           queue_widths))
        story.append(Paragraph("The full list is in the appendix. The Epicollect5 entry opens "
                               "the household's record for anyone with access to the project.",
                               small))

    story += [CondPageBreak(90 * mm), Paragraph("Losses over time", h2),
              drawing("monthly", frame_w)]
    s = damage.summary(df)
    gap_text = (f"Surveyors noted <b>{s['register_missing']}</b> incident(s) missing from "
                f"Gaj Rakshak and <b>{s['register_wrong']}</b> with the wrong location there.")
    if match is not None and len(match):
        gap_text += (f" Checked against the register by place and date, "
                     f"<b>{int((~match).sum())}</b> of {len(match)} surveys have no report nearby.")
    story += [Paragraph("Gaj Rakshak gap", h2), Paragraph(gap_text, body)]

    if figs.get("map"):
        legend = " &nbsp; ".join(f"<font color='{c}'>\u25cf</font> {esc(st)}"
                                 for st, c in STATUS_COLORS.items()
                                 if st in set(df["Compensation"]))
        story += [CondPageBreak(130 * mm), Paragraph("Where", h2),
                  KeepTogether([drawing("map", frame_w), Spacer(1, 3),
                                Paragraph(legend, small),
                                Paragraph("Circle area is estimated loss; colour is the claim's "
                                          "status. Fields are solid, houses ringed. Outlines "
                                          "are forest divisions, ranges and beats.", small)])]
    villages = damage.villages(df, match)
    vcols = VILLAGE_COLUMNS + (["No Register Match"] if "No Register Match" in villages else [])
    widths = [0.17, 0.14, 0.15, 0.07, 0.07, 0.07, 0.12, 0.09, 0.12]
    if len(vcols) > len(VILLAGE_COLUMNS):
        widths = [w * 0.9 for w in widths] + [0.1]
    vnum = ["Cases", "Crop", "House", "Estimated Loss", "Not Applied", "Repeat Households",
            "No Register Match"]
    story += [Paragraph(f"Villages: the {min(TOP_ROWS, len(villages))} with the largest loss "
                        f"of {len(villages)}", h3),
              table(villages[vcols].head(TOP_ROWS), vnum, widths)]

    story += [CondPageBreak(80 * mm), Paragraph("When and by how many", h2),
              pair(titled("Elephants present", "herd"), titled("Hour of damage", "hours"))]
    if "crops" in figs:
        check = damage.area_check(df)
        story += [CondPageBreak(90 * mm), Paragraph("Crop damage", h2),
                  pair(titled("Crops damaged", "crops", "A field with two crops counts under both."),
                       titled("Stage of the crop", "stages")),
                  pair(titled("Fencing at the damaged field", "fencing",
                              "Only damaged fields are surveyed: what was in place where damage "
                              "happened, not evidence of which fence works."),
                       titled("Damaged area: claimed against measured", "area",
                              f"Over {check['cases']} fields the calculated area is "
                              f"{check['ratio']:.0%} of what owners reported." if check else ""))]
    if "house_type" in figs:
        lit = df.loc[df["Kind"] == damage.HOUSE, "Light On"].dropna()
        story += [CondPageBreak(70 * mm), Paragraph("House damage", h2),
                  pair(titled("Type of house", "house_type"),
                       titled("Part of the house broken", "rooms",
                              "Kitchens and bakhari (grain stores) are where food is kept."))]
        if len(lit):
            story.append(Paragraph(f"A light was on at {int(lit.sum())} of {len(lit)} houses "
                                   f"({lit.mean():.0%}) when the elephant came.", body))
    story += [CondPageBreak(70 * mm), Paragraph("Survey timeliness", h2),
              drawing("lag", frame_w)]
    if notes:
        story += [Paragraph("Data notes", h2)] + [Paragraph(esc(n), small) for n in notes]
    if len(queue) > TOP_ROWS or len(villages) > TOP_ROWS:
        story.append(PageBreak())
        story.append(Paragraph("Appendix", h2))
    if len(queue) > TOP_ROWS:
        story += [Paragraph(f"A. Follow-up queue: all {len(queue)} case(s) with no claim", h3),
                  table(queue, ["Estimated Loss", "Days Since Damage"], queue_widths)]
    if len(villages) > TOP_ROWS:
        story += [Paragraph(f"B. All {len(villages)} villages", h3),
                  table(villages[vcols], vnum, widths)]
    story += [Spacer(1, 8), Paragraph(
        "Owner names, phone numbers, photos, surveyor emails and remark text were dropped when "
        "the export was read. This report identifies households only by their Epicollect5 entry.",
        small)]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(margin, 8 * mm, TITLE)
        canvas.drawRightString(page_w - margin, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=margin, rightMargin=margin,
                            topMargin=margin, bottomMargin=14 * mm, title=TITLE,
                            author="Elephant Conflict Intelligence")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()
