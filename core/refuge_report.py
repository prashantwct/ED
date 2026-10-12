"""Day refuges for the brief: the map and the section, no Streamlit.

The view's download menu and the HTML brief draw the same map from here,
so a refuge printed in the brief is the one on screen.
"""

from __future__ import annotations

from html import escape
from typing import Dict, List

import pandas as pd

from core import refuges as rf

CONFIDENCE_COLORS = {rf.CONFIRMED: "#8C1D18", rf.PROBABLE: "#C2570C", rf.POSSIBLE: "#C9A227"}
RAID_COLOR = "#5B6B62"

BRIEF_COLUMNS = [
    ("Refuge", "Refuge"), ("Confidence", "Confidence"), ("Beat", "Beat"),
    ("Days Used", "Days used"), ("Last Used", "Last used"), ("Group", "Group"),
    ("Cover Share %", "Cover"), ("Linked Raids", "Raids"), ("Raids Head", "Raids head"),
    ("Raid Distance (km, median)", "Reach (km)"), ("Villages Raided", "Villages raided"),
    ("Recommended Action", "Action"),
]


def map_svg(table: pd.DataFrame, links: pd.DataFrame, width: int = 900, height: int = 600) -> str:
    """The refuge map as vector SVG, for the download menu and the brief."""
    from html import escape

    from core.damage_report import _svg, _text, outline_parts, scale_bar
    from core.map_export import _fit_view, _projector

    lats = table["Latitude"].tolist() + links["Latitude"].tolist()
    lons = table["Longitude"].tolist() + links["Longitude"].tolist()
    lat, lon, zoom = _fit_view(lats, lons, width, height)
    project = _projector(lat, lon, zoom, width, height)
    parts = [f'<rect width="{width}" height="{height}" fill="#f4f6f3"/>']
    parts += outline_parts(project, width, height)
    ref = table.set_index("Refuge")
    for r in links.to_dict("records"):
        x0, y0 = project(ref.at[r["Refuge"], "Latitude"], ref.at[r["Refuge"], "Longitude"])
        x1, y1 = project(r["Latitude"], r["Longitude"])
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                     f'stroke="{RAID_COLOR}" stroke-opacity="0.45" stroke-width="1"/>')
        parts.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="2.2" fill="{RAID_COLOR}"/>')
    for r in table.to_dict("records"):
        x, y = project(r["Latitude"], r["Longitude"])
        color = CONFIDENCE_COLORS[r["Confidence"]]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{6 + min(r["Days Used"], 20) * 0.5:.1f}" '
                     f'fill="{color}" fill-opacity="0.85" stroke="#ffffff" stroke-width="1.5">'
                     f'<title>{escape(r["Refuge"])}</title></circle>')
        parts.append(_text(x + 10, y - 8, r["Refuge"], 11, weight="bold"))
    parts += scale_bar(lat, zoom, height)
    legend_x = width - 150
    for k, (name, color) in enumerate(CONFIDENCE_COLORS.items()):
        y = 20 + k * 18
        parts.append(f'<circle cx="{legend_x}" cy="{y - 4}" r="5" fill="{color}"/>')
        parts.append(_text(legend_x + 10, y, f"{name} refuge", 10.5))
    parts.append(f'<line x1="{legend_x - 6}" x2="{legend_x + 6}" y1="{20 + 3 * 18 - 4}" '
                 f'y2="{20 + 3 * 18 - 4}" stroke="{RAID_COLOR}"/>')
    parts.append(_text(legend_x + 10, 20 + 3 * 18, "Linked night raid", 10.5))
    return _svg(width, height, parts, "Elephant day refuges and linked raids")


def _cell(col: str, value: object) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "--"
    if isinstance(value, pd.Timestamp):
        return f"{value:%d %b %Y}"
    if col == "Cover Share %":
        return f"{float(value):.0f}%"
    if col == "Confidence":
        color = CONFIDENCE_COLORS.get(str(value), "#555")
        return f'<span style="color:{color};font-weight:600">{escape(str(value))}</span>'
    return escape(str(value))


def brief_section(result: Dict[str, object], top_n: int = 12) -> str:
    """The brief's Day Refuges section: findings, map, register, caveats.

    Only refuges with night conflict linked to them are in ``result`` (see
    :func:`core.refuges.find_refuges`), so every row here sends raids.
    """
    table, links = result["refuges"], result["links"]
    source = result.get("land_cover", "")
    sub = ("where herds lie up by day, and the raids each refuge sends"
           + (f" -- land cover: {source}" if source else ""))
    head = f'<h2>Day Refuges <span class="sub">{escape(sub)}</span></h2>'
    lines: List[str] = rf.headlines(result)
    findings = "".join(f"<li>{escape(line)}</li>" for line in lines)
    out = [head, f'<div class="headlines"><ul>{findings}</ul></div>']
    if table is None or table.empty:
        out.append('<p class="empty-note">No day refuge had night conflict linked to it in '
                   "this period.</p>")
    else:
        out.append("<div class='chart-title'>Refuges and the night raids linked to them</div>"
                   f"<div class='chart'>{map_svg(table, links)}</div>")
        cols = [(c, h) for c, h in BRIEF_COLUMNS if c in table]
        rows = "".join(
            "<tr>" + "".join(f"<td>{_cell(c, r.get(c))}</td>" for c, _ in cols) + "</tr>"
            for r in table.head(top_n).to_dict("records"))
        more = (f'<p class="empty-note">{len(table) - top_n} more refuge(s) in the app\'s '
                "Day refuges view.</p>" if len(table) > top_n else "")
        out.append("<table><thead><tr>" + "".join(f"<th>{escape(h)}</th>" for _, h in cols)
                   + f"</tr></thead><tbody>{rows}</tbody></table>{more}")
    caveats = [*result.get("land_cover_notes", []), *result["caveats"]]
    if caveats:
        out.append('<div class="caveats"><ul>'
                   + "".join(f"<li>{escape(c)}</li>" for c in caveats) + "</ul></div>")
    return "\n".join(out)
