"""Day refuges for the brief: the map and the section, no Streamlit.

The view's download menu and the HTML brief draw the same map from here,
so a refuge printed in the brief is the one on screen.
"""

from __future__ import annotations

from html import escape
from typing import Dict, List, Optional

import pandas as pd

from core import refuges as rf

CONFIDENCE_COLORS = {rf.CONFIRMED: "#8C1D18", rf.PROBABLE: "#C2570C", rf.POSSIBLE: "#C9A227"}
RAID_COLOR = "#5B6B62"
AREA_COLOR = "#2F5D8A"     # raid areas with no known refuge

BRIEF_COLUMNS = [
    ("Refuge", "Refuge"), ("Confidence", "Confidence"), ("Beat", "Beat"),
    ("Days Used", "Days used"), ("Last Used", "Last used"), ("Group", "Group"),
    ("Cover Share %", "Cover"), ("Linked Raids", "Raids"), ("Same-Night Raids", "Same night"),
    ("Raids Head", "Raids head"),
    ("Raid Distance (km, median)", "Reach (km)"), ("Villages Raided", "Villages raided"),
    ("Recommended Action", "Action"),
]


def map_svg(table: pd.DataFrame, links: pd.DataFrame, width: int = 900, height: int = 600,
            areas: Optional[pd.DataFrame] = None) -> str:
    """The refuge map as vector SVG, for the download menu and the brief."""
    from html import escape

    from core.damage_report import _svg, _text, outline_parts, scale_bar
    from core.map_export import _fit_view, _projector

    areas = areas if areas is not None else pd.DataFrame(columns=["Latitude", "Longitude"])
    frames = [f for f in (table, links, areas) if len(f)]
    lats = [v for f in frames for v in f["Latitude"].tolist()]
    lons = [v for f in frames for v in f["Longitude"].tolist()]
    lat, lon, zoom = _fit_view(lats, lons, width, height)
    project = _projector(lat, lon, zoom, width, height)
    parts = [f'<rect width="{width}" height="{height}" fill="#f4f6f3"/>']
    parts += outline_parts(project, width, height)
    for r in areas.to_dict("records"):
        x, y = project(r["Latitude"], r["Longitude"])
        size = 5 + min(r["Raids"], 30) * 0.4
        parts.append(f'<rect x="{x - size:.1f}" y="{y - size:.1f}" width="{2 * size:.1f}" '
                     f'height="{2 * size:.1f}" fill="{AREA_COLOR}" fill-opacity="0.25" '
                     f'stroke="{AREA_COLOR}" stroke-width="1.5"><title>{escape(r["Area"])}: '
                     f'{r["Raids"]} raids, no known refuge</title></rect>')
        parts.append(_text(x + size + 3, y + 4, r["Area"], 10.5, color=AREA_COLOR))
    ref = table.set_index("Refuge") if len(table) else table
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
    parts.append(f'<rect x="{legend_x - 5}" y="{20 + 4 * 18 - 9}" width="10" height="10" '
                 f'fill="{AREA_COLOR}" fill-opacity="0.25" stroke="{AREA_COLOR}"/>')
    parts.append(_text(legend_x + 10, 20 + 4 * 18, "Raid area, no known refuge", 10.5))
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


AREA_BRIEF_COLUMNS = [
    ("Area", "Area"), ("Beat", "Beat"), ("Raids", "Raids"), ("Nights", "Nights"),
    ("Last Raid", "Last raid"), ("Months", "Months"), ("Villages Raided", "Villages raided"),
    ("To Nearest Known Refuge (km)", "Known refuge (km)"),
    ("Search Patch (ha)", "Patch to search (ha)"), ("Recommended Action", "Action"),
]


def _table(df: pd.DataFrame, columns, top_n: int, noun: str) -> str:
    cols = [(c, h) for c, h in columns if c in df]
    rows = "".join(
        "<tr>" + "".join(f"<td>{_cell(c, r.get(c))}</td>" for c, _ in cols) + "</tr>"
        for r in df.head(top_n).to_dict("records"))
    more = (f'<p class="empty-note">{len(df) - top_n} more {noun} in the app\'s '
            "Day refuges view.</p>" if len(df) > top_n else "")
    return ("<table><thead><tr>" + "".join(f"<th>{escape(h)}</th>" for _, h in cols)
            + f"</tr></thead><tbody>{rows}</tbody></table>{more}")


def brief_section(result: Dict[str, object], top_n: int = 12) -> str:
    """The brief's Day Refuges section: findings, map, both registers, caveats.

    Every raid is accounted for: linked to a refuge, in a raid area with no
    known refuge, or counted as scattered (see
    :func:`core.refuges.find_refuges`).
    """
    table, links = result["refuges"], result["links"]
    areas = result.get("areas")
    areas = areas if areas is not None else pd.DataFrame()
    source = result.get("land_cover", "")
    sub = ("where herds lie up by day, the raids each refuge sends, and raids with no "
           "known refuge" + (f" -- land cover: {source}" if source else ""))
    head = f'<h2>Day Refuges <span class="sub">{escape(sub)}</span></h2>'
    lines: List[str] = rf.headlines(result)
    findings = "".join(f"<li>{escape(line)}</li>" for line in lines)
    out = [head, f'<div class="headlines"><ul>{findings}</ul></div>']
    if len(table) or len(areas):
        out.append("<div class='chart-title'>Refuges, their linked raids, and raid areas with "
                   "no known refuge</div>"
                   f"<div class='chart'>{map_svg(table, links, areas=areas)}</div>")
    if len(table):
        out.append("<div class='chart-title'>Day refuges that send raids</div>")
        out.append(_table(table, BRIEF_COLUMNS, top_n, "refuge(s)"))
    else:
        out.append('<p class="empty-note">No day refuge had a night raid linked to it in '
                   "this period.</p>")
    if len(areas):
        out.append("<div class='chart-title'>Raid areas with no known refuge -- where to "
                   "search by day</div>")
        out.append(_table(areas, AREA_BRIEF_COLUMNS, top_n, "raid area(s)"))
    caveats = [*result.get("land_cover_notes", []), *result["caveats"]]
    if caveats:
        out.append('<div class="caveats"><ul>'
                   + "".join(f"<li>{escape(c)}</li>" for c in caveats) + "</ul></div>")
    return "\n".join(out)
