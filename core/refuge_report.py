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


MAP_LABELS = 14            # most labels a map carries before it stops being read
LABEL_GAP_PX = 46          # labels closer than this to a placed one are dropped
LEGEND_H = 34


def map_svg(table: pd.DataFrame, links: pd.DataFrame, width: int = 1000, height: int = 640,
            areas: Optional[pd.DataFrame] = None, style_name: Optional[str] = None) -> str:
    """The refuge map as SVG over the basemap, for the download menu and the brief.

    The basemap is fetched and inlined as a JPEG (vector marks on top), so
    the file reads like the interactive map; without network it falls back
    to the forest outlines alone. Only division and range lines are drawn
    over the basemap -- beat lines at landscape scale are noise -- and only
    the refuges and raid areas that matter most are labelled, where a label
    has room.
    """
    import math

    from core import boundaries
    from core.damage_report import _svg, outline_parts, scale_bar
    from core.map_engine import DEFAULT_BASEMAP
    from core.map_export import _basemap_data_uri, _fit_view, _label, _projector

    areas = areas if areas is not None else pd.DataFrame(columns=["Latitude", "Longitude"])
    frames = [f for f in (table, areas) if len(f)] or [f for f in (links,) if len(f)]
    lats = [v for f in frames for v in f["Latitude"].tolist()]
    lons = [v for f in frames for v in f["Longitude"].tolist()]
    if not lats:
        return _svg(width, height + LEGEND_H, [], "No refuges or raid areas to map")
    lat, lon, zoom = _fit_view(lats, lons, width, height)
    project = _projector(lat, lon, zoom, width, height)

    parts = []
    basemap = None
    try:
        basemap = _basemap_data_uri(lat, lon, zoom, width, height,
                                    style_name or DEFAULT_BASEMAP)
    except Exception:  # noqa: BLE001 - a map without terrain is still a map
        basemap = None
    if basemap:
        parts.append(f'<image href="{basemap[0]}" x="0" y="0" width="{width}" '
                     f'height="{height}" preserveAspectRatio="none"/>')
        parts += outline_parts(project, width, height,
                               levels=(boundaries.RANGE, boundaries.DIVISION),
                               styles={boundaries.RANGE: ("#7a5a2a", 0.6),
                                       boundaries.DIVISION: ("#5b3d12", 1.4)})
    else:
        parts.append(f'<rect width="{width}" height="{height}" fill="#f1f3ef"/>')
        parts += outline_parts(project, width, height,
                               levels=(boundaries.RANGE, boundaries.DIVISION))

    def inside(x, y, m=6):
        return -m <= x <= width + m and -m <= y <= height + m

    labels = []   # (priority, x, y, text)
    for r in areas.to_dict("records"):
        x, y = project(r["Latitude"], r["Longitude"])
        if not inside(x, y):
            continue
        size = 4 + 1.4 * math.sqrt(min(r["Raids"], 100))
        parts.append(f'<rect x="{x - size:.1f}" y="{y - size:.1f}" width="{2 * size:.1f}" '
                     f'height="{2 * size:.1f}" fill="{AREA_COLOR}" fill-opacity="0.35" '
                     f'stroke="{AREA_COLOR}" stroke-width="1.4"><title>{escape(r["Area"])}: '
                     f'{r["Raids"]} raids, no known refuge</title></rect>')
        labels.append((float(r["Raids"]), x, y - size - 5, r["Area"]))

    ref = table.set_index("Refuge") if len(table) else table
    for r in links.to_dict("records"):
        if r["Refuge"] not in ref.index:
            continue
        x0, y0 = project(ref.at[r["Refuge"], "Latitude"], ref.at[r["Refuge"], "Longitude"])
        x1, y1 = project(r["Latitude"], r["Longitude"])
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                     f'stroke="{RAID_COLOR}" stroke-opacity="0.55" stroke-width="0.8"/>')
        parts.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="1.6" fill="{RAID_COLOR}"/>')

    order = {rf.POSSIBLE: 0, rf.PROBABLE: 1, rf.CONFIRMED: 2}
    for r in sorted(table.to_dict("records"), key=lambda r: order.get(r["Confidence"], 0)):
        x, y = project(r["Latitude"], r["Longitude"])
        if not inside(x, y):
            continue
        radius = 4 + 1.1 * math.sqrt(min(r["Linked Raids"], 60))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
                     f'fill="{CONFIDENCE_COLORS[r["Confidence"]]}" fill-opacity="0.9" '
                     f'stroke="#ffffff" stroke-width="1.3"><title>{escape(r["Refuge"])}: '
                     f'{r["Linked Raids"]} linked raids</title></circle>')
        labels.append((float(r["Linked Raids"]) + 100 * order.get(r["Confidence"], 0),
                       x, y - radius - 5, r["Refuge"]))

    placed = []
    for _, x, y, text in sorted(labels, key=lambda c: -c[0]):
        if len(placed) >= MAP_LABELS:
            break
        if all(math.hypot(x - px, y - py) >= LABEL_GAP_PX for px, py in placed):
            placed.append((x, y))
            parts.append(_label(x, y, text))
    parts += scale_bar(lat, zoom, height)
    if basemap:
        parts.append(f'<rect x="{width - 230}" y="{height - 15}" width="230" height="15" '
                     f'fill="#ffffff" fill-opacity="0.75"/>'
                     f'<text x="{width - 6}" y="{height - 4.5}" text-anchor="end" '
                     f'font-size="9.5" fill="#3d4a43">{escape(basemap[1])}</text>')

    # Legend in its own band below the map, so it never covers a refuge.
    y0 = height + LEGEND_H / 2
    parts.append(f'<rect x="0" y="{height}" width="{width}" height="{LEGEND_H}" fill="#ffffff"/>')
    x = 14
    for name, color in CONFIDENCE_COLORS.items():
        parts.append(f'<circle cx="{x + 6}" cy="{y0}" r="6" fill="{color}" stroke="#ffffff"/>')
        parts.append(f'<text x="{x + 17}" y="{y0 + 4}" font-size="12" fill="#2b3530">'
                     f"{name} refuge</text>")
        x += 150
    parts.append(f'<line x1="{x}" x2="{x + 14}" y1="{y0}" y2="{y0}" stroke="{RAID_COLOR}" '
                 f'stroke-width="1.5"/><circle cx="{x + 14}" cy="{y0}" r="2" '
                 f'fill="{RAID_COLOR}"/>')
    parts.append(f'<text x="{x + 22}" y="{y0 + 4}" font-size="12" fill="#2b3530">'
                 "Linked night raid</text>")
    x += 150
    parts.append(f'<rect x="{x}" y="{y0 - 6}" width="12" height="12" fill="{AREA_COLOR}" '
                 f'fill-opacity="0.35" stroke="{AREA_COLOR}"/>')
    parts.append(f'<text x="{x + 18}" y="{y0 + 4}" font-size="12" fill="#2b3530">'
                 "Raid area, no known refuge (size = raids)</text>")
    return _svg(width, height + LEGEND_H, parts,
                "Elephant day refuges, linked raids and raid areas")


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


def brief_section(result: Dict[str, object], top_n: int = 12,
                  style_name: Optional[str] = None) -> str:
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
                   f"<div class='chart'>{map_svg(table, links, areas=areas, style_name=style_name)}"
                   "</div>")
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
