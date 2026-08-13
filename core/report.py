"""Self-contained HTML brief.

Inline CSS, no external assets, so it can be emailed or printed without
the dashboard being online. Leads with what needs deciding, keeps the
evidence beside each claim, and closes with the data's limits.
"""

from __future__ import annotations

from datetime import date, datetime
from html import escape
from typing import Dict, List, Optional

import pandas as pd

from core.analytics import (
    classify_conflict,
    division_conflict_rate,
    night_day_comparison,
    severity_distribution,
)
from core.map_export import (
    filter_summary,
    sightings_map_svg,
    village_map_svg,
)
from core.intelligence import (
    TIER_CRITICAL,
    TIER_HIGH,
    TIER_WATCH,
    format_window,
    management_brief,
)
from core.map_engine import DEFAULT_BASEMAP

_CSS = """
    body { font-family: 'Segoe UI', Arial, sans-serif; color: #222; margin: 0; background: #f4f6f5; }
    .wrap { max-width: 960px; margin: 0 auto; padding: 32px; }
    .header { background: #1f5f3f; color: #fff; padding: 28px 32px; border-radius: 10px 10px 0 0; }
    .header h1 { margin: 0 0 6px 0; font-size: 26px; }
    .header p { margin: 0; opacity: 0.9; font-size: 14px; }
    .content { background: #fff; padding: 28px 32px; border-radius: 0 0 10px 10px;
               box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    h2 { font-size: 18px; color: #1f5f3f; border-bottom: 2px solid #e5ece8; padding-bottom: 6px;
         margin-top: 30px; }
    h2 .sub { font-weight: 400; font-size: 13px; color: #789; margin-left: 8px; }
    .kpi-grid { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0 8px 0; }
    .kpi-card { flex: 1; min-width: 130px; background: #f0f6f2; border: 1px solid #dce8e0;
                border-radius: 8px; padding: 14px 16px; }
    .kpi-card.alert { background: #fdf0ee; border-color: #f0cdc7; }
    .kpi-card .label { font-size: 12px; color: #567; text-transform: uppercase; letter-spacing: .04em; }
    .kpi-card .value { font-size: 24px; font-weight: 700; color: #1f5f3f; margin-top: 4px; }
    .kpi-card.alert .value { color: #b3261e; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0 20px 0; font-size: 13px; }
    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e8ece9; vertical-align: top; }
    th { background: #f0f6f2; color: #1f5f3f; font-weight: 600; }
    tr:hover { background: #fafcfb; }
    td.num, th.num { text-align: right; }
    .tier { font-weight: 700; padding: 2px 7px; border-radius: 4px; font-size: 11.5px;
            text-transform: uppercase; letter-spacing: .03em; white-space: nowrap; }
    .tier-critical { background: #b3261e; color: #fff; }
    .tier-high { background: #e8751a; color: #fff; }
    .tier-watch { background: #f5d76e; color: #4a3a00; }
    .tier-routine { background: #e7eeea; color: #46614f; }
    .trend-up { color: #b3261e; font-weight: 600; }
    .trend-down { color: #1f5f3f; }
    .conf-low { color: #a86a00; }
    .headlines { background: #f0f6f2; border-left: 4px solid #1f5f3f; padding: 14px 18px;
                 margin: 16px 0; border-radius: 0 6px 6px 0; }
    .headlines ul { margin: 0; padding-left: 18px; }
    .headlines li { margin: 5px 0; font-size: 14.5px; }
    .caveats { background: #fbfaf4; border: 1px solid #ece7d5; border-radius: 6px;
               padding: 12px 18px; margin-top: 12px; }
    .caveats li { font-size: 12.5px; color: #5a5545; margin: 5px 0; }
    .action { font-size: 12.5px; color: #33413a; }
    .footer { text-align: center; color: #99a; font-size: 11px; margin-top: 24px; }
    .empty-note { color: #789; font-style: italic; }
    .table-note { font-size: 12px; color: #5a6b62; margin: 6px 0 0 0; line-height: 1.5; }
    .header .scope { margin-top: 6px; font-size: 13px; opacity: 0.95; }
    .map-figure { margin: 12px 0 20px 0; border: 1px solid #dce8e0; border-radius: 8px;
                  overflow: hidden; background: #fff; }
    .map-figure svg { display: block; width: 100%; height: auto; }
"""

_TIER_CLASSES = {
    TIER_CRITICAL: "tier-critical",
    TIER_HIGH: "tier-high",
    TIER_WATCH: "tier-watch",
}

# Glyphs so the ranking survives a greyscale photocopy.
_TIER_GLYPHS = {
    TIER_CRITICAL: "▲",
    TIER_HIGH: "◆",
    TIER_WATCH: "●",
    "Routine": "○",
}


def generate_html_report(
    df: pd.DataFrame,
    start_date: Optional[date],
    end_date: Optional[date],
    recent_days: int = 90,
    hotspots: Optional[pd.DataFrame] = None,
    village_risk: Optional[pd.DataFrame] = None,
    filters: Optional[Dict[str, object]] = None,
    basemap_style_name: str = DEFAULT_BASEMAP,
) -> str:
    """Build a complete, styled conservation intelligence brief.

    Args:
        df: Filtered/enriched dataframe used to compute every section.
            Safe to call with an empty dataframe.
        start_date: Period start to display.
        end_date: Period end to display.
        recent_days: Escalation comparison window passed through to
            :func:`core.intelligence.management_brief`.
        hotspots: Output of :func:`core.hotspots.detect_hotspots`.
            Omitted from the document when None.
        village_risk: Output of :func:`core.hotspots.villages_at_risk`.
        filters: Active sidebar filters, named in the header so the
            reader knows which divisions the figures cover.
        basemap_style_name: Basemap for the embedded static maps.

    Returns:
        A complete HTML document as a string.
    """
    brief = management_brief(df, recent_days=recent_days)
    kpis = brief["kpis"]

    period_str = _format_period(start_date, end_date)
    night_pct_str = (
        "N/A" if pd.isna(kpis["night_pct"]) else f"{kpis['night_pct']:.1f}%"
    )
    deaths = int(kpis["human_deaths"])
    injuries = int(kpis["human_injuries"])
    casualty_class = "kpi-card alert" if (deaths or injuries) else "kpi-card"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Elephant Conflict Intelligence Brief</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>Elephant Conflict Intelligence Brief</h1>
    <p>Period covered: {escape(period_str)} &nbsp;|&nbsp; Generated: {datetime.now():%d %b %Y, %H:%M}</p>
    <p class="scope">{filter_summary(filters)}</p>
  </div>
  <div class="content">

    <h2>Assessment</h2>
    {_headlines_html(brief["headlines"])}

    <div class="kpi-grid">
      <div class="kpi-card"><div class="label">Reports</div><div class="value">{kpis['entries']:,}</div></div>
      <div class="kpi-card"><div class="label">Conflict Events</div><div class="value">{kpis['conflicts']:,}</div></div>
      <div class="{casualty_class}"><div class="label">People Killed</div><div class="value">{deaths}</div></div>
      <div class="{casualty_class}"><div class="label">People Injured</div><div class="value">{injuries}</div></div>
      <div class="kpi-card"><div class="label">Night Share</div><div class="value">{night_pct_str}</div></div>
    </div>

    <h2>Priority Beats <span class="sub">tier set by fixed rules; score orders within tier</span></h2>
    {_beat_table_html(brief["beats"])}

    <h2>Which Animals <span class="sub">damage rate by the kind of group seen</span></h2>
    {_composition_html(brief.get("composition"))}

    <h2>Spatial View <span class="sub">sightings by conflict type, hotspot footprints to scale</span></h2>
    {sightings_map_svg(df, hotspots, basemap_style_name)}

    <h2>Movement Hotspots <span class="sub">density clusters in the point data</span></h2>
    {_hotspot_html(hotspots)}

    <h2>Villages at Risk</h2>
    {village_map_svg(village_risk, hotspots, basemap_style_name)}
    {_village_risk_html(village_risk)}

    <h2>Escalating Beats <span class="sub">last {brief['coverage']['recent_days']} days vs the window before</span></h2>
    {_escalation_html(brief["escalating"])}

    <h2>Timing <span class="sub">when to staff</span></h2>
    {_temporal_html(brief["temporal"])}

    <h2>Conflict Type Breakdown</h2>
    {_conflict_breakdown_table(df)}

    <h2>Conflict Rate by Division</h2>
    {_division_html(division_conflict_rate(df))}

    <h2>Severity Distribution</h2>
    {_bands_table_html(severity_distribution(df))}

    <h2>Night vs. Day</h2>
    {_night_day_table_html(night_day_comparison(df))}

    <h2>How to read this</h2>
    {_caveats_html(brief["caveats"])}

    <div class="footer">
      Generated by the Elephant Sighting &amp; Conflict Dashboard. Figures reflect the
      filters active at the time of generation.
    </div>
  </div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
def _headlines_html(headlines: List[str]) -> str:
    if not headlines:
        return '<p class="empty-note">No findings for the current filters.</p>'
    items = "".join(f"<li>{escape(str(line))}</li>" for line in headlines)
    return f'<div class="headlines"><ul>{items}</ul></div>'


def _caveats_html(caveats: List[str]) -> str:
    if not caveats:
        return ""
    items = "".join(f"<li>{escape(str(line))}</li>" for line in caveats)
    return f'<div class="caveats"><ul>{items}</ul></div>'


def _beat_table_html(beats: pd.DataFrame, top_n: int = 15) -> str:
    if beats.empty:
        return '<p class="empty-note">No beat data available for the current filters.</p>'

    rows = []
    for row in beats.head(top_n).to_dict("records"):
        conf = str(row["Confidence"])
        conf_html = (
            f'<span class="conf-low">{escape(conf)}</span>' if conf == "Low" else escape(conf)
        )
        rows.append(
            "<tr>"
            f"<td><b>{escape(str(row['Beat']))}</b><br/>"
            f"<span class='sub'>{escape(str(row['Division']))} / {escape(str(row['Range']))}</span></td>"
            f"<td>{_tier_cell(row['Priority Tier'])}</td>"
            f"<td class='num'>{row['Priority Score']:.0f}</td>"
            f"<td class='num'>{int(row['Reports'])}</td>"
            f"<td class='num'>{int(row['Conflict Events'])}</td>"
            f"<td class='num'>{_pct(row['Adj. Conflict Rate %'])}</td>"
            f"<td class='num'>{_casualty_cell(row)}</td>"
            f"<td class='num'>{_pct(row['Night Conflict %'])}</td>"
            f"<td>{_trend_html(row['Trend'], row['Recent vs Prior'])}</td>"
            f"<td>{conf_html}</td>"
            f"<td class='action'>{escape(str(row['Recommended Action']))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Beat</th><th>Tier</th><th class='num'>Score</th><th class='num'>Reports</th>"
        "<th class='num'>Conflicts</th><th class='num'>Adj. rate</th>"
        "<th class='num'>Killed / injured<br/><span class='sub'>recent</span></th>"
        "<th class='num'>Night</th>"
        "<th>Trend</th><th>Confidence</th><th>Recommended action</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _composition_html(composition: Optional[pd.DataFrame]) -> str:
    """Bull against breeding herd, which decides how a response is run."""
    if composition is None or composition.empty:
        return '<p class="empty-note">No group composition recorded.</p>'

    rows = []
    for row in composition.to_dict("records"):
        rows.append(
            "<tr>"
            f"<td><b>{escape(str(row['Group Type']))}</b></td>"
            f"<td class='num'>{int(row['Sightings']):,}</td>"
            f"<td class='num'>{int(row['Conflict Events']):,}</td>"
            f"<td class='num'>{_pct(row['Damage Rate %'])}</td>"
            f"<td class='num'>{row['Human Deaths']:.0f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Group</th><th class='num'>Sightings</th>"
        "<th class='num'>Conflicts</th><th class='num'>Damage rate</th>"
        "<th class='num'>Killed</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        "<p class='table-note'>Only bulls carry tusks in this species, so the male "
        "count is the tusker count. A bull raids and occasionally kills; a "
        "breeding herd with calves is avoiding people. The two take opposite "
        "handling, and the beat table carries the split per beat.</p>"
    )


def _escalation_html(escalating: pd.DataFrame) -> str:
    if escalating.empty:
        return (
            '<p class="empty-note">No beat shows a materially higher conflict count '
            "than in the preceding window.</p>"
        )
    rows = "".join(
        "<tr>"
        f"<td><b>{escape(str(r['Beat']))}</b></td>"
        f"<td>{escape(str(r['Division']))} / {escape(str(r['Range']))}</td>"
        f"<td class='num trend-up'>{escape(str(r['Recent vs Prior']))}</td>"
        f"<td class='num'>{int(r['Human Deaths'])} / {int(r['People Injured'])}</td>"
        f"<td class='action'>{escape(str(r['Recommended Action']))}</td>"
        "</tr>"
        for r in escalating.to_dict("records")
    )
    return (
        "<table><thead><tr><th>Beat</th><th>Division / Range</th>"
        "<th class='num'>Recent vs prior</th><th class='num'>Killed / injured</th>"
        "<th>Recommended action</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _hotspot_html(hotspots: Optional[pd.DataFrame], top_n: int = 10) -> str:
    if hotspots is None:
        return (
            '<p class="empty-note">Hotspot clustering was not run for this brief.</p>'
        )
    if hotspots.empty:
        return (
            '<p class="empty-note">No density hotspots at the current settings -- '
            "sightings are too dispersed to form clusters.</p>"
        )

    rows = "".join(
        "<tr>"
        f"<td><b>{escape(str(r['Hotspot']))}</b></td>"
        f"<td>{_tier_cell(r['Tier'])}</td>"
        f"<td>{escape(str(r['Divisions']))}<br/>"
        f"<span class='sub'>{escape(str(r['Beats']))}</span></td>"
        f"<td class='num'>{int(r['Sightings']):,}</td>"
        f"<td class='num'>{int(r['Conflict Events']):,}</td>"
        f"<td class='num'>{r['Conflict Share %']:.0f}%</td>"
        f"<td class='num'>{int(r['Human Deaths'])} / {int(r['People Injured'])}</td>"
        f"<td class='num'>{_pct(r['Night Share %'])}</td>"
        f"<td class='num'>{r['Radius (km)']:.1f} km</td>"
        f"<td class='num'>{r['Centre Latitude']:.4f},<br/>{r['Centre Longitude']:.4f}</td>"
        "</tr>"
        for r in hotspots.head(top_n).to_dict("records")
    )
    return (
        "<table><thead><tr><th>ID</th><th>Tier</th><th>Where</th>"
        "<th class='num'>Sightings</th><th class='num'>Conflicts</th>"
        "<th class='num'>Share</th><th class='num'>Killed / injured</th>"
        "<th class='num'>Night</th><th class='num'>Radius</th>"
        "<th class='num'>Centre</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _village_risk_html(village_risk: Optional[pd.DataFrame], top_n: int = 15) -> str:
    if village_risk is None or village_risk.empty:
        return (
            '<p class="empty-note">No village-level ranking. Sighting exports carry '
            "no village field, so villages can only be identified from a centroids "
            "file (Village, Latitude, Longitude).</p>"
        )

    rows = "".join(
        "<tr>"
        f"<td>{escape(str(r['Village']))}</td>"
        f"<td>{_tier_cell(r['Tier'])}</td>"
        f"<td class='num'>{int(r['Conflict Events']):,}</td>"
        f"<td class='num'>{int(r['Human Deaths'])} / {int(r['People Injured'])}</td>"
        f"<td class='num'>{int(r['House Damage Events'])}</td>"
        f"<td class='num'>{int(r['Crop Damage Events'])}</td>"
        f"<td class='num'>{_pct(r['Night Share %'])}</td>"
        f"<td>{escape(str(r['Nearest Hotspot']))}"
        f"{' (inside)' if r['Inside Hotspot'] else ''}</td>"
        f"<td class='num'>{_km(r['Distance to Hotspot (km)'])}</td>"
        "</tr>"
        for r in village_risk.head(top_n).to_dict("records")
    )
    return (
        "<table><thead><tr><th>Village</th><th>Tier</th>"
        "<th class='num'>Conflicts</th><th class='num'>Killed / injured</th>"
        "<th class='num'>House</th><th class='num'>Crop</th>"
        "<th class='num'>Night</th><th>Hotspot</th>"
        "<th class='num'>Distance</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _tier_cell(tier: object) -> str:
    """Tier badge carrying a glyph as well as colour."""
    label = str(tier)
    css = _TIER_CLASSES.get(label, "tier-routine")
    glyph = _TIER_GLYPHS.get(label, "○")
    return f"<span class='tier {css}'>{glyph} {escape(label)}</span>"


def _temporal_html(temporal: Dict[str, object]) -> str:
    window = temporal.get("peak_window")
    months = temporal.get("peak_months") or []
    night_share = temporal.get("night_share", float("nan"))

    parts = [f"<p><b>Peak window:</b> {escape(format_window(window))}</p>"]
    if not pd.isna(night_share):
        parts.append(f"<p><b>Night share of conflict:</b> {night_share:.0f}%</p>")
    if months:
        parts.append(f"<p><b>Seasonal peak:</b> {escape(', '.join(months))}</p>")
    else:
        parts.append(
            "<p><b>Seasonal peak:</b> none marked -- conflict is spread through the year.</p>"
        )

    hourly = temporal.get("hourly")
    if hourly is not None and len(hourly) and float(hourly.sum()) > 0:
        cells = "".join(
            f"<td class='num'>{int(v)}</td>" for v in hourly.to_numpy()
        )
        heads = "".join(f"<th class='num'>{h:02d}</th>" for h in range(24))
        parts.append(
            "<table style='font-size:11.5px;'><thead><tr>"
            f"<th>Hour</th>{heads}</tr></thead><tbody><tr><td><b>Events</b></td>{cells}"
            "</tr></tbody></table>"
        )
    return "".join(parts)



def _division_html(rates: pd.DataFrame) -> str:
    if rates.empty:
        return '<p class="empty-note">No division data available for the current filters.</p>'
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(division))}</td>"
        f"<td class='num'>{int(r['Sightings'])}</td>"
        f"<td class='num'>{int(r['Conflict Events'])}</td>"
        f"<td class='num'>{int(r['Human Deaths'])}</td>"
        f"<td class='num'>{r['Conflict Rate %']:.1f}%</td>"
        "</tr>"
        for division, r in rates.iterrows()
    )
    return (
        "<table><thead><tr><th>Division</th><th class='num'>Reports</th>"
        "<th class='num'>Conflict events</th><th class='num'>Killed</th>"
        "<th class='num'>Conflict rate</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _bands_table_html(bands: pd.DataFrame) -> str:
    if bands.empty:
        return '<p class="empty-note">No severity data available for the current filters.</p>'
    rows = "".join(
        f"<tr><td>{escape(str(r.Band))}</td><td class='num'>{int(r.Count)}</td></tr>"
        for r in bands.itertuples()
    )
    return (
        "<table><thead><tr><th>Severity band</th><th class='num'>Reports</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _night_day_table_html(night_day: pd.DataFrame) -> str:
    if night_day.empty:
        return '<p class="empty-note">Night/day classification unavailable for the current filters.</p>'
    rows = "".join(
        f"<tr><td>{escape(str(row['Period']))}</td>"
        f"<td class='num'>{int(row['Entries'])}</td>"
        f"<td class='num'>{row['Avg Severity']:.1f}</td></tr>"
        for _, row in night_day.iterrows()
    )
    return (
        "<table><thead><tr><th>Period</th><th class='num'>Reports</th>"
        "<th class='num'>Avg severity</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _conflict_breakdown_table(df: pd.DataFrame) -> str:
    """Counts by conflict category, one row per report.

    Counted under the most severe category that applies, so rows sum to
    the report total.
    """
    if df.empty:
        return '<p class="empty-note">No incident data available for the current filters.</p>'

    counts = classify_conflict(df).value_counts()
    labels = {
        "Death": "Human fatality",
        "Injury": "Human injury",
        "House": "House / store damage",
        "Crop": "Crop or grain loss",
        "Presence": "Presence only (no damage)",
    }
    rows = "".join(
        f"<tr><td>{escape(label)}</td><td class='num'>{int(counts.get(key, 0))}</td></tr>"
        for key, label in labels.items()
    )
    return (
        "<table><thead><tr><th>Conflict type</th><th class='num'>Reports</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def _format_period(start_date: Optional[date], end_date: Optional[date]) -> str:
    if start_date is None or end_date is None or pd.isna(start_date) or pd.isna(end_date):
        return "N/A"
    try:
        return f"{pd.Timestamp(start_date):%d %b %Y} to {pd.Timestamp(end_date):%d %b %Y}"
    except Exception:  # noqa: BLE001 - report generation must never crash
        return "N/A"


def _casualty_cell(row: dict) -> str:
    """Recent killed/injured, with the period total when it is larger.

    Critical keys off the recent figure alone, so a beat showing "0 / 0"
    may still have killed someone earlier in the period.
    """
    recent = f"{int(row['Recent Deaths'])} / {int(row['Recent Injuries'])}"
    total_deaths = int(row["Human Deaths"])
    total_injuries = int(row["People Injured"])

    if total_deaths > int(row["Recent Deaths"]) or total_injuries > int(
        row["Recent Injuries"]
    ):
        return (
            f"{recent}<br/><span class='sub'>{total_deaths} / {total_injuries} "
            "in period</span>"
        )
    return recent


def _pct(value: object) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.0f}%"


def _km(value: object) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.1f} km"


def _trend_html(trend: object, detail: object) -> str:
    label = escape(str(trend))
    detail_str = escape(str(detail))
    if trend == "Escalating":
        return f"<span class='trend-up'>{label}</span><br/><span class='sub'>{detail_str}</span>"
    if trend == "Easing":
        return f"<span class='trend-down'>{label}</span><br/><span class='sub'>{detail_str}</span>"
    return f"{label}<br/><span class='sub'>{detail_str}</span>"
