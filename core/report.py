"""Self-contained HTML brief.

Inline CSS, no external assets, so it can be emailed or printed without
the dashboard being online. It reads in the order a range officer works:
the findings, then the action plan (one row per action, every place it
applies to, why and when), then the evidence -- where, when, and what
the register says about the elephants themselves -- and closes with a
short appendix and the data's limits.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from html import escape
from typing import Dict, List, Optional

import pandas as pd

from core import brief_insights as insights
from core import report_charts
from core.analytics import (
    classify_conflict,
    division_conflict_rate,
    division_monthly_conflict,
    monthly_conflict_breakdown,
    seasonal_matrix,
    window_comparison,
)
from core.charts import division_colors
from core.config import NIGHT_HOUR_END, NIGHT_HOUR_START
from core.map_export import (
    filter_summary,
    sightings_map_svg,
    village_map_svg,
)
from core.intelligence import (
    TIER_CRITICAL,
    TIER_HIGH,
    TIER_WATCH,
    management_brief,
)
from core.map_engine import DEFAULT_BASEMAP
from core.spacetime import PATTERN_COLORS, PATTERN_MEANING, PATTERN_ORDER

logger = logging.getLogger(__name__)

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
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
                gap: 12px; margin: 16px 0 8px 0; }
    .kpi-card { background: #f0f6f2; border: 1px solid #dce8e0;
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
    .chart { margin: 10px 0 6px 0; break-inside: avoid; page-break-inside: avoid; }
    .chart svg { display: block; width: 100%; height: auto; }
    .chart-title { font-size: 13.5px; font-weight: 600; color: #16221c; margin: 18px 0 2px 0; }
    .chart-grid { display: flex; gap: 20px; flex-wrap: wrap; }
    .chart-grid > div { flex: 1 1 400px; min-width: 0; }
    .kpi-card .change { font-size: 12px; margin-top: 4px; color: #5a6b62; }
    .kpi-card .change b.up { color: #b3261e; }
    .kpi-card .change b.down { color: #1f5f3f; }
    table.heatmap { font-size: 12px; }
    table.heatmap td.heat { text-align: center; padding: 7px 4px;
                            -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    table.heatmap td.heat-na { background: #fafbfa; }
    .findings { margin: 14px 0 6px 0; padding: 0; list-style: none; counter-reset: f; }
    .findings li { position: relative; padding: 9px 12px 9px 40px; margin: 6px 0;
                   background: #f6faf7; border-radius: 6px; font-size: 14.5px; line-height: 1.45; }
    .findings li::before { counter-increment: f; content: counter(f); position: absolute;
                           left: 12px; top: 9px; width: 19px; height: 19px; border-radius: 50%;
                           background: #1f5f3f; color: #fff; font-size: 11.5px; font-weight: 700;
                           text-align: center; line-height: 19px; }
    table.plan td { padding: 10px; }
    table.plan td.what { font-weight: 600; color: #16221c; width: 30%; }
    table.plan td.when { white-space: nowrap; color: #33413a; }
    table.plan .why { color: #5a6b62; font-size: 12px; margin-top: 3px; }
    .kind { display: inline-block; width: 9px; height: 9px; border-radius: 2px;
            margin-right: 7px; vertical-align: 1px;
            -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .part { margin: 38px 0 0 0; font-size: 11.5px; font-weight: 700; letter-spacing: .12em;
            text-transform: uppercase; color: #789; border-top: 3px solid #1f5f3f;
            padding-top: 8px; }
    .part + h2 { margin-top: 10px; }
    .notes { font-size: 13.5px; color: #33413a; line-height: 1.5; margin: 8px 0 4px 0;
             padding-left: 18px; }
    .notes li { margin: 4px 0; }
    td .sub, .sub { color: #789; font-size: 11.5px; }
    .more { font-size: 12px; color: #5a6b62; margin: -12px 0 18px 0; }
    @media print { .header, .kpi-card, .chart svg { -webkit-print-color-adjust: exact;
                   print-color-adjust: exact; }
                   .part { break-before: auto; } h2 { break-after: avoid; } }
"""

# Colour keys for the action plan, so the kinds of work read at a glance.
_ACTION_COLORS = {
    "respond": "#b3261e", "warn": "#e8751a", "bulls": "#7A4A2A", "herds": "#1f5f3f",
    "survey": "#5b6b62", "refuge": "#2E6B4A", "search": "#4F8C6A", "crops": "#EE8A3C",
    "homes": "#D04E14", "verify": "#8C9A93", "confirm": "#C5CDC8",
}

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
    coverage: Optional[pd.DataFrame] = None,
    coverage_stats: Optional[Dict[str, object]] = None,
    refuges: Optional[Dict[str, object]] = None,
    emerging: Optional[pd.DataFrame] = None,
    emerging_step: str = "Month",
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
        coverage: Output of :func:`core.coverage.village_coverage`. The
            section is omitted when no registry was supplied.
        coverage_stats: Its companion summary, used for the caveats.
        refuges: Output of :func:`core.refuges.find_refuges`. Only refuges
            with night conflict linked to them are shown; the section is
            omitted when None.
        emerging: Output of :func:`core.spacetime.emerging_hotspots`.
            The section is omitted when None.
        emerging_step: The cube's time step, named in the section.

    Returns:
        A complete HTML document as a string.
    """
    as_of = None if end_date is None or pd.isna(end_date) else pd.Timestamp(end_date)
    brief = management_brief(df, recent_days=recent_days, as_of=as_of)
    kpis = brief["kpis"]
    change = window_comparison(df, recent_days, as_of=as_of)
    findings = insights.key_findings(df, brief, change, recent_days, emerging, refuges)
    plan = insights.action_plan(brief, recent_days, emerging, refuges, village_risk)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Elephant Conflict Brief</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>Elephant Conflict Brief</h1>
    <p>{escape(_format_period(start_date, end_date))} &nbsp;|&nbsp; Generated {datetime.now():%d %b %Y, %H:%M}</p>
    <p class="scope">{filter_summary(filters)}</p>
  </div>
  <div class="content">

    {_kpi_strip(kpis, change, recent_days, brief, emerging)}

    <h2>Key Findings</h2>
    {_findings_html(findings)}

    <h2>Action Plan <span class="sub">what to do, where, and when</span></h2>
    {_action_plan_html(plan)}

    <div class="part">Where</div>
    <h2>Priority Beats <span class="sub">Critical and High tiers, worst first</span></h2>
    {_priority_beats_html(brief["beats"])}

    <h2>Spatial View <span class="sub">sightings by conflict type, hotspot footprints to scale</span></h2>
    {sightings_map_svg(df, hotspots, basemap_style_name)}

    {_emerging_section(emerging, emerging_step)}

    <h2>Villages at Risk <span class="sub">conflict within reach of each village</span></h2>
    {village_map_svg(village_risk, hotspots, basemap_style_name)}
    {_village_risk_html(village_risk)}

    <h2>Movement Hotspots <span class="sub">where reports cluster on the ground</span></h2>
    {_hotspot_html(hotspots)}

    <h2>Escalating Beats <span class="sub">{escape(insights.trend_basis_phrase(brief, recent_days))}</span></h2>
    {_escalation_chart(brief["escalating"])}
    {_escalation_html(brief["escalating"], recent_days, brief["coverage"].get("trend_basis", ""))}

    <div class="part">When</div>
    {_trend_section(df, kpis, recent_days, as_of)}
    {_rhythm_section(df)}

    <div class="part">The Elephants</div>
    {_ecology_section(df, brief)}

    {_refuge_section(refuges, basemap_style_name)}

    {_coverage_section(coverage, coverage_stats)}

    <div class="part">Appendix</div>
    <h2>Conflict Rate by Division</h2>
    {_division_chart(df, kpis)}
    {_division_html(division_conflict_rate(df))}

    <h2>Reports by Type</h2>
    {_conflict_breakdown_table(df)}

    <h2>Notes</h2>
    {_notes_html(df, brief)}

    <div class="footer">
      Elephant Conflict Intelligence dashboard. Figures follow the filters set when the
      brief was generated.
    </div>
  </div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
def _caveats_html(caveats: List[str]) -> str:
    if not caveats:
        return ""
    items = "".join(f"<li>{escape(str(line))}</li>" for line in caveats)
    return f'<div class="caveats"><ul>{items}</ul></div>'


def _kpi_strip(kpis: Dict[str, float], change: Dict[str, float], recent_days: int,
               brief: Dict[str, object], emerging: Optional[pd.DataFrame]) -> str:
    deaths, injuries = int(kpis["human_deaths"]), int(kpis["human_injuries"])
    harm = "kpi-card alert" if (deaths or injuries) else "kpi-card"
    beats = brief["beats"]
    acting = 0 if beats.empty else int(beats["Priority Tier"].isin([TIER_CRITICAL,
                                                                    TIER_HIGH]).sum())
    cards = [
        f'<div class="kpi-card"><div class="label">Reports</div>'
        f'<div class="value">{kpis["entries"]:,}</div></div>',
        f'<div class="kpi-card"><div class="label">Conflict events</div>'
        f'<div class="value">{kpis["conflicts"]:,}</div>'
        f'{_change_html(change, "conflicts", recent_days)}</div>',
        f'<div class="{harm}"><div class="label">People killed</div>'
        f'<div class="value">{deaths}</div></div>',
        f'<div class="{harm}"><div class="label">People injured</div>'
        f'<div class="value">{injuries}</div></div>',
        f'<div class="kpi-card"><div class="label">Beats to act on</div>'
        f'<div class="value">{acting}</div>'
        f'<div class="change">of {len(beats)} with reports</div></div>',
    ]
    if emerging is not None and len(emerging):
        from core.spacetime import NEW

        new = int((emerging["Pattern"] == NEW).sum())
        cards.append(f'<div class="kpi-card"><div class="label">New hotspots</div>'
                     f'<div class="value">{new}</div>'
                     f'<div class="change">this month</div></div>')
    return '<div class="kpi-grid">' + "".join(cards) + "</div>"


def _findings_html(lines: List[str]) -> str:
    if not lines:
        return ""
    return '<ol class="findings">' + "".join(
        f"<li>{escape(line)}</li>" for line in lines) + "</ol>"


def _action_plan_html(plan: List["insights.Action"]) -> str:
    if not plan:
        return ('<p class="empty-note">Nothing in the period calls for action beyond '
                "routine monitoring.</p>")
    rows = "".join(
        "<tr>"
        f"<td class='what'><span class='kind' style='background:"
        f"{_ACTION_COLORS.get(item.kind, '#8C9A93')}'></span>{escape(item.action)}</td>"
        f"<td>{escape(item.where)}<div class='why'>{escape(item.why)}</div></td>"
        f"<td class='when'>{escape(item.when)}</td>"
        "</tr>"
        for item in plan
    )
    return ("<table class='plan'><thead><tr><th>Action</th><th>Where, and why</th>"
            f"<th>When</th></tr></thead><tbody>{rows}</tbody></table>")


def _priority_beats_html(beats: pd.DataFrame, top_n: int = 12) -> str:
    if beats.empty:
        return '<p class="empty-note">No beat data available for the current filters.</p>'
    focus = beats[beats["Priority Tier"].isin([TIER_CRITICAL, TIER_HIGH])]
    shown = (focus if len(focus) else beats).head(top_n)
    rows = []
    for row in shown.to_dict("records"):
        conf = " <span class='conf-low'>(thin data)</span>" if row["Confidence"] == "Low" else ""
        rows.append(
            "<tr>"
            f"<td><b>{escape(str(row['Beat']))}</b><br/>"
            f"<span class='sub'>{escape(str(row['Division']))} / {escape(str(row['Range']))}</span></td>"
            f"<td>{_tier_cell(row['Priority Tier'])}</td>"
            f"<td class='num'>{int(row['Conflict Events'])}"
            f"<br/><span class='sub'>of {int(row['Reports'])} reports</span></td>"
            f"<td class='num'>{_casualty_cell(row)}</td>"
            f"<td>{escape(insights.beat_driver(row))}{conf}</td>"
            "</tr>"
        )
    more = len(focus) - len(shown)
    note = (f"<p class='more'>{insights.plural(more, 'more beat')} in the Critical or High "
            "tier in the dashboard's beat table.</p>") if more > 0 else ""
    return (
        "<table><thead><tr><th>Beat</th><th>Tier</th><th class='num'>Conflicts</th>"
        "<th class='num'>Killed / injured</th><th>What is driving it</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + note
    )


def _escalation_html(escalating: pd.DataFrame, recent_days: int, basis: str = "",
                     top_n: int = 12) -> str:
    if escalating.empty:
        return '<p class="empty-note">No beat rose clearly faster than its baseline.</p>'
    from core.intelligence import TREND_BASIS_YEAR

    before = (f"Same {recent_days} days a year earlier" if basis == TREND_BASIS_YEAR
              else f"{recent_days} before")
    ordered = escalating.assign(
        _rise=escalating["Recent Conflicts"] - escalating["Prior Conflicts"]
    ).sort_values("_rise", ascending=False)
    rows = "".join(
        "<tr>"
        f"<td><b>{escape(str(r['Beat']))}</b><br/>"
        f"<span class='sub'>{escape(str(r['Division']))} / {escape(str(r['Range']))}</span></td>"
        f"<td class='num trend-up'>{int(r['Recent Conflicts'])}</td>"
        f"<td class='num'>{int(r['Prior Conflicts'])}</td>"
        f"<td>{escape(insights.beat_driver(r))}</td>"
        "</tr>"
        for r in ordered.head(top_n).to_dict("records")
    )
    more = len(ordered) - min(len(ordered), top_n)
    note = f"<p class='more'>{insights.plural(more, 'more escalating beat')}.</p>" if more else ""
    return (
        "<table><thead><tr><th>Beat</th>"
        f"<th class='num'>Last {recent_days} days</th><th class='num'>{escape(before)}</th>"
        "<th>What is driving it</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>{note}"
    )


def _rhythm_section(df: pd.DataFrame) -> str:
    """When in the day elephants are seen, and when in the year they raid."""
    try:
        hours = insights.diel_activity(df)
        months = insights.damage_by_month(df)
        left = right = ""
        if hours.sum() >= 10:
            peak = insights.peak_hours(hours)
            left = (_chart(report_charts.diel_clock_svg(hours, NIGHT_HOUR_START, NIGHT_HOUR_END),
                           "When elephants are seen")
                    + f"<p class='table-note'>Direct sightings only: signs and damage are "
                      f"written up the morning after. Busiest two hours: {escape(peak or '')}.</p>")
        if months.to_numpy().sum() > 0:
            right = _chart(report_charts.damage_season_svg(months), "When crops and homes are hit")
        if not (left or right):
            return ""
        return ("<h2>Daily and Seasonal Rhythm</h2><div class='chart-grid'>"
                f"<div>{left}</div><div>{right}</div></div>")
    except Exception:  # noqa: BLE001 - report generation must never crash
        logger.exception("Rhythm charts failed")
        return ""


def _ecology_section(df: pd.DataFrame, brief: Dict[str, object]) -> str:
    """What the register says about the animals: who, how many, where, when."""
    try:
        notes = insights.ecology_notes(df)
        parts = ["<h2>Who, How Many, and Where They Range</h2>"]
        if notes:
            parts.append('<ul class="notes">' + "".join(
                f"<li>{escape(n)}</li>" for n in notes) + "</ul>")
        composition = brief.get("composition")
        sizes = insights.herd_sizes(df)
        left = right = ""
        if composition is not None and len(composition):
            left = (_chart(report_charts.herd_composition_svg(composition),
                           "Kind of group, and how often it raided")
                    + _composition_table(composition))
        if sizes.sum() >= 10:
            right = _chart(report_charts.bars_svg(
                list(sizes.index), list(sizes.values), "Group size",
                y_title="Direct sightings", width=report_charts.HALF_WIDTH),
                "Size of the groups seen")
        if left or right:
            parts.append(f"<div class='chart-grid'><div>{left}</div><div>{right}</div></div>")
        used = insights.range_by_month(df)
        if len(used) >= 3:
            parts.append(_chart(report_charts.range_use_svg(used),
                                "How much of the landscape elephants used, month by month"))
        grid = insights.presence_grid(df)
        if not grid.empty and grid.shape[1] >= 2:
            parts.append(_chart(
                report_charts.seasonal_heatmap_svg(
                    grid, left=120, steps=report_charts.PRESENCE_STEPS,
                    label="Elephant reports by division and month"),
                "Where elephants were through the year: reports by division and month"))
        return "".join(parts) if len(parts) > 1 else ""
    except Exception:  # noqa: BLE001 - report generation must never crash
        logger.exception("Ecology section failed")
        return ""


def _composition_table(composition: pd.DataFrame) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['Group Type']))}</td>"
        f"<td class='num'>{int(row['Sightings']):,}</td>"
        f"<td class='num'>{int(row['Conflict Events']):,}</td>"
        f"<td class='num'>{_pct(row['Damage Rate %'])}</td>"
        f"<td class='num'>{row['Human Deaths']:.0f}</td>"
        "</tr>"
        for row in composition.to_dict("records")
    )
    return ("<table><thead><tr><th>Group</th><th class='num'>Reports</th>"
            "<th class='num'>Conflict</th><th class='num'>Raided</th>"
            f"<th class='num'>Killed</th></tr></thead><tbody>{rows}</tbody></table>")


def _notes_html(df: pd.DataFrame, brief: Dict[str, object]) -> str:
    """The data's limits, said once and plainly."""
    window = brief["coverage"].get("critical_window_days", 90)
    notes = [
        "Counts follow patrol effort as well as elephant activity: a beat watched more "
        "closely reports more.",
        f"Critical means a death or injury in the last {window} days; an older casualty "
        "puts a beat in High.",
        "Signs and damage are written up the morning after, so they are dated to the night "
        "before and their hour is not used.",
        "Group type is read from the male, female and calf counts; conflict with none "
        "recorded is left out of the bull and herd shares.",
    ]
    beats = brief["beats"]
    if not beats.empty:
        thin = int((beats["Confidence"] == "Low").sum())
        if thin:
            notes.insert(1, f"{thin} of {len(beats)} beats have fewer than 10 reports; their "
                            "rates are pulled toward the landscape average and marked thin data.")
    return '<ul class="notes">' + "".join(f"<li>{escape(n)}</li>" for n in notes) + "</ul>"


def _hotspot_html(hotspots: Optional[pd.DataFrame], top_n: int = 10) -> str:
    if hotspots is None:
        return (
            '<p class="empty-note">Hotspot clustering was not run for this brief.</p>'
        )
    if hotspots.empty:
        return (
            '<p class="empty-note">No density hotspots at the current settings: '
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
        f"<td class='num'>{r['Radius (km)']:.1f} km</td>"
        f"<td class='num'>{r['Centre Latitude']:.4f},<br/>{r['Centre Longitude']:.4f}</td>"
        "</tr>"
        for r in hotspots.head(top_n).to_dict("records")
    )
    return (
        "<table><thead><tr><th>ID</th><th>Tier</th><th>Where</th>"
        "<th class='num'>Sightings</th><th class='num'>Conflicts</th>"
        "<th class='num'>Conflict share</th><th class='num'>Killed / injured</th>"
        "<th class='num'>Radius</th><th class='num'>Centre</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _emerging_section(emerging: Optional[pd.DataFrame], step: str, top_n: int = 10) -> str:
    if emerging is None:
        return ""
    head = (f"<h2>Emerging Hotspots <span class='sub'>places turning hot, {escape(step.lower())} "
            "by month</span></h2>") if step == "Month" else (
        f"<h2>Emerging Hotspots <span class='sub'>places turning hot, "
        f"{escape(step.lower())} by {escape(step.lower())}</span></h2>")
    named = emerging[emerging["Place"] != ""] if not emerging.empty else emerging
    if named.empty:
        return head + (
            '<p class="empty-note">No place was a significant hotspot in any '
            f"{escape(step.lower())} of the period, or the period is too short to say.</p>"
        )
    fmt = "%b %Y" if step == "Month" else "%d %b %Y"
    unit = step.lower()
    rows = "".join(
        "<tr>"
        f"<td><b>{escape(str(r['Place']))}</b></td>"
        f"<td><span style='color:{PATTERN_COLORS[r['Pattern']]}'>&#9632;</span> "
        f"{escape(r['Pattern'])}</td>"
        f"<td>{escape(str(r['Nearest Village']) or '-')}<br/>"
        f"<span class='sub'>{escape(str(r['Beats']))}</span></td>"
        f"<td class='num'>{int(r['Events']):,}</td>"
        f"<td class='num'>{int(r['Recent Events']):,}</td>"
        f"<td class='num'>{int(r['Hot Steps'])} / {int(r['Steps'])}</td>"
        f"<td>{escape(r['Trend'])}</td>"
        f"<td class='num'>{r['Last Hot']:{fmt}}</td>"
        "</tr>"
        for r in named.head(top_n).to_dict("records")
    )
    legend = "".join(
        f"<li><b>{escape(p)}</b>: {escape(PATTERN_MEANING[p].replace('step', unit))}</li>"
        for p in PATTERN_ORDER[:-1] if (named["Pattern"] == p).any())
    return (
        head
        + "<table><thead><tr><th>ID</th><th>Pattern</th><th>Nearest village</th>"
        f"<th class='num'>Events</th><th class='num'>Last 3 {escape(unit)}s</th>"
        f"<th class='num'>Hot {escape(unit)}s</th><th>Trend</th><th class='num'>Last hot</th>"
        f"</tr></thead><tbody>{rows}</tbody></table><ul class='notes'>{legend}</ul>"
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
        "<th>Nearest hotspot</th><th class='num'>To its centre</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _refuge_section(refuges: Optional[Dict[str, object]], style_name: str) -> str:
    if refuges is None:
        return ""
    from core.refuge_report import brief_section

    return brief_section(refuges, style_name=style_name)


def _coverage_section(
    coverage: Optional[pd.DataFrame], stats: Optional[Dict[str, object]]
) -> str:
    """Early-warning gaps, omitted entirely when no registry was loaded.

    An empty heading reading "no data" in a forwarded document invites
    the reader to conclude there are no gaps, which is the opposite of
    what it means.
    """
    if coverage is None or coverage.empty:
        return ""

    # Imported here: the brief is the only caller, and core.coverage
    # pulls in scipy that a report without a registry never needs.
    from core.coverage import COVERAGE_NONE, coverage_caveats, coverage_gaps

    stats = stats or {}
    gaps = coverage_gaps(coverage)

    if gaps.empty:
        body = (
            '<p class="empty-note">Every Critical and High village has at least '
            f"{int(stats.get('min_contacts', 3))} registered contacts.</p>"
        )
    else:
        rows = "".join(
            "<tr>"
            f"<td><b>{escape(str(r['Village']))}</b></td>"
            f"<td>{_tier_cell(r['Tier'])}</td>"
            f"<td class='num'>{int(r['Registered Contacts'])}</td>"
            f"<td>{escape(str(r['Coverage']))}</td>"
            f"<td class='num'>{int(r['Conflict Events']):,}</td>"
            f"<td class='num'>{int(r['Human Deaths'])} / {int(r['People Injured'])}</td>"
            "</tr>"
            for r in gaps.to_dict("records")
        )
        no_contact = int((gaps["Coverage"] == COVERAGE_NONE).sum())
        body = (
            "<table><thead><tr><th>Village</th><th>Tier</th>"
            "<th class='num'>Registered</th><th>Cover</th>"
            "<th class='num'>Conflicts</th><th class='num'>Killed / injured</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
            f"<p class='table-note'>{len(gaps)} village(s) in the worst two tiers "
            f"are short of early-warning contacts, {no_contact} with nobody "
            "enrolled at all.</p>"
        )

    return (
        '<h2>Early-Warning Coverage <span class="sub">who can be warned where '
        "it matters</span></h2>"
        + body
        + _caveats_html(coverage_caveats(stats))
    )


def _tier_cell(tier: object) -> str:
    """Tier badge carrying a glyph as well as colour."""
    label = str(tier)
    css = _TIER_CLASSES.get(label, "tier-routine")
    glyph = _TIER_GLYPHS.get(label, "○")
    return f"<span class='tier {css}'>{glyph} {escape(label)}</span>"


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


def _chart(svg: str, title: str = "") -> str:
    head = f"<div class='chart-title'>{escape(title)}</div>" if title else ""
    return f"{head}<div class='chart'>{svg}</div>"


def _change_html(change: Dict[str, float], key: str, days: int,
                 noun: str = "") -> str:
    """Recent window against the one before, for a KPI card."""
    recent, prior = change.get(f"{key}_recent"), change.get(f"{key}_prior")
    if recent is None or prior is None or pd.isna(recent) or pd.isna(prior):
        return ""
    diff = recent - prior
    css = "up" if diff > 0 else "down" if diff < 0 else ""
    pct = f" ({diff / prior:+.0%})" if prior > 0 else ""
    what = f" {noun}" if noun else ""
    return (
        f"<div class='change'><b class='{css}'>{diff:+,.0f}{pct}</b>{what} "
        f"last {days} days vs the {days} before</div>"
    )


def _trend_section(df: pd.DataFrame, kpis: Dict[str, float], recent_days: int,
                   as_of: Optional[pd.Timestamp]) -> str:
    """The trend charts, drawn as inline SVG so the brief stays one file.

    Never allowed to take the brief down: a chart that cannot be drawn is
    replaced by a note, and the tables further down still carry the data.
    """
    try:
        monthly = monthly_conflict_breakdown(df)
        if len(monthly) < 2:
            return (
                "<h2>Conflict Trends</h2><p class='empty-note'>The period covers "
                "less than two months, so there is no trend to draw.</p>"
            )
        parts = [
            "<h2>Conflict Trends <span class='sub'>is it getting worse, where, "
            "and when in the year</span></h2>",
            _chart(report_charts.conflict_type_trend_svg(monthly, recent_days, as_of),
                   "Conflict events per month, by type"),
            "<p class='table-note'>Darkest at the base is the most severe. A rising "
            "dark band matters more than a rising total. The shaded band is the "
            "escalation window.</p>",
            "<div class='chart-grid'><div>",
            _chart(report_charts.conflict_rate_svg(monthly, kpis["conflict_rate"]),
                   "Conflict rate"),
            "</div><div>",
        ]
        if float((monthly["Human Deaths"] + monthly["People Injured"]).sum()) > 0:
            parts.append(_chart(report_charts.casualty_svg(monthly), "Casualties"))
        else:
            parts.append("<div class='chart-title'>Casualties</div>"
                         "<p class='empty-note'>None recorded in the period.</p>")
        parts.append("</div></div>")
        parts.append("<div class='chart-title'>Seasonal pattern: conflict events "
                     "by month and year</div>")
        parts.append(report_charts.seasonal_heatmap_html(seasonal_matrix(df)))
        divisions = division_monthly_conflict(df)
        if divisions["Division"].nunique() >= 2:
            parts.append(_chart(
                report_charts.division_trend_svg(
                    divisions, division_colors(df["Division"].unique())
                ),
                "By division",
            ))
        return "".join(parts)
    except Exception:  # noqa: BLE001 - report generation must never crash
        logger.exception("Trend charts failed; brief generated without them")
        return (
            "<h2>Conflict Trends</h2><p class='empty-note'>The trend charts could "
            "not be drawn for this selection.</p>"
        )


def _escalation_chart(escalating: pd.DataFrame) -> str:
    if escalating.empty or "Recent Conflicts" not in escalating.columns:
        return ""
    try:
        return _chart(report_charts.escalation_svg(escalating))
    except Exception:  # noqa: BLE001 - the table below carries the same figures
        logger.exception("Escalation chart failed")
        return ""


def _division_chart(df: pd.DataFrame, kpis: Dict[str, float]) -> str:
    rates = division_conflict_rate(df)
    if rates.empty:
        return ""
    try:
        return _chart(report_charts.division_rate_svg(
            rates, kpis["conflict_rate"], division_colors(df["Division"].unique())
        ))
    except Exception:  # noqa: BLE001 - the table below carries the same figures
        logger.exception("Division chart failed")
        return ""


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
