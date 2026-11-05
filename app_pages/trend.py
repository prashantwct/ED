"""Is it getting worse? Monthly conflict, rates, casualties, season, divisions."""

from __future__ import annotations

import logging

import streamlit as st

from app_pages.common import _figure
from core import brief_insights, charts, report_charts
from core.analytics import (
    division_conflict_rate,
    division_monthly_conflict,
    seasonal_matrix,
)


logger = logging.getLogger(__name__)


def render(c) -> None:
    brief = c.brief
    division_palette = c.division_palette
    escalating = c.escalating
    filtered = c.history
    kpis = c.kpis_history
    monthly = c.monthly
    period_end = c.period_end
    recent_days = c.recent_days

    if len(monthly) < 2:
        st.info("The selected period covers less than two months, so there is no trend "
                "to draw.")
    else:
        # Each figure downloads from its SVG twin. Values are bound as
        # default arguments: the callable runs on click, after this script
        # has moved on and reassigned names like ``monthly``.
        _figure(
            "Conflict events per month, by type",
            charts.conflict_type_trend(monthly, recent_days=recent_days, as_of=period_end),
            "Conflict events per month by type",
            lambda m=monthly, d=recent_days, e=period_end:
            report_charts.conflict_type_trend_svg(m, d, e),
            note=f"Darkest at the base is the most severe; the shaded band is the last "
                 f"{recent_days} days (set on the Analyst page).",
        )

        rate_col, casualty_col = st.columns(2)
        with rate_col:
            _figure("Conflict rate", charts.conflict_rate_trend(monthly, kpis["conflict_rate"]),
                    "Conflict rate per month",
                    lambda m=monthly, r=kpis["conflict_rate"]:
                    report_charts.conflict_rate_svg(m, r),
                    subtitle="share of reports that were conflict")
        with casualty_col:
            if float((monthly["Human Deaths"] + monthly["People Injured"]).sum()) == 0:
                st.markdown("**People killed and injured**")
                st.success("No casualties recorded in the selected period.")
            else:
                _figure("People killed and injured", charts.casualty_trend(monthly),
                        "People killed and injured per month",
                        lambda m=monthly: report_charts.casualty_svg(m))

        season_col, division_col = st.columns(2)
        with season_col:
            _figure("Seasonal pattern", charts.seasonal_heatmap(seasonal_matrix(filtered)),
                    "Seasonal pattern",
                    lambda f=filtered: report_charts.seasonal_heatmap_svg(seasonal_matrix(f)),
                    note="Read down a column to compare the same month across years.")
        with division_col:
            division_long = division_monthly_conflict(filtered)
            if division_long["Division"].nunique() < 2:
                st.markdown("**By division**")
                st.info("One division selected. Widen the Division filter to compare.")
            else:
                _figure("By division", charts.division_trend(division_long, division_palette),
                        "Conflict events per month by division",
                        lambda d=division_long, c=division_palette:
                        report_charts.division_trend_svg(d, c),
                        subtitle="three-month average")

    esc_col, rate_div_col = st.columns(2)
    with esc_col:
        basis = brief_insights.trend_basis_phrase(brief, recent_days)
        if escalating.empty:
            st.markdown(f"**Escalating beats** -- {basis}")
            st.info("No beat rose clearly faster than its baseline.")
        else:
            _figure("Escalating beats", charts.escalation_dumbbell(escalating),
                    "Escalating beats", lambda e=escalating: report_charts.escalation_svg(e),
                    subtitle=basis,
                    note=f"Grey is the baseline, red the recent window; "
                         f"{brief_insights.plural(len(escalating), 'beat')} escalating.")
    with rate_div_col:
        rates = division_conflict_rate(filtered)
        if rates.empty:
            st.markdown("**Conflict rate by division**")
            st.info("No division data for the current filters.")
        else:
            _figure("Conflict rate by division",
                    charts.division_rate_bars(rates, kpis["conflict_rate"], division_palette),
                    "Conflict rate by division",
                    lambda r=rates, k=kpis["conflict_rate"], c=division_palette:
                    report_charts.division_rate_svg(r, k, c),
                    note="Rate, not raw volume: sighting counts mostly track reporting "
                         "effort.")
