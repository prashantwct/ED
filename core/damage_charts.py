"""Plotly figures for the damage and compensation view.

Same tokens as ``core.charts`` so the two views read as one product.
Two colour jobs here, kept apart:

* **Compensation status is a status**, so it uses the status scale --
  not applied and refused in the reds, in process amber, paid green --
  and every bar carries its label, never colour alone.
* **Crop against house is an identity**, in amber and brick: warm like
  the conflict view's damage colours, but far enough apart to be told
  apart in a stacked bar.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core import damage
from core.charts import BRAND, GRID, INK, INK_MUTED, SURFACE, _base_layout

# Validated as a pair: the ramp's own crop and house steps sit too close
# together (normal-vision dE 14) to tell apart side by side.
KIND_COLORS = {damage.CROP: "#D98E04", damage.HOUSE: "#A8360F"}

STATUS_COLORS: Dict[str, str] = {
    damage.NOT_APPLIED: "#8C1D18",
    damage.NOT_RECEIVED: "#C2570C",
    damage.IN_PROCESS: "#C9A227",
    damage.RECEIVED: BRAND,
    damage.NOT_RECORDED: "#9AA5A0",
}


def _lakh(values) -> np.ndarray:
    """Rupees in lakh, the unit a forest office budgets in."""
    return np.asarray(values, dtype=float) / 1e5


def compensation_bars(pipeline: pd.DataFrame, measure: str = "Cases") -> go.Figure:
    """Each kind as one bar split by compensation status."""
    fig = go.Figure()
    kinds = list(dict.fromkeys(pipeline["Kind"]))
    totals = pipeline.groupby("Kind")[measure].sum()
    for status in damage.STATUS_ORDER:
        part = pipeline[pipeline["Compensation"] == status].set_index("Kind").reindex(kinds)
        values = part[measure].fillna(0).to_numpy(float)
        if not values.any():
            continue
        share = values / totals.reindex(kinds).to_numpy(float) * 100
        label = [f"{v:,.0f}" if measure == "Cases" else damage.rupees(v) for v in values]
        fig.add_bar(
            y=kinds, x=values if measure == "Cases" else _lakh(values),
            orientation="h", name=status,
            marker=dict(color=STATUS_COLORS[status], line=dict(color=SURFACE, width=2)),
            text=[f"{s:.0f}%" if s >= 8 else "" for s in share], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="#fff", size=11),
            customdata=np.stack([label, share], axis=-1),
            hovertemplate=f"%{{y}} - {status}<br><b>%{{customdata[0]}}</b> "
            "(%{customdata[1]:.0f}%)<extra></extra>",
        )
    _base_layout(fig, 70 + 56 * len(kinds), barmode="stack", legend_traceorder="normal")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_xaxes(showgrid=True, gridcolor=GRID,
                     title_text="Cases" if measure == "Cases"
                     else "Estimated loss (Rs lakh)")
    return fig


def monthly_loss(monthly: pd.DataFrame) -> go.Figure:
    """Estimated loss per month of damage, crop and house stacked."""
    fig = go.Figure()
    for kind in (damage.CROP, damage.HOUSE):
        part = monthly[monthly["Kind"] == kind]
        if part.empty:
            continue
        fig.add_bar(
            x=part["Month"], y=_lakh(part["Loss"]), name=f"{kind} damage",
            marker=dict(color=KIND_COLORS[kind], line=dict(color=SURFACE, width=1)),
            customdata=np.stack([part["Cases"], part["Loss"]], axis=-1),
            hovertemplate=f"%{{x|%b %Y}}<br>{kind}: <b>Rs %{{customdata[1]:,.0f}}</b> "
            "in %{customdata[0]} case(s)<extra></extra>",
        )
    _base_layout(fig, 320, barmode="stack", hovermode="x unified")
    fig.update_yaxes(title_text="Estimated loss (Rs lakh)")
    fig.update_xaxes(dtick="M1" if monthly["Month"].nunique() <= 14 else "M2",
                     tickformat="%b\n%Y")
    return fig


def ranked_bars(table: pd.DataFrame, label: str, value: str = "Cases",
                color: str = BRAND, limit: int = 12, note: str = "") -> go.Figure:
    """Horizontal ranked bars, one colour, labelled at the end."""
    data = table.head(limit).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=data[value], y=data[label].astype(str), orientation="h",
        marker=dict(color=color), text=data[value], textposition="outside",
        textfont=dict(color=INK, size=11), cliponaxis=False,
        hovertemplate=f"<b>%{{y}}</b><br>{value}: %{{x:,}}{note}<extra></extra>",
    ))
    _base_layout(fig, 60 + 26 * len(data), showlegend=False, bargap=0.35)
    fig.update_layout(margin=dict(t=8, b=8, l=8, r=36))
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, rangemode="tozero")
    return fig


def by_kind_bars(table: pd.DataFrame, category: str, order: List[str], height: int = 260,
                 x_title: str = "") -> go.Figure:
    """Grouped bars of cases per category, one bar per kind."""
    fig = go.Figure()
    for kind in (damage.CROP, damage.HOUSE):
        part = table[table["Kind"] == kind].set_index(category).reindex(order)
        if part["Cases"].fillna(0).sum() == 0:
            continue
        fig.add_bar(
            x=order, y=part["Cases"].fillna(0), name=f"{kind} damage",
            marker=dict(color=KIND_COLORS[kind], line=dict(color=SURFACE, width=1)),
            hovertemplate=f"{kind}, %{{x}}: <b>%{{y}}</b><extra></extra>",
        )
    _base_layout(fig, height, barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title_text="Cases")
    fig.update_xaxes(title_text=x_title, title_font=dict(color=INK_MUTED), type="category")
    return fig


def hour_profile(hourly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for kind in (damage.CROP, damage.HOUSE):
        part = hourly[hourly["Kind"] == kind]
        if part["Cases"].sum() == 0:
            continue
        fig.add_bar(
            x=part["Hour"], y=part["Cases"], name=f"{kind} damage",
            marker=dict(color=KIND_COLORS[kind], line=dict(color=SURFACE, width=1)),
            hovertemplate=f"{kind}, %{{x}}:00 - <b>%{{y}}</b><extra></extra>",
        )
    _base_layout(fig, 260, barmode="stack", bargap=0.12)
    fig.add_vrect(x0=5.5, x1=17.5, fillcolor="#F2C94C", opacity=0.08, line_width=0,
                  layer="below")
    fig.add_annotation(x=11.5, y=1, yref="paper", yanchor="top", showarrow=False,
                       text="daylight", font=dict(size=11, color=INK_MUTED))
    fig.update_xaxes(title_text="Hour of damage (24h)", dtick=2, range=[-0.6, 23.6],
                     title_font=dict(color=INK_MUTED))
    fig.update_yaxes(title_text="Cases")
    return fig


def area_comparison(check: Dict[str, float]) -> go.Figure:
    """Damaged area as claimed, as measured, and as calculated."""
    labels = ["Owner's claim", "Surveyor's estimate", "Calculated"]
    values = [check["owner"], check["surveyor"], check["calculated"]]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker=dict(color=["#B7C2BC", "#6B8578", BRAND]),
        text=[f"{v:,.1f}" for v in values], textposition="outside",
        textfont=dict(color=INK, size=11), cliponaxis=False,
        hovertemplate="%{x}: <b>%{y:,.2f}</b><extra></extra>",
    ))
    _base_layout(fig, 260, showlegend=False, bargap=0.45)
    fig.update_yaxes(title_text="Total damaged area", rangemode="tozero")
    return fig
