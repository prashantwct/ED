"""The Overview page's words: what changed, where to act, whom to warn.

Built only from tables the analysis already produces -- the beat table,
the escalating beats, the village ranking and the emerging hotspot
labels -- so the Overview cannot say anything the brief does not. Pure
functions, tested without Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from core.brief_insights import beat_driver, names, plural
from core.intelligence import TIER_CRITICAL, TIER_HIGH, TIER_ORDER, recommended_actions

#: How far back "what changed" looks for beats that became Critical.
CHANGE_DAYS = 30
ACT_NOW_ROWS = 5


@dataclass(frozen=True)
class Change:
    """One thing that changed, and where on the app to look at it."""

    kind: str     # "critical", "escalating", "emerging"
    label: str    # what the chip says
    target: str   # the beat, or the place, it is about


def _pct_change(recent: float, before: float) -> Optional[float]:
    if before != before or recent != recent or before <= 0:
        return None
    return (recent - before) / before * 100


def change_sentence(change: Dict[str, float], days: int) -> Optional[str]:
    """``Conflict is up 25% on the same 90 days last year.``"""
    recent = change.get("conflicts_recent", float("nan"))
    for key, against in (("year", f"the same {days} days last year"),
                         ("prior", f"the {days} days before")):
        before = change.get(f"conflicts_{key}", float("nan"))
        if before != before:
            continue
        pct = _pct_change(recent, before)
        if pct is None:
            return (f"{plural(recent, 'conflict event')} in the last {days} days, "
                    f"against none in {against}.") if recent else None
        if abs(pct) < 5:
            return f"Conflict is level with {against} ({recent:.0f} events against {before:.0f})."
        way = "up" if pct > 0 else "down"
        return (f"Conflict is {way} {abs(pct):.0f}% on {against} "
                f"({recent:.0f} events against {before:.0f}).")
    return None


def headline(change: Dict[str, float], days: int, beats: pd.DataFrame,
             new_critical: List[str], villages: Optional[pd.DataFrame]) -> str:
    """Two or three sentences a manager reads before anything else."""
    parts: List[str] = []
    said = change_sentence(change, days)
    if said:
        parts.append(said)

    critical = int((beats["Priority Tier"] == TIER_CRITICAL).sum()) if len(beats) else 0
    if critical:
        line = f"{plural(critical, 'beat')} {'is' if critical == 1 else 'are'} Critical"
        if new_critical:
            fresh = len(new_critical)
            line += (f", {fresh} of {'it' if critical == 1 else 'them'} new in the last "
                     f"{CHANGE_DAYS} days")
        parts.append(line + ".")
    elif len(beats):
        parts.append("No beat is Critical.")

    if villages is not None and len(villages) and "Inside Hotspot" in villages:
        outside = villages[(villages["Tier"] == TIER_CRITICAL)
                           & ~villages["Inside Hotspot"].fillna(False).astype(bool)]
        if len(outside):
            verb = "lies" if len(outside) == 1 else "lie"
            parts.append(f"{plural(len(outside), 'Critical village')} {verb} outside every "
                         "hotspot, so hotspot patrols alone will not reach them.")
    return " ".join(parts) if parts else "No reports match this area and period."


def act_now(beats: pd.DataFrame, rows: int = ACT_NOW_ROWS) -> List[Dict[str, str]]:
    """The beats to act on first: worst tier, then highest score.

    Only Critical and High beats qualify; a list padded with Watch beats
    would dilute the ones that need staff today.
    """
    if beats is None or beats.empty:
        return []
    urgent = beats[beats["Priority Tier"].isin([TIER_CRITICAL, TIER_HIGH])].copy()
    if urgent.empty:
        return []
    urgent["_tier"] = urgent["Priority Tier"].map(TIER_ORDER.index)
    urgent = urgent.sort_values(["_tier", "Priority Score"], ascending=[True, False]).head(rows)
    # One action per card, the first the beat table would list; the rest are
    # on the Where to act table and in the brief.
    first = (recommended_actions(urgent, max_actions=1)
             if {"Night Conflict %", "Near Village %", "Confidence"} <= set(urgent.columns)
             else urgent.get("Recommended Action", pd.Series("", index=urgent.index)))
    out = []
    for (_, row), action in zip(urgent.iterrows(), first):
        out.append({
            "beat": str(row["Beat"]),
            "range": str(row.get("Range", "")),
            "division": str(row.get("Division", "")),
            "tier": str(row["Priority Tier"]),
            "why": beat_driver(row.to_dict()),
            "do": str(action),
        })
    return out


def new_critical(beats_now: pd.DataFrame, beats_before: Optional[pd.DataFrame]) -> List[str]:
    """Beats Critical now that were not Critical ``CHANGE_DAYS`` ago."""
    if beats_now is None or beats_now.empty:
        return []
    now = beats_now.loc[beats_now["Priority Tier"] == TIER_CRITICAL, "Beat"].astype(str)
    if beats_before is None or beats_before.empty:
        return list(now)
    before = set(beats_before.loc[beats_before["Priority Tier"] == TIER_CRITICAL, "Beat"]
                 .astype(str))
    return [b for b in now if b not in before]


def what_changed(beats_now: pd.DataFrame, beats_before: Optional[pd.DataFrame],
                 escalating: pd.DataFrame, emerging: Optional[pd.DataFrame],
                 limit: int = 3) -> List[Change]:
    """Newly Critical beats, the fastest-rising beats, new hotspot places."""
    changes: List[Change] = [
        Change("critical", f"Now Critical: {b}", b)
        for b in new_critical(beats_now, beats_before)[:limit]
    ]
    if escalating is not None and len(escalating):
        rising = escalating.assign(
            _rise=escalating["Recent Conflicts"] - escalating["Prior Conflicts"]
        ).sort_values("_rise", ascending=False).head(limit)
        for row in rising.to_dict("records"):
            changes.append(Change(
                "escalating",
                f"Rising: {row['Beat']} ({int(row['Recent Conflicts'])} vs "
                f"{int(row['Prior Conflicts'])})",
                str(row["Beat"])))
    if emerging is not None and len(emerging) and "Pattern" in emerging:
        fresh = emerging[emerging["Pattern"].isin(["New", "Intensifying"])].head(limit)
        for row in fresh.to_dict("records"):
            near = row.get("Nearest Village")
            where = f" near {near}" if isinstance(near, str) and near else ""
            changes.append(Change("emerging", f"{row['Pattern']} hotspot{where}",
                                  str(row.get("Place", ""))))
    return changes


def villages_to_warn(villages: Optional[pd.DataFrame]) -> int:
    if villages is None or villages.empty or "Tier" not in villages:
        return 0
    return int(villages["Tier"].isin([TIER_CRITICAL, TIER_HIGH]).sum())


def warn_list(villages: Optional[pd.DataFrame], limit: int = 6) -> str:
    """The worst villages by name, for a line under the Act now list."""
    if villages is None or villages.empty:
        return ""
    worst = villages[villages["Tier"].isin([TIER_CRITICAL, TIER_HIGH])]
    return names(worst["Village"].astype(str), limit=limit) if len(worst) else ""
