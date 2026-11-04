"""What the page is looking at: an area, a period and an incident type.

The period is split in two, because one date range was doing two jobs:

* the **focus window** is what the page counts and draws -- the headline
  numbers, the points on the map, the rows in the download;
* the **analysis history** is everything the rankings and trends compare
  against: the register from its first report up to the end of the focus
  window, for the same area and incident type.

Tiers, escalation and the year-on-year comparisons all need that history.
Cutting the analysis to the last 90 days removed the comparisons and
moved beats between tiers (on a test register, escalating beats fell from
4 to 1 and High beats from 22 to 3). With the two apart, a short focus
window is safe as the default: the period's end is the date the analysis
is "as of", and its start only decides what is counted on screen.

Pure functions only, so the rules are tested without Streamlit. The scope
also round-trips through URL query parameters, so a view can be shared;
the link carries the scope, never the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

from core.analytics import filter_dataframe

LAST_30, LAST_90, LAST_12M, WHOLE, CUSTOM = (
    "Last 30 days", "Last 90 days", "Last 12 months", "Whole register", "Custom")
PERIODS: Dict[str, Optional[int]] = {
    LAST_30: 30, LAST_90: 90, LAST_12M: 365, WHOLE: None, CUSTOM: None}
DEFAULT_PERIOD = LAST_90
# What the scope bar's buttons say: short enough for five in a row.
SHORT_LABELS = {LAST_30: "30 days", LAST_90: "90 days", LAST_12M: "12 months",
                WHOLE: "All", CUSTOM: "Custom"}
PERIOD_SLUGS = {LAST_30: "30d", LAST_90: "90d", LAST_12M: "12m", WHOLE: "all", CUSTOM: "custom"}
_SLUG_PERIODS = {v: k for k, v in PERIOD_SLUGS.items()}


@dataclass(frozen=True)
class Window:
    """An inclusive run of days."""

    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def label(self) -> str:
        return f"{_day(self.start)} to {_day(self.end)}"


def _day(d: date) -> str:
    return f"{d.day} {d:%b %Y}"


def focus_window(period: str, register_start: date, register_end: date,
                 custom: Optional[Sequence[date]] = None) -> Window:
    """The days the page counts and draws.

    Presets end on the register's last report, not today: a register
    fetched in the morning or exported last week still opens on its own
    latest days. A custom range is clipped to the register; an unusable
    one falls back to the default.
    """
    if period == CUSTOM and custom is not None and len(custom) == 2 and all(custom):
        start, end = min(custom), max(custom)
        start = max(start, register_start)
        end = min(end, register_end)
        if start <= end:
            return Window(start, end)
        period = DEFAULT_PERIOD
    if period == WHOLE:
        return Window(register_start, register_end)
    days = PERIODS.get(period) or PERIODS[DEFAULT_PERIOD]
    start = max(register_start, register_end - timedelta(days=days - 1))
    return Window(start, register_end)


def history(df: pd.DataFrame, window: Window, divisions: Sequence[str] = (),
            ranges: Sequence[str] = (), beats: Sequence[str] = (),
            min_severity: float = 0.0) -> pd.DataFrame:
    """The analysis frame: this area and incident type, up to the window's end."""
    first = df["Date"].min().date() if len(df) else window.start
    return filter_dataframe(df, date_range=(min(first, window.start), window.end),
                            divisions=list(divisions), ranges=list(ranges),
                            beats=list(beats), min_severity=min_severity)


def focus(history_frame: pd.DataFrame, window: Window) -> pd.DataFrame:
    """The focus window's rows. Keeps the history's index, so a mask over
    one lines up with the other."""
    if history_frame.empty:
        return history_frame
    return history_frame[history_frame["Date"] >= pd.Timestamp(window.start)]


def covers_register(window: Window, register_start: date) -> bool:
    return window.start <= register_start


def comparison_days(window: Window, register_start: date, recent_days: int) -> int:
    """How many days the headline change compares.

    A focus window shorter than the register compares itself with the
    same days a year earlier, so the number and its change are about the
    same days. The whole register has nothing before it to compare with,
    so it falls back to the recent window from Advanced analysis.
    """
    return recent_days if covers_register(window, register_start) else window.days


# ---------------------------------------------------------------------------
# URL round trip
# ---------------------------------------------------------------------------
def to_query(divisions: Sequence[str], ranges: Sequence[str], beats: Sequence[str],
             period: str, custom: Optional[Sequence[date]], incident: Optional[str],
             incident_options: Sequence[str]) -> Dict[str, object]:
    """Query parameters for a scope; defaults are left out to keep links short."""
    out: Dict[str, object] = {}
    if divisions:
        out["div"] = list(divisions)
    if ranges:
        out["range"] = list(ranges)
    if beats:
        out["beat"] = list(beats)
    if period and period != DEFAULT_PERIOD:
        out["period"] = PERIOD_SLUGS.get(period, PERIOD_SLUGS[DEFAULT_PERIOD])
    if period == CUSTOM and custom is not None and len(custom) == 2 and all(custom):
        out["from"], out["to"] = min(custom).isoformat(), max(custom).isoformat()
    if incident and incident_options and incident != incident_options[0] \
            and incident in incident_options:
        out["type"] = str(list(incident_options).index(incident))
    return out


def from_query(params: Mapping[str, List[str]], division_options: Sequence[str],
               range_options: Sequence[str], beat_options: Sequence[str],
               incident_options: Sequence[str]) -> Dict[str, object]:
    """A scope from query parameters, keeping only values this register has.

    ``params`` maps each key to all its values (a repeated key is a list).
    A link made against another register, or edited by hand, opens on
    whatever of it still makes sense rather than on an error.
    """
    def pick(key: str, options: Sequence[str]) -> List[str]:
        allowed = set(map(str, options))
        return [v for v in params.get(key, []) if v in allowed]

    out: Dict[str, object] = {
        "divisions": pick("div", division_options),
        "ranges": pick("range", range_options),
        "beats": pick("beat", beat_options),
        "period": _SLUG_PERIODS.get((params.get("period") or [""])[0], DEFAULT_PERIOD),
        "custom": None,
        "incident": incident_options[0] if incident_options else None,
    }
    if out["period"] == CUSTOM:
        try:
            out["custom"] = (date.fromisoformat(params["from"][0]),
                             date.fromisoformat(params["to"][0]))
        except (KeyError, IndexError, ValueError):
            out["period"] = DEFAULT_PERIOD
    try:
        index = int((params.get("type") or ["0"])[0])
        if 0 <= index < len(incident_options):
            out["incident"] = incident_options[index]
    except ValueError:
        pass
    return out


def area_label(divisions: Sequence[str], ranges: Sequence[str],
               beats: Sequence[str]) -> str:
    """``Anuppur``, ``Kotma``, ``3 beats``: the narrowest level chosen, in a few words."""
    def few(values: Sequence[str], noun: str) -> str:
        return values[0] if len(values) == 1 else f"{len(values)} {noun}"

    if beats:
        return few(list(beats), "beats")
    if ranges:
        return few(list(ranges), "ranges")
    if divisions:
        return few(list(divisions), "divisions")
    return "All divisions"
