"""What the brief says: its findings, the action plan, and the ecology.

The brief used to restate each beat's recommendation in every table the
beat appeared in -- the same forty words down twenty-five rows -- and
wrote its numbers in a template's voice ("6 human fatality(ies)"). This
turns the same analysis into what a range officer acts on:

* a handful of findings in plain sentences, each with its number;
* an action plan with one row per action, naming every place it applies
  to, why, and when, instead of one action string per beat;
* a short driver per beat (bulls or herds, crops or houses, the trend),
  so the tables say what is going on without repeating the plan;
* what the register says about the elephants themselves -- group sizes,
  when they are seen, how much of the landscape they use and where they
  are through the year.

Pure functions over the frames the app already builds, so the numbers
here are the dashboard's numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from core.analytics import (
    BULL_TYPE_GROUPS,
    classify_conflict,
    classify_group,
    indirect_mask,
    property_damage_mask,
)
from core.config import (
    BULL_SHARE_FOR_ALERT,
    MIN_CONFLICTS_FOR_COMPOSITION,
    NIGHT_HOUR_END,
    NIGHT_HOUR_START,
)

# Places named in one cell before the rest are counted instead.
MAX_NAMES = 6
# Beats a finding about concentration counts.
TOP_BEATS = 10
# Herd-size bands for the group-size chart.
SIZE_BANDS = [(1, 1, "1"), (2, 3, "2-3"), (4, 9, "4-9"), (10, 19, "10-19"), (20, 10**6, "20+")]

FOCUS_TIERS = ("Critical", "High")


def plural(count: float, one: str, many: Optional[str] = None) -> str:
    """``6 people``, ``1 person``: a count with its noun agreeing."""
    n = int(round(count))
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def names(items: Iterable[object], limit: int = MAX_NAMES) -> str:
    """``A, B and C``; past ``limit``, ``A, B, C and 4 more``."""
    seen: List[str] = []
    for item in items:
        text = str(item).strip()
        if text and text.lower() != "nan" and text not in seen:
            seen.append(text)
    if not seen:
        return ""
    if len(seen) > limit:
        return ", ".join(seen[:limit]) + f" and {len(seen) - limit} more"
    if len(seen) == 1:
        return seen[0]
    return ", ".join(seen[:-1]) + " and " + seen[-1]


def _village(name: object) -> str:
    """``Ghaghaud (0.9 km)`` -> ``Ghaghaud``."""
    return str(name).split(" (")[0].strip()


def _night_window() -> str:
    return f"{NIGHT_HOUR_START:02d}:00-{NIGHT_HOUR_END:02d}:00"


_MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                "Nov", "Dec"]


def _months(months: List[str]) -> str:
    """Peak months in calendar order, not ranked order."""
    ordered = sorted(months, key=lambda m: _MONTH_ORDER.index(m) if m in _MONTH_ORDER else 99)
    return names(ordered, limit=12)


def harm(deaths: float, injured: float) -> str:
    """``2 people killed and 3 injured``, leaving out a zero."""
    return " and ".join(part for part in (
        f"{plural(deaths, 'person', 'people')} killed" if deaths else "",
        f"{int(injured):,} injured" if injured else "",
    ) if part)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
def key_findings(
    df: pd.DataFrame,
    brief: Dict[str, object],
    change: Dict[str, float],
    recent_days: int,
    emerging: Optional[pd.DataFrame] = None,
    refuges: Optional[Dict[str, object]] = None,
) -> List[str]:
    """The few sentences worth reading first, each carrying its number."""
    if df.empty:
        return ["No reports match the filters, so there is nothing to brief on."]
    lines: List[str] = []
    for build in (_casualty_finding, _trend_finding, _loss_finding, _group_finding,
                  _concentration_finding):
        line = build(df, brief, change, recent_days)
        if line:
            lines.append(line)
    line = _emerging_finding(emerging)
    if line:
        lines.append(line)
    line = _refuge_finding(refuges)
    if line:
        lines.append(line)
    return lines


def _casualty_finding(df, brief, change, recent_days) -> Optional[str]:
    kpis = brief["kpis"]
    deaths, injured = int(kpis.get("human_deaths", 0)), int(kpis.get("human_injuries", 0))
    window = brief["coverage"].get("critical_window_days", 90)
    if not (deaths or injured):
        return "No one was killed or injured by elephants in the period."
    counts = classify_conflict(df).value_counts()
    incidents = int(counts.get("Death", 0) + counts.get("Injury", 0))
    beats = brief["beats"]
    recent = 0
    if not beats.empty and "Recent Deaths" in beats:
        recent = int(beats["Recent Deaths"].sum() + beats["Recent Injuries"].sum())
    when = (f"none in the last {window} days" if recent == 0
            else f"{recent:,} of them in the last {window} days")
    return f"{harm(deaths, injured).capitalize()} in {plural(incidents, 'incident')}; {when}."


def _trend_finding(df, brief, change, recent_days) -> Optional[str]:
    recent, prior = change.get("conflicts_recent"), change.get("conflicts_prior")
    if recent is None or prior is None or pd.isna(recent) or pd.isna(prior):
        return None
    recent, prior = int(recent), int(prior)
    if recent + prior == 0:
        return None
    if prior == 0:
        line = (f"{plural(recent, 'conflict event')} in the last {recent_days} days, "
                f"none in the {recent_days} before")
    else:
        change_pct = (recent - prior) / prior
        if abs(change_pct) < 0.1:
            line = (f"Conflict is level with the previous {recent_days} days "
                    f"({recent:,} events against {prior:,})")
        else:
            line = (f"Conflict is {'up' if recent > prior else 'down'} "
                    f"{abs(change_pct):.0%} on the previous {recent_days} days "
                    f"({recent:,} events against {prior:,})")
    beats = brief["beats"]
    if recent > prior and not beats.empty and "Recent Conflicts" in beats:
        rise = (beats["Recent Conflicts"] - beats["Prior Conflicts"]).sort_values(ascending=False)
        drivers = beats.loc[rise[rise > 0].index[:3], "Beat"]
        if len(drivers):
            line += f", most of the rise in {names(drivers)}"
    return line + "."


def _loss_finding(df, brief, change, recent_days) -> Optional[str]:
    counts = classify_conflict(df).value_counts()
    crop, house = int(counts.get("Crop", 0)), int(counts.get("House", 0))
    conflicts = crop + house + int(counts.get("Death", 0) + counts.get("Injury", 0))
    if not conflicts:
        return None
    line = f"Crop loss is {crop / conflicts:.0%} of conflict ({crop:,} reports)"
    if house:
        line += f" and house or grain-store damage {house / conflicts:.0%} ({house:,})"
    months = brief["temporal"].get("peak_months") or []
    if months:
        line += f"; it peaks in {_months(months)}"
    return line + "."


def _group_finding(df, brief, change, recent_days) -> Optional[str]:
    composition = brief.get("composition")
    if composition is None or composition.empty:
        return None
    known = composition[composition["Group Type"] != "Unrecorded"]
    conflicts = float(known["Conflict Events"].sum())
    if conflicts < MIN_CONFLICTS_FOR_COMPOSITION:
        return None
    bulls = known[known["Group Type"].isin(BULL_TYPE_GROUPS)]
    herds = known[known["Group Type"] == "Family herd"]
    line = (f"Lone bulls and bull parties cause {bulls['Conflict Events'].sum() / conflicts:.0%} "
            "of conflict where the group was recorded")
    deaths, total = float(bulls["Human Deaths"].sum()), float(known["Human Deaths"].sum())
    if total == 1:
        line += ", and the one death" if deaths else ", though not the one death"
    elif total:
        line += f", and {deaths:.0f} of the {plural(total, 'death')}"
    if not herds.empty:
        line += (f"; breeding herds cause {herds['Conflict Events'].sum() / conflicts:.0%}, "
                 f"raiding on {herds['Damage Rate %'].iloc[0]:.0f}% of their sightings "
                 f"against {bulls['Damage Rate %'].max():.0f}% for bulls")
    return line + "."


def _concentration_finding(df, brief, change, recent_days) -> Optional[str]:
    beats = brief["beats"]
    if beats.empty or len(beats) < TOP_BEATS * 1.5:
        return None
    ranked = beats.sort_values("Conflict Events", ascending=False)
    total = float(ranked["Conflict Events"].sum())
    if not total:
        return None
    share = ranked["Conflict Events"].head(TOP_BEATS).sum() / total
    return (f"{TOP_BEATS} of {len(beats)} beats hold {share:.0%} of all conflict, led by "
            f"{names(ranked['Beat'].head(4))}.")


def _emerging_finding(emerging: Optional[pd.DataFrame]) -> Optional[str]:
    if emerging is None or emerging.empty:
        return None
    from core.spacetime import CONSECUTIVE, INTENSIFYING, NEW

    new = emerging[emerging["Pattern"] == NEW]
    growing = emerging[emerging["Pattern"].isin([INTENSIFYING, CONSECUTIVE])]
    if new.empty and growing.empty:
        return None
    parts = []
    if len(new):
        last = pd.to_datetime(new["Last Hot"]).max()
        near = names((_village(v) for v in new["Nearest Village"] if v), limit=3)
        parts.append(f"{plural(len(new), 'place')} became a hotspot for the first time in "
                     f"{last:%b %Y}" + (f", near {near}" if near else ""))
    if len(growing):
        parts.append(f"{plural(len(growing), 'more place')} "
                     f"{'has' if len(growing) == 1 else 'have'} been hot for consecutive months")
    return "; ".join(parts) + "."


def _refuge_finding(refuges: Optional[Dict[str, object]]) -> Optional[str]:
    if not refuges:
        return None
    table = refuges.get("refuges")
    if table is None or table.empty or "Linked Raids" not in table:
        return None
    top = table.sort_values("Linked Raids", ascending=False).iloc[0]
    line = (f"{plural(len(table), 'forest day refuge')} send night raids; the busiest, "
            f"{top['Refuge']} at {top.get('Beat', '')}, is linked to "
            f"{plural(top['Linked Raids'], 'raid')}")
    areas = refuges.get("areas")
    if areas is not None and len(areas):
        line += f". {plural(len(areas), 'raid area')} have no known refuge"
    return line + "."


# ---------------------------------------------------------------------------
# The action plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    """One thing to do, everywhere it applies."""

    kind: str     # respond, warn, bulls, herds, survey, refuge, search, crops, homes, verify, confirm
    action: str
    where: str
    why: str
    when: str


def action_plan(
    brief: Dict[str, object],
    recent_days: int,
    emerging: Optional[pd.DataFrame] = None,
    refuges: Optional[Dict[str, object]] = None,
    village_risk: Optional[pd.DataFrame] = None,
) -> List[Action]:
    """Every action the analysis supports, most urgent first, one row each.

    Built from the same signals the per-beat recommendations read -- a
    recent casualty, who causes the conflict, what is lost, the trend,
    thin reporting -- but grouped by action, so a reader sees "night
    watch for bulls: these seven beats" once rather than seven times.
    """
    beats = brief["beats"]
    plan: List[Action] = []
    if beats.empty:
        return plan
    window = brief["coverage"].get("critical_window_days", 90)
    months = brief["temporal"].get("peak_months") or []
    focus = beats[beats["Priority Tier"].isin(FOCUS_TIERS) | (beats["Trend"] == "Escalating")]
    # Only where enough conflicts had their group recorded to say who.
    enough = focus["Group-Known Conflicts"] >= MIN_CONFLICTS_FOR_COMPOSITION
    bulls = focus["Bull-Type Conflict %"]

    hurt = beats[(beats["Recent Deaths"] > 0) | (beats["Recent Injuries"] > 0)]
    if len(hurt):
        plan.append(Action(
            "respond", "Keep the rapid-response team posted and alert the community",
            names(hurt["Beat"]),
            f"{harm(hurt['Recent Deaths'].sum(), hurt['Recent Injuries'].sum()).capitalize()} "
            f"in the last {window} days",
            "Now"))

    plan.extend(_emerging_actions(emerging))

    raiders = focus[enough & (bulls >= BULL_SHARE_FOR_ALERT)].sort_values(
        "Conflict Events", ascending=False)
    if len(raiders):
        share = raiders["Bull-Type Conflict %"]
        plan.append(Action(
            "bulls", "Identify the raiding bulls and watch their approach routes at night",
            names(raiders["Beat"]),
            f"Bulls behind {share.min():.0f}-{share.max():.0f}% of conflict in each; "
            f"{plural(raiders['Conflict Events'].sum(), 'conflict event')}",
            _night_window()))

    herds = focus[enough & (bulls <= 100 - BULL_SHARE_FOR_ALERT)].sort_values(
        "Conflict Events", ascending=False)
    if len(herds):
        plan.append(Action(
            "herds", "Give breeding herds passage: hold crowds back and do not drive them",
            names(herds["Beat"]),
            f"Herds with calves behind most of the conflict; "
            f"{plural(herds['Conflict Events'].sum(), 'conflict event')}",
            "While a herd is in the beat"))

    rising = beats[beats["Trend"] == "Escalating"].copy()
    if len(rising):
        rising["_rise"] = rising["Recent Conflicts"] - rising["Prior Conflicts"]
        rising = rising.sort_values("_rise", ascending=False)
        counts = ", ".join(f"{r['Beat']} {int(r['Recent Conflicts'])} vs {int(r['Prior Conflicts'])}"
                           for r in rising.head(3).to_dict("records"))
        plan.append(Action(
            "survey", "Re-survey escalating beats and establish what changed",
            names(rising["Beat"]),
            f"Conflict events, last {recent_days} days vs the {recent_days} before: {counts}",
            "This month"))

    plan.extend(_refuge_actions(refuges))

    if village_risk is not None and len(village_risk):
        exposed = village_risk[village_risk["Tier"].isin(FOCUS_TIERS)]
        if len(exposed):
            # Neighbouring villages share incidents, so their counts are
            # quoted one by one rather than summed into a total.
            worst = exposed.sort_values("Conflict Events", ascending=False).head(3)
            counts = ", ".join(f"{r['Village']} {int(r['Conflict Events'])}"
                               for r in worst.to_dict("records"))
            plan.append(Action(
                "warn", "Early-warning cover and community alerts for the most exposed villages",
                names(exposed["Village"]),
                f"Conflict events nearby: {counts}",
                f"Before {_months(months)}" if months else "Through the season"))

    crops = focus[(focus["Crop Damage Events"] > 0)
                  & (focus["Crop Damage Events"] >= focus["House Damage Events"])]
    if len(crops):
        crops = crops.sort_values("Crop Damage Events", ascending=False)
        plan.append(Action(
            "crops", "Crop guarding and fence checks on fields hit more than once",
            names(crops["Beat"]),
            f"{plural(crops['Crop Damage Events'].sum(), 'crop-loss report')}",
            f"Before and through {_months(months)}" if months else "At harvest"))

    homes = focus[focus["House Damage Events"] > focus["Crop Damage Events"]]
    if len(homes):
        plan.append(Action(
            "homes", "Secure grain stores and homesteads",
            names(homes.sort_values("House Damage Events", ascending=False)["Beat"]),
            f"{plural(homes['House Damage Events'].sum(), 'house or store damage report')}",
            _night_window()))

    past = beats[(beats["Human Deaths"] > 0) & (beats["Recent Deaths"] == 0)]
    if len(past):
        plan.append(Action(
            "verify", "Check that the measures taken after past deaths still hold",
            names(past["Beat"]),
            f"{plural(past['Human Deaths'].sum(), 'death')} earlier in the period, none in the "
            f"last {window} days",
            "This month"))

    thin = focus[focus["Confidence"] == "Low"]
    if len(thin):
        plan.append(Action(
            "confirm", "Confirm with the beat guard before committing staff",
            names(thin["Beat"]),
            "Ranked high on fewer than 10 reports", "Before deploying"))
    return plan


def _emerging_actions(emerging: Optional[pd.DataFrame]) -> List[Action]:
    if emerging is None or emerging.empty:
        return []
    from core.spacetime import CONSECUTIVE, INTENSIFYING, NEW

    hot = emerging[emerging["Pattern"].isin([NEW, INTENSIFYING, CONSECUTIVE])]
    if hot.empty:
        return []
    near = [_village(v) for v in hot["Nearest Village"] if v] or list(hot["Place"])
    new = int((hot["Pattern"] == NEW).sum())
    return [Action(
        "warn", "Alert the villages beside new and growing hotspots",
        names(near),
        f"{plural(new, 'new hotspot')}, {len(hot) - new} hot for consecutive months; "
        f"{plural(hot['Recent Events'].sum(), 'event')} in the last three months",
        "Now")]


def _refuge_actions(refuges: Optional[Dict[str, object]]) -> List[Action]:
    if not refuges:
        return []
    actions: List[Action] = []
    table = refuges.get("refuges")
    if table is not None and len(table) and "Linked Raids" in table:
        top = table.sort_values("Linked Raids", ascending=False).head(5)
        villages: List[str] = []
        for value in top.get("Villages Raided", pd.Series(dtype=str)):
            villages.extend(v.strip() for v in str(value).split(",") if v.strip())
        actions.append(Action(
            "refuge", "Dusk watch at the refuges that send most raids; warn their villages "
                      "before dark",
            names(f"{r['Refuge']} {r.get('Beat', '')}".strip() for r in top.to_dict("records")),
            f"{plural(top['Linked Raids'].sum(), 'raid')} linked; villages raided include "
            f"{names(villages, limit=5)}",
            "Dusk, on days the refuge is in use"))
    areas = refuges.get("areas")
    if areas is not None and len(areas) and "Raids" in areas:
        top = areas.sort_values("Raids", ascending=False).head(5)
        actions.append(Action(
            "search", "Search by day for the unreported refuges behind raid areas",
            names(f"{r['Area']} {r.get('Beat', '')}".strip() for r in top.to_dict("records")),
            f"{plural(top['Raids'].sum(), 'raid')} with no known refuge within a night's walk",
            "By day, after a raid"))
    return actions


# ---------------------------------------------------------------------------
# Per beat
# ---------------------------------------------------------------------------
def beat_driver(row: Dict[str, object]) -> str:
    """What is going on in a beat, in a few words: who, what, which way."""
    parts: List[str] = []
    bulls = row.get("Bull-Type Conflict %")
    if bulls == bulls and bulls is not None and row.get("Group-Known Conflicts", 0) >= \
            MIN_CONFLICTS_FOR_COMPOSITION:
        if bulls >= BULL_SHARE_FOR_ALERT:
            parts.append(f"bulls {bulls:.0f}%")
        elif bulls <= 100 - BULL_SHARE_FOR_ALERT:
            parts.append(f"herds {100 - bulls:.0f}%")
        else:
            parts.append("bulls and herds")
    crop, house = row.get("Crop Damage Events", 0), row.get("House Damage Events", 0)
    if house > crop:
        parts.append("houses")
    elif crop:
        parts.append("crops")
    if row.get("Recent Deaths", 0) or row.get("Recent Injuries", 0):
        parts.append("recent casualty")
    elif row.get("Human Deaths", 0):
        parts.append("past death")
    trend = row.get("Trend")
    if trend == "Escalating":
        parts.append(f"up, {int(row['Recent Conflicts'])} vs {int(row['Prior Conflicts'])}")
    elif trend == "Easing":
        parts.append(f"easing, {int(row['Recent Conflicts'])} vs {int(row['Prior Conflicts'])}")
    return ", ".join(parts).capitalize() if parts else "Presence only"


# ---------------------------------------------------------------------------
# Ecology
# ---------------------------------------------------------------------------
def _num(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _seen(df: pd.DataFrame) -> pd.Series:
    """Direct sightings: the elephants were there when the report was made.

    Signs and damage are written up the next morning, so neither their
    hour nor their group count describes the animals at that moment.
    """
    return ~(indirect_mask(df) | property_damage_mask(df))


def herd_sizes(df: pd.DataFrame) -> pd.Series:
    """Direct sightings by the size of the group seen."""
    sizes = _num(df, "Total Count")[_seen(df)]
    sizes = sizes[sizes > 0]
    return pd.Series({label: int(((sizes >= low) & (sizes <= high)).sum())
                      for low, high, label in SIZE_BANDS})


def calf_share(df: pd.DataFrame) -> Optional[float]:
    """Share of sexed direct sightings with calves in the group."""
    seen = df[_seen(df)]
    sexed = (_num(seen, "Male Count") + _num(seen, "Female Count") + _num(seen, "Calf Count")) > 0
    if sexed.sum() < 10:
        return None
    return float((_num(seen, "Calf Count")[sexed] > 0).mean())


def diel_activity(df: pd.DataFrame) -> pd.Series:
    """Direct sightings by hour of day, 0-23."""
    if "Hour" not in df:
        return pd.Series(0, index=range(24))
    hours = pd.to_numeric(df.loc[_seen(df), "Hour"], errors="coerce").dropna().astype(int)
    return hours.value_counts().reindex(range(24), fill_value=0).sort_index()


def range_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Beats with any report each month, and how many were used for the first time."""
    if df.empty or "Beat" not in df or "Date" not in df:
        return pd.DataFrame(columns=["Beats used", "First used"])
    dated = df.dropna(subset=["Date"])
    month = dated["Date"].dt.to_period("M")
    used = dated.groupby(month)["Beat"].nunique()
    first = dated.groupby("Beat")["Date"].min().dt.to_period("M").value_counts()
    out = pd.DataFrame({"Beats used": used}).join(first.rename("First used")).fillna(0)
    index = pd.period_range(month.min(), month.max(), freq="M")
    out = out.reindex(index, fill_value=0).astype(int)
    # In the register's first month every beat is new by definition; only
    # later firsts say the elephants moved somewhere they had not been.
    out.iloc[0, out.columns.get_loc("First used")] = 0
    out.index = out.index.to_timestamp()
    return out


def presence_grid(df: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """Reports by division (rows) and month (columns): where elephants were when."""
    if df.empty or "Division" not in df or "Date" not in df:
        return pd.DataFrame()
    dated = df.dropna(subset=["Date"])
    keep = dated["Division"].value_counts().head(top).index
    dated = dated[dated["Division"].isin(keep)]
    grid = pd.crosstab(dated["Division"], dated["Date"].dt.to_period("M"))
    grid = grid.reindex(columns=pd.period_range(grid.columns.min(), grid.columns.max(),
                                                freq="M"), fill_value=0)
    grid = grid.loc[keep]
    grid.columns = [p.strftime("%b %y") for p in grid.columns]
    return grid.astype(float)


def damage_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Crop and house or store damage by calendar month, all years together."""
    if df.empty or "Date" not in df:
        return pd.DataFrame(columns=["Crop", "House"])
    kind = classify_conflict(df)
    month = df["Date"].dt.month
    out = pd.DataFrame({
        "Crop": (kind == "Crop").groupby(month).sum(),
        "House": (kind == "House").groupby(month).sum(),
    }).reindex(range(1, 13), fill_value=0).fillna(0).astype(int)
    out.index = [pd.Timestamp(2000, m, 1).strftime("%b") for m in out.index]
    return out


def ecology_notes(df: pd.DataFrame) -> List[str]:
    """What the register says about the elephants, as plain sentences."""
    if df.empty:
        return []
    notes: List[str] = []
    groups = classify_group(df[_seen(df)])
    recorded = groups[groups != "Unrecorded"]
    if len(recorded) >= 10:
        share = recorded.value_counts(normalize=True)
        bulls = float(share.reindex(list(BULL_TYPE_GROUPS)).fillna(0).sum())
        herds = float(share.get("Family herd", 0.0))
        notes.append(f"Of {plural(len(recorded), 'direct sighting')} where the group was "
                     f"recorded, {bulls:.0%} were lone bulls or all-male parties and "
                     f"{herds:.0%} breeding herds with females or calves.")
    sizes = _num(df, "Total Count")[_seen(df)]
    sizes = sizes[sizes > 0]
    if len(sizes) >= 10:
        line = (f"The median group seen was {plural(sizes.median(), 'elephant')}, the "
                f"largest {int(sizes.max())}")
        calves = calf_share(df)
        if calves is not None:
            line += f"; calves were in {calves:.0%} of sexed sightings"
        notes.append(line + ".")
    hours = diel_activity(df)
    if hours.sum() >= 20:
        night = hours[(hours.index >= NIGHT_HOUR_START) | (hours.index < NIGHT_HOUR_END)].sum()
        dawn = hours.loc[4:8].sum()
        dusk = hours.loc[16:20].sum()
        notes.append(f"{night / hours.sum():.0%} of direct sightings fall between "
                     f"{_night_window()}; {(dawn + dusk) / hours.sum():.0%} at dawn "
                     "(04:00-09:00) or dusk (16:00-21:00), when herds move between cover "
                     "and fields.")
    used = range_by_month(df)
    if len(used) >= 3:
        recent_first = int(used["First used"].tail(3).sum())
        line = (f"Elephants were reported from {plural(df['Beat'].nunique(), 'beat')}, "
                f"{int(used['Beats used'].median())} in a typical month")
        if recent_first:
            line += f"; {plural(recent_first, 'beat')} saw them for the first time in the last three months"
        notes.append(line + ".")
    return notes


def peak_hours(hours: pd.Series, width: int = 2) -> Optional[str]:
    """The busiest run of ``width`` hours, as ``18:00-20:00``."""
    if hours.sum() == 0:
        return None
    values = hours.to_numpy(dtype=float)
    windows = np.array([values[[(h + k) % 24 for k in range(width)]].sum() for h in range(24)])
    start = int(windows.argmax())
    return f"{start:02d}:00-{(start + width) % 24:02d}:00"
