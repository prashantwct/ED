"""Severity scoring, conflict classification, KPIs and filters.

Pure functions over DataFrames so they can be tested and reused outside
Streamlit. Two domain rules drive everything downstream:

1. Human casualties dominate severity, weighted per person killed.
2. "Conflict" includes fatalities. A death-only report is a conflict.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import (
    DEATH_WEIGHT_PER_PERSON,
    SOLITARY_MAX_GROUP,
    NIGHT_HOUR_END,
    NIGHT_HOUR_START,
    SEVERITY_BAND_EDGES,
    SEVERITY_BAND_LABELS,
    SEVERITY_WEIGHTS,
)

DEATH_COUNT_COLS = ["Male Death Count", "Female Death Count", "Children Death Count"]
INJURY_COUNT_COLS = ["Male Injury Count", "Female Injury Count", "Children Injury Count"]
DAMAGE_FLAG_COLS = ["Crop Damage", "Grain Damage", "House Damage"]

# Most severe first; each row gets exactly one.
CONFLICT_CATEGORIES = ["Death", "Injury", "House", "Crop", "Presence"]


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Return ``column`` as a zero-filled float Series, or zeros if absent."""
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _people_count(df: pd.DataFrame, flag_col: str, count_cols: List[str]) -> pd.Series:
    """Count people affected per row, gated on the incident flag.

    The gate matters: exports contain rows with a death count filled in
    but the flag unset, which have matched same-day, same-beat follow-up
    reports of a death already logged. Summing the count columns directly
    double-counts those. Where the flag is set with no breakdown, assume
    one person.
    """
    flagged = _numeric(df, flag_col) > 0

    present = [c for c in count_cols if c in df.columns]
    if not present:
        return flagged.astype(float)

    counts = sum(_numeric(df, c) for c in present)
    gated = counts.where(flagged, 0.0)
    return gated.mask(flagged & (gated == 0), 1.0)


def human_deaths(df: pd.DataFrame) -> pd.Series:
    """People killed per row."""
    return _people_count(df, "Death", DEATH_COUNT_COLS)


def human_injuries(df: pd.DataFrame) -> pd.Series:
    """People injured per row.

    Uses the demographic breakdown when present, else treats ``Injury``
    as a count (respecting values above 1) rather than a flag.
    """
    if [c for c in INJURY_COUNT_COLS if c in df.columns]:
        return _people_count(df, "Injury", INJURY_COUNT_COLS)
    return _numeric(df, "Injury").clip(lower=0)


def conflict_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that are human-elephant conflict events, fatalities included."""
    if df.empty:
        return pd.Series(dtype=bool, index=df.index)

    mask = pd.Series(False, index=df.index)
    for col in DAMAGE_FLAG_COLS:
        mask = mask | (_numeric(df, col) > 0)
    return mask | (human_injuries(df) > 0) | (human_deaths(df) > 0)


def compute_severity(df: pd.DataFrame) -> pd.Series:
    """Per-row severity score. Absent columns count as zero."""
    return (
        (_numeric(df, "Total Count") > 0).astype(float) * SEVERITY_WEIGHTS["presence"]
        + (_numeric(df, "Crop Damage") > 0).astype(float) * SEVERITY_WEIGHTS["crop"]
        + (_numeric(df, "Grain Damage") > 0).astype(float) * SEVERITY_WEIGHTS["grain"]
        + (_numeric(df, "House Damage") > 0).astype(float) * SEVERITY_WEIGHTS["house"]
        + human_injuries(df) * SEVERITY_WEIGHTS["injury"]
        + human_deaths(df) * DEATH_WEIGHT_PER_PERSON
    )


def property_severity(df: pd.DataFrame) -> pd.Series:
    """Severity from crop, grain, house and presence only.

    Used where casualties are scored separately, so one fatality (100
    points, ~40 crop raids' worth) does not swamp the property picture.
    """
    return (
        (_numeric(df, "Total Count") > 0).astype(float) * SEVERITY_WEIGHTS["presence"]
        + (_numeric(df, "Crop Damage") > 0).astype(float) * SEVERITY_WEIGHTS["crop"]
        + (_numeric(df, "Grain Damage") > 0).astype(float) * SEVERITY_WEIGHTS["grain"]
        + (_numeric(df, "House Damage") > 0).astype(float) * SEVERITY_WEIGHTS["house"]
    )


def classify_conflict(df: pd.DataFrame) -> pd.Series:
    """Label each row with the most severe conflict type present.

    One category per row, so map colours and category counts do not
    double-count an incident involving both a death and crop damage.
    """
    if df.empty:
        return pd.Series(dtype=object, index=df.index, name="Conflict Category")

    conditions = [
        human_deaths(df) > 0,
        human_injuries(df) > 0,
        _numeric(df, "House Damage") > 0,
        (_numeric(df, "Crop Damage") > 0) | (_numeric(df, "Grain Damage") > 0),
    ]
    return pd.Series(
        np.select(conditions, ["Death", "Injury", "House", "Crop"], default="Presence"),
        index=df.index,
        name="Conflict Category",
    )


def classify_group(df: pd.DataFrame) -> pd.Series:
    """Label each sighting with the kind of group that was seen.

    The distinction that matters operationally is bull against breeding
    herd. A bull raids and occasionally kills; a herd with calves is
    avoiding people and needs safe passage, not deterrence. Only bulls
    carry tusks in this species, so ``Male Count`` is the tusker count.

    Rows with no composition recorded return "Unrecorded" rather than
    being folded into a group they might not belong to.
    """
    if df.empty:
        return pd.Series(dtype=object, index=df.index, name="Group Type")

    male = _numeric(df, "Male Count")
    female = _numeric(df, "Female Count")
    calves = _numeric(df, "Calf Count")
    total = _numeric(df, "Total Count")

    recorded = (male + female + calves + _numeric(df, "Unknown Count")) > 0
    conditions = [
        ~recorded,
        (calves > 0) | (female > 0),
        (male > 0) & (total <= 1),
        (male > 0) & (total <= SOLITARY_MAX_GROUP),
    ]
    return pd.Series(
        np.select(
            conditions,
            ["Unrecorded", "Family herd", "Lone bull", "Bull party"],
            default="Mixed / unsexed",
        ),
        index=df.index,
        name="Group Type",
    )


BULL_TYPE_GROUPS = ("Lone bull", "Bull party")


def is_bull_type(df: pd.DataFrame) -> pd.Series:
    """Whether each sighting is a lone bull or a small all-male party."""
    return classify_group(df).isin(BULL_TYPE_GROUPS)


def composition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Damage rate by group type, which is the evidence for the split."""
    if df.empty:
        return pd.DataFrame(
            columns=["Group Type", "Sightings", "Conflict Events",
                     "Damage Rate %", "Human Deaths"]
        )

    working = df.assign(
        _group=classify_group(df),
        _conflict=conflict_mask(df).astype(int),
        _deaths=human_deaths(df),
    )
    summary = (
        working.groupby("_group", observed=True)
        .agg(Sightings=("_conflict", "size"),
             **{"Conflict Events": ("_conflict", "sum"),
                "Human Deaths": ("_deaths", "sum")})
        .reset_index()
        .rename(columns={"_group": "Group Type"})
    )
    summary["Damage Rate %"] = (
        summary["Conflict Events"] / summary["Sightings"] * 100
    ).round(1)
    order = ["Lone bull", "Bull party", "Family herd", "Mixed / unsexed", "Unrecorded"]
    summary["_order"] = summary["Group Type"].map(
        {name: i for i, name in enumerate(order)}
    ).fillna(len(order))
    return (
        summary.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)[
            ["Group Type", "Sightings", "Conflict Events", "Damage Rate %",
             "Human Deaths"]
        ]
    )


def compute_is_night(df: pd.DataFrame) -> pd.Series:
    """Nullable boolean night flag from ``Hour``.

    Rows with no usable hour get ``pd.NA`` rather than defaulting to day,
    so KPIs can report "unknown" instead of counting them as daytime.
    """
    if "Hour" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")

    hour = pd.to_numeric(df["Hour"], errors="coerce")
    is_night = (hour >= NIGHT_HOUR_START) | (hour < NIGHT_HOUR_END)
    return is_night.astype("boolean").where(hour.notna(), pd.NA)


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Headline KPIs.

    ``night_known`` is the row count behind ``night_pct``: 40% over 12
    known rows and over 1,200 are very different claims.
    """
    if df.empty:
        return {
            "entries": 0,
            "conflicts": 0,
            "conflict_rate": float("nan"),
            "human_deaths": 0.0,
            "human_death_incidents": 0,
            "human_injuries": 0.0,
            "severity": 0.0,
            "night_pct": float("nan"),
            "night_known": 0,
        }

    deaths = human_deaths(df)
    conflicts = int(conflict_mask(df).sum())

    is_night = df.get("Is_Night")
    if is_night is not None and is_night.notna().any():
        known = is_night.dropna()
        night_pct = float(known.astype(bool).mean() * 100)
        night_known = int(len(known))
    else:
        night_pct, night_known = float("nan"), 0

    return {
        "entries": int(len(df)),
        "conflicts": conflicts,
        "conflict_rate": float(conflicts / len(df) * 100),
        "human_deaths": float(deaths.sum()),
        "human_death_incidents": int((deaths > 0).sum()),
        "human_injuries": float(human_injuries(df).sum()),
        "severity": float(_numeric(df, "Severity Score").sum()),
        "night_pct": night_pct,
        "night_known": night_known,
    }


def filter_dataframe(
    df: pd.DataFrame,
    date_range: Optional[tuple] = None,
    divisions: Optional[List[str]] = None,
    ranges: Optional[List[str]] = None,
    beats: Optional[List[str]] = None,
    min_severity: float = 0.0,
) -> pd.DataFrame:
    """Apply sidebar filters. Every filter is a no-op when empty."""
    out = df.copy()

    if date_range and len(date_range) == 2 and all(date_range):
        start, end = date_range
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        out = out[(out["Date"] >= pd.Timestamp(start)) & (out["Date"] <= end_ts)]

    if divisions:
        out = out[out["Division"].isin(divisions)]
    if ranges:
        out = out[out["Range"].isin(ranges)]
    if beats:
        out = out[out["Beat"].isin(beats)]
    if "Severity Score" in out.columns and min_severity > 0:
        out = out[out["Severity Score"] >= min_severity]

    return out.reset_index(drop=True)


def severity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per severity band.

    Fixed edges, not equal-width bins: with fatalities at 100 and
    sightings at 0.5, equal-width binning puts ~99% in the first bucket.
    """
    if df.empty or "Severity Score" not in df.columns:
        return pd.DataFrame(columns=["Band", "Count"])

    banded = pd.cut(
        df["Severity Score"],
        bins=SEVERITY_BAND_EDGES,
        labels=SEVERITY_BAND_LABELS,
        right=False,
        include_lowest=True,
    )
    counts = banded.value_counts().reindex(SEVERITY_BAND_LABELS, fill_value=0)
    return pd.DataFrame({"Band": SEVERITY_BAND_LABELS, "Count": counts.to_numpy()})


def night_day_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Entry counts and mean severity by night/day, with unknowns kept."""
    if df.empty or "Is_Night" not in df.columns:
        return pd.DataFrame(columns=["Period", "Entries", "Avg Severity"])

    labels = df["Is_Night"].map({True: "Night", False: "Day"}).astype(object)
    labels = labels.where(labels.notna(), "Unknown")

    grouped = (
        df.assign(Period=labels)
        .groupby("Period", observed=True)
        .agg(
            Entries=("Severity Score", "size"),
            **{"Avg Severity": ("Severity Score", "mean")},
        )
        .reset_index()
    )
    grouped["_order"] = grouped["Period"].map({"Day": 0, "Night": 1, "Unknown": 2})
    return grouped.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def division_conflict_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Sightings, conflict events and conflict rate per division.

    Raw volume mostly measures reporting activity, not conflict.
    """
    columns = ["Sightings", "Conflict Events", "Human Deaths", "Conflict Rate %"]
    if df.empty or "Division" not in df.columns:
        return pd.DataFrame(columns=columns)

    grouper = df["Division"]
    out = df.groupby(grouper, observed=True).size().rename("Sightings").to_frame()
    out["Conflict Events"] = conflict_mask(df).groupby(grouper, observed=True).sum()
    out["Human Deaths"] = human_deaths(df).groupby(grouper, observed=True).sum()
    out["Conflict Rate %"] = (out["Conflict Events"] / out["Sightings"] * 100).round(1)
    return out[columns].sort_values("Conflict Rate %", ascending=False)


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Sightings and conflict events by month."""
    columns = ["Sightings", "Conflict Events"]
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame(columns=columns)

    month = df["Date"].dt.to_period("M").astype(str)
    out = df.groupby(month, observed=True).size().rename("Sightings").to_frame()
    out["Conflict Events"] = conflict_mask(df).groupby(month, observed=True).sum()
    out.index.name = "Month"
    return out[columns].sort_index()


def hourly_conflict_profile(df: pd.DataFrame) -> pd.Series:
    """Conflict events by hour of day, zero-filled.

    Conflict rows only: a profile of all sightings mostly traces when
    patrols go out.
    """
    counts = pd.Series(0, index=pd.RangeIndex(24), dtype=int)
    counts.index.name = "Hour"
    counts.name = "Conflict Events"

    if df.empty or "Hour" not in df.columns:
        return counts

    hours = pd.to_numeric(df.loc[conflict_mask(df), "Hour"], errors="coerce").dropna()
    observed = hours.astype(int).value_counts()
    counts.update(observed.reindex(range(24)).dropna().astype(int))
    return counts


# --- Trend views ------------------------------------------------------------
# Conflict types drawn in the trend charts, most severe first. Presence is
# not conflict, so it is left out rather than stacked under the rest.
TREND_CATEGORIES = ["Death", "Injury", "House", "Crop"]


def _month_index(dates: pd.Series) -> pd.PeriodIndex:
    """Every month from the first report to the last, gaps included.

    A month with no conflict is a zero, not a missing point: dropping it
    draws a line straight across the quiet spell and hides it.
    """
    months = dates.dt.to_period("M")
    return pd.period_range(months.min(), months.max(), freq="M", name="Month")


def monthly_conflict_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Conflict events per month by type, with rate and casualties.

    One row per calendar month, zero-filled. Columns: one per
    ``TREND_CATEGORIES`` entry, ``Conflict Events``, ``Sightings``,
    ``Conflict Rate %``, ``Human Deaths``, ``People Injured`` and
    ``Rolling Conflicts`` (a trailing three-month mean, which is what
    separates a trend from a noisy month).
    """
    columns = TREND_CATEGORIES + [
        "Conflict Events", "Sightings", "Conflict Rate %",
        "Human Deaths", "People Injured", "Rolling Conflicts",
    ]
    if df.empty or "Date" not in df.columns or df["Date"].isna().all():
        return pd.DataFrame(columns=columns)

    dated = df[df["Date"].notna()]
    index = _month_index(dated["Date"])
    month = dated["Date"].dt.to_period("M")

    category = classify_conflict(dated)
    out = (
        pd.crosstab(month, category)
        .reindex(index=index, columns=TREND_CATEGORIES, fill_value=0)
        .astype(int)
    )
    out["Conflict Events"] = out[TREND_CATEGORIES].sum(axis=1)
    out["Sightings"] = month.value_counts().reindex(index, fill_value=0).astype(int)
    out["Conflict Rate %"] = (
        (out["Conflict Events"] / out["Sightings"].where(out["Sightings"] > 0) * 100)
        .round(1)
    )
    out["Human Deaths"] = human_deaths(dated).groupby(month).sum().reindex(index, fill_value=0)
    out["People Injured"] = (
        human_injuries(dated).groupby(month).sum().reindex(index, fill_value=0)
    )
    out["Rolling Conflicts"] = out["Conflict Events"].rolling(3, min_periods=1).mean().round(1)
    out.columns.name = None
    out.index = out.index.to_timestamp()
    out.index.name = "Month"
    return out[columns]


def seasonal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Conflict events by year (rows) and calendar month (columns).

    Reading down a column compares the same month across years, which is
    the only fair comparison for a seasonal species. Months outside the
    data's span are NaN, not zero, so they are not read as quiet.
    """
    month_names = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]
    if df.empty or "Date" not in df.columns or df["Date"].isna().all():
        return pd.DataFrame(columns=month_names)

    dated = df[df["Date"].notna()]
    conflicts = dated[conflict_mask(dated)]
    span = _month_index(dated["Date"])
    years = sorted(span.year.unique())

    counts = (
        pd.crosstab(conflicts["Date"].dt.year, conflicts["Date"].dt.month)
        if not conflicts.empty
        else pd.DataFrame()
    )
    matrix = pd.DataFrame(np.nan, index=years, columns=range(1, 13))
    for period in span:
        value = 0
        if period.year in counts.index and period.month in counts.columns:
            value = int(counts.loc[period.year, period.month])
        matrix.loc[period.year, period.month] = value
    matrix.columns = month_names
    matrix.index.name = "Year"
    return matrix


def division_monthly_conflict(df: pd.DataFrame) -> pd.DataFrame:
    """Conflict events per month per division, long form, zero-filled."""
    columns = ["Month", "Division", "Conflict Events"]
    if df.empty or "Division" not in df.columns or "Date" not in df.columns:
        return pd.DataFrame(columns=columns)

    dated = df[df["Date"].notna()]
    if dated.empty:
        return pd.DataFrame(columns=columns)

    index = _month_index(dated["Date"])
    conflicts = dated[conflict_mask(dated)]
    divisions = sorted(dated["Division"].astype(str).unique())
    grid = (
        pd.crosstab(
            conflicts["Date"].dt.to_period("M"), conflicts["Division"].astype(str)
        )
        if not conflicts.empty
        else pd.DataFrame()
    ).reindex(index=index, columns=divisions, fill_value=0)
    grid.index = grid.index.to_timestamp()
    grid.index.name = "Month"
    return (
        grid.reset_index()
        .melt(id_vars="Month", var_name="Division", value_name="Conflict Events")
        .astype({"Conflict Events": int})[columns]
    )


def window_comparison(
    df: pd.DataFrame, days: int, as_of: Optional[pd.Timestamp] = None
) -> Dict[str, float]:
    """Conflict, casualties and rate in the last ``days`` against the ``days`` before.

    Anchored to ``as_of`` (default: the last report) so the headline
    change matches the period the reader selected. Returns NaN prior
    figures when the data does not reach back a full prior window,
    rather than comparing against a half-empty one.
    """
    keys = ["conflicts", "casualties", "rate"]
    blank = {f"{k}_{w}": float("nan") for k in keys for w in ("recent", "prior")}
    if df.empty or "Date" not in df.columns or df["Date"].isna().all():
        return blank

    end = pd.Timestamp(as_of) if as_of is not None else df["Date"].max()
    end = end.normalize() + pd.Timedelta(days=1)
    recent_start = end - pd.Timedelta(days=days)
    prior_start = recent_start - pd.Timedelta(days=days)

    def figures(frame: pd.DataFrame) -> Dict[str, float]:
        conflicts = float(conflict_mask(frame).sum()) if len(frame) else 0.0
        return {
            "conflicts": conflicts,
            "casualties": float(human_deaths(frame).sum() + human_injuries(frame).sum())
            if len(frame) else 0.0,
            "rate": conflicts / len(frame) * 100 if len(frame) else float("nan"),
        }

    recent = figures(df[(df["Date"] >= recent_start) & (df["Date"] < end)])
    out = {f"{k}_recent": v for k, v in recent.items()}
    if df["Date"].min() > prior_start:
        out.update({f"{k}_prior": float("nan") for k in keys})
    else:
        prior = figures(df[(df["Date"] >= prior_start) & (df["Date"] < recent_start)])
        out.update({f"{k}_prior": v for k, v in prior.items()})
    return out
