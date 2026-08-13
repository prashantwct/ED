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
