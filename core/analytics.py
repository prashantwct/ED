"""Core analytics: severity scoring, conflict classification, KPIs, filters.

All functions here are pure (take a DataFrame, return a DataFrame/Series/
dict) so they can be tested and reused outside of Streamlit.

Two domain rules drive everything downstream and are worth stating up
front:

1. **Human casualties dominate.** A fatality is weighted per person
   killed, far above any property or crop loss. A ranking that lets
   accumulated crop damage outrank a death is not usable for
   protected-area management.
2. **"Conflict" includes fatalities.** A report whose only marker is a
   death still is a conflict event. Defining conflict as
   crop/house/injury only silently drops the worst incidents.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Points added to the severity score for each conflict signal present in
# a row. "Presence" alone (an elephant seen, no damage) contributes the
# smallest weight; human harm contributes the largest by a wide margin.
SEVERITY_WEIGHTS: Dict[str, float] = {
    "presence": 0.5,
    "grain": 1.5,
    "crop": 2.5,
    "house": 5.0,
    "injury": 25.0,
}

# Applied per person killed, not per incident, so a two-person fatality
# outranks a one-person one.
DEATH_WEIGHT_PER_PERSON = 100.0

DEATH_COUNT_COLS = ["Male Death Count", "Female Death Count", "Children Death Count"]
INJURY_COUNT_COLS = ["Male Injury Count", "Female Injury Count", "Children Injury Count"]

# Property/crop damage flags. Order matters only for readability.
DAMAGE_FLAG_COLS = ["Crop Damage", "Grain Damage", "House Damage"]

# Category hierarchy for map colouring and the priority table, most
# severe first. Each row gets exactly one category: the worst that applies.
CONFLICT_CATEGORIES = ["Death", "Injury", "House", "Crop", "Presence"]

# Severity band edges, chosen to line up with the weights above so each
# band means something concrete to a reader rather than being an
# equal-width slice of an arbitrary range:
#   < 1      presence only
#   1 -  5   crop / grain loss
#   5 - 25   house or store damage
#   25 - 100 human injury
#   >= 100   human fatality
SEVERITY_BAND_EDGES = [0.0, 1.0, 5.0, 25.0, 100.0, float("inf")]
SEVERITY_BAND_LABELS = [
    "Presence only",
    "Crop / grain loss",
    "Property damage",
    "Human injury",
    "Human fatality",
]

# Night is dusk-to-dawn elephant movement: 18:00-23:59 and 00:00-05:59.
NIGHT_HOUR_START = 18
NIGHT_HOUR_END = 6


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Return ``column`` as a zero-filled float Series aligned to ``df``.

    Missing columns become all-zero, so every downstream calculation
    degrades gracefully on a minimal CSV instead of raising.
    """
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _people_count(df: pd.DataFrame, flag_col: str, count_cols: List[str]) -> pd.Series:
    """Count people affected per row, gated on the incident flag.

    Args:
        df: Source dataframe.
        flag_col: The 0/1 incident flag (e.g. ``Death``).
        count_cols: Per-demographic count columns to sum when present.

    Returns:
        Float Series of people affected per row.

    The gate matters. Summing the count columns directly over-counts:
    field exports contain rows where a death count is filled in but the
    ``Death`` flag is not set, and in the cases reviewed these were
    same-day, same-beat follow-up reports of a fatality already logged
    elsewhere, not distinct deaths. Those rows are surfaced as a
    data-quality warning at load time rather than silently counted.

    Where the flag *is* set but no per-person breakdown was given, this
    assumes at least one person: the flag itself asserts the event
    happened, even without a demographic breakdown.
    """
    flagged = _numeric(df, flag_col) > 0

    present = [c for c in count_cols if c in df.columns]
    if not present:
        return flagged.astype(float)

    counts = sum(_numeric(df, c) for c in present)
    gated = counts.where(flagged, 0.0)
    # Flag set, no breakdown given -> at least one person.
    return gated.mask(flagged & (gated == 0), 1.0)


def human_deaths(df: pd.DataFrame) -> pd.Series:
    """People killed per row (see :func:`_people_count` for the gating rule)."""
    return _people_count(df, "Death", DEATH_COUNT_COLS)


def human_injuries(df: pd.DataFrame) -> pd.Series:
    """People injured per row.

    Uses ``Male/Female/Children Injury Count`` when the export carries
    them; otherwise falls back to treating ``Injury`` as a flag (one
    person). Some exports use ``Injury`` as a raw count instead of a
    flag -- values above 1 are respected in that case.
    """
    present = [c for c in INJURY_COUNT_COLS if c in df.columns]
    if present:
        return _people_count(df, "Injury", INJURY_COUNT_COLS)
    # No breakdown columns: honour Injury itself if it looks like a count.
    return _numeric(df, "Injury").clip(lower=0)


def conflict_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series marking rows that are human-elephant conflict events.

    A conflict is any crop, grain, house, injury or fatality marker.
    Fatalities are explicitly included: a death-only report with no
    property flag set is still -- obviously -- a conflict.
    """
    if df.empty:
        return pd.Series(dtype=bool, index=df.index)

    mask = pd.Series(False, index=df.index)
    for col in DAMAGE_FLAG_COLS:
        mask = mask | (_numeric(df, col) > 0)
    mask = mask | (human_injuries(df) > 0)
    mask = mask | (human_deaths(df) > 0)
    return mask


def compute_severity(df: pd.DataFrame) -> pd.Series:
    """Compute a per-row severity score from the damage/casualty markers.

    Any contributing column absent from ``df`` is treated as zero for
    every row, so the function degrades gracefully on minimal CSVs.

    Args:
        df: Sightings/conflict dataframe.

    Returns:
        A float Series aligned to ``df.index``.
    """
    deaths = human_deaths(df)
    injuries = human_injuries(df)

    return (
        (_numeric(df, "Total Count") > 0).astype(float) * SEVERITY_WEIGHTS["presence"]
        + (_numeric(df, "Crop Damage") > 0).astype(float) * SEVERITY_WEIGHTS["crop"]
        + (_numeric(df, "Grain Damage") > 0).astype(float) * SEVERITY_WEIGHTS["grain"]
        + (_numeric(df, "House Damage") > 0).astype(float) * SEVERITY_WEIGHTS["house"]
        + injuries * SEVERITY_WEIGHTS["injury"]
        + deaths * DEATH_WEIGHT_PER_PERSON
    )


def property_severity(df: pd.DataFrame) -> pd.Series:
    """Severity from crop, grain, house and presence only -- no casualties.

    Used where casualty pressure is already scored as its own signal, so
    that damage burden and human harm stay separable instead of one
    fatality swamping the property picture (a death contributes 100
    points, roughly forty crop-raids' worth).
    """
    return (
        (_numeric(df, "Total Count") > 0).astype(float) * SEVERITY_WEIGHTS["presence"]
        + (_numeric(df, "Crop Damage") > 0).astype(float) * SEVERITY_WEIGHTS["crop"]
        + (_numeric(df, "Grain Damage") > 0).astype(float) * SEVERITY_WEIGHTS["grain"]
        + (_numeric(df, "House Damage") > 0).astype(float) * SEVERITY_WEIGHTS["house"]
    )


def classify_conflict(df: pd.DataFrame) -> pd.Series:
    """Label each row with the single most severe conflict type present.

    A row with both crop damage and a fatality is labelled ``Death``,
    not ``Crop`` -- this drives map colouring and the priority table,
    where double-counting a row across categories would mislead.

    Returns:
        A string Series aligned to ``df.index``, values drawn from
        :data:`CONFLICT_CATEGORIES`.
    """
    if df.empty:
        return pd.Series(dtype=object, index=df.index, name="Conflict Category")

    conditions = [
        human_deaths(df) > 0,
        human_injuries(df) > 0,
        _numeric(df, "House Damage") > 0,
        (_numeric(df, "Crop Damage") > 0) | (_numeric(df, "Grain Damage") > 0),
    ]
    choices = ["Death", "Injury", "House", "Crop"]

    return pd.Series(
        np.select(conditions, choices, default="Presence"),
        index=df.index,
        name="Conflict Category",
    )


def compute_is_night(df: pd.DataFrame) -> pd.Series:
    """Derive a nullable boolean ``Is_Night`` flag from an ``Hour`` column.

    A row is night if its hour is >= 18 or < 6. Rows with no usable hour
    get ``pd.NA`` rather than being silently defaulted to day, so
    downstream KPIs can report "unknown" instead of misleadingly
    counting them as daytime.

    Args:
        df: Dataframe that may contain an ``Hour`` column (0-23).

    Returns:
        A nullable boolean Series aligned to ``df.index``.
    """
    if "Hour" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")

    hour = pd.to_numeric(df["Hour"], errors="coerce")
    is_night = (hour >= NIGHT_HOUR_START) | (hour < NIGHT_HOUR_END)
    # Preserve unknown-ness where hour itself was unknown.
    return is_night.astype("boolean").where(hour.notna(), pd.NA)


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Compute the headline KPIs shown at the top of the dashboard.

    Args:
        df: Enriched dataframe, expected to already contain a
            ``Severity Score`` column and ideally ``Is_Night``.

    Returns:
        Dict with ``entries``, ``conflicts``, ``conflict_rate``,
        ``human_deaths``, ``human_death_incidents``, ``human_injuries``,
        ``severity``, ``night_pct`` and ``night_known`` (the number of
        rows the night percentage is actually based on -- 40% over 12
        known rows and 40% over 1,200 are very different claims).
        ``night_pct`` is NaN when no row has a known hour.
    """
    empty = {
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
    if df.empty:
        return empty

    deaths = human_deaths(df)
    injuries = human_injuries(df)
    conflicts = int(conflict_mask(df).sum())

    is_night = df.get("Is_Night")
    if is_night is not None and is_night.notna().any():
        known = is_night.dropna()
        night_pct = float(known.astype(bool).mean() * 100)
        night_known = int(len(known))
    else:
        night_pct = float("nan")
        night_known = 0

    return {
        "entries": int(len(df)),
        "conflicts": conflicts,
        "conflict_rate": float(conflicts / len(df) * 100),
        "human_deaths": float(deaths.sum()),
        "human_death_incidents": int((deaths > 0).sum()),
        "human_injuries": float(injuries.sum()),
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
    """Apply sidebar filters to the dataframe.

    Every filter is optional / no-op when left as ``None`` or empty, so
    callers can pass through whatever the sidebar widgets currently hold
    without special-casing "no filter selected".

    Args:
        df: Source dataframe (expects ``Date``, ``Division``, ``Range``,
            ``Beat``, and ``Severity Score`` columns).
        date_range: Optional ``(start_date, end_date)`` tuple (inclusive).
        divisions: Optional list of Division values to keep.
        ranges: Optional list of Range values to keep.
        beats: Optional list of Beat values to keep.
        min_severity: Minimum ``Severity Score`` to keep (default 0).

    Returns:
        A filtered copy of ``df``.
    """
    out = df.copy()

    if date_range and len(date_range) == 2 and all(date_range):
        start, end = date_range
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        out = out[(out["Date"] >= start_ts) & (out["Date"] <= end_ts)]

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
    """Bucket rows into fixed, interpretable severity bands.

    Deliberately *not* equal-width bins. With fatalities weighted at 100
    and a presence sighting at 0.5, equal-width binning across the
    observed range dumps ~99% of rows into the first bucket and tells
    the reader nothing. The fixed edges in
    :data:`SEVERITY_BAND_EDGES` each correspond to a real incident type.

    Args:
        df: Dataframe with a ``Severity Score`` column.

    Returns:
        DataFrame with columns ``Band`` and ``Count``, one row per band
        (including empty bands, so the shape is stable across filters).
        Empty input yields an empty DataFrame.
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
    """Summarise entry counts and average severity by night vs. day.

    Args:
        df: Dataframe with ``Is_Night`` and ``Severity Score`` columns.

    Returns:
        DataFrame with columns ``Period``, ``Entries``, ``Avg Severity``.
        Rows with unknown ``Is_Night`` are reported separately as
        "Unknown" rather than dropped, so totals still reconcile.
    """
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
    order = {"Day": 0, "Night": 1, "Unknown": 2}
    grouped["_order"] = grouped["Period"].map(order)
    return grouped.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def division_conflict_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Sightings, conflict events and conflict rate per division.

    The normalised view. Raw sighting volume alone mostly measures
    reporting activity -- it favours whichever division files the most
    reports, not the one with the most actual conflict.

    Returns:
        DataFrame indexed by Division with ``Sightings``,
        ``Conflict Events``, ``Human Deaths`` and ``Conflict Rate %``,
        sorted by rate descending.
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
    """Total sightings vs. conflict events by month, for the trend chart."""
    columns = ["Sightings", "Conflict Events"]
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame(columns=columns)

    month = df["Date"].dt.to_period("M").astype(str)
    out = df.groupby(month, observed=True).size().rename("Sightings").to_frame()
    out["Conflict Events"] = conflict_mask(df).groupby(month, observed=True).sum()
    out.index.name = "Month"
    return out[columns].sort_index()


def hourly_conflict_profile(df: pd.DataFrame) -> pd.Series:
    """Conflict-event counts by hour of day (0-23), zero-filled.

    Only conflict rows are counted. A profile of *all* sightings mostly
    traces when staff are on patrol; restricting to conflict events is
    what identifies the risk window worth staffing.
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
