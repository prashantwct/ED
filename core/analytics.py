"""Core analytics: severity scoring, night/day classification, KPIs, filters.

All functions here are pure (take a DataFrame, return a DataFrame/Series/
dict) so they can be tested and reused outside of Streamlit.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Points added to the severity score for each conflict signal present in
# a row. "Presence" alone (an elephant seen, no damage) contributes the
# smallest weight; a human injury contributes the largest.
SEVERITY_WEIGHTS: Dict[str, float] = {
    "presence": 0.5,
    "crop": 2.5,
    "house": 5.0,
    "injury": 20.0,
}

# Night is defined as 18:00-23:59 or 00:00-06:00 inclusive, per field
# convention (dusk-to-dawn elephant movement window).
NIGHT_HOUR_START = 18
NIGHT_HOUR_END = 6


def compute_severity(df: pd.DataFrame) -> pd.Series:
    """Compute a per-row severity score from optional damage/injury flags.

    Any of ``Total Count``, ``Crop Damage``, ``House Damage``, ``Injury``
    that are absent from ``df`` are treated as zero for every row, so the
    function degrades gracefully on minimal CSVs.

    Args:
        df: Sightings/conflict dataframe.

    Returns:
        A float Series aligned to ``df.index``.
    """
    presence = (df.get("Total Count", pd.Series(0, index=df.index)) > 0).astype(int)
    crop = (df.get("Crop Damage", pd.Series(0, index=df.index)) > 0).astype(int)
    house = (df.get("House Damage", pd.Series(0, index=df.index)) > 0).astype(int)
    injury = (df.get("Injury", pd.Series(0, index=df.index)) > 0).astype(int)

    return (
        presence * SEVERITY_WEIGHTS["presence"]
        + crop * SEVERITY_WEIGHTS["crop"]
        + house * SEVERITY_WEIGHTS["house"]
        + injury * SEVERITY_WEIGHTS["injury"]
    )


def compute_is_night(df: pd.DataFrame) -> pd.Series:
    """Derive a boolean ``Is_Night`` flag from an ``Hour`` column.

    A row is night if its hour is >= 18 or <= 6 (inclusive on both ends,
    covering dusk through dawn). Rows with no usable hour get ``pd.NA``
    rather than being silently defaulted to day, so downstream KPIs can
    correctly report "unknown" instead of misleadingly counting them.

    Args:
        df: Dataframe that may contain an ``Hour`` column (0-23).

    Returns:
        A nullable boolean Series aligned to ``df.index``.
    """
    if "Hour" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")

    hour = pd.to_numeric(df["Hour"], errors="coerce")
    is_night = (hour >= NIGHT_HOUR_START) | (hour <= NIGHT_HOUR_END)
    # Preserve unknown-ness where hour itself was unknown.
    return is_night.astype("boolean").where(hour.notna(), pd.NA)


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Compute the headline KPIs shown at the top of the dashboard.

    Args:
        df: Enriched dataframe, expected to already contain
            ``Severity Score`` and ``Is_Night`` columns.

    Returns:
        Dict with ``entries``, ``conflicts``, ``severity``, and
        ``night_pct`` (``night_pct`` is ``float('nan')`` if no rows have
        a known Is_Night value).
    """
    if df.empty:
        return {"entries": 0, "conflicts": 0, "severity": 0.0, "night_pct": float("nan")}

    zeros = pd.Series(0, index=df.index)
    conflict_mask = (
        (df.get("Crop Damage", zeros) > 0)
        | (df.get("House Damage", zeros) > 0)
        | (df.get("Injury", zeros) > 0)
    )

    is_night = df.get("Is_Night")
    if is_night is not None and is_night.notna().any():
        night_pct = float(is_night.dropna().astype(bool).mean() * 100)
    else:
        night_pct = float("nan")

    return {
        "entries": int(len(df)),
        "conflicts": int(conflict_mask.sum()),
        "severity": float(df.get("Severity Score", pd.Series(dtype=float)).sum()),
        "night_pct": night_pct,
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


def severity_distribution(df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Bucket rows into severity bands for charting.

    Args:
        df: Dataframe with a ``Severity Score`` column.
        bins: Number of equal-width bands to create.

    Returns:
        DataFrame with columns ``Band`` and ``Count``, empty if input is
        empty or has no variation to bin.
    """
    if df.empty or "Severity Score" not in df.columns or df["Severity Score"].nunique() < 2:
        if df.empty or "Severity Score" not in df.columns:
            return pd.DataFrame(columns=["Band", "Count"])
        # Single unique value - report it as one band rather than erroring.
        value = df["Severity Score"].iloc[0]
        return pd.DataFrame({"Band": [f"{value:.1f}"], "Count": [len(df)]})

    banded = pd.cut(df["Severity Score"], bins=bins)
    counts = banded.value_counts().sort_index()
    return pd.DataFrame(
        {"Band": [str(interval) for interval in counts.index], "Count": counts.values}
    )


def night_day_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise conflict counts and average severity by night vs. day.

    Args:
        df: Dataframe with ``Is_Night`` and ``Severity Score`` columns.

    Returns:
        DataFrame with columns ``Period``, ``Entries``, ``Avg Severity``.
        Rows with unknown ``Is_Night`` are reported separately as
        "Unknown" rather than dropped, so totals still reconcile.
    """
    if df.empty or "Is_Night" not in df.columns:
        return pd.DataFrame(columns=["Period", "Entries", "Avg Severity"])

    labels = df["Is_Night"].map({True: "Night", False: "Day"})
    labels = labels.fillna("Unknown")

    grouped = (
        df.assign(Period=labels)
        .groupby("Period", observed=True)
        .agg(Entries=("Severity Score", "size"), **{"Avg Severity": ("Severity Score", "mean")})
        .reset_index()
    )
    order = {"Day": 0, "Night": 1, "Unknown": 2}
    grouped["_order"] = grouped["Period"].map(order)
    return grouped.sort_values("_order").drop(columns="_order").reset_index(drop=True)
