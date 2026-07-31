"""Beat-level risk scoring.

Combines three signals into a single 0-100 "Risk Score" per Beat so
field staff can quickly see where to prioritise patrols or mitigation:

* Total severity of incidents in the beat (50% weight)
* Share of incidents occurring at night (30% weight)
* Share of incidents near a village, when that data is available (20%)

The previous version of this file had broken indentation (module-level
code outside any function) and was never imported by app.py. It is now
a proper function with normalised, comparable scores across beats.
"""

from __future__ import annotations

import pandas as pd

SEVERITY_WEIGHT = 0.5
NIGHT_WEIGHT = 0.3
VILLAGE_WEIGHT = 0.2


def beat_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a ranked risk score per Beat.

    Args:
        df: Dataframe with at least ``Beat`` and ``Severity Score``
            columns. ``Is_Night`` (nullable boolean) and ``Near Village``
            (boolean) are used when present; their contribution is
            reweighted onto the remaining signals when absent, so a
            missing signal never silently zeroes out part of the score.

    Returns:
        DataFrame sorted by ``Risk Score`` descending, with columns
        ``Beat``, ``Entries``, ``Total Severity``, ``Night %``,
        ``Near Village %``, and ``Risk Score`` (0-100). Empty input
        yields an empty DataFrame with the same columns.
    """
    columns = ["Beat", "Entries", "Total Severity", "Night %", "Near Village %", "Risk Score"]
    if df.empty or "Beat" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    has_night = "Is_Night" in working.columns and working["Is_Night"].notna().any()
    has_village = "Near Village" in working.columns

    if has_night:
        working["_night_flag"] = working["Is_Night"].astype("boolean").fillna(False).astype(int)
    if has_village:
        working["_village_flag"] = working["Near Village"].astype(bool).astype(int)

    agg_spec = {
        "Entries": ("Severity Score", "size"),
        "Total Severity": ("Severity Score", "sum"),
    }
    grouped = working.groupby("Beat", observed=True).agg(**agg_spec)

    grouped["Night %"] = (
        working.groupby("Beat", observed=True)["_night_flag"].mean() * 100
        if has_night
        else float("nan")
    )
    grouped["Near Village %"] = (
        working.groupby("Beat", observed=True)["_village_flag"].mean() * 100
        if has_village
        else float("nan")
    )

    grouped = grouped.reset_index()

    # Normalise total severity to 0-100 across beats so it is comparable
    # in scale to the two percentage-based signals.
    max_severity = grouped["Total Severity"].max()
    if max_severity > 0:
        severity_norm = grouped["Total Severity"] / max_severity * 100
    else:
        severity_norm = pd.Series(0.0, index=grouped.index)

    # If a signal is unavailable for the whole dataset, redistribute its
    # weight proportionally across the signals that ARE available,
    # rather than silently treating the missing signal as zero risk.
    available_weights = {"severity": SEVERITY_WEIGHT}
    if has_night:
        available_weights["night"] = NIGHT_WEIGHT
    if has_village:
        available_weights["village"] = VILLAGE_WEIGHT
    weight_total = sum(available_weights.values())

    risk = severity_norm * (available_weights["severity"] / weight_total)
    if has_night:
        risk = risk + grouped["Night %"] * (available_weights["night"] / weight_total)
    if has_village:
        risk = risk + grouped["Near Village %"] * (available_weights["village"] / weight_total)

    grouped["Risk Score"] = risk.round(1)

    return grouped[columns].sort_values("Risk Score", ascending=False).reset_index(drop=True)
