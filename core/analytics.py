import numpy as np
import pandas as pd

# Ordered worst-to-least-severe. A human death is weighted per person
# killed (using the actual death-count fields) rather than as a flat
# indicator, so a two-person fatality outranks a one-person one instead
# of scoring identically to routine crop damage.
SEVERITY_WEIGHTS = {
    "presence": 0.5,
    "crop": 2.5,
    "grain": 1.5,
    "house": 5.0,
    "injury": 25.0,
}
DEATH_WEIGHT_PER_PERSON = 100.0

DEATH_COUNT_COLS = ["Male Death Count", "Female Death Count", "Children Death Count"]

# Category hierarchy for map coloring / the priority table, most severe first.
CONFLICT_CATEGORIES = ["Death", "Injury", "House", "Crop", "Presence"]


def _human_deaths(df):
    """
    Total people killed per row, gated by the Death flag.

    Counting straight from the death-count columns without that gate
    over-counts: this dataset has rows where Male/Female/Children Death
    Count is populated but Death itself is 0 -- in every case found here,
    a same-day, same-beat second report of a fatality already logged
    elsewhere (see the load-time data quality warning), not a distinct
    death. Gating on the flag avoids double-counting those.

    If Death is set but no per-person breakdown was given, this assumes
    at least 1 person rather than 0 -- the flag itself asserts a death
    happened even without a gender/age breakdown.
    """
    death_flag = pd.to_numeric(df.get("Death", 0), errors="coerce").fillna(0)
    is_death = death_flag > 0

    present = [c for c in DEATH_COUNT_COLS if c in df.columns]
    if not present:
        return is_death.astype(float)

    counts = df[present].sum(axis=1, min_count=1).fillna(0)
    result = counts.where(is_death, 0.0)
    result = result.mask(is_death & (result == 0), 1.0)
    return result


def compute_severity(df):
    deaths = _human_deaths(df)
    return (
        (df.get("Total Count", 0) > 0).astype(int) * SEVERITY_WEIGHTS["presence"]
        + (df.get("Crop Damage", 0) > 0).astype(int) * SEVERITY_WEIGHTS["crop"]
        + (df.get("Grain Damage", 0) > 0).astype(int) * SEVERITY_WEIGHTS["grain"]
        + (df.get("House Damage", 0) > 0).astype(int) * SEVERITY_WEIGHTS["house"]
        + (df.get("Injury", 0) > 0).astype(int) * SEVERITY_WEIGHTS["injury"]
        + deaths * DEATH_WEIGHT_PER_PERSON
    )


def classify_conflict(df):
    """
    One category per row -- the most severe type that applies, for map
    coloring and the priority table. A row with both crop damage and a
    death is labeled 'Death', not 'Crop'. Vectorized: evaluated once per
    category across the whole column, not once per row.
    """
    deaths = _human_deaths(df)
    injury = df.get("Injury", pd.Series(0, index=df.index)).fillna(0) > 0
    house = df.get("House Damage", pd.Series(0, index=df.index)).fillna(0) > 0
    crop = (df.get("Crop Damage", pd.Series(0, index=df.index)).fillna(0) > 0) | (
        df.get("Grain Damage", pd.Series(0, index=df.index)).fillna(0) > 0
    )

    conditions = [deaths > 0, injury, house, crop]
    choices = ["Death", "Injury", "House", "Crop"]
    return pd.Series(
        np.select(conditions, choices, default="Presence"),
        index=df.index,
        name="Conflict Category",
    )


def compute_kpis(df):
    deaths = _human_deaths(df)
    is_conflict = (
        (df.get("Crop Damage", 0) > 0)
        | (df.get("Grain Damage", 0) > 0)
        | (df.get("House Damage", 0) > 0)
        | (df.get("Injury", 0) > 0)
        | (deaths > 0)
    )
    return {
        "entries": len(df),
        "conflicts": int(is_conflict.sum()),
        "human_deaths": float(deaths.sum()),
        "human_death_incidents": int((deaths > 0).sum()),
        "severity": float(df["Severity Score"].sum()) if "Severity Score" in df else 0.0,
        "night_pct": float(df["Is_Night"].mean() * 100) if "Is_Night" in df.columns and len(df) else 0.0,
    }


def division_conflict_rate(df):
    """
    Sightings, conflict count, and conflict rate per division -- the
    normalized view, since raw sighting volume alone hides which
    divisions are actually more conflict-prone.
    """
    deaths = _human_deaths(df)
    is_conflict = (
        (df.get("Crop Damage", 0) > 0)
        | (df.get("Grain Damage", 0) > 0)
        | (df.get("House Damage", 0) > 0)
        | (df.get("Injury", 0) > 0)
        | (deaths > 0)
    )
    out = df.groupby("Division").size().rename("Sightings").to_frame()
    out["Conflict Events"] = is_conflict.groupby(df["Division"]).sum()
    out["Conflict Rate %"] = (out["Conflict Events"] / out["Sightings"] * 100).round(1)
    return out.sort_values("Conflict Rate %", ascending=False)


def monthly_trend(df):
    """Total sightings vs. conflict sightings by month, for the trend chart."""
    deaths = _human_deaths(df)
    is_conflict = (
        (df.get("Crop Damage", 0) > 0)
        | (df.get("Grain Damage", 0) > 0)
        | (df.get("House Damage", 0) > 0)
        | (df.get("Injury", 0) > 0)
        | (deaths > 0)
    )
    month = df["Date"].dt.to_period("M").astype(str)
    out = df.groupby(month).size().rename("Sightings").to_frame()
    out["Conflict Events"] = is_conflict.groupby(month).sum()
    return out.sort_index()


def hourly_conflict_profile(df):
    """Conflict-only sighting counts by hour of day (0-23), for the risk-window chart."""
    deaths = _human_deaths(df)
    is_conflict = (
        (df.get("Crop Damage", 0) > 0)
        | (df.get("Grain Damage", 0) > 0)
        | (df.get("House Damage", 0) > 0)
        | (df.get("Injury", 0) > 0)
        | (deaths > 0)
    )
    conflict_hours = df.loc[is_conflict, "Hour"].dropna()
    counts = conflict_hours.value_counts().reindex(range(24), fill_value=0).sort_index()
    counts.index.name = "Hour"
    return counts.rename("Conflict Events")
