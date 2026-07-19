import pandas as pd

# How much each factor contributes to a beat's ranking: mostly the
# accumulated severity of what's happened there, with a smaller boost
# for how night-heavy and how village-adjacent that activity is.
RISK_WEIGHTS = {"severity": 0.5, "night": 0.3, "near_village": 0.2}


def beat_risk_scores(df):
    """
    Aggregate risk score per beat, ranked worst first -- the list a
    manager can act on directly ("send the team here first").

    Safe by construction: 'Is_Night' and 'Near Village' are only used
    if actually present, so a missing centroids.csv (no village-distance
    data) degrades that term to zero instead of crashing.
    """
    if df.empty or "Beat" not in df.columns:
        return pd.DataFrame(columns=["Beat", "Sightings", "Risk Score"])

    grp = df.groupby("Beat")
    beats = grp.size().rename("Sightings")

    if "Severity Score" in df.columns:
        severity_sum = grp["Severity Score"].sum()
    else:
        severity_sum = pd.Series(0.0, index=beats.index)

    if "Is_Night" in df.columns:
        night_pct = grp["Is_Night"].mean() * 100
    else:
        night_pct = pd.Series(0.0, index=beats.index)

    if "Near Village" in df.columns:
        village_pct = grp["Near Village"].mean() * 100
    else:
        village_pct = pd.Series(0.0, index=beats.index)

    risk_score = (
        severity_sum * RISK_WEIGHTS["severity"]
        + night_pct * RISK_WEIGHTS["night"]
        + village_pct * RISK_WEIGHTS["near_village"]
    )

    out = pd.DataFrame({
        "Beat": beats.index,
        "Sightings": beats.values,
        "Risk Score": risk_score.reindex(beats.index).values,
    })
    return out.sort_values("Risk Score", ascending=False).reset_index(drop=True)
