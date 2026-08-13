"""Feature construction for the conflict models.

Two units of analysis, because they answer different questions:

``sighting_features`` scores a report that has already been filed -- use
it to triage which reports need a response team. ``village_month_panel``
scores a village before the month starts, from history alone -- use it to
decide where to place patrols and early warning.

Both are built so that no feature can see the outcome it predicts.
History windows are strictly earlier than the row they describe.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0088

CONFLICT_FLAGS = ["Crop Damage", "Grain Damage", "House Damage", "Injury", "Death"]

# Sighting-detail tokens that describe field evidence rather than the
# outcome. cropDamage/houseDamage/fenceDamage/anyOtherDamage are the
# target written into a second column, so they are excluded.
EVIDENCE_TOKENS = ["pugmark", "dung", "brokenbranches", "elephantsound"]

# Features that describe how a report came to exist rather than how
# dangerous the place is. Damage is usually discovered rather than
# witnessed, so these carry real signal and no forecasting value.
REPORT_ARTEFACTS = ["direct", "ev_pugmark", "ev_dung", "ev_brokenbranches",
                    "ev_elephantsound"]

EXPOSURE_RADIUS_KM = 2.0
CANDIDATE_RADIUS_KM = 5.0


def km_plane(lat: np.ndarray, lon: np.ndarray, ref_lat: float) -> np.ndarray:
    """Project to a local plane where one unit is one kilometre.

    Longitude degrees shrink with latitude; leaving them unscaled
    stretches east-west distances by 8% in this landscape.
    """
    x = np.radians(lon) * np.cos(np.radians(ref_lat)) * EARTH_RADIUS_KM
    y = np.radians(lat) * EARTH_RADIUS_KM
    return np.column_stack([x, y])


def conflict_target(df: pd.DataFrame) -> pd.Series:
    """Any recorded damage, injury or death.

    Casualties alone are unmodellable at this sample size -- the export
    behind this work holds five deaths and four injuries.
    """
    present = [c for c in CONFLICT_FLAGS if c in df.columns]
    return (df[present].fillna(0).sum(axis=1) > 0).astype(int)


def sighting_features(
    df: pd.DataFrame, centroids: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-report features. Returns (dataframe sorted by date, features)."""
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df["y"] = conflict_target(df)
    ref_lat = float(df["Latitude"].mean())

    counts = df.reindex(
        columns=["Male Count", "Female Count", "Calf Count", "Unknown Count"]
    ).fillna(0)
    total = df["Total Count"].fillna(0)
    f = pd.DataFrame(index=df.index)

    # Herd composition. Bulls range alone and raid; breeding herds with
    # calves avoid settlements.
    f["total_count"] = total
    f["male_count"] = counts["Male Count"]
    f["female_count"] = counts["Female Count"]
    f["calf_count"] = counts["Calf Count"]
    f["lone_male"] = ((total == 1) & (counts["Male Count"] == 1)).astype(int)
    f["all_male"] = ((counts["Male Count"] == total) & (total > 0)).astype(int)
    f["calf_present"] = (counts["Calf Count"] > 0).astype(int)
    f["male_fraction"] = np.where(
        total > 0, counts["Male Count"] / total.replace(0, np.nan), 0
    ).clip(0, 1)
    f["solitary"] = (total == 1).astype(int)
    f["log_group"] = np.log1p(total)

    hour = df["Hour"].fillna(12) if "Hour" in df.columns else pd.Series(12, index=df.index)
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["is_night"] = ((hour >= 18) | (hour < 6)).astype(int)
    f["month"] = df["Date"].dt.month

    f["direct"] = (df["Sighting Type"].astype(str).str.lower() == "direct").astype(int)
    detail = df.get("Sighting Type Detail", pd.Series("", index=df.index))
    detail = detail.fillna("").astype(str).str.lower()
    for token in EVIDENCE_TOKENS:
        f[f"ev_{token}"] = detail.str.contains(token).astype(int)

    points = km_plane(df["Latitude"].values, df["Longitude"].values, ref_lat)
    village_points = km_plane(
        centroids["Latitude"].values, centroids["Longitude"].values, ref_lat
    )
    village_tree = cKDTree(village_points)
    f["dist_village_km"] = village_tree.query(points, k=1)[0]
    f["villages_2km"] = [len(i) for i in village_tree.query_ball_point(points, r=2.0)]
    f["villages_5km"] = [len(i) for i in village_tree.query_ball_point(points, r=5.0)]

    f = f.join(_history(points, df))
    return df, f


def _history(points: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """Local conflict history, from records strictly earlier than each row."""
    neighbours = cKDTree(points).query_ball_point(points, r=EXPOSURE_RADIUS_KM)
    day = df["Date"].values.astype("datetime64[D]").astype(int)
    y = df["y"].values

    conflicts_90, sightings_90, conflicts_all, days_since = [], [], [], []
    for i, idx in enumerate(neighbours):
        past = np.array([k for k in idx if day[k] < day[i]], dtype=int)
        if past.size == 0:
            conflicts_90.append(0)
            sightings_90.append(0)
            conflicts_all.append(0)
            days_since.append(365)
            continue
        recent = past[day[past] >= day[i] - 90]
        sightings_90.append(len(recent))
        conflicts_90.append(int(y[recent].sum()))
        hits = past[y[past] == 1]
        conflicts_all.append(len(hits))
        days_since.append(int(day[i] - day[hits].max()) if hits.size else 365)

    conflicts_90 = np.array(conflicts_90)
    sightings_90 = np.array(sightings_90)
    return pd.DataFrame({
        "prior_conflicts_2km_90d": conflicts_90,
        "prior_sightings_2km_90d": sightings_90,
        "prior_conflicts_2km_all": conflicts_all,
        "days_since_conflict_2km": days_since,
        # Shrunk toward the landscape rate: one conflict in one report is
        # not a 100% rate.
        "prior_conflict_rate_2km": (conflicts_90 + 1.7) / (sightings_90 + 10.0),
    }, index=df.index)


def village_month_panel(
    df: pd.DataFrame, centroids: pd.DataFrame,
    exposure_km: float = EXPOSURE_RADIUS_KM,
    candidate_km: float = CANDIDATE_RADIUS_KM,
) -> pd.DataFrame:
    """One row per village per month, scored from the preceding months.

    The target is whether any conflict is recorded within ``exposure_km``
    of the village during that month. Every feature is computed from
    months strictly before it.
    """
    df = df.copy()
    df["conflict"] = conflict_target(df)
    df["period"] = df["Date"].dt.to_period("M")
    ref_lat = float(df["Latitude"].mean())

    sighting_points = km_plane(df["Latitude"].values, df["Longitude"].values, ref_lat)
    all_village_points = km_plane(
        centroids["Latitude"].values, centroids["Longitude"].values, ref_lat
    )
    sighting_tree = cKDTree(sighting_points)
    all_tree = cKDTree(all_village_points)

    # Only villages the elephants actually came near are candidates.
    reachable = all_tree.query_ball_tree(sighting_tree, r=candidate_km)
    keep = np.array([i for i, hits in enumerate(reachable) if hits], dtype=int)
    if keep.size == 0:
        return pd.DataFrame()
    villages = centroids.iloc[keep].reset_index(drop=True)
    village_points = all_village_points[keep]

    # Settlement density: a stand-in for the forest interface until real
    # landcover is available. Sparse neighbours imply a forest matrix.
    density_2km = np.array([len(i) - 1 for i in all_tree.query_ball_point(village_points, r=2.0)])
    density_5km = np.array([len(i) - 1 for i in all_tree.query_ball_point(village_points, r=5.0)])
    nearest_neighbour = all_tree.query(village_points, k=2)[0][:, 1]

    members = cKDTree(village_points).query_ball_tree(sighting_tree, r=exposure_km)
    periods = sorted(df["period"].unique())
    period_index = df["period"].map({p: i for i, p in enumerate(periods)}).values

    male = df["Male Count"].fillna(0).values
    total = df["Total Count"].fillna(0).values
    calf = df["Calf Count"].fillna(0).values
    hour = (df["Hour"].fillna(12) if "Hour" in df.columns
            else pd.Series(12, index=df.index)).values
    conflict = df["conflict"].values

    rows: List[dict] = []
    for month in range(1, len(periods)):
        for v in range(len(villages)):
            idx = np.array(members[v], dtype=int)
            if idx.size == 0:
                continue
            past = idx[period_index[idx] < month]
            current = idx[period_index[idx] == month]
            last_month = past[period_index[past] >= month - 1]
            last_quarter = past[period_index[past] >= month - 3]
            hits = past[conflict[past] == 1]

            rows.append({
                "village": villages["Village"].iloc[v],
                "period": str(periods[month]),
                "month_index": month,
                "month_of_year": periods[month].month,
                "y": int(conflict[current].sum() > 0),
                "conf_prev_month": int(conflict[last_month].sum()),
                "conf_prev_quarter": int(conflict[last_quarter].sum()),
                "conf_all": int(conflict[past].sum()),
                "sight_prev_month": len(last_month),
                "sight_prev_quarter": len(last_quarter),
                "sight_all": len(past),
                "ever_conflict": int(hits.size > 0),
                "months_since_conflict": (
                    month - period_index[hits].max() if hits.size else 99
                ),
                "male_share_hist": _mean(male[past] / np.maximum(total[past], 1)),
                "lone_male_share_hist": _mean((total[past] == 1) & (male[past] == 1)),
                "calf_share_hist": _mean(calf[past] > 0),
                "mean_group_hist": _mean(total[past]),
                "night_share_hist": _mean((hour[past] >= 18) | (hour[past] < 6)),
                "villages_2km": density_2km[v],
                "villages_5km": density_5km[v],
                "nearest_village_km": nearest_neighbour[v],
            })
    return pd.DataFrame(rows)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def panel_feature_columns(panel: pd.DataFrame) -> List[str]:
    return [c for c in panel.columns
            if c not in ("village", "period", "y", "month_index", "division")]
