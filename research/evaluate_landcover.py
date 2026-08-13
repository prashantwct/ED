"""Does real land cover improve the conflict forecast?

Answers three separate questions, because they have different answers:

1. Do the land-cover variables discriminate conflict villages at all?
2. Do they improve the month-ahead forecast?
3. Do they help where history cannot -- villages with no prior conflict?

Validation is rolling-origin: train on every month before the test
month, step forward, pool the out-of-fold predictions. A single split
on seven months of data lands on whichever two months happen to be
held out, and Jun-Jul is the low season for crop damage.

    python -m research.evaluate_landcover path/to/sightings.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from core.data_loader import load_and_validate_csv
from core.spatial import load_village_centroids
from research.evaluate import gradient_boosting, recall_at_capacity
from research.features import conflict_target, panel_feature_columns, village_month_panel
from research.landcover import CACHE_DIR, STANDING_CROP, LandCover, cropping_features, landcover_table

# Five variables chosen for mechanism rather than by search: the raiding
# surface, how far cover is, how much of it there is, how shredded it is,
# and water.
INTERFACE_BLOCK = ["crop_edge_frac_2km", "dist_forest_km", "tree_frac_2km",
                   "forest_edge_density_2km", "dist_water_km"]
CROPPING_BLOCK = ["standing_crop_index", "is_kharif", "is_rabi",
                  "exposed_cropland", "exposed_crop_edge"]
MIN_TEST_EVENTS = 3


def village_landcover(panel: pd.DataFrame, centroids: pd.DataFrame,
                      cache: Path = CACHE_DIR / "village_landcover.csv") -> pd.DataFrame:
    """Land-cover statistics per village, cached between runs."""
    names = sorted(panel["village"].unique())
    if cache.exists():
        table = pd.read_csv(cache, index_col=0)
        if set(names).issubset(table.index):
            return table.loc[names]

    located = centroids.drop_duplicates("Village").set_index("Village").loc[names]
    print(f"extracting land cover for {len(located)} villages ...")
    started = time.time()
    cover = LandCover.for_points(located["Latitude"].values, located["Longitude"].values)
    _ = cover.crop_forest_interface
    table = landcover_table(located, cover)
    table.index = located.index
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache)
    print(f"  done in {time.time() - started:.0f}s")
    return table


def attach(panel: pd.DataFrame, landcover: pd.DataFrame) -> pd.DataFrame:
    joined = panel.join(landcover, on="village")
    return pd.concat([joined, cropping_features(
        joined["month_of_year"], joined["crop_frac_2km"], joined["crop_edge_frac_2km"]
    )], axis=1)


# ---------------------------------------------------------------------------
# 1. Discrimination
# ---------------------------------------------------------------------------
def describe_hypotheses(panel: pd.DataFrame, landcover: pd.DataFrame,
                        df: pd.DataFrame) -> None:
    village = panel.groupby("village").agg(
        events=("y", "sum"), settlements_5km=("villages_5km", "first")
    ).join(landcover)
    village["any_conflict"] = village["events"] > 0
    hit = village[village["any_conflict"]]
    quiet = village[~village["any_conflict"]]

    print(f"\n{'=' * 74}\nDO THE LAND-COVER VARIABLES SEPARATE CONFLICT VILLAGES?")
    print(f"{len(hit)} villages with conflict, {len(quiet)} without\n")
    print(f"{'metric':<26}{'conflict':>10}{'quiet':>10}{'ratio':>8}{'p':>10}")
    for column in ["crop_edge_frac_2km", "crop_frac_2km", "built_frac_2km",
                   "tree_frac_2km", "tree_frac_5km", "forest_edge_density_2km",
                   "dist_forest_km", "dist_water_km"]:
        test = stats.mannwhitneyu(hit[column], quiet[column])
        a, b = hit[column].mean(), quiet[column].mean()
        mark = ("***" if test.pvalue < 0.001 else "**" if test.pvalue < 0.01
                else "*" if test.pvalue < 0.05 else "")
        print(f"{column:<26}{a:>10.4f}{b:>10.4f}{a / b if b else np.nan:>8.2f}"
              f"{test.pvalue:>10.4f} {mark}")

    for column, label in [("tree_frac_2km", "tree cover within 2 km"),
                          ("crop_edge_frac_2km", "crop-forest interface within 2 km")]:
        print(f"\n  share of villages ever in conflict, by {label}:")
        bands = pd.qcut(village[column], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
        grouped = village.groupby(bands, observed=True).agg(
            villages=("events", "size"), with_conflict=("any_conflict", "mean"),
            mean_value=(column, "mean"))
        for band, row in grouped.iterrows():
            bar = "#" * int(row["with_conflict"] * 40)
            print(f"    {band:<8} {row['mean_value']:>7.3f}  "
                  f"{row['with_conflict']:>5.1%}  {bar}")

    _crop_calendar(df)


def _crop_calendar(df: pd.DataFrame) -> None:
    """Does the assumed cropping calendar track observed crop damage?"""
    frame = df.copy()
    frame["crop_damage"] = (
        frame[["Crop Damage", "Grain Damage"]].fillna(0).sum(axis=1) > 0
    ).astype(int)
    monthly = frame.groupby(frame["Date"].dt.month).agg(
        reports=("crop_damage", "size"), crop_damage_rate=("crop_damage", "mean"))
    monthly["standing_crop_index"] = [STANDING_CROP[m] for m in monthly.index]

    print(f"\n  cropping calendar against observed crop damage:")
    print(f"    {'month':<7}{'reports':>9}{'damage rate':>13}{'calendar':>10}")
    for month, row in monthly.iterrows():
        print(f"    {month:<7}{int(row['reports']):>9}{row['crop_damage_rate']:>13.3f}"
              f"{row['standing_crop_index']:>10.2f}")
    result = stats.spearmanr(monthly["standing_crop_index"], monthly["crop_damage_rate"])
    print(f"    Spearman rho {result.statistic:+.2f}, p={result.pvalue:.3f} "
          f"on {len(monthly)} months -- directional, underpowered")


# ---------------------------------------------------------------------------
# 2 and 3. Forecast value
# ---------------------------------------------------------------------------
def rolling_origin(frame: pd.DataFrame, feature_sets: Dict[str, List[str]]
                   ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Train on every earlier month, test on this one, pool the results."""
    out_of_fold = {name: [] for name in feature_sets}
    truth, prior_conflict = [], []

    print(f"\n{'test month':<12}{'n':>5}{'events':>8}{'base':>7}", end="")
    for name in feature_sets:
        print(f"{name[:20]:>22}", end="")
    print()

    for month in sorted(frame["month_index"].unique()):
        train = frame["month_index"] < month
        test = frame["month_index"] == month
        y = frame.loc[test, "y"].values
        if month < 2 or y.sum() < MIN_TEST_EVENTS:
            continue
        truth.append(y)
        prior_conflict.append(frame.loc[test, "conf_all"].values)
        print(f"{frame.loc[test, 'period'].iloc[0]:<12}{test.sum():>5}"
              f"{y.sum():>8}{y.mean():>7.3f}", end="")
        for name, columns in feature_sets.items():
            model = gradient_boosting().fit(frame.loc[train, columns].values,
                                            frame.loc[train, "y"].values)
            predicted = model.predict_proba(frame.loc[test, columns].values)[:, 1]
            out_of_fold[name].append(predicted)
            print(f"{average_precision_score(y, predicted):>22.3f}", end="")
        print()

    return (np.concatenate(truth),
            {k: np.concatenate(v) for k, v in out_of_fold.items()},
            np.concatenate(prior_conflict))


def summarise(y: np.ndarray, predictions: Dict[str, np.ndarray], title: str) -> None:
    print(f"\n  {title} -- {len(y)} rows, {y.sum()} events, "
          f"{y.mean():.1%} base rate")
    print(f"  {'feature set':<32}{'ROC-AUC':>9}{'PR-AUC':>9}{'recall@20%':>12}")
    for name, p in predictions.items():
        print(f"  {name:<32}{roc_auc_score(y, p):>9.3f}"
              f"{average_precision_score(y, p):>9.3f}"
              f"{recall_at_capacity(y, p, 0.20):>12.3f}")
    print(f"  {'base rate':<32}{0.5:>9.3f}{y.mean():>9.3f}{0.2:>12.3f}")


def paired_delta(y: np.ndarray, before: np.ndarray, after: np.ndarray,
                 label: str) -> None:
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        deltas.append(average_precision_score(y[idx], after[idx])
                      - average_precision_score(y[idx], before[idx]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    print(f"  {label}: PR-AUC {np.mean(deltas):+.3f} 95% CI [{low:+.3f}, {high:+.3f}], "
          f"positive in {np.mean(np.array(deltas) > 0):.1%} of resamples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--centroids", default=None)
    args = parser.parse_args()

    df, _ = load_and_validate_csv(args.csv)
    centroids, _ = load_village_centroids(args.centroids)
    panel = village_month_panel(df, centroids)
    landcover = village_landcover(panel, centroids)
    frame = attach(panel, landcover)

    describe_hypotheses(panel, landcover, df)

    base = panel_feature_columns(panel)
    feature_sets = {
        "history only": base,
        "+ interface block (5)": base + INTERFACE_BLOCK,
        "+ all land cover (24)": base + list(landcover.columns),
        "+ land cover + cropping (29)": base + list(landcover.columns) + CROPPING_BLOCK,
    }

    print(f"\n{'=' * 74}\nDOES LAND COVER IMPROVE THE FORECAST?")
    y, predictions, prior = rolling_origin(frame, feature_sets)
    summarise(y, predictions, "pooled out-of-fold")
    paired_delta(y, predictions["history only"],
                 predictions["+ interface block (5)"], "interface block")

    cold = prior == 0
    print(f"\n{'=' * 74}\nCOLD START -- villages with no prior recorded conflict")
    summarise(y[cold], {k: v[cold] for k, v in predictions.items()},
              "no history to lean on")
    paired_delta(y[cold], predictions["history only"][cold],
                 predictions["+ interface block (5)"][cold], "interface block")
    summarise(y[~cold], {k: v[~cold] for k, v in predictions.items()},
              "history already exists")


if __name__ == "__main__":
    main()
