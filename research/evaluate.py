"""Train and score the conflict models against honest baselines.

Run it:

    python -m research.evaluate path/to/sightings.csv

Validation is by time, never by random split: rows near each other in
space and time are not independent, so a shuffled split lets the model
see the neighbours of what it is being tested on and reports a number
the field will not reproduce.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from core.data_loader import load_and_validate_csv
from core.spatial import load_village_centroids
from research.features import (
    REPORT_ARTEFACTS,
    panel_feature_columns,
    sighting_features,
    village_month_panel,
)

BOOTSTRAP_SAMPLES = 2000
RESPONSE_CAPACITY = 0.20


def recall_at_capacity(y: np.ndarray, p: np.ndarray,
                       capacity: float = RESPONSE_CAPACITY) -> float:
    """Share of conflicts caught when you can only act on the top slice.

    The metric that matters operationally: a range officer cannot visit
    every village, so what counts is how much of the conflict lands in
    the fraction they can actually cover.
    """
    n = max(int(len(y) * capacity), 1)
    return float(y[np.argsort(-p)[:n]].sum() / max(y.sum(), 1))


def score(name: str, y: np.ndarray, p: np.ndarray) -> Dict[str, object]:
    return {
        "model": name,
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "recall@20%": recall_at_capacity(y, p, 0.20),
        "recall@10%": recall_at_capacity(y, p, 0.10),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, stat: Callable,
                 seed: int = 0) -> tuple:
    """Percentile interval over the test set, which is the small part."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        values.append(stat(y[idx], p[idx]))
    return tuple(np.percentile(values, [2.5, 97.5]))


def gradient_boosting(seed: int = 0) -> HistGradientBoostingClassifier:
    """Shallow and heavily regularised: 1,761 rows is not much data."""
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=250, learning_rate=0.06,
        min_samples_leaf=30, l2_regularization=1.0, random_state=seed,
    )


def evaluate_sightings(df: pd.DataFrame, centroids: pd.DataFrame) -> None:
    df, features = sighting_features(df, centroids)
    y = df["y"].values
    month = df["Date"].dt.month.values
    train, test = month <= 5, month >= 6

    print(f"\n{'=' * 72}\nREPORT TRIAGE -- does this sighting involve damage?")
    print(f"train {train.sum()} reports ({y[train].mean():.1%} conflict) | "
          f"test {test.sum()} ({y[test].mean():.1%})")

    rows = [score("base rate", y[test], np.full(test.sum(), y[train].mean())),
            score("local conflict history only", y[test],
                  features["prior_conflict_rate_2km"].values[test])]

    for label, columns in [
        ("gradient boosting", list(features.columns)),
        ("  ... artefacts removed", [c for c in features.columns
                                     if c not in REPORT_ARTEFACTS]),
    ]:
        X = features[columns].values.astype(float)
        model = gradient_boosting().fit(X[train], y[train])
        rows.append(score(label, y[test], model.predict_proba(X[test])[:, 1]))

    print(pd.DataFrame(rows).set_index("model").round(3).to_string())
    print("\nThe artefact row is the honest one. 'direct' and the evidence\n"
          "tokens describe how the report was generated -- damage is found,\n"
          "not witnessed -- so they cannot be known ahead of an incident.")


def evaluate_villages(df: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    panel = village_month_panel(df, centroids)
    if panel.empty:
        print("No village-month panel could be built.")
        return panel

    features = panel_feature_columns(panel)
    horizon = panel["month_index"].max()
    train = panel["month_index"] <= horizon - 2
    test = panel["month_index"] > horizon - 2

    X_train = panel.loc[train, features].values
    X_test = panel.loc[test, features].values
    y_train = panel.loc[train, "y"].values
    y_test = panel.loc[test, "y"].values

    print(f"\n{'=' * 72}\nVILLAGE FORECAST -- conflict near this village next month?")
    print(f"{len(panel)} village-months over {panel['village'].nunique()} villages")
    print(f"train {train.sum()} ({y_train.mean():.1%}) | "
          f"test {test.sum()} ({y_test.mean():.1%}, {y_test.sum()} events)")

    rows = [score("base rate", y_test, np.full(test.sum(), y_train.mean()))]
    for label, column in [("rule: ever had conflict", "ever_conflict"),
                          ("rule: conflicts last quarter", "conf_prev_quarter"),
                          ("rule: sightings last month", "sight_prev_month")]:
        rows.append(score(label, y_test, panel.loc[test, column].values.astype(float)))

    model = gradient_boosting().fit(X_train, y_train)
    predicted = model.predict_proba(X_test)[:, 1]
    rows.append(score("gradient boosting", y_test, predicted))
    print(pd.DataFrame(rows).set_index("model").round(3).to_string())

    print("\nUncertainty on the held-out months:")
    for name, stat in [("ROC-AUC", roc_auc_score),
                       ("PR-AUC", average_precision_score),
                       ("recall@20%", lambda a, b: recall_at_capacity(a, b, 0.20))]:
        low, high = bootstrap_ci(y_test, predicted, stat)
        print(f"  {name:<12} {stat(y_test, predicted):.3f}  95% CI [{low:.3f}, {high:.3f}]")

    _compare_to_rule(y_test, predicted,
                     panel.loc[test, "conf_prev_quarter"].values.astype(float))
    _importance(model, X_test, y_test, features)
    return panel


def _compare_to_rule(y: np.ndarray, model_p: np.ndarray, rule_p: np.ndarray) -> None:
    """Does the model earn its complexity over the obvious heuristic?"""
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        deltas.append(average_precision_score(y[idx], model_p[idx])
                      - average_precision_score(y[idx], rule_p[idx]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    print(f"\n  vs 'conflicts last quarter', PR-AUC {np.mean(deltas):+.3f} "
          f"95% CI [{low:+.3f}, {high:+.3f}]"
          f"; ahead in {np.mean(np.array(deltas) > 0):.1%} of resamples")


def _importance(model, X: np.ndarray, y: np.ndarray, names: List[str]) -> None:
    result = permutation_importance(model, X, y, n_repeats=20, random_state=0,
                                    scoring="average_precision")
    print("\n  What the forecast leans on:")
    for i in np.argsort(-result.importances_mean)[:8]:
        print(f"    {names[i]:<24} {result.importances_mean[i]:+.4f} "
              f"± {result.importances_std[i]:.4f}")


def spatial_transfer(panel: pd.DataFrame, df: pd.DataFrame,
                     centroids: pd.DataFrame) -> None:
    """Hold out whole divisions: can this be pointed at a new landscape?"""
    from scipy.spatial import cKDTree

    from research.features import km_plane

    ref_lat = float(df["Latitude"].mean())
    sighting_points = km_plane(df["Latitude"].values, df["Longitude"].values, ref_lat)
    names = panel["village"].unique()
    located = centroids.set_index("Village").loc[names]
    village_points = km_plane(located["Latitude"].values,
                              located["Longitude"].values, ref_lat)
    nearest = cKDTree(sighting_points).query(village_points, k=1)[1]
    panel = panel.assign(
        division=panel["village"].map(dict(zip(names, df["Division"].values[nearest])))
    )

    features = panel_feature_columns(panel)
    print(f"\n{'=' * 72}\nTRANSFER TO AN UNSEEN LANDSCAPE")
    for division in panel["division"].value_counts().index[:3]:
        test = panel["division"] == division
        if panel.loc[test, "y"].sum() < 5:
            continue
        model = gradient_boosting().fit(
            panel.loc[~test, features].values, panel.loc[~test, "y"].values
        )
        predicted = model.predict_proba(panel.loc[test, features].values)[:, 1]
        actual = panel.loc[test, "y"].values
        print(f"  hold out {division:<16} n={test.sum():<5} events={actual.sum():<4} "
              f"ROC-AUC {roc_auc_score(actual, predicted):.3f}  "
              f"PR-AUC {average_precision_score(actual, predicted):.3f} "
              f"(base {actual.mean():.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Gajrakshak sightings export")
    parser.add_argument("--centroids", default=None, help="Village centroids CSV")
    args = parser.parse_args()

    df, warnings = load_and_validate_csv(args.csv)
    centroids, _ = load_village_centroids(args.centroids)
    print(f"{len(df)} reports, {df['Date'].min():%b %Y} to {df['Date'].max():%b %Y}")
    for warning in warnings[:5]:
        print(f"  data quality: {warning}")

    evaluate_sightings(df, centroids)
    panel = evaluate_villages(df, centroids)
    if not panel.empty:
        spatial_transfer(panel, df, centroids)


if __name__ == "__main__":
    main()
