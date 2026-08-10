"""Conservation intelligence: beat priorities, timing, village exposure.

Answers the questions a range officer acts on -- where to post the team,
when to run the shift, what is getting worse, and how much of it to
trust.

Three rules run through it:

* Tier decides, score only orders. Tiers come from fixed rules, so they
  mean the same thing across periods and filters. The continuous score
  only orders beats within a tier, because it is normalised against
  whatever is currently on screen.
* Raw rates from thin data are not evidence. One conflict in one report
  is a 100% rate, so rates are shrunk toward the landscape rate and every
  beat carries a confidence label.
* Counts measure patrol effort as much as elephants. Rates sit alongside
  volumes rather than replacing them.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.config import (
    CASUALTY_POINTS_PER_DEATH,
    CASUALTY_POINTS_PER_INJURY,
    CONFIDENCE_THRESHOLDS,
    CRITICAL_CASUALTY_WINDOW_DAYS,
    DEFAULT_COVERAGE_TARGET,
    DEFAULT_RECENT_DAYS,
    ESCALATION_RATIO,
    EVENTS_FOR_ESCALATION_HIGH,
    FALLBACK_PRIOR_STRENGTH,
    HISTORICAL_CASUALTY_DISCOUNT,
    HOUSE_EVENTS_FOR_HIGH,
    INJURIES_FOR_CRITICAL,
    MAX_PEAK_MONTHS,
    MIN_EVENTS_FOR_TREND,
    MIN_WINDOW_DAYS,
    NIGHT_SHARE_FOR_PATROL_SHIFT,
    PRIOR_STRENGTH_BOUNDS,
    RATE_MULTIPLE_FOR_HIGH,
    SCORE_WEIGHTS,
    SEASONAL_LIFT,
    VILLAGE_SHARE_FOR_EARLY_WARNING,
)
from core.analytics import (
    classify_conflict,
    compute_kpis,
    conflict_mask,
    human_deaths,
    human_injuries,
    hourly_conflict_profile,
    property_severity,
)

logger = logging.getLogger(__name__)

# --- Labels ---------------------------------------------------------------
TIER_CRITICAL = "Critical"
TIER_HIGH = "High"
TIER_WATCH = "Watch"
TIER_ROUTINE = "Routine"
TIER_ORDER = [TIER_CRITICAL, TIER_HIGH, TIER_WATCH, TIER_ROUTINE]

# --- Trend labels ----------------------------------------------------------
TREND_ESCALATING = "Escalating"
TREND_STABLE = "Stable"
TREND_EASING = "Easing"
TREND_UNKNOWN = "Insufficient data"

# --- Confidence labels -----------------------------------------------------
CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_LOW = "Low"


EASING_RATIO = 1 / ESCALATION_RATIO


BEAT_KEY_COLUMNS = ["Division", "Range", "Beat"]


# ---------------------------------------------------------------------------
# Rate shrinkage
# ---------------------------------------------------------------------------
def shrink_rates(
    successes: Sequence[float],
    trials: Sequence[float],
    prior_strength: Optional[float] = None,
) -> Dict[str, object]:
    """Shrink per-group rates toward the pooled rate (empirical Bayes).

    Otherwise one conflict in one report reads as a 100% rate and
    outranks 90 conflicts in 200 reports::

        adjusted = (conflicts + k * landscape_rate) / (reports + k)

    ``k`` is the prior strength in reports: a group needs roughly ``k``
    of its own before its observed rate outweighs the landscape rate. It
    is fitted by method of moments rather than picked.

    Args:
        successes: Conflict events per group.
        trials: Total reports per group.
        prior_strength: Override for ``k``. Fitted from data if None.

    Returns:
        Dict with ``adjusted`` (rates in 0-1), ``prior_mean`` and
        ``prior_strength``.
    """
    x = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)

    total_trials = float(n.sum())
    if total_trials <= 0 or len(n) == 0:
        return {
            "adjusted": np.zeros_like(x),
            "prior_mean": 0.0,
            "prior_strength": float(prior_strength or FALLBACK_PRIOR_STRENGTH),
        }

    prior_mean = float(x.sum() / total_trials)

    if prior_strength is None:
        prior_strength = _fit_prior_strength(x, n, prior_mean)

    adjusted = (x + prior_strength * prior_mean) / (n + prior_strength)
    return {
        "adjusted": adjusted,
        "prior_mean": prior_mean,
        "prior_strength": float(prior_strength),
    }


def _fit_prior_strength(x: np.ndarray, n: np.ndarray, prior_mean: float) -> float:
    """Method-of-moments estimate of the Beta-Binomial prior strength.

    Excess variation over binomial sampling noise means the groups really
    differ, warranting a weak prior; no excess warrants a strong one.
    """
    usable = n > 0
    if usable.sum() < 2 or prior_mean <= 0 or prior_mean >= 1:
        return FALLBACK_PRIOR_STRENGTH

    x_u, n_u = x[usable], n[usable]
    rates = x_u / n_u
    weights = n_u / n_u.sum()

    observed_var = float(np.sum(weights * (rates - prior_mean) ** 2))
    # Variance expected from binomial sampling alone at the pooled rate.
    binomial_var = float(
        prior_mean * (1 - prior_mean) * (len(n_u) - 1) / n_u.sum()
    )

    excess = observed_var - binomial_var
    if excess <= 0:
        # Groups look homogeneous: shrink hard toward the landscape rate.
        return PRIOR_STRENGTH_BOUNDS[1]

    k = prior_mean * (1 - prior_mean) / excess - 1
    if not np.isfinite(k):
        return FALLBACK_PRIOR_STRENGTH
    return float(np.clip(k, *PRIOR_STRENGTH_BOUNDS))


# ---------------------------------------------------------------------------
# Beat-level intelligence
# ---------------------------------------------------------------------------
def beat_intelligence(
    df: pd.DataFrame,
    recent_days: int = DEFAULT_RECENT_DAYS,
    critical_window_days: int = CRITICAL_CASUALTY_WINDOW_DAYS,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Build the per-beat priority table that drives deployment decisions.

    Args:
        df: Filtered, enriched dataframe. Expects at minimum ``Beat``;
            uses ``Date``, ``Is_Night``, ``Near Village``, the damage and
            casualty columns, and ``Severity Score`` when present.
        recent_days: Length of the recent window used for escalation
            detection. The window immediately before it is the baseline.
        critical_window_days: How recent a casualty must be to put a beat
            in the Critical tier. See
            :data:`CRITICAL_CASUALTY_WINDOW_DAYS`.
        as_of: End of the review period, which the Critical window is
            measured back from. Pass the period end (not the filtered
            frame's own maximum) so that narrowing to a beat or division
            cannot move the goalposts and change a tier. Defaults to the
            latest date in ``df``.

    Returns:
        One row per beat, sorted by tier then priority score, with the
        evidence behind the ranking kept alongside the ranking itself so
        a manager can see *why* a beat is where it is:

        ``Beat``, ``Division``, ``Range``, ``Reports``,
        ``Conflict Events``, ``Conflict Rate %``, ``Adj. Conflict Rate %``,
        ``Human Deaths``, ``People Injured``, ``Recent Deaths``,
        ``Recent Injuries``, ``House Damage Events``,
        ``Crop Damage Events``, ``Damage Burden``, ``Night Conflict %``,
        ``Near Village %``, ``Trend``, ``Recent vs Prior``,
        ``Priority Score``, ``Priority Tier``, ``Confidence``,
        ``Recommended Action``.

        ``Human Deaths`` counts the whole filtered period; ``Recent
        Deaths`` counts only the Critical window. Both are shown because
        a tier the reader cannot account for is a tier they will not
        trust.

        Empty input yields an empty frame with those columns.
    """
    if df.empty or "Beat" not in df.columns:
        return pd.DataFrame(columns=_beat_columns())

    working = df.copy()
    working["_conflict"] = conflict_mask(working)
    working["_deaths"] = human_deaths(working)
    working["_injuries"] = human_injuries(working)
    working["_property"] = property_severity(working)
    working["_category"] = classify_conflict(working)

    key_cols = [c for c in BEAT_KEY_COLUMNS if c in working.columns]
    if "Beat" not in key_cols:
        key_cols = ["Beat"]

    grouped = working.groupby(key_cols, observed=True, dropna=False)

    table = grouped.agg(
        Reports=("_conflict", "size"),
        **{
            "Conflict Events": ("_conflict", "sum"),
            "Human Deaths": ("_deaths", "sum"),
            "People Injured": ("_injuries", "sum"),
            "Damage Burden": ("_property", "sum"),
        },
    ).reset_index()

    recent_deaths, recent_injuries = _window_casualties(
        working, key_cols, critical_window_days, as_of=as_of
    )
    table["Recent Deaths"] = recent_deaths
    table["Recent Injuries"] = recent_injuries

    table["Conflict Events"] = table["Conflict Events"].astype(int)
    table["House Damage Events"] = (
        grouped["_category"].apply(lambda s: int((s == "House").sum())).to_numpy()
    )
    table["Crop Damage Events"] = (
        grouped["_category"].apply(lambda s: int((s == "Crop").sum())).to_numpy()
    )

    table["Conflict Rate %"] = (
        table["Conflict Events"] / table["Reports"] * 100
    ).round(1)

    shrunk = shrink_rates(table["Conflict Events"], table["Reports"])
    table["Adj. Conflict Rate %"] = np.round(np.asarray(shrunk["adjusted"]) * 100, 1)
    landscape_rate = shrunk["prior_mean"] * 100

    table["Night Conflict %"] = _share_by_group(
        working, key_cols, "Is_Night", conflict_only=True
    )
    table["Near Village %"] = _share_by_group(
        working, key_cols, "Near Village", conflict_only=False
    )

    trend = _beat_trends(working, key_cols, recent_days)
    table = table.merge(trend, on=key_cols, how="left")
    table["Trend"] = table["Trend"].fillna(TREND_UNKNOWN)
    table["Recent vs Prior"] = table["Recent vs Prior"].fillna("n/a")

    table["Confidence"] = table["Reports"].map(_confidence_label)
    table["Priority Score"] = _priority_score(table)
    table["Priority Tier"] = _priority_tier(table, landscape_rate)
    table["Recommended Action"] = _recommended_actions(table)

    for col in BEAT_KEY_COLUMNS:
        if col not in table.columns:
            table[col] = "Unknown"

    table["_tier_rank"] = table["Priority Tier"].map(
        {tier: i for i, tier in enumerate(TIER_ORDER)}
    )
    table = (
        table.sort_values(
            ["_tier_rank", "Priority Score", "Human Deaths"],
            ascending=[True, False, False],
        )
        .drop(columns="_tier_rank")
        .reset_index(drop=True)
    )

    logger.info(
        "Beat intelligence: %d beats, landscape conflict rate %.1f%%, prior k=%.1f.",
        len(table),
        landscape_rate,
        shrunk["prior_strength"],
    )
    return table[_beat_columns()]


def _beat_columns() -> List[str]:
    return [
        "Beat",
        "Division",
        "Range",
        "Priority Tier",
        "Priority Score",
        "Confidence",
        "Reports",
        "Conflict Events",
        "Conflict Rate %",
        "Adj. Conflict Rate %",
        "Human Deaths",
        "People Injured",
        "Recent Deaths",
        "Recent Injuries",
        "House Damage Events",
        "Crop Damage Events",
        "Damage Burden",
        "Night Conflict %",
        "Near Village %",
        "Trend",
        "Recent vs Prior",
        "Recommended Action",
    ]


def _share_by_group(
    df: pd.DataFrame,
    key_cols: List[str],
    column: str,
    conflict_only: bool,
) -> np.ndarray:
    """Percentage of rows in each group where ``column`` is true.

    Args:
        conflict_only: Restrict the denominator to conflict rows. Night
            share is only meaningful over conflict events -- the share of
            *all* sightings that happen at night largely reflects when
            patrols go out.

    Returns NaN for a group with no usable rows, so "unknown" stays
    distinguishable from "zero".
    """
    if column not in df.columns:
        return np.full(df.groupby(key_cols, observed=True, dropna=False).ngroups, np.nan)

    subset = df[df["_conflict"]] if conflict_only else df
    values = pd.to_numeric(subset[column], errors="coerce")

    shares = (
        values.groupby([subset[c] for c in key_cols], observed=True, dropna=False)
        .mean()
        .mul(100)
        .round(1)
    )
    all_groups = df.groupby(key_cols, observed=True, dropna=False).size().index
    return shares.reindex(all_groups).to_numpy(dtype=float)


def _window_casualties(
    df: pd.DataFrame,
    key_cols: List[str],
    window_days: int,
    as_of: Optional[pd.Timestamp] = None,
) -> tuple:
    """People killed and injured per beat within the last ``window_days``.

    The window ends at ``as_of``, the review period's end -- not at each
    beat's own last report. Otherwise selecting a beat that stopped
    reporting months ago would measure from its final entry and flip an
    old fatality back to Critical.

    With no usable dates the whole period counts. For a casualty rule,
    over-escalating is recoverable and missing a fatality is not.

    Returns:
        ``(deaths, injuries)`` aligned to the beat grouping order.
    """
    index = df.groupby(key_cols, observed=True, dropna=False).size().index

    if "Date" not in df.columns or df["Date"].isna().all():
        recent = df
    else:
        anchor = pd.Timestamp(as_of) if as_of is not None else df["Date"].max()
        cutoff = anchor - pd.Timedelta(days=window_days)
        recent = df[df["Date"] > cutoff]

    def _summed(column: str) -> np.ndarray:
        if recent.empty:
            return np.zeros(len(index), dtype=float)
        totals = recent.groupby(key_cols, observed=True, dropna=False)[column].sum()
        return totals.reindex(index, fill_value=0.0).to_numpy(dtype=float)

    return _summed("_deaths"), _summed("_injuries")


def _beat_trends(
    df: pd.DataFrame, key_cols: List[str], recent_days: int
) -> pd.DataFrame:
    """Compare conflict events in the recent window against the prior one.

    Equal-length windows, so the comparison is like-for-like. Short
    datasets shrink both to half the span; below ``MIN_WINDOW_DAYS`` no
    trend is claimed.
    """
    empty = pd.DataFrame(columns=key_cols + ["Trend", "Recent vs Prior"])
    if "Date" not in df.columns or df["Date"].isna().all():
        return empty

    end = df["Date"].max()
    start = df["Date"].min()
    span_days = int((end - start).days) + 1
    window = min(recent_days, span_days // 2)
    if window < MIN_WINDOW_DAYS:
        return empty

    recent_start = end - pd.Timedelta(days=window)
    prior_start = recent_start - pd.Timedelta(days=window)

    recent = df[(df["Date"] > recent_start) & (df["Date"] <= end)]
    prior = df[(df["Date"] > prior_start) & (df["Date"] <= recent_start)]

    recent_counts = recent[recent["_conflict"]].groupby(
        key_cols, observed=True, dropna=False
    ).size()
    prior_counts = prior[prior["_conflict"]].groupby(
        key_cols, observed=True, dropna=False
    ).size()

    index = df.groupby(key_cols, observed=True, dropna=False).size().index
    recent_counts = recent_counts.reindex(index, fill_value=0)
    prior_counts = prior_counts.reindex(index, fill_value=0)

    out = pd.DataFrame(index=index)
    out["_recent"] = recent_counts
    out["_prior"] = prior_counts

    total = out["_recent"] + out["_prior"]
    # Continuity correction so 3-vs-0 is expressible without dividing by zero.
    ratio = (out["_recent"] + 0.5) / (out["_prior"] + 0.5)

    trend = np.where(
        total < MIN_EVENTS_FOR_TREND,
        TREND_UNKNOWN,
        np.where(
            ratio >= ESCALATION_RATIO,
            TREND_ESCALATING,
            np.where(ratio <= EASING_RATIO, TREND_EASING, TREND_STABLE),
        ),
    )
    out["Trend"] = trend
    out["Recent vs Prior"] = [
        f"{int(r)} vs {int(p)}" for r, p in zip(out["_recent"], out["_prior"])
    ]

    return out.drop(columns=["_recent", "_prior"]).reset_index()


def _confidence_label(reports: float) -> str:
    if reports >= CONFIDENCE_THRESHOLDS[CONFIDENCE_HIGH]:
        return CONFIDENCE_HIGH
    if reports >= CONFIDENCE_THRESHOLDS[CONFIDENCE_MEDIUM]:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def _percentile_rank(values: pd.Series) -> pd.Series:
    """Rank a column onto 0-100. Constant input maps to 0, not 50."""
    if values.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=values.index)
    return values.rank(pct=True, na_option="bottom") * 100


def _priority_score(table: pd.DataFrame) -> pd.Series:
    """Blend the component signals into a 0-100 ordering score.

    Orders beats within a tier only: some components are relative to the
    beats currently in view, so the number moves when filters change.
    """
    # Recent casualties at full weight, older at half: otherwise a beat
    # with only historical fatalities orders above one with a fresh death.
    older_deaths = (table["Human Deaths"] - table["Recent Deaths"]).clip(lower=0)
    older_injuries = (table["People Injured"] - table["Recent Injuries"]).clip(lower=0)

    casualty = (
        table["Recent Deaths"] * CASUALTY_POINTS_PER_DEATH
        + table["Recent Injuries"] * CASUALTY_POINTS_PER_INJURY
        + older_deaths * CASUALTY_POINTS_PER_DEATH * HISTORICAL_CASUALTY_DISCOUNT
        + older_injuries * CASUALTY_POINTS_PER_INJURY * HISTORICAL_CASUALTY_DISCOUNT
    ).clip(upper=100.0)

    burden = _percentile_rank(table["Damage Burden"])
    intensity = table["Adj. Conflict Rate %"].fillna(0.0).clip(0, 100)

    night = table["Night Conflict %"].fillna(0.0)
    village = table["Near Village %"]
    exposure = np.where(village.notna(), (night + village.fillna(0)) / 2, night)

    trend = table["Trend"].map(
        {
            TREND_ESCALATING: 100.0,
            TREND_STABLE: 50.0,
            TREND_EASING: 0.0,
            TREND_UNKNOWN: 50.0,
        }
    ).fillna(50.0)

    score = (
        casualty * SCORE_WEIGHTS["casualty"]
        + burden * SCORE_WEIGHTS["burden"]
        + intensity * SCORE_WEIGHTS["intensity"]
        + exposure * SCORE_WEIGHTS["exposure"]
        + trend * SCORE_WEIGHTS["trend"]
    )
    return score.round(1)


def _priority_tier(table: pd.DataFrame, landscape_rate: float) -> pd.Series:
    """Assign a decision tier from absolute rules.

    Written out rather than derived from the distribution, so a tier
    means the same thing across divisions, periods and filters.

    Critical is the deploy-now tier: all its triggers use the recent
    casualty window. An older fatality still elevates the beat to High.
    """
    recent_deaths = table["Recent Deaths"]
    recent_injuries = table["Recent Injuries"]
    deaths = table["Human Deaths"]
    injuries = table["People Injured"]
    events = table["Conflict Events"]
    escalating = table["Trend"] == TREND_ESCALATING
    rate = table["Adj. Conflict Rate %"].fillna(0.0)
    confident = table["Confidence"].isin([CONFIDENCE_HIGH, CONFIDENCE_MEDIUM])

    # Guard the zero case explicitly. With no conflict anywhere the
    # landscape rate is 0, the threshold is 0, and "rate >= threshold"
    # is true for every beat -- which would promote a completely quiet
    # landscape wholesale into Watch and High.
    rate_threshold = landscape_rate * RATE_MULTIPLE_FOR_HIGH
    rate_elevated = (rate >= rate_threshold) & (rate > 0) & (landscape_rate > 0)

    critical = (
        (recent_deaths > 0)
        | ((recent_injuries > 0) & escalating)
        | (recent_injuries >= INJURIES_FOR_CRITICAL)
    )
    high = (
        (deaths > 0)  # a fatality anywhere in the period, however old
        | (injuries > 0)
        | (rate_elevated & confident)
        | (table["House Damage Events"] >= HOUSE_EVENTS_FOR_HIGH)
        | (escalating & (events >= EVENTS_FOR_ESCALATION_HIGH))
    )
    watch = escalating | rate_elevated

    return pd.Series(
        np.select(
            [critical, high, watch],
            [TIER_CRITICAL, TIER_HIGH, TIER_WATCH],
            default=TIER_ROUTINE,
        ),
        index=table.index,
    )


def _recommended_actions(table: pd.DataFrame, max_actions: int = 3) -> pd.Series:
    """Map the signal that put a beat in its tier onto an intervention."""
    actions: List[str] = []

    for row in table.to_dict("records"):
        items: List[str] = []
        room = lambda: len(items) < max_actions  # noqa: E731 - local readability

        # Mutually exclusive: stacking both would spend the action
        # budget restating "send the response team". A fatality inside
        # the window calls for deployment; one outside it calls for a
        # check that the mitigation put in place then still holds.
        if row["Recent Deaths"] > 0:
            items.append(
                "Post rapid-response team; process ex-gratia; issue community alert"
            )
        elif row["Recent Injuries"] > 0:
            items.append("Rapid-response team on call; check early-warning coverage")
        elif row["Human Deaths"] > 0:
            items.append(
                "Past fatality, none recent: verify the mitigation put in place is holding"
            )
        elif row["People Injured"] > 0:
            items.append("Past injury, none recent: confirm early-warning coverage")

        night = row["Night Conflict %"]
        if pd.notna(night) and night >= NIGHT_SHARE_FOR_PATROL_SHIFT and room():
            items.append(f"Shift patrol to the night window ({night:.0f}% of conflict)")

        village = row["Near Village %"]
        if pd.notna(village) and village >= VILLAGE_SHARE_FOR_EARLY_WARNING and room():
            items.append("Village early warning; assess barriers on approach routes")

        house, crop = row["House Damage Events"], row["Crop Damage Events"]
        if house > 0 and house >= crop and room():
            items.append("Secure grain stores; structural mitigation for homesteads")
        elif crop > 0 and room():
            items.append("Crop-guarding support; review fencing on repeat-hit plots")

        if row["Trend"] == TREND_ESCALATING and room():
            items.append("Escalating: re-survey beat and establish the cause")

        if (
            row["Confidence"] == CONFIDENCE_LOW
            and row["Priority Tier"] != TIER_ROUTINE
            and room()
        ):
            items.append("Thin reporting: confirm with beat guard before committing staff")

        if not items:
            items.append("Routine monitoring")

        actions.append("; ".join(items[:max_actions]))

    return pd.Series(actions, index=table.index)


# ---------------------------------------------------------------------------
# Temporal risk windows
# ---------------------------------------------------------------------------
def temporal_risk_windows(
    df: pd.DataFrame, coverage_target: float = DEFAULT_COVERAGE_TARGET
) -> Dict[str, object]:
    """Identify when conflict actually happens, for shift planning.

    Returns:
        Dict with:

        ``hourly``: conflict events by hour (0-23, zero-filled).
        ``peak_window``: the *shortest* contiguous block of hours that
            contains at least ``coverage_target`` of all conflict events,
            as ``{"start", "end", "hours", "share", "events"}``, or None
            if there is nothing to summarise. Shortest-that-covers is the
            useful framing -- "these 6 hours hold 64% of incidents" sizes
            a shift, whereas a single peak hour does not.
        ``monthly``: conflict events by calendar month.
        ``peak_months``: month names accounting for the largest share,
            up to half the annual total -- the seasonal concentration
            that crop-guarding support should be timed against.
        ``night_share``: percentage of conflict events at night.
    """
    result: Dict[str, object] = {
        "hourly": hourly_conflict_profile(df),
        "peak_window": None,
        "monthly": pd.Series(dtype=int),
        "peak_months": [],
        "night_share": float("nan"),
    }
    if df.empty:
        return result

    conflicts = df[conflict_mask(df)]
    if conflicts.empty:
        return result

    result["peak_window"] = _peak_hour_window(result["hourly"], coverage_target)

    if "Date" in conflicts.columns and conflicts["Date"].notna().any():
        monthly = (
            conflicts["Date"].dt.month.value_counts().reindex(range(1, 13), fill_value=0).sort_index()
        )
        monthly.index = [
            pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)
        ]
        monthly.name = "Conflict Events"
        monthly.index.name = "Month"
        result["monthly"] = monthly
        result["peak_months"] = _peak_months(monthly)

    if "Is_Night" in conflicts.columns and conflicts["Is_Night"].notna().any():
        known = conflicts["Is_Night"].dropna()
        result["night_share"] = float(known.astype(bool).mean() * 100)

    return result


def _peak_hour_window(
    hourly: pd.Series, coverage_target: float
) -> Optional[Dict[str, object]]:
    """Shortest circular block of hours covering ``coverage_target`` of events.

    Circular because the elephant activity window straddles midnight; a
    non-wrapping scan would report "18:00-23:59" and drop the 00:00-02:00
    tail that belongs to the same night.
    """
    counts = np.asarray(hourly.to_numpy(), dtype=float)
    total = counts.sum()
    if total <= 0:
        return None

    for length in range(1, 25):
        best_start, best_events = None, -1.0
        for start in range(24):
            window = [(start + i) % 24 for i in range(length)]
            events = counts[window].sum()
            if events > best_events:
                best_start, best_events = start, events
        if best_events / total >= coverage_target:
            share = best_events / total
            return {
                "start": int(best_start),
                "end": int((best_start + length) % 24),
                "hours": int(length),
                "share": float(share * 100),
                "events": int(best_events),
                # How much denser this window is than an even spread over
                # the day. A 12-hour window holding 64% of incidents is
                # only 1.3x uniform -- real, but not a reason to
                # restructure a shift. Without this the share alone reads
                # far more decisively than it should.
                "lift": float(share / (length / 24)),
            }
    return None


def _peak_months(monthly: pd.Series) -> List[str]:
    """Months carrying materially more conflict than an even spread.

    Returns an empty list when conflict is spread evenly through the
    year. Taking "the top months until they add to half the total" always
    returns something -- on flat data that is six months labelled a
    "seasonal peak", which is worse than saying nothing.
    """
    total = float(monthly.sum())
    if total <= 0:
        return []

    even_share = total / 12
    ordered = monthly.sort_values(ascending=False)
    return [
        str(month)
        for month, count in ordered.head(MAX_PEAK_MONTHS).items()
        if count >= even_share * SEASONAL_LIFT
    ]


def format_window(window: Optional[Dict[str, object]]) -> str:
    """Render a peak window with its share and how concentrated it really is."""
    if not window:
        return "not determinable from the available hours"
    return (
        f"{window['start']:02d}:00-{window['end']:02d}:00 "
        f"({window['hours']}h, {window['share']:.0f}% of conflict events, "
        f"{window['lift']:.1f}x an even spread)"
    )


# ---------------------------------------------------------------------------
# Village exposure
# ---------------------------------------------------------------------------
def village_exposure(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank villages by the conflict recorded in their immediate surroundings.

    Requires village-centroid enrichment; returns an empty frame without
    it. This is the layer that turns a beat-level posting decision into a
    village-level one -- which settlements get early-warning coverage,
    barrier assessment, or a compensation camp.

    Args:
        df: Enriched dataframe including ``Nearest Village`` and
            ``Distance to Village (km)``.
        top_n: Maximum villages to return.

    Returns:
        DataFrame with ``Village``, ``Conflict Events``, ``Human Deaths``,
        ``People Injured``, ``House Damage Events``, ``Crop Damage Events``
        and ``Median Distance (km)``, sorted worst first.
    """
    columns = [
        "Village",
        "Conflict Events",
        "Human Deaths",
        "People Injured",
        "House Damage Events",
        "Crop Damage Events",
        "Median Distance (km)",
    ]
    if df.empty or "Nearest Village" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    working["_conflict"] = conflict_mask(working)
    working["_deaths"] = human_deaths(working)
    working["_injuries"] = human_injuries(working)
    working["_category"] = classify_conflict(working)

    # Only incidents close enough for the village to be the meaningful
    # unit. Beyond the near-village radius the nearest centroid is just
    # the least-distant point on the map, not a place anyone lives next to.
    if "Near Village" in working.columns:
        working = working[working["Near Village"].fillna(False).astype(bool)]

    working = working[working["_conflict"]]
    if working.empty:
        return pd.DataFrame(columns=columns)

    grouped = working.groupby("Nearest Village", observed=True)
    out = grouped.agg(
        **{
            "Conflict Events": ("_conflict", "sum"),
            "Human Deaths": ("_deaths", "sum"),
            "People Injured": ("_injuries", "sum"),
        }
    )
    out["House Damage Events"] = grouped["_category"].apply(
        lambda s: int((s == "House").sum())
    )
    out["Crop Damage Events"] = grouped["_category"].apply(
        lambda s: int((s == "Crop").sum())
    )
    if "Distance to Village (km)" in working.columns:
        out["Median Distance (km)"] = grouped["Distance to Village (km)"].median().round(2)
    else:
        out["Median Distance (km)"] = float("nan")

    out = out.reset_index().rename(columns={"Nearest Village": "Village"})
    out["Conflict Events"] = out["Conflict Events"].astype(int)
    return (
        out[columns]
        .sort_values(
            ["Human Deaths", "People Injured", "Conflict Events"], ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# The brief
# ---------------------------------------------------------------------------
def management_brief(
    df: pd.DataFrame,
    recent_days: int = DEFAULT_RECENT_DAYS,
    critical_window_days: int = CRITICAL_CASUALTY_WINDOW_DAYS,
    as_of: Optional[pd.Timestamp] = None,
) -> Dict[str, object]:
    """Assemble the full conservation-intelligence brief for a period.

    This is the object the app panel and the HTML report both render, so
    the numbers a manager reads on screen and the numbers in the document
    they forward upwards cannot drift apart.

    Args:
        df: Filtered, enriched dataframe.
        recent_days: Escalation comparison window.
        critical_window_days: Casualty recency window behind the
            Critical tier.

    Returns:
        Dict with ``period``, ``coverage``, ``kpis``, ``beats``,
        ``priority_beats``, ``escalating``, ``temporal``, ``villages``,
        ``headlines`` (ready-to-read sentences) and ``caveats``.
    """
    kpis = compute_kpis(df)
    beats = beat_intelligence(
        df,
        recent_days=recent_days,
        critical_window_days=critical_window_days,
        as_of=as_of,
    )
    temporal = temporal_risk_windows(df)
    villages = village_exposure(df)

    priority = beats[beats["Priority Tier"].isin([TIER_CRITICAL, TIER_HIGH])]
    escalating = beats[beats["Trend"] == TREND_ESCALATING]

    period = {"start": None, "end": None, "days": 0}
    if not df.empty and "Date" in df.columns and df["Date"].notna().any():
        start, end = df["Date"].min(), df["Date"].max()
        period = {"start": start, "end": end, "days": int((end - start).days) + 1}

    coverage = {
        "reports": int(len(df)),
        "beats": int(len(beats)),
        "divisions": int(df["Division"].nunique()) if "Division" in df.columns else 0,
        "recent_days": recent_days,
        "critical_window_days": critical_window_days,
    }

    return {
        "period": period,
        "coverage": coverage,
        "kpis": kpis,
        "beats": beats,
        "priority_beats": priority,
        "escalating": escalating,
        "temporal": temporal,
        "villages": villages,
        "headlines": _headlines(
            kpis, beats, priority, escalating, temporal, critical_window_days
        ),
        "caveats": _caveats(df, kpis, beats, villages, critical_window_days),
    }


def _headlines(
    kpis: Dict[str, float],
    beats: pd.DataFrame,
    priority: pd.DataFrame,
    escalating: pd.DataFrame,
    temporal: Dict[str, object],
    critical_window_days: int = CRITICAL_CASUALTY_WINDOW_DAYS,
) -> List[str]:
    """The handful of sentences worth putting at the top of the brief."""
    lines: List[str] = []

    deaths = kpis.get("human_deaths", 0)
    injuries = kpis.get("human_injuries", 0)
    if deaths or injuries:
        lines.append(
            f"{int(deaths)} human fatality(ies) and {int(injuries)} injury(ies) "
            f"recorded across {kpis['human_death_incidents']} fatal incident(s)."
        )
    else:
        lines.append("No human fatalities or injuries recorded in this period.")

    rate = kpis.get("conflict_rate", float("nan"))
    if not pd.isna(rate):
        lines.append(
            f"{kpis['conflicts']:,} of {kpis['entries']:,} reports involved conflict "
            f"({rate:.1f}%)."
        )

    critical = beats[beats["Priority Tier"] == TIER_CRITICAL] if not beats.empty else beats
    if len(critical):
        names = ", ".join(critical["Beat"].head(5).astype(str))
        lines.append(
            f"{len(critical)} beat(s) in the Critical tier -- a casualty in the last "
            f"{critical_window_days} days: {names}."
        )
    elif len(priority):
        names = ", ".join(priority["Beat"].head(5).astype(str))
        lines.append(f"{len(priority)} beat(s) require priority attention: {names}.")

    if len(escalating):
        names = ", ".join(escalating["Beat"].head(5).astype(str))
        lines.append(f"{len(escalating)} beat(s) escalating versus the previous window: {names}.")

    window = temporal.get("peak_window")
    if window:
        lines.append(f"Conflict concentrates in {format_window(window)}.")

    months = temporal.get("peak_months") or []
    if months:
        lines.append(f"Seasonal peak: {', '.join(months)}.")
    elif not temporal.get("monthly", pd.Series(dtype=int)).empty:
        lines.append("No marked seasonal peak -- conflict is spread through the year.")

    return lines


def _caveats(
    df: pd.DataFrame,
    kpis: Dict[str, float],
    beats: pd.DataFrame,
    villages: pd.DataFrame,
    critical_window_days: int = CRITICAL_CASUALTY_WINDOW_DAYS,
) -> List[str]:
    """Limits to state every time, not only when something looks wrong."""
    caveats = [
        "Report counts reflect patrol and reporting effort as well as elephant "
        "activity. A beat with more reports may be better watched, not worse "
        "affected -- read the adjusted conflict rate alongside the raw volume."
    ]

    if not beats.empty:
        thin = int((beats["Confidence"] == CONFIDENCE_LOW).sum())
        if thin:
            caveats.append(
                f"{thin} of {len(beats)} beats have fewer than "
                f"{CONFIDENCE_THRESHOLDS[CONFIDENCE_MEDIUM]} reports. Their rates are "
                "shrunk toward the landscape average and marked low confidence."
            )

    night_known = kpis.get("night_known", 0)
    entries = kpis.get("entries", 0)
    if entries and night_known < entries:
        caveats.append(
            f"Night/day is known for {night_known:,} of {entries:,} reports; "
            "timing figures are based only on those."
        )

    if villages.empty:
        caveats.append(
            "No village-centroid data applied, so village exposure and the "
            "proximity component of the priority score are unavailable."
        )

    if not df.empty and "Date" in df.columns and df["Date"].notna().any():
        span = int((df["Date"].max() - df["Date"].min()).days) + 1
        if span < 2 * MIN_WINDOW_DAYS:
            caveats.append(
                f"The period covers {span} day(s) -- too short to compare against a "
                "previous window, so no trend is claimed."
            )

    caveats.append(
        f"Critical means a person was killed or injured in the last "
        f"{critical_window_days} days of the period shown. Beats with older "
        "casualties and nothing recent sit in High, not Critical -- check the "
        "'Human Deaths' column alongside 'Recent Deaths' before concluding a "
        "beat is safe."
    )

    if not beats.empty and "Recent Deaths" in beats.columns:
        historical_only = int(
            ((beats["Human Deaths"] > 0) & (beats["Recent Deaths"] == 0)).sum()
        )
        if historical_only:
            caveats.append(
                f"{historical_only} beat(s) had a fatality earlier in the period but "
                f"none in the last {critical_window_days} days, so they are High "
                "rather than Critical."
            )

    caveats.append(
        "Tiers are set by fixed rules and are comparable across periods and "
        "filters. The priority score only orders beats within a tier and will "
        "move when filters change."
    )
    return caveats
