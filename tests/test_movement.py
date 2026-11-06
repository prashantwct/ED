"""Where groups may be next, and whether that forecast earns its place.

The forecast has to beat a dull alternative -- "they stay near where they
were last seen" -- on the register itself, or say that it does not. These
tests build movement with and without structure and check that the
self-test tells the two apart, and that nothing from the future leaks
into a forecast.
"""

import numpy as np
import pandas as pd
import pytest

from core import movement as mv

START = pd.Timestamp("2025-10-01")


def _simulate(structured: bool, groups: int = 8, days: int = 150, seed: int = 3):
    """Groups circling a loop of fixed places, or wandering at random.

    Real elephants sit between the two: they return to refuges and fields,
    and they also travel.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(groups):
        base = np.array([23.0 + rng.uniform(0, 0.8), 81.0 + rng.uniform(0, 0.8)])
        angles = np.sort(rng.uniform(0, 2 * np.pi, 6))
        places = base + 0.04 * np.column_stack([np.sin(angles), np.cos(angles)])
        k, pos = 0, base.copy()
        size = int(rng.choice([1, 1, 2, 3]))
        for d in range(days):
            if structured:
                k = (k + (1 if rng.random() < 0.7 else 0)) % len(places)
                pos = places[k] + rng.normal(0, 0.003, 2)
            else:
                pos = pos + rng.normal(0, 0.03, 2)
            if rng.random() < 0.6:
                rows.append({"Date": START + pd.Timedelta(days=d), "Latitude": pos[0],
                             "Longitude": pos[1], "Total Count": size, "Male Count": size,
                             "Division": "D", "Range": "R", "Beat": f"B{g}",
                             "Nearest Village": f"V{g}"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def structured():
    return _simulate(True)


@pytest.fixture(scope="module")
def wandering():
    return _simulate(False)


# ---------------------------------------------------------------------------
# Nothing from the future
# ---------------------------------------------------------------------------
def test_the_tracker_only_looks_back(structured):
    """Units given to early reports do not change when later reports arrive,
    which is what lets the backtest reuse one tracking run."""
    cut = START + pd.Timedelta(days=60)
    early = structured[structured["Date"] < cut]
    alone = mv.assign_units(early)
    within_all = mv.assign_units(structured).loc[early.index]
    pd.testing.assert_series_equal(alone, within_all, check_names=False)


def test_the_usage_map_ignores_reports_from_the_day_forecast_onwards(structured):
    grid = mv.Grid.around(structured)
    as_of = START + pd.Timedelta(days=90)
    before = mv.usage_prior(structured, grid, as_of)
    altered = structured.copy()
    later = altered["Date"] >= as_of
    altered.loc[later, "Latitude"] += 0.2   # move every later report elsewhere
    assert np.allclose(before, mv.usage_prior(altered, grid, as_of))


# ---------------------------------------------------------------------------
# The forecast itself
# ---------------------------------------------------------------------------
def test_a_forecast_is_a_probability_over_the_grid(structured):
    grid = mv.Grid.around(structured)
    prior = mv.usage_prior(structured, grid, structured["Date"].max())
    origin = grid.project(23.4, 81.4)[0]
    p = mv.forecast(origin, 1, grid, prior, np.array([]))
    assert p.min() >= 0 and abs(p.sum() - 1) < 1e-9


def test_the_possible_zone_contains_the_likely_zone(structured):
    grid = mv.Grid.around(structured)
    prior = mv.usage_prior(structured, grid, structured["Date"].max())
    p = mv.forecast(grid.project(23.4, 81.4)[0], 1, grid, prior, np.array([]))
    likely, possible = mv.zone(p, mv.LIKELY_MASS), mv.zone(p, mv.POSSIBLE_MASS)
    assert likely.sum() < possible.sum()
    assert not (likely & ~possible).any()
    assert p[likely].sum() >= mv.LIKELY_MASS


def test_more_time_means_a_wider_zone():
    grid = mv.Grid(23.0, 0.0, 0.0, 80, 80)
    flat = np.full(80 * 80, 1 / (80 * 80))
    origin = np.array([40.0, 40.0])
    one = mv.zone(mv.forecast(origin, 1, grid, flat, np.array([]))).sum()
    three = mv.zone(mv.forecast(origin, 3, grid, flat, np.array([]))).sum()
    assert three > one


def test_without_the_usage_map_the_zone_is_a_disc_around_the_last_report():
    grid = mv.Grid(23.0, 0.0, 0.0, 60, 60)
    prior = np.random.default_rng(0).random(60 * 60)
    prior /= prior.sum()
    origin = np.array([30.0, 30.0])
    likely = mv.zone(mv.forecast(origin, 1, grid, prior, np.array([]), prior_weight=0.0))
    d = np.linalg.norm(grid.centres - origin, axis=1)
    assert d[likely].max() <= d[~likely].min() + grid.cell_km


def test_observed_steps_set_the_reach(structured):
    learned = mv.step_lengths(mv.steps(structured, mv.assign_units(structured)))
    assert len(learned) >= mv.MIN_STEPS
    r, density = mv.radial_profile(learned, 1)
    # Most of the mass sits within the longest observed daily step.
    ring = density * 2 * np.pi * np.maximum(r, 0.5)
    inside = ring[r <= learned.max()].sum() / ring.sum()
    assert inside > 0.9


# ---------------------------------------------------------------------------
# The track record: does it beat "they stay where they were"?
# ---------------------------------------------------------------------------
def test_on_groups_that_keep_to_their_places_the_forecast_beats_staying_put(structured):
    record = mv.backtest(structured, horizon_days=1)
    assert record.tested > 200
    assert record.forecast_rate > record.stay_rate
    assert record.beats_staying_put
    assert record.prior_weight > 0


def test_on_wandering_groups_it_does_not_claim_to_beat_staying_put(wandering):
    record = mv.backtest(wandering, horizon_days=1)
    assert record.tested > 200
    assert not record.beats_staying_put
    # It learns to ignore the usage map rather than be dragged by it.
    assert record.prior_weight == 0.0
    assert abs(record.forecast_rate - record.stay_rate) < 0.05


def test_a_short_register_has_no_track_record():
    short = _simulate(True, groups=2, days=40)
    record = mv.backtest(short)
    assert record.tested == 0 and not record.beats_staying_put


# ---------------------------------------------------------------------------
# The groups to forecast now
# ---------------------------------------------------------------------------
def test_only_groups_seen_recently_are_forecast(structured):
    as_of = structured["Date"].max()
    out = mv.next_moves(structured, as_of)
    assert len(out["groups"]) > 0
    assert (out["groups"]["Days Ago"] < mv.ACTIVE_DAYS).all()
    later = mv.next_moves(structured, as_of + pd.Timedelta(days=30))
    assert later["groups"].empty


def test_villages_in_a_groups_likely_zone_are_named(structured):
    as_of = structured["Date"].max()
    first = mv.next_moves(structured, as_of)["groups"].iloc[0]
    villages = pd.DataFrame({"Village": ["On the spot", "Far away"],
                             "Latitude": [first["Latitude"], first["Latitude"] + 1.0],
                             "Longitude": [first["Longitude"], first["Longitude"]]})
    out = mv.next_moves(structured, as_of, villages=villages)
    named = out["villages"]
    assert "On the spot" in set(named["Village"])
    assert "Far away" not in set(named["Village"])
    assert set(named["Band"]) <= {"Likely", "Possible"}


def test_the_map_cells_carry_a_chance_between_zero_and_one(structured):
    out = mv.next_moves(structured, structured["Date"].max())
    cells = out["cells"]
    assert len(cells) and cells["Chance"].between(0, 1).all()
    assert cells["Likely"].any()
