"""Where groups may be next, and whether each part of the forecast earns its place.

The forecast has to beat a dull alternative -- "they stay near where they
were last seen" -- on the register itself, or say that it does not. These
tests build movement whose ingredients are known (movement_fixtures) and
check that the forecast picks up each one, leaves out what is not there,
keeps its zones honest, and never sees the future.
"""

import numpy as np
import pandas as pd
import pytest

from core import movement as mv
from movement_fixtures import KM_PER_DEG, START, simulate


@pytest.fixture(scope="module")
def wandering():
    return simulate(groups=8, days=240, seed=1)


@pytest.fixture(scope="module")
def travelling():
    return simulate(groups=8, days=240, seed=1, persistence=0.85)


@pytest.fixture(scope="module")
def ranging():
    return simulate(groups=8, days=240, seed=1, home=0.5)


@pytest.fixture(scope="module")
def shuttling():
    """Groups going back and forth between a day refuge and a field."""
    rng = np.random.default_rng(5)
    rows = []
    for g in range(8):
        refuge = np.array([g % 4 * 35.0, g // 4 * 35.0])
        field = refuge + rng.normal(0, 1, 2) / np.sqrt(2) * 4.0
        for d in range(200):
            at = (field if d % 2 else refuge) + rng.normal(0, 0.6, 2)
            if rng.random() < 0.7:
                rows.append({"Date": START + pd.Timedelta(days=d),
                             "Latitude": 23.3 + at[1] / KM_PER_DEG,
                             "Longitude": 81.0 + at[0] / (KM_PER_DEG * np.cos(np.radians(23.3))),
                             "Total Count": 1, "Male Count": 1, "Division": "D", "Range": "R",
                             "Beat": "B", "Nearest Village": "V"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def records(wandering, travelling, ranging, shuttling):
    return {name: mv.backtest(frame, horizon_days=1) for name, frame in (
        ("wandering", wandering), ("travelling", travelling), ("ranging", ranging),
        ("shuttling", shuttling))}


# ---------------------------------------------------------------------------
# Nothing from the future
# ---------------------------------------------------------------------------
def test_the_tracker_only_looks_back(ranging):
    """Units given to early reports do not change when later reports arrive,
    which is what lets the backtest reuse one tracking run."""
    early = ranging[ranging["Date"] < START + pd.Timedelta(days=60)]
    alone = mv.assign_units(early)
    within_all = mv.assign_units(ranging).loc[early.index]
    pd.testing.assert_series_equal(alone, within_all, check_names=False)


def test_the_usage_map_ignores_reports_from_the_day_forecast_onwards(ranging):
    grid = mv.Grid.around(ranging)
    as_of = START + pd.Timedelta(days=90)
    before = mv.usage_prior(ranging, grid, as_of)
    altered = ranging.copy()
    altered.loc[altered["Date"] >= as_of, "Latitude"] += 0.2
    assert np.allclose(before, mv.usage_prior(altered, grid, as_of))


def test_the_season_map_only_reads_earlier_years():
    two_years = simulate(groups=4, days=500, seed=2, home=0.5, seasonal=True)
    grid = mv.Grid.around(two_years)
    as_of = START + pd.Timedelta(days=430)
    recent = mv.usage_prior(two_years, grid, as_of)
    season = mv.season_prior(two_years, grid, as_of, recent)
    assert season is not None
    altered = two_years.copy()
    altered.loc[altered["Date"] >= as_of - pd.Timedelta(days=mv.SEASON_MIN_AGE_DAYS),
                "Latitude"] += 0.3
    again = mv.season_prior(altered, grid, as_of, recent)
    assert np.allclose(season, again)


def test_a_short_register_has_no_season_map(ranging):
    grid = mv.Grid.around(ranging)
    as_of = ranging["Date"].max()
    assert mv.season_prior(ranging, grid, as_of, mv.usage_prior(ranging, grid, as_of)) is None


# ---------------------------------------------------------------------------
# The parts
# ---------------------------------------------------------------------------
def test_cells_within_reach_are_found_in_order():
    grid = mv.Grid(23.0, 0.0, 0.0, 50, 40)
    index, distance = grid.near(np.array([10.2, 20.7]), 5.0)
    assert (np.diff(index) > 0).all()
    assert distance.max() <= 5.0
    assert len(index) == pytest.approx(np.pi * 25, rel=0.15)


def test_lone_bulls_and_family_herds_get_their_own_step_lengths():
    mixed = simulate(groups=9, days=300, seed=3)
    model = mv.StepModel.learn(mv.steps(mixed, mv.assign_units(mixed)))
    assert {"lone bull", "family herd"} <= set(model.per_day)
    assert np.median(model.lengths("lone bull")) > np.median(model.lengths("family herd"))


def test_steady_travel_is_learned_as_turns_near_zero(travelling):
    model = mv.StepModel.learn(mv.steps(travelling, mv.assign_units(travelling)))
    mu, kappa = model.turn(1)
    assert abs(mu) < np.radians(25) and kappa > 1


def test_refuge_to_field_and_back_is_learned_as_turns_near_half_a_circle(shuttling):
    model = mv.StepModel.learn(mv.steps(shuttling, mv.assign_units(shuttling)))
    mu, kappa = model.turn(1)
    assert abs(abs(mu) - np.pi) < np.radians(25) and kappa > 1


def test_a_zone_holds_its_share_of_the_forecast():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    p /= p.sum()
    likely, possible = mv.zone(p, mv.LIKELY_MASS), mv.zone(p, mv.POSSIBLE_MASS)
    assert likely.sum() < possible.sum() and not (likely & ~possible).any()
    assert p[likely].sum() >= mv.LIKELY_MASS


def test_calibration_widens_a_zone_that_caught_too_little():
    # Past zones of half the forecast caught the target only when it
    # needed 0.5 or less: here that happened 30% of the time.
    scores = list(np.r_[np.full(30, 0.3), np.full(70, 0.9)])
    assert mv.calibrated_mass(scores, 0.5) > 0.5
    assert mv.calibrated_mass(scores[:10], 0.5) == 0.5   # too few to calibrate on


# ---------------------------------------------------------------------------
# The track record: does each part earn its place?
# ---------------------------------------------------------------------------
def test_wandering_groups_get_no_extras_and_no_claim(records):
    record = records["wandering"]
    assert record.tested > 200
    assert record.config == mv.BASELINE
    assert not record.beats_staying_put


def test_direction_of_travel_is_used_when_groups_travel(records):
    record = records["travelling"]
    assert record.config.heading
    assert record.beats_staying_put
    assert record.forecast_rate - record.stay_rate > 0.05
    assert record.gain_bits > 0.3


def test_places_are_used_when_groups_keep_to_a_range(records):
    record = records["ranging"]
    assert record.config.places > 0
    assert record.beats_staying_put


def test_shuttling_groups_are_forecast_far_better_than_staying_put(records):
    record = records["shuttling"]
    assert record.config.heading
    assert record.forecast_rate > 3 * record.stay_rate
    # To catch as many next reports, a circle had to be several times larger.
    assert record.circle_area_km2 > 2.5 * record.mean_area_km2


@pytest.mark.parametrize("name", ["wandering", "travelling", "ranging", "shuttling"])
def test_the_likely_zone_catches_about_half(records, name):
    """Calibrated: 'likely' should mean about half, whatever the movement."""
    assert records[name].forecast_rate == pytest.approx(mv.LIKELY_MASS, abs=0.07)


def test_a_short_register_has_no_track_record():
    record = mv.backtest(simulate(groups=2, days=40, seed=4))
    assert record.tested == 0 and not record.beats_staying_put


# ---------------------------------------------------------------------------
# The groups to forecast now
# ---------------------------------------------------------------------------
def test_only_groups_seen_recently_are_forecast(ranging):
    as_of = ranging["Date"].max()
    out = mv.next_moves(ranging, as_of)
    assert len(out["groups"]) > 0
    assert (out["groups"]["Days Ago"] < mv.ACTIVE_DAYS).all()
    assert mv.next_moves(ranging, as_of + pd.Timedelta(days=30))["groups"].empty


def test_each_group_has_a_heading_and_a_most_likely_spot(travelling, records):
    out = mv.next_moves(travelling, travelling["Date"].max(), record=records["travelling"])
    groups = out["groups"]
    assert set(groups["Heading"]) - {""} <= set(mv.COMPASS)
    assert groups["Most Likely"].str.len().gt(0).all()


def test_the_forecast_points_the_way_a_travelling_group_is_going(travelling, records):
    out = mv.next_moves(travelling, travelling["Date"].max(), record=records["travelling"])
    moving = out["groups"][out["groups"]["Heading"] != ""]
    agree = (moving["Most Likely"].str.split(" ").str[2] == moving["Heading"])
    assert agree.mean() >= 0.5


def test_villages_in_a_groups_likely_zone_are_named(ranging):
    as_of = ranging["Date"].max()
    first = mv.next_moves(ranging, as_of)["groups"].iloc[0]
    villages = pd.DataFrame({"Village": ["On the spot", "Far away"],
                             "Latitude": [first["Latitude"], first["Latitude"] + 1.0],
                             "Longitude": [first["Longitude"], first["Longitude"]]})
    named = mv.next_moves(ranging, as_of, villages=villages)["villages"]
    assert "On the spot" in set(named["Village"])
    assert "Far away" not in set(named["Village"])
    assert set(named["Band"]) <= {"Likely", "Possible"}


def test_the_map_cells_carry_a_chance_between_zero_and_one(ranging):
    cells = mv.next_moves(ranging, ranging["Date"].max())["cells"]
    assert len(cells) and cells["Chance"].between(0, 1).all()
    assert cells["Likely"].any()


def test_compass_points():
    assert mv.compass(0.0) == "E"
    assert mv.compass(np.pi / 2) == "N"
    assert mv.compass(-np.pi / 2) == "S"
    assert mv.compass(3 * np.pi / 4) == "NW"
