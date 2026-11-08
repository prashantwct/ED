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
    local = grid.near(np.array([10.2, 20.7]), 5.0)
    assert (np.diff(local.keys) > 0).all()
    assert local.distance.max() <= 5.0
    assert len(local.keys) == pytest.approx(np.pi * 25, rel=0.15)
    # The lattice key and the grid's own index name the same cells.
    assert (grid.key_of(grid.centres[local.flat]) == local.keys).all()


def test_cells_beyond_the_grid_are_still_reachable_at_the_floor():
    grid = mv.Grid(23.0, 0.0, 0.0, 10, 10)
    local = grid.near(np.array([1.0, 1.0]), 4.0)
    assert (local.flat == -1).any() and (local.flat >= 0).any()
    values = grid.at(np.full(100, 0.5), local)
    assert (values[local.flat == -1] == grid.floor()).all()


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
    """Seen yesterday, a shuttling group is somewhere else today; seen the
    day before, it is back. The page forecasts both, and so does the test,
    so staying put is right a fair share of the time -- but far less often."""
    record = records["shuttling"]
    assert record.config.heading
    assert record.forecast_rate > 1.6 * record.stay_rate
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


# ---------------------------------------------------------------------------
# What the methodology review fixed (docs/MOVEMENT_REVIEW.md)
# ---------------------------------------------------------------------------
def _report(day, x_km, y_km=0.0, total=1, males=None, females=0, calves=0, hour=8):
    km_lon = KM_PER_DEG * np.cos(np.radians(23.3))
    return {"Date": START + pd.Timedelta(days=day, hours=hour),
            "Latitude": 23.3 + y_km / KM_PER_DEG, "Longitude": 81.0 + x_km / km_lon,
            "Total Count": total, "Male Count": total if males is None else males,
            "Female Count": females, "Calf Count": calves}


def test_a_herd_reported_without_its_breakdown_stays_one_track():
    """Field reports often give only a total. That is not a different kind
    of group: the herd counted with its calves one day and as a bare 12 the
    next is still one herd."""
    rows = []
    for day in range(10):
        if day % 2:
            rows.append(_report(day, 2.0 * day, total=12, males=0))
        else:
            rows.append(_report(day, 2.0 * day, total=12, males=2, females=7, calves=3))
    units = mv.assign_units(pd.DataFrame(rows))
    assert units.nunique() == 1


def test_a_days_reports_are_matched_to_tracks_all_at_once():
    """Two lone bulls 8 km apart. Next morning the report filed first is
    a hair closer to the wrong bull; taken one at a time it would claim
    that bull's track and push the other report onto the wrong one."""
    df = pd.DataFrame([_report(0, 0.0), _report(0, 8.0),
                       _report(1, 3.9, hour=6),    # bull B, 4.1 km from where it was
                       _report(1, 0.5, hour=7)])   # bull A, barely moved
    units = mv.assign_units(df)
    assert units[3] == units[0]
    assert units[2] == units[1]


def test_two_people_reporting_one_herd_do_not_split_it_or_steal_a_track():
    df = pd.DataFrame([_report(0, 0.0, total=12), _report(0, 6.0, total=12),
                       _report(1, 0.3, total=12, hour=6), _report(1, 0.8, total=12, hour=9),
                       _report(1, 6.2, total=12, hour=10)])
    units = mv.assign_units(df)
    assert units[2] == units[3] == units[0]
    assert units[4] == units[1]


def test_distance_grows_with_time_as_the_register_shows(wandering, travelling):
    """A random walk goes twice as far in four days as in one; a group
    holding its heading, nearly four times. Assuming the first for both
    made multi-day zones too small for travelling groups."""
    walk = mv.scaling(mv.steps(wandering, mv.assign_units(wandering)))
    travel = mv.scaling(mv.steps(travelling, mv.assign_units(travelling)))
    assert walk < 0.7
    assert travel > 0.9


def test_the_next_three_days_cover_each_of_the_three():
    """A three-day forecast for a group seen today covers tomorrow too, not
    only the third day."""
    situation = mv.Situation(np.array([0.0, 0.0]), 3.0, "lone bull", span=3)
    assert situation.leads() == [1.0, 2.0, 3.0]
    assert mv.lead(since=1, horizon_days=3) == (4, 3)
    assert mv.lead(since=0, horizon_days=1) == (1, 1)
    grid = mv.Grid(23.5, -50.0, -50.0, 100, 100)
    model = mv.StepModel({"all": np.full(100, 4.0)}, {}, 1.0)
    near_start = []
    for span in (1, 3):
        local, parts = mv._components(mv.Situation(np.array([0.5, 0.5]), 3.0, "lone bull",
                                                   span=span), grid, model, {})
        p = mv._combine(parts, mv.BASELINE)
        near_start.append(p[local.distance <= 6.0].sum())
    assert near_start[1] > near_start[0] + 0.2


def test_each_morning_a_group_was_shown_is_a_forecast_to_score():
    """Seen on day 0, next reported on day 2: with a one-day horizon the
    page forecast it on the morning of day 1; with three days, on days 0
    and 1. The old backtest scored only the move, with its gap known."""
    steps = pd.DataFrame({"Unit": [0], "From Date": [START], "To Date": [START + pd.Timedelta(days=2)],
                          "Gap (days)": [2]})
    one = mv._cases(steps, 1, START, START + pd.Timedelta(days=30))
    assert list(one["Since"]) == [1] and list(one["Days"]) == [2] and list(one["Span"]) == [1]
    three = mv._cases(steps, 3, START, START + pd.Timedelta(days=30))
    assert list(three["Since"]) == [0, 1]
    assert list(three["Days"]) == [3, 4] and list(three["Span"]) == [3, 3]


def test_a_group_that_shuttles_is_expected_to_turn_back():
    """Some groups keep going, others go back the way they came. One mean
    turn for the register averages the two into no direction at all; the
    turn density keeps both, and a group's own turns pick between them."""
    turns = np.r_[np.random.default_rng(0).normal(0, 0.3, 200),
                  np.random.default_rng(1).normal(np.pi, 0.3, 200)]
    model = mv.StepModel({"all": np.full(100, 3.0)}, {1: (0.0, 0.0)},
                         turn_samples={1: np.mod(turns, 2 * np.pi)})
    centre = (np.arange(mv.TURN_BINS) + 0.5) * 2 * np.pi / mv.TURN_BINS
    both = model.turn_density(1)
    assert both[np.argmin(np.abs(centre - np.pi))] > 2 * both[mv.TURN_BINS // 4]
    shuttler = model.turn_density(1, own=np.full(6, np.pi))
    traveller = model.turn_density(1, own=np.zeros(6))
    assert abs(centre[np.argmax(shuttler)] - np.pi) < 0.3
    assert min(centre[np.argmax(traveller)], 2 * np.pi - centre[np.argmax(traveller)]) < 0.3


def test_zones_are_calibrated_on_recent_forecasts():
    """Early forecasts, made from a few dozen moves, needed more of the
    forecast to catch the group; they must not keep later zones too wide."""
    early = [0.95] * 500
    recent = [0.4] * mv.CALIBRATION_WINDOW
    assert mv.calibrated_mass(early + recent, 0.5) == pytest.approx(0.4)


def test_the_landscape_score_counts_every_report_not_only_linked_ones(records):
    for record in records.values():
        assert record.field_reports >= record.tested
        assert 0 <= record.field_rate <= 1


def test_on_the_landscape_too_travelling_groups_beat_staying_put(records):
    record = records["travelling"]
    assert record.field_rate > record.field_circle_rate + 0.05


def test_compass_points():
    assert mv.compass(0.0) == "E"
    assert mv.compass(np.pi / 2) == "N"
    assert mv.compass(-np.pi / 2) == "S"
    assert mv.compass(3 * np.pi / 4) == "NW"



# ---------------------------------------------------------------------------
# Training again as entries arrive
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def register():
    return simulate(groups=6, days=200, seed=7, persistence=0.6, home=0.4)


def _same(a: mv.TrackRecord, b: mv.TrackRecord) -> None:
    assert (a.tested, a.forecast_hits, a.stay_hits, a.discordant, a.circle_hits) == \
        (b.tested, b.forecast_hits, b.stay_hits, b.discordant, b.circle_hits)
    assert (a.field_reports, a.field_hits, a.field_circle_hits) == \
        (b.field_reports, b.field_hits, b.field_circle_hits)
    assert a.config == b.config
    assert a.masses == pytest.approx(b.masses)
    assert a.mean_area_km2 == pytest.approx(b.mean_area_km2)


def test_new_entries_retrain_only_the_months_they_touch(register):
    """A register that gained a fortnight of entries is trained again from
    the month they fall in, and comes out exactly as a fresh training."""
    cut = register["Date"].max() - pd.Timedelta(days=14)
    earlier = mv.train(register[register["Date"] <= cut], 1)
    updated = mv.train(register, 1, previous=earlier)
    fresh = mv.train(register, 1)
    assert updated.reused_months >= len(earlier.months) - 2
    assert 0 < updated.retrained_months <= 3
    _same(updated.record, fresh.record)


def test_an_edited_old_entry_retrains_from_its_month(register):
    trained = mv.train(register, 1)
    edited = register.copy()
    row = edited.index[edited["Date"] == START + pd.Timedelta(days=120)][0]
    edited.loc[row, "Latitude"] += 0.05
    again = mv.train(edited, 1, previous=trained)
    month = str(edited.loc[row, "Date"].to_period("M"))
    first_redone = [m.month for m in again.months][again.reused_months]
    assert first_redone <= month
    _same(again.record, mv.train(edited, 1).record)


def test_an_unchanged_register_is_not_trained_again(register):
    trained = mv.train(register, 1)
    again = mv.train(register, 1, previous=trained)
    assert again.retrained_months == 0
    assert again.data_key == trained.data_key


def test_a_training_for_another_horizon_is_not_reused(register):
    one_day = mv.train(register, 1)
    three = mv.train(register, 3, previous=one_day)
    assert three.reused_months == 0


def test_reports_far_away_do_not_change_months_already_scored(register):
    """A new area joining the register widens the grid; the cells and the
    usage maps around existing groups must not move with it."""
    trained = mv.train(register, 1)
    far = register.tail(30).copy()
    far["Latitude"] += 2.0
    far["Date"] = register["Date"].max() + pd.Timedelta(days=40)
    grown = pd.concat([register, far], ignore_index=True)
    again = mv.train(grown, 1, previous=trained)
    old_months = {m.month: m for m in trained.months}
    for month in again.months[:again.reused_months]:
        assert month is old_months[month.month]
    _same(mv.train(grown, 1).record, again.record)
