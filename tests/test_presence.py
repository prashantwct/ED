"""The presence model on landscapes whose truth is known.

Each test fixes one property the page relies on: duplicates are one
record, the effort-corrected surface sees past observer bias, a test
fortnight's forecast never sees the fortnights after it, the forecast
follows a moving herd and a seasonal shift, and it says when it is no
better than a simple map.
"""

import dataclasses
import io

import numpy as np
import pandas as pd
import pytest

from core import presence as pr
from presence_fixtures import START, simulate, to_xy, truth, village_points

TWO_SPOTS = ((-20.0, 0.0, 4.0, 1.0), (20.0, 0.0, 4.0, 1.0))


@pytest.fixture(scope="module")
def villages():
    return village_points()


@pytest.fixture(scope="module")
def biased(villages):
    """Two equal hotspots; villages crowd the eastern one, and people
    report less the farther they are from a village."""
    df = simulate(days=560, seed=0, villages=villages, hotspots=TWO_SPOTS)
    inputs = pr.prepare(df, df["Date"].max(), villages=villages, habitat=None)
    return df, inputs, pr.run(inputs)


def _west_share(inputs, share):
    x, _ = to_xy(inputs.lattice.lat, inputs.lattice.lon)
    return share[x < 0].sum()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def test_reports_of_one_place_on_one_day_are_one_record():
    lattice = pr.lattice_for([23.3], [81.0])
    cell = lattice.locate([23.3], [81.0])[0]
    lat, lon = lattice.lat[cell], lattice.lon[cell]
    day = pd.Timestamp("2025-03-01 08:00")
    df = pd.DataFrame({
        "Date": [day, day + pd.Timedelta(hours=5), day + pd.Timedelta(days=1)],
        "Latitude": [lat, lat + 0.001, lat], "Longitude": [lon, lon + 0.001, lon],
    })
    thinned = pr.thin(df, lattice)
    assert len(thinned) == 2
    assert thinned["Day"].nunique() == 2


def test_every_report_falls_in_a_cell_and_far_cells_are_left_out():
    lat = np.array([23.0, 23.0, 23.5])
    lon = np.array([81.0, 81.2, 81.0])
    lattice = pr.lattice_for(lat, lon, cell_km=2.0, domain_km=5.0)
    assert (lattice.locate(lat, lon) >= 0).all()
    x, y = lattice.xy(lat, lon)
    nearest = np.min(np.hypot(lattice.x[:, None] - x, lattice.y[:, None] - y), axis=1)
    inside = pr._inside_hull(np.column_stack([x, y]), np.column_stack([lattice.x, lattice.y]))
    assert np.all(inside | (nearest <= 5.0 + 2.0))
    assert lattice.locate(np.array([25.0]), np.array([81.0]))[0] == -1


def test_the_land_between_two_clusters_of_reports_is_kept():
    """Two clusters 60 km apart: a corridor between them must have cells
    to run through, though nobody reported there."""
    lat = np.array([23.0, 23.02, 23.0, 23.02])
    lon = np.array([81.0, 81.0, 81.6, 81.6])
    lattice = pr.lattice_for(lat, lon, cell_km=2.0, domain_km=5.0)
    assert lattice.locate(np.array([23.01]), np.array([81.3]))[0] >= 0


def test_a_large_landscape_coarsens_the_grid():
    lat = np.array([22.0, 25.0])
    lon = np.array([79.0, 83.0])
    lattice = pr.lattice_for(lat, lon, cell_km=1.0, domain_km=60.0, max_cells=2000)
    assert lattice.size <= 2000
    assert lattice.cell_km > 1.0


def test_the_last_fortnight_ends_on_the_period_end():
    as_of = pd.Timestamp("2025-06-30")
    blocks = pr.blocks_for(pd.Timestamp("2025-01-01"), as_of)
    index = blocks.index(pd.Series(pd.to_datetime(
        ["2025-06-30", "2025-06-17", "2025-06-16", "2025-01-01"])))
    assert list(index[:3]) == [blocks.count - 1, blocks.count - 1, blocks.count - 2]
    assert index[3] == 0
    assert blocks.start(blocks.count - 1) == pd.Timestamp("2025-06-17")


def test_the_counts_hold_every_record_once(biased):
    _, inputs, _ = biased
    assert inputs.counts.sum() == inputs.records
    assert inputs.records < inputs.reports  # the duplicates went


# ---------------------------------------------------------------------------
# Observer bias
# ---------------------------------------------------------------------------
def test_the_corrected_surface_sees_the_remote_hotspot(biased):
    """The two hotspots are equal, but the western one is far from the
    villages and draws few reports. Reports alone put a small share of use
    there; corrected for distance from villages, the share comes back
    close to half."""
    _, inputs, result = biased
    x, y = to_xy(inputs.lattice.lat, inputs.lattice.lon)
    true = truth(x, y, TWO_SPOTS)
    true = true / true.sum()
    reported = _west_share(inputs, pr.baseline_past(inputs, inputs.blocks.count))
    corrected = _west_share(inputs, result.use(corrected=True))
    assert reported < 0.25
    assert abs(corrected - _west_share(inputs, true)) < 0.12
    assert np.corrcoef(result.use(corrected=True), true)[0, 1] > \
        np.corrcoef(pr.baseline_past(inputs, inputs.blocks.count), true)[0, 1] + 0.05


def test_fewer_reports_far_from_villages_is_read_as_effort(biased):
    _, inputs, result = biased
    effects = result.effects().set_index("Term")
    village = effects.loc[pr.LABELS[pr.VILLAGE_KM]]
    assert village["Kind"] == "Reporting effort"
    assert village["Ratio"] < 1.0


def test_with_no_effort_term_correcting_changes_nothing(villages):
    df = simulate(days=300, seed=1, villages=villages, hotspots=TWO_SPOTS)
    inputs = pr.prepare(df, df["Date"].max(), villages=villages, habitat=None,
                        village_effort=False)
    result = pr.run(inputs)
    assert inputs.effort == []
    assert np.allclose(result.use(corrected=True), result.use(corrected=False))


def test_effort_points_are_a_covariate_of_their_own(villages):
    df = simulate(days=300, seed=1, villages=villages, hotspots=TWO_SPOTS)
    rng = np.random.default_rng(0)
    pings = villages.sample(200, replace=True, random_state=1)[["Latitude", "Longitude"]]
    pings = pings + rng.normal(0, 0.01, pings.shape)
    inputs = pr.prepare(df, df["Date"].max(), villages=villages, habitat=None,
                        effort_layers={"Uploaded effort": pings})
    assert "Uploaded effort" in inputs.names and "Uploaded effort" in inputs.effort


# ---------------------------------------------------------------------------
# Tested forecasts
# ---------------------------------------------------------------------------
def test_a_tested_fortnight_never_sees_the_fortnights_after_it(biased):
    """Doubling the records of the last fortnight must not change any
    earlier fortnight's forecast or the version chosen for it."""
    _, inputs, result = biased
    counts = inputs.counts.copy()
    counts[-1] *= 2
    changed = dataclasses.replace(
        inputs, counts=counts,
        dynamic=pr._dynamic(counts, inputs.near, inputs.lattice.kernel(pr.WIDE_KM)))
    record, _ = pr.validate(changed)
    before = {t.row: t for t in result.record.tests}
    for test in record.tests[:-1]:
        assert test.spec == before[test.row].spec
        assert test.log_score[pr.MODEL] == pytest.approx(
            before[test.row].log_score[pr.MODEL], abs=1e-9)


def test_the_forecast_follows_a_moving_herd(villages):
    """A hotspot drifting east: all past reports point at where it was, last
    fortnight's at where it is. The model keeps up with the latter."""
    df = simulate(days=560, seed=2, villages=villages, detect_km=None, per_day=6,
                  drift_km_per_day=0.12, wrap_km=35,
                  hotspots=((-30.0, 0.0, 3.0, 1.0), (20.0, 15.0, 4.0, 0.5)))
    inputs = pr.prepare(df, df["Date"].max(), villages=villages, habitat=None)
    record = pr.run(inputs).record
    assert record.tested >= pr.MIN_TESTED
    assert record.spec.dynamic
    assert record.bits() > record.bits("All past reports") + 1.0
    assert record.bits() > record.bits("Last fortnight's reports") - 0.1
    assert record.half_km2() < record.half_km2("All past reports") / 3


def test_the_forecast_beats_simple_maps_when_use_shifts_with_the_season(villages):
    df = simulate(days=730, seed=4, villages=villages, detect_km=None, per_day=5,
                  seasonal_shift_km=18,
                  hotspots=((-20.0, -9.0, 3.0, 1.0), (20.0, 0.0, 4.0, 0.6)))
    inputs = pr.prepare(df, df["Date"].max(), villages=villages, habitat=None)
    result = pr.run(inputs)
    assert inputs.seasonal
    assert result.record.beats_baselines
    assert len(result.seasons()) == 1 + len(pr.SEASONS)
    # The monsoon surface sits north of the winter one, as the herd does.
    _, y = to_xy(inputs.lattice.lat, inputs.lattice.lon)
    west = to_xy(inputs.lattice.lat, inputs.lattice.lon)[0] < 0
    monsoon = result.use("Monsoon (Jun-Sep)", corrected=False)
    winter = result.use("Winter (Oct-Feb)", corrected=False)
    assert (monsoon[west] * y[west]).sum() / monsoon[west].sum() > \
        (winter[west] * y[west]).sum() / winter[west].sum() + 5


def test_it_says_when_it_is_no_better_than_a_simple_map(biased):
    """Where use never changes, a map of all past reports is about as good
    as forecasting gets: the record must not claim more."""
    _, _, result = biased
    record = result.record
    assert abs(record.bits() - record.bits("All past reports")) < 0.15


def test_too_little_history_fits_nothing(villages):
    df = simulate(days=40, seed=5, villages=villages, per_day=1)
    inputs = pr.prepare(df, df["Date"].max(), villages=villages, habitat=None)
    result = pr.run(inputs)
    assert not result.fitted and result.record.tested == 0
    share = result.next_fortnight()
    assert share.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scores and bands
# ---------------------------------------------------------------------------
def test_ties_favour_no_one():
    share = np.full(10, 0.1)
    counts = np.array([4, 0, 0, 0, 0, 0, 0, 0, 0, 6], dtype=float)
    assert pr.capture(share, counts, 0.1) == pytest.approx(0.1)
    assert pr.area_for(share, counts, 0.5) == pytest.approx(5.0)
    sharp = np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5]) + 1e-6
    assert pr.area_for(sharp, counts, 0.5) <= 2.0


def test_bands_cover_a_quarter_of_the_landscape():
    share = np.random.default_rng(0).random(1000)
    band = pr.bands(share)
    assert (band == "Top 5%").sum() == 50
    assert (band == "Next 5%").sum() == 50
    assert (band == "Next 15%").sum() == 150
    assert share[band == "Top 5%"].min() > share[band == "Next 5%"].max()


# ---------------------------------------------------------------------------
# What leaves the model
# ---------------------------------------------------------------------------
def test_the_table_for_r_is_cells_by_fortnights(biased):
    _, inputs, result = biased
    table = result.model_table()
    t, c = inputs.counts.shape
    assert len(table) == t * c
    assert table["records"].sum() == inputs.records
    assert {"cell", "fortnight", "fortnight_start", "area_km2", pr.VILLAGE_KM,
            f"{pr.VILLAGE_KM}_is_effort", "lag_recent"} <= set(table.columns)


def test_effort_files_keep_only_coordinates():
    raw = io.StringIO("Name,Mobile,lat,LNG\nRam,9999999999,24.12345,81.98765\n"
                      "Sita,8888888888,,81.1\n")
    points = pr.read_effort(raw)
    assert list(points.columns) == ["Latitude", "Longitude"]
    assert len(points) == 1
    assert points.iloc[0]["Latitude"] == 24.123
    with pytest.raises(ValueError):
        pr.read_effort(io.StringIO("a,b\n1,2\n"))


def test_places_name_the_villages_around_the_most_expected(biased, villages):
    _, inputs, result = biased
    table = pr.places(inputs.lattice, result.next_fortnight(), villages)
    assert len(table) and set(table["Village"]) <= set(villages["Village"])
    assert table["Share"].is_monotonic_decreasing
