"""Tests for the land-cover features.

Built on synthetic rasters rather than the real tiles: 450 MB of
downloads is not a test dependency, and a hand-made landscape is the
only way to know what the right answer is.
"""

import numpy as np
import pandas as pd
import pytest

from research.landcover import (
    BUILT_CLASS,
    CROP_CLASS,
    FOREST_CLASS,
    STANDING_CROP,
    LandCover,
    cropping_features,
    tiles_for_bounds,
)

# 0.0001 degrees is roughly 11 m, close enough to WorldCover's 10 m that
# buffer sizes in these tests are the same order as the real thing.
STEP = 0.0001
NORTH, WEST = 23.0, 81.0
SIZE = 400


def _cover(array):
    return LandCover(array, WEST, NORTH, STEP, -STEP)


def _centre():
    """Lat/lon of the array's middle cell."""
    return NORTH - (SIZE // 2) * STEP, WEST + (SIZE // 2) * STEP


def _uniform(code):
    return _cover(np.full((SIZE, SIZE), code, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Tile addressing
# ---------------------------------------------------------------------------
def test_tile_name_for_a_point_in_central_india():
    assert tiles_for_bounds(81.5, 23.2, 81.6, 23.3) == ["N21E081"]


def test_bounding_box_spanning_a_tile_seam_returns_both():
    """The study landscape straddles 81 E and 24 N."""
    tiles = tiles_for_bounds(80.8, 22.9, 81.9, 24.3)
    assert set(tiles) == {"N21E078", "N21E081", "N24E078", "N24E081"}


def test_southern_and_western_hemispheres_are_addressed():
    assert tiles_for_bounds(-58.5, -34.5, -58.4, -34.4) == ["S36W060"]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def test_uniform_forest_reads_as_all_forest():
    stats = _uniform(FOREST_CLASS).stats(*_centre())
    assert stats["tree_frac_1km"] == pytest.approx(1.0)
    assert stats["crop_frac_1km"] == pytest.approx(0.0)


def test_half_forest_half_crop_splits_evenly():
    array = np.full((SIZE, SIZE), CROP_CLASS, dtype=np.uint8)
    array[:, : SIZE // 2] = FOREST_CLASS
    stats = _cover(array).stats(*_centre())
    assert stats["tree_frac_1km"] == pytest.approx(0.5, abs=0.02)
    assert stats["crop_frac_1km"] == pytest.approx(0.5, abs=0.02)


def test_buffers_are_circular_not_square():
    """A square buffer would over-count the corners by 4/pi."""
    cover = _uniform(FOREST_CLASS)
    _, mask = cover.window(*_centre(), 1.0)
    assert mask.sum() / mask.size == pytest.approx(np.pi / 4, abs=0.01)


def test_fractions_sum_to_one_across_classes():
    rng = np.random.default_rng(0)
    array = rng.choice([FOREST_CLASS, CROP_CLASS, BUILT_CLASS, 30],
                       size=(SIZE, SIZE)).astype(np.uint8)
    stats = _cover(array).stats(*_centre())
    total = sum(stats[f"{name}_frac_2km"]
                for name in ("tree", "crop", "built", "grass", "shrub", "water"))
    assert total == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# Configuration -- the part that carries the hypothesis
# ---------------------------------------------------------------------------
def test_interface_marks_only_cropland_that_touches_forest():
    """Cropland far from cover is not a raiding surface."""
    array = np.full((SIZE, SIZE), CROP_CLASS, dtype=np.uint8)
    array[:, :100] = FOREST_CLASS
    interface = _cover(array).crop_forest_interface

    assert not interface[:, :100].any(), "forest itself is not crop"
    assert interface[:, 100:105].any(), "crop beside forest is interface"
    assert not interface[:, 150:].any(), "crop far from forest is not"


def test_shredded_forest_has_more_edge_than_a_solid_block():
    """Same forest fraction, different configuration."""
    solid = np.full((SIZE, SIZE), CROP_CLASS, dtype=np.uint8)
    solid[:, : SIZE // 2] = FOREST_CLASS

    striped = np.full((SIZE, SIZE), CROP_CLASS, dtype=np.uint8)
    striped[:, ::2] = FOREST_CLASS

    centre = _centre()
    solid_stats = _cover(solid).stats(*centre)
    striped_stats = _cover(striped).stats(*centre)

    assert solid_stats["tree_frac_1km"] == pytest.approx(
        striped_stats["tree_frac_1km"], abs=0.02)
    assert (striped_stats["forest_edge_density_1km"]
            > 10 * solid_stats["forest_edge_density_1km"])


def test_distance_to_forest_is_zero_when_standing_in_it():
    assert _uniform(FOREST_CLASS).stats(*_centre())["dist_forest_km"] == 0.0


def test_distance_saturates_when_the_class_is_absent():
    """No forest anywhere must not read as forest at zero metres."""
    stats = _uniform(CROP_CLASS).stats(*_centre())
    assert stats["dist_forest_km"] == 10.0


def test_a_point_outside_the_raster_does_not_raise():
    stats = _uniform(FOREST_CLASS).stats(0.0, 0.0)
    assert stats["tree_frac_1km"] == 0.0


# ---------------------------------------------------------------------------
# Cropping calendar
# ---------------------------------------------------------------------------
def test_calendar_covers_every_month():
    assert sorted(STANDING_CROP) == list(range(1, 13))
    assert all(0.0 <= v <= 1.0 for v in STANDING_CROP.values())


def test_bare_season_scores_below_standing_rabi():
    """May stubble is not February wheat."""
    assert STANDING_CROP[5] < STANDING_CROP[2]


def test_exposed_cropland_needs_both_hectares_and_a_standing_crop():
    month = pd.Series([2, 5, 2, 5])
    crop = pd.Series([0.8, 0.8, 0.0, 0.0])
    edge = pd.Series([0.2, 0.2, 0.0, 0.0])
    out = cropping_features(month, crop, edge)

    assert out["exposed_cropland"].iloc[0] > out["exposed_cropland"].iloc[1], \
        "same fields, bare season, less exposure"
    assert out["exposed_cropland"].iloc[2] == 0.0, "no cropland, no exposure"
    assert out["is_rabi"].tolist() == [1, 0, 1, 0]
    assert out["is_kharif"].tolist() == [0, 0, 0, 0]
