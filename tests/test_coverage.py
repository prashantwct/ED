"""Tests for early-warning coverage.

Two properties carry the weight here.

The first is disclosure. These two exports are the only personal data
the app ever touches -- names, mobile numbers, email addresses and home
coordinates -- and the whole design rests on the claim that none of it
survives the loader. That claim is worth a test that fails loudly rather
than a comment saying it is true.

The second is placement. A count of registered contacts per village is
only useful if people land on the right village, and the two files
transliterate names differently, so the rules deciding that are tested
directly rather than through the table they produce.
"""

import io

import numpy as np
import pandas as pd
import pytest

from core.coverage import (
    COVERAGE_NONE,
    COVERAGE_OK,
    COVERAGE_THIN,
    UNASSIGNED,
    assign_to_villages,
    coverage_caveats,
    coverage_gaps,
    coverage_headlines,
    division_labels,
    division_staffing,
    enrolment_by_division,
    load_ews_registry,
    load_staff_roster,
    village_coverage,
)
from core.exceptions import DataValidationError

# 2 km east of 23.10 N is about 0.0195 degrees of longitude.
LAT, LON = 23.10, 81.75
KM_LON = 0.00975  # ~1 km at this latitude


def _csv(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode("utf-8"))


REGISTRY_CSV = (
    "Name,Mobile,Latitude,Longitude,Village,Division\n"
    "Ramesh Kumar,9876543210,23.1000,81.7500,Kusumhai,Anuppur\n"
    "Sita Bai,9876543211,23.1001,81.7501,Kusumhai,Anuppur\n"
    "Mohan Lal,9876543212,23.2000,81.8500,Tirchuli,Umaria\n"
)

ROSTER_CSV = (
    "Name,Mobile,Email,Role,Division,Range,Beat,Sub Beat,Latitude,Longitude\n"
    "A Ranger,9000000001,a@example.com,Range Mgr,Anuppur,-,-,-,23.10,81.75\n"
    "B Ranger,9000000002,b@example.com,Range Mgr,Bandhavgarh TR,Pali,-,-,23.20,81.85\n"
    "C Ranger,9000000003,c@example.com,Div Mgr,-,-,-,-,,\n"
)


def _villages(*rows):
    return pd.DataFrame(
        [{"Village": v, "Latitude": lat, "Longitude": lon} for v, lat, lon in rows]
    )


def _risk(*rows):
    """Minimal village_risk frame: (village, lat, lon, tier, events, deaths)."""
    return pd.DataFrame([
        {
            "Village": v, "Latitude": lat, "Longitude": lon, "Tier": tier,
            "Conflict Events": events, "Human Deaths": float(deaths),
            "People Injured": 0.0,
        }
        for v, lat, lon, tier, events, deaths in rows
    ])


# ---------------------------------------------------------------------------
# Disclosure
# ---------------------------------------------------------------------------
def test_the_registry_loader_drops_names_and_numbers():
    registry, _ = load_ews_registry(_csv(REGISTRY_CSV))
    assert "Name" not in registry.columns
    assert "Mobile" not in registry.columns
    assert set(registry.columns) == {"Latitude", "Longitude", "Village", "Division"}


def test_the_roster_loader_drops_names_numbers_and_email():
    roster, _ = load_staff_roster(_csv(ROSTER_CSV))
    for column in ("Name", "Mobile", "Email"):
        assert column not in roster.columns


def test_no_identifying_value_survives_anywhere_in_the_frame():
    """Not just the column headers -- the values themselves."""
    registry, _ = load_ews_registry(_csv(REGISTRY_CSV))
    rendered = registry.to_csv(index=False)
    for leaked in ("Ramesh", "Sita", "9876543210", "9876543212"):
        assert leaked not in rendered

    roster, _ = load_staff_roster(_csv(ROSTER_CSV))
    rendered = roster.to_csv(index=False)
    for leaked in ("Ranger", "9000000001", "example.com"):
        assert leaked not in rendered


def test_an_unanticipated_identifying_column_is_excluded_by_default():
    """The keep list is an allowlist, so a new column needs no new rule."""
    registry, _ = load_ews_registry(
        _csv(
            "Name,Mobile,Aadhaar,Latitude,Longitude,Village,Division\n"
            "R,9876543210,123456789012,23.1,81.75,Kusumhai,Anuppur\n"
        )
    )
    assert "Aadhaar" not in registry.columns
    assert "123456789012" not in registry.to_csv(index=False)


def test_coordinates_are_coarsened_to_about_a_hundred_metres():
    registry, _ = load_ews_registry(
        _csv(
            "Name,Mobile,Latitude,Longitude,Village,Division\n"
            "R,9876543210,23.1234567,81.7654321,K,Anuppur\n"
        )
    )
    assert registry.loc[0, "Latitude"] == pytest.approx(23.123)
    assert registry.loc[0, "Longitude"] == pytest.approx(81.765)


def test_the_roster_ignores_beat_columns_it_cannot_trust():
    roster, _ = load_staff_roster(_csv(ROSTER_CSV))
    assert "Beat" not in roster.columns
    assert "Sub Beat" not in roster.columns


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_a_registry_without_coordinates_is_rejected():
    with pytest.raises(DataValidationError, match="Latitude"):
        load_ews_registry(_csv("Name,Mobile,Village\nR,9876543210,K\n"))


def test_a_roster_without_a_division_is_rejected():
    with pytest.raises(DataValidationError, match="Division"):
        load_staff_roster(_csv("Name,Mobile,Email\nR,9,r@e.com\n"))


def test_unusable_registry_coordinates_are_dropped_and_reported():
    registry, warnings = load_ews_registry(
        _csv(
            "Name,Mobile,Latitude,Longitude,Village,Division\n"
            "A,9,23.1,81.75,K,Anuppur\n"
            "B,9,,,K,Anuppur\n"
            "C,9,999,81.75,K,Anuppur\n"
        )
    )
    assert len(registry) == 1
    assert any("2" in w for w in warnings)


def test_a_registry_with_no_usable_row_raises_rather_than_reporting_zero():
    with pytest.raises(DataValidationError, match="nobody can be placed"):
        load_ews_registry(
            _csv("Name,Mobile,Latitude,Longitude\nA,9,,\nB,9,,\n")
        )


def test_placeholder_labels_become_missing_rather_than_a_division_called_dash():
    roster, warnings = load_staff_roster(_csv(ROSTER_CSV))
    assert roster["Division"].isna().sum() == 1
    assert any("no division" in w for w in warnings)
    assert (
        division_staffing(pd.DataFrame(), roster)
        .set_index("Division")
        .loc[UNASSIGNED, "Registered Users"]
        == 1
    )


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------
def test_placement_is_exclusive_so_counts_can_be_audited():
    villages = _villages(("A", LAT, LON), ("B", LAT, LON + 3 * KM_LON))
    registry = pd.DataFrame({
        "Latitude": [LAT] * 5,
        "Longitude": [LON, LON, LON, LON + 3 * KM_LON, LON + 3 * KM_LON],
        "Village": ["A"] * 5,
    })
    counts, stats = assign_to_villages(registry, villages, radius_km=2.0)
    assert counts.sum() == stats["matched"] == 5
    assert stats["unmatched"] == 0


def test_a_registrant_beyond_the_radius_counts_towards_nobody():
    villages = _villages(("A", LAT, LON))
    registry = pd.DataFrame({
        "Latitude": [LAT, LAT],
        "Longitude": [LON, LON + 10 * KM_LON],
        "Village": ["A", "A"],
    })
    counts, stats = assign_to_villages(registry, villages, radius_km=2.0)
    assert counts.tolist() == [1, ]
    assert stats["unmatched"] == 1


def test_a_matching_name_inside_the_radius_beats_a_closer_stranger():
    """The two files transliterate differently; where they agree, trust it."""
    villages = _villages(("Bhamhani", LAT, LON + 1.5 * KM_LON), ("Kusumhai", LAT, LON))
    registry = pd.DataFrame({
        "Latitude": [LAT],
        "Longitude": [LON + 1.4 * KM_LON],  # nearest is Bhamhani
        "Village": ["Kusumhai"],
    })
    counts, stats = assign_to_villages(registry, villages, radius_km=2.0)
    assert counts.tolist() == [0, 1]
    assert stats["by_name"] == 1


def test_a_name_match_outside_the_radius_does_not_pull_a_registrant_across():
    villages = _villages(("Bhamhani", LAT, LON), ("Kusumhai", LAT, LON + 8 * KM_LON))
    registry = pd.DataFrame(
        {"Latitude": [LAT], "Longitude": [LON], "Village": ["Kusumhai"]}
    )
    counts, stats = assign_to_villages(registry, villages, radius_km=2.0)
    assert counts.tolist() == [1, 0]
    assert stats["by_name"] == 0


def test_placement_falls_back_to_distance_when_names_do_not_agree():
    villages = _villages(("Bhamhani", LAT, LON))
    registry = pd.DataFrame(
        {"Latitude": [LAT], "Longitude": [LON], "Village": ["Bamhani"]}
    )
    counts, _ = assign_to_villages(registry, villages, radius_km=2.0)
    assert counts.tolist() == [1]


def test_a_registry_with_no_village_column_still_places_people():
    villages = _villages(("A", LAT, LON))
    registry = pd.DataFrame({"Latitude": [LAT], "Longitude": [LON]})
    counts, stats = assign_to_villages(registry, villages, radius_km=2.0)
    assert counts.tolist() == [1] and stats["by_name"] == 0


# ---------------------------------------------------------------------------
# The coverage table
# ---------------------------------------------------------------------------
def test_two_villages_sharing_a_name_are_not_pooled():
    """The centroid loader keeps them apart deliberately; so must this."""
    far_lon = LON + 50 * KM_LON
    villages = _villages(("Amdari", LAT, LON), ("Amdari", LAT, far_lon))
    registry = pd.DataFrame({
        "Latitude": [LAT] * 3,
        "Longitude": [LON, LON, far_lon],
        "Village": ["Amdari"] * 3,
    })
    risk = _risk(
        ("Amdari", LAT, LON, "Critical", 4, 1),
        ("Amdari", LAT, far_lon, "High", 2, 0),
    )
    table, _ = village_coverage(risk, villages, registry, radius_km=2.0)
    assert table["Registered Contacts"].tolist() == [2, 1]


@pytest.mark.parametrize(
    "registered, expected",
    [(0, COVERAGE_NONE), (1, COVERAGE_THIN), (2, COVERAGE_THIN), (3, COVERAGE_OK)],
)
def test_the_coverage_label_turns_over_at_the_configured_minimum(registered, expected):
    villages = _villages(("A", LAT, LON))
    registry = pd.DataFrame({
        "Latitude": [LAT] * registered,
        "Longitude": [LON] * registered,
        "Village": ["A"] * registered,
    })
    risk = _risk(("A", LAT, LON, "Critical", 3, 1))
    table, summary = village_coverage(
        risk, villages, registry, radius_km=2.0, min_contacts=3
    )
    if registered == 0:
        # An empty registry is a loading failure, not a coverage reading.
        assert table.empty
        return
    assert table.loc[0, "Coverage"] == expected


def test_a_village_nobody_registered_in_reads_as_no_contact():
    villages = _villages(("A", LAT, LON), ("B", LAT, LON + 50 * KM_LON))
    registry = pd.DataFrame(
        {"Latitude": [LAT], "Longitude": [LON], "Village": ["A"]}
    )
    risk = _risk(("B", LAT, LON + 50 * KM_LON, "Critical", 6, 1))
    table, summary = village_coverage(risk, villages, registry, radius_km=2.0)
    assert table.loc[0, "Coverage"] == COVERAGE_NONE
    assert summary["no_contact"] == 1


def test_registrants_are_placed_against_every_centroid_not_only_exposed_ones():
    """Otherwise someone enrolled beside a quiet village is credited to a risky one."""
    quiet_lon = LON + 1.2 * KM_LON
    villages = _villages(("Risky", LAT, LON), ("Quiet", LAT, quiet_lon))
    registry = pd.DataFrame(
        {"Latitude": [LAT], "Longitude": [quiet_lon], "Village": ["Quiet"]}
    )
    risk = _risk(("Risky", LAT, LON, "Critical", 5, 1))
    table, _ = village_coverage(risk, villages, registry, radius_km=2.0)
    assert table.loc[0, "Registered Contacts"] == 0


def test_the_gap_list_is_the_worst_tiers_only_and_worst_first():
    villages = _villages(
        ("Crit", LAT, LON),
        ("High", LAT, LON + 50 * KM_LON),
        ("Watch", LAT, LON + 100 * KM_LON),
    )
    registry = pd.DataFrame(
        {"Latitude": [LAT], "Longitude": [LON + 200 * KM_LON], "Village": ["Far"]}
    )
    risk = _risk(
        ("High", LAT, LON + 50 * KM_LON, "High", 9, 0),
        ("Crit", LAT, LON, "Critical", 2, 1),
        ("Watch", LAT, LON + 100 * KM_LON, "Watch", 7, 0),
    )
    table, _ = village_coverage(risk, villages, registry, radius_km=2.0)
    gaps = coverage_gaps(table)
    assert gaps["Village"].tolist() == ["Crit", "High"]


def test_no_contact_outranks_thin_within_a_tier():
    lon_b = LON + 50 * KM_LON
    villages = _villages(("Thin", LAT, LON), ("Empty", LAT, lon_b))
    registry = pd.DataFrame(
        {"Latitude": [LAT], "Longitude": [LON], "Village": ["Thin"]}
    )
    risk = _risk(
        ("Thin", LAT, LON, "High", 20, 0),
        ("Empty", LAT, lon_b, "High", 1, 0),
    )
    table, _ = village_coverage(risk, villages, registry, radius_km=2.0)
    assert coverage_gaps(table)["Village"].tolist() == ["Empty", "Thin"]


def test_a_fully_covered_landscape_produces_no_gap_list():
    villages = _villages(("A", LAT, LON))
    registry = pd.DataFrame({
        "Latitude": [LAT] * 4, "Longitude": [LON] * 4, "Village": ["A"] * 4,
    })
    risk = _risk(("A", LAT, LON, "Critical", 3, 1))
    table, summary = village_coverage(risk, villages, registry, radius_km=2.0)
    assert coverage_gaps(table).empty
    assert summary["urgent"] == 0
    assert "at least" in " ".join(coverage_headlines(table, summary))


# ---------------------------------------------------------------------------
# Divisions
# ---------------------------------------------------------------------------
def test_the_two_spellings_of_a_reserve_division_are_one_row():
    """The sighting loader title-cases; the roster does not."""
    sightings = pd.DataFrame({
        "Division": ["Bandhavgarh Tr"] * 3,
        "Latitude": [LAT] * 3, "Longitude": [LON] * 3,
    })
    roster = pd.DataFrame({"Division": ["Bandhavgarh TR", "Bandhavgarh TR"]})
    table = division_staffing(sightings, roster)
    assert len(table) == 1
    assert table.loc[0, "Registered Users"] == 2
    assert table.loc[0, "Reports"] == 3
    # And it is shown the way the rest of the app shows it.
    assert table.loc[0, "Division"] == "Bandhavgarh Tr"


def test_a_division_reporting_with_nobody_registered_is_not_a_divide_by_zero():
    sightings = pd.DataFrame({
        "Division": ["Satna"], "Latitude": [LAT], "Longitude": [LON],
    })
    roster = pd.DataFrame({"Division": ["Anuppur"]})
    table = division_staffing(sightings, roster).set_index("Division")
    assert np.isnan(table.loc["Satna", "Reports per User"])
    assert table.loc["Satna", "Registered Users"] == 0
    assert table.loc["Anuppur", "Reports"] == 0


def test_enrolment_by_division_counts_people_and_villages():
    registry = pd.DataFrame({
        "Latitude": [LAT] * 3, "Longitude": [LON] * 3,
        "Village": ["A", "A", "B"],
        "Division": ["Anuppur", "Anuppur", "Umaria"],
    })
    table = enrolment_by_division(registry).set_index("Division")
    assert table.loc["Anuppur", "Registered"] == 2
    assert table.loc["Anuppur", "Villages"] == 1
    assert table.loc["Umaria", "Registered"] == 1


def test_division_labels_prefers_the_sighting_export_spelling():
    sightings = pd.DataFrame({"Division": ["Bandhavgarh Tr", "Anuppur"]})
    labels = division_labels(sightings)
    assert labels["bandhavgarhtr"] == "Bandhavgarh Tr"
    registry = pd.DataFrame({
        "Latitude": [LAT], "Longitude": [LON],
        "Village": ["A"], "Division": ["Bandhavgarh TR"],
    })
    assert enrolment_by_division(registry, labels).loc[0, "Division"] == "Bandhavgarh Tr"


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------
def test_every_entry_point_survives_empty_input():
    empty = pd.DataFrame()
    villages = _villages(("A", LAT, LON))
    risk = _risk(("A", LAT, LON, "Critical", 1, 1))

    table, summary = village_coverage(risk, villages, empty)
    assert table.empty and summary["exposed"] == 0
    assert coverage_gaps(table).empty
    assert coverage_headlines(table, summary) == []
    assert coverage_caveats(summary)  # still explains itself

    table, _ = village_coverage(pd.DataFrame(), villages, empty)
    assert table.empty
    assert enrolment_by_division(empty).empty
    assert division_staffing(empty, empty).empty


def test_coverage_without_centroids_returns_nothing_rather_than_guessing():
    registry = pd.DataFrame(
        {"Latitude": [LAT], "Longitude": [LON], "Village": ["A"]}
    )
    risk = _risk(("A", LAT, LON, "Critical", 1, 1))
    table, _ = village_coverage(risk, None, registry)
    assert table.empty


def test_the_original_risk_columns_are_preserved():
    villages = _villages(("A", LAT, LON))
    registry = pd.DataFrame(
        {"Latitude": [LAT], "Longitude": [LON], "Village": ["A"]}
    )
    risk = _risk(("A", LAT, LON, "Critical", 1, 1))
    table, _ = village_coverage(risk, villages, registry)
    assert set(risk.columns).issubset(table.columns)
    assert list(table.columns)[-2:] == ["Registered Contacts", "Coverage"]


def test_the_caveats_name_the_registrants_nobody_could_place():
    summary = {"registrants": 100, "unmatched": 7, "radius_km": 2.0, "by_name": 3}
    joined = " ".join(coverage_caveats(summary))
    assert "7 of 100" in joined
    assert "not a reachable person" in joined
