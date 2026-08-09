"""Tests for CSV ingestion, focused on the date-parsing regression.

Inferred date parsing on a day-first export both mangles dates and
silently deletes rows: pandas locks onto the layout implied by the first
value, so ``03/04/2026`` becomes 4 March and the later ``21/04/2026``
fails to parse and is dropped as "unreadable". For a dataset used to
time patrols and count fatalities, both failure modes are severe and
neither raises.
"""

import io

import pandas as pd
import pytest

from core.data_loader import load_and_validate_csv
from core.exceptions import DataValidationError

HEADER = "Date,Latitude,Longitude,Division,Range,Beat\n"


def _csv(*date_values):
    body = "".join(f"{d},22.3,80.6,Div,Rng,Beat\n" for d in date_values)
    return io.StringIO(HEADER + body)


def _dates(df):
    return df["Date"].dt.strftime("%Y-%m-%d").tolist()


def test_day_first_dates_are_parsed_correctly_and_no_rows_are_lost():
    df, _ = load_and_validate_csv(_csv("03/04/2026", "21/04/2026", "05/06/2026"))
    assert _dates(df) == ["2026-04-03", "2026-04-21", "2026-06-05"]
    assert len(df) == 3


def test_ambiguous_dates_are_flagged_rather_than_guessed_silently():
    """Every day value <= 12 means the file cannot say which layout it
    uses. Read it day-first, but say so."""
    df, warnings = load_and_validate_csv(_csv("03/04/2026", "05/06/2026"))
    assert _dates(df) == ["2026-04-03", "2026-06-05"]
    assert any("ambiguous" in w.lower() for w in warnings)


def test_iso_dates_still_work():
    df, _ = load_and_validate_csv(_csv("2026-04-03", "2026-04-21"))
    assert _dates(df) == ["2026-04-03", "2026-04-21"]


def test_dash_separated_day_first_dates_work():
    df, _ = load_and_validate_csv(_csv("03-04-2026", "21-04-2026"))
    assert _dates(df) == ["2026-04-03", "2026-04-21"]


def test_unparseable_dates_are_dropped_with_a_warning():
    df, warnings = load_and_validate_csv(_csv("03/04/2026", "not a date"))
    assert len(df) == 1
    assert any("date" in w.lower() for w in warnings)


def test_missing_required_column_raises_a_clear_error():
    bad = io.StringIO("Date,Latitude\n03/04/2026,22.3\n")
    with pytest.raises(DataValidationError) as exc:
        load_and_validate_csv(bad)
    assert "missing required column" in str(exc.value).lower()


def test_out_of_range_coordinates_are_dropped():
    csv = io.StringIO(
        HEADER + "03/04/2026,22.3,80.6,D,R,B\n04/04/2026,999,80.6,D,R,B\n"
    )
    df, warnings = load_and_validate_csv(csv)
    assert len(df) == 1
    assert any("latitude" in w.lower() for w in warnings)


def test_death_count_without_flag_is_reported_with_its_row_id():
    """This one decides whether a recorded death is counted. It must be
    named explicitly, not resolved silently."""
    csv = io.StringIO(
        "Date,Latitude,Longitude,Division,Range,Beat,ID,Death,Male Death Count\n"
        "03/04/2026,22.3,80.6,D,R,B,101,0,1\n"
    )
    _, warnings = load_and_validate_csv(csv)
    mismatch = [w for w in warnings if "death count filled in" in w]
    assert mismatch
    assert "101" in mismatch[0]


def test_death_flag_without_breakdown_is_reported():
    csv = io.StringIO(
        "Date,Latitude,Longitude,Division,Range,Beat,ID,Death,Male Death Count\n"
        "03/04/2026,22.3,80.6,D,R,B,202,1,0\n"
    )
    _, warnings = load_and_validate_csv(csv)
    assert any("no per-person breakdown" in w for w in warnings)


def test_time_column_is_converted_to_hour():
    csv = io.StringIO(
        "Date,Latitude,Longitude,Division,Range,Beat,Time\n"
        "03/04/2026,22.3,80.6,D,R,B,21:30\n"
    )
    df, _ = load_and_validate_csv(csv)
    assert df["Hour"].iloc[0] == 21


def test_out_of_range_hour_becomes_unknown():
    csv = io.StringIO(
        "Date,Latitude,Longitude,Division,Range,Beat,Hour\n"
        "03/04/2026,22.3,80.6,D,R,B,99\n"
    )
    df, warnings = load_and_validate_csv(csv)
    assert pd.isna(df["Hour"].iloc[0])
    assert any("0-23" in w for w in warnings)


def test_blank_beat_becomes_unknown_and_stays_sortable():
    """A single blank Beat used to take the whole app down.

    On the Arrow-backed string dtype, astype(str) leaves a missing value
    as NaN rather than "nan", so the null survived into Beat. The sidebar
    builds its filter options with sorted(df["Beat"].unique()), which
    then raises comparing float to str.
    """
    csv = io.StringIO(
        HEADER + "03/04/2026,22.3,80.6,Div,Rng,\n04/04/2026,22.3,80.6,Div,Rng,Bhola\n"
    )
    df, warnings = load_and_validate_csv(csv)

    assert df["Beat"].isna().sum() == 0
    assert "Unknown" in set(df["Beat"])
    sorted(df["Beat"].unique())  # must not raise
    assert any("no 'Beat' value" in w for w in warnings)


def test_blank_rows_are_not_dropped_silently():
    """Grouping them under Unknown keeps them in the totals."""
    csv = io.StringIO(
        HEADER + "03/04/2026,22.3,80.6,,Rng,Beat\n04/04/2026,22.3,80.6,Div,Rng,Beat\n"
    )
    df, _ = load_and_validate_csv(csv)
    assert len(df) == 2


def test_empty_file_raises():
    with pytest.raises(DataValidationError):
        load_and_validate_csv(io.StringIO(HEADER))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
