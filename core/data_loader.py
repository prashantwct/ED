"""CSV ingestion and validation for the Elephant Sighting & Conflict Dashboard.

The goal of this module is to turn an arbitrary, hand-maintained field CSV
into a clean, typed DataFrame - or fail with a clear, specific message
about *why* it failed. Nothing in here talks to Streamlit; it is pure
pandas so it can be unit tested and reused (e.g. from a CLI or notebook).

Anything this module cannot resolve confidently is returned as a warning
string rather than silently coerced. For a dataset that records human
fatalities, quietly dropping or reinterpreting a row is the worst
available outcome.
"""

from __future__ import annotations

import logging
from typing import BinaryIO, List, Optional, Tuple, Union

import pandas as pd

from core.exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Columns without which the dashboard cannot function at all.
REQUIRED_COLUMNS = {"Date", "Latitude", "Longitude", "Division", "Range", "Beat"}

# Columns that unlock extra features (severity components, casualty
# counts, night calc) but are not mandatory. Missing ones simply disable
# that feature rather than failing the load.
OPTIONAL_NUMERIC_COLUMNS = [
    "Total Count",
    "Crop Damage",
    "Grain Damage",
    "House Damage",
    "Injury",
    "Death",
    "Male Death Count",
    "Female Death Count",
    "Children Death Count",
    "Male Injury Count",
    "Female Injury Count",
    "Children Injury Count",
]

DEATH_COUNT_COLUMNS = ["Male Death Count", "Female Death Count", "Children Death Count"]

# Text columns that get normalised (title-cased, trimmed) if present.
TEXT_COLUMNS = ["Division", "Range", "Beat"]

VALID_LAT_RANGE = (-90.0, 90.0)
VALID_LON_RANGE = (-180.0, 180.0)

# Candidate date formats, most-likely first. These are field exports from
# Indian forest divisions, so day-first is the working assumption; the
# order here only breaks ties when several formats parse equally well.
DATE_FORMATS = [
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d %b %Y",
    "%d %B %Y",
    "%m/%d/%Y",
]

# Formats that are indistinguishable on a given file whenever every day
# component happens to be <= 12. If both parse the whole column we have
# to state the assumption rather than pick silently.
_AMBIGUOUS_PAIR = ("%d/%m/%Y", "%m/%d/%Y")


def load_and_validate_csv(
    file: Union[str, BinaryIO],
) -> Tuple[pd.DataFrame, List[str]]:
    """Load a sightings/conflict CSV and validate it for dashboard use.

    Args:
        file: A path, or a file-like object such as the one returned by
            ``st.file_uploader`` (must support ``.seek``/``.read``).

    Returns:
        A tuple of ``(dataframe, warnings)``. ``warnings`` is a list of
        human-readable strings describing non-fatal issues encountered
        during loading (rows dropped, dates assumed, casualty fields that
        disagree with each other). The list is empty when the file was
        clean.

    Raises:
        DataValidationError: If the file cannot be parsed as CSV, is
            empty, or is missing one of ``REQUIRED_COLUMNS``.
    """
    warnings: List[str] = []
    df = _read_csv_with_fallback_encoding(file)

    if df.empty:
        raise DataValidationError(
            "The uploaded file has no rows. Please check the export and try again."
        )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            "The CSV is missing required column(s): "
            f"{', '.join(sorted(missing))}. "
            f"Required columns are: {', '.join(sorted(REQUIRED_COLUMNS))}."
        )

    df = df.copy()
    original_rows = len(df)

    # --- Date -----------------------------------------------------------
    df["Date"], date_warnings = _parse_dates(df["Date"])
    warnings.extend(date_warnings)

    bad_dates = int(df["Date"].isna().sum())
    if bad_dates:
        warnings.append(
            f"Dropped {bad_dates} row(s) with an unreadable or missing 'Date' value."
        )
    df = df.dropna(subset=["Date"])

    # --- Coordinates ------------------------------------------------------
    for col in ["Latitude", "Longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    coord_na = df["Latitude"].isna() | df["Longitude"].isna()
    if coord_na.any():
        warnings.append(
            f"Dropped {int(coord_na.sum())} row(s) with missing or non-numeric coordinates."
        )
    df = df[~coord_na]

    lat_ok = df["Latitude"].between(*VALID_LAT_RANGE)
    lon_ok = df["Longitude"].between(*VALID_LON_RANGE)
    coord_out_of_range = ~(lat_ok & lon_ok)
    if coord_out_of_range.any():
        warnings.append(
            f"Dropped {int(coord_out_of_range.sum())} row(s) with coordinates "
            "outside valid latitude/longitude ranges."
        )
    df = df[~coord_out_of_range]

    if df.empty:
        raise DataValidationError(
            "No rows remained after removing invalid dates and coordinates. "
            "Please check the source data."
        )

    # --- Text / categorical columns ---------------------------------------
    #
    # Nulls are filled *before* the string conversion, not after. On the
    # Arrow-backed string dtype pandas now uses for CSV text columns,
    # `astype(str)` leaves a missing value as NaN rather than turning it
    # into the string "nan", so a trailing `.replace({"Nan": ...})` never
    # sees it. The null then survives into Beat/Division/Range, and the
    # first `sorted(df["Beat"].unique())` -- which is how the sidebar
    # builds its filter options -- raises on comparing float to str.
    # One blank beat in an export is enough to take the whole app down.
    for col in TEXT_COLUMNS:
        blank = df[col].isna()
        cleaned = (
            df[col]
            .astype("object")
            .where(~blank, "Unknown")
            .astype(str)
            .str.strip()
            .str.title()
            .replace({"": "Unknown", "Nan": "Unknown", "None": "Unknown", "<Na>": "Unknown"})
        )
        df[col] = cleaned
        if blank.any():
            warnings.append(
                f"{int(blank.sum())} row(s) have no '{col}' value; grouped under "
                "'Unknown' so they still appear in totals rather than vanishing."
            )

    # --- Optional numeric flag columns -------------------------------------
    for col in OPTIONAL_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Hour / Time, used later to derive Is_Night ------------------------
    warnings.extend(_derive_hour(df))

    # --- Casualty field cross-checks ---------------------------------------
    warnings.extend(_check_death_field_consistency(df))

    dropped = original_rows - len(df)
    if dropped:
        warnings.append(
            f"{dropped} of {original_rows} row(s) excluded overall; "
            f"{len(df):,} remain for analysis."
        )

    df = df.reset_index(drop=True)
    logger.info(
        "Loaded %d valid rows (from %d original) with %d warning(s).",
        len(df),
        original_rows,
        len(warnings),
    )
    return df, warnings


def _parse_dates(series: pd.Series) -> Tuple[pd.Series, List[str]]:
    """Parse a date column with an explicit, reported format choice.

    Inferred parsing is not safe here. Given a day-first export, pandas
    locks onto the format implied by the first value: a column starting
    ``03/04/2026`` is read as March 4th, and every later value whose day
    exceeds 12 (``21/04/2026``) fails outright and is dropped as
    "unreadable". The result is a mix of wrong dates and missing rows,
    with no error raised.

    So: try each candidate format against the whole column, keep the one
    that parses the most values, and say which was used whenever the
    choice was not obvious.

    Args:
        series: The raw ``Date`` column.

    Returns:
        ``(parsed_series, warnings)``.
    """
    warnings: List[str] = []

    if pd.api.types.is_datetime64_any_dtype(series):
        return series, warnings

    text = series.astype(str).str.strip()
    non_empty = int((text.notna() & (text != "") & (text.str.lower() != "nan")).sum())
    if non_empty == 0:
        return pd.to_datetime(series, errors="coerce"), warnings

    results = {}
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        results[fmt] = (parsed, int(parsed.notna().sum()))

    best_fmt, (best_parsed, best_hits) = max(
        results.items(), key=lambda item: item[1][1]
    )

    if best_hits == 0:
        # Nothing matched a known layout. Fall back to flexible parsing,
        # but pin day-first so a DD/MM export is not silently transposed.
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
        if parsed.notna().any():
            warnings.append(
                "Date values did not match any expected format; parsed them "
                "flexibly assuming day-first (DD/MM/YYYY). Verify a few dates "
                "against the source export before relying on the trend charts."
            )
        return parsed, warnings

    # Ambiguity check: if the day-first and month-first readings both
    # parse the same number of values, the file itself cannot tell us
    # which is right (every day component is <= 12).
    day_first, month_first = _AMBIGUOUS_PAIR
    if (
        best_fmt in _AMBIGUOUS_PAIR
        and results[day_first][1] == results[month_first][1]
        and results[day_first][1] > 0
        and not results[day_first][0].equals(results[month_first][0])
    ):
        best_fmt = day_first
        best_parsed = results[day_first][0]
        warnings.append(
            "Dates are ambiguous (every day value is 12 or lower, so DD/MM and "
            "MM/DD both parse). Read as day-first DD/MM/YYYY, the field-export "
            "convention. If this export is month-first, the monthly trend and "
            "date filter will be wrong -- confirm with whoever produced the file."
        )
    elif best_hits < non_empty:
        warnings.append(
            f"Parsed dates as {best_fmt}; {non_empty - best_hits} value(s) did "
            "not match that format and were dropped."
        )

    logger.info("Date column parsed with format %s (%d/%d values).", best_fmt, best_hits, non_empty)
    return best_parsed, warnings


def _derive_hour(df: pd.DataFrame) -> List[str]:
    """Populate ``df['Hour']`` in place from ``Hour`` or ``Time``.

    Returns:
        Warning strings describing anything that could not be parsed.
    """
    warnings: List[str] = []

    if "Hour" in df.columns:
        df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
        out_of_range = df["Hour"].notna() & ~df["Hour"].between(0, 23)
        if out_of_range.any():
            warnings.append(
                f"{int(out_of_range.sum())} row(s) had an 'Hour' outside 0-23; "
                "treated as unknown for night/day analysis."
            )
            df.loc[out_of_range, "Hour"] = pd.NA
        return warnings

    if "Time" in df.columns:
        # Try the known field formats first. Gajrakshak exports use
        # "HH:MM:SS"; older sheets use "HH:MM". Matching one of these
        # avoids dateutil's per-element fallback, which parses every row
        # individually and emits a "could not infer format" warning on a
        # file that is in fact perfectly consistent.
        parsed_time = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        for fmt in ("%H:%M:%S", "%H:%M"):
            missing = parsed_time.isna() & df["Time"].notna()
            if not missing.any():
                break
            parsed_time.loc[missing] = pd.to_datetime(
                df.loc[missing, "Time"], format=fmt, errors="coerce"
            )

        still_missing = parsed_time.isna() & df["Time"].notna()
        if still_missing.any():
            parsed_time.loc[still_missing] = pd.to_datetime(
                df.loc[still_missing, "Time"], errors="coerce", format="mixed"
            )
        df["Hour"] = parsed_time.dt.hour

        unparsed = int((parsed_time.isna() & df["Time"].notna()).sum())
        if unparsed > 0:
            warnings.append(
                f"Could not parse a time value for {unparsed} row(s); "
                "night/day classification will be unknown for those rows."
            )
        return warnings

    warnings.append(
        "No 'Hour' or 'Time' column found - night vs. day analysis will be unavailable."
    )
    return warnings


def _check_death_field_consistency(df: pd.DataFrame) -> List[str]:
    """Flag rows where the death counts and the ``Death`` flag disagree.

    Rows with a per-person death count filled in but ``Death`` not set
    are *not* counted as fatalities downstream (see
    ``analytics._people_count``): in the exports reviewed these matched a
    same-day, same-beat report of a fatality already logged elsewhere,
    so counting them would double-count a real death.

    That is a judgement call about someone's death, so it gets named
    explicitly here -- with row IDs where available -- rather than being
    resolved silently in either direction.
    """
    warnings: List[str] = []

    present_counts = [c for c in DEATH_COUNT_COLUMNS if c in df.columns]
    if not present_counts or "Death" not in df.columns:
        return warnings

    flag = pd.to_numeric(df["Death"], errors="coerce").fillna(0)
    count_sum = df[present_counts].sum(axis=1, min_count=1).fillna(0)

    mismatched = df.loc[(count_sum > 0) & (flag <= 0)]
    if len(mismatched):
        warnings.append(
            f"{len(mismatched)} row(s) have a death count filled in but the "
            f"'Death' flag not set{_id_suffix(mismatched)}. These are NOT counted "
            "as fatalities, on the basis that such rows have matched same-day, "
            "same-beat follow-up reports of a death already logged elsewhere in "
            "the export. Confirm with the field reporter that they are duplicates "
            "and not separate incidents."
        )

    flag_no_count = df.loc[(flag > 0) & (count_sum <= 0)]
    if len(flag_no_count):
        warnings.append(
            f"{len(flag_no_count)} row(s) have the 'Death' flag set with no "
            f"per-person breakdown{_id_suffix(flag_no_count)}. Counted as one "
            "fatality each; add the demographic counts if more people were killed."
        )

    return warnings


def _id_suffix(rows: pd.DataFrame, limit: int = 10) -> str:
    """Render a ' (ID 12, 34)' suffix when the export carries an ID column."""
    if "ID" not in rows.columns:
        return ""
    ids = [str(i) for i in rows["ID"].tolist()]
    shown = ", ".join(ids[:limit])
    if len(ids) > limit:
        shown += f", +{len(ids) - limit} more"
    return f" (ID {shown})"


def _read_csv_with_fallback_encoding(file: Union[str, BinaryIO]) -> pd.DataFrame:
    """Read a CSV, retrying with a Latin-1 fallback for legacy field exports."""
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        if hasattr(file, "seek"):
            file.seek(0)
        try:
            return pd.read_csv(file, encoding="ISO-8859-1")
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean error
            raise DataValidationError(
                "Could not read the file as CSV, even with a fallback encoding. "
                "Please confirm it is a valid, comma-separated CSV export."
            ) from exc
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The uploaded file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(
            "The file could not be parsed as CSV. Please confirm the export "
            "format and that it uses standard comma delimiters."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - last-resort, user-facing message
        raise DataValidationError(f"Could not read the uploaded file: {exc}") from exc
