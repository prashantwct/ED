"""CSV ingestion and validation for the Elephant Sighting & Conflict Dashboard.

The goal of this module is to turn an arbitrary, hand-maintained field CSV
into a clean, typed DataFrame - or fail with a clear, specific message
about *why* it failed. Nothing in here talks to Streamlit; it is pure
pandas so it can be unit tested and reused (e.g. from a CLI or notebook).
"""

from __future__ import annotations

import logging
from typing import BinaryIO, List, Tuple, Union

import pandas as pd

from core.exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Columns without which the dashboard cannot function at all.
REQUIRED_COLUMNS = {"Date", "Latitude", "Longitude", "Division", "Range", "Beat"}

# Columns that unlock extra features (severity components, night calc,
# etc.) but are not mandatory. Missing ones simply disable that feature.
OPTIONAL_NUMERIC_COLUMNS = ["Total Count", "Crop Damage", "House Damage", "Injury"]

# Text columns that get normalised (title-cased, trimmed) if present.
TEXT_COLUMNS = ["Division", "Range", "Beat"]

VALID_LAT_RANGE = (-90.0, 90.0)
VALID_LON_RANGE = (-180.0, 180.0)


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
        during loading (e.g. rows dropped for bad coordinates). The list
        is empty when the file was clean.

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

    # --- Date -----------------------------------------------------------
    original_rows = len(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    bad_dates = df["Date"].isna().sum()
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
    for col in TEXT_COLUMNS:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.title()
            .replace({"": "Unknown", "Nan": "Unknown"})
        )

    # --- Optional numeric flag columns -------------------------------------
    for col in OPTIONAL_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Hour / Time, used later to derive Is_Night ------------------------
    if "Hour" in df.columns:
        df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
    elif "Time" in df.columns:
        # Try the common "HH:MM" field format first (fast, no warnings);
        # fall back to flexible parsing for any values it couldn't handle.
        parsed_time = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
        still_missing = parsed_time.isna() & df["Time"].notna()
        if still_missing.any():
            parsed_time.loc[still_missing] = pd.to_datetime(
                df.loc[still_missing, "Time"], errors="coerce"
            )
        df["Hour"] = parsed_time.dt.hour
        if parsed_time.isna().any() and df["Time"].notna().any():
            unparsed = int(parsed_time.isna().sum() - df["Time"].isna().sum())
            if unparsed > 0:
                warnings.append(
                    f"Could not parse a time value for {unparsed} row(s); "
                    "night/day classification will be unknown for those rows."
                )
    else:
        warnings.append(
            "No 'Hour' or 'Time' column found - night vs. day analysis will be unavailable."
        )

    df = df.reset_index(drop=True)
    logger.info(
        "Loaded %d valid rows (from %d original) with %d warning(s).",
        len(df),
        original_rows,
        len(warnings),
    )
    return df, warnings


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
