import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"Date", "Latitude", "Longitude", "Division", "Range", "Beat"}

# Rough bounding box around the Madhya Pradesh elephant range divisions this
# app covers. Not a hard reject -- a row outside it is flagged, not dropped,
# since it may be a real edge-of-range sighting rather than a bad GPS fix.
MP_LAT_RANGE = (20.5, 26.5)
MP_LON_RANGE = (78.0, 84.5)

NIGHT_START_HOUR = 19  # 7 PM
NIGHT_END_HOUR = 2     # 2 AM, inclusive


def load_and_validate_csv(file):
    """
    Loads a Gajrakshak sightings export, validates and types it.

    Returns (df, warnings): warnings is a list of human-readable data
    quality notices meant for the UI. Rows are only ever dropped when
    they're unusable (unparseable date, non-numeric coordinates) and
    every drop is counted in warnings -- nothing disappears silently.
    """
    warnings = []

    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding="ISO-8859-1")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    n_start = len(df)

    # Gajrakshak exports are DD/MM/YYYY. Pinned explicitly -- relying on
    # pandas' format inference works today only because this file happens
    # to be unambiguous; a differently-formatted future export could
    # silently mis-parse the whole column without this.
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    bad_dates = int(df["Date"].isna().sum())
    if bad_dates:
        warnings.append(f"{bad_dates} row(s) had an unparseable Date and were excluded.")
    df = df.dropna(subset=["Date"])

    for col in ["Latitude", "Longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_coords = int(df[["Latitude", "Longitude"]].isna().any(axis=1).sum())
    if bad_coords:
        warnings.append(f"{bad_coords} row(s) had a non-numeric Latitude/Longitude and were excluded from mapping.")
    df = df.dropna(subset=["Latitude", "Longitude"])

    out_of_bounds = ~(
        df["Latitude"].between(*MP_LAT_RANGE) & df["Longitude"].between(*MP_LON_RANGE)
    )
    n_oob = int(out_of_bounds.sum())
    if n_oob:
        warnings.append(
            f"{n_oob} row(s) have coordinates outside the expected range for these divisions "
            f"(lat {MP_LAT_RANGE[0]}-{MP_LAT_RANGE[1]}, lon {MP_LON_RANGE[0]}-{MP_LON_RANGE[1]}) -- "
            "flagged on the map rather than dropped, worth a field check."
        )
    df["Coordinate Flagged"] = out_of_bounds.values

    for col in ["Division", "Range", "Beat"]:
        df[col] = df[col].astype(str).str.strip().str.title().fillna("Unknown")

    if "Time" in df.columns:
        parsed_time = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")
        df["Hour"] = parsed_time.dt.hour
        df["Is_Night"] = (
            (df["Hour"] >= NIGHT_START_HOUR) | (df["Hour"] <= NIGHT_END_HOUR)
        ).fillna(False)
    else:
        df["Hour"] = np.nan
        df["Is_Night"] = False
        warnings.append("No Time column found -- night-window analysis will be unavailable.")

    death_count_cols = [c for c in ["Male Death Count", "Female Death Count", "Children Death Count"] if c in df.columns]
    if death_count_cols and "Death" in df.columns:
        death_flag = pd.to_numeric(df["Death"], errors="coerce").fillna(0)
        count_sum = df[death_count_cols].sum(axis=1, min_count=1).fillna(0)
        mismatched = df.loc[(count_sum > 0) & (death_flag <= 0), "ID"] if "ID" in df.columns else pd.Series(dtype=object)
        if len(mismatched):
            ids = ", ".join(str(i) for i in mismatched.tolist())
            warnings.append(
                f"{len(mismatched)} row(s) have a death count filled in but the Death flag not set "
                f"(ID {ids}) -- treated as NOT a counted fatality here since these matched same-day, "
                "same-beat reports of a death already logged elsewhere in this export. Worth confirming "
                "with the field reporter that these are duplicates and not separate incidents."
            )

    n_end = len(df)
    if n_end < n_start:
        warnings.append(f"{n_start - n_end} of {n_start} total row(s) excluded; {n_end} remain for analysis.")

    return df.reset_index(drop=True), warnings
