"""Household damage surveys: the herd-crop-damage and herd-house-damage forms.

These are not sightings. Each row is a field team's visit to one
household after elephants damaged its crop or house: what was lost, what
it is worth, whether compensation has been applied for, and how long the
visit came after the damage. So they get their own analysis -- loss,
the compensation pipeline, the gap against the Gaj Rakshak register,
survey timeliness, repeat households -- rather than being folded into
the sighting statistics, where a damage survey would read as one more
conflict sighting and its rupee value would be lost.

**Personal data is dropped as the file is read.** The forms carry the
owner's name, mobile number, a photo and the surveyor's email. None of
that reaches the page or a download. Owner name and mobile are hashed
together in memory to recognise a household surveyed more than once,
and the hash is all that is kept. Free-text remarks are reduced to
whether they say the incident is missing from Gaj Rakshak, because the
text itself sometimes describes the people involved.

**Fields are found by question, not position.** Epicollect5 names a
column after its question's number and the start of its wording
("7_Date_of_Damage"); the number is ignored, so re-ordering a form does
not break the mapping. The same reader takes the CSV/ZIP export from
the project page and the entries the export API returns.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

CROP, HOUSE = "Crop", "House"

# Compensation as recorded, in the order a case moves through it.
NOT_APPLIED = "Not applied"
IN_PROCESS = "In process"
NOT_RECEIVED = "Not received"
RECEIVED = "Received"
NOT_RECORDED = "Not recorded"
STATUS_ORDER = [NOT_APPLIED, NOT_RECEIVED, IN_PROCESS, RECEIVED, NOT_RECORDED]

# What the remarks say about the Gaj Rakshak register.
REG_MISSING = "Not in Gaj Rakshak"
REG_WRONG_PLACE = "Wrong location in Gaj Rakshak"
REG_MENTIONED = "Gaj Rakshak mentioned"

NIGHT_START, NIGHT_END = 18, 6

# Question stem (number stripped, lower case) -> field. Two questions
# share a stem in each form ("Distance to nearest ...", "Presence of
# store..."); they are taken in form order. Fields after the personal
# ones are dropped below, never returned.
_FIELDS: List[Tuple[str, str]] = [
    ("survey_date", "survey_date"),
    ("survey_time", "survey_time"),
    ("village", "village"),
    ("owner", "name_of_the_owner"),
    ("mobile", "mobile_number"),
    ("damage_date", "date_of_damage"),
    ("damage_time", "time_of_damage"),
    ("total_land", "total_land"),
    ("land_cultivated", "land_cultivated"),
    ("irrigation", "irrigation"),
    ("farming", "type_of_farming"),
    ("fencing", "type_of_fencing"),
    ("crops", "crops"),
    ("crop_stage", "stage_of_crop"),
    ("area_owner", "damage_area_owner"),
    ("area_collected", "damage_area_colle"),
    ("area_calculated", "damage_area_calcu"),
    ("crop_quantity", "quantity_of_crop"),
    ("crop_value", "estimated_value_o"),
    ("house_type", "type_of_house"),
    ("light_on", "was_the_light_on"),
    ("house_area", "total_area_of_hou"),
    ("house_damaged_area", "damaged_area_of_h"),
    ("rooms", "which_area_of_hou"),
    ("house_value", "estimated_cost_of"),
    ("comp_status", "compensation_rece"),
    ("comp_amount", "compensation_amou"),
    ("elephants", "number_of_elephan"),
    ("remarks", "remarks"),
]

TIDY_COLUMNS = [
    "Entry", "Kind", "Damage Date", "Hour", "Survey Date", "Days to Survey",
    "Latitude", "Longitude", "Village", "Division", "Range", "Beat", "Beat Distance (km)",
    "Elephants", "Estimated Loss", "Compensation", "Compensation Paid",
    "Register Remark", "Household", "Repeat Household",
    "Crops", "Crop Stage", "Irrigated", "Farming", "Fencing",
    "Land Cultivated", "Area (owner)", "Area (surveyor)", "Area (calculated)",
    "House Type", "Light On", "Rooms Damaged", "House Damaged Area",
]


class DamageDataError(ValueError):
    """A file that is not a damage survey export, with a reason to show."""


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _stem(column: str) -> str:
    return re.sub(r"^\d+_", "", str(column).strip().lstrip("\ufeff")).lower()


def read_export(data: bytes, filename: str) -> List[pd.DataFrame]:
    """The form tables in an Epicollect5 CSV export, or a ZIP of them."""
    name = filename.lower()
    if name.endswith(".zip") or data[:2] == b"PK":
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as error:
            raise DamageDataError(f"{filename} is not a readable ZIP file.") from error
        tables = [
            _read_csv(archive.read(member), member)
            for member in archive.namelist()
            if member.lower().endswith(".csv") and not member.startswith("__MACOSX")
        ]
        if not tables:
            raise DamageDataError(f"{filename} holds no CSV file.")
        return tables
    return [_read_csv(data, filename)]


def _read_csv(data: bytes, name: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", dtype=str,
                            keep_default_na=False)
    except Exception as error:  # noqa: BLE001 - any parse failure is the user's file
        raise DamageDataError(f"Could not read {name} as CSV: {error}") from error
    frame.attrs["source"] = name
    return frame


def entries_frame(entries: Iterable[dict]) -> pd.DataFrame:
    """API entries in the CSV export's shape: location split into columns."""
    rows = []
    for entry in entries:
        row = {}
        for key, value in entry.items():
            if isinstance(value, dict) and "latitude" in value:
                row[f"lat_{key}"] = value.get("latitude")
                row[f"long_{key}"] = value.get("longitude")
            elif isinstance(value, (list, tuple)):
                row[key] = ", ".join(str(v) for v in value)
            else:
                row[key] = value
        rows.append(row)
    frame = pd.DataFrame(rows).astype(str).replace({"None": "", "nan": ""})
    return frame


def _field_map(columns: Iterable[str]) -> Dict[str, str]:
    """Field -> column, by question wording, in form order."""
    taken: Dict[str, str] = {}
    for column in columns:
        stem = _stem(column)
        for field, prefix in _FIELDS:
            if field not in taken and stem.startswith(prefix):
                taken[field] = column
                break
    return taken


def _location_columns(columns: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    lat = next((c for c in columns if c.lower().startswith("lat_")), None)
    lon = next((c for c in columns if c.lower().startswith("long_")), None)
    return lat, lon


# ---------------------------------------------------------------------------
# Tidying
# ---------------------------------------------------------------------------
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _dates(values: pd.Series) -> pd.Series:
    """dd/mm/yyyy as the CSV export writes it, or ISO as the API does."""
    text = values.fillna("").astype(str).str.strip()
    iso = text.str.match(_ISO)
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(text[iso].str[:10], format="%Y-%m-%d", errors="coerce")
    if (~iso).any():
        out[~iso] = pd.to_datetime(text[~iso], format="%d/%m/%Y", errors="coerce")
    return out


def _hours(values: pd.Series) -> pd.Series:
    found = values.fillna("").astype(str).str.extract(r"(?:T|^)(\d{1,2}):\d{2}")[0]
    hours = pd.to_numeric(found, errors="coerce")
    return hours.where(hours.between(0, 23))


def _numbers(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.replace("", np.nan), errors="coerce")


def _yes(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str).str.strip().str.lower()
    return text.map({"yes": True, "no": False}).astype("boolean")


def _status(status: pd.Series) -> pd.Series:
    text = status.fillna("").astype(str).str.strip().str.lower()
    mapped = text.map({
        "not applied": NOT_APPLIED, "in process": IN_PROCESS,
        "no": NOT_RECEIVED, "yes": RECEIVED, "received": RECEIVED,
    }).fillna(NOT_RECORDED)
    return mapped


_NEGATION = re.compile(
    r"\b(not|no|nahi|nahin|nhi|need to|needs to|missing)\b|nahi|nhi", re.IGNORECASE
)


def register_remark(remark: str) -> str:
    """What a free-text remark says about Gaj Rakshak, or ''."""
    text = str(remark or "")
    if not re.search(r"gaj\s*-?\s*rak", text, re.IGNORECASE):
        return ""
    if re.search(r"location", text, re.IGNORECASE):
        return REG_WRONG_PLACE
    if _NEGATION.search(text):
        return REG_MISSING
    return REG_MENTIONED


def _household(owner: pd.Series, mobile: pd.Series) -> pd.Series:
    """A one-way key for a household. The name and number are not kept."""
    names = owner.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    phones = mobile.fillna("").astype(str).str.replace(r"\D", "", regex=True).str[-10:]
    keys = []
    for name, phone in zip(names, phones):
        if not name and not phone:
            keys.append("")
            continue
        keys.append(hashlib.blake2b(f"{name}|{phone}".encode(), digest_size=6).hexdigest())
    return pd.Series(keys, index=owner.index)


def _kind(fields: Dict[str, str]) -> Optional[str]:
    if "crops" in fields or "crop_value" in fields:
        return CROP
    if "house_type" in fields or "house_value" in fields:
        return HOUSE
    return None


def tidy(tables: Iterable[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    """Combine crop and house survey tables into one de-identified frame.

    Returns the frame and notes on anything dropped or doubtful.
    """
    frames, notes = [], []
    for table in tables:
        source = table.attrs.get("source", "a table")
        fields = _field_map(table.columns)
        kind = _kind(fields)
        if kind is None or "damage_date" not in fields:
            raise DamageDataError(
                f"{source} does not look like a herd-crop-damage or "
                "herd-house-damage export: no crop or house questions, or no "
                "date of damage."
            )
        lat_col, lon_col = _location_columns(table.columns)
        col = lambda f: table[fields[f]] if f in fields else pd.Series("", index=table.index)  # noqa: E731

        out = pd.DataFrame(index=table.index)
        out["Entry"] = table.get("ec5_uuid", pd.Series("", index=table.index))
        out["Kind"] = kind
        out["Damage Date"] = _dates(col("damage_date"))
        out["Hour"] = _hours(col("damage_time"))
        out["Survey Date"] = _dates(col("survey_date"))
        out["Latitude"] = _numbers(table[lat_col]) if lat_col else np.nan
        out["Longitude"] = _numbers(table[lon_col]) if lon_col else np.nan
        out["Village"] = col("village").astype(str).str.strip().str.title().replace("", "Unknown")
        out["Elephants"] = _numbers(col("elephants"))
        value = col("crop_value") if kind == CROP else col("house_value")
        out["Estimated Loss"] = _numbers(value)
        paid = _numbers(col("comp_amount"))
        out["Compensation Paid"] = paid.fillna(0.0)
        out["Compensation"] = _status(col("comp_status"))
        out["Register Remark"] = col("remarks").map(register_remark)
        out["Household"] = _household(col("owner"), col("mobile"))

        if kind == CROP:
            out["Crops"] = col("crops").astype(str).str.strip()
            out["Crop Stage"] = col("crop_stage").astype(str).str.strip()
            out["Irrigated"] = _yes(col("irrigation"))
            out["Farming"] = col("farming").astype(str).str.strip()
            out["Fencing"] = col("fencing").astype(str).str.strip()
            out["Land Cultivated"] = _numbers(col("land_cultivated"))
            for target, field in (("Area (owner)", "area_owner"),
                                  ("Area (surveyor)", "area_collected"),
                                  ("Area (calculated)", "area_calculated")):
                area = _numbers(col(field))
                bad = int((area < 0).sum())
                if bad:
                    notes.append(f"{bad} negative {target.lower()} value(s) treated as blank.")
                out[target] = area.where(area >= 0)
        else:
            out["House Type"] = (col("house_type").astype(str).str.strip()
                                 .str.replace(r"\s*-\s*", "-", regex=True))
            out["Light On"] = _yes(col("light_on"))
            out["Rooms Damaged"] = col("rooms").astype(str).str.strip()
            out["House Damaged Area"] = _numbers(col("house_damaged_area"))
        frames.append(out)

    if not frames:
        raise DamageDataError("No survey tables were given.")
    data = pd.concat(frames, ignore_index=True).reindex(columns=TIDY_COLUMNS)

    # The same entry can arrive twice: a ZIP and a CSV, or an upload and a fetch.
    has_id = data["Entry"].fillna("").astype(str).str.len() > 0
    dupes = int(data[has_id].duplicated("Entry").sum())
    if dupes:
        notes.append(f"{dupes} entry(ies) appeared twice and were counted once.")
        data = data[~(has_id & data.duplicated("Entry"))].reset_index(drop=True)

    undated = int(data["Damage Date"].isna().sum())
    if undated:
        notes.append(f"{undated} survey(s) with no readable date of damage were left out "
                     "of the time charts.")
    lag = (data["Survey Date"] - data["Damage Date"]).dt.days
    early = lag < 0
    if early.any():
        notes.append(
            f"{int(early.sum())} survey(s) are dated before the damage they record, "
            "so the date of damage is probably mistyped; it is left blank, which "
            "keeps them out of the time charts and survey-delay figures."
        )
        data.loc[early, "Damage Date"] = pd.NaT
    data["Days to Survey"] = lag.where(~early)
    data["Repeat Household"] = data["Household"].ne("") & data.duplicated("Household", keep=False)
    data = _place(data)
    return data, notes


def _place(data: pd.DataFrame) -> pd.DataFrame:
    """Division, range and beat from the forest boundary polygons.

    Damaged fields and houses mostly sit on farmland between forest
    blocks, outside every beat polygon. For follow-up the useful answer
    is the beat whose staff cover that edge, so a point outside is given
    the nearest unit, and ``Beat Distance (km)`` says how far off it is
    (0 when inside).
    """
    from core import boundaries

    out = data.copy()
    if out.empty or not boundaries.available():
        out[["Division", "Range", "Beat"]] = "Unknown"
        out["Beat Distance (km)"] = np.nan
        return out
    lon = out["Longitude"].to_numpy(float)
    lat = out["Latitude"].to_numpy(float)
    for col, level in (("Division", boundaries.DIVISION), ("Range", boundaries.RANGE),
                       ("Beat", boundaries.BEAT)):
        names, km = boundaries.nearest(level, lon, lat)
        out[col] = np.where(names == "", "Unknown", names)
        if col == "Beat":
            out["Beat Distance (km)"] = np.round(km, 2)
    return out


def load(files: Iterable[Tuple[bytes, str]]) -> Tuple[pd.DataFrame, List[str]]:
    """Read and tidy any number of uploaded exports."""
    tables: List[pd.DataFrame] = []
    for data, name in files:
        tables.extend(read_export(data, name))
    return tidy(tables)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def summary(df: pd.DataFrame) -> Dict[str, float]:
    crop, house = df[df["Kind"] == CROP], df[df["Kind"] == HOUSE]
    hours = df["Hour"].dropna()
    return {
        "reports": len(df),
        "crop": len(crop),
        "house": len(house),
        "villages": int(df["Village"].nunique()),
        "households": int(df.loc[df["Household"] != "", "Household"].nunique()),
        "repeat_households": int(df.loc[df["Repeat Household"], "Household"].nunique()),
        "loss": float(df["Estimated Loss"].sum()),
        "loss_crop": float(crop["Estimated Loss"].sum()),
        "loss_house": float(house["Estimated Loss"].sum()),
        "not_applied": int((df["Compensation"] == NOT_APPLIED).sum()),
        "not_applied_loss": float(df.loc[df["Compensation"] == NOT_APPLIED, "Estimated Loss"].sum()),
        "in_process": int((df["Compensation"] == IN_PROCESS).sum()),
        "not_received": int((df["Compensation"] == NOT_RECEIVED).sum()),
        "paid_cases": int((df["Compensation Paid"] > 0).sum()),
        "paid": float(df["Compensation Paid"].sum()),
        "median_days": float(df["Days to Survey"].median()) if df["Days to Survey"].notna().any() else float("nan"),
        "register_missing": int((df["Register Remark"] == REG_MISSING).sum()),
        "register_wrong": int((df["Register Remark"] == REG_WRONG_PLACE).sum()),
        "night_share": float(((hours >= NIGHT_START) | (hours < NIGHT_END)).mean() * 100)
        if len(hours) else float("nan"),
        "crop_median_herd": float(crop["Elephants"].median()) if len(crop) else float("nan"),
        "house_lone": int((house["Elephants"] == 1).sum()),
    }


def headlines(df: pd.DataFrame, register_match: Optional[pd.Series] = None) -> List[str]:
    """Plain findings for the top of the damage view."""
    s = summary(df)
    lines = []
    if s["reports"]:
        lines.append(
            f"{s['reports']:,} household(s) surveyed in {s['villages']} village(s): "
            f"{s['crop']:,} crop and {s['house']:,} house damage, an estimated "
            f"{rupees(s['loss'])} lost."
        )
    by_kind = df.groupby("Kind")
    parts = []
    for kind, group in by_kind:
        n = int((group["Compensation"] == NOT_APPLIED).sum())
        if n:
            parts.append(f"{n} of {len(group)} {kind.lower()} cases")
    if parts:
        lines.append(
            f"No compensation claim made in {' and '.join(parts)}, covering "
            f"{rupees(s['not_applied_loss'])} of estimated loss -- the follow-up "
            "queue below lists them."
        )
    lines.append(
        f"Compensation paid in {s['paid_cases']} case(s) ({rupees(s['paid'])}); "
        f"{s['in_process']} in process, {s['not_received']} recorded as not received."
    )
    if s["register_missing"] or s["register_wrong"]:
        lines.append(
            f"Surveyors noted {s['register_missing']} incident(s) missing from Gaj "
            f"Rakshak" + (f" and {s['register_wrong']} with the wrong location there"
                          if s["register_wrong"] else "") + "."
        )
    if register_match is not None and len(register_match):
        lines.append(
            f"{int((~register_match).sum())} of {len(register_match)} surveyed incidents "
            "have no Gaj Rakshak report within the matching distance and days."
        )
    if s["repeat_households"]:
        lines.append(f"{s['repeat_households']} household(s) were damaged more than once.")
    if s["house"] and s["crop"]:
        lines.append(
            f"Houses were broken by a lone elephant in {s['house_lone']} of {s['house']} "
            f"cases; crop raids involved a median of {s['crop_median_herd']:.0f} elephants."
        )
    if s["median_days"] == s["median_days"]:
        lines.append(f"Median {s['median_days']:.0f} day(s) from damage to survey.")
    return lines


def rupees(value: float) -> str:
    """Indian grouping: lakh above one lakh, otherwise the full figure."""
    if value != value:
        return "-"
    if abs(value) >= 1e5:
        return f"Rs {value / 1e5:,.1f} lakh"
    return f"Rs {value:,.0f}"


def lakh(value: float) -> str:
    """A rupee figure for a narrow tile: '37.4 lakh', or '2,600' below one."""
    if value != value:
        return "-"
    return f"{value / 1e5:,.1f} lakh" if abs(value) >= 1e5 else f"{value:,.0f}"


def compensation_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Cases and estimated loss by kind and compensation status."""
    grouped = (
        df.groupby(["Kind", "Compensation"], observed=True)
        .agg(Cases=("Entry", "size"), Loss=("Estimated Loss", "sum"))
        .reset_index()
    )
    grouped["_o"] = grouped["Compensation"].map({s: i for i, s in enumerate(STATUS_ORDER)})
    return grouped.sort_values(["Kind", "_o"]).drop(columns="_o").reset_index(drop=True)


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Cases and loss per month of damage and kind, zero-filled."""
    dated = df[df["Damage Date"].notna()]
    if dated.empty:
        return pd.DataFrame(columns=["Month", "Kind", "Cases", "Loss"])
    months = pd.period_range(dated["Damage Date"].min(), dated["Damage Date"].max(), freq="M")
    rows = []
    for kind, group in dated.groupby("Kind"):
        per = group.groupby(group["Damage Date"].dt.to_period("M")).agg(
            Cases=("Entry", "size"), Loss=("Estimated Loss", "sum"))
        per = per.reindex(months, fill_value=0)
        per["Kind"] = kind
        rows.append(per)
    out = pd.concat(rows)
    out.index = out.index.to_timestamp()
    return out.rename_axis("Month").reset_index()[["Month", "Kind", "Cases", "Loss"]]


def exploded_counts(df: pd.DataFrame, column: str, order: Optional[List[str]] = None) -> pd.DataFrame:
    """Cases per answer of a multi-select question ("Rice, Kodo" counts both)."""
    values = (df[column].fillna("").astype(str).str.split(r",\s*").explode().str.strip())
    values = values[values != ""]
    counts = values.value_counts().rename_axis(column).reset_index(name="Cases")
    if order:
        counts["_o"] = counts[column].map({v: i for i, v in enumerate(order)}).fillna(len(order))
        counts = counts.sort_values(["_o", "Cases"], ascending=[True, False]).drop(columns="_o")
    return counts.reset_index(drop=True)


CROP_STAGES = ["Primary Stage", "Secondary Stage", "Flowering Stage", "Harvest Stage",
               "Post Harvest"]


def crop_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per crop: cases, and median loss of the cases it was part of."""
    crop = df[df["Kind"] == CROP]
    exploded = crop.assign(Crop=crop["Crops"].fillna("").str.split(r",\s*")).explode("Crop")
    exploded = exploded[exploded["Crop"].str.strip() != ""]
    return (
        exploded.groupby("Crop")
        .agg(Cases=("Entry", "size"), **{"Median Loss": ("Estimated Loss", "median")})
        .sort_values("Cases", ascending=False)
        .reset_index()
    )


def area_check(df: pd.DataFrame) -> Dict[str, float]:
    """Owner's claimed area against the surveyor's and the calculated one."""
    crop = df[df["Kind"] == CROP].dropna(subset=["Area (owner)", "Area (calculated)"])
    if crop.empty:
        return {}
    owner = crop["Area (owner)"].sum()
    return {
        "cases": len(crop),
        "owner": float(owner),
        "surveyor": float(crop["Area (surveyor)"].sum()),
        "calculated": float(crop["Area (calculated)"].sum()),
        "ratio": float(crop["Area (calculated)"].sum() / owner) if owner else float("nan"),
    }


HERD_BINS = [0, 1, 3, 9, 19, np.inf]
HERD_LABELS = ["1", "2-3", "4-9", "10-19", "20+"]


def herd_sizes(df: pd.DataFrame) -> pd.DataFrame:
    binned = pd.cut(df["Elephants"], HERD_BINS, labels=HERD_LABELS)
    return (
        df.assign(Herd=binned).groupby(["Kind", "Herd"], observed=False)
        .size().rename("Cases").reset_index()
    )


LAG_BINS = [-0.5, 2, 7, 14, 30, np.inf]
LAG_LABELS = ["0-2 days", "3-7", "8-14", "15-30", "Over 30"]


def survey_lag(df: pd.DataFrame) -> pd.DataFrame:
    binned = pd.cut(df["Days to Survey"], LAG_BINS, labels=LAG_LABELS)
    return (
        df.assign(Delay=binned).dropna(subset=["Delay"])
        .groupby(["Kind", "Delay"], observed=False).size().rename("Cases").reset_index()
    )


def hourly(df: pd.DataFrame) -> pd.DataFrame:
    hours = df.dropna(subset=["Hour"])
    grid = pd.MultiIndex.from_product([sorted(df["Kind"].unique()), range(24)],
                                      names=["Kind", "Hour"])
    return (
        hours.groupby(["Kind", hours["Hour"].astype(int)]).size()
        .reindex(grid, fill_value=0).rename("Cases").reset_index()
    )


def villages(df: pd.DataFrame, register_match: Optional[pd.Series] = None) -> pd.DataFrame:
    """One row per village, worst loss first."""
    work = df.assign(
        _crop=df["Kind"] == CROP, _house=df["Kind"] == HOUSE,
        _na=df["Compensation"] == NOT_APPLIED,
        _missing=df["Register Remark"] == REG_MISSING,
    )
    if register_match is not None:
        work["_unmatched"] = ~register_match.reindex(df.index).fillna(False).astype(bool)
    agg = {
        "Division": ("Division", lambda s: s.mode().iat[0] if len(s.mode()) else ""),
        "Beat": ("Beat", lambda s: s.mode().iat[0] if len(s.mode()) else ""),
        "Cases": ("Entry", "size"),
        "Crop": ("_crop", "sum"),
        "House": ("_house", "sum"),
        "Estimated Loss": ("Estimated Loss", "sum"),
        "Not Applied": ("_na", "sum"),
        "Repeat Households": ("Household", lambda s: int(s[s.duplicated(keep=False) & (s != "")].nunique())),
        "Not in Gaj Rakshak (remarks)": ("_missing", "sum"),
        "Last Damage": ("Damage Date", "max"),
    }
    if register_match is not None:
        agg["No Register Match"] = ("_unmatched", "sum")
    return (
        work.groupby("Village").agg(**agg)
        .sort_values("Estimated Loss", ascending=False).reset_index()
    )


def follow_up(df: pd.DataFrame, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Cases with no compensation claim, largest loss first.

    Identified by the Epicollect5 entry, which opens the record -- name
    and number included -- for someone with access to the project. The
    page never carries them.
    """
    queue = df[df["Compensation"] == NOT_APPLIED].copy()
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    queue["Days Since Damage"] = (as_of - queue["Damage Date"]).dt.days
    cols = ["Entry", "Village", "Beat", "Kind", "Damage Date", "Estimated Loss",
            "Days Since Damage", "Repeat Household"]
    return queue.sort_values("Estimated Loss", ascending=False)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Against the Gaj Rakshak register
# ---------------------------------------------------------------------------
def match_register(df: pd.DataFrame, register: pd.DataFrame, radius_km: float = 1.0,
                   days: int = 3) -> pd.Series:
    """Whether each survey has a register report near it in place and time.

    A register row counts if it lies within ``radius_km`` of the surveyed
    field or house and within ``days`` of the date of damage. Surveys
    with no date or position are left unmatched.
    """
    result = pd.Series(False, index=df.index)
    if register is None or register.empty or df.empty:
        return result
    reg = register.dropna(subset=["Date", "Latitude", "Longitude"])
    if reg.empty:
        return result
    reg_lat = np.radians(reg["Latitude"].to_numpy(float))
    reg_lon = np.radians(reg["Longitude"].to_numpy(float))
    reg_day = reg["Date"].dt.normalize().to_numpy("datetime64[D]")

    usable = df["Damage Date"].notna() & df["Latitude"].notna() & df["Longitude"].notna()
    for idx in df.index[usable]:
        day = np.datetime64(df.at[idx, "Damage Date"].normalize(), "D")
        near_time = np.abs((reg_day - day).astype(int)) <= days
        if not near_time.any():
            continue
        lat = np.radians(df.at[idx, "Latitude"])
        lon = np.radians(df.at[idx, "Longitude"])
        dlat = reg_lat[near_time] - lat
        dlon = reg_lon[near_time] - lon
        a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(reg_lat[near_time]) * np.sin(dlon / 2) ** 2
        km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        result.at[idx] = bool((km <= radius_km).any())
    return result
