"""Early-warning coverage: which exposed villages have someone to call.

The early-warning system pushes alerts to villagers who have registered a
mobile number, and the forest department's own staff are registered as
app users. Both are separate exports from the sighting data, and the two
have never been read against each other. A village can sit at the top of
the risk ranking with nobody enrolled in it and nothing in the app would
say so.

**This module handles personal data.** The villager registry carries
names and mobile numbers; the staff export adds email addresses. Both
loaders drop every identifying column at the point of reading, before
the frame reaches anything else, and coordinates are rounded to roughly
a hundred metres -- enough to place someone in a village, not enough to
place them at a house. Nothing downstream can display or export a
contact because nothing downstream is ever given one.

Registrants are placed against village centroids spatially rather than
by name. The two files transliterate differently, so on the reference
export only 41% of nearest-centroid names agreed with the registry's own
village field, while 95% of registrants fell within 2 km of some
centroid. A name match inside that radius is preferred where one exists;
otherwise the nearest centroid takes it.
"""

from __future__ import annotations

import logging
import math
import re
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from core.config import (
    EWS_MATCH_RADIUS_KM,
    KM_PER_DEG_LAT,
    KM_PER_DEG_LON_EQUATOR,
    MIN_EWS_CONTACTS,
    REGISTRY_COORD_PRECISION,
    VALID_LAT_RANGE,
    VALID_LON_RANGE,
)
from core.csv_io import read_csv_resilient
from core.exceptions import DataValidationError
from core.intelligence import TIER_CRITICAL, TIER_HIGH, TIER_ORDER

logger = logging.getLogger(__name__)

Source = Union[str, BinaryIO]

# Columns kept from each export. Everything else -- Name, Mobile, Email --
# is dropped before the frame is returned. This is an allowlist rather
# than a blocklist so a new identifying column in a future export is
# excluded by default instead of by having been anticipated.
REGISTRY_KEEP = ["Latitude", "Longitude", "Village", "Division"]
ROSTER_KEEP = ["Division", "Range", "Role", "Latitude", "Longitude"]

REGISTRY_REQUIRED = {"Latitude", "Longitude"}
ROSTER_REQUIRED = {"Division"}

COVERAGE_NONE = "No contact"
COVERAGE_THIN = "Thin"
COVERAGE_OK = "Covered"

# Ordered worst first, so a sort by this reads as a work queue.
COVERAGE_ORDER = [COVERAGE_NONE, COVERAGE_THIN, COVERAGE_OK]

# Exports use a literal dash where a field was never filled in.
_PLACEHOLDERS = {"", "-", "--", "n/a", "na", "null", "none"}


def _clean_label(value: object) -> Optional[str]:
    """Normalise an administrative label, or ``None`` if it is a blank."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return None if text.lower() in _PLACEHOLDERS else text


def _normalise_name(value: object) -> str:
    """Fold a place name for comparison across two transliterations."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def division_labels(df: pd.DataFrame) -> Dict[str, str]:
    """Preferred spelling of each division, keyed by its folded form.

    The sighting loader title-cases division names, so its "Bandhavgarh
    Tr" and the registry's "Bandhavgarh TR" are the same division written
    two ways. Joining on the fold keeps them one row; displaying the
    sighting export's spelling keeps this page agreeing with every other
    table in the app.
    """
    if df is None or df.empty or "Division" not in df.columns:
        return {}
    names = {str(v).strip() for v in df["Division"].dropna().tolist()}
    return {_normalise_name(n): n for n in names if n}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_and_strip(
    source: Source, keep: List[str], required: set, label: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Read a CSV and immediately discard every column not in ``keep``."""
    frame, warnings = read_csv_resilient(source, on_error=DataValidationError)

    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(
            f"The {label} is missing required column(s): "
            f"{', '.join(sorted(missing))}."
        )

    dropped = [c for c in frame.columns if c not in keep]
    kept = [c for c in keep if c in frame.columns]
    # Slice before anything else touches the frame, so the identifying
    # columns exist only inside this function.
    frame = frame[kept].copy()
    if dropped:
        logger.info(
            "%s: discarded %d non-essential column(s) at load.", label, len(dropped)
        )
    return frame, warnings


def _clean_coordinates(
    frame: pd.DataFrame, warnings: List[str], label: str, required: bool
) -> pd.DataFrame:
    """Coerce, range-check and coarsen coordinates.

    Rounding is deliberate: a hundred metres places someone in a village
    without placing them at a doorstep, and nothing here needs better.
    """
    if "Latitude" not in frame.columns or "Longitude" not in frame.columns:
        return frame

    frame["Latitude"] = pd.to_numeric(frame["Latitude"], errors="coerce")
    frame["Longitude"] = pd.to_numeric(frame["Longitude"], errors="coerce")

    out_of_range = ~(
        frame["Latitude"].between(*VALID_LAT_RANGE)
        & frame["Longitude"].between(*VALID_LON_RANGE)
    )
    frame.loc[out_of_range, ["Latitude", "Longitude"]] = np.nan

    missing = int(frame["Latitude"].isna().sum())
    if missing:
        message = (
            f"{missing:,} row(s) in the {label} have no usable location"
            + (" and are not counted against any village." if required else ".")
        )
        warnings.append(message)
    if required:
        frame = frame.dropna(subset=["Latitude", "Longitude"])

    frame["Latitude"] = frame["Latitude"].round(REGISTRY_COORD_PRECISION)
    frame["Longitude"] = frame["Longitude"].round(REGISTRY_COORD_PRECISION)
    return frame


def load_ews_registry(source: Source) -> Tuple[pd.DataFrame, List[str]]:
    """Load the early-warning villager registry, de-identified.

    Args:
        source: Path or uploaded file. Expected columns are ``Latitude``
            and ``Longitude``, optionally ``Village`` and ``Division``.

    Returns:
        ``(frame, warnings)``. One row per registrant, carrying only a
        coarsened location and the labels it declared -- never a name or
        a number.

    Raises:
        DataValidationError: The file is unreadable, or has no
            coordinates to place registrants by.
    """
    frame, warnings = _read_and_strip(
        source, REGISTRY_KEEP, REGISTRY_REQUIRED, "early-warning registry"
    )
    frame = _clean_coordinates(frame, warnings, "registry", required=True)

    if frame.empty:
        raise DataValidationError(
            "No registry rows have a usable location, so nobody can be placed "
            "against a village."
        )

    for column in ("Village", "Division"):
        if column in frame.columns:
            frame[column] = frame[column].map(_clean_label)

    logger.info("Loaded %d de-identified registry rows.", len(frame))
    return frame.reset_index(drop=True), warnings


def load_staff_roster(source: Source) -> Tuple[pd.DataFrame, List[str]]:
    """Load the registered app-user list, de-identified.

    ``Beat`` and ``Sub Beat`` are not kept. In the reference export they
    held a single placeholder value for 1,356 of 1,357 users, and
    ``Range`` was a placeholder for 1,327 -- so the roster can only be
    read at division level, and pretending otherwise would invent a
    precision the file does not have.

    Args:
        source: Path or uploaded file. ``Division`` is required.

    Returns:
        ``(frame, warnings)``, one row per registered user with no
        identifying column.

    Raises:
        DataValidationError: The file is unreadable or has no
            ``Division`` column.
    """
    frame, warnings = _read_and_strip(
        source, ROSTER_KEEP, ROSTER_REQUIRED, "user list"
    )
    frame = _clean_coordinates(frame, warnings, "user list", required=False)

    for column in ("Division", "Range", "Role"):
        if column in frame.columns:
            frame[column] = frame[column].map(_clean_label)

    unplaced = int(frame["Division"].isna().sum())
    if unplaced:
        warnings.append(
            f"{unplaced:,} registered user(s) have no division recorded and are "
            "reported as Unassigned."
        )

    logger.info("Loaded %d de-identified user rows.", len(frame))
    return frame.reset_index(drop=True), warnings


# ---------------------------------------------------------------------------
# Placing registrants against villages
# ---------------------------------------------------------------------------
def _plane(lat: np.ndarray, lon: np.ndarray, lon_scale: float) -> np.ndarray:
    return np.column_stack([lon * lon_scale, lat * KM_PER_DEG_LAT])


def assign_to_villages(
    registry: pd.DataFrame,
    villages: pd.DataFrame,
    radius_km: float = EWS_MATCH_RADIUS_KM,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Place each registrant against one village centroid.

    Assignment is exclusive -- a registrant belongs to one village, so
    the counts partition and can be audited against the file's own row
    count. Where a centroid whose name matches the registrant's declared
    village lies within ``radius_km``, that one wins over a marginally
    closer centroid with a different name.

    Args:
        registry: Output of :func:`load_ews_registry`.
        villages: Centroid table with ``Village``, ``Latitude``,
            ``Longitude``.
        radius_km: How far a registrant may sit from a centroid and
            still be counted as that village's.

    Returns:
        ``(counts, stats)``. ``counts`` is one integer per row of
        ``villages``; ``stats`` reports how the placement went.
    """
    counts = np.zeros(len(villages), dtype=int)
    stats = {"registrants": len(registry), "matched": 0, "by_name": 0, "unmatched": 0}

    if registry.empty or villages.empty:
        stats["unmatched"] = len(registry)
        return counts, stats

    reference_lat = float(
        np.nanmean(
            np.concatenate([
                villages["Latitude"].to_numpy(dtype=float),
                registry["Latitude"].to_numpy(dtype=float),
            ])
        )
    )
    lon_scale = KM_PER_DEG_LON_EQUATOR * math.cos(math.radians(reference_lat))

    village_pts = _plane(
        villages["Latitude"].to_numpy(dtype=float),
        villages["Longitude"].to_numpy(dtype=float),
        lon_scale,
    )
    registry_pts = _plane(
        registry["Latitude"].to_numpy(dtype=float),
        registry["Longitude"].to_numpy(dtype=float),
        lon_scale,
    )

    tree = cKDTree(village_pts)
    distances, nearest = tree.query(registry_pts, k=1)
    in_range = distances <= radius_km

    village_names = villages["Village"].map(_normalise_name).to_numpy()
    declared = (
        registry["Village"].map(_normalise_name).to_numpy()
        if "Village" in registry.columns
        else np.array([""] * len(registry))
    )

    # Only rows that could still be improved need the radius query: a
    # nearest centroid whose name already agrees cannot be bettered.
    candidates = [
        i
        for i in np.flatnonzero(in_range)
        if declared[i] and village_names[nearest[i]] != declared[i]
    ]
    if candidates:
        neighbourhoods = tree.query_ball_point(registry_pts[candidates], r=radius_km)
        for i, neighbours in zip(candidates, neighbourhoods):
            same_name = [j for j in neighbours if village_names[j] == declared[i]]
            if not same_name:
                continue
            offsets = registry_pts[i] - village_pts[same_name]
            nearest[i] = same_name[int(np.argmin((offsets**2).sum(axis=1)))]
            stats["by_name"] += 1

    assigned = nearest[in_range]
    if len(assigned):
        np.add.at(counts, assigned, 1)

    stats["matched"] = int(in_range.sum())
    stats["unmatched"] = int((~in_range).sum())
    logger.info(
        "Placed %d of %d registrants within %.1f km of a centroid (%d re-assigned "
        "on a name match).",
        stats["matched"], stats["registrants"], radius_km, stats["by_name"],
    )
    return counts, stats


def _coverage_label(count: int, min_contacts: int) -> str:
    if count <= 0:
        return COVERAGE_NONE
    if count < min_contacts:
        return COVERAGE_THIN
    return COVERAGE_OK


def village_coverage(
    village_risk: pd.DataFrame,
    villages: Optional[pd.DataFrame],
    registry: Optional[pd.DataFrame],
    radius_km: float = EWS_MATCH_RADIUS_KM,
    min_contacts: int = MIN_EWS_CONTACTS,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Read the early-warning registry against the village risk ranking.

    Args:
        village_risk: Output of :func:`core.hotspots.villages_at_risk`.
        villages: The full centroid table. Registrants are placed against
            all of it, not only the exposed subset, so somebody enrolled
            beside a quiet village is not credited to a risky one nearby.
        registry: Output of :func:`load_ews_registry`.
        radius_km: Placement radius passed to :func:`assign_to_villages`.
        min_contacts: Registrations below which a village counts as thin.
            One number is not coverage: it is off, or its owner is away.

    Returns:
        ``(table, summary)``. ``table`` is ``village_risk`` in the same
        order with ``Registered Contacts`` and ``Coverage`` added.
    """
    columns = list(village_risk.columns) + ["Registered Contacts", "Coverage"]
    empty = pd.DataFrame(columns=columns)
    summary: Dict[str, object] = {
        "registrants": 0, "matched": 0, "by_name": 0, "unmatched": 0,
        "villages_reached": 0, "exposed": 0, "no_contact": 0, "thin": 0,
        "covered": 0, "urgent": 0, "min_contacts": min_contacts,
        "radius_km": radius_km,
    }

    if village_risk.empty or villages is None or villages.empty:
        return empty, summary
    if registry is None or registry.empty:
        return empty, summary

    counts, stats = assign_to_villages(registry, villages, radius_km=radius_km)
    summary.update(stats)
    summary["villages_reached"] = int((counts > 0).sum())

    lookup = pd.DataFrame({
        "_key": _village_key(villages),
        "Registered Contacts": counts,
    }).groupby("_key", as_index=False)["Registered Contacts"].sum()

    table = village_risk.copy()
    table["_key"] = _village_key(table)
    table = table.merge(lookup, on="_key", how="left").drop(columns="_key")
    table["Registered Contacts"] = (
        table["Registered Contacts"].fillna(0).astype(int)
    )
    table["Coverage"] = [
        _coverage_label(int(n), min_contacts) for n in table["Registered Contacts"]
    ]

    summary["exposed"] = int(len(table))
    for label, key in (
        (COVERAGE_NONE, "no_contact"), (COVERAGE_THIN, "thin"), (COVERAGE_OK, "covered")
    ):
        summary[key] = int((table["Coverage"] == label).sum())
    summary["urgent"] = int(len(coverage_gaps(table)))
    summary["at_risk"] = int(table["Tier"].isin([TIER_CRITICAL, TIER_HIGH]).sum())

    return table[columns], summary


def _village_key(frame: pd.DataFrame) -> pd.Series:
    """Join key identifying a centroid, not just a village name.

    Names repeat -- the same name in two places is two places, and the
    centroid loader keeps them apart deliberately. Coordinates are
    rounded into the key rather than compared as floats, because the two
    frames reach them by different routes.
    """
    return (
        frame["Village"].astype(str).str.strip()
        + "@"
        + frame["Latitude"].astype(float).round(5).astype(str)
        + ","
        + frame["Longitude"].astype(float).round(5).astype(str)
    )


def coverage_gaps(table: pd.DataFrame) -> pd.DataFrame:
    """Villages that are exposed and short of contacts, worst first.

    Restricted to Critical and High. A Watch or Routine village with
    nobody enrolled is a backlog item; a Critical one is this week's job,
    and a list that mixes the two gets read as neither.
    """
    if table.empty:
        return table

    gaps = table[
        table["Tier"].isin([TIER_CRITICAL, TIER_HIGH])
        & table["Coverage"].isin([COVERAGE_NONE, COVERAGE_THIN])
    ].copy()
    if gaps.empty:
        return gaps

    gaps["_tier"] = gaps["Tier"].map({t: i for i, t in enumerate(TIER_ORDER)})
    gaps["_cov"] = gaps["Coverage"].map({c: i for i, c in enumerate(COVERAGE_ORDER)})
    return (
        gaps.sort_values(
            ["_tier", "_cov", "Human Deaths", "Conflict Events"],
            ascending=[True, True, False, False],
        )
        .drop(columns=["_tier", "_cov"])
        .reset_index(drop=True)
    )


UNASSIGNED = "Unassigned"


def _folded_division(frame: pd.DataFrame) -> pd.Series:
    """Division as a join key, with blanks collected under one label."""
    if "Division" not in frame.columns:
        return pd.Series([UNASSIGNED] * len(frame), index=frame.index)
    return (
        frame["Division"]
        .map(lambda v: _normalise_name(v) if _clean_label(v) else UNASSIGNED)
        .replace("", UNASSIGNED)
    )


def _display_division(fold: str, labels: Dict[str, str], fallback: Dict[str, str]) -> str:
    if fold == UNASSIGNED:
        return UNASSIGNED
    return labels.get(fold) or fallback.get(fold) or fold


def enrolment_by_division(
    registry: pd.DataFrame, labels: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Registrations and villages reached, per division.

    Uses the registry's own division label rather than the spatial
    placement, so it covers the whole landscape the registry spans --
    including divisions the current sighting filter excludes.
    """
    columns = ["Division", "Registered", "Villages"]
    if registry.empty:
        return pd.DataFrame(columns=columns)

    working = registry.copy()
    working["_fold"] = _folded_division(working)
    own = {
        f: str(v).strip()
        for f, v in zip(working["_fold"], working.get("Division", working["_fold"]))
        if _clean_label(v)
    }

    grouped = working.groupby("_fold", observed=True)
    table = grouped.size().rename("Registered").to_frame()
    table["Villages"] = (
        grouped["Village"].nunique() if "Village" in working.columns else 0
    )
    table = table.reset_index()
    table["Division"] = [
        _display_division(f, labels or {}, own) for f in table["_fold"]
    ]
    return (
        table.sort_values("Registered", ascending=False)
        .reset_index(drop=True)[columns]
    )


def division_staffing(df: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Registered app users against reporting volume, per division.

    Division only. ``Range`` was a placeholder for 98% of the reference
    export's users, so a per-range figure would be an artefact of how the
    accounts were created rather than of where anybody works.

    Returns:
        ``Division``, ``Registered Users``, ``Reports``, ``Conflict
        Events``, ``Reports per User``. Divisions appearing in only one
        of the two sources are still listed, since a division reporting
        with no registered user -- or the reverse -- is the finding.
    """
    columns = ["Division", "Registered Users", "Reports", "Conflict Events",
               "Reports per User"]
    if roster.empty:
        return pd.DataFrame(columns=columns)

    from core.analytics import conflict_mask  # local: avoids a cycle at import

    roster = roster.copy()
    roster["_fold"] = _folded_division(roster)
    own = {
        f: str(v).strip()
        for f, v in zip(roster["_fold"], roster.get("Division", roster["_fold"]))
        if _clean_label(v)
    }
    users = roster["_fold"].value_counts().rename("Registered Users")

    labels = division_labels(df)
    if df is None or df.empty or "Division" not in df.columns:
        reports = pd.Series(dtype="int64", name="Reports")
        conflicts = pd.Series(dtype="int64", name="Conflict Events")
    else:
        working = df.copy()
        working["_fold"] = _folded_division(working)
        working["_conflict"] = conflict_mask(working)
        grouped = working.groupby("_fold", observed=True)
        reports = grouped.size().rename("Reports")
        conflicts = grouped["_conflict"].sum().rename("Conflict Events")

    table = (
        pd.concat([users, reports, conflicts], axis=1)
        .fillna(0)
        .astype("int64")
        .rename_axis("_fold")
        .reset_index()
    )
    table["Division"] = [
        _display_division(f, labels, own) for f in table["_fold"]
    ]
    # Reports with nobody registered is the finding, so it must not
    # divide to infinity or vanish behind a zero.
    table["Reports per User"] = np.where(
        table["Registered Users"] > 0,
        (table["Reports"] / table["Registered Users"].replace(0, np.nan)).round(1),
        np.nan,
    )
    return table.sort_values(
        ["Reports", "Registered Users"], ascending=False
    ).reset_index(drop=True)[columns]


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------
def coverage_headlines(table: pd.DataFrame, summary: Dict[str, object]) -> List[str]:
    """What the coverage read says, in the order it should be acted on."""
    lines: List[str] = []
    if table.empty:
        return lines

    urgent = int(summary.get("urgent", 0))
    at_risk = int(summary.get("at_risk", 0))
    if urgent and at_risk:
        no_contact = int(
            (
                table["Tier"].isin([TIER_CRITICAL, TIER_HIGH])
                & (table["Coverage"] == COVERAGE_NONE)
            ).sum()
        )
        lines.append(
            f"{urgent} of {at_risk} Critical or High villages are short of "
            f"early-warning contacts, {no_contact} of them with nobody enrolled "
            "at all."
        )
    elif at_risk:
        lines.append(
            f"All {at_risk} Critical or High village(s) have at least "
            f"{summary['min_contacts']} registered contacts."
        )

    worst = coverage_gaps(table)
    if not worst.empty:
        top = worst.head(3)
        named = ", ".join(
            f"{row['Village']} ({int(row['Registered Contacts'])} registered, "
            f"{int(row['Conflict Events'])} incidents)"
            for row in top.to_dict("records")
        )
        lines.append(f"Enrol first: {named}.")

    exposed = int(summary.get("exposed", 0))
    covered = int(summary.get("covered", 0))
    if exposed:
        lines.append(
            f"{covered} of {exposed} exposed villages ({covered / exposed * 100:.0f}%) "
            f"have {summary['min_contacts']} or more contacts registered."
        )
    return lines


def coverage_caveats(summary: Dict[str, object]) -> List[str]:
    """What the coverage figures do not mean."""
    notes = [
        "A registration is not a reachable person. A number that has changed, "
        "or a handset that is off overnight, still counts here.",
    ]

    registrants = int(summary.get("registrants", 0))
    unmatched = int(summary.get("unmatched", 0))
    radius = summary.get("radius_km", EWS_MATCH_RADIUS_KM)
    if registrants and unmatched:
        notes.append(
            f"{unmatched:,} of {registrants:,} registrants ({unmatched / registrants * 100:.0f}%) "
            f"sit more than {radius:g} km from any known centroid and count "
            "towards no village."
        )

    by_name = int(summary.get("by_name", 0))
    notes.append(
        "Registrants are placed by GPS, preferring a centroid whose name matches "
        f"the one they gave{f' -- {by_name:,} were re-assigned that way' if by_name else ''}. "
        "The two files transliterate village names differently, so the rest are "
        "placed by distance alone and a settlement with a close neighbour may "
        "have its count split."
    )
    notes.append(
        "Only villages exposed to conflict under the current filters are listed. "
        "The registry covers the whole landscape."
    )
    return notes
