"""Crop and house damage reports from Epicollect5.

Two public Epicollect5 projects collect damage reports in the field,
separately from the Gajrakshak sighting register: ``herd-crop-damage``
and ``herd-house-damage``. This reads them over the Epicollect5 export
API and returns rows in the dashboard's own columns, so they go through
``core.data_loader`` like any upload.

**The form is read, not assumed.** Field names are whatever the project
creator typed as questions, so nothing here hard-codes them. The
project definition is fetched first; dates, times and coordinates are
found by input type, and the rest by what the question says ("division",
"beat", "injured", ...). Each project fixes its own damage type, so a
crop report is a crop-damage event whatever its form calls the field.

**Division, range and beat come from the map when the form lacks them.**
A point is placed in the vendored forest boundary polygons, the same
layers the map draws. A report on farmland outside every beat polygon is
labelled as such rather than pinned to the nearest beat it is not in.

**The API's limits are kept to.** Epicollect5 allows 5 requests a minute
per kind and asks clients to stay near half that, and to fetch only what
changed. So requests of one kind are spaced at least
``REQUEST_INTERVAL_SECONDS`` apart, pages are the maximum 500 entries,
and after the first pull a project is synced incrementally on
``uploaded_at``, which covers both new and edited entries. What has been
fetched is kept for the life of the server process, so reruns and other
sessions reuse it instead of asking again.

**Private projects** need a client app made by the project's creator or
a manager, and a client app opens only the project it was made in. So
each project reads its own pair, named from its slug --
``EPICOLLECT_HERD_CROP_DAMAGE_CLIENT_ID`` and ``..._CLIENT_SECRET`` --
from the environment or Streamlit secrets, falling back to a shared
``EPICOLLECT_CLIENT_ID``/``EPICOLLECT_CLIENT_SECRET``. The pair is
exchanged for a bearer token, kept until shortly before it expires
because the API issues only 10 an hour. A public project needs none.

One project failing does not stop the other: what loaded is returned,
and the failure is reported beside it.
"""

from __future__ import annotations

import io
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://five.epicollect.net"

# Slug -> the damage type every report in that project records.
PROJECTS: Dict[str, str] = {
    "herd-crop-damage": "Crop Damage",
    "herd-house-damage": "House Damage",
}

PER_PAGE = 500  # the API maximum
# 5 requests/minute per kind is the cap; the docs ask for about half.
REQUEST_INTERVAL_SECONDS = 24.0
# A sync newer than this is reused without asking the server at all.
MIN_RESYNC_SECONDS = 15 * 60
TIMEOUT_SECONDS = 60

FIELD_TIMEZONE = "Asia/Kolkata"

OUTSIDE_BEATS = "Outside Beat Boundaries"
UNPLACED = "Unknown"

SOURCE_COLUMN = "Source"

# Output columns, in the dashboard's spelling.
COLUMNS = [
    "Date", "Hour", "Latitude", "Longitude", "Division", "Range", "Beat",
    "Total Count", "Male Count", "Female Count", "Calf Count",
    "Crop Damage", "Grain Damage", "House Damage", "Injury", "Death",
    "Village (reported)", SOURCE_COLUMN,
]


class EpicollectError(RuntimeError):
    """A pull that failed, with a message meant for the screen."""


# ---------------------------------------------------------------------------
# HTTP, paced to the API's limits
# ---------------------------------------------------------------------------
_LAST_REQUEST: Dict[str, float] = {}
_LOCK = threading.Lock()


def _setting(name: str) -> str:
    import os

    value = os.environ.get(name, "")
    if value:
        return value.strip()
    try:
        import streamlit as st

        return str(st.secrets.get(name, "")).strip()
    except Exception:  # no secrets file, or no Streamlit context
        return ""


def credential_names(slug: str) -> Tuple[str, str]:
    """The secret names holding one project's client app credentials.

    Epicollect5 client apps belong to a single project, so each private
    project needs its own pair: herd-crop-damage reads
    EPICOLLECT_HERD_CROP_DAMAGE_CLIENT_ID and ..._CLIENT_SECRET.
    """
    stem = "EPICOLLECT_" + re.sub(r"[^A-Z0-9]+", "_", slug.upper()).strip("_")
    return f"{stem}_CLIENT_ID", f"{stem}_CLIENT_SECRET"


def _credentials(slug: str) -> Optional[Tuple[str, str]]:
    """The project's own pair, else the shared one, else None."""
    for id_name, secret_name in (credential_names(slug),
                                 ("EPICOLLECT_CLIENT_ID", "EPICOLLECT_CLIENT_SECRET")):
        client_id, secret = _setting(id_name), _setting(secret_name)
        if client_id and secret:
            return client_id, secret
    return None


# slug -> (bearer token, monotonic expiry). Tokens last two hours and
# the API issues at most 10 an hour, so one is reused until near expiry
# rather than requested per pull.
_TOKENS: Dict[str, Tuple[str, float]] = {}
TOKEN_MARGIN_SECONDS = 300


class _Client:
    """One pull's HTTP session: paced, authenticated per project."""

    def __init__(self, base_url: str, interval: float,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        import requests

        self.base_url = base_url.rstrip("/")
        self.interval = interval
        self.sleep = sleep
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        self.requests = 0

    def _pace(self, kind: str) -> None:
        with _LOCK:
            last = _LAST_REQUEST.get(kind)
            wait = 0.0 if last is None else self.interval - (time.monotonic() - last)
            if wait > 0:
                self.sleep(wait)
            _LAST_REQUEST[kind] = time.monotonic()

    def _token(self, slug: str, renew: bool = False) -> Optional[str]:
        """A bearer token for this project, if credentials are configured."""
        cached = _TOKENS.get(slug)
        if cached and not renew and time.monotonic() < cached[1]:
            return cached[0]
        credentials = _credentials(slug)
        if credentials is None:
            return None
        self._pace("token")
        response = self.session.post(
            f"{self.base_url}/api/oauth/token",
            json={"grant_type": "client_credentials", "client_id": credentials[0],
                  "client_secret": credentials[1]},
            timeout=TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            id_name, _ = credential_names(slug)
            raise EpicollectError(
                f"Epicollect5 refused the client id and secret configured for "
                f"{slug} (HTTP {response.status_code}). Check {id_name} and its "
                "secret against the client app on that project's details page; "
                "a client app only opens the project it was made in."
            )
        payload = response.json()
        lifetime = float(payload.get("expires_in") or 7200)
        token = str(payload["access_token"])
        _TOKENS[slug] = (token, time.monotonic() + lifetime - TOKEN_MARGIN_SECONDS)
        return token

    def get(self, kind: str, path: str, params: Optional[dict] = None,
            slug: str = "") -> dict:
        # Configured credentials are used from the first request, not
        # after a refusal: a wasted call costs a twelfth of the minute.
        token = self._token(slug) if slug else None
        for attempt in range(2):
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            self._pace(kind)
            self.requests += 1
            response = self.session.get(
                f"{self.base_url}{path}", params=params, headers=headers,
                timeout=TIMEOUT_SECONDS,
            )
            if response.status_code == 429:
                raise EpicollectError(
                    "Epicollect5 is rate-limiting this server. Wait a few "
                    "minutes and refresh; what was already fetched is kept."
                )
            if response.status_code in (401, 403, 404) and attempt == 0 and token:
                token = self._token(slug, renew=True)  # expired or revoked
                continue
            if response.status_code != 200:
                raise EpicollectError(_error_message(response, path, slug, bool(token)))
            return response.json()
        raise EpicollectError(_error_message(response, path, slug, bool(token)))


def _error_message(response, path: str, slug: str = "", authenticated: bool = False) -> str:
    detail = ""
    try:
        errors = response.json().get("errors") or []
        if errors:
            detail = f": {errors[0].get('title') or errors[0].get('code')}"
    except ValueError:
        pass
    if response.status_code in (401, 403, 404):
        id_name, secret_name = credential_names(slug or "project")
        if authenticated:
            return (
                f"Epicollect5 would not return {path} (HTTP {response.status_code}"
                f"{detail}) even with the configured client app. Check that "
                f"{id_name} belongs to a client app made in {slug}, not another "
                "project."
            )
        return (
            f"Epicollect5 would not return {path} (HTTP {response.status_code}"
            f"{detail}). {slug or 'The project'} is private: its creator or a "
            "manager has to add a client app on the project's details page, and "
            f"this app needs that app's id and secret as {id_name} and "
            f"{secret_name} in its secrets."
        )
    return f"Epicollect5 returned HTTP {response.status_code} for {path}{detail}."


# ---------------------------------------------------------------------------
# Project structure and field roles
# ---------------------------------------------------------------------------
@dataclass
class _Field:
    key: str  # the entry key, e.g. "3_Beat_name"
    type: str
    question: str


def _forms(structure: dict) -> List[dict]:
    project = (structure.get("data") or structure).get("project") or {}
    return project.get("forms") or []


def _project_name(structure: dict, slug: str) -> str:
    project = (structure.get("data") or structure).get("project") or {}
    return str(project.get("name") or slug)


def _words(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


# Question wording -> dashboard column. First match wins. Casualties come
# before the animal counts, so "male persons injured" is an injury and
# not a bull count; the counts need the question to be about elephants,
# so "number of houses damaged" is not mistaken for herd size.
_ROLE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Division", re.compile(r"\bdivision\b")),
    ("Range", re.compile(r"\brange\b")),
    ("Beat", re.compile(r"\bbeat\b")),
    ("Village (reported)", re.compile(r"\b(village|gram|gaon)\b")),
    ("Death", re.compile(r"\b(death|deaths|died|dead|killed|fatal)")),
    ("Injury", re.compile(r"\b(injur|hurt|wounded)")),
    ("Grain Damage", re.compile(r"\b(grain|stored food|ration)")),
    ("Calf Count", re.compile(r"\b(calf|calves|juvenile)")),
    ("Female Count", re.compile(r"\b(female|cow|cows)\b.*\b(elephant|hathi)|\b(elephant|hathi).*\b(female|cow|cows)\b")),
    ("Male Count", re.compile(r"\b(bull|bulls|tusker|tuskers)\b|\bmale\b.*\b(elephant|hathi)|\b(elephant|hathi).*\bmale\b")),
    ("Total Count", re.compile(r"\b(elephants?|hathi|herd)\b")),
]
_NUMERIC_ROLES = {"Calf Count", "Female Count", "Male Count", "Total Count"}
_TEXT_TYPES = {"text", "textarea", "dropdown", "radio", "searchsingle"}
_NUMBER_TYPES = {"integer", "decimal"}


def _field_roles(form: dict) -> Dict[str, _Field]:
    """Which entry key carries each dashboard column, read off the form.

    Entry keys are numbered by the input's position in the form, then
    the question text; matching on the number keeps this independent of
    how Epicollect shortens the question into a key.
    """
    roles: Dict[str, _Field] = {}
    for position, item in enumerate(form.get("inputs") or [], start=1):
        kind = str(item.get("type") or "").lower()
        question = str(item.get("question") or "")
        f = _Field(key=f"{position}_", type=kind, question=question)
        if kind == "location":
            roles.setdefault("_location", f)
            continue
        if kind == "date":
            roles.setdefault("_date", f)
            continue
        if kind == "time":
            roles.setdefault("_time", f)
            continue
        words = _words(question)
        for role, pattern in _ROLE_PATTERNS:
            if not pattern.search(words):
                continue
            if role in _NUMERIC_ROLES and kind not in _NUMBER_TYPES:
                continue
            if role in ("Division", "Range", "Beat", "Village (reported)") \
                    and kind not in _TEXT_TYPES:
                continue
            roles.setdefault(role, f)
            break
    return roles


def _value(entry: dict, f: Optional[_Field]):
    """An entry's answer to a field, found by the field's position prefix."""
    if f is None:
        return None
    for key, value in entry.items():
        if key.startswith(f.key):
            return value
    return None


# Hindi "haan" in both common spellings, escaped to keep the source ASCII.
_YES = {"yes", "y", "true", "1", "haan", "han", "ha", "\u0939\u093e\u0901", "\u0939\u093e\u0902"}


def _number(value) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in _YES:
        return 1.0
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group()) if match else 0.0


def _parse_date(value, fallback) -> Optional[pd.Timestamp]:
    for candidate in (value, fallback):
        if not candidate:
            continue
        stamp = pd.to_datetime(str(candidate), errors="coerce", utc=True, dayfirst=False)
        if not pd.isna(stamp):
            # Upload stamps are UTC; the day that matters is the field's.
            return stamp.tz_convert(FIELD_TIMEZONE).tz_localize(None)
    return None


def _hour(value) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"(?:T|\s|^)(\d{1,2}):(\d{2})", str(value))
    return int(match.group(1)) if match else None


def entries_to_frame(structure: dict, entries: Iterable[dict], slug: str,
                     damage_column: str) -> pd.DataFrame:
    """One project's entries in the dashboard's columns.

    Division, Range and Beat are left blank where the form has no such
    question; :func:`place_in_boundaries` fills them in afterwards.
    """
    forms = _forms(structure)
    roles = _field_roles(forms[0]) if forms else {}
    name = _project_name(structure, slug)

    rows = []
    for entry in entries:
        location = _value(entry, roles.get("_location")) or {}
        if not isinstance(location, dict):
            location = {}
        try:
            lat = float(location.get("latitude"))
            lon = float(location.get("longitude"))
        except (TypeError, ValueError):
            lat = lon = float("nan")

        stamp = _parse_date(_value(entry, roles.get("_date")), entry.get("created_at"))
        # Only a time question gives the hour. A date answer is midnight,
        # and created_at is when the phone saved it, not when it happened.
        hour = _hour(_value(entry, roles.get("_time")))

        row = {
            "Date": stamp.strftime("%Y-%m-%d") if stamp is not None else "",
            "Hour": hour,
            "Latitude": lat,
            "Longitude": lon,
            SOURCE_COLUMN: f"Epicollect5: {name}",
            "Crop Damage": 0, "Grain Damage": 0, "House Damage": 0,
            "Injury": 0.0, "Death": 0.0,
        }
        row[damage_column] = 1
        for role in ("Division", "Range", "Beat", "Village (reported)"):
            value = _value(entry, roles.get(role))
            row[role] = str(value).strip() if value not in (None, "") else ""
        for role in ("Total Count", "Male Count", "Female Count", "Calf Count"):
            if role in roles:
                row[role] = _number(_value(entry, roles[role]))
        for role in ("Injury", "Death", "Grain Damage"):
            if role in roles:
                row[role] = _number(_value(entry, roles[role]))
        rows.append(row)

    frame = pd.DataFrame(rows, columns=COLUMNS)
    if "Total Count" not in roles:
        # A damage report is evidence elephants were there even when the
        # form does not count them; presence scores off a non-zero count.
        frame["Total Count"] = frame["Total Count"].fillna(1)
    return frame


# ---------------------------------------------------------------------------
# Placing points in the forest boundaries
# ---------------------------------------------------------------------------
def _rings(geometry: dict) -> List[List[np.ndarray]]:
    """Polygons as lists of rings (outer first), as arrays of lon/lat."""
    coords = geometry.get("coordinates") or []
    polys = coords if geometry.get("type") == "MultiPolygon" else [coords]
    return [[np.asarray(ring, dtype=float) for ring in poly] for poly in polys]


def _inside(ring: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Even-odd ray cast of many points against one ring."""
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    inside = np.zeros(len(lon), dtype=bool)
    for xa, ya, xb, yb in zip(x, y, x2, y2):
        crosses = (ya > lat) != (yb > lat)
        if not crosses.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = xa + (lat - ya) * (xb - xa) / (yb - ya)
        inside ^= crosses & (lon < x_at)
    return inside


def _locate(level: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    from core import boundaries

    names = np.full(len(lon), "", dtype=object)
    for feature in boundaries.load(level)["features"]:
        for polygon in _rings(feature["geometry"]):
            outer = polygon[0]
            box = ((lon >= outer[:, 0].min()) & (lon <= outer[:, 0].max())
                   & (lat >= outer[:, 1].min()) & (lat <= outer[:, 1].max())
                   & (names == ""))
            if not box.any():
                continue
            idx = np.flatnonzero(box)
            hit = _inside(outer, lon[idx], lat[idx])
            for hole in polygon[1:]:
                hit &= ~_inside(hole, lon[idx], lat[idx])
            names[idx[hit]] = str(feature["properties"].get("name") or "")
    return names


def place_in_boundaries(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill blank Division, Range and Beat from the boundary polygons."""
    from core import boundaries

    out = frame.copy()
    if out.empty or not boundaries.available():
        for col in ("Division", "Range", "Beat"):
            out[col] = out[col].replace("", UNPLACED).fillna(UNPLACED)
        return out

    lon = pd.to_numeric(out["Longitude"], errors="coerce").to_numpy(float)
    lat = pd.to_numeric(out["Latitude"], errors="coerce").to_numpy(float)
    valid = np.isfinite(lon) & np.isfinite(lat)
    for col, level, missing in (
        ("Division", boundaries.DIVISION, UNPLACED),
        ("Range", boundaries.RANGE, UNPLACED),
        ("Beat", boundaries.BEAT, OUTSIDE_BEATS),
    ):
        blank = (out[col].fillna("").astype(str).str.strip() == "").to_numpy() & valid
        if blank.any():
            found = _locate(level, lon[blank], lat[blank])
            values = out[col].to_numpy(dtype=object)
            values[blank] = np.where(found == "", missing, found)
            out[col] = values
        out[col] = out[col].replace("", missing).fillna(missing)
    return out


# ---------------------------------------------------------------------------
# Incremental sync, kept for the life of the process
# ---------------------------------------------------------------------------
@dataclass
class _Store:
    structure: dict = field(default_factory=dict)
    entries: Dict[str, dict] = field(default_factory=dict)
    newest_upload: str = ""
    synced_at: float = 0.0


_STORES: Dict[str, _Store] = {}


def _entries_page(payload: dict) -> Tuple[List[dict], int]:
    data = payload.get("data") or {}
    entries = data.get("entries") if isinstance(data, dict) else None
    if entries is None and isinstance(data, dict):
        entries = data.get("data")
    if entries is None and isinstance(data, list):
        entries = data
    meta = payload.get("meta") or {}
    return list(entries or []), int(meta.get("last_page") or 1)


def _sync(client: _Client, slug: str, force: bool,
          progress: Optional[Callable[[str], None]]) -> _Store:
    store = _STORES.setdefault(slug, _Store())
    if not force and store.entries and time.time() - store.synced_at < MIN_RESYNC_SECONDS:
        return store

    if not store.structure:
        if progress:
            progress(f"Reading the {slug} form")
        store.structure = client.get("project", f"/api/export/project/{slug}",
                                     slug=slug)

    forms = _forms(store.structure)
    if not forms:
        raise EpicollectError(f"The {slug} project has no forms to read.")
    params = {"form_ref": forms[0].get("ref"), "per_page": PER_PAGE,
              "sort_by": "uploaded_at", "sort_order": "ASC"}
    if store.newest_upload:
        # uploaded_at moves on edits too, so this picks up corrections.
        params.update(filter_by="uploaded_at", filter_from=store.newest_upload)

    page, last_page = 1, 1
    while page <= last_page:
        if progress:
            progress(f"Fetching {slug}, page {page}"
                     + (f" of {last_page}" if last_page > 1 else ""))
        payload = client.get(
            "entries", f"/api/export/entries/{slug}", {**params, "page": page},
            slug=slug,
        )
        entries, last_page = _entries_page(payload)
        for entry in entries:
            uid = entry.get("ec5_uuid") or entry.get("uuid")
            if uid:
                store.entries[str(uid)] = entry
            uploaded = str(entry.get("uploaded_at") or "")
            if uploaded > store.newest_upload:
                store.newest_upload = uploaded
        page += 1
    store.synced_at = time.time()
    return store


@dataclass(frozen=True)
class EpicollectResult:
    """A completed pull, as CSV bytes for the shared loader."""

    data: bytes
    rows: int
    per_project: Dict[str, int]
    requests: int
    fetched_at: str
    errors: Dict[str, str] = field(default_factory=dict)


def fetch_damage_reports(
    projects: Optional[Dict[str, str]] = None,
    *,
    force: bool = False,
    base_url: str = BASE_URL,
    interval: float = REQUEST_INTERVAL_SECONDS,
    progress: Optional[Callable[[str], None]] = None,
) -> EpicollectResult:
    """Sync the damage projects and return them in dashboard columns.

    A project that cannot be read is reported in ``errors`` and the
    others still load.

    Raises:
        EpicollectError: with a message for the screen, when no project
            could be read. Entries already synced are kept, so a later
            retry only asks for what is new.
    """
    projects = projects or PROJECTS
    try:
        client = _Client(base_url, interval)
    except ImportError as error:
        raise EpicollectError(
            f"Fetching from Epicollect5 needs the 'requests' package ({error})."
        ) from error

    frames, counts, errors = [], {}, {}
    for slug, damage_column in projects.items():
        try:
            store = _sync(client, slug, force, progress)
        except EpicollectError as error:
            errors[slug] = str(error)
            continue
        except Exception as error:  # noqa: BLE001 - the network is the world
            logger.exception("Epicollect5 fetch failed for %s", slug)
            import requests

            if isinstance(error, requests.ConnectionError):
                errors[slug] = (
                    f"Could not reach {base_url} from this server. It may be "
                    "offline, or outbound access to it may be blocked where "
                    "the app is hosted."
                )
            else:
                errors[slug] = (
                    f"Could not read Epicollect5: {type(error).__name__}: {error}"
                )
            continue
        frame = entries_to_frame(
            store.structure, store.entries.values(), slug, damage_column
        )
        counts[slug] = len(frame)
        frames.append(frame)

    if errors and not frames:
        messages = list(dict.fromkeys(errors.values()))
        raise EpicollectError("\n\n".join(messages))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    combined = place_in_boundaries(combined)
    buffer = io.StringIO()
    combined.to_csv(buffer, index=False)
    return EpicollectResult(
        data=buffer.getvalue().encode("utf-8"),
        rows=len(combined),
        per_project=counts,
        requests=client.requests,
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        errors=errors,
    )


def reset() -> None:
    """Forget everything synced (tests, and a full re-pull)."""
    _STORES.clear()
    _LAST_REQUEST.clear()
    _TOKENS.clear()
