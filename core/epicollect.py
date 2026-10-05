"""Reading the herd damage survey projects from Epicollect5.

Two Epicollect5 projects hold household damage surveys made in the
field, separately from the Gaj Rakshak sighting register:
``herd-crop-damage`` and ``herd-house-damage``. This syncs their entries
over the export API and hands them, as the API returns them, to
``core.damage``, which reads them the same way as the CSV export.

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

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)

BASE_URL = "https://five.epicollect.net"

# The household damage survey projects.
PROJECTS: Tuple[str, ...] = ("herd-crop-damage", "herd-house-damage")

PER_PAGE = 500  # the API maximum
# 5 requests/minute per kind is the cap; the docs ask for about half.
REQUEST_INTERVAL_SECONDS = 24.0
# A sync newer than this is reused without asking the server at all.
MIN_RESYNC_SECONDS = 15 * 60
TIMEOUT_SECONDS = 60

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


def _forms(structure: dict) -> List[dict]:
    project = (structure.get("data") or structure).get("project") or {}
    return project.get("forms") or []


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
class EntriesResult:
    """A completed sync: each project's entries, and what failed."""

    entries: Dict[str, List[dict]]
    requests: int
    fetched_at: str
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return sum(len(v) for v in self.entries.values())


def fetch_entries(
    projects: Optional[Iterable[str]] = None,
    *,
    force: bool = False,
    base_url: str = BASE_URL,
    interval: float = REQUEST_INTERVAL_SECONDS,
    progress: Optional[Callable[[str], None]] = None,
) -> EntriesResult:
    """Sync the damage projects and return their entries as the API gives them.

    A project that cannot be read is reported in ``errors`` and the
    others still load.

    Raises:
        EpicollectError: with a message for the screen, when no project
            could be read. Entries already synced are kept, so a later
            retry only asks for what is new.
    """
    projects = list(projects or PROJECTS)
    try:
        client = _Client(base_url, interval)
    except ImportError as error:
        raise EpicollectError(
            f"Fetching from Epicollect5 needs the 'requests' package ({error})."
        ) from error

    entries: Dict[str, List[dict]] = {}
    errors: Dict[str, str] = {}
    for slug in projects:
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
        entries[slug] = list(store.entries.values())

    if errors and not entries:
        raise EpicollectError("\n\n".join(dict.fromkeys(errors.values())))
    return EntriesResult(
        entries=entries,
        requests=client.requests,
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        errors=errors,
    )


def reset() -> None:
    """Forget everything synced (tests, and a full re-pull)."""
    _STORES.clear()
    _LAST_REQUEST.clear()
    _TOKENS.clear()
