"""Pull the sightings register off the Forest Alerts admin site.

Gaj Rakshak (mpforest.forestalerts.com) is a Laravel + Inertia.js app. The
admin pages have no fillable HTML form and no separate REST API -- instead
every page request returns its data as JSON when asked for an Inertia
response. This script logs in with email/password the way that page does.

**The listing's Export button comes first.** It is one request for the
whole window, where reading the listing is one request per page with a
pause between each -- minutes against seconds on a register of a few
thousand rows. :meth:`ForestAlertsClient.export_sightings` finds the
button's route (from the route table Laravel ships to the page, failing
that a few likely paths), downloads the file for the window and reads it
whatever format it arrives in. Where no export can be found, the listing
is still read page by page, over the same window, so a fetch never comes
back empty-handed because a route was guessed wrong.

Because it authenticates itself, it needs no copied browser cookie and can
run unattended on a schedule. Point it at a daily cron / Task Scheduler job
and it refreshes the whole register each run.

Run it:

    export FORESTALERTS_EMAIL=you@example.com
    export FORESTALERTS_PASSWORD=...
    python -m tools.fetch_sightings -o data/sightings.csv

The window runs from the day the register opens (``START_DATE``) to today
unless narrowed, so a re-run is a full refresh rather than an increment to
be merged.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
import io
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote

import requests

from tools.export_source import (
    ExportFormatError,
    detect_format,
    records_from_csv_text,
    suggested_filename,
    to_csv_text,
)

logger = logging.getLogger("fetch_sightings")

BASE_URL = "https://mpforest.forestalerts.com"
LOGIN_PATH = "/login"
SIGHTINGS_PATH = "/admin/sightings"
DEFAULT_OUT = Path("data/sightings.csv")

# The register opens on 1 October 2025; there is nothing before it.
START_DATE = "2025-10-01"

# The listing's filters, sent empty as the unfiltered listing sends them,
# with the window beside them. Named rather than omitted: a filter the
# site defaults to non-empty would otherwise narrow the pull unseen.
LISTING_FILTERS: Dict[str, str] = {
    "beat": "",
    "damage_type": "",
    "division": "",
    "elephant_visible": "",
    "range": "",
    "search": "",
}

# Where the Export button's route is tried when the page's own route
# table does not name it. A wrong guess costs one 404.
EXPORT_PATHS = (
    "/admin/sightings/export",
    "/admin/sightings-export",
    "/admin/export/sightings",
)

# What a download asks for: the file types an admin panel exports, then
# anything, so a server that keys its response on Accept sends a file.
_EXPORT_ACCEPT = (
    "text/csv, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, "
    "application/vnd.ms-excel, application/octet-stream;q=0.9, */*;q=0.8"
)

# What makes a downloaded file the sightings register and not another.
_SIGHTING_COLUMNS = frozenset({"Date", "Latitude", "Longitude"})

# The route table Ziggy's @routes directive writes into the page.
_ZIGGY = re.compile(r"\bZiggy\s*=\s*\{")

# data-page="{...}" is where Inertia hides the page payload (and its version)
# in the initial HTML document.
_DATA_PAGE = re.compile(r'data-page="([^"]+)"')

# The CSV is written with the column names the dashboard expects
# (core/data_loader.py: REQUIRED_COLUMNS, OPTIONAL_NUMERIC_COLUMNS), so the
# output uploads straight into the app with no renaming.

# Nested {..,"name":..} objects -> the dashboard column that takes the name.
_NAME_FIELD_MAP = {
    "state": "State",
    "region": "Region",
    "circle": "Circle",
    "division": "Division",
    "range": "Range",
    "beat": "Beat",
    "sub_beat": "Sub Beat",
}

# Flat scalar API fields -> dashboard column names.
_FIELD_MAP = {
    "id": "ID",
    "date": "Date",
    "time": "Time",
    "title": "Title",
    "elephant_visible": "Elephant Visible",
    "total_count": "Total Count",
    "male_count": "Male Count",
    "female_count": "Female Count",
    "calf_count": "Calf Count",
    "unknown_count": "Unknown Count",
    "crop_damage": "Crop Damage",
    "grain_damage": "Grain Damage",
    "house_damage": "House Damage",
    "injury": "Injury",
    "death": "Death",
    "male_death_count": "Male Death Count",
    "female_death_count": "Female Death Count",
    "is_approved": "Is Approved",
    "createdBy": "Reported By",
    "createdByMobile": "Reported By Mobile",
    "created_at": "Created At",
    "image_url": "Image URL",
}

# Preferred CSV column order: the dashboard's required columns first, then the
# rest in a sensible reading order. Anything unmapped is appended after these.
_COLUMN_ORDER = [
    "ID", "Date", "Time", "Latitude", "Longitude",
    "State", "Division", "Range", "Beat", "Sub Beat",
    "Title", "Sighting Type", "Elephant Visible",
    "Total Count", "Male Count", "Female Count", "Calf Count", "Unknown Count",
    "Crop Damage", "Grain Damage", "House Damage", "Injury", "Death",
    "Male Death Count", "Female Death Count",
    "Is Approved", "Reported By", "Reported By Mobile", "Created At", "Image URL",
]


class FetchError(RuntimeError):
    """A step failed in a way the caller should see spelled out."""


class ExportUnavailable(FetchError):
    """No route on the site handed back an export file.

    Not a failure of the fetch as a whole: the listing can still be read
    page by page. The message lists what was tried, for whoever pins the
    real route.
    """


def _routes_in(node: Any, found: Dict[str, Tuple[str, Tuple[str, ...]]]) -> None:
    """Collect ``name -> (uri, methods)`` from every Ziggy-style route table."""
    if isinstance(node, dict):
        if node and all(isinstance(v, dict) and "uri" in v for v in node.values()):
            for name, route in node.items():
                methods = tuple(m.upper() for m in route.get("methods") or ("GET",))
                found[str(name)] = (str(route["uri"]), methods)
            return
        for value in node.values():
            _routes_in(value, found)
    elif isinstance(node, list):
        for value in node:
            _routes_in(value, found)


def _export_strings(node: Any, found: List[str]) -> None:
    """Every string in the payload that reads as a link to an export."""
    if isinstance(node, dict):
        for value in node.values():
            _export_strings(value, found)
    elif isinstance(node, list):
        for value in node:
            _export_strings(value, found)
    elif isinstance(node, str) and "export" in node.lower() and (
            node.startswith("/") or node.startswith("http")) and " " not in node:
        found.append(node)


def _ziggy_from_html(page_html: str) -> Dict[str, Any]:
    """The route table ``@routes`` wrote into the page, if it did."""
    match = _ZIGGY.search(page_html or "")
    if not match:
        return {}
    try:
        table, _ = json.JSONDecoder().raw_decode(page_html, match.end() - 1)
    except ValueError:
        return {}
    return table if isinstance(table, dict) else {}


def export_candidates(
    payload: Optional[Dict[str, Any]],
    page_html: str = "",
    explicit: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Where the Export button might point, best lead first, as (path, method).

    A path configured by hand wins. Then routes from the table Laravel
    ships to the page (Ziggy, in the Inertia props or written by
    ``@routes``) whose name or path says both export and sightings, then
    links to an export found anywhere in the props, then the usual
    paths. A route with a parameter in it cannot be filled from here and
    is skipped.
    """
    routes: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
    _routes_in(payload or {}, routes)
    _routes_in(_ziggy_from_html(page_html), routes)
    links: List[str] = []
    _export_strings((payload or {}).get("props", {}), links)

    ranked: List[Tuple[int, str, str]] = []
    if explicit:
        ranked.append((0, explicit, "GET"))
    for name, (uri, methods) in routes.items():
        words = f"{name} {uri}".lower()
        if "export" not in words or "{" in uri:
            continue
        method = "GET" if "GET" in methods or not methods else methods[0]
        ranked.append((1 if "sight" in words else 3, "/" + uri.lstrip("/"), method))
    for link in links:
        ranked.append((2 if "sight" in link.lower() else 3, link, "GET"))
    for path in EXPORT_PATHS:
        ranked.append((4, path, "GET"))

    seen = set()
    out: List[Tuple[str, str]] = []
    for _, path, method in sorted(ranked, key=lambda item: item[0]):
        key = (path.split("?")[0].rstrip("/"), method)
        if key not in seen:
            seen.add(key)
            out.append((path, method))
    return out


class ForestAlertsClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        *,
        timeout: float = 30.0,
        verify: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = verify
        self.session.headers.update({
            "User-Agent": "ED-sightings-fetch/1.0 (+wildlife monitoring)",
        })
        self._version: Optional[str] = None
        # The listing page as last loaded, kept for the routes it names.
        self._payload: Optional[Dict[str, Any]] = None
        self._page_html: str = ""

    # -- low level --------------------------------------------------------

    def _url(self, path: str) -> str:
        return path if path.startswith("http") else f"{self.base_url}{path}"

    def _xsrf_header(self) -> Dict[str, str]:
        """Laravel checks X-XSRF-TOKEN against the (URL-encoded) cookie."""
        token = self.session.cookies.get("XSRF-TOKEN")
        return {"X-XSRF-TOKEN": unquote(token)} if token else {}

    def _page_version(self, path: str = SIGHTINGS_PATH) -> str:
        """Read the current Inertia asset version out of a full page load.

        The version changes whenever the site is redeployed; an Inertia
        request carrying a stale version is answered with 409, so we always
        read a fresh one rather than hard-coding it.
        """
        resp = self.session.get(
            self._url(path),
            headers={"X-Requested-With": "XMLHttpRequest"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        match = _DATA_PAGE.search(resp.text)
        if not match:
            raise FetchError(
                f"No Inertia payload on {path}; the site's shape may have "
                "changed, or the session is not authenticated."
            )
        payload = json.loads(html.unescape(match.group(1)))
        version = payload.get("version")
        if not version:
            raise FetchError("Inertia payload carried no version string.")
        self._version = version
        if path == SIGHTINGS_PATH:
            self._payload, self._page_html = payload, resp.text
        return version

    def _inertia_headers(self) -> Dict[str, str]:
        return {
            "X-Inertia": "true",
            "X-Inertia-Version": self._version or "",
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json",
        }

    # -- public steps -----------------------------------------------------

    def login(self, email: str, password: str) -> None:
        """Authenticate the session with email/password.

        Mirrors what the login form does: GET /login to seed the CSRF
        cookie and asset version, then POST the credentials as an Inertia
        request.
        """
        version = self._page_version(LOGIN_PATH)
        headers = {
            **self._inertia_headers(),
            **self._xsrf_header(),
            "Content-Type": "application/json",
        }
        resp = self.session.post(
            self._url(LOGIN_PATH),
            json={"email": email, "password": password, "remember": True},
            headers=headers,
            timeout=self.timeout,
            allow_redirects=True,
        )

        # Inertia replies 422 (with validation errors) on a bad login, or
        # 302/200 on success. Laravel rotates the session cookie on success,
        # so the surest check is whether we can now read an admin page.
        if resp.status_code == 422:
            raise FetchError(
                "Login rejected (bad email or password). The server's reply: "
                + _first_error(resp)
            )
        if resp.status_code >= 400 and resp.status_code != 409:
            raise FetchError(
                f"Login POST returned HTTP {resp.status_code}. Check the "
                "email, password, and that the account is active."
            )
        # Prove we are in: a fresh version read off an admin page will fail
        # if the session is still anonymous (it redirects to /login).
        self._page_version(SIGHTINGS_PATH)
        logger.info("Signed in as %s", email)

    def export_sightings(
        self,
        start_date: str,
        end_date: str,
        export_path: Optional[str] = None,
    ) -> Tuple[List[Dict[str, str]], str]:
        """Download the listing's export for the window, as dashboard rows.

        Returns the rows, renamed to the dashboard's columns through the
        same synonym table the scraper uses, and the URL that served
        them. Each candidate route is asked once with the listing's
        filters and the window; one that answers with a page rather than
        a file (the listing itself, a login shell, a queued-export
        notice) is passed over.

        Raises:
            ExportUnavailable: No candidate handed back a file.
        """
        from tools.scrape_sightings import map_record

        if self._payload is None:
            self._page_version(SIGHTINGS_PATH)
        params = {**LISTING_FILTERS, "start_date": start_date, "end_date": end_date}
        tried: List[str] = []
        for path, method in export_candidates(self._payload, self._page_html, export_path):
            url = self._url(path if path.startswith("http") else "/" + path.lstrip("/"))
            headers = {"Accept": _EXPORT_ACCEPT}
            if method == "POST":
                resp = self.session.post(url, data=params, timeout=self.timeout,
                                         headers={**headers, **self._xsrf_header()})
            else:
                resp = self.session.get(url, params=params, headers=headers,
                                        timeout=self.timeout)
            if resp.status_code != 200:
                tried.append(f"{method} {path}: HTTP {resp.status_code}")
                continue
            filename = suggested_filename(resp.headers.get("Content-Disposition", ""))
            try:
                fmt = detect_format(resp.content, resp.headers.get("Content-Type", ""),
                                    filename)
                if fmt == "html":
                    tried.append(f"{method} {path}: a page, not a file")
                    continue
                text = to_csv_text(resp.content, fmt)
            except ExportFormatError as error:
                tried.append(f"{method} {path}: {error}")
                continue
            rows = [map_record(row) for row in records_from_csv_text(text)]
            # A generic export route may hand back some other register;
            # sightings are the file whose rows carry a date and a place.
            if rows and not _SIGHTING_COLUMNS.issubset(rows[0]):
                tried.append(f"{method} {path}: a file without sighting columns")
                continue
            logger.info("Export from %s: %s, %d bytes", path, fmt, len(resp.content))
            return rows, resp.url
        raise ExportUnavailable(
            "The site's export could not be found, so the listing was read page "
            "by page instead. Tried: " + "; ".join(tried) + ". Set "
            "FORESTALERTS_EXPORT_PATH to the address the Export button requests "
            "(browser developer tools, Network tab) to use it directly."
        )

    def fetch_sightings(
        self,
        delay: float = 0.5,
        on_page: Optional[Callable[[int, int], None]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Page through the sightings listing and return every row.

        ``on_page(page, rows_so_far)`` is called after each page, so a
        caller (the app's status panel) can report progress. A window, when
        given, is sent with the listing's filters as the page itself
        sends it.
        """
        if self._version is None:
            self._page_version(SIGHTINGS_PATH)
        window: Dict[str, str] = {}
        if start_date or end_date:
            window = {**LISTING_FILTERS, "start_date": start_date or START_DATE,
                      "end_date": end_date or ""}

        rows: List[Dict[str, Any]] = []
        page = 1
        last_page = 1
        while page <= last_page:
            data = self._get_page(page, window)
            block = data["props"]["sightings"]
            rows.extend(block.get("data", []))
            meta = block.get("meta", {})
            last_page = int(meta.get("last_page", page))
            total = meta.get("total")
            logger.info(
                "page %d/%d  (%d rows so far%s)",
                page, last_page, len(rows),
                f" of {total}" if total else "",
            )
            if on_page is not None:
                on_page(page, len(rows))
            page += 1
            if page <= last_page and delay:
                time.sleep(delay)
        return rows

    def _get_page(self, page: int, window: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """One listing page as Inertia JSON, refreshing version on 409."""
        for attempt in (1, 2):
            resp = self.session.get(
                self._url(SIGHTINGS_PATH),
                params={**(window or {}), "page": page},
                headers=self._inertia_headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 409 and attempt == 1:
                # Asset version moved under us -- read it again and retry.
                self._page_version(SIGHTINGS_PATH)
                continue
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError as exc:  # pragma: no cover - server-shaped
                raise FetchError(
                    f"Page {page} did not return JSON; the session may have "
                    "expired mid-run."
                ) from exc
        raise FetchError(f"Page {page} kept returning 409 (version mismatch).")


def _first_error(resp: requests.Response) -> str:
    try:
        payload = resp.json()
    except ValueError:
        return resp.text[:200]
    errors = payload.get("props", {}).get("errors") or payload.get("errors")
    if isinstance(errors, dict) and errors:
        return "; ".join(f"{k}: {v}" for k, v in errors.items())
    return json.dumps(errors or payload)[:200]


# -- flattening to CSV ----------------------------------------------------

def flatten(row: Dict[str, Any]) -> Dict[str, Any]:
    """Turn one nested sighting into a flat dict under dashboard column names."""
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if key in _NAME_FIELD_MAP:
            out[_NAME_FIELD_MAP[key]] = (
                value.get("name") if isinstance(value, dict) else ""
            )
        elif key == "geolocation":
            geo = value if isinstance(value, dict) else {}
            out["Latitude"] = geo.get("lat")
            out["Longitude"] = geo.get("lng")
        elif key == "sighting_type":
            # usually a list (["pugMark"]), occasionally a {"0": "..."} object
            if isinstance(value, dict):
                out["Sighting Type"] = ", ".join(str(v) for v in value.values())
            elif isinstance(value, list):
                out["Sighting Type"] = ", ".join(str(v) for v in value)
            else:
                out["Sighting Type"] = value
        elif key in _FIELD_MAP:
            out[_FIELD_MAP[key]] = value
        elif key.endswith("_id"):
            # foreign-key ids duplicate the named objects above; drop them.
            continue
        elif isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


def _columns_for(flat: List[Dict[str, Any]]) -> List[str]:
    """Preferred dashboard columns first, then any extras in first-seen order."""
    seen: List[str] = []
    seen_set = set()
    for r in flat:
        for k in r:
            if k not in seen_set:
                seen_set.add(k)
                seen.append(k)
    columns = [c for c in _COLUMN_ORDER if c in seen_set]
    columns += [c for c in seen if c not in set(columns)]
    return columns


def rows_to_csv_text(rows: Iterable[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Flatten raw sighting rows into CSV text and its column list.

    Shared by the command-line writer and the in-app fetch, so both paths
    produce byte-identical output.
    """
    flat = [flatten(r) for r in rows]
    if not flat:
        return "", []
    columns = _columns_for(flat)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(flat)
    return buffer.getvalue(), columns


def write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    rows = list(rows)
    text, _ = rows_to_csv_text(rows)
    if not text:
        logger.warning("No rows returned; nothing written.")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel opens the Hindi/Devanagari names correctly.
    path.write_text(text, encoding="utf-8-sig", newline="")
    return len(rows)


def run(
    *,
    email: str,
    password: str,
    out: Path,
    base_url: str = BASE_URL,
    delay: float = 0.5,
    timeout: float = 30.0,
    verify: bool = True,
) -> int:
    client = ForestAlertsClient(base_url, timeout=timeout, verify=verify)
    client.login(email, password)
    rows = client.fetch_sightings(delay=delay)
    written = write_csv(rows, out)
    logger.info("Wrote %d sightings to %s", written, out)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                        help=f"CSV output path (default: {DEFAULT_OUT}).")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--email", default=os.environ.get("FORESTALERTS_EMAIL"),
                        help="Prefer the FORESTALERTS_EMAIL environment variable.")
    parser.add_argument("--password", default=os.environ.get("FORESTALERTS_PASSWORD"),
                        help="Prefer the FORESTALERTS_PASSWORD environment variable.")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to pause between pages (default: 0.5).")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip TLS verification (last resort).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    if not args.email or not args.password:
        parser.error(
            "No credentials. Set FORESTALERTS_EMAIL and FORESTALERTS_PASSWORD, "
            "or pass --email/--password."
        )

    try:
        run(
            email=args.email,
            password=args.password,
            out=args.out,
            base_url=args.base_url,
            delay=args.delay,
            timeout=args.timeout,
            verify=not args.no_verify,
        )
    except (FetchError, requests.RequestException) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
