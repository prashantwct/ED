"""Pulling the register straight off Forest Alerts, from inside the app.

The landing page offers a sign-in as an alternative to uploading a CSV.
This is the bridge between that form and ``tools.fetch_sightings``: it
signs in, reads the listing, and hands back the same CSV bytes the
uploader would have produced, so everything downstream is unchanged.
Both paths end in ``core.data_loader``, and there is no second parser.

**The site is a Laravel + Inertia app.** Its login page is a script
shell with no postable form and its listing renders in the browser, so
the client signs in the way the page itself does -- POST /login with the
XSRF cookie and Inertia headers. Nothing has to be configured for that.

**The Export button, then the listing.** A fetch asks for a window --
by default from the day the register opens to the viewer's today -- and
downloads the listing's export for it: one request, where reading the
listing is one per page with a pause between. Only if no export route
can be found is the listing read page by page, over the same window, and
the result says which happened and why. ``FORESTALERTS_EXPORT_PATH``
pins the route once it is known.

**The dates on the result are what arrived.** The window is what was
asked for; ``start_date`` and ``end_date`` are the span of the rows that
came back, read off them afterwards, so the screen never reports a
window as covered when it was not.

**Credentials are a request parameter and nothing else.** They are read
off the form, used for one sign-in, and never written to session state,
a cache, a log or disk. That is why this module returns bytes rather
than caching itself on the credentials: a Streamlit cache keyed on a
password is a password at rest. The bytes are cached instead, by the
same ``_load`` the uploader already goes through.

**The app signs in as the person using it**, and their own account
decides what they may pull. Baking a shared login into ``secrets.toml``
would turn a tool with no access control into a public mirror of the
register, so there is deliberately no way to configure one.

The import of the client is deferred to the call. Nothing here runs
until someone actually signs in, and a dashboard that cannot rank beats
because an outbound-HTTP dependency failed to import is worse than one
without the fetch button.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

if TYPE_CHECKING:  # pandas is imported where it is used, not at load
    import pandas as pd

logger = logging.getLogger(__name__)

# A pause between page requests. Someone waiting at a screen is the
# reason to keep it short, and somebody else's production server is the
# reason it is not zero. Only the page-by-page fallback pays it.
PAGE_DELAY_SECONDS = 0.5

# The register opens on 1 October 2025, so a fetch starts there unless
# asked to start later.
START_DATE = date(2025, 10, 1)

SOURCE_EXPORT, SOURCE_LISTING = "export", "listing"


def observed_window(dates: "pd.Series") -> Tuple[str, str]:
    """The span the fetched rows actually cover, as ISO dates.

    Read off what arrived rather than stated in advance: the listing is
    not filtered by date, so any window named before the call would be a
    guess the screen then reports as fact.
    """
    import pandas as pd

    parsed = pd.to_datetime(dates, dayfirst=True, errors="coerce").dropna()
    if parsed.empty:
        return "", ""
    return parsed.min().date().isoformat(), parsed.max().date().isoformat()


def _unreachable(site: str, error: Exception) -> str:
    """A failed call, said in a sentence rather than dumped.

    A connection refused or timed out is the one failure a reader can
    actually do something about, and ``HTTPConnectionPool(host=...):
    Max retries exceeded with url: /login (Caused by NewConnectionError
    (...))`` does not tell them what. Anything else keeps its type and
    message, which is the most that can honestly be said about it.
    """
    try:
        import requests

        network = (requests.exceptions.ConnectionError,
                   requests.exceptions.Timeout)
    except ImportError:  # pragma: no cover - the client imported it already
        network = ()
    if network and isinstance(error, network):
        return (
            f"Could not reach {site}. Check the network where the app is "
            "running, then try again. The register can also be uploaded as "
            "a CSV instead."
        )
    return f"The fetch failed: {type(error).__name__}: {error}"


@dataclass(frozen=True)
class FetchResult:
    """A completed pull: the CSV, and what to say about it on screen."""

    data: bytes
    name: str
    rows: int
    pages: int
    columns: List[str]
    start_date: str
    end_date: str
    fetched_at: str
    # The window asked for, as ISO dates.
    window_start: str = ""
    window_end: str = ""
    # SOURCE_EXPORT, or SOURCE_LISTING when no export could be found.
    source: str = SOURCE_EXPORT
    # Why the listing was read instead of the export, when it was.
    note: str = ""


class FetchError(RuntimeError):
    """The pull did not happen, with a reason worth showing a user."""


def fetch_sightings(
    email: str = "",
    password: str = "",
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    today: Optional[date] = None,
    export_path: Optional[str] = None,
    base_url: Optional[str] = None,
    delay: float = PAGE_DELAY_SECONDS,
    progress: Optional[Callable[[int, int], None]] = None,
    stage: Optional[Callable[[str], None]] = None,
) -> FetchResult:
    """Sign in, download the register for a window, and return it as CSV bytes.

    The window runs from ``start_date`` (default: the day the register
    opens) to ``end_date`` (default: ``today``). ``today`` is the
    viewer's local date; the server's is only the fallback, because a
    server on UTC is still on yesterday for the first hours of an Indian
    morning. ``export_path`` pins the Export button's route. ``stage``
    hears each step in words, ``progress`` each page of the fallback.

    ``base_url`` overrides the site and ``delay`` the pause between page
    requests; both exist so this can be driven against the test replica
    and a staging instance. The app sets neither.

    Raises:
        FetchError: With a message meant for the screen -- a rejected
            sign-in, a network failure. Never a traceback.
    """
    try:
        from tools.fetch_sightings import (
            BASE_URL,
            ExportUnavailable,
            FetchError as ClientError,
            ForestAlertsClient,
            rows_to_csv_text,
        )
        from tools.scrape_sightings import order_columns
    except ImportError as error:  # requests missing from the deployment
        raise FetchError(
            "The fetch feature needs the 'requests' package, which is not "
            f"installed here ({error}). Upload the CSV export instead, or "
            "add requests to requirements.txt and reboot the app."
        ) from error

    if not (email and password):
        raise FetchError(
            "Enter your Forest Alerts email and password to fetch the register. "
            "The sign-in happens once and nothing is stored."
        )

    stamp = (today or date.today()).isoformat()
    first_day = start_date or START_DATE
    last_day = end_date or today or date.today()
    if last_day < first_day:
        raise FetchError(
            f"The end date ({last_day:%d %b %Y}) is before the start date "
            f"({first_day:%d %b %Y})."
        )
    window = (first_day.isoformat(), last_day.isoformat())

    def _say(text: str) -> None:
        if stage is not None:
            stage(text)

    pages = 0

    def _on_page(page: int, rows: int) -> None:
        nonlocal pages
        pages = page
        if progress is not None:
            progress(page, rows)

    site = base_url or BASE_URL
    source, note = SOURCE_EXPORT, ""
    try:
        client = ForestAlertsClient(site, verify=True)
        client.login(email, password)
        _say(f"Downloading the export for {first_day:%d %b %Y} to {last_day:%d %b %Y}...")
        try:
            records, _ = client.export_sightings(*window, export_path=export_path)
            pages = 1
            text, columns = _rows_to_csv(records, order_columns)
        except ExportUnavailable as gap:
            logger.warning("%s", gap)
            source, note = SOURCE_LISTING, str(gap)
            _say("No export found; reading the listing page by page instead...")
            records = client.fetch_sightings(delay=delay, on_page=_on_page,
                                             start_date=window[0], end_date=window[1])
            text, columns = rows_to_csv_text(records)
    except ClientError as error:
        # Deliberately not chained into the UI: the message is written to
        # be read by a forest officer, and a traceback is not.
        raise FetchError(str(error)) from error
    except Exception as error:  # noqa: BLE001 -- the network is the world
        logger.exception("Fetch from Forest Alerts failed")
        raise FetchError(_unreachable(site, error)) from error

    if not records:
        raise FetchError(
            f"No sightings were recorded between {first_day:%d %b %Y} and "
            f"{last_day:%d %b %Y}. Widen the dates and fetch again."
        )

    import pandas as pd

    frame = pd.read_csv(io.StringIO(text), dtype=str) if text.strip() else pd.DataFrame()
    first, last = observed_window(frame.get("Date", pd.Series(dtype=str)))
    span = f"{first}-to-{last}" if first else stamp
    return FetchResult(
        data=text.encode("utf-8-sig"),
        name=f"forestalerts-{span}.csv",
        rows=len(records),
        pages=pages,
        columns=columns,
        start_date=first,
        end_date=last,
        fetched_at=stamp,
        window_start=window[0],
        window_end=window[1],
        source=source,
        note=note,
    )


def _rows_to_csv(records, order_columns) -> Tuple[str, List[str]]:
    """Export rows, already under the dashboard's names, as CSV text.

    An export with no rows is an empty window, not a failure: it comes
    back as an empty file the caller reports.
    """
    import csv

    if not records:
        return "", []
    columns = order_columns(records)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow({column: record.get(column, "") for column in columns})
    return buffer.getvalue(), columns
