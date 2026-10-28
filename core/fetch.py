"""Pulling the register straight off Forest Alerts, from inside the app.

The landing page offers a sign-in as an alternative to uploading a CSV.
This is the bridge between that form and ``tools.fetch_sightings``: it
signs in, reads the listing, and hands back the same CSV bytes the
uploader would have produced, so everything downstream is unchanged.
Both paths end in ``core.data_loader``, and there is no second parser.

**The site is a Laravel + Inertia app.** Its login page is a script
shell with no postable form and its listing renders in the browser, so
the client signs in the way the page itself does -- POST /login with the
XSRF cookie and Inertia headers -- and reads the listing as Inertia
JSON, page by page. Nothing has to be configured for that: there is no
endpoint to discover and no cookie to paste.

**A fetch is the whole register.** The listing is not filtered by date,
so there is no window to ask for and none to advertise. What comes back
is everything the account can see; the dates on the result are the span
of the rows that arrived, read off them afterwards rather than promised
in advance.

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
# reason it is not zero.
PAGE_DELAY_SECONDS = 0.5

# Only used to describe the window when the scraper cannot be imported at
# all; the real value is read from it below.
FALLBACK_START_DATE = "2025-10-01"


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


class FetchError(RuntimeError):
    """The pull did not happen, with a reason worth showing a user."""


def fetch_sightings(
    email: str = "",
    password: str = "",
    *,
    base_url: Optional[str] = None,
    delay: float = PAGE_DELAY_SECONDS,
    progress: Optional[Callable[[int, int], None]] = None,
) -> FetchResult:
    """Sign in, read the register, and return it as CSV bytes.

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
            FetchError as ClientError,
            ForestAlertsClient,
            rows_to_csv_text,
        )
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

    pages = 0

    def _on_page(page: int, rows: int) -> None:
        nonlocal pages
        pages = page
        if progress is not None:
            progress(page, rows)

    site = base_url or BASE_URL
    try:
        client = ForestAlertsClient(site, verify=True)
        client.login(email, password)
        # The site's listing is the whole register; it is not windowed by
        # date here, so a fetch is always a full, gap-free pull.
        records = client.fetch_sightings(delay=delay, on_page=_on_page)
        text, columns = rows_to_csv_text(records)
    except ClientError as error:
        # Deliberately not chained into the UI: the message is written to
        # be read by a forest officer, and a traceback is not.
        raise FetchError(str(error)) from error
    except Exception as error:  # noqa: BLE001 -- the network is the world
        logger.exception("Fetch from Forest Alerts failed")
        raise FetchError(_unreachable(site, error)) from error

    stamp = date.today().isoformat()
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
    )
