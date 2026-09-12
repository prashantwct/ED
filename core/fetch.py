"""Pulling the register straight off Forest Alerts, from inside the app.

The landing page offers a sign-in as an alternative to uploading a CSV.
This is the bridge between that form and ``tools.scrape_sightings``: it
logs in, walks the listing, and hands back the same CSV bytes the
uploader would have produced, so everything downstream is unchanged.
Both paths end in ``core.data_loader``, and there is no second parser.

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

The import of the scraper is deferred to the call. Nothing here runs
until someone actually signs in, and a dashboard that cannot rank beats
because an outbound-HTTP dependency failed to import is worse than one
without the fetch button.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# A pause between page requests. Someone waiting at a screen is the
# reason to keep it short, and somebody else's production server is the
# reason it is not zero.
PAGE_DELAY_SECONDS = 0.5

# Only used to describe the window when the scraper cannot be imported at
# all; the real value is read from it below.
FALLBACK_START_DATE = "2025-10-01"


def window(end_date: Optional[str] = None) -> Tuple[str, str]:
    """The window a fetch would use, for saying so on screen.

    Read from the scraper rather than restated, so the sentence on the
    landing page cannot drift from what the fetch actually asks for.
    """
    try:
        from tools.scrape_sightings import START_DATE, today
    except ImportError:
        return FALLBACK_START_DATE, end_date or date.today().isoformat()
    return START_DATE, end_date or today()


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
    cookie: str = "",
    end_date: Optional[str] = None,
    base_url: Optional[str] = None,
    delay: float = PAGE_DELAY_SECONDS,
    progress: Optional[Callable[[int, int], None]] = None,
) -> FetchResult:
    """Sign in, scrape the register, and return it as CSV bytes.

    ``base_url`` overrides the site and ``delay`` the pause between page
    requests; both exist so this can be driven against the test replica
    and a staging instance. The app sets neither.

    Raises:
        FetchError: With a message meant for the screen -- a rejected
            sign-in, a listing that renders in JavaScript, a network
            failure. Never a traceback.
    """
    try:
        from tools.scrape_sightings import (
            BASE_URL,
            START_DATE,
            ScrapeError,
            records_to_csv,
            scrape,
            today,
        )
    except ImportError as error:  # requests missing from the deployment
        raise FetchError(
            "The fetch feature needs the 'requests' package, which is not "
            f"installed here ({error}). Upload the CSV export instead, or "
            "add requests to requirements.txt and reboot the app."
        ) from error

    window_end = end_date or today()
    pages = 0

    def _on_page(page: int, rows: int) -> None:
        nonlocal pages
        pages = page
        if progress is not None:
            progress(page, rows)

    try:
        records = scrape(
            base_url=base_url or BASE_URL,
            email=email or None,
            password=password or None,
            cookie=cookie or None,
            start_date=START_DATE,
            end_date=window_end,
            delay=delay,
            progress=_on_page,
        )
        text, columns = records_to_csv(records)
    except ScrapeError as error:
        # Deliberately not chained into the UI: the message is written to
        # be read by a forest officer, and a traceback is not.
        raise FetchError(str(error)) from error
    except Exception as error:  # noqa: BLE001 -- the network is the world
        logger.exception("Fetch from Forest Alerts failed")
        raise FetchError(
            f"The fetch failed: {type(error).__name__}: {error}"
        ) from error

    stamp = date.today().isoformat()
    return FetchResult(
        data=text.encode("utf-8"),
        name=f"forestalerts-{START_DATE}-to-{window_end}.csv",
        rows=len(records),
        pages=pages,
        columns=columns,
        start_date=START_DATE,
        end_date=window_end,
        fetched_at=stamp,
    )
