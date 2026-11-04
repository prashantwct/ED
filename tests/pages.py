"""Opening one of the app's pages under AppTest.

The app's pages are functions registered with ``st.navigation`` and
identified by their URL path. ``AppTest.switch_page`` only takes file
paths, and AppTest does not carry an ``st.switch_page`` over to the next
run the way a browser's address bar does, so tests name the page here.
It sets the same hash Streamlit computes for a function page.
"""

from streamlit.util import calc_hash


def open_page(at, url_path: str):
    """Run ``at`` on the page at ``url_path`` (``"overview"`` for the home page)."""
    at._page_hash = calc_hash(url_path)
    return at
