"""Fixtures shared across the test suite."""

import threading
from http.server import ThreadingHTTPServer

import pytest

from replica import Handler


@pytest.fixture
def site():
    """A running replica; yields its base URL."""
    Handler.requests = []
    Handler.expire_on_page = None
    Handler.expired = set()
    Handler.advertise = (1, 2, 3)
    Handler.login_path = "/login"
    Handler.login_html = None
    Handler.api_requests = []
    Handler.api_needs_token = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
