"""A local stand-in for the Forest Alerts admin site.

The real site is not reachable from CI, and code that is only ever
exercised against it is code nobody can change. This serves a listing
over ``http.server`` so the scraper and the app's fetch bridge can both
be driven end to end against something that behaves like the real thing.

It deliberately does not serve tidy HTML. The header spellings differ
from the dashboard's, the date cell carries a 12-hour time, there is a
decorative summary table above the listing and a ``<table>`` inside a
``<script>``, a cell uses an entity and a ``<br>``, and asking for a page
past the end re-serves the last one -- each of which is a way a real
admin panel has of breaking a parser that assumed its own output.
"""

import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

EMAIL = "ranger@example.org"
PASSWORD = "correct horse"
CSRF = "tok-abc123"
SESSION_COOKIE = "fa_session=live"
PAGE_SIZE = 4
TOTAL_ROWS = 9  # three pages, the last one short


def _row(index):
    """One sighting, in the shapes the listing actually prints."""
    return {
        "id": 100 + index,
        "date": f"{(index % 28) + 1:02d}-10-2025 07:{index % 60:02d} PM",
        "lat": round(23.10 + index * 0.01, 4),
        "lng": round(81.20 + index * 0.01, 4),
        "division": "Shahdol" if index % 2 else "Anuppur",
        "range": "Jaisinghnagar" if index % 2 else "Kotma",
        "beat": f"Beat {index:02d}",
        "total": index % 5,
        "male": index % 2,
        "crop": index % 3,
        "death": 1 if index == 3 else 0,
    }


ROWS = [_row(index) for index in range(TOTAL_ROWS)]

LOGIN_PAGE = f"""<!doctype html><html><head><title>Sign in</title></head><body>
<form method="post" action="/login">
  <input type="hidden" name="_token" value="{CSRF}">
  <input type="email" name="email" value="">
  <input type="password" name="password">
  <button type="submit">Login</button>
</form></body></html>"""


def _listing(page, rows, advertise=(1, 2, 3)):
    """A page of the listing, wrapped in the noise a real one carries."""
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['id']}</td>"
            f"<td>{row['date']}</td>"
            f"<td>{row['lat']}</td><td>{row['lng']}</td>"
            f"<td>{row['division']}</td><td>{row['range']}</td>"
            # An entity and a line break inside a cell.
            f"<td>{row['beat']}&nbsp;<br>Sub</td>"
            f"<td>{row['total']}</td><td>{row['male']}</td>"
            f"<td>{row['crop']}</td><td>{row['death']}</td>"
            "<td><a href='/admin/sightings/" f"{row['id']}" "'>View</a></td>"
            "</tr>"
        )
    nav = "".join(
        f"<a href='?page={n}&start_date=2025-10-01'>{n}</a>" for n in (advertise or ())
    )
    return f"""<!doctype html><html><head><title>Sightings</title>
<script>var t = "<table><tr><td>not a table</td></tr></table>";</script></head><body>
<table class="summary"><tr><th>Filter</th><th>Value</th></tr>
  <tr><td>Division</td><td>All</td></tr></table>
<table class="listing">
  <thead><tr>
    <th>S.No</th><th>Sighting Date</th><th>Latitude</th><th>Longitude</th>
    <th>Division Name</th><th>Range Name</th><th>Beat Name</th>
    <th>Total Elephants</th><th>Tuskers</th><th>Crop Damage</th>
    <th>Human Death</th><th>Action</th>
  </tr></thead>
  <tbody>{''.join(body)}</tbody>
</table>
<nav>{nav}</nav>
<p>Showing page {page}</p></body></html>"""


# The shapes a login page takes when a script, not the server, builds
# it. Each one renders a password box a person can see and type into,
# and none of them can be posted by anything but a browser.
NAMELESS_LOGIN_PAGE = """<!doctype html><html><head><title>Sign in</title></head>
<body><form method="post" action="/login">
  <input type="email" id="email">
  <input type="password" id="password">
  <button>Login</button>
</form></body></html>"""

LOOSE_LOGIN_PAGE = """<!doctype html><html><head><title>Sign in</title></head>
<body><div class="card">
  <input type="email" name="email">
  <input type="password" name="password">
  <button onclick="submitLogin()">Login</button>
</div></body></html>"""

APP_SHELL_PAGE = """<!doctype html><html><head><title>Forest Alerts</title>
<script src="/build/app.js"></script></head>
<body><div id="app"></div><script>window.boot();</script></body></html>"""


# What the real site turned out to be: a shell around a mount point,
# with the rows arriving over an API after the page boots.
SPA_SHELL = """<!doctype html><html><head><title>Forest Alerts</title>
<script src="/build/app.js"></script></head>
<body><div id="app"></div></body></html>"""

# A bundle, with its routes surviving minification as string literals and
# plenty of near-misses around them.
APP_BUNDLE = """
var e={LOGIN:"/api/login",ME:"/api/me"};
function t(){return r.get("/api/sightings",{params:n})}
function o(){return r.get("/api/divisions")}
var i="/api/ranges",s="/api/beats/list";
var css="/build/app.css",img="/img/logo.png",help="https://example.org/help";
"""

API_TOKEN = "tok-bearer-99"


def _api_row(row):
    """A sighting as an API returns one: nested lookups, mixed types."""
    return {
        "id": row["id"],
        # ISO, the way an API sends a timestamp -- the HTML listing
        # above sends a 12-hour clock in the date cell instead, so both
        # shapes are covered.
        "sighting_date": f"2025-10-{(row['id'] - 99):02d}T19:0{row['id'] % 10}:00",
        "latitude": row["lat"],
        "longitude": row["lng"],
        "division": {"id": 1, "name": row["division"]},
        "range": {"id": 2, "name": row["range"]},
        "beat": {"id": 3, "name": row["beat"]},
        "total_elephants": row["total"],
        "tuskers": row["male"],
        "crop_damage": bool(row["crop"]),
        "human_death": row["death"],
        "photos": ["a.jpg", "b.jpg"],
        "reporter": {"id": 7},
    }


class Handler(BaseHTTPRequestHandler):
    """A small, deliberately awkward stand-in for the admin site."""

    requests = []          # every listing query the scraper sent
    expire_on_page = None  # serve the login page once for this page number
    expired = set()
    advertise = (1, 2, 3)  # what the paginator links to
    login_path = "/login"  # where the login form is served, if anywhere
    login_html = None      # override the login page body
    api_requests = []      # every API query the scraper sent
    api_needs_token = True # whether /api/sightings checks Authorization

    def log_message(self, *args):  # keep pytest output clean
        pass

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send(self, body, status=200, headers=(), content_type="text/html; charset=utf-8"):
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _authenticated(self):
        return SESSION_COOKIE in (self.headers.get("Cookie") or "")

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if parsed.path == "/build/app.js":
            return self._send(APP_BUNDLE, content_type="application/javascript")
        if parsed.path == "/spa":
            return self._send(SPA_SHELL)
        if parsed.path == "/api/sightings":
            return self._api_sightings(query)
        if parsed.path == self.login_path:
            return self._send(self.login_html or LOGIN_PAGE)
        if parsed.path in ("/login", "/admin/login", "/auth/login", "/"):
            return self._send("<h1>Not found</h1>", status=404)
        if parsed.path == "/admin/no-table":
            # A 200 with nothing to parse, which is what a listing that
            # renders its rows in JavaScript looks like from here.
            return self._send("<html><body><p>Loading...</p></body></html>")
        if parsed.path != "/admin/sightings":
            return self._send("<h1>Not found</h1>", status=404)
        if not self._authenticated():
            return self._send(LOGIN_PAGE)

        page = int(query.get("page", ["1"])[0])
        if "page" in query:  # the redirect after login is not a listing fetch
            type(self).requests.append(query)
        if self.expire_on_page == page and page not in self.expired:
            # A session that lapses mid-walk: a 200 carrying the login form.
            type(self).expired.add(page)
            return self._send(LOGIN_PAGE)

        start = (page - 1) * PAGE_SIZE
        rows = ROWS[start:start + PAGE_SIZE]
        if not rows:
            # Out of range: this framework clamps back to the last page
            # rather than returning an empty table.
            rows = ROWS[-PAGE_SIZE:]
        return self._send(_listing(page, rows, advertise=self.advertise))

    def _api_sightings(self, query):
        authorised = (
            not self.api_needs_token
            or f"Bearer {API_TOKEN}" == (self.headers.get("Authorization") or "")
            or SESSION_COOKIE in (self.headers.get("Cookie") or "")
        )
        if not authorised:
            return self._send_json({"message": "Unauthenticated."}, status=401)

        page = int(query.get("page", ["1"])[0])
        type(self).api_requests.append(query)
        start = (page - 1) * PAGE_SIZE
        rows = ROWS[start:start + PAGE_SIZE]
        last = -(-TOTAL_ROWS // PAGE_SIZE)
        return self._send_json({
            "data": {
                "current_page": page,
                "last_page": last,
                "total": TOTAL_ROWS,
                "next_page_url": f"/api/sightings?page={page + 1}" if page < last else None,
                "data": [_api_row(row) for row in rows],
            }
        })

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        if urlparse(self.path).path == "/api/login":
            sent = json.loads(raw)
            if sent.get("email") != EMAIL or sent.get("password") != PASSWORD:
                return self._send_json({"message": "Invalid credentials."}, status=422)
            return self._send_json({"data": {"token": API_TOKEN}})
        form = parse_qs(raw, keep_blank_values=True)
        if form.get("_token", [""])[0] != CSRF:
            return self._send("<h1>419 Page Expired</h1>", status=419)
        if form.get("email", [""])[0] != EMAIL or form.get("password", [""])[0] != PASSWORD:
            return self._send(LOGIN_PAGE, status=200)
        self._send(
            "<html><body>Redirecting</body></html>",
            status=302,
            headers=[("Set-Cookie", f"{SESSION_COOKIE}; Path=/"),
                     ("Location", "/admin/sightings")],
        )
