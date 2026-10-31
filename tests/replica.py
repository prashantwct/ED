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
from html import escape
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

EMAIL = "ranger@example.org"
PASSWORD = "correct horse"
CSRF = "tok-abc123"
SESSION_COOKIE = "fa_session=live"
LOGIN_PATH_INERTIA = "/login"
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

# The listing's Export button. Its headers are the site's own spellings,
# not the dashboard's, and it arrives with a filename rather than a
# useful content type -- both of which is how real exports turn up.
EXPORT_HEADER = (
    "S.No,Sighting Date,Sighting Time,Latitude,Longitude,Division Name,"
    "Range Name,Beat Name,Total Elephants,Tuskers,Crop Damage,Human Death"
)


def _export_row(row):
    return ",".join(str(part) for part in (
        row["id"],
        f"{(row['id'] - 99):02d}-10-2025",
        f"07:0{row['id'] % 10} PM",
        row["lat"], row["lng"],
        row["division"], row["range"], row["beat"],
        row["total"], row["male"], row["crop"], row["death"],
    ))


def _export_day(row):
    return f"2025-10-{row['id'] - 99:02d}"


def export_csv_text(start_date="", end_date=""):
    """The export over a window, filtered the way the site filters it."""
    rows = [row for row in ROWS
            if (not start_date or _export_day(row) >= start_date)
            and (not end_date or _export_day(row) <= end_date)]
    return "\n".join([EXPORT_HEADER, *(_export_row(row) for row in rows)]) + "\n"


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


# --- the Inertia surface -------------------------------------------------
# The live site is a Laravel + Inertia app: the HTML is a shell carrying a
# data-page payload, and the rows arrive as JSON on a second request. The
# form-and-table routes above stay for the scraper tools that still read
# them, and the two are told apart by the headers the client sends rather
# than by the path, because the real site serves both off /login and
# /admin/sightings exactly as this does.
INERTIA_VERSION = "inertia-v1"
XSRF_TOKEN = "xsrf-tok-123"
INERTIA_SESSION = "fa_inertia=live"


def _data_page(component, props=None, version=INERTIA_VERSION, head=""):
    """The shell Inertia boots from, payload and all."""
    payload = {"component": component, "props": props or {}, "url": "/",
               "version": version}
    attr = escape(json.dumps(payload), quote=True)
    return (f'<!doctype html><html><head><title>Forest Alerts</title>{head}</head>'
            f'<body><div id="app" data-page="{attr}"></div></body></html>')


# A Vite build as Laravel's resolvePageComponent leaves it: the main
# bundle maps each page to a lazy chunk, and the sightings page's chunk
# is the one whose Export button names the route.
VITE_HEAD = '<script type="module" src="/build/assets/app-7f3c.js"></script>'
VITE_BUNDLE = (
    'const Ze=(f,d)=>f();const pages={'
    '"./Pages/Admin/Users/Index.vue":()=>Ze(()=>import("./Index-u1.js"),[]),'
    '"./Pages/Admin/Sightings/Index.vue":()=>Ze(()=>import("./Index-s9.js"),[])};'
)


def sightings_chunk(route):
    return ('export default{methods:{exportCsv(){window.location.href='
            f'`{route}?start_date=${{this.f.start}}&end_date=${{this.f.end}}`}}}}}};'
            'const other="/admin/users/export";')


def _inertia_row(row):
    """A sighting in the shape the Inertia listing sends it.

    Date and time arrive in separate fields and the lookups are nested,
    which is what ``tools.fetch_sightings.flatten`` has to undo.
    """
    day = row["id"] - 99
    return {
        "id": row["id"],
        "date": f"{day:02d}-10-2025",
        "time": f"19:{row['id'] % 60:02d}:00",
        "geolocation": {"lat": row["lat"], "lng": row["lng"]},
        "division": {"id": 1, "name": row["division"]},
        "range": {"id": 2, "name": row["range"]},
        "beat": {"id": 3, "name": row["beat"]},
        "sighting_type": ["directSighting"],
        "total_count": row["total"],
        "male_count": row["male"],
        "crop_damage": row["crop"],
        "death": row["death"],
        "created_at": "2025-10-01T00:00:00Z",
    }


def _inertia_page(page, size=PAGE_SIZE):
    start = (page - 1) * size
    last = max(1, -(-len(ROWS) // size))
    return {
        "component": "Admin/Sightings/Index",
        "version": INERTIA_VERSION,
        "props": {"sightings": {
            "data": [_inertia_row(r) for r in ROWS[start:start + size]],
            "meta": {"current_page": page, "last_page": last,
                     "per_page": size, "total": len(ROWS)},
        }},
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
    export_requests = []   # every export query the scraper sent
    export_as = "csv"      # what the Export button hands back
    export_route = "/admin/sightings/export"  # where the Export button points
    ziggy_routes = None    # the route table the listing page ships, if any
    vite = False           # whether the listing page loads a Vite build
    per_page = "ignore"    # what the listing does with ?per_page: ignore|honour|reject
    inertia_partials = []  # the partial-reload props each listing page asked for
    inertia_pages = []     # every Inertia listing page requested
    inertia_queries = []   # the query each of those pages was asked with
    inertia_stale_once = False  # answer the first listing page with a 409

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

    def _inertia(self):
        """Whether this request is the JSON half of an Inertia visit."""
        return (self.headers.get("X-Inertia") or "").lower() == "true"

    def _boots_inertia(self):
        """Whether it is the page load the client reads the version off."""
        return (self.headers.get("X-Requested-With") == "XMLHttpRequest"
                and not self._inertia())

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if parsed.path == LOGIN_PATH_INERTIA and self._boots_inertia():
            return self._send(_data_page("Auth/Login"), headers=[
                ("Set-Cookie", f"XSRF-TOKEN={XSRF_TOKEN}; Path=/")])
        if parsed.path == "/admin/sightings" and (
                self._inertia() or self._boots_inertia()):
            return self._inertia_sightings(query)
        if parsed.path == "/build/assets/app-7f3c.js" and self.vite:
            return self._send(VITE_BUNDLE, content_type="application/javascript")
        if parsed.path == "/build/assets/Index-s9.js" and self.vite:
            return self._send(sightings_chunk(self.export_route),
                              content_type="application/javascript")
        if parsed.path == "/build/app.js":
            return self._send(APP_BUNDLE, content_type="application/javascript")
        if parsed.path == "/spa":
            return self._send(SPA_SHELL)
        if parsed.path == "/api/sightings":
            return self._api_sightings(query)
        if parsed.path == self.export_route:
            return self._export(query)
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

    def _export(self, query):
        if not (self._authenticated()
                or INERTIA_SESSION in (self.headers.get("Cookie") or "")
                or f"Bearer {API_TOKEN}" == (self.headers.get("Authorization") or "")):
            return self._send_json({"message": "Unauthenticated."}, status=401)
        type(self).export_requests.append(query)

        if self.export_as == "xlsx":
            body = _xlsx_bytes()
            name = "sightings.xlsx"
            # Served with no useful type, which is common enough that
            # guessing from the header alone would fail here.
            content_type = "application/octet-stream"
        elif self.export_as == "html":
            body = b"<html><body><h1>Session expired</h1></body></html>"
            name, content_type = "login.html", "text/html"
        else:
            body = export_csv_text(query.get("start_date", [""])[0],
                                   query.get("end_date", [""])[0]).encode("utf-8")
            name, content_type = "sightings.csv", "application/octet-stream"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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

    def _inertia_login(self, raw):
        """Laravel's own login: 422 with errors, or a rotated session."""
        if self.headers.get("X-XSRF-TOKEN") != XSRF_TOKEN:
            return self._send_json({"message": "CSRF token mismatch."}, status=419)
        sent = json.loads(raw or "{}")
        if sent.get("email") != EMAIL or sent.get("password") != PASSWORD:
            return self._send_json(
                {"errors": {"email": "These credentials do not match our records."}},
                status=422)
        body = json.dumps({"component": "Admin/Sightings/Index", "props": {}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Set-Cookie", f"{INERTIA_SESSION}; Path=/")
        self.end_headers()
        self.wfile.write(body)

    def _inertia_sightings(self, query):
        cookies = self.headers.get("Cookie") or ""
        if INERTIA_SESSION not in cookies:
            # Anonymous: the real site serves the login shell, which is how
            # the client notices a sign-in that did not take.
            return self._send(_data_page("Auth/Login"))
        page = int(query.get("page", ["1"])[0])
        if self._boots_inertia():
            props = {"ziggy": {"routes": self.ziggy_routes}} if self.ziggy_routes else None
            return self._send(_data_page("Admin/Sightings/Index", props,
                                         head=VITE_HEAD if self.vite else ""))
        type(self).inertia_pages.append(page)
        type(self).inertia_queries.append(query)
        type(self).inertia_partials.append(self.headers.get("X-Inertia-Partial-Data"))
        if self.inertia_stale_once and len(self.inertia_pages) == 1:
            return self._send_json({"message": "version mismatch"}, status=409)
        if "per_page" in query and self.per_page == "reject":
            return self._send_json({"message": "The per page field is invalid."},
                                   status=422)
        if "per_page" in query and self.per_page == "honour":
            return self._send_json(_inertia_page(page, int(query["per_page"][0])))
        return self._send_json(_inertia_page(page))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        if urlparse(self.path).path == LOGIN_PATH_INERTIA and self._inertia():
            return self._inertia_login(raw)
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


def _xlsx_bytes():
    """The same rows as a real .xlsx, so the spreadsheet path is exercised
    rather than described."""
    import io

    import pandas as pd

    frame = pd.read_csv(io.StringIO(export_csv_text()), dtype=str)
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()
