# Elephant Conflict Intelligence

Streamlit app that turns Gajrakshak elephant sighting/conflict exports into
decisions a protected-area manager can act on: which beats to resource, which
shift to staff, which villages to warn, and what is getting worse.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then either upload a sightings CSV, or sign in to Forest Alerts on the landing
page and let the app pull one. Village centroids for the Shahdol–Anuppur
landscape ship with the repo and load automatically.

Tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Input data

**Sightings CSV (uploaded).** Required columns: `Date, Latitude, Longitude,
Division, Range, Beat`. Optional columns unlock features rather than being
mandatory: `Time` or `Hour`, `Total Count`, `Crop Damage`, `Grain Damage`,
`House Damage`, `Injury`, `Death`, `Male/Female/Children Death Count`, `ID`.

Dates are parsed by trying each candidate format against the whole column,
day-first first. Genuinely ambiguous files (every day ≤ 12) are read day-first
and flagged. Non-UTF-8 files fall back to cp1252 then Latin-1.

**Damage and compensation surveys (a separate view).** The Epicollect5
[herd-crop-damage](https://five.epicollect.net/project/herd-crop-damage) and
[herd-house-damage](https://five.epicollect.net/project/herd-house-damage)
projects are household surveys, not sightings: what each household lost, its
estimated rupee value, where its compensation claim stands, and how long the
survey came after the damage. They get their own view (sidebar -> View, or the
button on the landing page) rather than being folded into the sighting
statistics, and it answers different questions:

- *Compensation* -- cases and loss by claim status, and a follow-up queue of
  households that have not applied, largest loss first.
- *Gaj Rakshak gap* -- surveys whose remarks say the incident is missing from
  the register, and, when the register is loaded in the conflict view, every
  survey checked against it by place and date (distance and days set in the
  sidebar).
- *Villages* -- loss, unclaimed cases and repeat households per village, with
  the beat each point falls in or the nearest within 10 km (damaged fields
  mostly sit on farmland between beat blocks), and a map sized by loss and
  coloured by claim status.
- *Crop and house detail* -- crops and crop stage, fencing at the damaged field,
  the owner's claimed area against the surveyor's and the calculated one, house
  type, rooms broken, light on or off, herd size and hour, and survey delay.

The whole study downloads as an HTML file and an A4 PDF ("Prepare HTML and
PDF" under the assessment). Both are vector throughout: every chart and the map
is drawn as SVG -- inline in the HTML, and converted to PDF drawing operators by
svglib and ReportLab rather than pasted in as a picture -- so they stay sharp at
any zoom and print size. The map has no raster basemap for the same reason;
forest boundary outlines, clipped to the frame, give the place. Long tables show
their top 20 in the body and in full in an appendix.

Upload the CSV or ZIP export from each project's Data page, or fetch directly.
Owner names, phone numbers, photos, surveyor emails and remark text are dropped
as the file is read: a repeat household is recognised by a one-way hash of name
and number, remarks are reduced to what they say about Gaj Rakshak, and
downloads carry only the Epicollect5 entry ID, which opens the full record for
someone with access to the project. `core/damage.py` finds fields by question
wording rather than number, so a re-ordered form still reads.

Fetching syncs over the export API within its limits: requests of one kind
24 s apart (half the documented 5 a minute), 500-entry pages, and after the
first pull only entries uploaded since. The projects are private, so each needs
its own client app's credentials -- `EPICOLLECT_HERD_CROP_DAMAGE_CLIENT_ID` /
`..._CLIENT_SECRET` and `EPICOLLECT_HERD_HOUSE_DAMAGE_CLIENT_ID` /
`..._CLIENT_SECRET` (a shared `EPICOLLECT_CLIENT_ID` / `EPICOLLECT_CLIENT_SECRET`
is the fallback). The project's creator or a manager makes the app under the
project's Details -> Apps. Tokens are reused until near expiry, as the API
issues ten an hour, and a project that cannot be read is reported while the
other still loads.

**Fetching the export instead of uploading it.** Where no export button
exists, the same rows can be pulled off the Forest Alerts admin listing — from
the landing page, or from the command line:

```bash
export FORESTALERTS_EMAIL=... FORESTALERTS_PASSWORD=...
python -m tools.scrape_sightings -o data/sightings_scraped.csv
```

On the landing page it is a sign-in below the uploader. Both paths run the same
scraper and end in the same `core.data_loader`, so there is no second parser and
no way for the two to drift.

**The app signs in as the person using it.** Their own Forest Alerts account
decides what they can pull, which is why there is deliberately no way to
configure a shared login in `secrets.toml`: this app has no access control of
its own, and a stored credential would turn it into a public mirror of the
register. Credentials are a request parameter and nothing else — used for one
sign-in, then gone. The form clears on submit, the fetch is not cached (a cache
keyed on a password is a password at rest), and only the resulting CSV bytes
reach session state, where they go through the same cached load as an upload.
`tests/test_app_fetch.py` asserts the password is absent from session state
after a real round trip rather than merely unrendered.

The form asks for an email and a password, and nothing else. The site is a
Laravel + Inertia app, so `tools/fetch_sightings.py` signs in the way the
login page itself does — `POST /login` with the XSRF cookie and the Inertia
headers — and reads the listing as Inertia JSON, page by page. There is no
endpoint to discover on the Network tab and no browser cookie to paste, so
the fields that used to collect them are gone: a form that takes what nothing
reads teaches a reader the fetch needs setting up, and it does not.

The window is not a parameter of the job. The listing is not filtered by
date, so every run is a full refresh rather than an increment to be merged;
a full pull cannot develop a gap, and the register is small enough that this
is cheaper than being clever. Nothing is promised in advance about the span:
the dates on the result are read off the rows that arrived, which is why the
loaded page can say what the pull actually covers rather than what was asked
for.

A connection that is refused or times out says so in a sentence naming the
site, with the CSV upload as the way round it. Anything else keeps its type
and message, because guessing "check the network" at an unrelated bug sends
the reader the wrong way.

Column headers are read rather than assumed — each is normalised and looked up
in a synonym table to find the dashboard's name for it, so `Sighting Date`,
`Tuskers` and `No. of Elephants` land in `Date`, `Male Count` and `Total Count`.
A header nobody anticipated is carried through under its own name rather than
dropped, because the file is meant to be a raw capture that also happens to
load. Pagination stops on evidence: asking for a page past the end returns
either nothing or a repeat depending on the framework, so the walk stops when a
page yields no row it has not already seen, and rows are fingerprinted and
de-duplicated across pages.

The run finishes by loading its own output back through `core.data_loader`,
which is the only check worth making — the scrape is correct when the dashboard
accepts it, not when it parsed. `--no-verify` skips it.

**Forest Alerts renders in the browser.** `--probe` settled it: `/` and
`/login` both return a four-script shell around `<div id="app">` with no form,
no table and zero characters of visible text, and `/admin/login` and
`/auth/login` are not served at all. Nothing renders server-side, so there is
no HTML to scrape — **the rows come from a JSON API the page calls after it
boots, and that API is the only thing there is to read.**

**The listing's Export button is the best source, and the one to try first.**
Whatever it returns is the register as the department itself publishes it: one
request, no pagination to walk, no row shape to infer, and the columns the
people who run the system chose.

```bash
python -m tools.scrape_sightings \
  --export-path /admin/sightings/export \
  --cookie 'laravel_session=...' \
  -o data/sightings_scraped.csv
```

What comes back is not known in advance, so the format is detected rather than
assumed — from the filename the server suggests, then the content type, then
the first bytes, because a download served as `application/octet-stream` is
common enough that trusting the header alone would fail on it. CSV, TSV, JSON,
`.xlsx` and `.xls` are all read. A spreadsheet goes through pandas, which reads
cell *types* rather than printed strings: an Excel date is a serial number, and
reading it by hand is how every date in a file ends up wrong. An HTML page
where a file was expected is reported as a failed download rather than an
export of no rows — that is what a lapsed session looks like.

Failing an export, `--api-path` points the scraper at the JSON endpoint the
page calls to fill its table; failing that, it parses the HTML listing as
before, which still serves any deployment that renders server-side.

```bash
python -m tools.scrape_sightings \
  --api-path /api/sightings \
  --header 'Authorization: Bearer ...' \
  -o data/sightings_scraped.csv
```

All three sources converge on the same column mapping, the same CSV and the
same `core.data_loader`.

The payload's wrapper is read rather than assumed — a bare list, `data`,
`results`, `rows`, `items`, or Laravel's doubly-nested `{"data": {"data":
[...]}}` — and a named wrapper wins over some other list in the same payload.
Nested lookups give up their label, so `{"division": {"id": 3, "name":
"Shahdol"}}` becomes a `Division` of "Shahdol" with `division.id` kept beside
it. Booleans become 1/0, because `crop_damage: true` has to be countable.
Where the API reports `next_page_url: null` the walk stops on its word;
otherwise it falls back to the same no-new-rows rule as the HTML path.

**Finding the endpoints.** Neither can be discovered from outside a browser —
open the listing with developer tools on the Network tab, filter to Fetch/XHR,
**click Export**, and read the request it makes. `--probe` narrows the search
by fetching the page's own script bundles and listing the API routes quoted
inside them, export routes first and then the ones naming sightings. Those are
leads, not answers: a route in a bundle may be one the app never calls.

**Signing in.** There is no form to post, so `--cookie` (a session cookie from
a signed-in browser) or `--header 'Authorization: Bearer ...'` is the way in.
Where the app's own sign-in endpoint is known, `--api-login-path` posts
`{email, password}` as JSON to it and uses whatever comes back — a bearer
token if it returns one, the session cookie if it sets one.

**The login form, where a site has one.** `/login`, `/admin/login`,
`/auth/login` and `/` are tried in turn, a password box is recognised whether
or not it has a `name` attribute and whether or not it sits inside a `<form>`,
and the failure reports what each candidate page actually held rather than
which one it wanted.

`--probe` prints that report on demand and needs no credentials:

```bash
python -m tools.scrape_sightings --probe
```

For each candidate it gives the status, title, forms, named and nameless
inputs, script and mount-point counts, visible text length, table rows, and a
verdict — a submittable form, a password box nothing can post, a JavaScript
application shell, or simply not a login page. `--dump-dir` keeps the HTML
alongside it.

A password box with no `name`, or one outside any `<form>`, is a form wired up
in JavaScript: the server never receives a nameless field, so no script can
sign in. That is what `--cookie` is for.

On the command line, prefer the environment variables over `--password`, which
is visible in `ps` and shell history.

**Early-warning registry (optional upload).** The villager registry export:
`Latitude, Longitude` required, `Village` and `Division` used when present.
Unlocks the coverage section below.

**Registered app users (optional upload).** The user-list export; `Division`
required. `Beat`, `Sub Beat` and `Range` are ignored — in the export this was
built against they were placeholders for 1,356 of 1,357 users, all of them, and
1,327 respectively, so the roster can only be read at division level.

Both files carry personal data: names and mobile numbers, and email addresses
in the user list. `core/coverage.py` keeps an **allowlist** of columns and
discards everything else while reading, before the frame reaches anything else,
so a new identifying column in a future export is excluded by default rather
than by having been anticipated. Coordinates are rounded to three decimals,
about 110 m — enough to place someone in a village, not at a house. Only
per-village counts reach the page, the brief or any download. Neither file is
stored, and both file shapes are in `.gitignore`.

**Forest Alerts endpoints (optional).** `FORESTALERTS_EXPORT_PATH`,
`FORESTALERTS_API_PATH` and `FORESTALERTS_API_LOGIN_PATH`, in the environment
or `.streamlit/secrets.toml`, pre-fill the landing page's endpoint fields. They are configuration rather than
secrets; the credentials are never stored.

**MapTiler key (optional).** Set `MAPTILER_KEY` in `.streamlit/secrets.toml`
locally, or under Manage app → Settings → Secrets on Streamlit Cloud. See
`.streamlit/secrets.toml.example`. Without it both the on-screen maps and the
brief's embedded maps fall back to the keyless Carto basemap, so the choice of
terrain, satellite or topographic styling is what the key buys — not whether
there is a basemap at all.

The key is fetched by the browser to load tiles, so it is visible to anyone
using the app and cannot be kept secret. Restrict it by origin in the MapTiler
dashboard (Keys → Allowed origins) rather than relying on secrecy.

**Forest boundaries (bundled).** `data/boundaries/*.geojson.gz` — division,
range, beat and reserve outlines for the Shahdol and Rewa circles, converted
from the MP forest department shapefiles by `tools/build_boundaries.py` and
committed so the app needs no GIS stack at runtime. 612 KB for 1,377 polygons.

The source is a custom Transverse Mercator, reprojected to WGS84. Geometry is
simplified per level and coordinates rounded to about a metre, because a beat
outline at landscape zoom cannot show detail finer than its own pixel.

Every sighting falls inside a division and a range polygon. **Beats cover
forest land only**, so a sighting on farmland between two blocks — a third of
them — sits in no beat at all. The shapefiles also suffix reserve ranges with
their zone, so the export's "Kallwah" is "Kallwah Core" here; joins normalise
that away.

**Village centroids (bundled).** `data/centroids.csv` — 9,229 villages,
`Village, Latitude, Longitude`, UTF-8. Constant reference data, loaded by
default from a repo-relative path so it resolves regardless of working
directory. The sidebar uploader overrides it for a different landscape.

Sighting exports carry no village field, so villages can only be named against
this file. The free-text description is not mined for them: it is inconsistent
(in a 1,761-row export, 5 rows used "निवासी" and 29 used "ग्राम") and carries
victim names.

## Landing page

The pre-upload screen carries the project identity: a MOVE-MP masthead over a
field photograph, the partner marks, what the tool does, and what to upload.
Below the uploader is the Forest Alerts sign-in, which fetches the register
instead. Once an export is loaded, either way, the screen collapses to a
one-line bar so it belongs to the data, with a line saying where that data came
from and a button to discard a fetch.

Assets live in `assets/`. `hero-elephant.jpg` and `logo-wct.png` are taken from
the project's own progress report. **`logo-mpfd.png` is not supplied** — an
official state emblem is not something to approximate, so the Forest Department
currently renders as a wordmark. Drop the official file in under that name and
it is picked up automatically; the same applies to any partner added to
`PARTNERS` in `core/landing.py`.

## What it produces

**How the page reads.** The headline numbers and the assessment stay at the
top, with the Generate brief button beside them. Everything else sits in five
tabs, one per question a manager asks, so no question pushes another off the
screen:

* *Is it getting worse?* -- monthly conflict by type, conflict rate,
  casualties, the seasonal grid, divisions over time, escalating beats and
  the conflict rate by division.
* *Where to act* -- the beat tier counts, the layer filters, then a table
  (beats, hotspots, villages, animals) beside the spatial view. One tick on a
  row shows only that place on the map.
* *New or growing?* -- each place's emerging hotspot pattern (new,
  intensifying, persistent, consecutive, sporadic, diminishing, historical) as
  counts, a table, a hexagon map, and a place-by-time heatmap.
* *Villages at risk* -- the village ranking beside its map, and early-warning
  coverage when a registry is loaded.
* *When* -- the casualty peak window, how many reports count as night raids,
  the seasonal peak, the hour-of-day and night/day charts.
* *Data* -- monthly sightings, severity, division and escalation tables, the
  filtered rows and their download.

Each chart's download sits as an icon in its title row rather than a button
under it, and wide tables show the columns a decision needs, with every
column in their CSV.

**Conflict trends.** First on the trends tab, before any ranking: monthly
conflict stacked by type (fatality darkest, at the base, so a growing dark band
reads before a growing total) with a three-month average and the escalation
window shaded; the monthly conflict rate against the period average, so more
patrolling is not mistaken for more conflict; people killed and injured per
month; a year-by-month heatmap for seasonality; and each division's monthly
conflict, labelled at the line end. The headline numbers carry monthly
sparklines and their change over the escalation window, where a rise reads red.
Escalating beats are drawn as prior-to-recent dumbbells rather than a "7 vs 2"
column, and division conflict rates as ranked bars against the landscape rate.

**Beat priorities.** Every beat gets a decision tier — Critical, High, Watch,
Routine — with the evidence behind it: reports, conflict events, adjusted
conflict rate, casualties recent and total, night share, village proximity,
which animals are causing it, trend, and a recommended action.

**Timing.** Crop, grain and house damage is found and written down the
morning after the raid, so a damage report's hour is when it was recorded.
Every damage report counts as night conflict, in the night shares and in
each beat's Night Conflict %, whatever hour it was entered at. The
hour-of-day chart and the "when to staff" peak window are built from
injuries and deaths only, which are reported as they happen; the timing
panel and brief state how many damage events were counted as night raids.

**Which animals.** Damage rate split by the kind of group seen. Only bulls
carry tusks in this species, so the male count is the tusker count. In the
export this was built against, a lone bull carried a 28% damage rate and a
small all-male party 36%, against 3% for a breeding herd with calves. Each
beat carries the share of its conflict driven by bull-type groups, because the
two need opposite handling — a bull is identified and deterred, a herd is given
passage and not driven.

**Movement hotspots.** Density clusters in the point data rather than the
administrative grid, so a herd working a corridor along a beat boundary reads
as one hotspot instead of moderate pressure in two beats.

**Emerging hotspots.** A space-time cube: reports binned into hexagons
(2 km apart by default) by month, fortnight or week. Getis-Ord Gi* scores
every cell and step against the whole cube over the cell, its six neighbours,
this step and the one before, with a false discovery rate correction across
all bins. A Mann-Kendall test on each cell's Gi* series gives its trend, and
the run of hot steps plus that trend gives the label. Hot side only, since a
cold spot in counts this sparse is just a quiet place. Two guards keep it
honest: a cell counts as hot only if it had a report itself, so Gi* does not
paint empty neighbours; and a trailing step less than half covered by the data
is dropped, not read as everything cooling at once. Fewer than six steps gives
no labels. Settings sit under *Space-time trend* in the sidebar; the brief
gets a table and the assessment a one-line headline.

**Villages at risk.** Every conflict incident within a chosen radius of each
village, so a settlement between two hotspots is credited with both.

**Early-warning coverage.** The risk ranking read against the registry: which
exposed villages have nobody enrolled to warn. Registrants are placed against
village centroids spatially, not by name — the two files transliterate
differently, and on the reference data only 41% of nearest-centroid names
agreed with the registry's own village field while 95% of registrants fell
within 2 km of some centroid. Where a same-named centroid does lie inside that
radius it wins over a marginally closer stranger, which moved 898 of 7,261
people. Placement is exclusive, so the counts partition and can be audited
against the file's row count.

The output is an enrolment queue: Critical and High villages with no contact or
fewer than three, worst tier and thinnest cover first. On the reference export
that was seven villages, five of them Critical with nobody enrolled at all.
Below three is "Thin" rather than covered, because one number is a handset that
is off overnight.

**Timing.** The shortest contiguous block of hours holding a target share of
conflict, reported with how concentrated it actually is. Seasonality likewise,
and silent when there is none.

**Forest boundaries.** Hollow outlines — no fill — in amber, chosen to hold up
against both pale terrain and satellite; reserve outlines get a pale casing
under the line, which the administrative levels cannot afford because it
doubles their geometry payload.

`line_width_units` is passed as a *quoted* string. pydeck turns a bare string
prop into a deck.gl accessor, so `"pixels"` was serialised as `"@@=pixels"` — a
per-feature function where a unit belongs. The widths that produced drew the
outlines as rounded blobs, or made them vanish. There is a test asserting no
layer prop leaves as an accessor unless it is one.

Which level shows follows the zoom — divisions across a landscape, beats once
the view is tight enough to read them. Streamlit's map reports clicks but not
viewport changes, so that follows the view the filters produce rather than live
pinch-zoom; the sidebar can pin a level instead.

**Two maps.** The spatial view's layers are chosen in the *Where to act*
tab, not the sidebar. *Events* (what happened) and *Animals* (lone bull,
bull party, family herd, ...) combine, so "crop or grain loss" with "lone
bull" maps exactly those sightings. *Villages by tier* switches each tier on
or off (Routine starts off), and hotspot footprints have their own toggle.
Each conflict class has its own symbol as well as its colour, the same
on the map, in its legend and in the downloaded map: a diamond with a cross
for a human fatality, a medical cross for an injury, a house for house or
store damage, a leaf for crop or grain loss, and a small dot for presence
only. Casualties draw largest and on top. Ticking a row in the beat,
hotspot or village table shows only that place: the map zooms to it,
outlines the beat or rings the hotspot or the village's exposure radius,
and draws only the sightings inside it. Filters cut before drawing, so
hotspots stay only where a shown sighting falls in them and villages only
within the exposure radius of one; the download is the same filtered map.
The table draws before the map, so one tick is enough. The village-risk map leads with the settlements: circle size
is conflict count, colour is tier, and the worst are labelled. Basemap is
selectable (terrain, satellite, hybrid, topographic, streets, minimal).

**A brief.** One structured object renders both the in-app panel and the
downloadable HTML, so the numbers on screen and in the forwarded document
cannot drift apart. The brief names the divisions it covers in its header and
embeds both maps as inline SVG over a basemap stitched from map tiles at
generation time and inlined as a JPEG, so it stays a single self-contained file
that prints.
The trend charts, escalation dumbbells, hourly profile and division rates are
drawn into the brief as inline SVG in the dashboard's colours -- no chart
library, no script -- with the seasonal grid as a shaded table, so the brief
still opens offline and prints as it looks on screen.

**Every chart and map downloads.** A Download menu under each figure, in both
views, gives SVG (vector), PNG at four times its drawn size, and PDF (vector).
Each is exported from an SVG twin of the figure -- the brief's and the damage
study's chart builders -- rather than captured from the interactive canvas,
which the server cannot render. resvg draws the PNG with the DejaVu Sans fonts
in `assets/fonts`, so text renders the same on a host with no fonts installed;
svglib and ReportLab draw the PDF. Files are built only when a button is
pressed. The spatial view downloads as sightings and hotspots over the basemap.

**Day refuges.** A third view finds where herds lie up by day, from the
register loaded in the Conflict dashboard and Esri's 10 m Sentinel-2 land
cover (read online by year, or uploaded as the GeoTIFFs Esri's Land Cover
Explorer downloads). A place is a candidate refuge only if it passes every
test, and each decoy it rejects is planted in the tests:

* Inside forest: every sighting and sign that is not a conflict event --
  direct sightings and indirect signs (pug marks, dung, broken branches,
  sounds) alike, at any hour -- counts as refuge evidence when it falls in
  notified forest (the vendored forest beat boundaries), or, where they do
  not cover the register, in tree, rangeland or flooded-vegetation cover in
  Esri land cover. A sign is stale about the hour but not about the forest
  block: elephants used it. Signs and sightings in fields are feeding,
  raiding or passing, never refuge evidence.
* Evidence clustered (DBSCAN, 0.75 km) and seen on at least three separate
  days, so one day's passage is dropped; the block must reappear when the
  cluster radius changes (0.5-1 km).
* Cover: tree, flooded-vegetation and rangeland share within 250 m, and the
  size of the tree patch it sits in. Notified forest with no cover on the
  ground is capped at Possible and flagged for ground verification.
* Confirmed needs five days, two-thirds stability, at least one direct
  sighting and passing cover; without land cover nothing is Confirmed. A
  refuge known from signs alone is flagged to be confirmed by day.
* Raids: a block no raid links to is left out, and counted in the caveats.

Only a direct sighting puts elephants at a place at the hour entered.
Everything else -- pug marks, dung, broken branches, sounds, and damage
or casualties reported indirectly -- is a stale trace of an earlier visit:
it never counts as a day sighting, counts as night use of the place, and
is dated to the night before its entry. The dashboard follows the same
rule: stale reports are night events in every night share, and the
hour-of-day chart and "when to staff" window use directly sighted
injuries and deaths only.

Damage is written down the morning after the raid, so a crop, grain or
house damage entry is dated to the night before it (or to the same night
if entered after 18:00, and to the night before if it has no hour). Damage
entries are never counted as day sightings, so a 10 am damage report in a
field cannot become a "refuge"; they count as night use of the place.
Injuries and deaths keep their reported hour and are raids only at night.

A raid within 5 km is linked to a refuge with forest evidence the day its
night began or the day after (adjustable to a wider window). Each link says
how close in time it is, and a refuge in use the same night wins over a
nearer one that was not. Every run also re-links the raids with each
refuge's dates shifted by 20-60 days and reports that count as the chance
baseline: on the October 2026 register, 358 of 603 raids linked against
about 91 by chance, while a 7-day window linked more raids but mostly by
coincidence.
This gives the direction raids head, their median distance and the villages
hit.

Every raid is accounted for: linked to a refuge, in a raid area, or
scattered. Raids no refuge accounts for are grouped into raid areas (three
or more raids within 1.5 km). Each is a place elephants raid with no
reported day refuge within reach -- in a register where one division
reports almost only at night, most raids are of this kind. With land cover,
each area names the nearest tree patch of 20 ha or more within reach, with
its distance, direction and coordinates, as the place to search by day; the
KML carries these as search targets for field phones. Each refuge carries its beat, nearest village,
flags (small isolated patch, near settlement, breeding herd, lone bull) and
a recommended action: the dusk-watch edge, the villages to warn, do-not-drive
for herds with calves. It downloads as CSV, GeoJSON and KML for field phones,
with the linked raids, and a land-cover card per refuge shows what is around
it. Clusters of sightings say where herds were reported, not where every herd
is: a refuge nobody reports from will not appear.

The view opens on the **Dashboard** tab: the atlas plate itself, embedded and
filled with the current run. `assets/refuge_dashboard.html` is the plate, a
D3 sheet with a register panel, a village finder, a watch list and the
charts, vendored with its data lifted out and five placeholders in its place.
`core/refuge_dashboard.py` puts the analysis back in — one FeatureCollection
of refuges, raid areas and a line per linked raid, under `core/refuges.py`'s
own column names, so it is a pass-through rather than a translation. The
same bytes are offered as a download: the file carries its own register, so
it opens on a phone in a range office with no app behind it.

The plate is vendored rather than generated. It is a design, and the honest
way to keep a design is to keep it; rebuilding that sheet out of Python every
release would be a slow way of losing it.

Two things there are easy to get wrong. The day every recency band is
measured from comes off the register's own last day, not the clock, so a file
that ends in March does not open with every refuge two seasons stale. And the
register is an uploaded CSV, so a village or beat name is attacker-controllable
text that ends up inside a `<script>` in a frame with same-origin access to
the app — `json.dumps` does not escape `<`, so the embedded JSON escapes
`<`, `>`, `&` and the two line separators JSON allows and JavaScript does not.
They are ordinary JSON string escapes, so the value a reader sees is
unchanged; it simply cannot carry a tag.

The **Map and tables** tab keeps what the plate does not do: the pydeck map
over a real basemap with the forest boundary outlines under the marks,
sortable registers, the CSV/GeoJSON/KML downloads and the land-cover card per
refuge. It is laid out as a field atlas plate too: the map sheet on the left,
a register panel beside it. The panel leads with **what to act on tonight** --
the refuges used in the last seven days, newest first, each with the edge to
watch at dusk and the villages to warn before dark, and the lot copyable as a
message, because that is how it reaches a range office.

Below that the **map key is the controls, not a legend**. A key that only
explains wastes the space next to a map, so each block takes its marks off:
confidence, group, last used within 7/30/60 days, the raid period, and the
raid areas and refuge-to-raid lines. The map can be coloured by confidence
(how much the site is believed), group (who uses it) or last used (whether it
is live) -- three different questions, and the reader picks rather than the
map deciding. Group takes the Okabe-Ito identity colours; last used takes one
hue stepped dark-to-pale, because recency is ordinal and identity is not.
The legend lists only what is actually drawn: offering four confidence levels
on a map showing one teaches the reader something untrue.

Under the key sit **when raids start**, hour by hour with the night hours
darker and the peak called out, and **raids by month** stacked by what the
raid cost. Both follow the key, as do the register, the raid-area table and
every download -- a file matches the plate it came from rather than quietly
carrying rows the reader had switched off. The map layers are unchanged:
the same basemap, the same forest boundary outlines, the same pydeck layers
underneath.

The brief carries the same refuges: findings, the map of refuges and their
linked raids, and the register with recommended actions. It uses the land
cover and settings last chosen in the Day refuges view, or otherwise Esri's
latest year read online. Read online, a year Esri has not published for a
tile (a 404) falls back to the latest earlier year, and the view and brief
say which year was used. Set `ESRI_LULC_URL` to read from a mirror, using the
same `{year}`, `{next}` and `{tile}` fields.

## Design rules

**Composition informs the score, never the tier.** Tiers record harm that has
already happened; group composition predicts harm that has not. Letting a tier
move on composition alone would break the guarantee below. It changes the
ordering within a tier and the recommended action, and there is a test holding
that line.

**Tier decides, score only orders.** Tiers come from fixed rules, so "Critical"
means the same thing in April as in October and in one division as the next.
The continuous score orders beats *within* a tier and nothing more — a score
normalised against what is currently on screen moves when a filter changes, and
a posting decision must not hang on it.

**Critical means recent.** A beat reaches Critical only on a casualty in the
last 90 days of the period under review. An older fatality still puts it in
High. The window is anchored to the period's end, not each beat's own last
report — otherwise selecting a beat that stopped reporting months ago would
measure from its final entry and flip an old fatality back to Critical. It is a
fixed constant, not the adjustable escalation slider.

**Raw rates from thin data are not evidence.** One conflict in one report is a
100% rate. Beat rates are shrunk toward the landscape rate with an
empirical-Bayes estimator whose strength is fitted from how much beats actually
differ, and every beat carries a confidence label.

**Counts measure patrol effort as much as elephants.** More reports can mean
more staff walking the beat. Rates sit alongside volumes rather than replacing
them, and the brief says so every time.

**Watch what the map ships.** Streamlit serialises the whole deck spec into
the page on every rerun, so anything attached per-row is multiplied by the row
count. Tooltip styling lives in one stylesheet rather than inline on 1,761
features, the spec is minified because pydeck pretty-prints by default, and
boundary geometry is filtered to the viewport. Together that is the difference
between 5.8 MB and 1.6 MB, and between a map that loads and one that does not.

**Personal data is dropped at the door, not at the display.** The registry and
user-list loaders slice to an allowlist before returning, so no name, number or
email exists anywhere downstream — not in a cache, not in session state, not in
the deck spec. A rule enforced where the data enters cannot be forgotten by the
next feature that reads it, and there is a test asserting the values are absent
from the loaded frame rather than merely unrendered.

**Credentials are a request parameter, never state.** The Forest Alerts
sign-in is read off the form, used for one login, and kept nowhere: not in
session state, not in a cache, not in a log, not on disk. The fetch is
deliberately uncached for that reason — its CSV output is cached instead. The
same reasoning as the rule above, applied to a secret rather than to someone
else's name.

**Colour is never the only signal.** Tier badges carry a shape, a glyph and a
word, so rankings survive greyscale printing and colour vision deficiency. Map
categories use the Okabe-Ito colourblind-safe palette with size as a parallel
channel.

## Layout

```
app.py               Streamlit entry point: layout and widgets only
core/config.py       Every tunable parameter
core/csv_io.py       Encoding-tolerant CSV reading
core/data_loader.py  Schema validation, date/time parsing, data-quality warnings
core/fetch.py        The landing-page sign-in, bridged to the scraper
core/epicollect.py   Syncing the Epicollect5 damage projects over the API
core/damage.py       Damage surveys: reading, de-identifying, analysis
core/damage_charts.py Figures for the damage view
core/damage_view.py  The damage and compensation view
core/damage_report.py The damage study as vector HTML and PDF
core/exports.py      SVG / PNG / PDF downloads for any chart or map
core/esri_landcover.py Windowed reads of Esri 10 m land cover
core/refuges.py      Day refuges: detection, cover, raid links, actions
core/refuge_view.py  The day refuges view
core/refuge_report.py Day refuges map and brief section
core/refuge_dashboard.py The vendored atlas plate, filled with a run
core/analytics.py    Severity, conflict classification, KPIs, filters, trends
core/charts.py       Plotly figures for the trend views
core/report_charts.py The same views as static SVG for the brief
core/intelligence.py Beat priorities, escalation, timing, the brief
core/hotspots.py     DBSCAN clustering and village risk
core/spacetime.py    Space-time cube and emerging hotspot labels
core/coverage.py     Early-warning enrolment, de-identified at load
core/spatial.py      Centroid loading and nearest-village enrichment
core/boundaries.py   Vendored forest boundary layers
core/landing.py      Pre-upload masthead and the loaded-state identity bar
core/map_engine.py   Interactive pydeck maps and tooltips
core/map_export.py   Static SVG maps embedded in the brief
core/report.py       Self-contained HTML brief
core/ui.py           Design tokens, SVG icons, shared components
assets/              Hero photograph and partner logos
data/boundaries/     Division, range, beat and reserve outlines
data/centroids.csv   Bundled village centroids
tools/api_source.py  JSON payload shapes, and API routes read from bundles
tools/export_source.py  The Export button's download, whatever format it is
tools/               Data preparation and the scraper, not imported by the app
tests/replica.py     A local stand-in for the Forest Alerts admin site
tests/               Unit tests, run with `pytest -q`
```

All tunables live in `core/config.py`. They encode domain judgement, not fact,
and were set against a 1,761-row Anuppur/Bandhavgarh export.

## Deployment

**Streamlit Cloud.** Point it at `app.py`. `requirements.txt` is pinned,
`runtime.txt` fixes the Python version, and `.streamlit/config.toml` caps
uploads at 10 MB and disables usage stats.

The landing-page fetch makes the deployment's only outbound request, so
`requests` is a runtime dependency rather than a tools-only one, and `openpyxl`
alongside it because an admin panel's Export button usually returns `.xlsx`.
The host has to be able to reach `mpforest.forestalerts.com`. Where it cannot, the fetch
fails with that reason on screen and the uploader still works.

**Reboot after changing anything `app.py` reaches across a module boundary
for.** Streamlit re-executes `app.py` on every rerun but leaves already
imported modules in `sys.modules`, so a deploy can leave a new entry point
running against the previous `core.*`. Manage app → Reboot restarts the
interpreter and clears it; nothing in the tree substitutes for that, because
the two halves really are different versions.

It has bitten twice, and the shape is the same both times. A new constant:
`from core.config import NEW_NAME` raises `ImportError` before anything
renders. A changed signature: `landing.fetch_panel()` raises `TypeError:
missing 2 required positional arguments` against the definition it replaced.
Both read as a bug in code that is, on disk, correct and consistent — so
check the error against `git show <previous-commit>:<file>` before chasing
it. If it names the shape the file used to have, the container is stale and
a reboot is the whole fix.

Two things keep the blast radius down. The entry point imports only
long-standing names from `core.config`, and it reaches optional subsystems
lazily, so a dashboard that cannot do early-warning coverage can still rank
beats — there is a test holding that line. Neither helps where the entry
point must call into a module to draw the page at all, as it must for
`core.landing`; there the window is simply visible, and the reboot ends it.

**Logging.** Writes to `app.log` beside the app and to stdout. Set `ED_LOG_PATH`
to relocate it on a read-only container.

**Secrets.** Only `MAPTILER_KEY`, and only for basemaps. Set it via Streamlit
secrets; never commit it. The app works without it.

## Known limits

- **No authentication.** Field exports carry victim names in the free-text
  description and named reporters, and the filtered-data download exposes both.
  Put the app behind auth or keep it private before sharing a link. This is the
  most important item on this list.
- **Basemap needs network.** Tiles come from MapTiler (or the keyless Carto
  fallback), so an air-gapped deployment gets points on a plain background. The
  brief says so in the figure rather than leaving it ambiguous. Its basemap is
  fetched when the brief is generated, so a brief made offline stays that way
  even if it is opened on a connected machine later.
- **Map labels are thinned.** deck.gl has no label collision handling, so
  labels are kept only where they clear 70 px of each other at the starting
  zoom, capped at 8. On a landscape-wide view that is often only three or four.
  Unlabelled villages are still plotted and still have tooltips.
- **Hotspot clustering is unweighted.** Every sighting counts the same, so a
  hotspot marks where reporting concentrates. The conflict-share and casualty
  columns are what separate a busy patrol route from a dangerous place.
- **Village casualty columns do not sum.** Search radii overlap, so one fatality
  is credited to every village near it. The app states this under the table.
- **A registration is not a reachable person.** Coverage counts enrolments. A
  number that has changed hands, or a handset that is off overnight, counts the
  same as a phone that rings.
- **5% of registrants sit more than 2 km from any centroid** and count towards
  no village. Where two settlements are close, a count can also land on the
  neighbour.
- **Staffing is division-level only.** The user list's Range and Beat fields are
  placeholders, and a low reports-per-user figure is as likely to mean accounts
  created and never used as a quiet division.
- **No persistent store.** Fresh upload per session, so month-over-month
  comparison means re-uploading.
- **Effort is not modelled.** Rates control for how many reports a beat filed,
  not how many patrols produced them.
- **Night is a fixed 18:00–06:00 window**, not sunset-to-sunrise, which shifts
  by over an hour across the year in central India.
- **`Movement Direction` is unused** — inconsistently cased and mostly blank.
- **Group composition is often unrecorded.** Reports with no male/female/calf
  breakdown are labelled Unrecorded rather than guessed at, so the bull share
  is computed only from what was actually counted. Beats with fewer than three
  conflict events get no composition claim at all.
- **Composition is not individual identification.** A lone bull seen twice may
  be two bulls. `research/herds.py` chains sightings into movement units, but
  those are inferred from time, place and group size — not identified animals.
