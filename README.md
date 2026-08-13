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

Then upload a sightings CSV. Village centroids for the Shahdol–Anuppur
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

**MapTiler key (optional).** Set `MAPTILER_KEY` in `.streamlit/secrets.toml`
locally, or under Manage app → Settings → Secrets on Streamlit Cloud. See
`.streamlit/secrets.toml.example`. Without it both the on-screen maps and the
brief's embedded maps fall back to the keyless Carto basemap, so the choice of
terrain, satellite or topographic styling is what the key buys — not whether
there is a basemap at all.

The key is fetched by the browser to load tiles, so it is visible to anyone
using the app and cannot be kept secret. Restrict it by origin in the MapTiler
dashboard (Keys → Allowed origins) rather than relying on secrecy.

**Village centroids (bundled).** `data/centroids.csv` — 9,229 villages,
`Village, Latitude, Longitude`, UTF-8. Constant reference data, loaded by
default from a repo-relative path so it resolves regardless of working
directory. The sidebar uploader overrides it for a different landscape.

Sighting exports carry no village field, so villages can only be named against
this file. The free-text description is not mined for them: it is inconsistent
(in a 1,761-row export, 5 rows used "निवासी" and 29 used "ग्राम") and carries
victim names.

## What it produces

**Beat priorities.** Every beat gets a decision tier — Critical, High, Watch,
Routine — with the evidence behind it: reports, conflict events, adjusted
conflict rate, casualties recent and total, night share, village proximity,
which animals are causing it, trend, and a recommended action.

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

**Villages at risk.** Every conflict incident within a chosen radius of each
village, so a settlement between two hotspots is credited with both.

**Timing.** The shortest contiguous block of hours holding a target share of
conflict, reported with how concentrated it actually is. Seasonality likewise,
and silent when there is none.

**Two maps.** The spatial view carries three toggleable layers — sightings
coloured by what happened, hotspot footprints drawn at true radius, and
villages by tier. The village-risk map leads with the settlements: circle size
is conflict count, colour is tier, and the worst are labelled. Basemap is
selectable (terrain, satellite, hybrid, topographic, streets, minimal).

**A brief.** One structured object renders both the in-app panel and the
downloadable HTML, so the numbers on screen and in the forwarded document
cannot drift apart. The brief names the divisions it covers in its header and
embeds both maps as inline SVG over a basemap stitched from map tiles at
generation time and inlined as a JPEG, so it stays a single self-contained file
that prints.

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
core/analytics.py    Severity, conflict classification, KPIs, filters
core/intelligence.py Beat priorities, escalation, timing, the brief
core/hotspots.py     DBSCAN clustering and village risk
core/spatial.py      Centroid loading and nearest-village enrichment
core/map_engine.py   Interactive pydeck maps
core/map_export.py   Static SVG maps embedded in the brief
core/report.py       Self-contained HTML brief
core/ui.py           Design tokens, SVG icons, shared components
data/centroids.csv   Bundled village centroids
tests/               Unit tests, run with `pytest -q`
```

All tunables live in `core/config.py`. They encode domain judgement, not fact,
and were set against a 1,761-row Anuppur/Bandhavgarh export.

## Deployment

**Streamlit Cloud.** Point it at `app.py`. `requirements.txt` is pinned,
`runtime.txt` fixes the Python version, and `.streamlit/config.toml` caps
uploads at 10 MB and disables usage stats.

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
