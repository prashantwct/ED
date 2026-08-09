# Elephant Conflict Intelligence

Streamlit app for turning Gajrakshak elephant sighting/conflict exports into
decisions a protected-area manager can act on: which beats to resource, which
shift to staff, which villages to warn, and what is getting worse.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Test it

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## What it produces

The dashboard reports what happened. The intelligence layer
(`core/intelligence.py`) answers what to do about it.

**Beat priorities.** Every beat gets a decision *tier* — Critical, High, Watch,
Routine — plus the evidence behind it: reports, conflict events, adjusted
conflict rate, people killed and injured, night share, village proximity, trend,
and a recommended action.

**Escalation.** Each beat's most recent N days are compared against the N days
immediately before, like for like. A beat that has been large for years needs
steady resourcing; a small beat that has doubled needs someone to go and find
out why. Only the second is news.

**Timing.** The shortest contiguous block of hours holding a target share of
conflict events, reported with how concentrated it actually is. Seasonality is
reported the same way, and stays silent when there isn't any.

**Movement hotspots.** Density clusters found in the point data itself, not in
the administrative grid. A herd working a corridor along a beat boundary shows
up as moderate pressure in two beats but as the one hotspot it actually is
here. Each hotspot reports its centre, footprint, conflict share, casualties
and night share.

**Villages at risk.** Every conflict incident within a chosen radius of each
village, so a settlement sitting between two hotspots is credited with both —
nearest-village attribution hides exactly that case. Requires village
centroids; see the note below.

**A brief.** One structured object renders both the in-app panel and the
downloadable HTML, so the numbers on screen and the numbers in the document
forwarded upwards cannot drift apart.

### Village data is required to name villages

The sighting export has no village field. Beat, Range and Division are the only
place names in it, and the free-text description is inconsistent — in a real
1,761-row export only 5 rows used "निवासी" and 29 used "ग्राम", and those lines
carry victim names. Scraping villages out of it would be both unreliable and a
privacy problem, so the app does not.

Village ranking therefore needs a centroids CSV with `Village, Latitude,
Longitude`. The Census village directory, a state forest department village
layer, or OpenStreetMap place nodes all work. Without it, hotspots are still
located, sized and tiered — they just cannot be named, and the app says so
instead of returning an empty table with no explanation.

### Design rules

**Tier decides, score only orders.** Tiers come from fixed, written-down rules,
so "Critical" means the same thing in April as in October and in one division as
in the next. The continuous priority score orders beats *within* a tier and
nothing more. Scores normalised against whatever is currently on screen move
when a filter changes — a number like "67.3" is not a fact about the beat, and a
posting decision must not hang on it.

**Critical means recent.** A beat reaches Critical only if someone was killed or
injured there in the last 90 days of the period under review. A fatality two
years ago is history and should not hold a beat at the top of the list forever;
it still puts the beat in High, because somewhere that has killed someone is not
a routine beat either. Both the recent and the whole-period casualty counts are
shown, so the tier is always accountable to figures on screen.

The window is anchored to the end of the review period, not to each beat's own
last report. Otherwise selecting a beat that stopped reporting six months ago
would measure its window from its own final entry and flip an old fatality back
to Critical purely because it was selected. It is also a fixed constant rather
than the adjustable escalation window — a tier that moves when you drag a slider
is not a tier.

**Raw rates from thin data are not evidence.** One conflict in one report is a
100% conflict rate. Beat rates are shrunk toward the landscape rate using an
empirical-Bayes estimator whose strength is fitted from how much beats actually
differ from one another, and every beat carries an explicit confidence label.

**Counts measure patrol effort as much as elephants.** More reports from a beat
can mean more conflict or simply more staff walking it. Nothing here can fully
separate the two, so rates sit alongside volumes rather than replacing them, and
the brief says so every time rather than only on bad days.

## What changed (August 2026 review)

The previous pass documented a set of fixes in this README, but the last upload
to `core/` reverted the analytics to a pre-fix state while the README and tests
kept describing the fixed one. The test suite did not run at all — it failed at
import on four functions that no longer existed — so nothing caught it.

### Correctness

- **Human deaths were scoring below crop damage again.** A two-person fatality
  scored 0.5, identical to a plain footprint sighting; ordinary crop damage
  scored 3.0. Deaths are weighted per person killed (100 points/person) from the
  actual `Male/Female/Children Death Count` fields.
- **The conflict KPI dropped pure-fatality reports.** The mask was
  `Crop OR House OR Injury` — a death-only report counted as no conflict at all.
  Death and Grain Damage are both included now.
- **Day-first dates were silently mangled and rows deleted.** Inferred parsing
  locks onto the layout implied by the first value: in a `DD/MM/YYYY` export,
  `03/04/2026` was read as 4 March and `21/04/2026` failed to parse and was
  dropped as "unreadable". Formats are now tried explicitly against the whole
  column, and genuine ambiguity is reported rather than guessed.
- **Nearest-village search was biased and its distances wrong.** The neighbour
  tree was built on unscaled lat/lon, which treats a degree of longitude as
  equal to a degree of latitude. At 22 °N that overstates east-west distance by
  8.1% and pulls the "nearest" village north-south. Longitude is now scaled by
  cos(latitude) and distances reported with haversine.
- **The near-village radius was too tight to fire.** A village is not a point;
  at 0.5 km from the centroid almost nothing qualified and the proximity signal
  contributed nothing. Now 2 km, which is what centroid-only data supports.
- **Map points were invisible.** This landscape spans ~150 km, which the
  adaptive view fits at ~950 m/pixel — a 60 m radius is 0.06 px. Points now have
  a pixel floor, are coloured by conflict category rather than a severity ramp
  that renders every non-fatal incident the same green, and are log-scaled so
  property damage stays distinguishable next to a fatality.
- **The report interpolated CSV values into HTML unescaped.** Beat and village
  names are now escaped; the file gets emailed onward.
- **Severity bands were equal-width.** With fatalities at 100 and sightings at
  0.5, that put ~everything in bucket one. Bands are now fixed at the boundaries
  that mean something: presence, crop/grain, property, injury, fatality.
- **The reset-filters button did nothing.** It called `st.rerun()` without
  clearing the widget state it was meant to reset.
- **Cascading filters kept stale selections.** Widening Division back out left
  Range holding values no longer offered, silently excluding rows while the UI
  looked reset. Selections are now pruned before the dependent widget renders.
- `use_container_width` (deprecated, removal date passed) replaced with `width=`.

- **A single blank Beat took the whole app down.** On the Arrow-backed string
  dtype pandas now uses, `astype(str)` leaves a missing value as NaN rather than
  the string `"nan"`, so the null survived normalisation into `Beat`. The
  sidebar builds its filter options with `sorted(df["Beat"].unique())`, which
  then raised `'<' not supported between 'float' and 'str'`. Found by running
  the app against a real export, not by reading the code. Blanks now become
  "Unknown" and are reported.
- **`HH:MM:SS` times fell back to per-element dateutil parsing**, emitting a
  "could not infer format" warning on a file that was in fact perfectly
  consistent. Both field formats are now tried directly.

### Added

- `core/intelligence.py` — the layer described above.
- `core/hotspots.py` — DBSCAN clustering of movement (via scipy's KD-tree, no
  new dependency) plus village risk ranking. Parameters were tuned against a
  real 1,761-row export rather than picked for roundness: DBSCAN chains, and at
  a 2.5 km neighbour distance 94% of points collapsed into 7 "hotspots", the
  largest with a 17.7 km radius — a region, not a patrol target. At 1.0 km the
  radii settle at 1–3 km. Anything over 5 km is flagged in the output as
  probable chaining.
- `core/ui.py` — design tokens, inline SVG icons and shared components. Tier is
  never signalled by colour alone: every badge carries a shape, a glyph and a
  word, so the ranking survives a black-and-white photocopy and readers with a
  colour vision deficiency. Map categories use the Okabe-Ito colourblind-safe
  sequence rather than a red-to-green ramp, with size carrying the same signal
  in parallel. Data columns use tabular figures. Both light and dark themes are
  defined.
- Data-quality warnings surfaced in-app, including the death-count/death-flag
  mismatch, named with row IDs. Those rows are excluded from the death count on
  the basis that they have matched same-day, same-beat follow-up reports of a
  death already logged elsewhere — a judgement call about someone's death, so it
  is stated rather than resolved silently.
- Monthly trend, hourly conflict profile, and conflict-rate-by-division.
- 103 tests across six files, covering the regressions above and the properties
  that make the ranking usable: shrinkage, tier stability under filtering,
  casualty recency, escalation, cluster compactness, kilometre-accurate radii,
  midnight-wrapping windows, and the flat-data cases.

### Known limits

- **No authentication.** The data contains named victims of fatal incidents and
  named field reporters, and the filtered-data download exposes them. This needs
  addressing before the app is deployed anywhere shareable.
- **No basemap.** The map renders points on a blank background (`map_style=None`),
  so there is no terrain, boundary or river context. Fixing this means choosing
  a tile provider, which is a licensing and network-access decision — relevant
  if the deployment is air-gapped.
- **No persistent store.** Still a fresh CSV upload per session, so
  month-over-month comparison across sessions requires re-uploading files.
- **Effort is not modelled.** Conflict rates control for how many reports a beat
  filed, but not for how many patrols produced them. A beat that files reports
  only when something goes wrong will look worse than one that logs every walk.
- **Night is a fixed 18:00–06:00 window,** not sunset-to-sunrise, which shifts by
  more than an hour across the year in central India.
- `core/risk_engine.py` is superseded by `core/intelligence.py` and no longer
  used by the app or the report. It is kept only so existing callers keep
  working, and is a candidate for deletion.
- `centroids.csv` still does not exist; `centroids.sample.csv` shows the format.
  Enrichment no-ops without it and the app says so.
- **Hotspot clustering is unweighted.** Every sighting counts the same toward
  cluster density, so a hotspot marks where reporting concentrates, which is not
  identical to where elephants concentrate. The conflict share and casualty
  columns are what separate a busy patrol route from a dangerous place.
- **`Movement Direction` is not used.** The field exists but its values are
  inconsistently cased (`northWest`, `north_east`, `northEast`) and it is blank
  on most rows, so corridor direction is not yet derived from it.
