# Elephant Sighting & Conflict Dashboard

Streamlit app for reviewing Gajrakshak elephant sighting exports: severity scoring,
conflict-rate-by-division, night-window risk, and a beat-level priority ranking.

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

The test suite is a regression net for the specific bugs this rewrite fixed (see
below) -- if any of them come back, a test should fail rather than the app
quietly shipping a wrong number again.

## What changed (July 2026 review)

### Fixed
- **Human deaths now outrank routine crop damage in severity scoring.**
  Previously `Death` had no weight at all -- a fatality scored 0.5 (identical to
  a plain footprint sighting), while ordinary crop damage scored 3.0. Deaths are
  now weighted per person killed (100 points/person), using the actual
  `Male/Female/Children Death Count` fields rather than a flat indicator, so a
  two-person fatality outranks a one-person one.
- **The "conflicts" KPI no longer drops pure-fatality reports.** The old
  definition was `Crop OR House OR Injury` -- Death wasn't in the condition, so
  a death-only report (no crop/house/injury flag) didn't count as a conflict at
  all. Now included.
- **`risk_engine.py` was not valid Python** (an indentation error, confirmed by
  a direct import attempt) and wasn't wired into `app.py` either. Rewritten,
  tested, and now feeds the "Priority beats" table.
- **Death-count vs. Death-flag mismatch in the source data.** Two rows have
  `Male/Female Death Count` populated but the `Death` flag not set. Both matched
  a same-day, same-beat, same-gender report of a fatality already logged
  elsewhere in the export -- almost certainly a duplicate/follow-up entry, not a
  new death. These are excluded from the death count (to avoid double-counting)
  but surfaced as a named, ID-specific data quality warning in the app rather
  than silently resolved either way -- worth confirming with the field reporters.
- **Map points are now visible at landscape scale.** The data spans roughly
  150 km x 140 km; the old radius-in-metres encoding made every point a few
  pixels or less at the zoom level used. `radius_min_pixels`/`radius_max_pixels`
  now floor point size regardless of zoom, and points are colored by conflict
  category (death/injury/house/crop/presence) instead of uniform red.
- Explicit `%d/%m/%Y` date parsing instead of implicit format inference (worked
  by luck on this file; not guaranteed on a future export).
- Cascading filter bug: narrowing the Division filter correctly narrowed the
  Range options, but widening Division back out left Range holding its stale,
  narrower selection -- silently keeping rows excluded even though the UI
  looked "reset." Fixed via explicit session-state sync.
- `use_container_width` (deprecated, removal date already passed) replaced with
  `width=`.

### Added
- Sidebar filters: Division, Range (cascading, see above), date range.
- Monthly trend and hourly conflict-profile charts (Plotly was already a listed
  dependency and was never actually used).
- "Priority beats" table -- the direct payoff of fixing `risk_engine.py`: beats
  ranked by accumulated severity, night share, and village proximity.
- Conflict-rate-by-division table (not just raw sighting counts, which favor
  whichever division has the most reporting activity rather than the most
  actual conflict).
- Grain Damage is now included as its own conflict type (previously ignored
  entirely). This moves the headline conflict count from 363 to 378 --
  intentional, not a regression: 15 rows had grain damage with no other
  conflict flag set.
- Data-quality warnings are now surfaced in-app (bad dates, bad coordinates,
  the death-count/flag mismatch above) instead of silently coerced or dropped.
- `tests/test_analytics.py` -- smoke + regression coverage for the above.

### Explicitly out of scope for this pass (see the review discussion for the
fuller phased plan)
- No persistent store -- still a fresh CSV upload per session, so month-over-month
  comparison across sessions still requires keeping and re-uploading files.
- No authentication/access control. The data contains named victims of fatal
  incidents and named field reporters; if this is ever deployed somewhere
  shareable, that needs addressing before the data does.
- Map is still point-based (no clustering/hex-binning) -- fine at
  division/range zoom, still visually dense at full-landscape zoom with all
  ~2,100 points loaded at once.
- `centroids.csv` (village locations, for the "Near Village" risk factor) still
  doesn't exist. `attach_nearest_village()` continues to safely no-op without
  it, and the app now says so explicitly instead of staying silent about it.
