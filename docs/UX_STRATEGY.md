# UI/UX strategy: from analyst's notebook to field console

**Status:** proposal · **Scope:** the Streamlit app (`app.py`, `core/ui.py`,
`core/landing.py`, `core/refuge_view.py`, `core/damage_view.py`) · **Leaves
alone:** the analytics, tier rules, privacy rules and the brief's content

---

## 1. Summary

The analysis behind this app is careful and well argued. The interface is laid
out like an analyst's notebook: every result is on screen, every caveat is
written out, and every model parameter can be changed. The people it is meant
for (a DFO before the weekly review, a range officer deciding where tonight's
patrol goes) need three answers in under a minute: **what changed, where to
act, and whom to warn.** At the moment those answers are spread across one long
page and six tabs, behind about 22 sidebar controls, and among roughly 60
status, caption and info messages.

Comparable tools (EarthRanger, SMART Connect, Global Forest Watch, NASA FIRMS,
Chhattisgarh's Gajsanket, ArcGIS operational dashboards) settle on the same
few patterns:

1. **The map is the main canvas**, and the lists and charts are indexes into it.
2. **One scope (area and period) is picked once**, shown at all times, and
   applies to every view.
3. **There is a simple default mode**, and the advanced analysis sits one step
   away.
4. **The default time window is the current one** ("last 24 hours" or "last 7
   days"), not the whole archive.
5. **Every view has a URL** that can be shared.

The strategy takes the current app toward those patterns in five moves, in
order of payoff:

| # | Move | What the user notices |
|---|------|-----------------------|
| 1 | **Add a "This week" overview as the home page** | Opening the app answers "what changed, where to act, whom to warn" on one screen |
| 2 | **Use one scope bar on every view** (area · period · incident type) | Filters stop living in a long sidebar, carry across views, and go into the brief and the URL |
| 3 | **Split viewer and analyst modes** | Managers no longer see clustering distances or FDR checkboxes, and analysts lose nothing |
| 4 | **Make the map page full-width with click-to-profile** | Clicking a beat or village opens why it is ranked, its trend and what to do |
| 5 | **Treat the phone as a first-class screen and unify the three views** | A range officer can use it in the field, and refuges and damage feel like part of the same tool |

A phased plan (section 10) begins with about a dozen fixes that need no
restructuring and can ship in days.

---

## 2. How this was assessed

* **Ran the app** at commit `ebfb014` on Streamlit 1.61.1, at 1440×900
  (desktop) and on an iPhone 13 profile (390×844). It was loaded with a
  **synthetic 1,580-row register** generated over the real beat boundaries for
  Anuppur, North and South Shahdol and Umaria. The screenshots and counts below
  come from that file. **No real data was used.**
* **Read the UI code**: `app.py` (1,645 lines), `core/ui.py`,
  `core/landing.py`, `core/refuge_view.py`, `core/damage_view.py`.
* **Benchmarked** against published documentation for comparable conservation
  and operational dashboards (section 4, sources in section 13).
* The test container had no external network. The basemap, the Esri land-cover
  read and the refuge atlas's D3/font CDN were all blocked. Blank basemaps are
  not counted as findings. The *dependence* on those resources is counted,
  because district and range offices have the same problem on bad days.

---

## 3. What already works (keep it)

The redesign should keep these. Several of them are better than what the
benchmark tools do.

* **Tabs named after questions** ("Is it getting worse?", "Where to act").
  This is the right instinct. The strategy extends it from tabs to pages.
* **Tier badges that never rely on colour alone** (shape + glyph + word), the
  Okabe-Ito category palette, and tabular figures (`core/ui.py`).
* **"Tier decides, score only orders"** and the other integrity rules in the
  README. The simplification below must not blur them.
* **Picking a table row focuses the map** (`app.py:386`, `_map_pick`). This is
  the seed of the map/list linkage described in section 6.
* **A brief that prints**, plus per-chart SVG/PNG downloads in the chart's
  title row.
* **Personal data is dropped at the door** and credentials are never stored.
  Any feature that "remembers" state has to respect this (section 11).
* **Recent accessibility work** (commit `2938c63`): 44 px targets, focus
  rings, larger minimum text, reduced motion.

---

## 4. What similar dashboards do

| Product | Built for | Pattern worth borrowing | Lesson for this app |
|---|---|---|---|
| **EarthRanger (web)** | Protected-area control rooms | Map-first. A left panel switches between *Events*, *Patrols* and *Map Layers*. The filters set in the Events tab also drive the event feed and the map. A **time slider** sits on the map toolbar. Export takes "what is currently filtered". Layer on/off state survives a refresh. | Map and list are **two views of one filtered set**. Time control belongs on the map. Remember what the user turned on. |
| **SMART / SMART Connect** | Patrol data across 1,000+ sites | **Standard report templates** for patrol and threat summaries. Cloud dashboards let **non-SMART decision-makers** read results without learning the tool. Regional managers compare several areas. | Separate the *viewer* audience from the *analyst*. Managers get canned answers, and analysts get the workbench. |
| **Global Forest Watch** | NGOs, governments, communities | **Area of interest**: pick a country or protected area, draw one, or upload one, then get a dashboard and **subscribe to alerts** for it. *Places to Watch* is a curated short list of priority areas with a short story for each. | Let users fix **"my area"** once. Lead with a **short ranked list with a reason per item**, not a full table. |
| **NASA FIRMS** | Fire managers, public | **Basic mode is the default** and Advanced, Alerts and Downloads are separate modes. The map **opens on the last 24 hours**. Share-URL and capture tools are built in. | Start in the **operational window**, keep power tools one click away, and give every view a **shareable link**. |
| **Gajsanket (Chhattisgarh)** / **Gaj Rakshak (MP)** | Forest staff and villagers | Herd routes over a chosen time window, filtered by herd. Alerts go to registered villagers **within ~10 km** of a tracked herd. Works offline. | Frame the product around **herds, tonight and villages to warn**. Plan for **intermittent connectivity**. |
| **ArcGIS Dashboards (Esri guidance)** | Operational GIS dashboards | Start from "the questions the user wants answered at a glance". Keep the map simple: **one operational layer that contrasts with the basemap**. Aim for consistency, clarity, visual hierarchy and added context. Link components through actions and **URL parameters**. | Each page answers named questions. Fewer simultaneous layers. Use the URL for state. |
| **Global Fishing Watch** | Fisheries managers, public | Saved **workspaces**, area reports, and one consistent word ("apparent") for uncertainty instead of paragraphs of caveats. | State uncertainty **once, consistently and briefly**. Put the full method in its own place. |

The UX principle these share is **progressive disclosure**: put the common
task up front and move advanced or rarely used features to a secondary screen.
That makes the tool easier to learn and less error-prone (Nielsen). For
dashboards it usually means **4–6 headline numbers** with detail reached by
drilling down.

---

## 5. Diagnosis: ten problems, with evidence

Severity: **H** = blocks or misleads the primary user, **M** = slows them,
**L** = polish.

### P1 (H): The most important view is the furthest away

The manager's core question is "where do I act?". It lives on the **second
tab**, and its map starts about **1,480 px down the page** at 1440×900, which is
1.6 screens of scrolling *after* a click. The default tab is the trend charts.
Above the tabs, the uploader, a "Loaded N rows" banner, five KPI cards and a
ten-bullet assessment take the whole first screen. *(app.py:420–497, 727–776,
816–870)*

### P2 (H): The sidebar mixes five kinds of control

There are **about 22 controls** in one scrolling column on the conflict view:

| Kind | Controls | Who needs them |
|---|---|---|
| View switch | View radio | Everyone, often |
| Data sources | Village centroids uploader, villager registry, app-user list | Analyst, rarely |
| Scope filters | Date, Division, Range, Beat, Minimum incident type, Reset | Everyone, often |
| Model parameters | Escalation window, hotspot neighbour distance, minimum sightings per hotspot, village exposure radius, time step, cell size, count, multiple-testing correction | Analyst, rarely |
| Map styling | Basemap, boundaries on/off, boundary level | Occasionally |

Two problems follow. First, the frequently used controls sit between the
rarely used ones. Second, **a manager can change a clustering parameter by
accident**, and two officers can then compare different results without
knowing it. Nothing on the page or in the brief says the settings are
non-standard. *(app.py:329, 500–690)*

### P3 (H): Status chatter takes the best space

After a successful load the user sees a green "Loaded 1,580 valid rows" banner,
a yellow sidebar warning that "2740 village name(s) appear more than once…", a
green "Enriched with 9,229 village centroids", and a developer note under every
map: "No MAPTILER_KEY set, so the basemap selector is inactive…". The uploader
stays at the top of the page. In total `app.py` contains **36
success/info/warning calls**. Most of them confirm that nothing is wrong, which
teaches users to ignore the yellow and green boxes that do matter. *(app.py:497,
539, 545; core/map_engine.py:301)*

### P4 (M): Explanation outweighs signal

Besides the warnings there are **23 captions**, chart notes, and "How to read
this" expanders in `app.py` alone, and 10 more captions in each of the refuge
and damage views. Some caveats raise doubt without helping the decision. One
example under the villages table: *"These rows total 53 deaths against 16
recorded."* The statement is accurate, but a reader skimming it concludes the
numbers are wrong.

### P5 (H): The assessment is a flat list

The assessment has ten equally weighted bullets. It mixes outcomes ("16 people
killed and 18 injured"), priorities ("4 beats in the Critical tier…"), method
notes ("565 conflict reports … are counted as night raids: they are written up
after the night") and trivia ("Seasonal peak: Oct, Dec, Nov", listed in rank
order, so it reads as unsorted). Nothing is clickable, so the beat names in a
finding cannot be opened or shown on the map.

### P6 (M): Analyst vocabulary

Users meet terms such as *Escalation window*, *Hotspot neighbour distance*,
*Minimum sightings per hotspot*, *Correct for multiple testing*, *Space-time
cube*, *Sporadic / Historical / Consecutive / Diminishing*, *Conflict rate*
(it means "share of reports that involved damage or harm", which nobody would
guess) and *Seen at night* (with a help icon to explain which reports count).
A 3D space-time cube with rotation, focus and per-step highlighting sits in the
same tab as the manager's "is this new?" question. *(app.py:631–667,
1195–1340)*

### P7 (M): Three views feel like three apps

* The **view switch is a radio button in the sidebar**. It is easy to miss and
  hidden entirely on phones.
* **Day refuges** has its own Division filter (`refuge_divisions`), separate
  from the conflict filters, so the user picks the area twice. Its embedded
  atlas has **different typography, controls and visual language** (Barlow,
  Atkinson Hyperlegible, IBM Plex Mono, its own "Colour by" and "Go to" bars).
  It also loads **D3 and fonts from public CDNs**, so its map is blank without
  internet. Land-cover failures are shown as raw errors ("CURL error: CONNECT
  tunnel failed, response 403"). *(core/refuge_view.py:468;
  assets/refuge_dashboard.html)*
* **Damage & compensation** opens empty with its own uploader. It does not say
  whether the register already loaded will be used.

### P8 (H): Phones are an afterthought

On an iPhone 13 (664 px of visible page) the filters and the view switch are
behind the `>>` sidebar toggle. The masthead, uploader and banner use about
390 CSS px, and the **five KPI cards stack to about 1,030 CSS px**. The
assessment therefore starts **2.2 screens down**. The map, after switching to
the *Where to act* tab, starts **5.7 screens down**. Wide tables truncate (the
villages table's last column shows as "Hots… N/A"). Range officers and beat
guards are the people most likely to open a shared link on a phone.

### P9 (M): No "now" in the default view

The default period is **the whole register** (two years in the test file). The
KPI deltas compare 90-day windows, and nothing says what changed since the last
visit or since last week. The operational questions are about **tonight, this
week and this season**. FIRMS and Gajsanket both open on the most recent
window.

### P10 (M): No memory, no sharing, rough edges

* No URL captures the view. "Look at Jamudi beat, last 30 days" cannot be
  sent, and a reload loses every filter.
* KPI delta labels truncate at desktop width ("vs sa…", "vs same 9…").
* Streamlit's **"Deploy" developer button** showed in the test run.
  `client.toolbarMode` is unset, so it defaults to `auto`, which shows
  developer options on localhost and to the app's developer on Community
  Cloud. Anyone who runs the app locally (a training laptop, an office
  machine) sees it. Setting `viewer` hides it everywhere.
* Map legends sit *below* the maps instead of on them. Each of the four maps
  has a slightly different legend layout.

---

## 6. Design principles for the redesign

1. **Answer, then evidence, then method.** Each page leads with a one-line
   answer, follows with the evidence (map, ranked list, one chart), and keeps
   the method behind a link. The brief's "findings → action plan → ecology"
   structure already does this, and the app should match it.
2. **The map is the canvas and lists are its index.** A list row, a map
   feature and a finding naming a beat all lead to the same profile.
3. **One scope, everywhere.** Area and period are chosen once, shown always,
   and carried into every view, the brief and the URL.
4. **Defaults are decisions.** Viewers get the standard parameters. Analysts
   can change them, and a non-standard setting is **badged on screen and
   stamped on every output**.
5. **Use the user's words.** Beat, range, tonight, patrol and warn are the
   vocabulary. Statistical terms belong in tooltips and the method page.
6. **Quiet when fine, loud when not.** A message appears only if the user needs
   to do something. Confirmations collapse into a status pill.
7. **The phone is a first-class screen.** The first phone screen has to
   contain the first action.
8. **Keep the integrity rules.** Tier logic, colour-not-alone, the privacy
   rules and "score only orders" are unchanged. This is a presentation change.

---

## 7. Target information architecture

### 7.1 Navigation

Replace the sidebar radio and the six tabs with **top navigation**
(`st.navigation(..., position="top")`, available in the pinned Streamlit
1.61.1):

```
Overview │ Map │ Villages │ Patterns │ Day refuges │ Damage │ Data & brief │ ⚙ Analyst
```

| Page | Replaces | Main question |
|---|---|---|
| **Overview** (home) | Headline KPIs + Assessment | What changed, where to act, whom to warn? |
| **Map** | *Where to act* tab | Where exactly, and why there? |
| **Villages** | *Villages at risk* tab | Which villages, and are they enrolled for warnings? |
| **Patterns** | *Is it getting worse?* + *When* + *New or growing?* (2D parts) | Is it getting worse, when does it happen, what is new? |
| **Day refuges** | Day refuges view | Where do they lie up, and what do they raid? |
| **Damage** | Damage & compensation view | Who lost what, and who has not claimed? |
| **Data & brief** | *Data* tab + Generate brief + data-quality notes | Can I trust this, and how do I send it? |
| **⚙ Analyst** | Analysis settings, space-time settings, centroid/registry/roster uploads, 3D cube | (Analysts only) Tune, explore, validate |

### 7.2 The scope bar (on every page)

One row under the navigation, the same on every page:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 📍 Anuppur ▸ all ranges ▾ │ Last 7d · 30d · [90d] · Season · Custom │ All incidents ▾ │ ↺ │ 🔗 Share │
└──────────────────────────────────────────────────────────────────────────────┘
  Register: 1,580 reports · 1 Oct 2024 – 31 Oct 2026 · fetched 09:41  ⓘ 2 data notes
```

* **Area** is a popover (`st.popover`) with the cascading Division ▸ Range ▸
  Beat pickers. The bar shows only the resulting label.
* **Period** is a segmented control with presets. *Custom* opens the date
  picker. **The default is "Last 90 days"**, which matches the Critical-tier
  window. That is safe only because the period is no longer the analysis
  frame: tiers and escalation use the history before it (phase 0, item 0.11;
  done in phase 1).
* **Incident type** is a plain-language select: *All reports / Damage or
  worse / Injury or death*.
* **Share** copies a URL carrying the page and the scope (`st.query_params`).
  The link restores the *view*, not the data: the recipient still uploads or
  fetches the register. That is consistent with the "nothing is stored" rule.
* The **source line** replaces the persistent uploader, the "Loaded" banner
  and the sidebar success/warning boxes. *Data notes* opens the data-quality
  list. Replacing the file happens from this popover.
* If analyst settings are non-standard, a **"Custom analysis" badge** appears
  here and links to the ⚙ Analyst page.

The sidebar stays collapsed by default and holds only map styling on the Map
page.

---

## 8. Page specifications

### 8.1 First run (landing)

* Keep the masthead, but **shorten the hero to about 200 px**, so both ways in
  are visible at 1366×768 without scrolling.
* Show **two equal cards side by side**: *Sign in to Forest Alerts* (with the
  date window) and *Upload the export (CSV)*. They replace the uploader,
  divider and expander stacked one above the other.
* Add a third, smaller action, **"Explore with sample data"**: a synthetic,
  person-free register shipped in the repo. It makes training sessions,
  demonstrations and screenshot tests possible without real data.
* Make "Damage surveys" a link under the cards instead of a full-width button.
* Move the capability cards to an "About this tool" section below the fold.

### 8.2 Overview (new home)

Desktop wireframe:

```
┌ Scope bar ───────────────────────────────────────────────────────────────────┐
│ Conflict is up 25% on the same 90 days last year. 4 beats are Critical,      │
│ 2 of them new this month. 9 Critical villages lie outside every hotspot.     │  ← one generated sentence
├──────────────┬──────────────┬──────────────┬──────────────┬──────────────────┤
│ Conflict     │ People       │ Beats        │ Villages     │  [Generate brief]│
│ events  587  │ killed/inj.  │ Critical  4  │ to warn  17  │  [Share link]    │
│ ▲25% vs ly   │ 16 / 18      │ High     22  │ Critical 9   │                  │
├──────────────┴──────────────┴──────────────┴──────────────┴──────────────────┤
│ ACT NOW                         │                                            │
│ ▲ Jamudi · Anuppur              │                                            │
│   Death 12 Oct; lone bull;      │              MAP (60%)                     │
│   night raids ↑. Night patrol,  │   tiers + recent casualties only;          │
│   warn Podaki, Audhera  [Map]   │   click → profile                          │
│ ▲ Bhundakona · Amarkantak …     │                                            │
│ ◆ Masiyari · Amjhor …           │                                            │
│ (top 5, "See all 26 →" Map)     │                                            │
├─────────────────────────────────┴────────────────────────────────────────────┤
│ WHAT CHANGED (since 30 days ago)                                             │
│ • New Critical: Badahar, Tikaria   • New hotspot near Khati   • Escalating:  │
│   Dhanwahi +4, Badahar +3, Jarha +3                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

* **Four headline numbers** instead of five. *Reports* measures effort, not
  outcome, and moves to Data & brief. *Seen at night* moves to Patterns.
  *Beats Critical/High* and *Villages to warn* are added, because they are the
  numbers a manager acts on. Delta labels shorten to fit ("▲25% vs last yr").
* **Act now** shows the top five beats by tier, then by score. Each card
  carries the tier badge, a one-line reason assembled from fields the beat
  table already has (last casualty, composition, trend), the per-beat
  `Recommended Action` that `core/intelligence.py` already computes, and a
  *Map* link. A toggle switches to **"By action"**, which is the brief's
  grouped plan (`brief_insights.action_plan`: "night watch for bulls: these
  seven beats"). That is the form a DFO hands out at a staff meeting.
* **What changed** has three to six clickable chips (new Critical beats, new
  hotspots, escalating beats). The comparison point is the start of the
  selected period, or a "since" date.
* The current ten-bullet assessment is split up. Its **outcome** items feed the
  headline sentence, its **priority** items feed *Act now*, its **trend**
  items feed *What changed*, and its **method** items move to the method page
  linked from *Data & brief*.

Phone wireframe (390 px):

```
┌──────────────────────────────┐
│ ☰  Elephant Conflict  🔗     │
│ Anuppur · last 90 days  ✎    │  ← scope as one tappable line
├──────────────────────────────┤
│ Conflict up 25% on last year │
│ 4 beats Critical (2 new).    │
├──────────────┬───────────────┤
│ Conflict 587 │ Killed/inj.   │  ← 2×2 compact tiles, no sparklines
│ ▲25%         │ 16 / 18       │
├──────────────┼───────────────┤
│ Critical 4   │ To warn 17    │
├──────────────┴───────────────┤
│ ACT NOW                      │
│ ▲ Jamudi — Night patrol;     │
│   warn Podaki, Audhera   ›   │
│ ▲ Bhundakona …           ›   │
├──────────────────────────────┤
│ [ Open map ]  [ Get brief ]  │
└──────────────────────────────┘
```

The first action ("Jamudi: night patrol") should be **inside the first phone
screen**.

### 8.3 Map

* **Full-width map** at least 70 vh tall. A left panel (about 30%, collapsible)
  switches between **Beats / Villages / Hotspots / Animals** lists. This is the
  current table→map focus with the roles reversed: the map is the main element.
* **Click a map feature to open its profile.** `st.pydeck_chart` supports
  `on_select="rerun"` in 1.61. Today only the tables use selection
  (`app.py:1022, 1075, 1110`).
* **Layer controls sit on the map** in a compact panel, the way EarthRanger
  does it. *Events* and *Animals* pills, village tiers and hotspot footprints
  move there from above the table. Following Esri's "one operational layer"
  guidance, open with **tiers + recent casualties** only. Other layers are one
  tap away.
* **The legend goes inside the map frame** (bottom-left), drawn the same way on
  every map.
* **A time slider** under the map scrubs within the scope period by week, so
  movement can be read over time without leaving the map.
* The "No MAPTILER_KEY set" note moves to ⚙ Analyst. When there is no key,
  viewers simply get the fallback basemap.

### 8.4 Beat and village profiles (new drill-down)

Opened from any list row, map feature, *Act now* card or *What changed* chip.
It appears as a side panel on desktop and a full-screen sheet on phone
(`st.dialog`).

```
▲ CRITICAL   Jamudi beat · Anuppur range · Anuppur division
Why: Death on 12 Oct (night, lone bull). Conflict ↑ 3→7 vs last year.
Do:  Night patrol 18:00–03:00 on the Podaki–Audhera edge; warn 3 villages.
──────────────────────────────────────────────────────────
[sparkline: conflict per month]   [mini-map: beat outline + incidents]
Incidents (last 90 days): 7  ·  Killed/injured: 1/0  ·  Night share: 80%
Animals: lone bull 5, herd 2       Villages within 3 km: Podaki, Audhera, …
──────────────────────────────────────────────────────────
[Show on map]  [Filter everything to this beat]  [Download beat sheet]
```

All of this comes from tables that already exist (`brief["beats"]`,
`village_risk`, `composition`). No new analysis is needed.

### 8.5 Villages

* Show the essential columns at desktop width: tier, village, incidents
  nearby, killed/injured, nearest hotspot, enrolled for warnings (when the
  registry is loaded). Everything else goes in the CSV.
* The overlapping-radius caveat becomes a **column-header tooltip**
  ("Incidents within 3 km; neighbouring villages share incidents") instead of a
  sentence about deaths not adding up.
* Add a **"Villages to warn tonight"** preset: Critical/High villages within a
  night's reach of an active hotspot or a refuge used in the last 7 days. Add a
  **"Copy as message"** button that produces WhatsApp-ready text. The refuge
  atlas already has one, so the pattern is proven.

### 8.6 Patterns

* One section each for **Getting worse?**, **When?** and **New or growing?**,
  with in-page anchors instead of tabs inside tabs.
* **Lead with one chart per section.** The monthly conflict-by-type chart heads
  *Getting worse?*. The rest go in a two-column grid below it.
* Rename emerging-hotspot labels for viewers: *New*, *Getting worse*,
  *Ongoing*, *On and off*, *Easing*, *Past*. Each has a tooltip with the
  technical name.
* The **3D space-time cube moves to ⚙ Analyst**. Patterns keeps the flat hex
  map and the place-by-time heatmap.

### 8.7 Day refuges

* It reads the **global scope bar**, and its own Division filter is removed.
* **Restyle the embedded atlas** to the app's tokens (font, colours, control
  styles), or at least to the same type family and button style.
* **Self-host D3 and fonts** (inline or under `assets/`), so the atlas works
  without CDN access, as the brief already does.
* Replace raw land-cover errors with: *"Land cover for this area could not be
  downloaded. Refuges are still shown; cover type is missing. [Upload a tile]
  [Details]"*.

### 8.8 Damage & compensation

* It reads the global scope bar. When a register is loaded, say so in the
  empty state ("The register you loaded will be used to check the Gaj Rakshak
  gap").
* Use the empty state to explain the steps: **1** get the export from
  Epicollect5 (or fetch), **2** upload, **3** what you will see.

### 8.9 Data & brief

* **Brief builder**: scope summary (from the bar), section checkboxes, and
  *Generate* → *Download*. The brief is stamped with the scope and any
  non-standard analyst settings.
* **Data notes**: every data-quality warning, centroid note and load message,
  listed once with a count. This is where the sidebar and banner messages go.
* The filtered rows table, the download, and the monthly, severity, division
  and escalation tables from today's *Data* tab.
* **About these numbers**: the "How to read this" caveats, written once and
  linked from wherever a number could mislead.

### 8.10 ⚙ Analyst

* All model parameters: escalation window, hotspot distance and minimum size,
  village radius, space-time step, cell, metric and FDR. Each has a
  plain-language label, the technical name in the help text, and **"Reset to
  standard"**.
* Uploads for centroids, villager registry and app-user list, with the
  privacy note.
* The 3D space-time cube and other exploratory views.
* Basemap and API-key diagnostics.
* No password is needed (the app has no access control of its own). The split
  is about attention, not security. The page should not be the default
  landing.

---

## 9. Content and language guide

### 9.1 Label changes

| Current | Proposed (viewer) | Notes |
|---|---|---|
| Conflict rate | **Share of reports with damage or harm** | Short form "Damage share" in tiles |
| Seen at night | **Night sightings** | Help text: "Direct sightings 18:00–06:00" |
| Killed / injured | **People killed / injured** | |
| Minimum incident type | **Incident type** | Options: All reports · Damage or worse · Injury or death |
| Escalation window (days) | **Compare the last … days** | Analyst page |
| Hotspot neighbour distance (km) | **Merge activity closer than … km** | Analyst page |
| Minimum sightings per hotspot | **Smallest hotspot (reports)** | Analyst page |
| Village exposure radius (km) | **Count incidents within … km of a village** | Analyst page |
| Correct for multiple testing | **Strict test (fewer false alarms)** | Analyst page |
| Is it getting worse? / New or growing? / When | Sections of **Patterns** | Keep the questions as section titles |
| Sporadic / Historical / Consecutive / Diminishing / Intensifying / Persistent / New | On and off / Past / Ongoing / Easing / Getting worse / Ongoing / New | Technical term in tooltip |
| Space-time cube | (Analyst only) | |
| Generate brief | **Get the brief** (primary) | One primary button per page |

### 9.2 Message rules

* **Success:** only after an action the user took (for example "Brief ready").
  A successful load goes in the source line.
* **Info:** only for empty states, and always with the next step ("No
  casualties in this period" is fine on its own).
* **Warning:** only when the user should do something different. Data-quality
  notes go to *Data notes* with a count in the scope bar.
* **Error:** plain sentence, consequence, fix. Technical detail goes behind
  *Details*.
* **Caveats:** inline only where the caveat changes the decision. Otherwise
  link to *About these numbers*.

### 9.3 Language

Field staff in Shahdol, Anuppur and Umaria work mainly in Hindi. A **Hindi /
English toggle** for the Overview page, tier names, actions and the brief would
widen the audience more than any layout change. Plan it in phase 3, and design
labels now so they can be translated (no text inside images, room for longer
strings).

---

## 10. Phased plan

### Phase 0: quick wins (1–3 days, no restructuring) — done (0.11 in phase 1)

| # | Change | Status |
|---|---|---|
| 0.1 | Hide Streamlit's Deploy/dev toolbar: `[client] toolbarMode = "viewer"` | Done (`.streamlit/config.toml`) |
| 0.2 | Remove the "Loaded N valid rows" banner. Show rows and dates in the source note | Done. The source line reads "sightings.csv, 1,580 reports, 1 Oct 2024 to 31 Oct 2026" |
| 0.3 | Collapse the uploader once a file is loaded (show the file name and "Replace") | Done. The keyed uploader moves into a *Replace file* popover and keeps its file |
| 0.4 | Move the centroid duplicate warning and "Enriched with…" out of the sidebar into a "Data notes (n)" expander | Done. Load, centroid and spatial notes sit behind one *Data notes (n)* popover on the source line. Registry and roster feedback stays beside their uploaders |
| 0.5 | Reorder tabs: **Where to act** first | Done. Order: Where to act, Villages at risk, Is it getting worse?, New or growing?, When, Data |
| 0.6 | Put *Analysis settings* and *Space-time trend* in one collapsed "Advanced analysis" expander, with the labels from §9.1 | Done. The sidebar order is now Filters, Map, Advanced analysis, Optional data. *Reset to standard* is inside |
| 0.7 | Show a "Custom analysis settings" badge when any parameter is off its default, and stamp it into the brief | Done (`core/analysis_settings.py`). The badge sits above the headline numbers and the brief's scope line names each change |
| 0.8 | Shorten KPI delta labels so they do not truncate ("vs last yr") | Done. The full comparison is in each card's help |
| 0.9 | Show the MAPTILER note only once and only in the sidebar's map section, not under every map | Done. The basemap selector is disabled and its help says why |
| 0.10 | Plain-language land-cover error with details folded | Done (`core/refuge_view.py`) |
| 0.11 | Default period preset "Last 90 days", with "Whole register" one click away | **Moved to phase 1, and done there** (see below) |
| 0.12 | Drop trailing low-value columns from the villages table at desktop width | Done. Village, tier, incidents, killed, injured, plus nearest hotspot with distance when there are hotspots. The overlap caveat moved to the column headers |

**Why 0.11 moved.** The date filter is not only a display filter. Tiers,
escalation and the headline comparisons are all worked out *within* the
selected period. On the synthetic register, "last 90 days" removed the KPI
comparisons (no earlier data in the frame), cut escalating beats from 4 to 1,
and dropped High-tier beats from 22 to 3. That is an analysis change, not a
quick win. Phase 1 has to separate the **focus window** (what the page shows)
from the **analysis history** (the baselines tiers and trends compare
against) before a short default period is safe.

**Exit check, measured after phase 0** (1440×900, synthetic register): the
first screen shows the source line, five headline numbers and the start of
the assessment, with no confirmation banners and no developer chrome. *Where
to act* now opens first without a click, and its map starts about 1,350 px
down (it was about 1,480 px, behind a tab click). The top of *Where to act*
is still below the first screen. That needs phase 1's Overview, which
replaces the ten-bullet assessment.

### Phase 1: structure (1–2 weeks) — done

| Item | Status |
|---|---|
| Top navigation, one module per page | Done. `st.navigation(position="top")` with eight pages: Overview, Where to act, Villages, Patterns, Day refuges, Damage, Data & brief, Analyst. The page modules live in `app_pages/`; a folder named `pages/` would trigger Streamlit's older automatic multipage mode |
| Shared, cached analysis | Done. `app.py` builds one context per run, and the slow steps (beat ranking, village ranking, hotspots, cube) are cached on the frames they read (`app_pages/common.py`). Moving between pages reuses them |
| Scope bar synced to the URL | Done. Area popover, period presets (30 days, 90 days, 12 months, all, custom), incident type, Reset, and a Share popover showing the link. The scope is written to the query string on every run and read from it when a register first loads (`core/scope.py`) |
| Focus window vs analysis history | Done, and with it **0.11: the default period is now the last 90 days**. Tiers, escalation, hotspots, villages, trends and the brief use the register up to the period's end; the period's start only sets what is counted and drawn. A test holds that a 90-day view ranks beats exactly as the whole register does |
| Overview | Done. Headline sentence, four numbers, *Act now* (beats or actions) beside the map, *Warn first*, and *What changed* in the last 30 days (`core/overview.py`) |
| Data & brief, Analyst | Done. Data & brief holds the brief, the full assessment, "About these numbers", data notes and the tables. Analyst holds every setting, the 3D cube, and diagnostics |

Implementation notes and deviations:

* **Links, not `st.switch_page`.** In a real browser, navigating from inside
  a run with `st.switch_page` dropped the uploaded file, and the next page
  fell back to the landing screen. The Overview's links are real page links
  (`st.page_link`) that carry the scope, plus `focus=beat:<name>` when they
  should open the map on a beat. The entry point applies the focus once and
  drops it from the address.
* **State across pages.** Streamlit forgets a widget's value on any run that
  does not draw it. The scope bar and source line are drawn on every page.
  Settings and map styling are written back to session state at the top of
  every run, so they survive pages that do not draw them.
* **Optional uploads stay in the sidebar,** not on the Analyst page. A file
  uploader cannot be written back to session state, and moving the
  registry's bytes there would break the rule that personal data never
  reaches session state. In the sidebar they are drawn on every conflict
  page and keep their files.
* **Day refuges and Damage** keep their own controls until phase 3. They
  show the source line, so the register survives a visit to them, which it
  did not before.
* **Tests** name a page with `tests/pages.py` (`open_page`). AppTest's own
  `switch_page` takes only file paths, and it does not carry an
  `st.switch_page` over to the next run the way a browser's address bar does.

**Exit check, measured** (1440×900, synthetic register): the headline
sentence, the four numbers, *Act now* and its first beat card, with its
*Map* link, are on the first screen. The usability sessions (T1–T3) have
not been run; they need people, not code.

### Phase 2: map-first and drill-down (2–3 weeks)

* Full-width **Map** page with list panel, in-map legend and layer panel, and
  pydeck `on_select`.
* **Beat and village profiles** (§8.4) opened from every entry point.
* Clickable **What changed** and *Act now*.
* "Villages to warn tonight" plus **Copy as message**.

**Exit check:** T1–T4 pass. The median time for T1 is under 60 s.

### Phase 3: unify and reach (3–4 weeks)

* Day refuges and Damage read the scope bar. Restyle the refuge atlas and
  self-host its assets.
* Phone layout (§8.2 phone wireframe) for Overview, Map and profiles.
* Sample-data mode and a short first-run tour (three steps: scope bar → Act
  now → brief).
* Hindi/English toggle for Overview, tiers, actions and the brief.

**Exit check:** T5 (phone) passes. SUS ≥ 75 across personas.

### Phase 4: beyond the dashboard (needs an owner decision)

* **Saved areas and scheduled briefs** (the Global Forest Watch subscription
  pattern) would require storing a scope and a contact, and possibly the
  register. That conflicts with the current "nothing is stored" rule
  (README, *Design rules*). The project owners should decide this before any
  work starts. A middle option: the brief carries its own share link, and
  recipients bookmark scoped URLs.

---

## 11. Implementation notes (Streamlit 1.61.1)

All of the APIs below are present in the pinned version (checked with
`inspect` against the installed package):

| Need | API |
|---|---|
| Top navigation, page modules | `st.navigation(pages, position="top")`, `st.Page` |
| Scope bar layout | `st.container(horizontal=True)`, `st.popover`, `st.segmented_control`, `st.pills` |
| Shareable state | `st.query_params` (read on start, write on change) |
| Profiles | `st.dialog` (modal) or a right-hand column for desktop |
| Map click | `st.pydeck_chart(..., on_select="rerun", selection_mode="single-object")` |
| Avoid full reruns on map interaction | `@st.fragment` around the map and its panel |
| Hide dev chrome | `[client] toolbarMode = "viewer"` |

Guardrails:

* **Presentation only.** No change to `core/analytics.py`,
  `core/intelligence.py`, `core/hotspots.py` or `core/spacetime.py` is needed
  for phases 0–2. `core/ui.py` already states this separation, and the tests
  that hold the tier rules stay as they are.
* **Privacy.** Query params carry only scope (division, range, beat, period,
  type, page): no names, no coordinates of people, no credentials.
* **Tests.** Add `streamlit.testing.v1.AppTest` cases for navigation and scope
  persistence. Add a Playwright screenshot job at 1440×900 and 390×844 against
  the sample dataset, so layout regressions show up in review.
* **Performance.** The README's "watch what the map ships" rule still applies.
  A full-width map must keep the viewport-filtered boundaries and the minified
  deck spec.

---

## 12. Measuring success

Run short moderated sessions with **five people per persona** (DFO/SDO, range
officer, monitoring-cell analyst) **before phase 0 and after each phase**,
using the sample dataset.

| Task | Persona | Target |
|---|---|---|
| T1: "Which three beats need extra patrols this week, and why?" | DFO | ≥ 4/5 succeed, median < 60 s |
| T2: "Which villages in Anuppur should be warned tonight?" | Range officer | ≥ 4/5, median < 60 s |
| T3: "Is conflict in your division worse than last year?" | DFO | ≥ 4/5, median < 30 s |
| T4: "Send a brief for North Shahdol, last 30 days, to your CCF" | DFO / analyst | ≥ 4/5, median < 2 min |
| T5: T2 on a phone | Range officer | ≥ 4/5, median < 90 s |
| T6: "Change the hotspot distance, then put it back" | Analyst | 5/5; viewers never do it by accident |

Also record the **System Usability Scale** (target ≥ 75), the number of
unprompted questions about terms (target: halve it), and whether participants
notice the Custom-analysis badge.

The app disables usage statistics on purpose (`gatherUsageStats = false`), so
in-product analytics are not assumed. If the owners want them, use anonymous
counts of page and preset use, never scope values tied to a person.

---

## 13. Risks and open decisions

| Risk / decision | Mitigation |
|---|---|
| Analysts lose quick access to parameters | ⚙ Analyst page is one click away. Its settings persist for the session and are badged |
| Fewer caveats lead to over-reading | Caveats stay in the brief and *About these numbers*. Inline caveats remain where they change a decision (e.g. "counts track patrol effort") |
| Streamlit layout limits (no true sticky header, rerun model) | Fragments for the map. Scope bar repeated at the top of each page. Revisit a JS front end only if offline field use becomes primary |
| Saved areas vs "nothing stored" | Owner decision (phase 4). Scoped URLs are the interim |
| Hindi translation quality | Use the forest department's own terms for tiers and actions. Review with field staff |

---

## 14. Sources

Benchmarks:

* EarthRanger Web, glossary and help pages: [Glossary](https://earthranger-support.helpjuice.com/en_US/earthranger-glossary),
  [EarthRanger Web](https://earthranger-support.helpjuice.com/en_US/earthranger-web),
  [Filter events](https://earthranger-support.helpjuice.com/en_US/filtering-reports),
  [Time slider](https://earthranger-support.helpjuice.com/en_US/timeslider),
  [Map layers](https://earthranger-support.helpjuice.com/map-layers),
  [Mobile 2.8.0 notes](https://earthranger-support.helpjuice.com/en_US/earthranger-mobile-version-2-8-0)
* SMART: [Global Conservation, SMART overview](https://globalconservation.org/methodology/global-park-defense/smart-spatial-monitoring-and-reporting-tool/),
  [WCS, SMART at 1,000+ sites](https://newsroom.wcs.org/News-Releases/articleType/ArticleView/articleId/16647/SMART-Technology-Transforms-Conservation-at-Over-1000-of-the-Worlds-Most-Important-Biodiversity-Sites.aspx),
  [Mongabay, SMART Connect](https://news.mongabay.com/2017/10/smart-and-well-connected-reserve-patrol-data-system-adds-communications-capacity/amp/)
* Global Forest Watch: [Introducing My GFW](https://blog.globalforestwatch.org/features/introducing-my-gfw-a-customized-forest-monitoring-experience),
  [Places to Watch](https://globalforestwatch.org/blog/data-and-tools/places-to-watch-curating-data-driven-stories-for-action),
  [Mongabay, early warning for forest loss](https://news.mongabay.com/2017/11/an-early-warning-system-for-locating-forest-loss/)
* NASA FIRMS: [Updated user interface](https://www.earthdata.nasa.gov/news/blog/updated-user-interface-firms),
  [FIRMS data tool focus](https://earthdata.nasa.gov/learn/articles/firms-data-tool-focus)
* Global Fishing Watch: [User guide](https://globalfishingwatch.org/user-guide/),
  [Global reports on the map](https://globalfishingwatch.org/platform-update/introducing-global-reports-on-the-global-fishing-watch-map/)
* Elephant alert systems in India: [NITI Frontier Tech, Gajsanket](https://frontiertech.niti.gov.in/story/gajsanket-how-ai-alerts-and-foot-patrols-ended-a-deadly-elephant-conflict-in-chhattisgarh/),
  [Drishti IAS, Chhattisgarh tracking](https://www.drishtiias.com/state-pcs-current-affairs/hi-tech-monitoring-of-the-movement-of-elephants-in-chhattisgarh/print_manually),
  [Free Press Journal, MP Gajrakshak app](https://www.freepressjournal.in/bhopal/madhya-pradesh-forest-department-launches-gajrakhshak-app-to-track-elephants-will-alert-villagers-in-real-time),
  [GK Today, Gaj Rakshak in Bandhavgarh](https://www.gktoday.in/gaj-rakshak-app-enhances-elephant-monitoring-in-bandhavgarh),
  [WION, Assam's Haati app](https://www.wionews.com/india-news/assam-launches-mobile-app-haati-to-reduce-human-elephant-conflict-and-resulting-deaths-749169)
* Human–wildlife conflict data systems: [World Bank GWP HWC database exchange](https://www.worldbank.org/en/events/2024/08/06/knowledge-exchange-human-wildlife-conflict-hwc-database)

Design guidance:

* Esri: [Strategies and best practices for dashboards on smartphones](https://www.esri.com/arcgis-blog/products/arcgis-online/decision-support/strategies-best-practices-for-using-dashboards-on-your-smartphone),
  [Design impactful ArcGIS Dashboards (slides)](https://www.esri.com/content/dam/esrisites/en-us/media/powerpoints/design-impactful-arcgis-dashboards-with-best-practices.pdf),
  [Designing effective dashboards (slides)](https://www.esri.com/content/dam/esrisites/en-us/events/conferences/2022/midwest-uc/arcgis-dashboards-designing-effective-dashboards.pdf)
* Progressive disclosure: [Nielsen Norman Group](https://www.nngroup.com/videos/progressive-disclosure/),
  [The Decision Lab](https://thedecisionlab.com/reference-guide/design/progressive-disclosure),
  [Dashboard design principles (BI textbook)](https://ljkelly3141.github.io/bi-book/15.4-dashboard-design-principles.html)
* Streamlit: [Hiding the Deploy button / toolbarMode](https://discuss.streamlit.io/t/how-to-hide-or-remove-the-deploy-button-that-appears-at-the-top-right-corner-of-the-streamlit-app/55325)
  (the `st.navigation` signature, including `position="top"`, was checked
  against the installed 1.61.1 package)
