# Movement forecast: a methodology review

*Next moves* forecasts where a group reported in the last three days may be
next (`core/movement.py`). This review read the method end to end, measured
each suspected weakness on synthetic movement where the true identity of
every group is known (`tests/movement_fixtures.py` plus scenarios built for
the review), and fixed what the measurements showed to matter. Each finding
gives the evidence, the fix and the result. What remains open is listed at the
end.

## How the forecast worked before

1. A greedy tracker chained reports into *groups*: each report, in filing
   order, extended the cheapest compatible open track within reach.
2. Step lengths (distance / √days), turns relative to the last move (one
   von Mises per gap), a recency-weighted usage map, the same season of
   earlier years, and the group's own range formed the parts of a forecast.
3. A backtest forecast each month from the months before it. Every move
   between consecutive reports was scored once, with the gap to the next
   report known. Parts were chosen by log score and the zones were
   calibrated (split conformal) on all past forecasts.

## Findings

### 1. The backtest scored a forecast the page never makes (serious)

The page forecasts each group from *today*: a group seen two days ago, for
"the next 3 days", is forecast 3 to 5 days after its last report. The
backtest instead scored each move once, from the last report, and told the
forecast exactly when the next report would come. Its calibration, the
"catches about half" on the page, was therefore calibrated on a different
forecast.

**Evidence.** Forecasts rebuilt exactly as the page makes them and scored
against each group's *true* next report:

| | Page claimed | Page forecasts achieved |
|---|---|---|
| Home range, next 3 days | 52% | **69%** (zones too wide) |
| Steady travel, next 3 days | 51% | **65%** (from 85% to 24% by when the group was next seen) |
| Steady travel, next 24 h | 51% | **45%** |

**Fix.** The backtest forecasts each morning as the page does: every group
last seen in the past three days, from that morning, over the horizon. A move
is the outcome of every morning the page showed the group (`_cases`). Zones
are calibrated on those forecasts.

### 2. Distance was assumed to grow as √days (serious for travelling groups)

√days is a random walk. A group holding its heading goes nearly four times as
far in four days, not twice. Multi-day zones were too small for travelling
groups, which caused the 24% above.

**Fix.** The register's own scaling is learned: distance ∝ days^α, with α fitted
to the median distance at each gap (`scaling`), between ½ and 1. On the
synthetic worlds α comes out at 0.58 for a random walk, 0.72 at moderate
persistence and 1.0 for steady travel.

### 3. "Next 3 days" covered only the third day

The forecast for a three-day horizon was where the group would be *on* day
3. A report the next morning, the most common case, fell outside it.

**Fix.** A forecast spans every day of its horizon: the day-by-day movement
kernels, averaged (`Situation.span`, `_movement`).

### 4. Calibration on all past forecasts kept zones too wide

The first months' forecasts, made from a few dozen moves, needed far more of
the forecast to catch the group: 32% in the first month, before calibration
was possible. Calibrating on every past forecast carried that forward, and
later months caught 49–67% against a 50% target.

**Fix.** Zones are calibrated on the most recent 300 forecasts. Months from the
third on now catch 50–54%.

**Result for findings 1–4**, the same independent check as above:

| | Claimed | Achieved, before | Achieved, after |
|---|---|---|---|
| Home range, next 3 days | 52% | 69% | **58%** |
| Steady travel, next 3 days | 48% | 65% | **52%** |
| Home range, next 24 h | 50% | 50% | 55% |
| Steady travel, next 24 h | 48% | 45% | 54% |

The remaining few points come from one simplification: the backtest learns
step lengths once a month, while the page learns them each day.

### 5. Unrecorded composition split tracks (serious on real registers)

Field reports often give only a total. The tracker treated "no breakdown" as
a class of its own, so a family herd reported once with its calves and once
as a bare 12 became two groups.

**Evidence.** With half of reports lacking a breakdown, the tracker linked
68% of true moves (84% with full composition) and split nine herds into 122
units.

**Fix.** Composition is compared only where both reports recorded it. The
count must still agree. Linked moves are back to 83–85%, and units drop to
about 40.

### 6. Reports were matched one at a time (moderate where herds share a landscape)

The greedy tracker let the first report filed on a day claim the nearest
track, even when a later report that day was the better match. Where herds
of the same make-up shared overlapping ranges, 31% of linked moves joined two
different herds. Zones grew to 34 km² against 20 km² with correct identities,
because steps between two herds are long.

**Fix.** Tracking now goes day by day:
- Each day's reports are grouped into sightings: compatible reports within
  3 km are several people reporting one herd.
- The day's sightings are matched to open tracks all at once, the cheapest
  set of links (the Hungarian method).
- Similar counts are preferred over merely compatible ones.

**Result.** The share of linked moves joining a herd to itself, before → after:

| Overlapping herds | Greedy | Whole-day matching |
|---|---|---|
| Identical make-up | 0.69 | **0.81** |
| Different sizes | 0.77 | **0.89** |
| Half without breakdown | 0.73 | **0.81** |
| Different sizes, with duplicate reports* | 0.82 | **0.88** |

\* Greedy here already includes the fix for finding 5.

### 7. Two herds within 12 km on the same day were merged

A same-day radius of 12 km, meant for two observers on opposite sides of one
herd, joined any two compatible groups that close on the same day.

**Fix.** The same-day reach is the 3 km of a sighting plus what a herd can
cover in the hours between the reports: the register's `Hour` column where
filled in, else the time in `Date`. Without times, two herds 6 km apart on one
morning stay two herds.

### 8. One mean turn averaged mixed behaviour away (serious in mixed populations)

The direction model was a single von Mises distribution of turns. In a
population where some groups travel and others shuttle between a refuge and
fields, the two cancel: the learned concentration was 0.02, which means no
direction at all.

**Fix.** The turn density is a smoothed density of the register's turns
(about 20°), so it can have both peaks. A group's own recent turns are mixed
in: five of its own count as much as the register's. A group that has been
shuttling is expected to turn back; one that has been travelling, to keep
going.

**Result** on a population half travelling, half shuttling: the likely zone
still catches 51%, but it shrank from 41 to 26 km². A same-size circle now
catches 33%, and the gain over distance alone rose from 0.34 to 0.85 bits per
forecast. On pure steady travel it costs a little (0.79 → 0.73 bits); on
shuttling it gains (1.89 → 2.01).

### 9. The likeliest spot was nearly always "near its last report"

Moves spread over rings around the start, so the single most probable 1 km
cell is almost always the group's own. **Fix.** The likeliest spot is the
3 × 3 km block holding the most probability.

### 10. Every score depended on the tracker and on a group being seen again

Hit rates are conditional on the group being reported again and on the
tracker linking that report. Neither is what a field team experiences.

**Fix, added alongside.** A landscape score is computed for each morning's
whole map: of every report in the days after, including groups nobody had
seen, how many fell in some group's likely zone, against same-size circles
round each group's last report. The page states it.

## The forecast now, on synthetic movement

Likely zone against a same-size circle round the last report (per group); then
the whole map against all reports (landscape). Next 24 hours:

| Movement | Likely zone | Circle | Area | Map | Circles | Parts chosen |
|---|---|---|---|---|---|---|
| Wandering at random | 50% | 50% | 43 km² | 47% | 46% | none: no claim |
| Steady travel | **50%** | 36% | 41 km² | **46%** | 33% | direction |
| Keeping to a home range | **52%** | 47% | 35 km² | **48%** | 43% | places |
| Refuge to field and back | **52%** | 29% | 10 km² | **51%** | 28% | places, direction, own range |
| Half travel, half shuttle | **51%** | 33% | 26 km² | **48%** | 31% | places, direction |
| Mixed, two years | 51% | 50% | 46 km² | 47% | 46% | places (same season) |

Next 3 days, steady travel: 48% against 34% per group, 31% against 19% for the
map.

These figures are not comparable with the earlier README table. That table
scored moves with their gap known, which this review found to be a different,
easier forecast.

## Added since: grid memory with seasons

Every move is remembered as from one 2 km grid (on the national lattice, so a
grid keeps its name whatever register is loaded) to the next (`Routes`). To
forecast a group, the moves that began in its grid, or within about 1.5 km of
its position, are weighted:
- recent ones more (half the weight after two years),
- moves whose length in days matches the forecast's more,
- with the season on, moves made within about 30 days of today's date in any
  year more.

A move counts only from the day its second report came in, so the backtest
never remembers a move it could not have known about.

The first version multiplied the memory into the forecast, as the other parts
are. That lost to the forecast without it: remembered destinations already say
how far groups go, so multiplying counted distance twice. The memory is now
mixed in as a forecast of its own, with weight n / (n + k) for n remembered
moves. The backtest chooses k (2 or 8), the season on or off, or no memory at
all, as it chooses every other part.

| World | Best without memory | With memory | Chosen |
|---|---|---|---|
| Loops walked one way round, at random times | +1.05 bits | **+1.30 bits** | memory, trusted from two moves |
| Out in October, back in June, same corridors | +0.35 | +0.36 | memory, seasonal |
| Wandering at random | 0 | -0.02 | no memory |
| Home ranges | +0.17 | +0.10 | no memory |

(Bits per forecast over distance alone, next 24 hours.) The memory earns its
place where direction depends on the place, not on the date. Where it depends
on the date, the usage map of the same weeks of earlier years already says most
of it. The page shows each group's grid, its three likeliest next grids with
their chance, and what the memory holds for its grid, whether or not the
forecast uses it.

## Still open

- **Fitted weights, and habitat.** The parts are switched on or off from a
  small grid (places at 0, ½ or 1, direction and own range on or off). A
  step-selection function would fit their weights continuously by
  conditional logit and could add habitat: forest, distance to fields,
  rivers. That is integrated step-selection analysis (Avgar et al. 2016).
- **Time of day.** Herds rest in forest by day and raid fields by night. A
  day sighting in forest says the next night's report is likely at the
  forest edge. Report hours are used for tracking, not yet for forecasting.
- **Daily step model in the backtest.** The backtest learns step lengths once a
  month and the page each day. This is the source of the few points by which
  it still over-covers.
- **The usage map carries reporting effort.** That is right for forecasting
  the next *report*, which is effort-biased too. It is not a map of where
  elephants are; the effort-corrected surface on *Range & corridors* is that.
- **Validation on known identities.** All of the above is measured on
  simulation, which shares the method's assumptions. Collared or individually
  identified herds would test the tracker as nothing synthetic can. Even a
  season of one collared herd would settle how often real groups are merged
  or split.
