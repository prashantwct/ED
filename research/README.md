# Conflict prediction: strategy and results

The app ships a **rule-based tier system**, not a trained model. Tiers come
from fixed thresholds, and the continuous score only orders beats within a
tier. That was a deliberate choice — a rule means the same thing in April as
in October — but it leaves predictive signal on the table.

This directory is the experiment that measures how much. It is not wired into
the app. Nothing here changes what a manager currently sees.

```bash
pip install -r requirements-dev.txt
python -m research.evaluate path/to/sightings.csv            # history features
python -m research.evaluate_landcover path/to/sightings.csv  # + ESA WorldCover
```

The land-cover run downloads about 450 MB of raster on first use and
caches it in `.data_cache/` (gitignored). Village statistics are cached
too, so only the first run is slow.

## Two models, because there are two questions

**Report triage.** A sighting has been filed — does it involve damage? Scored
at the moment a report arrives, to decide whether a response team goes out.

**Village forecast.** Will conflict occur within 2 km of this village next
month? Scored from history only, before the month begins. This is the one that
supports patrol placement and early warning.

The distinction matters more than it looks. Triage scores an event that has
already happened; only the forecast is a prediction.

## What the data supports

From a 1,761-report export, Jan–Jul 2026, Shahdol–Anuppur landscape.

**The three field hypotheses hold, and they are the top features.** Permutation
importance on the held-out months, for the village forecast:

| Feature | Importance | Hypothesis |
|---|---|---|
| `villages_5km` | +0.077 | settlement–forest interface |
| `mean_group_hist` | +0.054 | small parties raid, herds don't |
| `lone_male_share_hist` | +0.046 | lone tuskers |
| `conf_prev_quarter` | +0.046 | repeat conflict |

Raw damage rates, before any modelling, against a 17.1% landscape baseline:

| | n | damage rate |
|---|---|---|
| Lone male (total = 1, male = 1) | 482 | **28.4%** |
| All-male party | 767 | **30.1%** |
| Calves present | 295 | **3.1%** |
| Group of 5 or more | 375 | **2.4%** |

The bull/herd split is the strongest single signal in the dataset — roughly a
tenfold difference in damage rate between an all-male party and a breeding
herd. It is not currently used anywhere in the app's scoring.

## Results

Validation is **by time, never by random split**. Rows near each other in space
and time are not independent; a shuffled split lets the model see the
neighbours of what it is tested on and reports a number the field will not
reproduce.

### Village forecast — train Feb–May, test Jun–Jul

| Model | ROC-AUC | PR-AUC | recall@20% |
|---|---|---|---|
| Base rate | 0.500 | 0.123 | 0.141 |
| Rule: ever had conflict | 0.620 | 0.160 | 0.297 |
| Rule: conflicts last quarter | 0.656 | 0.336 | 0.453 |
| **Gradient boosting** | **0.732** | **0.403** | **0.531** |

95% CI on the held-out months: ROC-AUC [0.652, 0.810], PR-AUC [0.292, 0.531],
recall@20% [0.420, 0.648]. On this split the model gains +0.069 PR-AUC over the
best simple rule.

**That gain does not survive a better protocol.** See the rolling-origin table
below: pooled over every month, the rule wins. The +0.069 was a property of
which two months were held out, which is exactly the hazard a single split
carries on seven months of data.

`recall@20%` is the operational metric: a range officer cannot visit every
village, so what counts is how much conflict falls in the fraction they can
cover. **Ranking villages by the model puts 53% of next month's conflict in the
top 20%** — against 45% for the last-quarter rule and 14% for no ranking at all.

Ranked for July, the top 15 of 260 villages carried a **60% hit rate against 7%
for the rest**.

### Rolling origin — the number to quote

The table above holds out Jun–Jul. On seven months that is one arbitrary
window, and it happens to be the *low* season for crop damage. Training on
every month before the test month and stepping forward pools 1,300 rows and
179 events, and gives a soberer figure:

| Model | ROC-AUC | PR-AUC | recall@20% |
|---|---|---|---|
| Base rate | 0.500 | 0.138 | 0.200 |
| Gradient boosting, history features | 0.650 | 0.243 | 0.352 |
| **Rule: conflicts last quarter** | 0.613 | **0.290** | 0.385 |
| **Rule: conflicts all time** | 0.620 | 0.281 | **0.419** |
| + interface block (5 land-cover) | 0.643 | 0.226 | 0.318 |
| + all land cover (24) | 0.613 | 0.214 | 0.346 |
| + land cover + cropping (29) | 0.601 | 0.205 | 0.341 |

Two corrections to earlier versions of this file. **0.650 is the honest
headline, not 0.732** — the single split was the favourable window. And **the
simple rule beats the model** on the metrics that decide a watch list: the
gradient booster is behind by 0.044 PR-AUC, 95% CI [−0.091, +0.005], ahead in
only 3.6% of resamples. It leads on ROC-AUC, which on a 13.8% base rate is
dominated by easy negatives and is the wrong metric for ranking.

### Transfer to a landscape never seen in training

Hold out an entire division:

| Held out | n | events | ROC-AUC | PR-AUC | base rate |
|---|---|---|---|---|---|
| Bandhavgarh TR | 570 | 83 | 0.721 | 0.367 | 0.146 |
| Anuppur | 534 | 87 | 0.666 | 0.311 | 0.163 |
| Satna | 132 | 13 | 0.729 | 0.225 | 0.098 |

Performance holds up on divisions the model never trained on, at 2–2.5× the
base rate. This is the result that matters for rolling it out beyond the
division it was fitted on.

### Report triage — and a caveat that eats most of it

| Model | ROC-AUC | PR-AUC | recall@20% |
|---|---|---|---|
| Base rate | 0.500 | 0.116 | 0.262 |
| Local conflict history only | 0.738 | 0.230 | 0.344 |
| Gradient boosting | 0.866 | 0.408 | 0.689 |
| … report artefacts removed | 0.818 | 0.317 | 0.525 |

The strongest single feature was `Sighting Type = Direct`, with a **negative**
coefficient: indirect reports are likelier to carry damage. That is not a risk
factor. Damage is *discovered* — you find the trampled field, not the elephant
standing in it. A model leaning on it has partly learned how reports get
generated.

Removing those features costs 0.09 PR-AUC. **The 0.317 row is the honest one.**

## Land cover and cropping

Acquired: **ESA WorldCover v200**, 10 m global land cover for 2021, CC-BY 4.0,
four 3° tiles (N21/N24 × E078/E081) from the public S3 bucket, mosaicked to the
study window — 17,889 × 15,210 pixels. Geolocation checks out: built-up
fraction correlates +0.54 with village-centroid density and tree cover −0.71
with built-up.

Landscape composition: 38.1% cropland, 33.8% tree cover, 23.0% grassland, 2.8%
water, 0.7% built-up.

Cropping *season* is not observed — WorldCover says where cropland is, not what
is standing in it, and no Sentinel-2 archive was reachable. The calendar in
`landcover.py` encodes the Madhya Pradesh cropping year as documented domain
knowledge, and is labelled as such.

### It refines the field hypothesis rather than confirming it

Comparing the 129 villages that saw conflict against the 116 that did not:

| Metric within 2 km | Conflict | Quiet | Ratio | p |
|---|---|---|---|---|
| Crop–forest interface | 0.0790 | 0.0494 | **1.60** | <0.0001 |
| Cropland fraction | 0.4095 | 0.2883 | 1.42 | <0.0001 |
| Built-up fraction | 0.0095 | 0.0070 | 1.36 | 0.0004 |
| **Tree cover** | 0.3500 | 0.4342 | **0.81** | 0.031 |
| Forest edge density | 0.0912 | 0.0847 | 1.08 | 0.10 |
| Distance to forest | 0.0537 | 0.0602 | 0.89 | 0.96 |

Share of villages ever in conflict, by tree cover within 2 km:

| Q1 low | Q2 | Q3 | Q4 | Q5 high |
|---|---|---|---|---|
| 53.1% | 61.2% | 57.1% | 55.1% | **36.7%** |

…and by crop–forest interface:

| Q1 low | Q2 | Q3 | Q4 | Q5 high |
|---|---|---|---|---|
| 30.6% | 42.9% | 42.9% | **77.6%** | 69.4% |

**"Villages surrounded by forest are more prone to conflict" is not what the
land cover says.** Villages deepest in forest are the *safest* — the top
tree-cover quintile has the lowest conflict rate of all. What carries risk is
cropland with forest beside it: the interface quintiles run 30.6% to 77.6%.
The mechanism is a field a bull can feed in with cover to retreat into, and it
is a sharper target than "forest" for deciding where fencing and crop-guarding
go.

The cropping calendar tracks observed crop damage directionally — Spearman
+0.67 over seven months, p=0.10, peaking in February (28.5% of reports) and
bottoming in May (4.9%) — but seven points cannot carry more than a direction.

### It does not improve the forecast

Every land-cover variant scores below history alone, on rolling origin and on
the single split alike. The interface block costs −0.018 PR-AUC, 95% CI
[−0.044, +0.007]. Not a significant loss, but no gain either, and adding all 24
is clearly worse.

Land cover alone, with no history at all, reaches PR-AUC 0.222 against a 0.123
base rate — so the signal is real. It is simply **redundant**: the village
density it was meant to replace already measures the same settlement–forest
gradient (−0.50 with tree cover, +0.54 with built-up), and for *recorded*
conflict it measures it better, because centroid density also tracks how many
people are present to file a report.

### Cold start stays unsolved

The hope was that land cover would flag villages with no conflict history —
exactly what history-based forecasting is blind to. It does not. On 798
village-months with no prior recorded conflict, history-only scores PR-AUC
0.137 and adding the interface block gives 0.129, against a 0.109 base rate.
**Neither model can predict a village's first conflict.** Where history exists,
both do markedly better (0.319 and 0.285 against a 0.183 base).

## Movement units

`herds.py` chains sightings into tracks the way radar tracks aircraft: a report
joins the best-matching open track if it is close enough in space given the
elapsed time, and if the group composition is compatible. Composition is what
stops single-linkage chaining the whole landscape into one blob — a party of
bulls does not become a breeding herd with calves overnight.

`python -m research.herd_map <csv>` draws the ranges over land cover.

**These are movement units, not identified animals.** No collars, no ear-notch
photographs, no DNA — only a time, a place and a rough count. Two bulls working
the same valley on alternating days are one track here, and there is a test
asserting exactly that so it cannot be quietly forgotten.

### What is stable, and what is not

The **unit count is an artefact of the thresholds** — 147 to 484 depending on
the speed and gap settings. Do not quote it as a population estimate.

What does not move, at any setting tried:

| Setting | Units | Top 10 share | Bull-type share | Deaths from bull-type |
|---|---|---|---|---|
| 10 km/day, 2-day gap | 484 | 43% | **94%** | 6 / 6 |
| 15 km/day, 4-day gap | 287 | 55% | **94%** | 6 / 6 |
| 20 km/day, 7-day gap | 167 | 72% | **94%** | 6 / 6 |
| 30 km/day, 7-day gap | 147 | 75% | **94%** | 6 / 6 |

**Lone bulls and bull parties account for 94% of conflict events and every one
of the six human deaths**, however the tracker is tuned. Family herds, at the
default setting, produced 10 conflict events across 94 units and no deaths.

Conflict is also concentrated: at the default setting the top 10 units of 287
carry 55% of all conflict events and the top 20 carry 75%. The exact figure
moves with the thresholds, but the shape does not.

### Does the movement look like an elephant?

For the 117 units seen three or more times: median 3.8 km/day, 90th percentile
8.3, maximum 25.2. Median longest single step 9.5 km. Median observed range
22 km², rising to 380 km² at the 90th percentile. Those are plausible figures
for Asian elephants in central India, which is a check on the tracker rather
than a finding.

Ranges are the area covered **within the observation window** — median span
8 days — not annual home ranges. Only 18 of 287 units cross a division
boundary, so most are one division's problem to manage.

## Known limits

- **Casualties are not modellable here.** Five deaths and four injuries in
  1,761 reports. Any model of fatality risk at this sample size would be
  fitting noise. The target throughout is *any damage* (crop, grain, house,
  injury, death), n = 301.
- **Probabilities are not calibrated.** The model under-predicts on held-out
  months — a predicted 20% came back as an observed 33%, because Jun–Jul
  carries more conflict than the Feb–May it trained on. **Use it as a ranking,
  not as a probability.** Fix with isotonic recalibration on a rolling window
  once there are more months.
- **Land cover is a 2021 snapshot** scored against 2026 conflict. Cropland
  boundaries move. It is also a single epoch, so it cannot express the thing
  that most likely matters — a field that was forest five years ago.
- **Seven months.** One season. Nothing here can speak to inter-annual
  variation, and `month_of_year` is fitted on a single pass through the crop
  calendar.
- **No true absences.** The data records where elephants *were seen*, not where
  they were. The village forecast sidesteps this by conditioning on villages
  the elephants came near, but a true occupancy model would need survey effort.
- **Null damage flags are read as zero**, following the app's convention. 555
  reports have all five flags blank. If blank means "not asked" rather than
  "no damage", the base rate is understated and every number here shifts.

## What would move the needle

Land cover was the previous top recommendation. It has now been acquired and
tested, and it did not pay off in forecast accuracy — so the list is reordered
around what the land-cover result actually taught: **the binding constraint is
sample size and reporting consistency, not feature richness.** Adding more
covariates to 179 events makes things worse, however good the covariate.

1. **More months.** Seven is enough to detect signal, not to fit it. Every
   negative result here is on 179 events, and the intervals show it. Two full
   years is the single change that would let any of the rest matter.
2. **Record the flags consistently.** 555 reports with all five damage flags
   blank is the largest quality loss in the dataset, and it costs more than any
   modelling choice.
3. **Tusker identity.** If bulls are individually identified — many are, in
   practice — a repeat-offender feature would likely dominate everything above.
   A handful of animals usually drive most raiding, and unlike land cover this
   is not redundant with anything already measured.
4. **Observed crop phenology** rather than an assumed calendar. Sentinel-2 NDVI
   per village per month would replace a 12-number constant with what was
   actually standing. The calendar tracks damage at rho +0.67; measuring it
   would settle whether the residual matters.
5. **Where patrols went.** Every rate here conflates elephants with observers,
   and effort data would separate them. This is what village-centroid density
   is quietly standing in for.
6. **Multi-date land cover.** A single epoch cannot express recent conversion
   of forest to field, which is the mechanism the interface result implies.

## Recommendation

**Do not adopt the machine-learning model.** Under rolling-origin validation it
is beaten by counting conflicts at each village over the last quarter — a rule
a range officer can compute by hand, that needs no scikit-learn, no retraining
and no monitoring. The model is ahead in 3.6% of resamples. Complexity has to
earn its place and this does not.

Adopt instead, in order of confidence:

1. **The counting rule as the watch list.** Rank villages by conflict in the
   last quarter, show the tier beside it, publish deciles rather than
   probabilities. It puts 38.5% of next month's conflict in the top 20% of
   villages against 20% for no ranking. The app's existing village tier is
   already close to this; the change is to make it a forward-looking list and
   backtest it monthly.
2. **The bull/herd distinction, everywhere.** 94% of conflict and every human
   death came from lone bulls and bull parties, and the finding holds at every
   tracker setting. Nothing else in this work is that robust, and the app's
   scoring currently ignores group composition entirely. A report of a lone
   bull near a village deserves a different response from a breeding herd with
   calves, and right now it gets the same one.
3. **The crop-forest interface as a siting layer.** Not a forecast input --
   it made the model worse -- but a static map of where cropland meets cover.
   Interface quintiles run 30.6% to 77.6% for conflict. That is where fencing
   and crop-guarding belong.
4. **Movement units for narrative, not for numbers.** Useful for asking "which
   animal is doing this and where does it go", and for briefing field staff.
   Never as a population estimate.

Revisit the model when there are two years of data. Every negative result here
rests on 179 events, and the intervals show it. The honest summary is that at
this sample size the simple thing wins, and the useful findings came from
looking at the data rather than from fitting it.
