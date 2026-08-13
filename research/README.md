# Conflict prediction: strategy and results

The app ships a **rule-based tier system**, not a trained model. Tiers come
from fixed thresholds, and the continuous score only orders beats within a
tier. That was a deliberate choice — a rule means the same thing in April as
in October — but it leaves predictive signal on the table.

This directory is the experiment that measures how much. It is not wired into
the app. Nothing here changes what a manager currently sees.

```bash
pip install -r requirements-dev.txt
python -m research.evaluate path/to/sightings.csv
```

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
recall@20% [0.420, 0.648]. Against the best simple rule the model gains
**+0.069 PR-AUC, 95% CI [+0.008, +0.140]**, ahead in 98.9% of resamples.

Real but modest. The interval nearly touches zero, on 64 test events.

`recall@20%` is the operational metric: a range officer cannot visit every
village, so what counts is how much conflict falls in the fraction they can
cover. **Ranking villages by the model puts 53% of next month's conflict in the
top 20%** — against 45% for the last-quarter rule and 14% for no ranking at all.

Ranked for July, the top 15 of 260 villages carried a **60% hit rate against 7%
for the rest**.

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
- **The forest interface is a proxy.** There is no landcover layer, so
  `villages_5km` and `nearest_village_km` stand in for it — sparse settlement
  implies a forest matrix. It is the top feature, which is precisely why real
  landcover should replace it.
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

In descending order of expected value per unit of effort:

1. **Landcover.** Forest cover fraction, edge density, and distance to forest
   boundary within 2 km of each village. The settlement-density proxy is the
   single most important feature; replacing it with the real thing is the
   biggest available gain.
2. **More months.** Seven is enough to detect signal, not enough to trust the
   seasonal term. Two full years would allow proper rolling-origin validation.
3. **Crop calendar by village.** Sowing and harvest dates for the dominant
   crop. Damage risk is a function of what is standing in the field.
4. **Tusker identity.** If bulls are individually identified — many are, in
   practice — a repeat-offender feature would likely dominate. A handful of
   animals usually drive most raiding.
5. **Record the flags consistently.** 555 blank rows is the largest single
   quality loss in the dataset, and it costs more than any modelling choice.
6. **Water and terrain.** Distance to perennial water, slope, ridge lines.

## Recommendation

Do **not** replace the tier system. It is legible, stable, and a manager can
explain a posting decision from it.

Add the village forecast alongside it, as a **ranked watch list** — deciles,
not probabilities — with the tier still shown. It earns its place on the
recall@20% result and it transfers across divisions, which is the harder test.

Before that ships it needs: recalibration on a rolling window, a monthly
backtest that re-scores last month's list against what actually happened, and
a plain statement in the UI that the ranking reflects where *reporting* and
conflict have concentrated, not where elephants certainly are.
