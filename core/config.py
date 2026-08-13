"""Tunable parameters for the conservation intelligence pipeline.

Everything here encodes a domain judgement rather than a fact. Values
were set against a 1,761-row Anuppur/Bandhavgarh export; a different
landscape may warrant different ones. Change them here, not in the
modules that read them.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Village centroids shipped with the repo. Constant reference data, so it
# loads by default; the sidebar uploader overrides it.
DEFAULT_CENTROIDS_PATH = DATA_DIR / "centroids.csv"

# Field deployments need a log that survives the terminal closing.
# Overridable so a read-only container can point it elsewhere.
LOG_PATH = Path(os.getenv("ED_LOG_PATH", REPO_ROOT / "app.log"))


# --- Severity ---------------------------------------------------------------
# Points per conflict signal. Human harm dominates by design: a ranking
# where accumulated crop damage outranks a death is not usable.
SEVERITY_WEIGHTS = {
    "presence": 0.5,
    "grain": 1.5,
    "crop": 2.5,
    "house": 5.0,
    "injury": 25.0,
}
DEATH_WEIGHT_PER_PERSON = 100.0

# Band edges aligned to the weights, so each band means one incident type.
SEVERITY_BAND_EDGES = [0.0, 1.0, 5.0, 25.0, 100.0, float("inf")]
SEVERITY_BAND_LABELS = [
    "Presence only",
    "Crop / grain loss",
    "Property damage",
    "Human injury",
    "Human fatality",
]

# Dusk-to-dawn movement window.
NIGHT_HOUR_START = 18
NIGHT_HOUR_END = 6


# --- Beat priority ----------------------------------------------------------
# Escalation compares the most recent N days against the N before.
DEFAULT_RECENT_DAYS = 90
MIN_WINDOW_DAYS = 14
MIN_EVENTS_FOR_TREND = 5
ESCALATION_RATIO = 1.5

# How recent a casualty must be for Critical. Fixed, not the adjustable
# escalation window: a tier that moves when a slider moves is not a tier.
CRITICAL_CASUALTY_WINDOW_DAYS = 90

# Reports needed before a beat's own rate outweighs the landscape rate.
CONFIDENCE_THRESHOLDS = {"High": 30, "Medium": 10}

# Multiple of the landscape conflict rate that promotes a beat.
RATE_MULTIPLE_FOR_HIGH = 1.5
INJURIES_FOR_CRITICAL = 3
HOUSE_EVENTS_FOR_HIGH = 5
EVENTS_FOR_ESCALATION_HIGH = 10

# Priority score weights. Only orders beats within a tier.
SCORE_WEIGHTS = {
    "casualty": 0.35,
    "burden": 0.20,
    "intensity": 0.20,
    "exposure": 0.10,
    "trend": 0.05,
    "composition": 0.10,
}
CASUALTY_POINTS_PER_DEATH = 60.0
CASUALTY_POINTS_PER_INJURY = 25.0
HISTORICAL_CASUALTY_DISCOUNT = 0.5

# Empirical-Bayes prior strength, in reports, when it cannot be fitted.
FALLBACK_PRIOR_STRENGTH = 10.0
PRIOR_STRENGTH_BOUNDS = (1.0, 200.0)


# --- Recommended actions ----------------------------------------------------
NIGHT_SHARE_FOR_PATROL_SHIFT = 60.0
VILLAGE_SHARE_FOR_EARLY_WARNING = 50.0

# Group composition. Bulls range alone or in small parties and raid;
# breeding herds with calves keep away from settlements. In the export
# these tunables were fitted against, an all-male party carried a 30.1%
# damage rate against 3.1% where calves were present, and bull-type
# groups accounted for every human death. Only bulls carry tusks in this
# species, so "Male Count" is the tusker count.
SOLITARY_MAX_GROUP = 3
# Above this share of a beat's conflict, the response is a bull problem
# and should be written as one.
BULL_SHARE_FOR_ALERT = 60.0
# Below this many conflict events the share is too noisy to act on.
MIN_CONFLICTS_FOR_COMPOSITION = 3


# --- Temporal ---------------------------------------------------------------
# Share of conflict a recommended patrol window should cover.
DEFAULT_COVERAGE_TARGET = 0.60
# A month must beat an even spread by this much to count as seasonal.
SEASONAL_LIFT = 1.25
MAX_PEAK_MONTHS = 4


# --- Hotspots ---------------------------------------------------------------
# DBSCAN chains: past the scale at which sightings are continuously
# distributed, separate concentrations merge through the density between
# them. On the reference export, 2.5 km put 94% of points into 7 clusters,
# the largest 17.7 km across. 1.0 km keeps radii at 1-3 km.
DEFAULT_EPS_KM = 1.0
DEFAULT_MIN_SAMPLES = 15
MAX_SENSIBLE_RADIUS_KM = 5.0

CONFLICT_SHARE_FOR_HIGH = 0.25
EVENTS_FOR_WATCH = 5


# --- Spatial ----------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON_EQUATOR = 111.32

# A centroid is a point; a village is not. Its built extent plus adjacent
# fields commonly runs several hundred metres from the centre.
NEAR_VILLAGE_THRESHOLD_KM = 2.0
DEFAULT_VILLAGE_RADIUS_KM = 3.0

VALID_LAT_RANGE = (-90.0, 90.0)
VALID_LON_RANGE = (-180.0, 180.0)


# --- Map --------------------------------------------------------------------
# Radii are metres but clamped to a pixel range: this landscape fits at
# roughly 1 km per pixel, where a 60 m radius is invisible.
MIN_RADIUS_M = 60
MAX_RADIUS_M = 500
RADIUS_MIN_PIXELS = 3
RADIUS_MAX_PIXELS = 14


# --- Ingestion --------------------------------------------------------------
# Day-first is the field-export convention here; order only breaks ties.
DATE_FORMATS = [
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d %b %Y",
    "%d %B %Y",
    "%m/%d/%Y",
]
TIME_FORMATS = ["%H:%M:%S", "%H:%M"]

# cp1252 before latin-1: it is what Excel on Windows emits.
FALLBACK_ENCODINGS = ["utf-8", "cp1252", "ISO-8859-1"]
