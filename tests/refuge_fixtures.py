"""A synthetic landscape with known day refuges and known decoys.

A 10 m land-cover raster in UTM 44N (Esri's projection and class codes)
over farmland with two forest patches and a village, and a sighting
register in which the truth is planted:

* Refuge A -- a large forest block, used by day on ten days; the herd
  raids fields 2-3 km east on the nights it is there, and the damage is
  entered the next morning, as it is in the field.
* Refuge B -- a 9 ha tree patch beside the village, used on four days.
* Feeding ground -- crop fields seen mostly at night, with a few day
  sightings on three days. Must not pass as a refuge.
* Passage -- a herd seen eight times on one afternoon. Must be dropped.
* Open-field day use -- day sightings in open crop on five days. May be
  found, but must be flagged as having no cover.
* Morning damage entries -- crop damage in open fields, written down the
  morning after on five days. Under the entry's hour it looks like day
  use; it is a night raid and must not become a refuge.
* Pug marks -- indirect signs logged by day in open crop on six days.
  Stale traces of night visits: never day rest.
* Signs-only refuge -- pug marks and dung inside the forest block on four
  days, nobody sighting the herd, raids south on the same nights. A
  refuge, flagged for confirmation by day.
* Unknown refuge -- raids on five nights in fields 4 km east of the big
  forest block, with no daytime sighting anywhere within reach. A raid
  area to search, with the forest block as the place to look.
* Quiet refuge -- forest used by day on five days with no raid within
  reach. A refuge, but not a management one: it must be left out.
"""

import numpy as np
import pandas as pd

ORIGIN_E, ORIGIN_N = 500_000.0, 2_640_000.0   # UTM 44N, near 23.87 N 81.0 E
SIZE = 900                                     # pixels of 10 m: a 9 km square
CRS = "EPSG:32644"


def _to_latlon(east, north):
    from rasterio.warp import transform

    lon, lat = transform(CRS, "EPSG:4326", [east], [north])
    return lat[0], lon[0]


# Feature centres in metres from the raster's south-west corner.
REFUGE_A = (2000.0, 6500.0)       # inside the big block
REFUGE_B = (6500.0, 2500.0)       # inside the small patch
FEEDING = (5000.0, 6500.0)        # open crop
PASSAGE = (4000.0, 1000.0)
OPEN_DAY = (7500.0, 7500.0)
VILLAGE = (7200.0, 2200.0)
DAMAGE_FIELD = (5500.0, 4000.0)   # open crop, damage entered in the morning
UNKNOWN_RAIDS = (8500.0, 8500.0)  # raided, no day sighting within 5 km
PUGMARKS = (3000.0, 2500.0)       # fresh pug marks logged by day in open crop
FOREST_SIGNS = (3800.0, 4600.0)   # pug marks and dung inside the forest block, no sighting
QUIET = (1000.0, 4500.0)          # inside the big block; used after the raiding season


def landcover_tif(path) -> str:
    import rasterio
    from rasterio.transform import from_origin

    codes = np.full((SIZE, SIZE), 5, dtype=np.uint8)          # crops
    codes[100:500, 50:450] = 2                                 # big forest block (16 km2)
    rows = slice(SIZE - 265, SIZE - 235)                       # small patch: 300 m square
    codes[rows, 635:665] = 2
    codes[SIZE - 240:SIZE - 200, 700:740] = 7                  # village
    codes[600:620, :] = 1                                      # a river
    with rasterio.open(path, "w", driver="GTiff", height=SIZE, width=SIZE, count=1,
                       dtype="uint8", crs=CRS,
                       transform=from_origin(ORIGIN_E, ORIGIN_N + SIZE * 10, 10, 10)) as ds:
        ds.write(codes, 1)
    return str(path)


def point(xy, rng, jitter_m=150.0):
    e = ORIGIN_E + xy[0] + rng.normal(0, jitter_m)
    n = ORIGIN_N + xy[1] + rng.normal(0, jitter_m)
    return _to_latlon(e, n)


def register(seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    def add(xy, date, hour, jitter=150.0, total=3, male=0, female=2, calf=1, crop=0, house=0,
            kind=None):
        lat, lon = point(xy, rng, jitter)
        # Damage is found after the fact; everything else here was seen.
        kind = kind or ("Indirect" if crop or house else "Direct")
        rows.append({"Date": pd.Timestamp(date), "Hour": hour, "Latitude": lat, "Longitude": lon,
                     "Sighting Type": kind,
                     "Division": "Test", "Range": "Test", "Beat": "Test",
                     "Total Count": total, "Male Count": male, "Female Count": female,
                     "Calf Count": calf, "Crop Damage": crop, "House Damage": house,
                     "Injury": 0, "Death": 0})

    base = pd.Timestamp("2026-01-01")
    # Refuge A: a family herd, ten days, raiding east each evening.
    for k in range(10):
        day = base + pd.Timedelta(days=3 * k)
        for hour in (10, 12, 14):
            add(REFUGE_A, day, hour)
        # The night of `day` raid, found and entered at 08:00 next morning.
        add((REFUGE_A[0] + 2500, REFUGE_A[1] + rng.normal(0, 300)),
            day + pd.Timedelta(days=1), 8, crop=1)
    # Refuge B: a lone bull, four days, raiding the village.
    for k in range(4):
        day = base + pd.Timedelta(days=5 * k + 1)
        for hour in (11, 13):
            add(REFUGE_B, day, hour, jitter=60, total=1, male=1, female=0, calf=0)
        add(VILLAGE, day + pd.Timedelta(days=1), 10, jitter=80, total=1, male=1, female=0,
            calf=0, house=1)
    # Feeding ground: night-heavy, a little day use on three days.
    for k in range(12):
        add(FEEDING, base + pd.Timedelta(days=2 * k), 22, crop=1)
        add(FEEDING, base + pd.Timedelta(days=2 * k), 2, crop=1)
    for k in range(3):
        for hour in (10, 13):
            add(FEEDING, base + pd.Timedelta(days=4 * k), hour)
    # Passage: one afternoon.
    for hour in (9, 10, 10, 11, 12, 13, 14, 15):
        add(PASSAGE, base + pd.Timedelta(days=20), hour, jitter=100)
    # Day use in open crop, raiding fields 1.5 km south each evening.
    for k in range(5):
        for hour in (10, 14):
            add(OPEN_DAY, base + pd.Timedelta(days=4 * k + 2), hour, jitter=80)
        add((OPEN_DAY[0], OPEN_DAY[1] - 1500), base + pd.Timedelta(days=4 * k + 2), 20,
            jitter=100, crop=1)
    # Damage in a field, entered the morning after: night raids, not day use.
    for k in range(5):
        for hour in (10, 12):
            add(DAMAGE_FIELD, base + pd.Timedelta(days=6 * k + 4), hour, jitter=80, crop=1)
    # Pug marks and dung logged at 10:00 and 13:00 on six days: stale signs
    # of night visits, which would pass for day rest if taken at their hour.
    for k in range(6):
        for hour in (10, 13):
            add(PUGMARKS, base + pd.Timedelta(days=3 * k + 1), hour, jitter=80,
                kind="Indirect")
    # A refuge known only from signs inside the forest, raiding fields 1.6 km
    # south on the same nights; damage entered the next morning.
    for k in range(4):
        day = base + pd.Timedelta(days=70 + 3 * k)
        for hour in (8, 11):
            add(FOREST_SIGNS, day, hour, jitter=80, kind="Indirect")
        add((FOREST_SIGNS[0], FOREST_SIGNS[1] - 1600), day, 8, jitter=120, crop=1)
    # Raids from a refuge nobody reports, entered the next morning.
    for k in range(5):
        add(UNKNOWN_RAIDS, base + pd.Timedelta(days=61 + k), 8, jitter=200, crop=1)
    # A quiet refuge: forest, day use on five days, never a raid near it.
    for k in range(5):
        for hour in (10, 13):
            add(QUIET, base + pd.Timedelta(days=5 * k + 40), hour, jitter=80)
    # Dawn and dusk movement near the refuges: must not be counted as rest.
    for k in range(10):
        add(REFUGE_A, base + pd.Timedelta(days=3 * k), 6)
        add(REFUGE_A, base + pd.Timedelta(days=3 * k), 17)
    return pd.DataFrame(rows)


def villages() -> pd.DataFrame:
    lat, lon = _to_latlon(ORIGIN_E + VILLAGE[0], ORIGIN_N + VILLAGE[1])
    lat2, lon2 = _to_latlon(ORIGIN_E + REFUGE_A[0] + 2600, ORIGIN_N + REFUGE_A[1])
    return pd.DataFrame({"Village": ["Patchgaon", "Eastpur"],
                         "Latitude": [lat, lat2], "Longitude": [lon, lon2]})


def latlon(xy):
    return _to_latlon(ORIGIN_E + xy[0], ORIGIN_N + xy[1])
