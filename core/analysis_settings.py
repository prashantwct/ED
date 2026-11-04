"""The analysis settings a viewer can leave alone, and saying when they were not.

The sidebar lets an analyst tune the clustering distance, the comparison
window and the rest. A manager who changes one by accident sees different
tiers, hotspots and villages than a colleague looking at the same export,
with nothing on the page to say why. So each setting has a standard value,
and anything off it is named on the page and in the brief.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping

from core.config import DEFAULT_CUBE_CELL_KM, DEFAULT_CUBE_STEP
from core.hotspots import DEFAULT_EPS_KM, DEFAULT_MIN_SAMPLES, DEFAULT_VILLAGE_RADIUS_KM
from core.intelligence import DEFAULT_RECENT_DAYS
from core.spacetime import METRICS


@dataclass(frozen=True)
class Setting:
    """One tunable: its widget key, a short name for badges, its standard value."""

    key: str
    name: str
    default: object
    unit: str = ""

    def phrase(self, value: object) -> str:
        """``Recent window 60 days (standard 90)``."""
        return f"{self.name} {_show(value)}{self.unit} (standard {_show(self.default)})"

    def is_standard(self, value: object) -> bool:
        if isinstance(self.default, bool) or isinstance(value, bool):
            return bool(value) == bool(self.default)
        if isinstance(self.default, (int, float)) and isinstance(value, (int, float)):
            return abs(float(value) - float(self.default)) < 1e-9
        return value == self.default


def _show(value: object) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


RECENT_DAYS = Setting("cfg_recent_days", "Recent window", DEFAULT_RECENT_DAYS, " days")
VILLAGE_RADIUS = Setting("cfg_village_radius", "Village radius", DEFAULT_VILLAGE_RADIUS_KM, " km")
EPS_KM = Setting("cfg_eps_km", "Hotspot merge distance", DEFAULT_EPS_KM, " km")
MIN_SAMPLES = Setting("cfg_min_samples", "Smallest hotspot", DEFAULT_MIN_SAMPLES, " reports")
CUBE_STEP = Setting("cfg_cube_step", "Time step", DEFAULT_CUBE_STEP)
CUBE_CELL_KM = Setting("cfg_cube_cell_km", "Hexagon size", DEFAULT_CUBE_CELL_KM, " km")
CUBE_METRIC = Setting("cfg_cube_metric", "Counting", METRICS[0])
CUBE_FDR = Setting("cfg_cube_fdr", "Strict test", True)

# In sidebar order, so the badge reads in the order the controls do.
SETTINGS = (RECENT_DAYS, VILLAGE_RADIUS, EPS_KM, MIN_SAMPLES,
            CUBE_STEP, CUBE_CELL_KM, CUBE_METRIC, CUBE_FDR)
KEYS = tuple(s.key for s in SETTINGS)


def non_standard(values: Mapping[str, object]) -> List[str]:
    """One phrase per setting that is off its standard value.

    ``values`` maps widget keys to their current values -- session state,
    in the app. A key that is absent has never been touched, so it is at
    its standard value.
    """
    return [s.phrase(values[s.key]) for s in SETTINGS
            if s.key in values and not s.is_standard(values[s.key])]
