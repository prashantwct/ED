"""Train the next-move forecast on historic exports and report how it does.

The app trains the forecast on whatever register is loaded, every time it
changes. This does the same from the command line, for exports kept
outside the app -- several years in several files, say -- and prints the
track record for each horizon:

    python -m tools.train_movement old_2025.csv register_2026.csv
    python -m tools.train_movement *.csv --horizon 1 3 --json summary.json

Files are read through the app's own loader and merged; a report that
appears in more than one file (same ID, or the same date, place and count
where there is no ID) is kept once. ``--json`` writes the summary: counts,
rates, areas and the parts chosen -- no report, place or person.

``--state`` saves the trainings (JSON: scores and settings, no reports) and
``--previous`` reads them back next time, so only the months whose reports
changed since are trained again, as the app does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from core import movement as mv
from core.data_loader import load_and_validate_csv

_SAME_REPORT = ["Date", "Latitude", "Longitude", "Total Count"]


def merge(paths: Sequence[Path]) -> pd.DataFrame:
    """Every file through the app's loader, then each report once."""
    frames = []
    for path in paths:
        with open(path, "rb") as handle:
            frame, warnings = load_and_validate_csv(handle)
        for warning in warnings:
            print(f"{path.name}: {warning}", file=sys.stderr)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    if "ID" in merged.columns and merged["ID"].notna().all():
        merged = merged.drop_duplicates(subset=["ID"], keep="last")
    else:
        merged = merged.drop_duplicates(
            subset=[c for c in _SAME_REPORT if c in merged.columns], keep="last")
    return merged.sort_values("Date", kind="stable").reset_index(drop=True)


def summary(training: mv.Training) -> dict:
    record = training.record
    return {
        "horizon_days": training.horizon_days,
        "reports": training.reports,
        "first_report": str(training.first_report.date()) if training.reports else None,
        "last_report": str(training.last_report.date()) if training.reports else None,
        "moves": training.moves,
        "months_scored": record.months,
        "months_retrained": training.retrained_months,
        "forecasts_tested": record.tested,
        "likely_zone_hit_rate": round(record.forecast_rate, 3) if record.tested else None,
        "likely_zone_km2": round(record.mean_area_km2, 1) if record.tested else None,
        "same_size_circle_hit_rate": round(record.stay_rate, 3) if record.tested else None,
        "circle_km2_for_same_hits": (round(record.circle_area_km2, 1)
                                     if record.circle_area_km2 == record.circle_area_km2
                                     else None),
        "beats_staying_put": bool(record.beats_staying_put),
        "parts_used": record.config.parts(),
        "zone_masses": [round(m, 3) for m in record.masses],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="sightings exports (CSV)")
    parser.add_argument("--horizon", nargs="+", type=int, default=[1, 2, 3],
                        help="days ahead to forecast (default: 1 2 3)")
    parser.add_argument("--json", type=Path, help="write the summary here")
    parser.add_argument("--state", type=Path,
                        help="save the trainings here, to retrain only what changes next time")
    parser.add_argument("--previous", type=Path, help="trainings saved earlier with --state")
    args = parser.parse_args(argv)

    register = merge(args.files)
    print(f"{len(register):,} reports from {len(args.files)} file(s), "
          f"{register['Date'].min():%d %b %Y} to {register['Date'].max():%d %b %Y}")
    earlier: List[mv.Training] = []
    if args.previous and args.previous.exists():
        earlier = [mv.Training.from_dict(item)
                   for item in json.loads(args.previous.read_text())]

    trainings, rows = [], []
    for horizon in args.horizon:
        training = mv.train(register, horizon, previous=earlier)
        trainings.append(training)
        row = summary(training)
        rows.append(row)
        if not row["forecasts_tested"]:
            print(f"{horizon} day(s): too little history to test the forecast")
            continue
        print(f"{horizon} day(s): likely zone {row['likely_zone_hit_rate']:.0%} at "
              f"{row['likely_zone_km2']:.0f} km2; same-size circle "
              f"{row['same_size_circle_hit_rate']:.0%}; "
              f"{'beats' if row['beats_staying_put'] else 'does not beat'} staying put; "
              f"uses {', '.join(row['parts_used']) or 'distance alone'}; "
              f"{row['months_retrained']} of {len(training.months)} month(s) trained")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    if args.state:
        args.state.write_text(json.dumps([t.to_dict() for t in trainings]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
