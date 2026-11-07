"""Fit the presence model on historic exports, and write what R needs.

The app fits it on whatever register is loaded. This does the same from
the command line, for exports kept outside the app, and writes files
to take further elsewhere -- inlabru, R-INLA, mgcv, Circuitscape:

    python -m tools.fit_presence register_2025.csv register_2026.csv --out presence/

* ``model_inputs.csv``: one row per cell per fortnight -- thinned
  records, covariates (and which are reporting effort), recent-past terms.
* ``surfaces.csv``: next fortnight's forecast and the effort-corrected
  use for each season, per cell.
* ``corridors.geojson``: least-cost paths between core areas.
* ``summary.json``: the track record and what was fitted -- counts,
  areas and scores, no report, place or person.

``--effort`` adds points where people who report are (app-open pings,
patrol tracks): only their coordinates are read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from core import corridors as cr
from core import presence as pr
from tools.train_movement import merge

CENTROIDS = Path(__file__).resolve().parent.parent / "data" / "centroids.csv"


def summary(result: pr.Result) -> dict:
    record = result.record
    inputs = result.inputs
    tested = record.tested > 0
    return {
        "reports": inputs.reports,
        "records_after_thinning": inputs.records,
        "cell_km": inputs.lattice.cell_km,
        "cells": inputs.lattice.size,
        "fortnights": inputs.blocks.count,
        "covariates": inputs.names,
        "reporting_effort": inputs.effort,
        "fitted": result.fitted,
        "fortnights_tested": record.tested,
        "half_records_km2": round(record.half_km2(), 1) if tested else None,
        "half_records_km2_simple_maps": (
            {name: round(record.half_km2(name), 1) for name in pr.BASELINES}
            if tested else None),
        "bits_per_record_over_better_simple_map": (round(record.gain_bits(), 3)
                                                   if tested else None),
        "beats_simple_maps": bool(record.beats_baselines),
        "parts_used": record.spec.parts() if tested else [],
        "effects": result.effects()[["Term", "Kind", "Ratio"]].round(3).to_dict("records"),
    }


def _corridors(result: pr.Result) -> dict:
    lattice = result.inputs.lattice
    features = []
    for season in result.seasons():
        net = cr.network(lattice, result.use(season), flow=False)
        for link in net.links:
            features.append({"type": "Feature", "properties": {
                "season": season, "from_core": int(link.a), "to_core": int(link.b),
                "length_km": round(link.length_km, 2), "cost": round(link.cost, 1)},
                "geometry": {"type": "LineString", "coordinates": [
                    [round(float(lon), 5), round(float(lat), 5)]
                    for lat, lon in zip(lattice.lat[link.path], lattice.lon[link.path])]}})
    return {"type": "FeatureCollection", "features": features}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="sightings exports (CSV)")
    parser.add_argument("--out", type=Path, default=Path("presence"),
                        help="folder for the outputs (default: presence/)")
    parser.add_argument("--as-of", help="last day to read (default: the last report)")
    parser.add_argument("--effort", type=Path, help="CSV of points where reporters are")
    parser.add_argument("--villages", type=Path, default=CENTROIDS,
                        help="village centroids (default: the ones shipped with the app)")
    parser.add_argument("--villages-are-habitat", action="store_true",
                        help="do not count distance from villages as reporting effort")
    args = parser.parse_args(argv)

    register = merge(args.files)
    as_of = pd.Timestamp(args.as_of) if args.as_of else register["Date"].max()
    villages = pd.read_csv(args.villages) if args.villages and args.villages.exists() else None
    layers = {"Uploaded effort": pr.read_effort(args.effort)} if args.effort else {}
    inputs = pr.prepare(register, as_of, villages=villages, effort_layers=layers,
                        village_effort=not args.villages_are_habitat)
    result = pr.run(inputs)

    args.out.mkdir(parents=True, exist_ok=True)
    result.model_table().to_csv(args.out / "model_inputs.csv", index=False)
    row = summary(result)
    (args.out / "summary.json").write_text(json.dumps(row, indent=2))
    print(f"{inputs.reports:,} reports, {inputs.records:,} records after thinning, "
          f"{inputs.lattice.size:,} cells of {inputs.lattice.cell_km:g} km, "
          f"{inputs.blocks.count} fortnights")
    if not result.fitted:
        print("Too little history to fit the model; wrote the inputs only.")
        return 0
    lattice = inputs.lattice
    surfaces = pd.DataFrame({"cell": range(lattice.size), "latitude": lattice.lat.round(5),
                             "longitude": lattice.lon.round(5),
                             "next_fortnight": result.next_fortnight()})
    for season in result.seasons():
        surfaces[f"use {season}"] = result.use(season)
    surfaces.to_csv(args.out / "surfaces.csv", index=False)
    (args.out / "corridors.geojson").write_text(json.dumps(_corridors(result)))
    if row["fortnights_tested"]:
        rival = result.record.best_baseline()
        print(f"Half of each tested fortnight's records fell in {row['half_records_km2']:,.0f} "
              f"km2 of the forecast ({rival.lower()}: "
              f"{row['half_records_km2_simple_maps'][rival]:,.0f} km2); "
              f"{'beats' if row['beats_simple_maps'] else 'does not beat'} the simple maps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
