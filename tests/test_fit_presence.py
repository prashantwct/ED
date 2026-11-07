"""The command-line fit: files for R, and a summary with no places in it."""

import json

import pandas as pd

from presence_fixtures import simulate, village_points
from tools import fit_presence


def test_it_writes_inputs_surfaces_corridors_and_a_summary(tmp_path):
    villages = village_points()
    register = simulate(days=420, seed=3, villages=villages)
    register = register.assign(Division="D", Range="R", Beat="B")
    register["Date"] = register["Date"].dt.strftime("%d-%m-%Y")
    path = tmp_path / "register.csv"
    register.to_csv(path, index=False)
    centroids = tmp_path / "villages.csv"
    villages.to_csv(centroids, index=False)
    out = tmp_path / "out"
    assert fit_presence.main([str(path), "--out", str(out), "--villages", str(centroids)]) == 0
    table = pd.read_csv(out / "model_inputs.csv")
    assert {"cell", "fortnight", "records", "village_km_is_effort"} <= set(table.columns)
    assert len(pd.read_csv(out / "surfaces.csv")) == table["cell"].nunique()
    assert json.loads((out / "corridors.geojson").read_text())["type"] == "FeatureCollection"
    text = (out / "summary.json").read_text()
    row = json.loads(text)
    assert row["fitted"] and row["fortnights_tested"] > 0
    assert "Latitude" not in text and "latitude" not in text
