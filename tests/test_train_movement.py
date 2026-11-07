"""The command-line trainer: historic files merged, each report once."""

import json

import pandas as pd

from movement_fixtures import simulate
from tools import train_movement


def _write(frame, path):
    out = frame.copy()
    out["Date"] = out["Date"].dt.strftime("%d-%m-%Y")
    out.to_csv(path, index=False)
    return path


def test_overlapping_exports_are_merged_with_each_report_once(tmp_path):
    register = simulate(groups=4, days=160, seed=3, home=0.5)
    register["ID"] = range(len(register))
    half = len(register) // 2
    first = _write(register.iloc[: half + 50], tmp_path / "old.csv")
    second = _write(register.iloc[half:], tmp_path / "new.csv")
    merged = train_movement.merge([first, second])
    assert len(merged) == len(register)


def test_the_summary_reports_rates_and_parts_but_no_places(tmp_path):
    register = simulate(groups=4, days=160, seed=3, home=0.5)
    path = _write(register, tmp_path / "register.csv")
    out = tmp_path / "summary.json"
    assert train_movement.main([str(path), "--horizon", "1", "--json", str(out)]) == 0
    rows = json.loads(out.read_text())
    assert rows[0]["forecasts_tested"] > 0
    assert 0 <= rows[0]["likely_zone_hit_rate"] <= 1
    text = out.read_text()
    assert "Latitude" not in text and "Longitude" not in text


def test_a_saved_state_retrains_only_new_months(tmp_path):
    register = simulate(groups=4, days=200, seed=3, home=0.5)
    cut = register["Date"].max() - pd.Timedelta(days=10)
    early = _write(register[register["Date"] <= cut], tmp_path / "early.csv")
    full = _write(register, tmp_path / "full.csv")
    state, out = tmp_path / "state.json", tmp_path / "summary.json"
    train_movement.main([str(early), "--horizon", "1", "--state", str(state)])
    train_movement.main([str(full), "--horizon", "1", "--previous", str(state),
                         "--json", str(out)])
    row = json.loads(out.read_text())[0]
    assert 0 < row["months_retrained"] <= 2


def test_a_saved_training_reads_back_the_same(tmp_path):
    from core import movement as mv

    register = simulate(groups=4, days=200, seed=3, home=0.5)
    trained = mv.train(register, 1)
    back = mv.Training.from_dict(json.loads(json.dumps(trained.to_dict())))
    assert back.record.forecast_hits == trained.record.forecast_hits
    assert back.record.config == trained.record.config
    assert back.record.masses == trained.record.masses
    again = mv.train(register, 1, previous=back)
    assert again.retrained_months == 0
