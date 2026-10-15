import pandas as pd

from core import map_select


def _frame():
    # Two sightings at A (a lone bull raiding crop, a herd passing), one
    # injury by a lone bull at B, 20 km east.
    return pd.DataFrame({
        "Latitude": [23.0, 23.0, 23.0],
        "Longitude": [81.0, 81.001, 81.2],
        "Beat": ["A", "A", "B"],
        "Crop Damage": [1, 0, 0],
        "Injury": [0, 0, 1],
        "Male Count": [1, 2, 1],
        "Female Count": [0, 3, 0],
        "Calf Count": [0, 1, 0],
        "Total Count": [1, 6, 1],
    })


def _hotspots():
    return pd.DataFrame({
        "Hotspot": ["H1", "H2"], "Centre Latitude": [23.0, 23.0],
        "Centre Longitude": [81.0, 81.2], "Radius (km)": [1.0, 1.0],
    })


def _villages():
    return pd.DataFrame({
        "Village": ["Near A", "Near B", "Quiet"], "Tier": ["High", "Critical", "Routine"],
        "Latitude": [23.01, 23.0, 23.0], "Longitude": [81.0, 81.21, 81.1],
    })


ALL = dict(events=map_select.EVENT_CLASSES, groups=map_select.GROUP_CLASSES,
           tiers=["Critical", "High", "Watch", "Routine"], village_km=3.0)


def test_nothing_narrowed_keeps_every_layer():
    shown, hot, vil = map_select.select(_frame(), _hotspots(), _villages(), focus=None, **ALL)
    assert len(shown) == 3 and len(hot) == 2 and len(vil) == 3


def test_event_and_animal_filters_combine():
    kw = {**ALL, "events": ["Injury"], "groups": ["Lone bull"]}
    shown, hot, vil = map_select.select(_frame(), _hotspots(), _villages(), focus=None, **kw)
    assert shown["Beat"].tolist() == ["B"]
    # Context layers follow the sightings: only B's hotspot and village.
    assert hot["Hotspot"].tolist() == ["H2"]
    assert vil["Village"].tolist() == ["Near B"]


def test_crop_by_lone_bull_drops_the_herd():
    kw = {**ALL, "events": ["Crop"], "groups": ["Lone bull"]}
    shown, _, _ = map_select.select(_frame(), _hotspots(), _villages(), focus=None, **kw)
    assert len(shown) == 1 and shown.iloc[0]["Crop Damage"] == 1


def test_picked_hotspot_shows_only_its_data():
    frame = _frame()
    focus = map_select.focus_for("hotspot", "H1", frame, _hotspots(), _villages(), 3.0)
    shown, hot, vil = map_select.select(frame, _hotspots(), _villages(), focus=focus, **ALL)
    assert shown["Beat"].tolist() == ["A", "A"]
    assert hot["Hotspot"].tolist() == ["H1"]
    assert vil["Village"].tolist() == ["Near A"]
    assert focus["bounds"] is not None


def test_picked_beat_and_village():
    frame = _frame()
    focus = map_select.focus_for("beat", "B", frame, _hotspots(), _villages(), 3.0)
    shown, _, _ = map_select.select(frame, _hotspots(), _villages(), focus=focus, **ALL)
    assert shown["Beat"].tolist() == ["B"]

    focus = map_select.focus_for("village", "Near A", frame, _hotspots(), _villages(), 3.0)
    shown, _, vil = map_select.select(frame, _hotspots(), _villages(), focus=focus, **ALL)
    assert shown["Beat"].tolist() == ["A", "A"]
    assert vil["Village"].tolist() == ["Near A"]
    assert focus["ring"][2] == 3.0


def test_village_tiers_are_a_layer():
    kw = {**ALL, "tiers": ["Critical"]}
    _, _, vil = map_select.select(_frame(), _hotspots(), _villages(), focus=None, **kw)
    assert vil["Village"].tolist() == ["Near B"]


def test_unknown_pick_is_no_focus():
    assert map_select.focus_for("beat", "Z", _frame(), _hotspots(), _villages(), 3.0) is None
    assert map_select.focus_for("hotspot", "H9", _frame(), _hotspots(), _villages(), 3.0) is None
