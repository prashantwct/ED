"""Saying when the analysis was run on something other than its standard settings."""

from core import analysis_settings as cfg


def test_untouched_settings_are_standard():
    """A key that is absent has never been moved from its default."""
    assert cfg.non_standard({}) == []


def test_settings_at_their_defaults_are_standard():
    values = {s.key: s.default for s in cfg.SETTINGS}
    assert cfg.non_standard(values) == []


def test_a_changed_setting_is_named_with_its_standard_value():
    said = cfg.non_standard({cfg.EPS_KM.key: 1.5})
    assert said == ["Hotspot merge distance 1.5 km (standard 1)"]


def test_a_slider_float_equal_to_the_default_is_standard():
    """Slider values come back as floats; 1.0 and 1 are the same setting."""
    assert cfg.non_standard({cfg.EPS_KM.key: float(cfg.EPS_KM.default)}) == []
    assert cfg.non_standard({cfg.RECENT_DAYS.key: float(cfg.RECENT_DAYS.default)}) == []


def test_switches_read_as_on_and_off():
    said = cfg.non_standard({cfg.CUBE_FDR.key: False})
    assert said == ["Strict test off (standard on)"]


def test_changes_are_listed_in_sidebar_order():
    said = cfg.non_standard({cfg.CUBE_CELL_KM.key: 4.0, cfg.RECENT_DAYS.key: 60})
    assert [p.split(" ")[0] for p in said] == ["Recent", "Hexagon"]


def test_unrelated_session_keys_are_ignored():
    assert cfg.non_standard({"flt_divisions": ["Anuppur"], "view": "x"}) == []
