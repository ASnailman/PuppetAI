"""ConfigManager: bad values must be caught here, not deep inside pyautogui."""

import json

from puppet_functions.config_manager import DEFAULTS, ConfigManager


def write(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_missing_file_falls_back_to_defaults(tmp_path):
    cfg = ConfigManager(path=str(tmp_path / "nope.json"))
    assert cfg.scroll_gain == DEFAULTS["scroll_gain"]


def test_partial_file_keeps_defaults_for_the_rest(tmp_path):
    cfg = ConfigManager(path=write(tmp_path, {"scroll_gain": 1234.0}))
    assert cfg.scroll_gain == 1234.0
    assert cfg.pinch_on == DEFAULTS["pinch_on"]


def test_malformed_json_does_not_crash(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{ not json", encoding="utf-8")
    assert ConfigManager(path=str(path)).pinch_on == DEFAULTS["pinch_on"]


def test_wrong_type_is_replaced_by_the_default(tmp_path):
    cfg = ConfigManager(path=write(tmp_path, {"scroll_gain": "fast"}))
    assert cfg.scroll_gain == DEFAULTS["scroll_gain"]


def test_out_of_range_values_are_clamped(tmp_path):
    cfg = ConfigManager(path=write(tmp_path, {"pinch_on": 99.0, "gesture_dwell_ms": -5}))
    assert cfg.pinch_on == 1.5
    assert cfg.gesture_dwell_ms == 0


def test_ints_are_accepted_where_floats_are_expected(tmp_path):
    cfg = ConfigManager(path=write(tmp_path, {"scroll_gain": 500}))
    assert cfg.scroll_gain == 500.0


def test_booleans_are_not_accepted_as_numbers(tmp_path):
    cfg = ConfigManager(path=write(tmp_path, {"scroll_gain": True}))
    assert cfg.scroll_gain == DEFAULTS["scroll_gain"]


def test_bad_active_region_falls_back(tmp_path):
    cfg = ConfigManager(path=write(tmp_path, {"active_region": [0.1, 0.2]}))
    assert cfg.active_region == DEFAULTS["active_region"]


def test_hot_reload_picks_up_a_change(tmp_path):
    path = write(tmp_path, {"scroll_gain": 100.0})
    cfg = ConfigManager(path=path)
    assert cfg.scroll_gain == 100.0
    assert cfg.maybe_reload() is False

    # Force a distinct mtime; some filesystems have coarse timestamps.
    import os
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"scroll_gain": 200.0}, f)
    os.utime(path, (0, 0))

    assert cfg.maybe_reload() is True
    assert cfg.scroll_gain == 200.0


def test_save_round_trips(tmp_path):
    path = str(tmp_path / "config.json")
    cfg = ConfigManager(path=path)
    cfg.save()
    assert ConfigManager(path=path).as_dict() == cfg.as_dict()


def test_unknown_key_raises_attribute_error(tmp_path):
    cfg = ConfigManager(path=str(tmp_path / "nope.json"))
    try:
        cfg.definitely_not_a_setting
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")
