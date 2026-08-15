"""The camera view can be toggled at runtime, without config.json losing its say.

Only the visibility decision is covered here — there is no OpenCV window and no
webcam involved, so the suite stays offline. What matters is the precedence: a
tray toggle beats the config file, until the config file itself changes.
"""

import json
import os
import time

import pytest

pytest.importorskip("cv2")
pytest.importorskip("mediapipe")
pytest.importorskip("pyautogui")

from puppet_functions.config_manager import ConfigManager  # noqa: E402
from puppet_overlay import Pipeline  # noqa: E402


def make_pipeline(tmp_path, **overrides):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(overrides), encoding="utf-8")
    config = ConfigManager(path=str(path))
    return Pipeline(config, overlay=None, screen_w=1920, screen_h=1080), path


_clock = [time.time()]


def rewrite(pipeline, path, **values):
    """Edit config.json and let the pipeline pick it up.

    Both clocks have to move: the file needs an mtime the config manager can
    see is new, and `_poll_config` only stats the file once a second.
    """
    path.write_text(json.dumps(values), encoding="utf-8")
    _clock[0] += 10
    os.utime(str(path), (_clock[0], _clock[0]))
    pipeline._poll_config(_clock[0])


def test_the_feed_is_off_by_default(tmp_path):
    pipeline, _ = make_pipeline(tmp_path)
    assert pipeline.feed_visible is False


def test_config_can_open_it_at_startup(tmp_path):
    pipeline, _ = make_pipeline(tmp_path, show_debug_feed=True)
    assert pipeline.feed_visible is True


def test_toggling_shows_then_hides_it(tmp_path):
    pipeline, _ = make_pipeline(tmp_path)
    assert pipeline.toggle_feed() is True
    assert pipeline.feed_visible is True
    assert pipeline.toggle_feed() is False
    assert pipeline.feed_visible is False


def test_toggling_off_beats_a_config_that_says_on(tmp_path):
    """Otherwise the window would be impossible to close while the file says true."""
    pipeline, _ = make_pipeline(tmp_path, show_debug_feed=True)
    pipeline.toggle_feed()
    assert pipeline.feed_visible is False


def test_an_unrelated_config_edit_leaves_the_toggle_alone(tmp_path):
    """Tuning thresholds with the camera view open must not close it."""
    pipeline, path = make_pipeline(tmp_path)
    pipeline.toggle_feed()
    rewrite(pipeline, path, scroll_gain=1234.0)

    assert pipeline.config.scroll_gain == 1234.0
    assert pipeline.feed_visible is True


def test_changing_show_debug_feed_takes_control_back(tmp_path):
    """A *changed* value wins, both ways. Rewriting the same value is not a
    change the config manager can see, so it deliberately doesn't count."""
    pipeline, path = make_pipeline(tmp_path, show_debug_feed=True)
    pipeline.toggle_feed()                      # hidden, against the file
    rewrite(pipeline, path, show_debug_feed=False)
    assert pipeline.feed_visible is False       # and it stays hidden

    pipeline.toggle_feed()                      # shown
    pipeline.toggle_feed()                      # and hidden again, explicitly
    rewrite(pipeline, path, show_debug_feed=True)
    assert pipeline.feed_visible is True


def test_toggling_does_not_disturb_pause(tmp_path):
    pipeline, _ = make_pipeline(tmp_path)
    pipeline.toggle_feed()
    assert pipeline.paused is False
    pipeline.toggle_pause()
    pipeline.toggle_feed()
    assert pipeline.paused is True
