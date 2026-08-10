"""Loads, validates and hot-reloads config.json.

Defaults live here, not in the JSON file, so a missing or partial config still
starts the app. Values are range-checked on load: a typo that used to sail
straight through into pyautogui now gets a warning and the default instead.

The file is re-read while running whenever its timestamp changes, so you can
tune sensitivity with the app open and feel the difference immediately.
"""

import json
import os

# key -> (type, minimum, maximum). None means unbounded.
_SCHEMA = {
    # --- camera -----------------------------------------------------------
    "camera_index":          (int,   0,     8),
    "camera_width":          (int,   160,   4096),
    "camera_height":         (int,   120,   2160),
    "detection_confidence":  (float, 0.1,   1.0),
    "tracking_confidence":   (float, 0.1,   1.0),

    # --- cursor -----------------------------------------------------------
    # Fraction of the camera frame mapped to the full screen. Shrink it to
    # cover the screen with less arm movement; grow it for finer control.
    "active_region":         (list,  None,  None),
    # Lower = steadier cursor when still, but slower to get going.
    "one_euro_min_cutoff":   (float, 0.1,   30.0),
    # Higher = less lag when moving fast, at the cost of a little jitter.
    "one_euro_beta":         (float, 0.0,   1.0),
    # Where the cursor is taken from: "virtual" (a palm-anchored point that
    # doesn't move when you pinch), "knuckle", or "fingertip".
    "cursor_source":         (str,   None,  None),
    "cursor_offset_palms":   (float, 0.0,   3.0),

    # --- gesture thresholds (all in palm-lengths) -------------------------
    "finger_ext_on":         (float, 0.0,   1.5),
    "finger_ext_off":        (float, -0.5,  1.5),
    "thumb_out_on":          (float, 0.5,   3.0),
    "thumb_out_off":         (float, 0.5,   3.0),
    # Raise pinch_on if clicks are hard to trigger; lower it if they misfire.
    "pinch_on":              (float, 0.05,  1.5),
    "pinch_off":             (float, 0.05,  2.0),
    # Fingertip-to-wrist distance separating a pinch from a closed fist.
    "pinch_reach_on":        (float, 0.5,   2.5),
    "pinch_reach_off":       (float, 0.5,   2.5),
    "scale_smoothing":       (float, 0.01,  1.0),
    "gesture_dwell_ms":      (int,   0,     1000),

    # --- click / drag -----------------------------------------------------
    "drag_slop_palm":        (float, 0.01,  2.0),
    # Tracking drops a frame now and then; don't tear down a click over it.
    "hand_lost_grace_ms":    (int,   0,     2000),
    # Settle time between moving the cursor and pressing the button.
    "click_press_delay_ms":  (int,   0,     300),
    # Pinch distance at which the cursor starts slowing down. Must be larger
    # than pinch_on; the gap between them is the approach ramp.
    "pinch_damp_start":      (float, 0.05,  2.5),
    "click_settle_ms":       (int,   0,     1000),

    # --- scroll / zoom ----------------------------------------------------
    "scroll_deadzone":       (float, 0.0,   2.0),
    "scroll_gain":           (float, 1.0,   10000.0),
    "zoom_deadzone":         (float, 0.0,   2.0),
    "zoom_gain":             (float, 1.0,   10000.0),

    # --- interface --------------------------------------------------------
    "show_debug_feed":       (bool,  None,  None),
    "overlay_dot_radius":    (float, 1.0,   50.0),
    "show_overlay":          (bool,  None,  None),
    # Live pinch/reach numbers on the HUD, for checking thresholds against
    # your own hand rather than against assumed proportions.
    "show_diagnostics":      (bool,  None,  None),
}

DEFAULTS = {
    "camera_index": 0,
    "camera_width": 640,
    "camera_height": 480,
    "detection_confidence": 0.6,
    "tracking_confidence": 0.6,

    "active_region": [0.15, 0.15, 0.85, 0.85],
    "one_euro_min_cutoff": 1.2,
    "one_euro_beta": 0.02,
    "cursor_source": "virtual",
    "cursor_offset_palms": 1.0,

    "finger_ext_on": 0.25,
    "finger_ext_off": 0.10,
    "thumb_out_on": 1.25,
    "thumb_out_off": 1.10,
    "pinch_on": 0.45,
    "pinch_off": 0.65,
    "pinch_reach_on": 1.15,
    "pinch_reach_off": 1.02,
    "scale_smoothing": 0.3,
    "gesture_dwell_ms": 80,

    "drag_slop_palm": 0.12,
    "hand_lost_grace_ms": 250,
    "click_press_delay_ms": 25,
    "pinch_damp_start": 0.80,
    "click_settle_ms": 150,

    "scroll_deadzone": 0.15,
    "scroll_gain": 900.0,
    "zoom_deadzone": 0.25,
    "zoom_gain": 90.0,

    "show_debug_feed": False,
    "overlay_dot_radius": 9.0,
    "show_overlay": True,
    "show_diagnostics": False,
}

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config.json",
)


def _validate(key, value):
    """Return `value` if it fits the schema, else the default (with a warning)."""
    spec = _SCHEMA.get(key)
    if spec is None:
        return value  # unknown keys pass through untouched
    expected, lo, hi = spec

    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    # bool is a subclass of int, so an explicit check keeps `true` out of ints.
    if not isinstance(value, expected) or (expected is not bool and isinstance(value, bool)):
        print(f"[config] {key}: expected {expected.__name__}, got {value!r} — using default")
        return DEFAULTS[key]

    if key == "active_region":
        if len(value) != 4 or not all(isinstance(v, (int, float)) for v in value):
            print(f"[config] active_region must be [x0, y0, x1, y1] — using default")
            return DEFAULTS[key]
        return [min(1.0, max(0.0, float(v))) for v in value]

    if key == "cursor_source" and value not in ("virtual", "knuckle", "fingertip"):
        print(f"[config] cursor_source={value!r} unknown — using default")
        return DEFAULTS[key]

    if lo is not None and value < lo:
        print(f"[config] {key}={value} below minimum {lo} — clamped")
        return lo
    if hi is not None and value > hi:
        print(f"[config] {key}={value} above maximum {hi} — clamped")
        return hi
    return value


class ConfigManager:
    def __init__(self, path=None):
        self._path = path or _CONFIG_PATH
        self._data = dict(DEFAULTS)
        self._mtime = 0.0
        self.reload()

    def reload(self):
        """Re-read the file, falling back to defaults for anything missing."""
        data = dict(DEFAULTS)
        try:
            with open(self._path, encoding="utf-8") as f:
                loaded = json.load(f)
            for key, value in loaded.items():
                data[key] = _validate(key, value)
            self._mtime = os.path.getmtime(self._path)
        except FileNotFoundError:
            pass  # first run — defaults are fine
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[config] could not read {self._path}: {exc} — using defaults")
        self._data = data

    def maybe_reload(self):
        """Reload if the file changed on disk. Returns True if it did."""
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            return False
        if mtime == self._mtime:
            return False
        self.reload()
        return True

    def save(self):
        """Write the current values back out (used by future tuning UI)."""
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        self._mtime = os.path.getmtime(self._path)

    def as_dict(self):
        return dict(self._data)

    def __getattr__(self, key):
        # Only reached for names not found normally, so `_data` is safe to use
        # — but guard anyway in case of unpickling or early access.
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self.__dict__["_data"][key]
        except KeyError:
            raise AttributeError(key) from None
