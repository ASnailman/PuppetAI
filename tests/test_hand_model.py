"""HandModel: the scale- and rotation-invariance that replaces rigid posture.

These are the tests that guard the core fix. If they break, the app goes back
to only working at one distance, with the hand held perfectly upright.
"""

import math

from conftest import make_hand, pose_click, pose_move, pose_open_palm, pose_scroll
from puppet_functions.hand_model import HandModel


def _read(config, landmarks):
    """One-shot read with a fresh model, so no hysteresis state carries over."""
    return HandModel(config).update(landmarks)


def test_extended_and_curled_fingers_are_distinguished(config):
    f = _read(config, make_hand(index=True, middle=False, ring=False, pinky=False))
    assert f.index_extended
    assert not f.middle_extended
    assert not f.ring_extended
    assert not f.pinky_extended


def test_open_palm_and_fist(config):
    assert _read(config, pose_open_palm()).open_palm
    assert _read(config, make_hand(index=False)).fist


def test_thumb_out_vs_tucked(config):
    assert _read(config, make_hand(index=True, middle=True, thumb='out')).thumb_out
    assert not _read(config, make_hand(index=True, middle=True, thumb='in')).thumb_out


def test_pinch_detected_only_when_pinching(config):
    assert _read(config, pose_click()).index_pinch
    assert not _read(config, pose_move()).index_pinch


def test_middle_pinch_is_separate_from_index_pinch(config):
    f = _read(config, make_hand(index=True, middle=True, pinch='middle'))
    assert f.middle_pinch
    assert not f.index_pinch


def test_same_pose_at_different_distances_reads_identically(config):
    """A hand twice as far from the camera must produce the same booleans.

    This is what the old absolute thresholds (0.12, 0.10, 0.02) could not do.
    """
    near = _read(config, pose_click(scale=0.30))
    far = _read(config, pose_click(scale=0.08))
    for field in ('index_extended', 'middle_extended', 'thumb_out', 'index_pinch'):
        assert getattr(near, field) == getattr(far, field), field
    # Normalised distances agree too, not just the thresholded booleans.
    assert abs(near.index_pinch_distance - far.index_pinch_distance) < 1e-6


def test_rotated_hand_reads_identically(config):
    """Tilt your wrist and the app must not forget which fingers are up."""
    upright = _read(config, pose_scroll())
    for degrees in (-60, -30, 30, 60, 120):
        tilted = _read(config, pose_scroll(rotation=math.radians(degrees)))
        assert tilted.index_extended == upright.index_extended
        assert tilted.middle_extended == upright.middle_extended
        assert tilted.thumb_out == upright.thumb_out


def _pinch_pulls_the_index_finger(landmarks, amount=0.35):
    """Reproduce the reported problem: a real pinch doesn't just move the
    thumb, it drags the index finger down to meet it."""
    for idx in (6, 7, 8):        # index PIP, DIP, TIP
        landmarks[idx].x += 0.15 * amount * 0.5
        landmarks[idx].y += 0.15 * amount
    landmarks[4].x, landmarks[4].y = landmarks[8].x + 0.02, landmarks[8].y + 0.02
    return landmarks


def test_the_cursor_point_does_not_move_when_the_index_finger_does(config):
    """The reported bug: reaching the thumb across pulls the index finger with
    it, and the cursor — being the fingertip — went along for the ride."""
    pointing = _read(config, pose_move())
    pinching = _read(config, _pinch_pulls_the_index_finger(pose_move()))

    cursor_drift = math.hypot(pinching.pointer_x - pointing.pointer_x,
                              pinching.pointer_y - pointing.pointer_y)
    tip_drift = math.hypot(pinching.index_tip_x - pointing.index_tip_x,
                           pinching.index_tip_y - pointing.index_tip_y)

    assert tip_drift > 0.04          # the finger really did move
    assert cursor_drift < 1e-9       # ...and the cursor did not
    assert pinching.index_pinch      # ...and it still registers as a pinch


def test_curling_the_fingers_does_not_move_the_cursor(config):
    """Same property, stated generally: the cursor is rigid to the palm."""
    open_hand = _read(config, make_hand(index=True, middle=True))
    curled = _read(config, make_hand(index=True, middle=False))
    assert abs(open_hand.pointer_x - curled.pointer_x) < 1e-9
    assert abs(open_hand.pointer_y - curled.pointer_y) < 1e-9


def test_the_cursor_point_still_follows_the_hand(config):
    """...but it must not be so rigid that the hand can't steer it."""
    left = _read(config, pose_move(center=(0.3, 0.5)))
    right = _read(config, pose_move(center=(0.7, 0.5)))
    assert right.pointer_x - left.pointer_x > 0.35


def test_the_cursor_point_sits_near_the_fingertip(config):
    """It should feel like pointing, so the virtual point lands roughly where
    the fingertip does rather than back at the wrist."""
    f = _read(config, pose_move())
    offset = math.hypot(f.pointer_x - f.index_tip_x, f.pointer_y - f.index_tip_y)
    assert offset < 0.6 * f.scale


def test_cursor_source_is_configurable(config, tmp_path):
    import json
    from puppet_functions.config_manager import ConfigManager

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"cursor_source": "fingertip"}), encoding="utf-8")
    tip_cfg = ConfigManager(path=str(path))

    f = HandModel(tip_cfg).update(pose_move())
    assert abs(f.pointer_x - f.index_tip_x) < 1e-9


def test_pinch_strength_rises_as_fingers_close(config):
    apart = _read(config, pose_move()).pinch_strength
    together = _read(config, pose_click()).pinch_strength
    assert together > apart
    assert 0.0 <= apart <= 1.0 and 0.0 <= together <= 1.0


def test_degenerate_landmarks_do_not_divide_by_zero(config):
    collapsed = make_hand(scale=0.0)
    f = _read(config, collapsed)   # every landmark at the same point
    assert f.scale > 0


def test_hysteresis_holds_through_a_marginal_frame(config, model):
    """A finger hovering at the boundary must not flicker frame to frame."""
    model.update(pose_click())
    assert model.update(pose_click()).index_pinch
    # Nudge the thumb into the hysteresis gap: still counts as pinched.
    marginal = pose_click()
    marginal[4].x += 0.15 * 0.40   # lands ~0.53 palm lengths away: inside the gap
    assert model.update(marginal).index_pinch


def test_reset_clears_state(config, model):
    model.update(pose_click())
    model.reset()
    assert not model.update(pose_move()).index_pinch
