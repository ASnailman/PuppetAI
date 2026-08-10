"""GestureRecognizer: correct postures, and the right amount of stubbornness."""

from conftest import (make_hand, pose_click, pose_fist, pose_move,
                      pose_open_palm, pose_right_click, pose_scroll, pose_zoom)
from puppet_functions.gesture_recognizer import (MOVE, NEUTRAL, SCROLL, ZOOM,
                                                 GestureRecognizer)
from puppet_functions.hand_model import HandModel


def classify(config, landmarks):
    frame = HandModel(config).update(landmarks)
    return GestureRecognizer().classify(frame), frame


def settle(config, recognizer, landmarks, model=None, frames=6, start=0.0, step=0.033):
    """Feed the same pose for a few frames, as a real hand would."""
    model = model or HandModel(config)
    state = None
    t = start
    for _ in range(frames):
        state = recognizer.update(model.update(landmarks), t)
        t += step
    return state


def test_each_pose_classifies_correctly(config):
    assert classify(config, pose_move())[0] == MOVE
    assert classify(config, pose_scroll())[0] == SCROLL
    assert classify(config, pose_zoom())[0] == ZOOM
    assert classify(config, pose_fist())[0] == NEUTRAL
    assert classify(config, pose_open_palm())[0] == NEUTRAL


def test_pinching_reports_a_pinch_not_a_posture_change(config):
    posture, frame = classify(config, pose_click())
    assert posture == MOVE          # you are still pointing
    assert frame.index_pinch        # ...and also clicking


def test_a_realistically_bent_pinch_still_clicks(config):
    """The regression that mattered: a real pinch bends the index finger at the
    PIP joint to meet the thumb. Past about 60 degrees of bend that pushes the
    finger under the 'extended' threshold — so gating the click on extension
    swallowed exactly the firm, deliberate pinches users make when a light one
    didn't seem to register."""
    for amount in (0.3, 0.5, 0.7, 0.9, 1.0):
        frame = HandModel(config).update(
            make_hand(index=True, pinch='index', curl=amount))
        assert GestureRecognizer().update(frame, 0.0).index_pinch, \
            f"click lost at curl={amount}"


def test_the_pinch_gate_has_margin_at_both_ends(config):
    """The gate separates pinch from fist, and it has to do so with room to
    spare — hand proportions differ between people, so a threshold that only
    just clears the hardest pinch will fail for somebody.

    Guards against the regression where the threshold sat *between* a firm and
    a very firm pinch, so pinching harder made the click less likely to fire.
    """
    hardest = min(
        HandModel(config).update(
            make_hand(index=True, pinch='index', curl=c)).index_reach
        for c in (0.3, 0.5, 0.7, 0.9, 1.0)
    )
    fist = HandModel(config).update(pose_fist()).index_reach

    assert hardest > config.pinch_reach_on * 1.10, \
        f"only {hardest:.2f} vs threshold {config.pinch_reach_on}"
    assert fist < config.pinch_reach_on * 0.95


def test_a_firm_pinch_is_past_the_old_extension_gate(config):
    """Proves the test above isn't vacuous: at this bend the finger genuinely
    no longer counts as extended, yet it must still click."""
    frame = HandModel(config).update(make_hand(index=True, pinch='index', curl=0.8))
    assert not frame.index_extended     # the old gate would have refused
    assert frame.index_reaching         # the new one still passes
    assert GestureRecognizer().update(frame, 0.0).index_pinch


def test_a_fist_never_clicks(config):
    """In a real fist the thumb rests on the curled index finger, which is
    close enough to look like a pinch — so pinches require an extended finger."""
    r = GestureRecognizer()
    frame = HandModel(config).update(pose_fist())
    state = r.update(frame, 0.0)
    assert not state.index_pinch
    assert not state.middle_pinch


def test_pointing_does_not_look_like_a_right_click(config):
    """The tucked thumb sits right next to the curled middle fingertip."""
    r = GestureRecognizer()
    frame = HandModel(config).update(pose_move())
    assert not r.update(frame, 0.0).middle_pinch


def test_right_click_pose_is_not_mistaken_for_zoom(config):
    """Thumb on the middle fingertip reads as far from the palm, which would
    otherwise look identical to a splayed thumb."""
    frame = HandModel(config).update(pose_right_click())
    r = GestureRecognizer()
    assert r.update(frame, 0.0).middle_pinch
    assert r.classify(frame) != ZOOM


def test_ring_and_pinky_are_ignored_while_pointing(config):
    """A relaxed hand lets these drift; demanding a state from them dropped
    tracking constantly in the old version."""
    for ring in (True, False):
        for pinky in (True, False):
            if ring and pinky:
                continue   # that plus index/middle would be an open palm
            posture, _ = classify(config, make_hand(index=True, ring=ring, pinky=pinky))
            assert posture == MOVE


def test_stable_posture_is_reached(config):
    r = GestureRecognizer(dwell_ms=80)
    assert settle(config, r, pose_move()).posture == MOVE


def test_single_bad_frame_does_not_change_the_posture(config):
    r = GestureRecognizer(dwell_ms=120)
    model = HandModel(config)
    t = 0.0
    for _ in range(8):
        r.update(model.update(pose_move()), t)
        t += 0.033
    # One glitched frame — MediaPipe does this when a hand is partly occluded.
    state = r.update(model.update(pose_open_palm()), t)
    assert state.posture == MOVE


def test_sustained_change_does_switch(config):
    r = GestureRecognizer(dwell_ms=80)
    model = HandModel(config)
    settle(config, r, pose_move(), model=model)
    state = settle(config, r, pose_scroll(), model=model, start=1.0)
    assert state.posture == SCROLL


def test_leaving_scroll_needs_more_evidence_than_entering_it(config):
    """Scroll is sticky: a wobble mid-scroll must not drop you out."""
    r = GestureRecognizer(dwell_ms=120)
    model = HandModel(config)
    settle(config, r, pose_scroll(), model=model, frames=10)
    t = 1.0
    # Alternate frames — a clear majority never forms.
    for pose in (pose_move(), pose_scroll(), pose_move(), pose_scroll()):
        state = r.update(model.update(pose), t)
        t += 0.033
    assert state.posture == SCROLL


def test_pinch_needs_no_dwell_at_all(config):
    """Clicks bypass the vote entirely — that's the click-latency fix."""
    r = GestureRecognizer(dwell_ms=200)
    model = HandModel(config)
    r.update(model.update(pose_move()), 0.0)
    state = r.update(model.update(pose_click()), 0.033)
    assert state.index_pinch      # fires on the very next frame


def test_reset_returns_to_neutral(config):
    r = GestureRecognizer()
    settle(config, r, pose_scroll())
    r.reset()
    assert r._posture == NEUTRAL
