"""GestureActions: what actually reaches the mouse.

Nothing here touches the real cursor — a recording backend stands in for
pyautogui, so every click, drag and scroll is asserted exactly.
"""

from dataclasses import dataclass

import pytest

from puppet_functions.gesture_actions import GestureActions
from puppet_functions.gesture_recognizer import GestureState
from puppet_functions.pointer_mapper import PointerMapper

W, H = 1920, 1080


@dataclass
class Frame:
    """Minimal stand-in for a HandFrame — these are the fields actions read.

    Pinch distances default to 'wide open' so the cursor isn't damped unless a
    test deliberately closes them.
    """
    pointer_x: float = 0.5
    pointer_y: float = 0.5
    scale: float = 0.15
    index_pinch_distance: float = 1.0
    middle_pinch_distance: float = 1.0


def state(posture='MOVE', index_pinch=False, middle_pinch=False):
    return GestureState(posture, index_pinch, middle_pinch, 0.0)


@pytest.fixture
def actions(config, backend):
    mapper = PointerMapper(W, H, config)
    return GestureActions(config, mapper, backend), mapper


def test_move_drives_the_cursor(actions, backend):
    act, _ = actions
    act.execute(state('MOVE'), Frame(), now=0.0)
    assert 'moveTo' in backend.names()


def test_neutral_moves_nothing(actions, backend):
    act, _ = actions
    act.execute(state('NEUTRAL'), Frame(), now=0.0)
    assert backend.names() == []


def test_the_press_carries_its_own_coordinates(actions, backend):
    """The button event must name the point it belongs to, not rely on a
    separate moveTo having already landed."""
    act, mapper = actions
    for i in range(60):
        act.execute(state('MOVE'), Frame(0.4, 0.6), now=i / 60)
    aimed = mapper.position

    act.execute(state('MOVE', index_pinch=True), Frame(0.4, 0.6), now=1.0)
    down = [a for n, a in backend.calls if n == 'mouseDown']
    assert down == [aimed]

    act.execute(state('MOVE'), Frame(0.4, 0.6), now=1.1)
    up = [a for n, a in backend.calls if n == 'mouseUp']
    assert up == [aimed]


def test_a_pinch_presses_immediately(actions, backend):
    """The button goes down on the pinch itself — no waiting to see whether
    the release will arrive, and nothing to classify retroactively."""
    act, _ = actions
    act.execute(state('MOVE'), Frame(), now=0.0)
    act.execute(state('MOVE', index_pinch=True), Frame(), now=0.10)
    assert backend.count('mouseDown') == 1
    assert backend.count('mouseUp') == 0

    act.execute(state('MOVE'), Frame(), now=0.16)
    assert backend.count('mouseUp') == 1


def test_a_long_hold_is_still_a_click_not_a_drag(actions, backend):
    """Holding the pinch a beat longer must not silently become a drag —
    that was the single most common way a click went missing."""
    act, mapper = actions
    act.execute(state('MOVE', index_pinch=True), Frame(0.5, 0.5), now=0.0)
    for i in range(40):                      # 1.3 seconds, hand held still
        act.execute(state('MOVE', index_pinch=True), Frame(0.5, 0.5), now=i / 30)
        assert mapper.frozen                 # never slipped into drag mode
    act.execute(state('MOVE'), Frame(0.5, 0.5), now=2.0)

    assert backend.count('mouseDown') == backend.count('mouseUp') == 1


def test_a_click_survives_a_dropped_tracking_frame(actions, backend):
    """A frame where the pinch flickers off must not release the button early.
    The pipeline's grace window covers hand loss; hysteresis covers this."""
    act, _ = actions
    act.execute(state('MOVE', index_pinch=True), Frame(), now=0.0)
    act.execute(state('MOVE', index_pinch=True), Frame(), now=0.03)
    assert backend.count('mouseUp') == 0
    act.execute(state('MOVE'), Frame(), now=0.06)
    assert backend.count('mouseUp') == 1


def test_the_cursor_is_frozen_while_pinching(actions, backend):
    """The fix for clicks landing beside their target."""
    act, mapper = actions
    for i in range(60):
        act.execute(state('MOVE'), Frame(0.5, 0.5), now=i / 60)
    anchored = mapper.position

    act.execute(state('MOVE', index_pinch=True), Frame(0.5, 0.5), now=1.0)
    # The thumb tugs the fingertip as it closes — the cursor must not follow.
    x, y, _ = act.execute(state('MOVE', index_pinch=True), Frame(0.62, 0.58), now=1.05)
    assert (x, y) == anchored


def test_the_cursor_slows_as_the_pinch_approaches(actions):
    """The lurch happens *during* the approach, before the click ever fires —
    so the gain has to fade out on the way in, not snap off at the end."""
    act, mapper = actions
    for i in range(60):
        act.execute(state('MOVE'), Frame(0.5, 0.5), now=i / 60)
    start = mapper.position

    # Thumb closing in (1.0 -> 0.45), while the hand also drifts sideways.
    travelled = []
    t = 1.0
    for d in (0.90, 0.75, 0.60, 0.50, 0.46):
        t += 1 / 30
        x, _, _ = act.execute(state('MOVE'), Frame(0.62, 0.5, index_pinch_distance=d), now=t)
        travelled.append(abs(x - start[0]))

    # Movement per frame keeps shrinking as the fingers close.
    steps = [b - a for a, b in zip(travelled, travelled[1:])]
    assert all(later <= earlier + 1 for earlier, later in zip(steps, steps[1:]))

    # And by the time the pinch is about to trigger, it's essentially stopped.
    assert steps[-1] < 2


def test_the_cursor_stays_put_after_a_click_releases(actions, backend):
    """Fingers spring apart on release; the cursor must not chase them."""
    act, mapper = actions
    act.execute(state('MOVE', index_pinch=True), Frame(0.5, 0.5), now=0.0)
    act.execute(state('MOVE'), Frame(0.5, 0.5), now=0.05)      # button released
    assert backend.count('mouseUp') == 1
    anchored = mapper.position

    # Well inside the 150 ms settle window: the hand recoils, the cursor doesn't.
    for i in range(4):
        x, y, _ = act.execute(state('MOVE'), Frame(0.75, 0.75), now=0.06 + i * 0.02)
        assert (x, y) == anchored

    # After it expires, tracking resumes.
    act.execute(state('MOVE'), Frame(0.75, 0.75), now=0.5)
    assert not mapper.frozen


def test_dragging_is_never_damped(actions):
    """A drag holds a tight pinch by definition — damping it would freeze the
    drag solid."""
    act, _ = actions
    pinched = Frame(0.5, 0.5, index_pinch_distance=0.3)
    act.execute(state('MOVE', index_pinch=True), pinched, now=0.0)
    # Move past the slop to enter drag mode, then keep moving.
    act.execute(state('MOVE', index_pinch=True),
                Frame(0.5 + 0.15 * 0.5, 0.5, index_pinch_distance=0.3), now=0.1)
    x0, _, _ = act.execute(state('MOVE', index_pinch=True),
                           Frame(0.5 + 0.15 * 0.5, 0.5, index_pinch_distance=0.3), now=0.2)
    x1, _, _ = act.execute(state('MOVE', index_pinch=True),
                           Frame(0.80, 0.5, index_pinch_distance=0.3), now=0.3)
    assert x1 > x0


def test_two_quick_pinches_produce_two_presses(actions, backend):
    """We let the OS's own double-click timer do the work; detecting it here
    would mean delaying every single click by the double-click window."""
    act, _ = actions
    t = 0.0
    for _ in range(2):
        act.execute(state('MOVE', index_pinch=True), Frame(), now=t)
        act.execute(state('MOVE'), Frame(), now=t + 0.05)
        t += 0.12
    assert backend.count('mouseDown') == backend.count('mouseUp') == 2
    # Down/up strictly alternate — no overlapping presses.
    assert [n for n in backend.names() if n.startswith('mouse')] == \
        ['mouseDown', 'mouseUp', 'mouseDown', 'mouseUp']


def test_moving_while_pinched_starts_a_drag(actions, backend):
    act, mapper = actions
    act.execute(state('MOVE', index_pinch=True), Frame(0.5, 0.5), now=0.0)
    assert mapper.frozen
    # 0.5 palm-lengths of travel, well past the 0.35 slop.
    act.execute(state('MOVE', index_pinch=True), Frame(0.5 + 0.15 * 0.5, 0.5), now=0.05)
    assert not mapper.frozen                 # cursor released to follow the hand
    assert backend.count('mouseDown') == 1

    act.execute(state('MOVE'), Frame(0.5 + 0.15 * 0.5, 0.5), now=0.20)
    assert backend.count('mouseUp') == 1


def test_a_small_wobble_does_not_turn_a_click_into_a_drag(actions, backend):
    act, mapper = actions
    act.execute(state('MOVE', index_pinch=True), Frame(0.5, 0.5), now=0.0)
    act.execute(state('MOVE', index_pinch=True), Frame(0.5 + 0.15 * 0.05, 0.5), now=0.05)
    assert mapper.frozen
    act.execute(state('MOVE'), Frame(0.5, 0.5), now=0.10)
    assert backend.count('mouseDown') == backend.count('mouseUp') == 1


def test_right_click_fires_once_per_tap(actions, backend):
    act, _ = actions
    for t in (0.0, 0.03, 0.06):
        act.execute(state('MOVE', middle_pinch=True), Frame(), now=t)
    assert backend.count('rightClick') == 1
    act.execute(state('MOVE'), Frame(), now=0.09)
    act.execute(state('MOVE', middle_pinch=True), Frame(), now=0.12)
    assert backend.count('rightClick') == 2


def test_scroll_inside_the_deadzone_does_nothing(actions, backend):
    act, _ = actions
    act.execute(state('SCROLL'), Frame(0.5, 0.5), now=0.0)
    act.execute(state('SCROLL'), Frame(0.5, 0.5 + 0.15 * 0.05), now=0.1)
    assert backend.count('scroll') == 0


def test_scroll_direction_follows_the_hand(actions, backend):
    act, _ = actions
    act.execute(state('SCROLL'), Frame(0.5, 0.5), now=0.0)
    for i in range(1, 20):
        # Hand well below the anchor -> page scrolls down -> negative wheel.
        act.execute(state('SCROLL'), Frame(0.5, 0.5 + 0.15 * 0.8), now=i * 0.05)
    deltas = [a[0] for n, a in act._be.calls if n == 'scroll']
    assert deltas and all(d < 0 for d in deltas)

    act.release_all()
    backend.calls.clear()
    act.execute(state('SCROLL'), Frame(0.5, 0.5), now=5.0)
    for i in range(1, 20):
        act.execute(state('SCROLL'), Frame(0.5, 0.5 - 0.15 * 0.8), now=5.0 + i * 0.05)
    deltas = [a[0] for n, a in backend.calls if n == 'scroll']
    assert deltas and all(d > 0 for d in deltas)


def test_scroll_speed_grows_with_distance(actions, config, backend):
    def total(offset_palms):
        mapper = PointerMapper(W, H, config)
        be = type(backend)()
        act = GestureActions(config, mapper, be)
        act.execute(state('SCROLL'), Frame(0.5, 0.5), now=0.0)
        for i in range(1, 21):
            act.execute(state('SCROLL'), Frame(0.5, 0.5 + 0.15 * offset_palms),
                        now=i * 0.05)
        return sum(abs(a[0]) for n, a in be.calls if n == 'scroll')

    assert total(0.9) > total(0.4) > 0


def test_scroll_pins_the_cursor(actions):
    act, mapper = actions
    act.execute(state('SCROLL'), Frame(0.5, 0.5), now=0.0)
    assert mapper.frozen
    act.execute(state('MOVE'), Frame(0.5, 0.5), now=0.1)
    assert not mapper.frozen


def test_zoom_holds_ctrl_and_releases_it(actions, backend):
    act, _ = actions
    act.execute(state('ZOOM'), Frame(0.5, 0.5), now=0.0)
    for i in range(1, 20):
        act.execute(state('ZOOM'), Frame(0.5, 0.5 + 0.15 * 0.9), now=i * 0.05)
    assert backend.count('keyDown') == 1
    assert backend.count('scroll') > 0
    assert backend.count('keyUp') == 0

    act.execute(state('MOVE'), Frame(), now=2.0)
    assert backend.count('keyUp') == 1


def test_release_all_lets_go_of_everything(actions, backend):
    act, mapper = actions
    act.execute(state('MOVE', index_pinch=True), Frame(), now=0.0)
    act.execute(state('MOVE', index_pinch=True), Frame(), now=0.5)
    assert backend.count('mouseDown') == 1

    act.release_all()
    assert backend.count('mouseUp') == 1
    assert not mapper.frozen
    # Idempotent: pausing twice must not emit a second mouseUp.
    act.release_all()
    assert backend.count('mouseUp') == 1


def test_losing_the_hand_mid_drag_never_leaves_a_button_down(actions, backend):
    act, _ = actions
    act.execute(state('MOVE', index_pinch=True), Frame(), now=0.0)
    act.execute(state('MOVE', index_pinch=True), Frame(), now=0.4)
    act.release_all()                      # what the pipeline does on hand loss
    assert backend.count('mouseDown') == backend.count('mouseUp') == 1
