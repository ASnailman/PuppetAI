"""End-to-end: synthetic hand poses in, real action sequence out.

This is the regression test for the refactor as a whole — it exercises
HandModel -> GestureRecognizer -> PointerMapper -> GestureActions exactly as
the live pipeline does, just with scripted landmarks instead of a webcam.
"""

from conftest import (FakeBackend, make_hand, pose_click, pose_fist, pose_move,
                      pose_right_click, pose_scroll)
from puppet_functions.gesture_actions import GestureActions
from puppet_functions.gesture_recognizer import GestureRecognizer
from puppet_functions.hand_model import HandModel
from puppet_functions.pointer_mapper import PointerMapper

W, H = 1920, 1080
DT = 1 / 30    # a typical webcam frame interval


class Rig:
    """The live pipeline, minus the camera and MediaPipe."""

    def __init__(self, config):
        self.model = HandModel(config)
        self.recognizer = GestureRecognizer(config.gesture_dwell_ms)
        self.mapper = PointerMapper(W, H, config)
        self.backend = FakeBackend()
        self.actions = GestureActions(config, self.mapper, self.backend)
        self.t = 0.0
        self.labels = []

    def feed(self, landmarks, frames=1):
        for _ in range(frames):
            self.t += DT
            hand = self.model.update(landmarks)
            state = self.recognizer.update(hand, self.t)
            _, _, label = self.actions.execute(state, hand, self.t)
            self.labels.append(label)
        return self

    def lose_hand(self):
        """What the pipeline does when the hand leaves the frame."""
        self.actions.release_all()
        self.recognizer.reset()
        self.model.reset()
        return self


def test_point_then_click(config):
    rig = Rig(config).feed(pose_move(), 10)
    assert rig.backend.count('moveTo') > 0

    rig.feed(pose_click(), 2).feed(pose_move(), 2)
    assert rig.backend.count('mouseDown') == rig.backend.count('mouseUp') == 1
    assert 'CLICK' in rig.labels


def test_a_click_made_with_a_curled_finger(config):
    """As above, but with the index finger bent in to meet the thumb, the way a
    real hand pinches. This is what the extension gate used to swallow."""
    rig = Rig(config).feed(pose_move(), 10)
    rig.feed(make_hand(index=True, pinch='index', curl=0.8), 3)
    assert rig.backend.count('mouseDown') == 1
    rig.feed(pose_move(), 3)
    assert rig.backend.count('mouseUp') == 1


def test_holding_a_pinch_still_is_a_click_not_a_drag(config):
    rig = Rig(config).feed(pose_move(), 10)
    rig.feed(pose_click(), 45)               # 1.5 seconds, hand perfectly still
    assert 'DRAG' not in rig.labels
    rig.feed(pose_move(), 3)
    assert rig.backend.count('mouseDown') == rig.backend.count('mouseUp') == 1


def test_the_cursor_holds_still_through_a_whole_click(config):
    """End-to-end version of the reported problem: a click, including the
    approach and the release, must not move the cursor at all."""
    rig = Rig(config).feed(pose_move(), 20)
    aimed = rig.mapper.position

    # The thumb closes in over several frames, dragging the index finger down
    # to meet it — the movement that made the old cursor uncontrollable.
    for gap, pull in ((0.90, 0.10), (0.65, 0.20), (0.45, 0.30), (0.25, 0.35)):
        landmarks = pose_move()
        for idx in (6, 7, 8):
            landmarks[idx].x += 0.15 * pull * 0.5
            landmarks[idx].y += 0.15 * pull
        landmarks[4].x = landmarks[8].x + 0.15 * gap
        landmarks[4].y = landmarks[8].y
        rig.feed(landmarks, 1)
        assert rig.mapper.position == aimed, f"cursor moved at gap {gap}"

    rig.feed(pose_move(), 2)          # fingers spring back apart
    assert rig.backend.count('mouseDown') == rig.backend.count('mouseUp') == 1
    assert rig.mapper.position == aimed

    moved = [a for n, a in rig.backend.calls if n == 'moveTo']
    assert all(pos == aimed for pos in moved[-4:])


def test_moving_a_pinched_hand_drags(config):
    rig = Rig(config).feed(pose_move(), 10)
    rig.feed(pose_click(), 3)
    assert rig.backend.count('mouseDown') == 1

    rig.feed(pose_click(center=(0.72, 0.5)), 8)
    assert 'DRAG' in rig.labels

    rig.feed(pose_move(center=(0.72, 0.5)), 3)
    assert rig.backend.count('mouseUp') == 1


def test_a_drag_actually_carries_the_cursor_with_the_hand(config):
    """Pressing and releasing in the right places isn't enough — the cursor has
    to travel with the hand in between, smoothly and without a jump."""
    rig = Rig(config).feed(pose_move(center=(0.40, 0.5)), 20)
    grabbed = rig.mapper.position

    rig.feed(pose_click(center=(0.40, 0.5)), 2)
    assert rig.backend.count('mouseDown') == 1

    # Walk the hand across in small steps, as a real drag does.
    path = [rig.mapper.position]
    for i in range(1, 13):
        rig.feed(pose_click(center=(0.40 + 0.02 * i, 0.5)), 2)
        path.append(rig.mapper.position)

    assert 'DRAG' in rig.labels
    # The cursor moved a long way with the hand...
    assert path[-1][0] - grabbed[0] > W * 0.15
    # ...monotonically, and with no single teleporting step.
    steps = [b[0] - a[0] for a, b in zip(path, path[1:])]
    assert all(s >= 0 for s in steps), steps
    assert max(steps) < W * 0.10, f"jumped {max(steps)} px in one step"

    rig.feed(pose_move(center=(0.64, 0.5)), 3)
    assert rig.backend.count('mouseUp') == 1


def test_scroll_run(config):
    rig = Rig(config).feed(pose_move(), 6).feed(pose_scroll(), 6)
    # Now push the hand down from where the scroll began.
    lowered = pose_scroll(center=(0.5, 0.5 + 0.15 * 0.8))
    rig.feed(lowered, 20)
    assert rig.backend.count('scroll') > 0
    assert all(a[0] < 0 for n, a in rig.backend.calls if n == 'scroll')


def test_right_click_from_the_two_finger_pose(config):
    rig = Rig(config).feed(pose_move(), 6).feed(pose_right_click(), 4)
    assert rig.backend.count('rightClick') == 1
    assert rig.backend.count('mouseDown') == 0


def test_a_fist_stops_everything(config):
    # The first frames of the fist still move: the posture vote needs its
    # dwell window (~80 ms) before it commits. After that, nothing.
    rig = Rig(config).feed(pose_move(), 8).feed(pose_fist(), 8)
    settled = rig.backend.count('moveTo')
    rig.feed(pose_fist(), 10)
    assert rig.backend.count('moveTo') == settled
    assert rig.labels[-1] == 'NEUTRAL'


def test_hand_lost_mid_click_releases_the_button(config):
    rig = Rig(config).feed(pose_move(), 6).feed(pose_click(), 6)
    assert rig.backend.count('mouseDown') == 1
    rig.lose_hand()
    assert rig.backend.count('mouseUp') == 1


def test_re_acquiring_the_hand_does_not_sling_the_cursor(config):
    """After the hand reappears elsewhere, the cursor must be there at once —
    not creep across the screen from where the hand used to be."""
    elsewhere = pose_move(center=(0.75, 0.75))

    # Where a brand-new mapper puts that pose, i.e. the correct answer.
    reference = Rig(config)
    reference.feed(elsewhere, 1)
    expected = reference.mapper.position

    rig = Rig(config).feed(pose_move(center=(0.25, 0.25)), 30)
    stale = rig.mapper.position
    rig.lose_hand()
    rig.mapper.reset()          # what the pipeline does on re-acquisition
    rig.feed(elsewhere, 1)

    assert abs(rig.mapper.position[0] - expected[0]) <= 2
    assert abs(rig.mapper.position[1] - expected[1]) <= 2
    assert abs(rig.mapper.position[0] - stale[0]) > W * 0.2


def test_a_full_session_never_leaves_input_held(config):
    """Wander through every gesture, then stop — nothing may still be down."""
    rig = Rig(config)
    for pose in (pose_move(), pose_click(), pose_move(), pose_scroll(),
                 pose_right_click(), pose_fist(), pose_move()):
        rig.feed(pose, 8)
    rig.actions.release_all()

    assert rig.backend.count('mouseDown') == rig.backend.count('mouseUp')
    assert rig.backend.count('keyDown') == rig.backend.count('keyUp')
