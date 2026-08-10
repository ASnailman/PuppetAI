"""Decides what the hand is *asking for*, from the features in `hand_model`.

Two things separate this from the old recogniser:

* It reads only `HandFrame` booleans, never raw landmarks, so there is no
  hand-orientation or left/right-handed assumption anywhere in here.
* Postures and pinches are stabilised at different speeds.  A posture change
  (move -> scroll) is a deliberate act and can afford a short settling delay;
  a click must feel instant.  The old code pushed clicks through the same
  3-unanimous-frames buffer as postures, which is a large part of why clicking
  felt late and unresponsive.
"""

from collections import deque
from dataclasses import dataclass

# Postures — what the hand's shape means. Pinches are tracked separately.
NEUTRAL = 'NEUTRAL'
MOVE = 'MOVE'
SCROLL = 'SCROLL'
ZOOM = 'ZOOM'

# Postures you're mid-way through *using*. Leaving them needs stronger evidence
# so a momentary tracking wobble can't drop you out halfway through a scroll.
_STICKY = (SCROLL, ZOOM)

_ENTER_RATIO = 0.6   # fraction of recent frames that must agree to switch
_LEAVE_RATIO = 0.75  # ...raised to this when abandoning a sticky posture


def _index_pinch(frame):
    """A pinch only counts when the fingertip is out where pinches happen.

    Some gate is needed, because in a closed fist the thumb rests against the
    curled index finger and reads as a pinch. But that gate must NOT be
    "is the finger extended" — pinching *is* curling the index down to meet
    the thumb, so an extension test goes false at the exact moment the user
    wants to click, and the click is swallowed.

    Reach (fingertip-to-wrist distance) answers the question that actually
    separates the two poses, and a pinching finger passes it comfortably.
    """
    return frame.index_pinch and frame.index_reaching


def _middle_pinch(frame):
    return frame.middle_pinch and frame.middle_reaching


@dataclass(frozen=True)
class GestureState:
    """What the recogniser believes right now."""

    posture: str          # debounced hand shape
    index_pinch: bool     # thumb touching index tip  -> left click / drag
    middle_pinch: bool    # thumb touching middle tip -> right click
    pinch_strength: float # 0..1, for the overlay


class GestureRecognizer:
    def __init__(self, dwell_ms=80):
        # Window over which posture votes are counted. 80 ms is roughly 2-3
        # camera frames: long enough to swallow a single bad detection, short
        # enough that switching gestures still feels immediate.
        self._dwell_s = dwell_ms / 1000.0
        self._votes = deque()          # (timestamp, posture)
        self._posture = NEUTRAL

    def classify(self, frame) -> str:
        """Instantaneous posture for a single frame — no debouncing.

        Order matters: pinches are checked before the postures they could be
        confused with, so that tapping the thumb onto the middle finger reads
        as a right click rather than as a thumb-out zoom.
        """
        # A fully open hand and a closed fist are both "do nothing".  Having an
        # open palm mean rest matters: it lets you stop the cursor by simply
        # relaxing, rather than by holding a deliberate pose.
        if frame.open_palm or frame.fist:
            return NEUTRAL

        two_fingers = frame.index_extended and frame.middle_extended

        if two_fingers and not _middle_pinch(frame) and not _index_pinch(frame):
            # Thumb splayed away from the palm distinguishes zoom from scroll.
            return ZOOM if frame.thumb_out else SCROLL

        if frame.index_extended:
            # Ring and pinky are deliberately ignored here — they curl and
            # uncurl on their own while pointing, and demanding a specific
            # state from them was a major source of dropped tracking.
            return MOVE

        return NEUTRAL

    def update(self, frame, now) -> GestureState:
        """Feed one frame; returns the debounced state to act on."""
        candidate = self.classify(frame)

        # Keep only votes inside the dwell window.
        self._votes.append((now, candidate))
        cutoff = now - self._dwell_s
        while self._votes and self._votes[0][0] < cutoff:
            self._votes.popleft()

        if candidate != self._posture:
            agree = sum(1 for _, p in self._votes if p == candidate)
            ratio = _LEAVE_RATIO if self._posture in _STICKY else _ENTER_RATIO
            # `len(self._votes) > 1` stops the very first frame of a new
            # posture from committing on a sample size of one.
            if len(self._votes) > 1 and agree / len(self._votes) >= ratio:
                self._posture = candidate

        return GestureState(
            posture=self._posture,
            # Pinches pass straight through: `hand_model` already debounced
            # them with hysteresis, and adding dwell on top would only add lag.
            index_pinch=_index_pinch(frame),
            middle_pinch=_middle_pinch(frame),
            pinch_strength=frame.pinch_strength,
        )

    def reset(self):
        """Clear vote history — called when the hand leaves the frame."""
        self._votes.clear()
        self._posture = NEUTRAL
