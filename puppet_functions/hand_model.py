"""Turns raw MediaPipe landmarks into stable, human-meaningful hand features.

This module is the heart of the "you shouldn't have to hold your hand exactly
right" fix.  Two ideas do all the work:

1. **Scale normalisation.** Every distance is divided by the palm length
   (wrist -> middle knuckle).  Thresholds therefore become "fractions of a
   palm" instead of "fractions of the camera frame", so the exact same gesture
   registers whether you are 40 cm or 90 cm from the webcam.

2. **Rotation invariance.** Finger extension is measured as how much further
   the fingertip reaches from the wrist than its own middle joint does — a
   comparison of two distances, which does not care which way the hand is
   pointing.  The old `tip.y < knuckle.y` test only worked with the hand held
   upright, which is what made the app feel so rigid.

Everything is measured in 2D (x, y).  MediaPipe's z is a relative depth
estimate with far more noise than the in-plane coordinates, and including it
made thresholds jump around for no benefit.
"""

from dataclasses import dataclass

from .filters import EMA, Hysteresis

# MediaPipe hand landmark indices, spelled out so this module (and its tests)
# don't need to import mediapipe.  These indices are fixed by the model.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# (tip, pip) pairs used by the extension metric, in finger order.
_FINGER_JOINTS = {
    'index':  (INDEX_TIP, INDEX_PIP),
    'middle': (MIDDLE_TIP, MIDDLE_PIP),
    'ring':   (RING_TIP, RING_PIP),
    'pinky':  (PINKY_TIP, PINKY_PIP),
}

# A palm length can never legitimately be this small; anything at or below it
# means the detection collapsed and we must not divide by it.
_MIN_SCALE = 1e-6

# Pinch distance (in palm units) at which the fingers are considered fully
# open.  Only used to turn the raw distance into a 0..1 strength for the HUD.
_PINCH_OPEN_REF = 1.0


def _dist(a, b):
    """2D Euclidean distance between two landmarks."""
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


@dataclass(frozen=True)
class HandFrame:
    """One frame's worth of interpreted hand state.

    Immutable and cheap — the gesture recogniser reads only this, never raw
    landmarks, which keeps all the geometry in one place.
    """

    # Normalised (0..1) image position the cursor should track. NOT the
    # fingertip by default — see HandModel._cursor_point.
    pointer_x: float
    pointer_y: float

    # The actual index fingertip, kept for drawing and diagnostics.
    index_tip_x: float
    index_tip_y: float

    # Palm length in image units. Also a rough proxy for "how close am I to the
    # camera", used to keep scroll gain distance-independent.
    scale: float

    # Which fingers are extended (hysteresis-filtered, so they don't flicker).
    index_extended: bool
    middle_extended: bool
    ring_extended: bool
    pinky_extended: bool
    thumb_out: bool

    # Pinch distances in palm units, plus their debounced on/off states.
    index_pinch_distance: float
    middle_pinch_distance: float
    index_pinch: bool
    middle_pinch: bool

    # Is the fingertip far enough from the wrist for a pinch there to be a
    # real pinch rather than a closed fist? See the note above _cursor_point.
    index_reach: float
    index_reaching: bool
    middle_reaching: bool

    # 0..1, for the overlay's pinch ring — lets the user *see* how close to a
    # click they are instead of guessing.
    pinch_strength: float

    @property
    def open_palm(self):
        """All four fingers out — the natural "I'm done, relax" posture."""
        return (self.index_extended and self.middle_extended
                and self.ring_extended and self.pinky_extended)

    @property
    def fist(self):
        return not (self.index_extended or self.middle_extended
                    or self.ring_extended or self.pinky_extended)


class HandModel:
    """Stateful wrapper that produces a `HandFrame` per video frame.

    The state is only the hysteresis triggers and the smoothed palm scale;
    everything derived is recomputed each frame.
    """

    def __init__(self, config=None):
        get = (lambda k, d: getattr(config, k, d)) if config is not None else (lambda k, d: d)

        # Extension: tip reaches this many palm-lengths further from the wrist
        # than its own PIP joint.  A straight finger sits near +0.5, a curled
        # one near -0.4, so the 0.25/0.10 gap is comfortably inside the void
        # between them while still catching a lazily half-straightened finger.
        ext_on = get('finger_ext_on', 0.25)
        ext_off = get('finger_ext_off', 0.10)
        self._ext = {
            name: Hysteresis(ext_on, ext_off, invert=True)
            for name in _FINGER_JOINTS
        }

        # The thumb doesn't curl like the other fingers — it swings across the
        # palm.  Measuring how far the thumb tip sits from the pinky knuckle
        # captures that spread far more reliably than a reach metric does.
        self._thumb = Hysteresis(
            get('thumb_out_on', 1.25),
            get('thumb_out_off', 1.10),
            invert=True,
        )

        # Pinch: thumb tip to fingertip, in palm units.  Touching tips land
        # around 0.2 (fingers have thickness); a relaxed open hand is ~1.0.
        # The 0.45/0.65 gap means a resting hand can never accidentally click,
        # yet a light tap is enough to trigger one.
        pinch_on = get('pinch_on', 0.45)
        pinch_off = get('pinch_off', 0.65)
        self._index_pinch = Hysteresis(pinch_on, pinch_off)
        self._middle_pinch = Hysteresis(pinch_on, pinch_off)
        self._pinch_on = pinch_on

        # How far a fingertip must sit from the wrist for a thumb touching it
        # to count as a pinch. This is what separates a pinch from a fist —
        # see the note above _cursor_point for why extension can't do that job.
        reach_on = get('pinch_reach_on', 1.15)
        reach_off = get('pinch_reach_off', 1.02)
        self._index_reach = Hysteresis(reach_on, reach_off, invert=True)
        self._middle_reach = Hysteresis(reach_on, reach_off, invert=True)

        # Palm scale is smoothed: a single bad frame shrinking the "palm" would
        # otherwise rescale every threshold at once.
        self._scale = EMA(get('scale_smoothing', 0.3))

        # See _cursor_point. 'virtual' keeps the cursor still while you pinch.
        self._cursor_source = get('cursor_source', 'virtual')
        self._cursor_offset = get('cursor_offset_palms', 1.0)

    def update(self, landmarks) -> HandFrame:
        """Compute this frame's features from 21 MediaPipe landmarks."""
        wrist = landmarks[WRIST]

        # Palm length is the reference unit for everything below.  It is a good
        # choice because it barely changes as fingers move, unlike, say, the
        # hand's bounding box.
        raw_scale = _dist(wrist, landmarks[MIDDLE_MCP])
        scale = self._scale.update(raw_scale)
        if scale < _MIN_SCALE:
            scale = _MIN_SCALE

        # Extension per finger: positive when the tip reaches past its own PIP
        # joint, away from the wrist.  Purely distance-based, so tilting or
        # rotating the hand changes nothing.
        extended = {}
        for name, (tip_i, pip_i) in _FINGER_JOINTS.items():
            reach = (_dist(landmarks[tip_i], wrist) - _dist(landmarks[pip_i], wrist)) / scale
            extended[name] = self._ext[name].update(reach)

        thumb_spread = _dist(landmarks[THUMB_TIP], landmarks[PINKY_MCP]) / scale
        thumb_out = self._thumb.update(thumb_spread)

        thumb_tip = landmarks[THUMB_TIP]
        index_pinch_d = _dist(thumb_tip, landmarks[INDEX_TIP]) / scale
        middle_pinch_d = _dist(thumb_tip, landmarks[MIDDLE_TIP]) / scale

        index_reach = _dist(landmarks[INDEX_TIP], wrist) / scale

        cursor_x, cursor_y = self._cursor_point(landmarks, wrist, scale)

        return HandFrame(
            pointer_x=cursor_x,
            pointer_y=cursor_y,
            index_tip_x=landmarks[INDEX_TIP].x,
            index_tip_y=landmarks[INDEX_TIP].y,
            scale=scale,
            index_extended=extended['index'],
            middle_extended=extended['middle'],
            ring_extended=extended['ring'],
            pinky_extended=extended['pinky'],
            thumb_out=thumb_out,
            index_pinch_distance=index_pinch_d,
            middle_pinch_distance=middle_pinch_d,
            index_pinch=self._index_pinch.update(index_pinch_d),
            middle_pinch=self._middle_pinch.update(middle_pinch_d),
            index_reach=index_reach,
            index_reaching=self._index_reach.update(index_reach),
            middle_reaching=self._middle_reach.update(
                _dist(landmarks[MIDDLE_TIP], wrist) / scale),
            pinch_strength=self._strength(index_pinch_d),
        )

    # `reach` (computed inline in update) is dist(fingertip, wrist) / scale.
    #
    # It exists because "is the finger extended?" is the wrong question to ask
    # about a pinch. Pinching *is* bending the index finger down to meet the
    # thumb, so an extension test goes false exactly when the user is trying to
    # click, and the click never fires. Reach asks the question that actually
    # matters — is the fingertip out where a pinch happens, or balled up in a
    # fist?
    #
    # Measured against adult hand proportions (palm 9.5 cm, index 9.1 cm),
    # by how far the finger is bent at the PIP joint:
    #
    #   pointing straight  1.91      firm pinch (80°)   1.44
    #   relaxed point      1.83      very firm (95°)    1.29
    #   light pinch (50°)  1.70      fist (140°)        1.03
    #   normal pinch (65°) 1.58      tight fist (160°)  1.06
    #
    # Hence 1.15 on / 1.02 off: it clears the hardest pinch a user can make by
    # a comfortable margin and still rejects a balled fist. An earlier 1.35 sat
    # *between* "firm pinch" and "very firm pinch" — so pinching harder, which
    # is exactly what people do when a click doesn't seem to register, made the
    # click less likely to fire rather than more.

    def _cursor_point(self, landmarks, wrist, scale):
        """Where the cursor should sit, in normalised image coordinates.

        The obvious answer — the index fingertip — is the wrong one, because it
        is the exact landmark a pinch has to move.  Reaching your thumb across
        pulls the fingertip with it, so the cursor lurches at the worst
        possible moment: while you are trying to click something.

        The default 'virtual' source instead projects a point one palm-length
        out from the index knuckle, along the wrist->knuckle axis.  It sits
        roughly where your fingertip would be, and it rotates as you point, but
        it is rigid with respect to the palm — curling or pinching your fingers
        does not move it at all.
        """
        mcp = landmarks[INDEX_MCP]
        source = self._cursor_source

        if source == 'fingertip':
            tip = landmarks[INDEX_TIP]
            return tip.x, tip.y
        if source == 'knuckle':
            return mcp.x, mcp.y

        dx, dy = mcp.x - wrist.x, mcp.y - wrist.y
        length = (dx * dx + dy * dy) ** 0.5
        if length < _MIN_SCALE:
            return mcp.x, mcp.y   # degenerate frame — no usable direction
        reach = self._cursor_offset * scale
        return mcp.x + dx / length * reach, mcp.y + dy / length * reach

    def _strength(self, distance):
        """Map a pinch distance to 0 (wide open) .. 1 (touching)."""
        span = _PINCH_OPEN_REF - self._pinch_on
        if span <= 0:
            return 0.0
        return _clamp01((_PINCH_OPEN_REF - distance) / span)

    def reset(self):
        """Forget all debounced state — called when the hand leaves the frame."""
        for h in self._ext.values():
            h.reset()
        self._thumb.reset()
        self._index_pinch.reset()
        self._middle_pinch.reset()
        self._index_reach.reset()
        self._middle_reach.reset()
        self._scale.reset()
