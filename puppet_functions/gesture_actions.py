"""Turns recognised gestures into real mouse and keyboard events.

Each interaction is a small explicit engine rather than an `if` in a loop,
because each one needs its own memory (when did the pinch start, where was the
scroll anchored, is a button currently held).

The `backend` is injected — it defaults to pyautogui, but the tests pass a
recorder object instead, which is how the whole action layer is verified
without touching the real cursor.
"""

import time

# Distance in palm-lengths the hand may drift during a pinch before we decide
# the user meant to drag rather than click. Small on purpose: the cursor is
# pinned until this is crossed, so a large value means the start of every drag
# is dead time where the hand moves and nothing happens.
_DEFAULT_DRAG_SLOP = 0.12


def _dist(ax, ay, bx, by):
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


class GestureActions:
    def __init__(self, config, mapper, backend=None):
        if backend is None:
            import pyautogui
            backend = pyautogui
        self._be = backend
        self._mapper = mapper

        get = (lambda k, d: getattr(config, k, d)) if config is not None else (lambda k, d: d)
        # How far the hand may drift with the button held before we treat it as
        # a drag and let the cursor follow again.
        self._drag_slop = get('drag_slop_palm', _DEFAULT_DRAG_SLOP)
        # Scroll only begins once the hand has moved this far from where the
        # gesture started, so simply forming the posture doesn't scroll.
        self._scroll_deadzone = get('scroll_deadzone', 0.15)
        self._scroll_gain = get('scroll_gain', 900.0)
        self._zoom_deadzone = get('zoom_deadzone', 0.25)
        self._zoom_gain = get('zoom_gain', 90.0)
        # Cursor damping ramps between these two pinch distances, so the
        # cursor has already stopped by the time the click actually fires.
        self._damp_start = get('pinch_damp_start', 0.80)
        self._pinch_on = get('pinch_on', 0.45)
        # How long the cursor stays pinned after a click, so the fingers
        # springing back apart doesn't fling it.
        self._settle_s = get('click_settle_ms', 150) / 1000.0
        self._settle_until = 0.0
        # Pause between moving the cursor and pressing the button. Costs about
        # one camera frame; without it clicks can land before the target has
        # registered the cursor arriving.
        self._press_delay = get('click_press_delay_ms', 25) / 1000.0

        # Click / drag state. `_button_down` means we have issued a mouseDown
        # that is still outstanding; `_drag_mode` means the hand has since
        # moved far enough that the cursor should follow it.
        self._button_down = False
        self._drag_mode = False
        self._pinch_origin = (0.0, 0.0)
        self._right_held = False

        # Scroll / zoom state
        self._anchor_y = None
        self._scroll_residue = 0.0   # fractional clicks carried between frames
        self._ctrl_down = False
        self._last_time = None
        self._label = 'NEUTRAL'

    # ------------------------------------------------------------------ main

    def execute(self, state, frame, now=None):
        """Apply one frame of gesture state. Returns (x, y, label)."""
        if now is None:
            now = time.time()
        dt = 0.0 if self._last_time is None else max(0.0, now - self._last_time)
        self._last_time = now

        posture = state.posture

        # Scroll and zoom pin the cursor: you want the page under the pointer to
        # scroll, not to drift to a different element as your hand travels.
        if posture in ('SCROLL', 'ZOOM'):
            if self._anchor_y is None:
                self._anchor_y = frame.pointer_y
                self._scroll_residue = 0.0
                self._mapper.freeze()
        elif self._anchor_y is not None:
            self._end_scroll()

        if posture != 'ZOOM':
            self._release_ctrl()

        # The post-click settle window has expired — let the cursor move again.
        if self._settle_until and now >= self._settle_until:
            self._settle_until = 0.0
            self._mapper.unfreeze()

        # While a button is held the cursor must track the hand exactly, so the
        # mapper's continuity offset is frozen rather than decaying away.
        self._mapper.hold_offset(self._button_down)

        x, y = self._mapper.map(frame.pointer_x, frame.pointer_y, now,
                                self._cursor_damping(frame))

        self._label = posture
        self._handle_click(state, frame, now, x, y)
        self._handle_right_click(state)

        if posture == 'SCROLL':
            self._handle_scroll(frame, dt, ctrl=False)
        elif posture == 'ZOOM':
            self._handle_scroll(frame, dt, ctrl=True)
        elif posture == 'MOVE' or self._button_down:
            # Keep driving the cursor while a button is held even if the
            # posture wobbles — a pinched hand often reads as NEUTRAL, and a
            # drag must not stall part-way through.
            self._be.moveTo(x, y)

        return x, y, self._label

    # --------------------------------------------------------------- clicking

    def _cursor_damping(self, frame):
        """Slow the cursor to a stop as a pinch closes.

        Even with a palm-anchored cursor, reaching the thumb across shifts the
        whole hand slightly. Freezing only at the moment the click fires is too
        late — the drift has already happened during the approach. So the
        cursor's gain fades to zero across the gap between `pinch_damp_start`
        and `pinch_on`, and is fully stopped before the click lands.

        A drag is exempt: it needs the cursor to keep following the hand.
        """
        if self._drag_mode:
            return 1.0
        distance = min(frame.index_pinch_distance, frame.middle_pinch_distance)
        if distance >= self._damp_start:
            return 1.0
        span = self._damp_start - self._pinch_on
        if span <= 0 or distance <= self._pinch_on:
            return 0.0
        return (distance - self._pinch_on) / span

    def _handle_click(self, state, frame, now, x, y):
        """Press on pinch, release on unpinch. A click is those two events.

        The earlier design fired `click()` on release, and had to decide
        retroactively whether the pinch had been a click or a drag. That made
        every click depend on three fragile things at once: detecting the
        release, doing it inside a time limit, and not being interrupted. Hold
        a beat too long and you got a drag; a single dropped tracking frame and
        the click vanished entirely.

        Pressing down immediately is both instant and impossible to lose: the
        button is down because you pinched, and it comes up when you let go.
        Drag then falls out of the same two events for free, and a quick second
        pinch lands inside the OS's own double-click window without us needing
        to time anything.
        """
        if state.index_pinch and not self._button_down:
            self._button_down = True
            self._drag_mode = False
            self._pinch_origin = (frame.pointer_x, frame.pointer_y)
            # Pin the cursor, then press exactly where the user was aiming.
            # The coordinates are passed to mouseDown as well as moveTo: if the
            # move were dropped or raced, the press would otherwise land
            # wherever the cursor happened to be.
            self._mapper.freeze()
            self._be.moveTo(x, y)
            # Give Windows a moment to settle the new cursor position before
            # the button event. The original version of this app got this for
            # free from pyautogui's default 0.1 s PAUSE; we set PAUSE = 0 for
            # responsiveness, and without a settle here a press can land before
            # the target has seen the cursor arrive — a click that visibly
            # does nothing.
            if self._press_delay:
                time.sleep(self._press_delay)
            self._be.mouseDown(x, y)
            self._label = 'CLICK'
            return

        if state.index_pinch and self._button_down:
            if not self._drag_mode:
                moved = _dist(frame.pointer_x, frame.pointer_y,
                              *self._pinch_origin) / max(frame.scale, 1e-6)
                # Moving a real distance with the button held means a drag, so
                # let the cursor loose again. Until then it stays pinned, which
                # is what keeps a stationary click on target.
                if moved > self._drag_slop:
                    self._drag_mode = True
                    self._mapper.unfreeze()
            self._label = 'DRAG' if self._drag_mode else 'CLICK'
            return

        if not state.index_pinch and self._button_down:
            self._button_down = False
            self._drag_mode = False
            self._be.mouseUp(x, y)
            # Stay pinned a moment longer: fingers spring apart as they
            # release, and letting the cursor chase that would spoil your aim
            # right when you are about to click again.
            self._mapper.freeze()
            self._settle_until = now + self._settle_s

    def _handle_right_click(self, state):
        """Thumb tapped to the middle fingertip. Fires once per tap."""
        if state.middle_pinch and not state.index_pinch:
            if not self._right_held:
                self._right_held = True
                self._be.rightClick(*self._mapper.position)
                self._label = 'RIGHT_CLICK'
        elif not state.middle_pinch:
            self._right_held = False

    # ---------------------------------------------------------------- scroll

    def _handle_scroll(self, frame, dt, ctrl):
        """Displacement-driven scrolling: how far you hold your hand from the
        anchor sets the *speed*, not a one-off jump.

        The response is quadratic, so a small offset creeps along for precise
        positioning while a larger one moves at a genuinely useful pace.
        """
        if self._anchor_y is None or dt <= 0:
            return

        deadzone = self._zoom_deadzone if ctrl else self._scroll_deadzone
        gain = self._zoom_gain if ctrl else self._scroll_gain

        # In palm-lengths, so the same physical hand movement scrolls the same
        # amount regardless of how close to the camera you are.
        offset = (frame.pointer_y - self._anchor_y) / max(frame.scale, 1e-6)
        if abs(offset) <= deadzone:
            return

        excess = abs(offset) - deadzone
        # y grows downward in image space, so a hand below the anchor should
        # scroll the page down, i.e. a negative wheel delta.
        direction = -1.0 if offset > 0 else 1.0
        amount = direction * gain * excess * excess * dt

        self._scroll_residue += amount
        clicks = int(self._scroll_residue)
        if clicks == 0:
            # Keep the fraction: truncating every frame would make slow
            # scrolling emit nothing at all, forever.
            return
        self._scroll_residue -= clicks

        if ctrl:
            self._press_ctrl()
        self._be.scroll(clicks)

    def _end_scroll(self):
        self._anchor_y = None
        self._scroll_residue = 0.0
        self._mapper.unfreeze()

    def _press_ctrl(self):
        if not self._ctrl_down:
            self._be.keyDown('ctrl')
            self._ctrl_down = True

    def _release_ctrl(self):
        if self._ctrl_down:
            self._be.keyUp('ctrl')
            self._ctrl_down = False

    # ---------------------------------------------------------------- safety

    def release_all(self):
        """Let go of everything we might be holding.

        Called on pause, on losing the hand, and on shutdown. Without it, a
        hand that vanishes mid-drag would leave the mouse button stuck down
        and make the desktop unusable.
        """
        if self._button_down:
            self._button_down = False
            self._drag_mode = False
            self._be.mouseUp()
        self._release_ctrl()
        self._right_held = False
        self._anchor_y = None
        self._scroll_residue = 0.0
        self._settle_until = 0.0
        self._mapper.unfreeze()
        self._label = 'NEUTRAL'
