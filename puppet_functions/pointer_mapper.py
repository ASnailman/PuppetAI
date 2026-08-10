"""Maps a fingertip position in the camera frame to a screen pixel.

Three jobs, each fixing a specific complaint about how the cursor felt:

* **Active region.** The old code mapped the whole camera frame to the whole
  screen, which meant the screen's corners lived at the very edge of the
  camera's view — precisely where MediaPipe's tracking is worst, and where
  your hand is uncomfortable to hold.  Mapping a smaller, central box to the
  full screen means every pixel is reachable from a relaxed arm position.

* **1-Euro smoothing** instead of a fixed-alpha EMA, so the cursor is calm
  while your hand hovers and still keeps up when you throw it across the desk.

* **Freezing.** Pinching your thumb onto your index finger physically drags the
  index tip a little.  Without freezing, that tiny drag moves the cursor in the
  instant before the click lands, so clicks miss small targets — which is what
  forced such careful hand placement.  We latch the position at pinch-down.
"""

import math

from .filters import OneEuroFilter

# Time constant for the continuity offset to wash out, in seconds. Long enough
# that the correction is invisible, short enough that absolute aiming is back
# well before the next deliberate click.
_OFFSET_TAU = 0.35


def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


class PointerMapper:
    def __init__(self, screen_w, screen_h, config=None):
        get = (lambda k, d: getattr(config, k, d)) if config is not None else (lambda k, d: d)

        self._screen_w = screen_w
        self._screen_h = screen_h

        # [x0, y0, x1, y1] in normalised camera coordinates. The default 15%
        # inset keeps your hand well inside the frame at every screen corner.
        region = get('active_region', [0.15, 0.15, 0.85, 0.85])
        self._x0, self._y0, self._x1, self._y1 = (float(v) for v in region)
        # Degenerate regions would divide by zero; fall back to the full frame.
        if self._x1 - self._x0 < 1e-3 or self._y1 - self._y0 < 1e-3:
            self._x0, self._y0, self._x1, self._y1 = 0.0, 0.0, 1.0, 1.0

        min_cutoff = get('one_euro_min_cutoff', 1.2)
        beta = get('one_euro_beta', 0.02)
        self._fx = OneEuroFilter(min_cutoff, beta)
        self._fy = OneEuroFilter(min_cutoff, beta)

        # Kept as floats so partial (damped) movements don't quantise away.
        self._px = float(screen_w // 2)
        self._py = float(screen_h // 2)
        self._x = screen_w // 2
        self._y = screen_h // 2
        self._frozen = False
        self._snap = True
        self._ox = self._oy = 0.0     # continuity offset, see unfreeze()
        self._rebase = False
        self._hold_offset = False
        self._last_now = 0.0

    def map(self, nx, ny, now, damping=1.0):
        """Normalised camera coords -> screen pixels.

        `damping` scales how much of the new position is taken: 1.0 is normal
        tracking, 0.0 is a full stop. Callers ramp it down as a pinch closes,
        so the cursor settles *before* the click rather than lurching into it.

        While frozen the last position is returned unchanged, but the filters
        keep running, and `unfreeze` rebases so tracking resumes seamlessly.
        """
        # Stretch the active region across the full screen, then clamp so a
        # hand outside the region parks the cursor at the edge instead of
        # flying off it.
        rx = _clamp01((nx - self._x0) / (self._x1 - self._x0))
        ry = _clamp01((ny - self._y0) / (self._y1 - self._y0))

        # The filters are fed on EVERY frame, including frozen ones. Skipping
        # them while frozen left them holding stale state, so the first sample
        # after unfreezing arrived with a huge dt, the adaptive cutoff opened
        # right up, and the cursor teleported to wherever the hand had got to.
        tx = self._fx.filter(rx * self._screen_w, now)
        ty = self._fy.filter(ry * self._screen_h, now)

        # Advance the clock on every frame, frozen ones included. Otherwise the
        # first frame after a freeze sees a dt spanning the whole freeze and
        # decays the continuity offset away in one step — reintroducing the
        # jump the offset exists to prevent.
        dt = max(now - self._last_now, 0.0)
        self._last_now = now

        if self._snap:
            # First sample after a reset: take it whole, whatever the caller
            # asked for, or the cursor would crawl from its stale position.
            self._snap = False
            damping = 1.0

        if self._frozen:
            return self._x, self._y

        # First frame after an unfreeze: adopt whatever offset makes the cursor
        # carry on from exactly where it was pinned.
        if self._rebase:
            self._rebase = False
            self._ox = self._px - tx
            self._oy = self._py - ty

        # Offset keeps the cursor continuous across an unfreeze (see unfreeze).
        # It decays back to zero so absolute aiming is restored, unless a
        # button is held — a drag has to track the hand exactly.
        if not self._hold_offset and (self._ox or self._oy):
            keep = math.exp(-dt / _OFFSET_TAU)
            self._ox *= keep
            self._oy *= keep
            if abs(self._ox) < 0.5 and abs(self._oy) < 0.5:
                self._ox = self._oy = 0.0

        tx += self._ox
        ty += self._oy

        if damping >= 1.0:
            self._px, self._py = tx, ty
        elif damping > 0.0:
            self._px += damping * (tx - self._px)
            self._py += damping * (ty - self._py)
        # damping == 0: hold position entirely.

        # Guard against the filter overshooting a pixel past the edge.
        self._x = max(0, min(self._screen_w - 1, int(self._px)))
        self._y = max(0, min(self._screen_h - 1, int(self._py)))
        return self._x, self._y

    def freeze(self):
        """Pin the cursor where it is (used for the duration of a click)."""
        self._frozen = True

    def unfreeze(self):
        """Resume tracking from exactly where the cursor was pinned.

        Simply clearing the flag would snap the cursor to wherever the hand had
        wandered to during the freeze — after a click that meant a visible
        teleport, and the *next* click then landed somewhere the user never
        aimed at. Instead we record the gap between the pinned position and the
        hand's current one as an offset, which makes the resumption seamless
        and then decays away so absolute aiming still holds.
        """
        if self._frozen:
            self._frozen = False
            # The offset is computed on the next mapped frame rather than here,
            # so it cancels that frame's filtered value exactly. Doing it now
            # would leave one frame of hand movement unaccounted for.
            self._rebase = True

    def hold_offset(self, hold):
        """Stop the offset decaying. Used while a mouse button is held, so a
        drag follows the hand exactly instead of creeping toward absolute."""
        self._hold_offset = bool(hold)

    @property
    def frozen(self):
        return self._frozen

    @property
    def position(self):
        return self._x, self._y

    def reset(self, nx=None, ny=None, now=None):
        """Drop filter history so the next sample is taken at face value.

        Called when the hand is re-acquired: without this, the filter would
        interpolate from wherever the hand was before it vanished and sling the
        cursor across the screen.
        """
        self._frozen = False
        self._snap = True
        self._ox = self._oy = 0.0
        self._hold_offset = False
        self._fx.reset()
        self._fy.reset()
        if nx is not None and ny is not None and now is not None:
            self.map(nx, ny, now)

