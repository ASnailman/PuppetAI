"""Signal filters shared across the pipeline.

Everything here is deliberately dependency-free and stateless-per-instance so it
can be unit tested without a webcam.
"""

import math


class OneEuroFilter:
    """1-Euro filter (Casiez et al., 2012) — adaptive low-pass smoothing.

    The problem with a fixed-alpha EMA is that one alpha has to serve two
    conflicting jobs: kill jitter while the hand is still, and stay responsive
    while it moves.  Pick a low alpha and the cursor lags; pick a high one and
    it shakes.

    The 1-Euro filter fixes this by making the cutoff frequency a function of
    the signal's own speed: slow movement -> low cutoff -> heavy smoothing;
    fast movement -> high cutoff -> almost no smoothing (and therefore no lag).
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        # min_cutoff: smoothing floor when the hand is perfectly still.
        #             Lower = steadier cursor, but slower to start moving.
        # beta:       how aggressively the cutoff opens up with speed.
        #             Higher = less lag when moving fast, slightly more jitter.
        # d_cutoff:   cutoff for the internal speed estimate; 1.0 is the
        #             reference implementation's default and rarely needs tuning.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        """Convert a cutoff frequency (Hz) + timestep into an EMA alpha."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, timestamp):
        """Feed one sample; returns the smoothed value."""
        x = float(x)

        # First sample (or post-reset): nothing to smooth against yet.
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = timestamp
            self._dx_prev = 0.0
            return x

        dt = timestamp - self._t_prev
        # Guard against duplicate/backwards timestamps from a stalled camera.
        if dt <= 0:
            dt = 1e-3
        self._t_prev = timestamp

        # Smooth the derivative first — a noisy speed estimate would make the
        # adaptive cutoff itself jittery.
        dx = (x - self._x_prev) / dt
        dx_hat = self._dx_prev + self._alpha(self.d_cutoff, dt) * (dx - self._dx_prev)
        self._dx_prev = dx_hat

        # Faster movement -> higher cutoff -> less smoothing.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self._x_prev + self._alpha(cutoff, dt) * (x - self._x_prev)
        self._x_prev = x_hat
        return x_hat

    def reset(self, x=None, timestamp=None):
        """Drop all history so the next sample is taken verbatim.

        Used when the hand is re-acquired: interpolating from where the hand
        *was* a second ago would sling the cursor across the screen.
        """
        self._x_prev = None if x is None else float(x)
        self._t_prev = timestamp
        self._dx_prev = 0.0

    @property
    def value(self):
        """Last emitted value, or None before the first sample."""
        return self._x_prev


class Hysteresis:
    """Schmitt trigger — a boolean that refuses to chatter.

    A plain `value < threshold` test flickers on and off when the input hovers
    at the threshold, which is exactly where a relaxed hand naturally sits.  A
    Schmitt trigger uses two thresholds with a gap between them, so once it has
    committed to a state the input has to travel a real distance to change it.

    Configured for "smaller value means on" (distances: pinch, gaps), or
    "larger value means on" (extension) via `invert`.
    """

    def __init__(self, on_threshold, off_threshold, invert=False, initial=False):
        # invert=False -> turns ON below `on_threshold`, OFF above `off_threshold`
        #                 (so off_threshold must be the larger number)
        # invert=True  -> turns ON above `on_threshold`, OFF below `off_threshold`
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.invert = bool(invert)
        self._state = bool(initial)

    def update(self, value):
        if self.invert:
            if self._state:
                if value < self.off_threshold:
                    self._state = False
            elif value > self.on_threshold:
                self._state = True
        else:
            if self._state:
                if value > self.off_threshold:
                    self._state = False
            elif value < self.on_threshold:
                self._state = True
        return self._state

    @property
    def state(self):
        return self._state

    def reset(self, state=False):
        self._state = bool(state)


class EMA:
    """Plain exponential moving average, for scalars that don't need 1-Euro
    (FPS readout, hand-scale smoothing)."""

    def __init__(self, alpha=0.2):
        self.alpha = float(alpha)
        self._value = None

    def update(self, x):
        x = float(x)
        if self._value is None:
            self._value = x
        else:
            self._value += self.alpha * (x - self._value)
        return self._value

    @property
    def value(self):
        return self._value

    def reset(self):
        self._value = None
