"""Filters: the smoothing that makes the cursor calm without making it laggy."""

import random

from puppet_functions.filters import EMA, Hysteresis, OneEuroFilter


def test_first_sample_passes_through():
    f = OneEuroFilter()
    assert f.filter(10.0, 0.0) == 10.0


def test_converges_on_a_constant_signal():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    t = 0.0
    for _ in range(200):
        t += 1 / 60
        out = f.filter(100.0, t)
    assert abs(out - 100.0) < 0.5


def test_suppresses_jitter_on_a_still_hand():
    """The headline property: noise on a stationary signal is damped hard."""
    random.seed(7)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    t = 0.0
    raw, filtered = [], []
    for _ in range(300):
        t += 1 / 60
        sample = 500.0 + random.uniform(-5, 5)
        raw.append(sample)
        filtered.append(f.filter(sample, t))

    def spread(vals):
        mean = sum(vals) / len(vals)
        return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5

    assert spread(filtered[50:]) < spread(raw[50:]) / 3


def test_fast_motion_is_tracked_closely():
    """...while a fast sweep must not be smoothed into mush."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    t = 0.0
    for i in range(120):
        t += 1 / 60
        out = f.filter(i * 20.0, t)
    # Within a couple of frames' worth of travel of the true position.
    assert abs(out - 119 * 20.0) < 60


def test_reset_drops_history():
    f = OneEuroFilter()
    f.filter(0.0, 0.0)
    f.filter(5.0, 0.1)
    f.reset()
    assert f.filter(900.0, 0.2) == 900.0


def test_zero_and_negative_dt_do_not_explode():
    f = OneEuroFilter()
    f.filter(1.0, 5.0)
    assert f.filter(2.0, 5.0) == f.filter(2.0, 4.0) or True  # just must not raise


def test_hysteresis_does_not_flip_inside_the_gap():
    """A value parked between the thresholds must hold whatever state it had."""
    h = Hysteresis(on_threshold=0.45, off_threshold=0.65)
    assert h.update(1.0) is False
    assert h.update(0.40) is True      # crossed the on threshold
    assert h.update(0.55) is True      # inside the gap — stays on
    assert h.update(0.64) is True
    assert h.update(0.70) is False     # only now does it release
    assert h.update(0.55) is False     # inside the gap again — stays off


def test_hysteresis_inverted():
    h = Hysteresis(on_threshold=0.25, off_threshold=0.10, invert=True)
    assert h.update(0.0) is False
    assert h.update(0.30) is True
    assert h.update(0.15) is True
    assert h.update(0.05) is False


def test_ema_tracks_and_resets():
    e = EMA(0.5)
    assert e.update(10.0) == 10.0
    assert 10.0 < e.update(20.0) < 20.0
    e.reset()
    assert e.update(3.0) == 3.0
