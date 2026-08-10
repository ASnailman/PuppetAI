"""Shared test fixtures — most importantly, a synthetic hand builder.

`make_hand()` produces the same 21 landmarks MediaPipe would, for a pose you
describe in words. That's what lets the whole gesture stack be tested without a
webcam, and lets us assert the property that matters most: the same pose,
scaled or rotated, must be read the same way.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puppet_functions.config_manager import ConfigManager  # noqa: E402
from puppet_functions.hand_model import HandModel  # noqa: E402


class LM:
    """Stands in for a MediaPipe NormalizedLandmark."""

    __slots__ = ('x', 'y', 'z')

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


# Canonical hand, in "palm units": wrist at the origin, fingers pointing up
# (-y, because image coordinates grow downward), palm length exactly 1.0.
_WRIST = (0.0, 0.0)
_MCP = {
    'index':  (-0.35, -0.95),
    'middle': (0.0, -1.0),
    'ring':   (0.30, -0.95),
    'pinky':  (0.60, -0.85),
}
# Joint positions along the finger's own direction, as a multiple of palm length.
_EXTENDED = {'pip': 0.45, 'dip': 0.75, 'tip': 1.00}
# A curled finger is built with the same PIP-bend model as a pinch, just bent
# much further — 140 degrees, which folds the fingertip back down against the
# palm the way a real fist does. Modelling it as "a short straight finger"
# left the tip much too far from the wrist and made the fist look far more
# distinguishable from a pinch than it really is.
_FIST_CURL = 140.0 / 90.0

_THUMB_CMC = (-0.15, -0.20)
_THUMB_DIRS = {
    'out': (-0.707, -0.707),   # splayed away from the palm
    'in':  (0.20, -0.98),      # tucked alongside the fingers
}
_THUMB_JOINTS = {'mcp': 0.35, 'ip': 0.60, 'tip': 0.85}


def _unit(v):
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n)


def _rotate(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def make_hand(index=True, middle=False, ring=False, pinky=False,
              thumb='in', pinch=None, scale=0.15, rotation=0.0,
              center=(0.5, 0.5), curl=0.0):
    """Build 21 landmarks for a described pose.

    index/middle/ring/pinky : True = extended, False = curled
    thumb                   : 'in' (tucked) or 'out' (splayed)
    pinch                   : None, 'index' or 'middle' — puts the thumb tip
                              onto that fingertip, overriding `thumb`
    curl                    : 0..1, how far the *pinched* finger bends in to
                              meet the thumb. Real pinches are not made with a
                              ramrod-straight finger, and pretending they are
                              hid a bug where the click was gated on the
                              finger still counting as extended.
    scale                   : palm length in normalised image units
    rotation                : radians; the whole hand is rotated about the wrist
    """
    pts = [None] * 21
    pts[0] = _WRIST

    fingers = {'index': index, 'middle': middle, 'ring': ring, 'pinky': pinky}
    slots = {'index': (5, 6, 7, 8), 'middle': (9, 10, 11, 12),
             'ring': (13, 14, 15, 16), 'pinky': (17, 18, 19, 20)}

    for name, extended in fingers.items():
        mcp = _MCP[name]
        d = _unit(mcp)                      # fingers point away from the wrist
        i_mcp, i_pip, i_dip, i_tip = slots[name]
        pts[i_mcp] = mcp

        # Fingers bend at the PIP joint. `bend` is 0..1 mapped to 0..90
        # degrees, so a pinch (~0.7) and a fist (~1.55) are the same model at
        # different angles — which is what makes the two poses genuinely hard
        # to tell apart, as they are in real life.
        bend = None
        if extended:
            if curl and name == (pinch or 'index'):
                bend = curl
        else:
            bend = _FIST_CURL

        if bend is None:
            for idx, key in ((i_pip, 'pip'), (i_dip, 'dip'), (i_tip, 'tip')):
                pts[idx] = (mcp[0] + d[0] * _EXTENDED[key],
                            mcp[1] + d[1] * _EXTENDED[key])
            continue

        angle = bend * math.pi / 2
        pip = (mcp[0] + d[0] * 0.45, mcp[1] + d[1] * 0.45)
        d1 = _rotate(d, angle)
        dip = (pip[0] + d1[0] * 0.30, pip[1] + d1[1] * 0.30)
        d2 = _rotate(d, angle * 1.5)
        tip = (dip[0] + d2[0] * 0.25, dip[1] + d2[1] * 0.25)
        pts[i_pip], pts[i_dip], pts[i_tip] = pip, dip, tip

    d = _unit(_THUMB_DIRS[thumb])
    for idx, key in ((2, 'mcp'), (3, 'ip'), (4, 'tip')):
        pts[idx] = (_THUMB_CMC[0] + d[0] * _THUMB_JOINTS[key],
                    _THUMB_CMC[1] + d[1] * _THUMB_JOINTS[key])
    pts[1] = _THUMB_CMC

    if pinch is not None:
        # Park the thumb tip a fingertip's width from the target tip — real
        # fingers have thickness, so a "touching" pinch is never zero.
        target = pts[8 if pinch == 'index' else 12]
        pts[4] = (target[0] + 0.12, target[1] + 0.10)

    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    out = []
    for (x, y) in pts:
        rx = x * cos_r - y * sin_r
        ry = x * sin_r + y * cos_r
        out.append(LM(center[0] + rx * scale, center[1] + ry * scale))
    return out


# Convenience poses matching the documented gesture set.
def pose_move(**kw):        return make_hand(index=True, **kw)
def pose_click(**kw):       return make_hand(index=True, pinch='index', **kw)
def pose_scroll(**kw):      return make_hand(index=True, middle=True, thumb='in', **kw)
def pose_zoom(**kw):        return make_hand(index=True, middle=True, thumb='out', **kw)
def pose_right_click(**kw): return make_hand(index=True, middle=True, pinch='middle', **kw)
def pose_fist(**kw):        return make_hand(index=False, **kw)
def pose_open_palm(**kw):   return make_hand(index=True, middle=True, ring=True,
                                             pinky=True, thumb='out', **kw)


@pytest.fixture
def config():
    """Defaults only — never reads the user's real config.json."""
    return ConfigManager(path=os.path.join(os.path.dirname(__file__), '_no_such_config.json'))


@pytest.fixture
def model(config):
    return HandModel(config)


class FakeBackend:
    """Records what would have been sent to the real mouse/keyboard."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args))
        return record

    def names(self):
        return [name for name, _ in self.calls]

    def count(self, name):
        return sum(1 for n, _ in self.calls if n == name)


@pytest.fixture
def backend():
    return FakeBackend()
