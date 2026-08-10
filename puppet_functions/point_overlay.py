"""Transparent always-on-top overlay: the app's entire user interface.

It draws a dot where PuppetAI thinks you're pointing, a ring that closes as
your pinch tightens, and a small status readout. The ring matters more than it
sounds — being able to see how close you are to a click removes the guesswork
that made the old click feel unpredictable.

All public methods are safe to call from the capture thread; they marshal onto
Tk's own loop via `after`.
"""

import tkinter as tk

# Dot colour per state. Distinct hues make the current mode obvious at a glance
# without having to read the label.
_COLORS = {
    'MOVE':        '#ff4d4d',
    'CLICK':       '#3ddc84',
    'DRAG':        '#3ddc84',
    'RIGHT_CLICK': '#ffa726',
    'SCROLL':      '#42a5f5',
    'ZOOM':        '#ffee58',
    'NEUTRAL':     '#9e9e9e',
}

_GUIDE = (
    "  PUPPET AI — GESTURE GUIDE\n"
    "  ─────────────────────────────────────\n"
    "  MOVE         point with your index finger\n"
    "  CLICK        tap thumb to index fingertip\n"
    "  DOUBLE       tap it twice, quickly\n"
    "  DRAG         hold the pinch and move\n"
    "  RIGHT CLICK  tap thumb to middle fingertip\n"
    "  SCROLL       index + middle up, move hand\n"
    "               up or down from where you started\n"
    "  ZOOM         same, but with the thumb out\n"
    "  REST         open your hand or make a fist\n"
    "  ─────────────────────────────────────\n"
    "  G  hide this guide     Q  quit  (on the video window)\n"
)

# Rendering tick. Faster than the camera on purpose: the dot is interpolated
# between tracking updates so it glides instead of stepping at 30 fps.
_TICK_MS = 16
_GLIDE = 0.5   # fraction of the remaining distance covered each tick


class PointOverlay:
    def __init__(self, dot_radius=9.0):
        self.root = tk.Tk()
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'white')
        self.root.overrideredirect(True)

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")

        self._radius = dot_radius
        self._guide_visible = False

        # Where the dot is drawn now vs. where tracking says it should be.
        self._x = self._tx = sw / 2
        self._y = self._ty = sh / 2
        self._label = 'NEUTRAL'
        self._strength = 0.0
        self._hand = False
        self._paused = False
        self._fps = 0.0
        self._diagnostics = None

        self.canvas = tk.Canvas(self.root, width=sw, height=sh, bg='white',
                                highlightthickness=0)
        self.canvas.pack()

        r = dot_radius
        # Ring first so the dot paints on top of it.
        self.ring = self.canvas.create_oval(0, 0, r, r, outline='#3ddc84',
                                            width=2, state='hidden')
        self.dot = self.canvas.create_oval(0, 0, r * 2, r * 2,
                                           fill=_COLORS['NEUTRAL'], outline='')

        self.label_bg = self.canvas.create_rectangle(10, 10, 150, 38,
                                                     fill='black', outline='')
        self.label = self.canvas.create_text(20, 24, text='NEUTRAL', anchor='w',
                                             fill='white',
                                             font=('Segoe UI', 11, 'bold'))

        self.guide_bg = self.canvas.create_rectangle(10, 50, 620, 400,
                                                     fill='black', outline='',
                                                     state='hidden')
        self.guide = self.canvas.create_text(26, 66, text=_GUIDE, anchor='nw',
                                             fill='white', font=('Consolas', 12),
                                             state='hidden')

        self.root.bind('<g>', self._toggle_guide)
        self.root.bind('<G>', self._toggle_guide)

        self.root.after(_TICK_MS, self._tick)

    # ------------------------------------------------------- thread-safe API

    def update_state(self, label, x, y, strength=0.0, hand=True,
                     paused=False, fps=0.0, diagnostics=None):
        """Called from the capture thread once per processed frame."""
        self.root.after(0, self._apply_state, label, x, y, strength, hand,
                        paused, fps, diagnostics)

    def toggle_guide(self):
        self.root.after(0, self._toggle_guide)

    def close(self):
        self.root.after(0, self.root.destroy)

    # ---------------------------------------------------------- Tk-side only

    def _apply_state(self, label, x, y, strength, hand, paused, fps, diagnostics):
        self._tx, self._ty = x, y
        self._label = label
        self._strength = strength
        self._hand = hand
        self._paused = paused
        self._fps = fps
        self._diagnostics = diagnostics

    def _tick(self):
        """Glide the dot toward its target and repaint the status line."""
        # Snap the last pixel or two rather than easing into them forever: the
        # dot is what the user aims with, so it has to end up exactly where the
        # click will be sent, not fractionally short of it.
        if abs(self._tx - self._x) < 2 and abs(self._ty - self._y) < 2:
            self._x, self._y = self._tx, self._ty
        else:
            self._x += (self._tx - self._x) * _GLIDE
            self._y += (self._ty - self._y) * _GLIDE

        r = self._radius
        visible = self._hand and not self._paused
        self.canvas.itemconfigure(self.dot, state='normal' if visible else 'hidden')
        if visible:
            self.canvas.coords(self.dot, self._x - r, self._y - r,
                               self._x + r, self._y + r)
            self.canvas.itemconfigure(self.dot,
                                      fill=_COLORS.get(self._label, '#ff4d4d'))

            # The ring shrinks onto the dot as the pinch closes, so you can see
            # a click coming before it fires.
            if self._strength > 0.05:
                gap = r + 4 + (1.0 - self._strength) * 20
                self.canvas.coords(self.ring, self._x - gap, self._y - gap,
                                   self._x + gap, self._y + gap)
                self.canvas.itemconfigure(self.ring, state='normal',
                                          width=1 + 2 * self._strength)
            else:
                self.canvas.itemconfigure(self.ring, state='hidden')
        else:
            self.canvas.itemconfigure(self.ring, state='hidden')

        if self._paused:
            text = 'PAUSED'
        elif not self._hand:
            text = 'no hand   ·   show your hand to the camera'
        else:
            text = f'{self._label}   ·   {self._fps:.0f} fps'
        if self._diagnostics:
            text += f'\n{self._diagnostics}'
        self.canvas.itemconfigure(self.label, text=text)
        bbox = self.canvas.bbox(self.label)
        if bbox:
            self.canvas.coords(self.label_bg, bbox[0] - 10, bbox[1] - 6,
                               bbox[2] + 10, bbox[3] + 6)

        self.root.after(_TICK_MS, self._tick)

    def _toggle_guide(self, event=None):
        self._guide_visible = not self._guide_visible
        state = 'normal' if self._guide_visible else 'hidden'
        self.canvas.itemconfigure(self.guide_bg, state=state)
        self.canvas.itemconfigure(self.guide, state=state)
        if self._guide_visible:
            self.root.after(10, self._fit_guide_bg)

    def _fit_guide_bg(self):
        bbox = self.canvas.bbox(self.guide)
        if bbox:
            pad = 14
            self.canvas.coords(self.guide_bg, bbox[0] - pad, bbox[1] - pad,
                               bbox[2] + pad, bbox[3] + pad)

    def run(self):
        """Enter the Tk loop, first making the window click-through.

        WS_EX_TRANSPARENT tells Windows to skip this window during hit-testing,
        so the clicks PuppetAI generates land on the app underneath instead of
        on the overlay itself. Only OR the flag in — tkinter already set
        WS_EX_LAYERED via -transparentcolor, and re-adding it would clear the
        colour key and turn the overlay opaque.

        `winfo_id()` is NOT the window Windows hit-tests against: on Windows,
        Tk's root widget lives in a child HWND, wrapped in a separate top-level
        HWND that Tk manages internally. Applying the style to the child does
        nothing for hit-testing, since Windows resolves clicks against the
        top-level first — the overlay stayed opaque to mouse input, which is
        why it silently ate every hover and click on the whole desktop. Walk up
        to the real top-level with GetParent before setting the style.
        """
        try:
            import ctypes
            self.root.update()
            child_hwnd = self.root.winfo_id()
            hwnd = ctypes.windll.user32.GetParent(child_hwnd) or child_hwnd
            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                                                style | WS_EX_TRANSPARENT)
        except Exception:
            pass  # non-Windows, or the call failed — overlay still works
        self.root.mainloop()
