"""PuppetAI — control your computer with your hand.

Entry point. Deliberately thin: it wires the pieces together and owns the
threading, while every decision about what a hand is doing lives in
`puppet_functions/`.

    camera -> MediaPipe -> HandModel -> GestureRecognizer -> GestureActions
                                                  \\-> PointOverlay

Tk insists on owning the main thread, so the overlay runs there and the vision
pipeline runs on a worker.
"""

import sys
import threading
import time

import cv2 as cv
import mediapipe as mp
import pyautogui

from puppet_functions.camera import Camera, CameraError
from puppet_functions.config_manager import ConfigManager
from puppet_functions.filters import EMA
from puppet_functions.gesture_actions import GestureActions
from puppet_functions.gesture_recognizer import GestureRecognizer
from puppet_functions.hand_model import HandModel
from puppet_functions.point_overlay import PointOverlay
from puppet_functions.pointer_mapper import PointerMapper

try:
    import pystray
    from PIL import Image, ImageDraw
    _HAS_TRAY = True
except ImportError:
    _HAS_TRAY = False

# Re-read config.json at most this often; cheap, but no need to stat every frame.
_CONFIG_POLL_S = 1.0


def _tray_icon_image():
    img = Image.new('RGB', (64, 64), 'black')
    ImageDraw.Draw(img).ellipse([8, 8, 56, 56], fill='#e63946')
    return img


class Pipeline:
    """Owns the vision loop and the state shared with the UI thread."""

    def __init__(self, config, overlay, screen_w, screen_h):
        self.config = config
        self.overlay = overlay
        self.paused = False
        self.stop_event = threading.Event()

        self._screen = (screen_w, screen_h)
        self._build()
        self._fps = EMA(0.2)
        self._last_config_poll = 0.0

    def _build(self):
        """(Re)create everything that reads config, so a hot reload takes effect."""
        self.hand_model = HandModel(self.config)
        self.recognizer = GestureRecognizer(self.config.gesture_dwell_ms)
        self.mapper = PointerMapper(*self._screen, self.config)
        self.actions = GestureActions(self.config, self.mapper)

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            # Never leave a button or modifier held when input stops.
            self.actions.release_all()
        return self.paused

    def stop(self):
        self.stop_event.set()

    def run(self):
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=self.config.detection_confidence,
            min_tracking_confidence=self.config.tracking_confidence,
        )

        try:
            cam = Camera(self.config.camera_index,
                         self.config.camera_width,
                         self.config.camera_height)
        except CameraError as exc:
            print(f"[puppet] {exc}", file=sys.stderr)
            self.stop_event.set()
            self.overlay.close()
            return

        hand_present = False
        last_t = time.time()
        last_seen = 0.0
        # MediaPipe drops the odd frame, especially mid-pinch when the hand
        # self-occludes. Tearing down state on the first missing frame would
        # release the mouse button in the middle of a click.
        grace_s = self.config.hand_lost_grace_ms / 1000.0

        try:
            while not self.stop_event.is_set():
                frame = cam.read()
                if frame is None:
                    time.sleep(0.005)   # camera thread hasn't delivered yet
                    continue

                now = time.time()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    self._fps.update(1.0 / dt)

                self._poll_config(now)

                # Mirror the image so moving your hand right moves the cursor
                # right; MediaPipe wants RGB, OpenCV gives BGR.
                frame = cv.flip(frame, 1)
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)

                landmarks = None
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0]

                if landmarks is not None and not self.paused:
                    last_seen = now
                    if not hand_present:
                        # Fresh acquisition: snap the filters to the hand's
                        # actual position instead of gliding from wherever it
                        # was before it disappeared.
                        self.mapper.reset()
                        self.recognizer.reset()
                        self.hand_model.reset()
                        hand_present = True

                    hand = self.hand_model.update(landmarks.landmark)
                    state = self.recognizer.update(hand, now)
                    x, y, label = self.actions.execute(state, hand, now)
                    self.overlay.update_state(label, x, y, hand.pinch_strength,
                                              True, False, self._fps.value or 0.0,
                                              self._diagnostics(hand, state))
                else:
                    # Pausing is deliberate, so act on it at once. A missing
                    # hand only counts once it has been missing a while.
                    if hand_present and (self.paused or now - last_seen > grace_s):
                        self.actions.release_all()
                        self.recognizer.reset()
                        self.hand_model.reset()
                        hand_present = False
                    px, py = self.mapper.position
                    # During the grace period `hand_present` is still true, so
                    # a one-frame dropout doesn't flash "no hand" on the HUD.
                    self.overlay.update_state('NEUTRAL', px, py, 0.0,
                                              hand_present, self.paused,
                                              self._fps.value or 0.0)

                if self.config.show_debug_feed:
                    if landmarks is not None:
                        mp_draw.draw_landmarks(
                            frame, landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style())
                    cv.imshow('PuppetAI — camera', frame)
                    self._handle_keys()
                    self._feed_open = True
                elif getattr(self, '_feed_open', False):
                    # Debug feed was switched off via a config hot reload.
                    cv.destroyAllWindows()
                    self._feed_open = False
        finally:
            self.actions.release_all()
            cam.release()
            cv.destroyAllWindows()
            hands.close()
            self.overlay.close()

    def _diagnostics(self, hand, state):
        """Live threshold readout, so the click can be checked against the
        user's actual hand instead of assumed proportions.

        `pinch` must fall below pinch_on to click; `reach` must stay above
        pinch_reach_on. If a click isn't firing, this says which one is at
        fault — and both are adjustable in config.json while running.
        """
        if not self.config.show_diagnostics:
            return None
        return (f'pinch {hand.index_pinch_distance:4.2f} (need <{self.config.pinch_on:.2f})   '
                f'reach {hand.index_reach:4.2f} (need >{self.config.pinch_reach_on:.2f})   '
                f'click {"DOWN" if state.index_pinch else "up"}')

    def _handle_keys(self):
        """Keyboard shortcuts, read from the OpenCV debug window."""
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stop_event.set()
        elif key == ord('g'):
            self.overlay.toggle_guide()
        elif key == ord('p'):
            self.toggle_pause()

    def _poll_config(self, now):
        """Pick up edits to config.json without a restart."""
        if now - self._last_config_poll < _CONFIG_POLL_S:
            return
        self._last_config_poll = now
        if self.config.maybe_reload():
            print("[config] reloaded")
            self.actions.release_all()
            self._build()


def main():
    # A gesture can legitimately fling the cursor into a corner; the default
    # failsafe would abort the app. PAUSE=0 removes pyautogui's 100 ms delay
    # after every call, which would otherwise cap us at 10 actions a second.
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0

    config = ConfigManager()
    screen_w, screen_h = pyautogui.size()
    overlay = PointOverlay(dot_radius=config.overlay_dot_radius)
    if not config.show_overlay:
        # Tk still owns the main loop; we just hide the window.
        overlay.root.withdraw()
    pipeline = Pipeline(config, overlay, screen_w, screen_h)

    threading.Thread(target=pipeline.run, daemon=True).start()

    tray = None
    if _HAS_TRAY:
        def on_pause(icon, item):
            pipeline.toggle_pause()

        def on_guide(icon, item):
            overlay.toggle_guide()

        def on_quit(icon, item):
            pipeline.stop()
            icon.stop()

        tray = pystray.Icon('PuppetAI', _tray_icon_image(), 'PuppetAI', pystray.Menu(
            pystray.MenuItem(lambda item: 'Resume' if pipeline.paused else 'Pause', on_pause),
            pystray.MenuItem('Show guide', on_guide),
            pystray.MenuItem('Quit', on_quit),
        ))
        tray.run_detached()

    try:
        overlay.run()
    finally:
        pipeline.stop()
        if tray is not None:
            tray.stop()


if __name__ == "__main__":
    main()
