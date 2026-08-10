"""Threaded webcam capture that always hands back the newest frame.

OpenCV buffers frames internally, so a `cap.read()` in a loop that can't keep
up returns progressively older images — the hand tracking ends up reacting to
where your hand was two or three frames ago. No amount of filtering downstream
can undo that; the only fix is to keep draining the camera on its own thread
and throw away everything but the latest frame.
"""

import threading

import cv2 as cv

# How many device indices to probe when the configured one doesn't open.
_MAX_PROBE_INDEX = 4


class CameraError(RuntimeError):
    """No usable webcam. Raised loudly — the old code failed silently."""


class Camera:
    def __init__(self, index=0, width=None, height=None):
        self._cap = self._open(index, width, height)
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    @staticmethod
    def _open(index, width, height):
        """Try the requested index first, then fall back to scanning."""
        candidates = [index] + [i for i in range(_MAX_PROBE_INDEX) if i != index]
        for i in candidates:
            cap = cv.VideoCapture(i)
            if cap.isOpened():
                if width:
                    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
                if height:
                    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
                # Ask the driver for a shallow buffer too; not all backends
                # honour it, which is why the drain thread still exists.
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        raise CameraError(
            f"Could not open a webcam (tried index {index}, then 0-{_MAX_PROBE_INDEX - 1}). "
            "Check that no other app is using the camera and that camera "
            "permissions are enabled."
        )

    def _pump(self):
        """Continuously read, keeping only the most recent frame."""
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        """Latest frame, or None if none has arrived yet."""
        with self._lock:
            return self._frame

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()
