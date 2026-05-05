"""Camera source with latest-frame-wins ring buffer.

A background thread continuously reads from the device; consumers always get
the *most recent* frame, never a stale buffered one. Solves the legacy bug
where YOLO inferred on 200ms-old frames during busy scenes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from ..settings import CameraSettings
from ..types import Frame

log = logging.getLogger(__name__)


class CameraOpenError(RuntimeError):
    """Raised when no backend can open the requested camera."""


class CameraSource:
    """Threaded camera reader with single-slot ring buffer.

    Usage:
        with CameraSource(settings) as cam:
            for _ in range(100):
                frame = cam.read(timeout_s=1.0)
                if frame is None:
                    break  # disconnected
    """

    _BACKENDS = (cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_MSMF)

    def __init__(self, settings: CameraSettings) -> None:
        self._s = settings
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[Frame] = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_read_ok: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def open(self) -> None:
        for backend in self._BACKENDS:
            cap = cv2.VideoCapture(self._s.index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._s.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._s.height)
                cap.set(cv2.CAP_PROP_FPS, self._s.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self._s.buffer_size)
                self._cap = cap
                log.info("Camera opened: index=%d backend=%d", self._s.index, backend)
                self._start_reader()
                return
        raise CameraOpenError(f"Could not open camera index {self._s.index}")

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Public API ────────────────────────────────────────────────────────
    def read(self, timeout_s: float = 1.0) -> Optional[Frame]:
        """Block until a new frame is available or *timeout_s* elapses."""
        if not self._new_frame.wait(timeout=timeout_s):
            return None
        with self._lock:
            self._new_frame.clear()
            return self._frame

    def is_alive(self) -> bool:
        """True if the reader thread captured a frame in the last 2 seconds."""
        return time.time() - self._last_read_ok < 2.0

    # ── Internal ──────────────────────────────────────────────────────────
    def _start_reader(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="CameraReader"
        )
        self._thread.start()

    def _reader_loop(self) -> None:
        assert self._cap is not None  # noqa: S101
        consecutive_fails = 0
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok or image is None:
                consecutive_fails += 1
                if consecutive_fails > 30:
                    log.error("Camera read failed 30× — device likely disconnected")
                    time.sleep(0.5)
                else:
                    time.sleep(0.05)
                continue
            consecutive_fails = 0
            self._last_read_ok = time.time()
            self._frame_id += 1
            frame = Frame(image=image, frame_id=self._frame_id, captured_at=self._last_read_ok)
            with self._lock:
                self._frame = frame
            self._new_frame.set()
