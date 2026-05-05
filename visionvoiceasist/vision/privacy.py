"""PII (Personally Identifiable Information) redaction.

Blurs human faces and inferred licence-plate regions in a frame *before*
any image is transmitted to a cloud API.  This is a GDPR Article 25
(data-protection-by-design) measure: only de-identified images leave the
device.

Detection pipeline:
  1. Haar-cascade face detector  — OpenCV built-in, no model download.
  2. Heuristic plate locator     — contour + aspect-ratio filter on the
                                   lower 40 % of the frame, where plates
                                   most commonly appear.

When cv2 is unavailable (lightweight environments) the class degrades
gracefully to a no-op, returning the original frame unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

try:
    import cv2  # type: ignore[import-untyped]

    _CV2_OK = True
except ImportError:  # pragma: no cover
    _CV2_OK = False


@dataclass(frozen=True)
class RedactionStats:
    """Summary of PII regions found and blurred in a single frame."""

    faces_blurred: int
    plates_blurred: int

    @property
    def total(self) -> int:
        return self.faces_blurred + self.plates_blurred


_EMPTY = RedactionStats(0, 0)


class PrivacyFilter:
    """Blur detected faces and probable licence plates in a BGR frame.

    Args:
        blur_ksize: Gaussian kernel side length (must be a positive odd
                    integer).  Larger values produce stronger blurring.
                    Even values are rounded up to the next odd integer.
    """

    _HAAR = "haarcascade_frontalface_default.xml"

    def __init__(self, blur_ksize: int = 31) -> None:
        self._k: int = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self._cascade: Any = None

        if _CV2_OK:
            path: str = cv2.data.haarcascades + self._HAAR
            cc: Any = cv2.CascadeClassifier(path)
            if not cc.empty():
                self._cascade = cc
                log.debug("PrivacyFilter: Haar cascade loaded")
            else:
                log.warning("PrivacyFilter: cascade missing — face blurring disabled")

    # ── Public API ────────────────────────────────────────────────────────────

    def redact(self, frame: np.ndarray) -> tuple[np.ndarray, RedactionStats]:
        """Return *(redacted_frame, stats)*.

        The returned frame is always a copy; the original is never mutated.
        If cv2 is unavailable or *frame* is empty, returns the original
        array unchanged (zero-copy) with zero stats.
        """
        if not _CV2_OK or frame.size == 0:
            return frame, _EMPTY

        out = frame.copy()
        n_faces = self._blur_faces(out)
        n_plates = self._blur_plate_candidates(out)
        return out, RedactionStats(n_faces, n_plates)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _blur_region(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        """Gaussian-blur a rectangular ROI in-place."""
        roi = img[y : y + h, x : x + w]
        img[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (self._k, self._k), 0)

    def _blur_faces(self, frame: np.ndarray) -> int:
        if self._cascade is None:
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if not isinstance(faces, np.ndarray) or len(faces) == 0:
            return 0
        for x, y, w, h in faces:
            self._blur_region(frame, int(x), int(y), int(w), int(h))
        return int(len(faces))

    def _blur_plate_candidates(self, frame: np.ndarray) -> int:
        """Contour-based heuristic for rectangular plate-shaped regions.

        Typical European/AZ plate: ~520 × 110 mm → aspect ≈ 4.7.
        Standard US plate: ~305 × 152 mm → aspect ≈ 2.0.
        We scan the lower 40 % of the frame where plates usually appear.
        """
        fh, _fw = frame.shape[:2]
        scan_y = int(fh * 0.60)
        roi = frame[scan_y:, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = cw / ch if ch > 0 else 0.0
            if 1.8 <= aspect <= 5.5 and 60 <= cw <= 350 and 15 <= ch <= 90:
                self._blur_region(roi, x, y, cw, ch)
                count += 1
        return count
