"""Tesseract OCR wrapper with multi-stage preprocessing."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

try:
    import pytesseract  # type: ignore[import-untyped]

    _PYTESS = True
except ImportError:  # pragma: no cover
    _PYTESS = False


class OcrModule:
    """Reads text using Tesseract with `aze+eng` language pack.

    Returns None if no text is detected or pytesseract is unavailable.
    """

    MIN_TEXT_LEN = 4

    def __init__(self, lang: str = "aze+eng", psm: int = 6) -> None:
        self._lang = lang
        self._psm = psm
        self._ok = _PYTESS
        if not self._ok:
            log.warning("pytesseract not installed; OCR disabled.")

    @property
    def is_ready(self) -> bool:
        return self._ok

    def read(self, frame: np.ndarray) -> Optional[str]:
        if not self._ok:
            return None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            text = pytesseract.image_to_string(
                thresh, lang=self._lang, config=f"--psm {self._psm}"
            ).strip()
        except Exception as exc:  # noqa: BLE001
            log.debug("OCR failure: %s", exc)
            return None
        return text if len(text) >= self.MIN_TEXT_LEN else None
