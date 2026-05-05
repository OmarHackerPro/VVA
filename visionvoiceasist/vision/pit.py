"""Pit / stair / threshold detector — classical CV, no ML.

Combines:
    1. Canny edge density on the lower 48 % of the frame.
    2. Dark-mask threshold (likely shadows = depth discontinuities).
    3. Frame-to-frame mean-intensity delta (sudden floor change).

Returns a (message, priority) tuple if any condition fires, else None.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from ..i18n import Messages
from ..settings import ThresholdSettings
from ..types import Priority

log = logging.getLogger(__name__)


class PitDetector:
    """Three-stage classical pit / stair / threshold detector."""

    def __init__(self, thresholds: ThresholdSettings) -> None:
        self._t = thresholds
        self._prev_mean: Optional[float] = None
        self._kernel_lg = np.ones((9, 9), np.uint8)
        self._kernel_sm = np.ones((5, 5), np.uint8)

    def detect(
        self, frame: np.ndarray
    ) -> Optional[tuple[str, Priority]]:
        h, _ = frame.shape[:2]
        roi_y = int(h * self._t.pit_floor_ratio)
        floor = frame[roi_y:, :]
        gray = cv2.cvtColor(floor, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (13, 13), 0)

        edges = cv2.Canny(blurred, 25, 90)
        edge_density = float(np.sum(edges > 0)) / max(1, edges.size)

        _, dark = cv2.threshold(blurred, self._t.pit_dark, 255, cv2.THRESH_BINARY_INV)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, self._kernel_lg)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, self._kernel_sm)

        combined = cv2.bitwise_or(dark, edges)
        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in cnts if cv2.contourArea(c) > self._t.pit_min_area]

        cur_mean = float(np.mean(blurred))
        result: Optional[tuple[str, Priority]] = None

        if big and edge_density > 0.12:
            result = (Messages.PIT_STAIRS, Priority.CRITICAL)
        elif big and len(big) >= 2:
            result = (Messages.PIT_OBSTACLE, Priority.HIGH)
        elif self._prev_mean is not None and abs(cur_mean - self._prev_mean) > 22:
            result = (Messages.PIT_THRESHOLD, Priority.HIGH)

        self._prev_mean = cur_mean
        return result

    def reset(self) -> None:
        self._prev_mean = None
