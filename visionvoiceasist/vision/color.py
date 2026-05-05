"""HSV-based colour analysis: pixel colour name and traffic-light state."""

from __future__ import annotations

import numpy as np

from ..i18n import Messages


class ColorAnalyzer:
    """Pure functions on HSV pixels / regions."""

    @staticmethod
    def from_hsv(h: int, s: int, v: int) -> str:
        """Return Azerbaijani colour name from HSV components (OpenCV ranges)."""
        if v < 42:
            return "qara"
        if s < 28 and v > 168:
            return "ağ"
        if s < 28:
            return "boz"
        if h < 10 or h > 165:
            return "qırmızı"
        if 10 <= h < 30:
            return "narıncı"
        if 30 <= h < 73:
            return "sarı"
        if 73 <= h < 95:
            return "açıq yaşıl"
        if 95 <= h < 130:
            return "göy"
        if 130 <= h < 150:
            return "mavi"
        if 150 <= h <= 165:
            return "bənövşəyi"
        return "rəngli"

    @staticmethod
    def traffic_light(hsv_roi: np.ndarray) -> str:
        """Classify a traffic-light ROI (HSV) and return Az message."""
        if hsv_roi.size == 0:
            return Messages.TRAFFIC_LIGHT_OFF
        avg_v = float(np.mean(hsv_roi[:, :, 2]))
        avg_h = float(np.mean(hsv_roi[:, :, 0]))
        if avg_v < 50:
            return Messages.TRAFFIC_LIGHT_OFF
        if avg_h < 15 or avg_h > 160:
            return Messages.TRAFFIC_LIGHT_RED
        if 40 < avg_h < 90:
            return Messages.TRAFFIC_LIGHT_GREEN
        if 20 < avg_h < 35:
            return Messages.TRAFFIC_LIGHT_YELLOW
        return Messages.TRAFFIC_LIGHT_OFF
