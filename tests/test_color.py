"""Tests for ColorAnalyzer — HSV colour classification and traffic light."""

from __future__ import annotations

import numpy as np
import pytest

from visionvoiceasist.i18n import Messages
from visionvoiceasist.vision.color import ColorAnalyzer


class TestFromHsv:
    """from_hsv uses OpenCV HSV ranges: H∈[0,179], S∈[0,255], V∈[0,255]."""

    def test_black(self) -> None:
        assert ColorAnalyzer.from_hsv(0, 0, 20) == "qara"

    def test_white(self) -> None:
        assert ColorAnalyzer.from_hsv(0, 10, 200) == "ağ"

    def test_gray(self) -> None:
        assert ColorAnalyzer.from_hsv(0, 20, 120) == "boz"

    def test_red_low_hue(self) -> None:
        assert ColorAnalyzer.from_hsv(5, 200, 180) == "qırmızı"

    def test_red_high_hue(self) -> None:
        assert ColorAnalyzer.from_hsv(170, 200, 180) == "qırmızı"

    def test_orange(self) -> None:
        assert ColorAnalyzer.from_hsv(20, 200, 180) == "narıncı"

    def test_yellow(self) -> None:
        assert ColorAnalyzer.from_hsv(50, 200, 180) == "sarı"

    def test_light_green(self) -> None:
        assert ColorAnalyzer.from_hsv(80, 200, 180) == "açıq yaşıl"

    def test_blue(self) -> None:
        assert ColorAnalyzer.from_hsv(110, 200, 180) == "göy"

    def test_navy(self) -> None:
        assert ColorAnalyzer.from_hsv(140, 200, 180) == "mavi"

    def test_purple(self) -> None:
        assert ColorAnalyzer.from_hsv(155, 200, 180) == "bənövşəyi"

    def test_boundary_black_v42(self) -> None:
        # V == 42 is not < 42, so should NOT be black
        result = ColorAnalyzer.from_hsv(0, 0, 42)
        assert result != "qara"

    def test_boundary_white_v168(self) -> None:
        # s < 28 and v > 168: v=169 counts
        assert ColorAnalyzer.from_hsv(0, 10, 169) == "ağ"


class TestTrafficLight:
    def _roi(self, h: float, s: float, v: float) -> np.ndarray:
        """Create a 5×5 HSV ROI filled with a single pixel value."""
        roi = np.full((5, 5, 3), [h, s, v], dtype=np.float32)
        return roi

    def test_empty_roi_returns_off(self) -> None:
        roi = np.zeros((0, 0, 3), dtype=np.float32)
        assert ColorAnalyzer.traffic_light(roi) == Messages.TRAFFIC_LIGHT_OFF

    def test_dark_roi_returns_off(self) -> None:
        assert ColorAnalyzer.traffic_light(self._roi(0, 200, 20)) == Messages.TRAFFIC_LIGHT_OFF

    def test_red_light(self) -> None:
        # h near 0 (red), high V
        assert ColorAnalyzer.traffic_light(self._roi(5, 200, 180)) == Messages.TRAFFIC_LIGHT_RED

    def test_green_light(self) -> None:
        # h ~70 (green), high V
        assert ColorAnalyzer.traffic_light(self._roi(70, 200, 180)) == Messages.TRAFFIC_LIGHT_GREEN

    def test_yellow_light(self) -> None:
        # h ~28 (yellow), high V
        assert ColorAnalyzer.traffic_light(self._roi(28, 200, 180)) == Messages.TRAFFIC_LIGHT_YELLOW

    def test_high_hue_red(self) -> None:
        # h near 170 → still red
        assert ColorAnalyzer.traffic_light(self._roi(165, 200, 180)) == Messages.TRAFFIC_LIGHT_RED
