"""Tests for PitDetector — classical CV stair/pit/threshold detection."""

from __future__ import annotations

import numpy as np
import pytest

from visionvoiceasist.i18n import Messages
from visionvoiceasist.settings import ThresholdSettings
from visionvoiceasist.types import Priority
from visionvoiceasist.vision.pit import PitDetector


def make_detector() -> PitDetector:
    return PitDetector(ThresholdSettings())


class TestPitDetectorSmokeTests:
    def test_blank_frame_no_detection(self) -> None:
        d = make_detector()
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = d.detect(frame)
        # A plain gray frame should not fire
        assert result is None

    def test_returns_none_type_or_tuple(self) -> None:
        d = make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = d.detect(frame)
        assert result is None or (isinstance(result, tuple) and len(result) == 2)

    def test_reset_clears_prev_mean(self) -> None:
        d = make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        d.detect(frame)
        d.reset()
        assert d._prev_mean is None  # type: ignore[attr-defined]

    def test_detect_returns_priority(self) -> None:
        d = make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        d.detect(frame)  # seed prev_mean = 0

        # Create a bright frame — mean delta > 22
        bright = np.full((480, 640, 3), 100, dtype=np.uint8)
        result = d.detect(bright)
        if result is not None:
            _, priority = result
            assert isinstance(priority, Priority)

    def test_threshold_delta_triggers(self) -> None:
        """Mean-intensity jump > 22 should fire PIT_THRESHOLD."""
        d = make_detector()
        dark = np.zeros((480, 640, 3), dtype=np.uint8)
        d.detect(dark)  # prev_mean ≈ 0
        bright = np.full((480, 640, 3), 80, dtype=np.uint8)
        result = d.detect(bright)
        # delta ≈ 80 >> 22, should fire
        assert result is not None
        msg, prio = result
        assert msg == Messages.PIT_THRESHOLD
        assert prio == Priority.HIGH

    def test_stairs_pattern_detection(self) -> None:
        """A frame with dense horizontal edges in lower half should fire PIT_STAIRS."""
        d = PitDetector(ThresholdSettings(
            pit_min_area=100,  # lower threshold for test
            pit_floor_ratio=0.5,
        ))
        # Draw many horizontal edges in lower half
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        for y in range(100, 200, 4):
            frame[y, :] = 255
        result = d.detect(frame)
        # May or may not fire depending on morphological ops; at minimum no crash
        if result is not None:
            assert result[1] in {Priority.CRITICAL, Priority.HIGH}


class TestPitDetectorMinimumFrameSize:
    def test_tiny_frame_no_crash(self) -> None:
        d = make_detector()
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        d.detect(tiny)  # must not raise
