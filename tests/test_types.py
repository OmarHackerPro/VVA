"""Tests for core data types."""

from __future__ import annotations

import time

import numpy as np
import pytest

from visionvoiceasist.types import (
    BBox,
    Detection,
    Frame,
    HapticEvent,
    HealthReport,
    OperatingMode,
    Priority,
    SpeechEvent,
)


class TestBBox:
    def test_basic_properties(self) -> None:
        b = BBox(10, 20, 110, 70)
        assert b.width == 100
        assert b.height == 50
        assert b.cx == 60
        assert b.cy == 45
        assert b.area == 5000

    def test_area_pct(self) -> None:
        b = BBox(0, 0, 64, 48)
        assert b.area_pct(640, 480) == pytest.approx(0.01, rel=1e-3)

    def test_zero_frame_area_pct(self) -> None:
        b = BBox(0, 0, 100, 100)
        assert b.area_pct(0, 0) == 0.0

    def test_negative_box_area_is_zero(self) -> None:
        b = BBox(100, 100, 50, 50)
        assert b.area == 0

    def test_frozen(self) -> None:
        b = BBox(0, 0, 10, 10)
        with pytest.raises(Exception):
            b.x1 = 5  # type: ignore[misc]


class TestDetection:
    def test_frozen(self) -> None:
        d = Detection(
            label_eng="car", label_az="maşın",
            bbox=BBox(0, 0, 100, 100), conf=0.9, area_pct=0.05,
        )
        with pytest.raises(Exception):
            d.conf = 0.5  # type: ignore[misc]

    def test_timestamp_auto(self) -> None:
        before = time.time()
        d = Detection(
            label_eng="car", label_az="maşın",
            bbox=BBox(0, 0, 10, 10), conf=0.8, area_pct=0.01,
        )
        assert d.timestamp >= before


class TestSpeechEvent:
    def test_defaults(self) -> None:
        e = SpeechEvent(text="salam")
        assert e.priority == Priority.NORMAL
        assert e.force is False
        assert e.pan == 0.0

    def test_pan_range(self) -> None:
        e = SpeechEvent(text="x", pan=-1.0)
        assert e.pan == -1.0


class TestFrame:
    def test_valid_frame(self, black_frame: np.ndarray) -> None:
        f = Frame(image=black_frame, frame_id=0, captured_at=time.time())
        assert f.image.shape == (480, 640, 3)

    def test_invalid_shape_raises(self) -> None:
        bad = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="HxWx3 BGR"):
            Frame(image=bad, frame_id=0, captured_at=time.time())

    def test_frozen(self, black_frame: np.ndarray) -> None:
        f = Frame(image=black_frame, frame_id=0, captured_at=time.time())
        with pytest.raises(Exception):
            f.frame_id = 99  # type: ignore[misc]


class TestHealthReport:
    def test_defaults(self) -> None:
        r = HealthReport(component="cam", healthy=True)
        assert r.detail == ""
        assert r.timestamp > 0

    def test_unhealthy(self) -> None:
        r = HealthReport(component="tts", healthy=False, detail="pyaudio missing")
        assert not r.healthy


class TestPriority:
    def test_ordering(self) -> None:
        assert Priority.CRITICAL.value < Priority.HIGH.value
        assert Priority.HIGH.value < Priority.NORMAL.value
        assert Priority.NORMAL.value < Priority.LOW.value


class TestOperatingMode:
    def test_values(self) -> None:
        assert OperatingMode.ONLINE.value == "online"
        assert OperatingMode.OFFLINE.value == "offline"
        assert OperatingMode.DEGRADED.value == "degraded"
