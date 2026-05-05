"""Shared pytest fixtures for VisionVoiceAsist tests."""

from __future__ import annotations

import time

import numpy as np
import pytest

from visionvoiceasist.events import EventBus, EventType
from visionvoiceasist.settings import (
    AiSettings,
    CameraSettings,
    Settings,
    ThresholdSettings,
    TimingSettings,
)
from visionvoiceasist.types import BBox, Detection, Frame, Priority


@pytest.fixture
def bus() -> EventBus:
    b = EventBus()
    yield b
    b.clear()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        camera=CameraSettings(width=640, height=480),
        ai=AiSettings(gemini_key="test-key"),
        show_gui=False,
    )


@pytest.fixture
def thresholds() -> ThresholdSettings:
    return ThresholdSettings()


@pytest.fixture
def black_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame() -> np.ndarray:
    return np.full((480, 640, 3), 255, dtype=np.uint8)


@pytest.fixture
def frame_obj(black_frame: np.ndarray) -> Frame:
    return Frame(image=black_frame, frame_id=1, captured_at=time.time())


def make_det(
    label: str = "car",
    x1: int = 100,
    y1: int = 100,
    x2: int = 200,
    y2: int = 200,
    conf: float = 0.9,
    area_pct: float = 0.05,
    track_id: int | None = None,
    frame_id: int = 0,
) -> Detection:
    from visionvoiceasist.i18n import az_label
    return Detection(
        label_eng=label,
        label_az=az_label(label),
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        conf=conf,
        area_pct=area_pct,
        frame_id=frame_id,
        track_id=track_id,
    )
