"""Core data types — pure, hashable, side-effect-free.

These types are the contract between modules. They contain no business logic
beyond simple computed properties.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Priority(Enum):
    """Speech / event priority — lower value = higher priority.

    CRITICAL drains the queue and triggers haptic burst.
    HIGH preempts NORMAL/LOW; LOW is best-effort.
    """

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class OperatingMode(Enum):
    """Runtime operating mode tracked by the AI router."""

    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class BBox:
    """Bounding box in pixel coordinates (top-left, bottom-right inclusive)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def area_pct(self, frame_w: int, frame_h: int) -> float:
        denom = frame_w * frame_h
        return float(self.area) / denom if denom > 0 else 0.0


@dataclass(frozen=True)
class Detection:
    """Single object detection from the vision pipeline."""

    label_eng: str
    label_az: str
    bbox: BBox
    conf: float
    area_pct: float
    frame_id: int = 0
    track_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SpeechEvent:
    """Event published to the SpeechEngine."""

    text: str
    priority: Priority = Priority.NORMAL
    force: bool = False
    pan: float = 0.0  # -1.0 (left) to +1.0 (right) for spatial audio
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class HapticEvent:
    """Event published to the haptic motor."""

    priority: Priority
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Frame:
    """Camera frame with monotonic id."""

    image: np.ndarray
    frame_id: int
    captured_at: float

    def __post_init__(self) -> None:
        # Frozen dataclass: validate without mutating.
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"Frame must be HxWx3 BGR, got shape {self.image.shape}")


@dataclass(frozen=True)
class HealthReport:
    """Snapshot of subsystem health."""

    component: str
    healthy: bool
    detail: str = ""
    timestamp: float = field(default_factory=time.time)
