"""Frozen settings object — single source of truth for configuration.

Settings are loaded once from environment variables (.env file or shell) and
passed to every component via dependency injection. Never mutate at runtime.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover — optional dep
    pass


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, "").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class CameraSettings:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass(frozen=True)
class YoloSettings:
    model: str = "yolov8n.pt"
    conf: float = 0.35
    iou: float = 0.45
    device: str = "auto"  # auto | cpu | cuda | mps | openvino


@dataclass(frozen=True)
class TimingSettings:
    yolo_summary_s: float = 4.0
    gemini_s: float = 15.0
    ocr_s: float = 22.0
    pit_s: float = 0.40
    battery_s: float = 60.0
    health_s: float = 30.0


@dataclass(frozen=True)
class ThresholdSettings:
    critical_area: float = 0.55
    warning_area: float = 0.24
    approach_slope: float = 0.035
    pit_dark: int = 36
    pit_min_area: int = 5500
    pit_floor_ratio: float = 0.52


@dataclass(frozen=True)
class TtsSettings:
    elevenlabs_voice: str = "21m00Tcm4TlvDq8ikWAM"
    stability: float = 0.65
    similarity: float = 0.85
    style: float = 0.25
    http_timeout_s: float = 10.0
    http_retries: int = 3
    pool_workers: int = 2


@dataclass(frozen=True)
class GpioSettings:
    pin: int = 18
    critical_pattern: tuple[tuple[float, float], ...] = (
        (0.15, 0.05),
        (0.15, 0.05),
        (0.15, 0.05),
        (0.15, 0.05),
    )
    warning_pattern: tuple[tuple[float, float], ...] = ((0.30, 0.10),)


@dataclass(frozen=True)
class DashboardSettings:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass(frozen=True)
class AiSettings:
    gemini_key: str = ""
    elevenlabs_key: str = ""
    local_vlm: str = "moondream2"
    offline_mode: str = "auto"  # auto | always | never


@dataclass(frozen=True)
class Settings:
    """Top-level immutable configuration.

    Construct via :py:meth:`from_env` to load `.env` + shell env, or compose
    sub-settings manually for tests.
    """

    camera: CameraSettings = field(default_factory=CameraSettings)
    yolo: YoloSettings = field(default_factory=YoloSettings)
    timing: TimingSettings = field(default_factory=TimingSettings)
    thresholds: ThresholdSettings = field(default_factory=ThresholdSettings)
    tts: TtsSettings = field(default_factory=TtsSettings)
    gpio: GpioSettings = field(default_factory=GpioSettings)
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)
    ai: AiSettings = field(default_factory=AiSettings)

    log_level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    temp_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()))
    show_gui: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables (with defaults)."""
        return cls(
            camera=CameraSettings(
                index=_env_int("VVA_CAM_INDEX", 0),
                width=_env_int("VVA_CAM_WIDTH", 640),
                height=_env_int("VVA_CAM_HEIGHT", 480),
                fps=_env_int("VVA_CAM_FPS", 30),
            ),
            yolo=YoloSettings(
                conf=_env_float("VVA_YOLO_CONF", 0.35),
                device=_env("VVA_YOLO_DEVICE", "auto"),
            ),
            ai=AiSettings(
                gemini_key=_env("GEMINI_KEY"),
                elevenlabs_key=_env("ELEVENLABS_KEY"),
                local_vlm=_env("VVA_LOCAL_VLM", "moondream2"),
                offline_mode=_env("VVA_OFFLINE_MODE", "auto"),
            ),
            dashboard=DashboardSettings(
                enabled=_env_bool("VVA_DASHBOARD_ENABLED", False),
                host=_env("VVA_DASHBOARD_HOST", "0.0.0.0"),
                port=_env_int("VVA_DASHBOARD_PORT", 8080),
            ),
            log_level=_env("VVA_LOG_LEVEL", "INFO"),
            show_gui=_env_bool("VVA_SHOW_GUI", True) and not _is_headless(),
        )

    def with_overrides(self, **kwargs: object) -> "Settings":
        """Return a new Settings with selected fields overridden."""
        return replace(self, **kwargs)


def _is_headless() -> bool:
    """Detect headless environments (Termux, SSH-no-X)."""
    if "com.termux" in os.environ.get("PREFIX", ""):
        return True
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return True
    return False
