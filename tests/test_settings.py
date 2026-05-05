"""Tests for Settings loading and override mechanics."""

from __future__ import annotations

import os

import pytest

from visionvoiceasist.settings import (
    AiSettings,
    CameraSettings,
    DashboardSettings,
    Settings,
    ThresholdSettings,
    TimingSettings,
    YoloSettings,
)


class TestSettingsDefaults:
    def test_default_construction(self) -> None:
        s = Settings()
        assert s.camera.width == 640
        assert s.camera.height == 480
        assert s.yolo.conf == pytest.approx(0.35)
        assert s.show_gui is True

    def test_frozen(self) -> None:
        s = Settings()
        with pytest.raises(Exception):
            s.show_gui = False  # type: ignore[misc]

    def test_with_overrides(self) -> None:
        s = Settings()
        s2 = s.with_overrides(show_gui=False, log_level="DEBUG")
        assert s2.show_gui is False
        assert s2.log_level == "DEBUG"
        assert s.show_gui is True  # original unchanged

    def test_nested_settings_accessible(self) -> None:
        s = Settings(
            camera=CameraSettings(width=1280, height=720),
            ai=AiSettings(gemini_key="abc"),
        )
        assert s.camera.width == 1280
        assert s.ai.gemini_key == "abc"


class TestSettingsFromEnv:
    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in [
            "VVA_CAM_INDEX", "VVA_CAM_WIDTH", "VVA_CAM_HEIGHT",
            "VVA_CAM_FPS", "VVA_YOLO_CONF", "VVA_YOLO_DEVICE",
            "GEMINI_KEY", "ELEVENLABS_KEY", "VVA_LOCAL_VLM",
            "VVA_OFFLINE_MODE", "VVA_DASHBOARD_ENABLED",
            "VVA_DASHBOARD_HOST", "VVA_DASHBOARD_PORT",
            "VVA_LOG_LEVEL", "VVA_SHOW_GUI", "DISPLAY",
        ]:
            monkeypatch.delenv(key, raising=False)
        s = Settings.from_env()
        assert s.camera.index == 0
        assert s.yolo.device == "auto"
        assert s.dashboard.enabled is False
        assert s.log_level == "INFO"

    def test_from_env_reads_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VVA_CAM_WIDTH", "1280")
        monkeypatch.setenv("VVA_YOLO_CONF", "0.50")
        monkeypatch.setenv("GEMINI_KEY", "my-secret-key")
        monkeypatch.setenv("VVA_DASHBOARD_ENABLED", "true")
        monkeypatch.setenv("VVA_SHOW_GUI", "false")
        s = Settings.from_env()
        assert s.camera.width == 1280
        assert s.yolo.conf == pytest.approx(0.50)
        assert s.ai.gemini_key == "my-secret-key"
        assert s.dashboard.enabled is True
        assert s.show_gui is False

    def test_bad_int_env_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VVA_CAM_WIDTH", "not-a-number")
        s = Settings.from_env()
        assert s.camera.width == 640  # default

    def test_bad_float_env_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VVA_YOLO_CONF", "??")
        s = Settings.from_env()
        assert s.yolo.conf == pytest.approx(0.35)  # default


class TestThresholdSettings:
    def test_defaults(self) -> None:
        t = ThresholdSettings()
        assert t.critical_area == pytest.approx(0.55)
        assert t.warning_area == pytest.approx(0.24)
        assert t.approach_slope == pytest.approx(0.035)

    def test_frozen(self) -> None:
        t = ThresholdSettings()
        with pytest.raises(Exception):
            t.critical_area = 0.9  # type: ignore[misc]


class TestTimingSettings:
    def test_defaults(self) -> None:
        t = TimingSettings()
        assert t.yolo_summary_s == pytest.approx(4.0)
        assert t.gemini_s == pytest.approx(15.0)
        assert t.pit_s == pytest.approx(0.40)
