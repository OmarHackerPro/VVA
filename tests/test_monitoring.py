"""Tests for FpsCounter and BatteryWatcher."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from visionvoiceasist.events import EventBus, EventType
from visionvoiceasist.monitoring import BatteryWatcher, FpsCounter
from visionvoiceasist.types import SpeechEvent


class TestFpsCounter:
    def test_initial_fps_is_zero(self) -> None:
        f = FpsCounter()
        assert f.fps == 0.0

    def test_fps_nonzero_after_ticks(self) -> None:
        f = FpsCounter()
        time.sleep(0.01)
        f.tick()
        time.sleep(0.01)
        f.tick()
        assert f.fps > 0.0

    def test_fps_approximately_correct(self) -> None:
        f = FpsCounter(window=4)
        for _ in range(5):
            time.sleep(0.05)  # ~20 FPS
            f.tick()
        # Allow wide tolerance for CI timing variance
        assert 5.0 < f.fps < 50.0

    def test_fps_window_size(self) -> None:
        f = FpsCounter(window=3)
        for _ in range(10):
            time.sleep(0.005)
            f.tick()
        assert f.fps > 0

    def test_fps_zero_dt_does_not_crash(self) -> None:
        """If two ticks arrive at the same timestamp, no division by zero."""
        f = FpsCounter()
        with patch("visionvoiceasist.monitoring.time") as mock_time:
            mock_time.time.return_value = 1000.0
            f._t_last = 1000.0  # type: ignore[attr-defined]
            f.tick()  # dt == 0 → should skip silently


class TestBatteryWatcher:
    def _make_watcher(self) -> tuple[BatteryWatcher, list[SpeechEvent]]:
        bus = EventBus()
        events: list[SpeechEvent] = []
        bus.subscribe(EventType.SPEECH, events.append)
        return BatteryWatcher(bus), events

    def _mock_battery(self, pct: float, plugged: bool = True) -> MagicMock:
        battery = MagicMock()
        battery.percent = pct
        battery.power_plugged = plugged
        return battery

    def test_no_battery_returns_none(self) -> None:
        watcher, _ = self._make_watcher()
        with patch("psutil.sensors_battery", return_value=None):
            assert watcher.check() is None

    def test_returns_percentage(self) -> None:
        watcher, _ = self._make_watcher()
        with patch("psutil.sensors_battery", return_value=self._mock_battery(75)):
            assert watcher.check() == 75

    def test_critical_fires_speech(self) -> None:
        watcher, events = self._make_watcher()
        with patch("psutil.sensors_battery", return_value=self._mock_battery(8)):
            watcher.check()
        assert len(events) == 1
        assert "8" in events[0].text

    def test_critical_fires_only_once(self) -> None:
        watcher, events = self._make_watcher()
        mock_bat = self._mock_battery(5)
        with patch("psutil.sensors_battery", return_value=mock_bat):
            watcher.check()
            watcher.check()
        assert len(events) == 1

    def test_warning_fires_speech(self) -> None:
        watcher, events = self._make_watcher()
        with patch("psutil.sensors_battery", return_value=self._mock_battery(15)):
            watcher.check()
        assert len(events) == 1

    def test_warning_fires_every_call(self) -> None:
        """Unlike critical, warning fires on every check (different design)."""
        watcher, events = self._make_watcher()
        with patch("psutil.sensors_battery", return_value=self._mock_battery(15)):
            watcher.check()
            watcher.check()
        assert len(events) == 2

    def test_recovery_above_25_resets_critical_warn_flag(self) -> None:
        watcher, events = self._make_watcher()
        with patch("psutil.sensors_battery", return_value=self._mock_battery(5)):
            watcher.check()
        with patch("psutil.sensors_battery", return_value=self._mock_battery(80)):
            watcher.check()  # resets _warned
        with patch("psutil.sensors_battery", return_value=self._mock_battery(5)):
            watcher.check()  # should fire again
        assert len(events) == 2

    def test_normal_battery_no_speech(self) -> None:
        watcher, events = self._make_watcher()
        with patch("psutil.sensors_battery", return_value=self._mock_battery(60)):
            watcher.check()
        assert len(events) == 0
