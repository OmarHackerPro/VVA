"""Tests for HealthMonitor, liveness_probe, and make_static_probe."""

from __future__ import annotations

import time

import pytest

from visionvoiceasist.events import EventBus, EventType
from visionvoiceasist.health import HealthMonitor, liveness_probe, make_static_probe
from visionvoiceasist.types import HealthReport, Priority, SpeechEvent


class TestMakeStaticProbe:
    def test_healthy_probe(self) -> None:
        probe = make_static_probe("cam", healthy=True, detail="ok")
        result = probe()
        assert isinstance(result, HealthReport)
        assert result.healthy is True
        assert result.component == "cam"
        assert result.detail == "ok"

    def test_unhealthy_probe(self) -> None:
        probe = make_static_probe("tts", healthy=False, detail="broken")
        result = probe()
        assert result.healthy is False
        assert result.detail == "broken"


class TestLivenessProbe:
    def test_never_started_is_unhealthy(self) -> None:
        probe = liveness_probe("yolo", [], stale_after_s=10.0)
        result = probe()
        assert result.healthy is False
        assert "never started" in result.detail

    def test_recent_beat_is_healthy(self) -> None:
        beats = [time.time()]
        probe = liveness_probe("yolo", beats, stale_after_s=10.0)
        result = probe()
        assert result.healthy is True

    def test_stale_beat_is_unhealthy(self) -> None:
        old_ts = time.time() - 20.0
        beats = [old_ts]
        probe = liveness_probe("yolo", beats, stale_after_s=5.0)
        result = probe()
        assert result.healthy is False
        assert "ago" in result.detail

    def test_only_last_beat_matters(self) -> None:
        beats = [time.time() - 100.0, time.time()]
        probe = liveness_probe("cam", beats, stale_after_s=5.0)
        assert probe().healthy is True


class TestHealthMonitor:
    def test_register_and_force_check(self, bus: EventBus) -> None:
        monitor = HealthMonitor(bus, interval_s=999.0)
        monitor.register("cam", make_static_probe("cam", healthy=True))
        received: list[HealthReport] = []
        bus.subscribe(EventType.HEALTH, received.append)
        monitor.force_check()
        assert len(received) == 1
        assert received[0].component == "cam"
        assert received[0].healthy is True

    def test_degraded_speech_fires_on_unhealthy(self, bus: EventBus) -> None:
        monitor = HealthMonitor(bus, interval_s=999.0)
        monitor.register("cam", make_static_probe("cam", healthy=False))
        speech_events: list[SpeechEvent] = []
        bus.subscribe(EventType.SPEECH, speech_events.append)
        monitor.force_check()
        assert any(isinstance(e, SpeechEvent) for e in speech_events)
        degraded_msgs = [e for e in speech_events if e.force]
        assert len(degraded_msgs) >= 1

    def test_degraded_speech_fires_only_once(self, bus: EventBus) -> None:
        monitor = HealthMonitor(bus, interval_s=999.0)
        monitor.register("cam", make_static_probe("cam", healthy=False))
        speech_events: list[SpeechEvent] = []
        bus.subscribe(EventType.SPEECH, speech_events.append)
        monitor.force_check()
        monitor.force_check()
        # Should be exactly 1 degraded announcement (edge-triggered)
        assert len([e for e in speech_events if e.force]) == 1

    def test_recovery_resets_degraded_flag(self, bus: EventBus) -> None:
        monitor = HealthMonitor(bus, interval_s=999.0)
        state = {"healthy": False}

        def dynamic_probe() -> HealthReport:
            return HealthReport(component="cam", healthy=state["healthy"])

        monitor.register("cam", dynamic_probe)
        speech_events: list[SpeechEvent] = []
        bus.subscribe(EventType.SPEECH, speech_events.append)

        monitor.force_check()  # unhealthy → degraded announcement
        state["healthy"] = True
        monitor.force_check()  # recovered → flag reset

        # Now goes unhealthy again → should announce again
        state["healthy"] = False
        monitor.force_check()
        degraded = [e for e in speech_events if e.force]
        assert len(degraded) == 2

    def test_probe_exception_treated_as_unhealthy(self, bus: EventBus) -> None:
        monitor = HealthMonitor(bus, interval_s=999.0)

        def broken_probe() -> HealthReport:
            raise RuntimeError("crash")

        monitor.register("broken", broken_probe)
        health_events: list[HealthReport] = []
        bus.subscribe(EventType.HEALTH, health_events.append)
        monitor.force_check()
        assert any(not r.healthy for r in health_events)

    def test_snapshot(self, bus: EventBus) -> None:
        monitor = HealthMonitor(bus, interval_s=999.0)
        monitor.register("cam", make_static_probe("cam", healthy=True))
        snap = monitor.snapshot()
        assert "cam" in snap
        assert snap["cam"] is True

    def test_start_stop(self, bus: EventBus) -> None:
        monitor = HealthMonitor(bus, interval_s=999.0)
        monitor.register("x", make_static_probe("x", healthy=True))
        monitor.start()
        assert monitor._thread is not None  # type: ignore[attr-defined]
        assert monitor._thread.is_alive()
        monitor.stop()
        monitor._thread.join(timeout=2.0)
        assert not monitor._thread.is_alive()

    def test_start_idempotent(self, bus: EventBus) -> None:
        monitor = HealthMonitor(bus, interval_s=999.0)
        monitor.start()
        t1 = monitor._thread  # type: ignore[attr-defined]
        monitor.start()  # should not create a second thread
        assert monitor._thread is t1  # type: ignore[attr-defined]
        monitor.stop()
