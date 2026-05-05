"""Health monitor — periodically polls subsystems and announces degraded mode.

Critical to safety: an accessibility tool must never fail silently. If TTS,
camera, or AI is degraded, the user is told via the surviving channels (or
haptic + system beep if all audio is dead).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Optional

from .events import EventBus, EventType
from .i18n import Messages
from .types import HealthReport, Priority, SpeechEvent

log = logging.getLogger(__name__)

ProbeFn = Callable[[], HealthReport]


class HealthMonitor:
    """Polls registered probes; on degradation, publishes audible alert.

    Producers register a probe with :py:meth:`register`. The monitor runs on
    a daemon thread and emits a SPEECH event the first time any probe goes
    unhealthy and again when all probes recover.
    """

    def __init__(self, bus: EventBus, interval_s: float = 30.0) -> None:
        self._bus = bus
        self._interval = interval_s
        self._probes: dict[str, ProbeFn] = {}
        self._last_state: dict[str, bool] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._was_degraded = False

    def register(self, name: str, probe: ProbeFn) -> None:
        """Register a probe by *name*. Probes must not block."""
        with self._lock:
            self._probes[name] = probe
            self._last_state[name] = True

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="HealthMonitor"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._tick()
            self._stop.wait(self._interval)

    def _tick(self) -> None:
        with self._lock:
            probes = dict(self._probes)
        any_unhealthy = False
        for name, probe in probes.items():
            try:
                report = probe()
            except Exception as exc:  # noqa: BLE001
                log.exception("Probe %s raised", name)
                report = HealthReport(component=name, healthy=False, detail=str(exc))
            self._bus.publish(EventType.HEALTH, report)
            prev = self._last_state.get(name, True)
            if prev and not report.healthy:
                log.warning("Component %s became unhealthy: %s", name, report.detail)
            elif not prev and report.healthy:
                log.info("Component %s recovered", name)
            self._last_state[name] = report.healthy
            if not report.healthy:
                any_unhealthy = True

        # Edge-trigger degraded-mode announcement
        if any_unhealthy and not self._was_degraded:
            self._bus.publish(
                EventType.SPEECH,
                SpeechEvent(text=Messages.DEGRADED, priority=Priority.HIGH, force=True),
            )
            self._was_degraded = True
        elif not any_unhealthy and self._was_degraded:
            self._was_degraded = False

    def force_check(self) -> None:
        """Run all probes immediately (for tests / status command)."""
        self._tick()

    def snapshot(self) -> dict[str, bool]:
        with self._lock:
            return dict(self._last_state)


def make_static_probe(name: str, healthy: bool, detail: str = "") -> ProbeFn:
    """Helper for tests: a probe that always returns the same value."""
    def probe() -> HealthReport:
        return HealthReport(component=name, healthy=healthy, detail=detail)

    return probe


def liveness_probe(name: str, last_beat: list[float], stale_after_s: float) -> ProbeFn:
    """Return a probe that fails if *last_beat[-1]* is older than *stale_after_s*.

    The producer should append `time.time()` to *last_beat* on every successful
    cycle. (List-of-floats is shared mutable state — that's intentional.)
    """
    def probe() -> HealthReport:
        if not last_beat:
            return HealthReport(component=name, healthy=False, detail="never started")
        age = time.time() - last_beat[-1]
        return HealthReport(
            component=name,
            healthy=age <= stale_after_s,
            detail=f"last beat {age:.1f}s ago",
        )

    return probe
