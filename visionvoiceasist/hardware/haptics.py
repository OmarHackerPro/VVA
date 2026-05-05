"""GPIO vibration motor with priority-driven patterns.

If RPi.GPIO is not available (most dev machines), the motor gracefully
becomes a no-op while still logging the pattern, so behaviour is testable.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time

from ..events import EventBus, EventType
from ..settings import GpioSettings
from ..types import HapticEvent, HealthReport, Priority

log = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO  # type: ignore[import-untyped]

    _GPIO_OK = True
except Exception:  # noqa: BLE001 — common everywhere except Pi
    _GPIO_OK = False


class HapticMotor:
    """3-tier haptic motor: critical / warning / silent.

    Critical: short rapid bursts (4×).
    Warning: single longer pulse.
    Other priorities: silent.
    """

    def __init__(self, settings: GpioSettings, bus: EventBus) -> None:
        self._s = settings
        self._bus = bus
        self._ready = False
        self._init()
        atexit.register(self._cleanup)
        bus.subscribe(EventType.HAPTIC, self.handle_event)

    def _init(self) -> None:
        if not _GPIO_OK:
            log.info("RPi.GPIO unavailable — haptics in no-op mode")
            return
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._s.pin, GPIO.OUT)
            GPIO.output(self._s.pin, GPIO.LOW)
            self._ready = True
            log.info("Haptic motor ready on GPIO %d", self._s.pin)
        except Exception as exc:  # noqa: BLE001
            log.warning("GPIO init failed: %s", exc)

    def health(self) -> HealthReport:
        if not _GPIO_OK:
            return HealthReport(
                component="haptics", healthy=True, detail="no-op (no GPIO)"
            )
        return HealthReport(
            component="haptics", healthy=self._ready,
            detail="ready" if self._ready else "init failed",
        )

    def handle_event(self, event: HapticEvent) -> None:
        if event.priority is Priority.CRITICAL:
            self.critical()
        elif event.priority is Priority.HIGH:
            self.warning()

    def critical(self) -> None:
        self._pulse(self._s.critical_pattern)

    def warning(self) -> None:
        self._pulse(self._s.warning_pattern)

    def _pulse(self, pattern: tuple[tuple[float, float], ...]) -> None:
        if not self._ready:
            log.debug("Haptic pulse (no-op): %s", pattern)
            return
        threading.Thread(
            target=self._pulse_blocking, args=(pattern,), daemon=True,
            name="HapticPulse",
        ).start()

    def _pulse_blocking(self, pattern: tuple[tuple[float, float], ...]) -> None:
        for on_t, off_t in pattern:
            try:
                GPIO.output(self._s.pin, GPIO.HIGH)
                time.sleep(on_t)
                GPIO.output(self._s.pin, GPIO.LOW)
                time.sleep(off_t)
            except Exception as exc:  # noqa: BLE001
                log.warning("Haptic pulse failed: %s", exc)
                self._ready = False
                return

    def _cleanup(self) -> None:
        if not _GPIO_OK or not self._ready:
            return
        try:
            GPIO.output(self._s.pin, GPIO.LOW)
            GPIO.cleanup(self._s.pin)
        except Exception:  # noqa: BLE001
            pass
