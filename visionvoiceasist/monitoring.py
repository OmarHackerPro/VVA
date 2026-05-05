"""System monitoring: FPS counter, battery watcher."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

import psutil

from .events import EventBus, EventType
from .i18n import Messages
from .types import Priority, SpeechEvent

log = logging.getLogger(__name__)


class FpsCounter:
    """Rolling-mean FPS over the last N frame intervals."""

    def __init__(self, window: int = 20) -> None:
        self._buf: deque[float] = deque(maxlen=window)
        self._t_last = time.time()
        self.fps = 0.0

    def tick(self) -> None:
        now = time.time()
        dt = now - self._t_last
        self._t_last = now
        if dt > 0:
            self._buf.append(1.0 / dt)
            self.fps = sum(self._buf) / len(self._buf)


class BatteryWatcher:
    """Periodic battery checks → SPEECH events."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._warned = False

    def check(self) -> Optional[int]:
        bat = psutil.sensors_battery()
        if bat is None:
            return None
        pct = int(bat.percent)
        if pct <= 10 and not self._warned:
            self._bus.publish(
                EventType.SPEECH,
                SpeechEvent(text=Messages.BATTERY_CRITICAL.format(pct=pct),
                            priority=Priority.CRITICAL, force=True),
            )
            self._warned = True
        elif 10 < pct <= 20:
            self._bus.publish(
                EventType.SPEECH,
                SpeechEvent(text=Messages.BATTERY_WARN.format(pct=pct),
                            priority=Priority.HIGH, force=True),
            )
        elif pct > 25:
            self._warned = False
        return pct
