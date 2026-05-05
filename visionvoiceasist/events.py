"""Thread-safe in-process event bus — pub/sub with weak typing.

Producers (vision, AI, hardware monitors) publish events; consumers (speech,
overlay, dashboard) subscribe. Decoupled; no direct method calls.

Example:
    >>> bus = EventBus()
    >>> bus.subscribe(EventType.SPEECH, lambda e: print(e.text))
    >>> bus.publish(EventType.SPEECH, SpeechEvent("salam"))
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class EventType(Enum):
    """All event channels in the system."""

    # Vision pipeline
    DETECTIONS = "detections"            # payload: list[Detection]
    PIT_WARNING = "pit_warning"          # payload: tuple[str, Priority]
    OCR_TEXT = "ocr_text"                # payload: str
    AI_DESCRIPTION = "ai_description"    # payload: str
    APPROACH_ALERT = "approach_alert"    # payload: tuple[str, Priority]

    # Output channels
    SPEECH = "speech"                    # payload: SpeechEvent
    HAPTIC = "haptic"                    # payload: HapticEvent

    # IoT / Smart City
    V2X_ALERT = "v2x_alert"             # payload: V2xAlert (iot.v2x_client)

    # System / lifecycle
    HEALTH = "health"                    # payload: HealthReport
    MODE_CHANGE = "mode_change"          # payload: OperatingMode
    FRAME = "frame"                      # payload: Frame (dashboard)
    SHUTDOWN = "shutdown"                # payload: None


Subscriber = Callable[[Any], None]


class EventBus:
    """Synchronous pub/sub bus.

    Subscribers are invoked on the *publisher's* thread. For long-running
    work, subscribers must offload to their own queues (the SpeechEngine
    does this).
    """

    def __init__(self) -> None:
        self._subs: dict[EventType, list[Subscriber]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, event_type: EventType, callback: Subscriber) -> None:
        """Register *callback* for *event_type*. Idempotent."""
        with self._lock:
            if callback not in self._subs[event_type]:
                self._subs[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Subscriber) -> None:
        with self._lock:
            if callback in self._subs[event_type]:
                self._subs[event_type].remove(callback)

    def publish(self, event_type: EventType, payload: Any = None) -> None:
        """Publish *payload* on *event_type*.

        Subscribers are isolated: a raising subscriber does not prevent
        others from receiving the event.
        """
        with self._lock:
            subs = list(self._subs[event_type])
        for cb in subs:
            try:
                cb(payload)
            except Exception:  # noqa: BLE001 — bus must isolate failures
                log.exception("Subscriber for %s raised", event_type.value)

    def clear(self) -> None:
        with self._lock:
            self._subs.clear()
