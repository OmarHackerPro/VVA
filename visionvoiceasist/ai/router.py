"""AI router — picks online (Gemini) or offline (local VLM) per request.

Lifecycle:
    - When the cloud model fails N times in a row OR a network probe fails,
      switch to OFFLINE mode and announce the change to the user.
    - When the cloud model succeeds again, switch back to ONLINE.
"""

from __future__ import annotations

import logging
import socket
import threading
import time

import numpy as np

from ..events import EventBus, EventType
from ..i18n import Messages
from ..settings import AiSettings
from ..types import OperatingMode, Priority, SpeechEvent
from .base import VisionLanguageModel, VlmResult

log = logging.getLogger(__name__)


class AiRouter:
    """Routes describe/query calls to whichever backend is healthy.

    The router is *synchronous*; threading and queueing are the caller's
    responsibility.
    """

    def __init__(
        self,
        settings: AiSettings,
        bus: EventBus,
        *,
        cloud: VisionLanguageModel,
        offline: VisionLanguageModel,
    ) -> None:
        self._s = settings
        self._bus = bus
        self._cloud = cloud
        self._offline = offline
        self._mode = OperatingMode.ONLINE
        self._mode_lock = threading.Lock()

    @property
    def mode(self) -> OperatingMode:
        return self._mode

    def is_active(self) -> bool:
        """True if at least one backend can serve a request."""
        return self._cloud.is_available() or self._offline.is_available()

    def describe(self, image: np.ndarray, prompt: str | None = None) -> VlmResult:
        return self._dispatch("describe", image, prompt or "", "")

    def query(self, image: np.ndarray, question: str) -> VlmResult:
        return self._dispatch("query", image, "", question)

    def _dispatch(
        self,
        op: str,
        image: np.ndarray,
        prompt: str,
        question: str,
    ) -> VlmResult:
        forced = self._s.offline_mode
        if forced == "always":
            return self._call(self._offline, op, image, prompt, question)
        if forced == "never":
            return self._call(self._cloud, op, image, prompt, question)

        # auto: prefer cloud, fall back to offline.
        if self._cloud.is_available() and self._has_network():
            result = self._call(self._cloud, op, image, prompt, question)
            if result.success:
                self._set_mode(OperatingMode.ONLINE)
                return result
            log.info("Cloud failed (%s); switching offline", result.error)
        result = self._call(self._offline, op, image, prompt, question)
        if result.success:
            self._set_mode(OperatingMode.OFFLINE)
        return result

    @staticmethod
    def _call(
        model: VisionLanguageModel,
        op: str,
        image: np.ndarray,
        prompt: str,
        question: str,
    ) -> VlmResult:
        if op == "describe":
            return model.describe(image, prompt) if prompt else model.describe(image)
        return model.query(image, question)

    def _set_mode(self, new_mode: OperatingMode) -> None:
        with self._mode_lock:
            if self._mode == new_mode:
                return
            old = self._mode
            self._mode = new_mode
        log.info("Mode change: %s → %s", old.value, new_mode.value)
        self._bus.publish(EventType.MODE_CHANGE, new_mode)
        if new_mode == OperatingMode.OFFLINE:
            self._bus.publish(
                EventType.SPEECH,
                SpeechEvent(text=Messages.NETWORK_LOST, priority=Priority.HIGH, force=True),
            )
        elif new_mode == OperatingMode.ONLINE and old == OperatingMode.OFFLINE:
            self._bus.publish(
                EventType.SPEECH,
                SpeechEvent(text=Messages.NETWORK_BACK, priority=Priority.NORMAL, force=True),
            )

    @staticmethod
    def _has_network(timeout_s: float = 1.0) -> bool:
        """Cheap connectivity probe: TCP-handshake to 8.8.8.8:53."""
        try:
            with socket.create_connection(("8.8.8.8", 53), timeout=timeout_s):
                return True
        except OSError:
            return False
