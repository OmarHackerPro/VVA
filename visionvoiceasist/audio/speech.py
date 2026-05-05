"""Speech engine — priority-preemptive, thread-pool synthesis, never silent.

Architecture:
    - Subscribers publish :py:class:`SpeechEvent` to the bus.
    - The engine has a :py:class:`PriorityQueue` of pending speech tasks.
    - A pool of synth workers picks tasks in priority order. Critical events
      drain the queue *before* enqueuing themselves so they preempt.
    - Each provider is tried in fallback order. If **all** providers fail,
      we publish a HEALTH event so the HealthMonitor can surface the issue
      and at least play an emergency beep.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from ..events import EventBus, EventType
from ..i18n import Messages
from ..settings import Settings
from ..types import HealthReport, Priority, SpeechEvent
from .player import AudioPlayer
from .providers import (
    BeepFallbackProvider,
    ElevenLabsProvider,
    EspeakProvider,
    Pyttsx3Provider,
    TermuxTtsProvider,
    TtsProvider,
)

log = logging.getLogger(__name__)


@dataclass(order=True)
class _Task:
    priority: int
    seq: int
    event: SpeechEvent = field(compare=False)


class SpeechEngine:
    """Priority queue + thread pool for TTS.

    Wire it up:
        engine = SpeechEngine(settings, bus)
        bus.subscribe(EventType.SPEECH, engine.enqueue)
        engine.start()
    """

    def __init__(
        self,
        settings: Settings,
        bus: EventBus,
        *,
        providers: Optional[list[TtsProvider]] = None,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._player = AudioPlayer()
        self._providers = providers or self._default_providers()

        self._q: queue.PriorityQueue[_Task] = queue.PriorityQueue()
        self._recent: deque[str] = deque(maxlen=7)
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._stop = threading.Event()
        self._workers = ThreadPoolExecutor(
            max_workers=settings.tts.pool_workers, thread_name_prefix="TtsSynth"
        )
        self._consumer: Optional[threading.Thread] = None
        self._stats = {"total": 0, "skipped": 0, "failed": 0,
                       **{p.name: 0 for p in self._providers}}
        self._stats_lock = threading.Lock()
        self._last_speak_at = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._consumer is not None:
            return
        self._stop.clear()
        self._consumer = threading.Thread(
            target=self._consume_loop, daemon=True, name="SpeechConsumer"
        )
        self._consumer.start()

    def stop(self) -> None:
        self._stop.set()
        if self._consumer:
            self._consumer.join(timeout=2.0)
        self._workers.shutdown(wait=False, cancel_futures=True)

    # ── Public ────────────────────────────────────────────────────────────
    def enqueue(self, event: SpeechEvent) -> None:
        """Add *event* to the queue. Critical events drain pending tasks first."""
        if not event.text:
            return
        if not event.force and event.text in self._recent:
            self._stat_inc("skipped")
            return
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        if event.priority is Priority.CRITICAL:
            self._drain_pending()
        self._q.put(_Task(priority=event.priority.value, seq=seq, event=event))

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def health(self) -> HealthReport:
        """Probe: healthy if at least one provider is available *and* we spoke recently."""
        any_available = any(p.is_available() for p in self._providers)
        if not any_available:
            return HealthReport(
                component="speech", healthy=False, detail="no provider available"
            )
        # If queue is non-empty but consumer hasn't drained for >30s, degraded.
        if self._q.qsize() > 5 and time.time() - self._last_speak_at > 30:
            return HealthReport(
                component="speech", healthy=False, detail="queue stalled"
            )
        return HealthReport(component="speech", healthy=True)

    # ── Internals ─────────────────────────────────────────────────────────
    def _default_providers(self) -> list[TtsProvider]:
        s = self._settings
        out: list[TtsProvider] = []
        if s.ai.elevenlabs_key:
            out.append(
                ElevenLabsProvider(
                    s.tts, s.ai.elevenlabs_key, self._player, s.temp_dir
                )
            )
        out.append(Pyttsx3Provider())
        out.append(TermuxTtsProvider())
        out.append(EspeakProvider())
        out.append(BeepFallbackProvider(self._player, s.temp_dir))
        return out

    def _drain_pending(self) -> None:
        drained = 0
        while not self._q.empty():
            try:
                self._q.get_nowait()
                self._q.task_done()
                drained += 1
            except queue.Empty:
                break
        if drained:
            log.debug("Drained %d pending speech tasks (CRITICAL preemption)", drained)

    def _consume_loop(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._workers.submit(self._handle_task, task)

    def _handle_task(self, task: _Task) -> None:
        event = task.event
        self._recent.append(event.text)
        self._stat_inc("total")
        success = False
        for provider in self._providers:
            if not provider.is_available():
                continue
            try:
                if provider.speak(event.text, pan=event.pan):
                    self._stat_inc(provider.name)
                    success = True
                    break
            except Exception as exc:  # noqa: BLE001
                log.warning("Provider %s raised: %s", provider.name, exc)
        if not success:
            self._stat_inc("failed")
            self._bus.publish(
                EventType.HEALTH,
                HealthReport(
                    component="speech",
                    healthy=False,
                    detail=f"all providers failed for: {event.text[:30]}",
                ),
            )
            log.error("All TTS providers failed for: %s", event.text[:60])
            # Last-resort beep — even BeepFallbackProvider counts as success
            # above, but if even that fails we need to log loudly.
        self._last_speak_at = time.time()
        self._q.task_done()

    def _stat_inc(self, key: str) -> None:
        with self._stats_lock:
            self._stats[key] = self._stats.get(key, 0) + 1

    def _emit_failure_message(self) -> None:
        """Surface failure as a final speech attempt (lowest-priority providers)."""
        # Try beep one more time directly.
        for p in self._providers:
            if p.name == "beep":
                p.speak(Messages.TTS_FAILED, pan=0.0)
                break
