"""Interactive voice-query service: hold button → speak → AI answers.

Strategy:
    1. User presses & holds a key/GPIO button (managed by the runtime).
    2. We capture audio for up to *max_record_s* seconds.
    3. Local Whisper transcribes (offline-capable).
    4. AiRouter answers the question, conditioned on the latest frame.
    5. Speech engine speaks the answer.

The Whisper dependency is **optional** — if not installed we degrade
gracefully with an audible apology rather than crashing.
"""

from __future__ import annotations

import logging
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from ..events import EventBus, EventType
from ..i18n import Messages
from ..types import Priority, SpeechEvent
from .router import AiRouter

log = logging.getLogger(__name__)


class VoiceQueryService:
    """Push-to-talk voice query.

    Audio capture is intentionally minimal — production deployment should
    swap in a real audio backend (sounddevice, pyaudio). The interface
    below is the contract; implementations can be replaced.
    """

    def __init__(
        self,
        bus: EventBus,
        router: AiRouter,
        *,
        sample_rate: int = 16000,
        max_record_s: float = 5.0,
    ) -> None:
        self._bus = bus
        self._router = router
        self._sr = sample_rate
        self._max_s = max_record_s
        self._whisper: object | None = None
        self._latest_frame: np.ndarray | None = None
        self._bus.subscribe(EventType.FRAME, self._on_frame)

    def _on_frame(self, frame: object) -> None:
        # We keep only the latest frame for grounded queries.
        if hasattr(frame, "image"):
            self._latest_frame = frame.image  # type: ignore[attr-defined]

    def is_available(self) -> bool:
        try:
            import whisper  # noqa: PLC0415, F401

            return True
        except ImportError:
            return False

    def _load_whisper(self) -> None:
        if self._whisper is not None:
            return
        import whisper  # noqa: PLC0415

        log.info("Loading Whisper tiny model (offline)")
        self._whisper = whisper.load_model("tiny")

    def transcribe_wav(self, wav_path: Path) -> str:
        """Transcribe a recorded WAV file to text."""
        self._load_whisper()
        assert self._whisper is not None  # noqa: S101
        result = self._whisper.transcribe(str(wav_path), fp16=False)  # type: ignore[attr-defined]
        return str(result.get("text", "")).strip()

    def handle_query(self, audio_samples: np.ndarray, sample_rate: int) -> Optional[str]:
        """Run end-to-end: STT → AI → speech publish. Returns the answer text."""
        if self._latest_frame is None:
            self._publish(Messages.DEGRADED, Priority.HIGH)
            return None
        if not self.is_available():
            self._publish("Səs tanıma sistemi qoşulmayıb.", Priority.HIGH)
            return None

        wav_path = self._save_wav(audio_samples, sample_rate)
        try:
            question = self.transcribe_wav(wav_path)
        finally:
            wav_path.unlink(missing_ok=True)

        if not question:
            self._publish("Sualınızı eşitmədim.", Priority.NORMAL)
            return None
        log.info("User question: %s", question)
        result = self._router.query(self._latest_frame, question)
        answer = result.text or "Cavab tapmadım."
        self._publish(answer, Priority.NORMAL)
        return answer

    @staticmethod
    def _save_wav(samples: np.ndarray, sample_rate: int) -> Path:
        import tempfile  # noqa: PLC0415

        path = Path(tempfile.mktemp(suffix=".wav", prefix="vva_query_"))
        if samples.dtype != np.int16:
            samples = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())
        return path

    def _publish(self, text: str, priority: Priority) -> None:
        self._bus.publish(
            EventType.SPEECH,
            SpeechEvent(text=text, priority=priority, force=True, timestamp=time.time()),
        )
