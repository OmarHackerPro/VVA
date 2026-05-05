"""TTS provider strategy.

Each provider implements :py:class:`TtsProvider`. The :py:class:`SpeechEngine`
tries them in priority order until one succeeds.

Strategy hierarchy (best to fallback):
    1. ElevenLabs (cloud, best quality, multilingual)
    2. pyttsx3 (offline, all platforms, OS-native voices)
    3. eSpeak (offline, Linux/Termux, lowest fidelity but always works)
"""

from __future__ import annotations

import logging
import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import requests

from ..settings import TtsSettings
from ..utils.retry import with_retry
from ..utils.safe_subprocess import have, run
from .player import AudioPlayer

log = logging.getLogger(__name__)


class TtsProvider(ABC):
    """Abstract TTS provider. Either synthesises to file *or* speaks directly."""

    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider can serve a request right now."""

    @abstractmethod
    def speak(self, text: str, *, pan: float = 0.0) -> bool:
        """Synthesise + play *text*. Return True on success."""


# ── ElevenLabs (cloud) ─────────────────────────────────────────────────────
class ElevenLabsProvider(TtsProvider):
    """Cloud TTS — synthesises to MP3 then plays via AudioPlayer."""

    name = "elevenlabs"
    URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"

    def __init__(
        self, settings: TtsSettings, api_key: str, player: AudioPlayer, temp_dir: Path
    ) -> None:
        self._s = settings
        self._key = api_key
        self._player = player
        self._tmp = temp_dir

    def is_available(self) -> bool:
        return bool(self._key)

    def speak(self, text: str, *, pan: float = 0.0) -> bool:
        if not self.is_available():
            return False
        try:
            mp3_path = self._synthesize(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("ElevenLabs synthesis failed: %s", exc)
            return False
        try:
            self._player.play(mp3_path, pan=pan)
        finally:
            mp3_path.unlink(missing_ok=True)
        return True

    @with_retry(attempts=3, base_delay=0.5, max_delay=4.0,
                exceptions=(requests.RequestException, RuntimeError))
    def _synthesize(self, text: str) -> Path:
        url = self.URL.format(voice=self._s.elevenlabs_voice)
        headers = {"xi-api-key": self._key, "Content-Type": "application/json"}
        body = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": self._s.stability,
                "similarity_boost": self._s.similarity,
                "style": self._s.style,
                "use_speaker_boost": True,
            },
        }
        r = requests.post(url, headers=headers, json=body, timeout=self._s.http_timeout_s)
        if r.status_code == 401:
            raise RuntimeError("ElevenLabs API key rejected (HTTP 401)")
        if r.status_code != 200:
            raise RuntimeError(f"ElevenLabs HTTP {r.status_code}: {r.text[:120]}")

        fd, name = tempfile.mkstemp(suffix=".mp3", dir=str(self._tmp), prefix="vva_")
        path = Path(name)
        try:
            with open(fd, "wb") as f:
                f.write(r.content)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path


# ── pyttsx3 (offline, cross-platform) ──────────────────────────────────────
class Pyttsx3Provider(TtsProvider):
    """Offline TTS via pyttsx3. Runs in its own thread to be thread-safe."""

    name = "pyttsx3"

    def __init__(self) -> None:
        self._engine: object | None = None
        self._lock = threading.Lock()
        self._init_attempted = False

    def is_available(self) -> bool:
        if not self._init_attempted:
            self._init()
        return self._engine is not None

    def _init(self) -> None:
        self._init_attempted = True
        try:
            import pyttsx3  # noqa: PLC0415

            engine = pyttsx3.init()
            engine.setProperty("rate", 155)
            engine.setProperty("volume", 1.0)
            self._engine = engine
            log.info("pyttsx3 ready")
        except Exception as exc:  # noqa: BLE001
            log.info("pyttsx3 unavailable: %s", exc)

    def speak(self, text: str, *, pan: float = 0.0) -> bool:  # noqa: ARG002
        if not self.is_available():
            return False
        # pyttsx3 is not thread-safe — serialise.
        with self._lock:
            try:
                assert self._engine is not None  # noqa: S101
                self._engine.say(text)  # type: ignore[attr-defined]
                self._engine.runAndWait()  # type: ignore[attr-defined]
                return True
            except Exception as exc:  # noqa: BLE001
                log.warning("pyttsx3 speak failed: %s", exc)
                return False


# ── eSpeak (offline, Linux/Termux) ─────────────────────────────────────────
class EspeakProvider(TtsProvider):
    """Linux fallback TTS via eSpeak. Always uses argv list — no shell."""

    name = "espeak"

    def is_available(self) -> bool:
        return have("espeak") or have("espeak-ng")

    def speak(self, text: str, *, pan: float = 0.0) -> bool:  # noqa: ARG002
        if not text:
            return False
        binary = "espeak-ng" if have("espeak-ng") else "espeak"
        argv = [binary, "-v", "az", "-s", "155", "-a", "200", text]
        try:
            res = run(argv, timeout=15.0)
        except FileNotFoundError:
            return False
        if res.returncode != 0:
            # Fallback: try without language flag (default voice).
            res = run([binary, "-s", "155", "-a", "200", text], timeout=15.0)
        return res.returncode == 0


# ── Termux TTS (Android) ───────────────────────────────────────────────────
class TermuxTtsProvider(TtsProvider):
    """Android Termux native TTS — `termux-tts-speak`."""

    name = "termux"

    def is_available(self) -> bool:
        return have("termux-tts-speak")

    def speak(self, text: str, *, pan: float = 0.0) -> bool:  # noqa: ARG002
        if not text:
            return False
        try:
            res = run(["termux-tts-speak", "-l", "az", text], timeout=15.0)
        except FileNotFoundError:
            return False
        return res.returncode == 0


# ── Beep-only emergency fallback ───────────────────────────────────────────
class BeepFallbackProvider(TtsProvider):
    """Last-resort audible alert when no TTS engine is available.

    Generates a square-wave beep via the AudioPlayer. This is what protects
    the user from silent failure when every real TTS engine is down.
    """

    name = "beep"

    def __init__(self, player: AudioPlayer, temp_dir: Path) -> None:
        self._player = player
        self._tmp = temp_dir

    def is_available(self) -> bool:
        return True  # Always available; we synthesise the beep ourselves.

    def speak(self, text: str, *, pan: float = 0.0) -> bool:  # noqa: ARG002
        wav = self._tmp / "vva_beep.wav"
        try:
            self._make_beep(wav)
            self._player.play(wav, pan=pan)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Beep fallback failed: %s", exc)
            return False
        finally:
            wav.unlink(missing_ok=True)

    @staticmethod
    def _make_beep(path: Path, freq_hz: int = 880, duration_s: float = 0.4) -> None:
        import math  # noqa: PLC0415
        import struct  # noqa: PLC0415
        import wave  # noqa: PLC0415

        sr = 22050
        n = int(sr * duration_s)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            for i in range(n):
                v = int(32767 * 0.4 * math.sin(2 * math.pi * freq_hz * i / sr))
                wf.writeframesraw(struct.pack("<h", v))
