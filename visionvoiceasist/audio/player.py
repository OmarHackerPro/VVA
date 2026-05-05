"""Cross-platform MP3 / WAV playback — no shell-injection footguns."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from ..utils.safe_subprocess import have, run

log = logging.getLogger(__name__)


class AudioPlayer:
    """Plays audio files. Tries pygame first; falls back to a system player.

    Note: pygame's mixer is intentionally NOT initialised at import time —
    that would fail on headless boxes without an audio device. It's lazily
    initialised on first :py:meth:`play` call.
    """

    def __init__(self) -> None:
        self._pygame_ready = False
        self._tried_pygame = False

    def _ensure_pygame(self) -> bool:
        if self._tried_pygame:
            return self._pygame_ready
        self._tried_pygame = True
        try:
            import pygame  # noqa: PLC0415

            pygame.mixer.init()
            self._pygame = pygame
            self._pygame_ready = True
            log.debug("pygame mixer initialised")
        except Exception as exc:  # noqa: BLE001
            log.info("pygame unavailable (%s); using system player", exc)
        return self._pygame_ready

    def play(self, path: Path, *, pan: float = 0.0) -> None:
        """Play *path* synchronously (blocks until finished).

        Args:
            path: Audio file path (WAV / MP3).
            pan: Stereo pan in [-1, 1]. Only honoured by pygame.
        """
        if self._ensure_pygame():
            self._play_pygame(path, pan)
            return
        self._play_system(path)

    def _play_pygame(self, path: Path, pan: float) -> None:
        self._pygame.mixer.music.load(str(path))  # type: ignore[attr-defined]
        # Stereo pan via volume balance (rough HRTF substitute).
        if hasattr(self._pygame.mixer, "Channel"):
            try:
                ch = self._pygame.mixer.Channel(0)
                left = max(0.0, 1.0 - max(0.0, pan))
                right = max(0.0, 1.0 + min(0.0, pan))
                ch.set_volume(left, right)
            except Exception:  # noqa: BLE001
                pass
        self._pygame.mixer.music.play()
        while self._pygame.mixer.music.get_busy():
            time.sleep(0.05)

    @staticmethod
    def _play_system(path: Path) -> None:
        if sys.platform == "win32":
            try:
                import winsound  # noqa: PLC0415

                # winsound only does WAV reliably; for MP3 we'd need ffmpeg.
                if path.suffix.lower() == ".wav":
                    winsound.PlaySound(str(path), winsound.SND_FILENAME)
                    return
            except ImportError:
                pass
            log.warning("No audio backend on Windows for %s", path)
            return
        if sys.platform == "darwin" and have("afplay"):
            run(["afplay", str(path)], timeout=15.0)
            return
        if have("mpg123") and path.suffix.lower() == ".mp3":
            run(["mpg123", "-q", str(path)], timeout=15.0)
            return
        if have("ffplay"):
            run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
                timeout=15.0)
            return
        if have("aplay") and path.suffix.lower() == ".wav":
            run(["aplay", "-q", str(path)], timeout=15.0)
            return
        log.warning("No audio player available on this system")
