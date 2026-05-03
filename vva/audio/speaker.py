#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import queue
import threading
import requests
from dataclasses import dataclass, field
from collections import deque
from typing import Dict

from config import Config, Priority, PYGAME_OK, PYTTSX3_OK, IS_TERMUX, IS_WINDOWS, IS_MAC, log
from hardware.glasses import VibrationMotor

if PYGAME_OK:
    import pygame

if PYTTSX3_OK:
    import pyttsx3

# ════════════════════════════════════════════════════════════════════════════
#  AUDIO OYNADICISI — Çarpaz Platforma (Windows / Linux / Mac / Termux)
# ════════════════════════════════════════════════════════════════════════════
class AudioPlayer:
    """
    Platformaya uyğun audio oynatma:
      Windows  → pygame.mixer
      Linux    → mpg123 → aplay
      Termux   → termux-media-player
      Mac      → afplay
    """

    @staticmethod
    def play_mp3(path: str):
        try:
            if PYGAME_OK:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                return

            if IS_WINDOWS:
                import winsound
                # mp3 → wav lazımdır; pygame yoxdursa playsound cəhd et
                try:
                    from playsound import playsound
                    playsound(path)
                    return
                except ImportError:
                    pass
                os.system(f'start /wait "" "{path}" >nul 2>&1')

            elif IS_TERMUX:
                os.system(f"termux-media-player play {path}")
                time.sleep(2)  # oxuma qurtarana qədər gözlə

            elif IS_MAC:
                os.system(f"afplay {path}")

            else:  # Linux
                ret = os.system(f"mpg123 -q {path} 2>/dev/null")
                if ret != 0:
                    os.system(f"cvlc --play-and-exit {path} 2>/dev/null")

        except Exception as e:
            log.warning("Audio oxuma xətası: %s", e)


# ════════════════════════════════════════════════════════════════════════════
#  NİTQ MÜHƏRRİKİ — ElevenLabs → pyttsx3 → eSpeak
# ════════════════════════════════════════════════════════════════════════════
@dataclass(order=True)
class _Task:
    priority : int
    seq      : int
    text     : str  = field(compare=False)
    force    : bool = field(compare=False, default=False)


class SpeechEngine:
    """
    Prioritet əsaslı TTS:
      1) ElevenLabs (bulud, yüksək keyfiyyət)
      2) pyttsx3    (offline, Windows/Mac/Linux)
      3) eSpeak     (offline, Linux/Termux)
      4) Termux TTS (termux-tts-speak)
    """

    def __init__(self, vibration: VibrationMotor):
        self._q      : queue.PriorityQueue = queue.PriorityQueue()
        self._recent : deque = deque(maxlen=Config.RECENT_SPEECH_SIZE)
        self._seq    = 0
        self._vib    = vibration
        self._lock   = threading.Lock()
        self._stats  = {"total": 0, "elevenlabs": 0, "pyttsx3": 0, "espeak": 0, "skipped": 0}

        # pyttsx3 mühərriki (bir dəfə yarad, thread-safe saxla)
        self._pyttsx3_engine = None
        if PYTTSX3_OK:
            try:
                eng = pyttsx3.init()
                eng.setProperty("rate", 155)
                eng.setProperty("volume", 1.0)
                # Azərbaycan səsi yoxdursa ingilis istifadə et
                self._pyttsx3_engine = eng
                log.info("pyttsx3 TTS hazırdır.")
            except Exception as e:
                log.warning("pyttsx3 init xətası: %s", e)

        worker = threading.Thread(target=self._loop, daemon=True, name="SpeechWorker")
        worker.start()
        log.info("SpeechEngine hazırdır.")

    def speak(self, text: str, priority: Priority = Priority.NORMAL, force: bool = False):
        if not text:
            return
        with self._lock:
            if not force and text in self._recent:
                self._stats["skipped"] += 1
                return
            self._seq += 1
            task = _Task(priority=priority.value, seq=self._seq, text=text, force=force)

        if priority == Priority.CRITICAL:
            self._drain()
            self._vib.critical()
        elif priority == Priority.HIGH:
            self._vib.warning()

        self._q.put(task)

    def stats(self) -> Dict:
        return dict(self._stats)

    def _drain(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                break

    def _loop(self):
        while True:
            try:
                task: _Task = self._q.get(timeout=1)
                with self._lock:
                    self._recent.append(task.text)
                    self._stats["total"] += 1
                self._say(task.text)
                self._q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                log.error("SpeechWorker xətası: %s", e)

    def _say(self, text: str):
        # 1) ElevenLabs (bulud)
        if Config.ELEVENLABS_KEY and not Config.ELEVENLABS_KEY.startswith("SIZIN"):
            try:
                self._elevenlabs(text)
                self._stats["elevenlabs"] += 1
                return
            except Exception as e:
                log.warning("ElevenLabs xətası: %s | Offline TTS istifadə olunur.", e)

        # 2) pyttsx3 (offline, çarpaz platforma)
        if self._pyttsx3_engine:
            try:
                self._pyttsx3_speak(text)
                self._stats["pyttsx3"] += 1
                return
            except Exception as e:
                log.warning("pyttsx3 xətası: %s", e)

        # 3) Termux TTS
        if IS_TERMUX:
            try:
                os.system(f'termux-tts-speak -l az "{text}" 2>/dev/null')
                self._stats["espeak"] += 1
                return
            except Exception:
                pass

        # 4) eSpeak (Linux)
        self._espeak(text)
        self._stats["espeak"] += 1

    def _elevenlabs(self, text: str):
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{Config.ELEVENLABS_VOICE}"
        headers = {
            "xi-api-key": Config.ELEVENLABS_KEY,
            "Content-Type": "application/json"
        }
        body = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability":        Config.TTS_STABILITY,
                "similarity_boost": Config.TTS_SIMILARITY,
                "style":            Config.TTS_STYLE,
                "use_speaker_boost": True,
            },
        }
        r = requests.post(url, headers=headers, json=body, timeout=10)
        if r.status_code == 401:
            raise RuntimeError("API açarı yanlışdır — elevenlabs.io-dan yeni açar alın")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:100]}")

        tmp = os.path.join(Config.TEMP_DIR, "vva_audio.mp3")
        with open(tmp, "wb") as f:
            f.write(r.content)
        AudioPlayer.play_mp3(tmp)

    def _pyttsx3_speak(self, text: str):
        """pyttsx3 tək thread-dən istifadə edilməlidir."""
        engine = self._pyttsx3_engine
        engine.say(text)
        engine.runAndWait()

    @staticmethod
    def _espeak(text: str):
        safe = text.replace('"', "'").replace("`", "'")
        ret = os.system(f'espeak -v az -s 155 -a 200 "{safe}" 2>/dev/null')
        if ret != 0:
            os.system(f'espeak -s 155 -a 200 "{safe}" 2>/dev/null')
