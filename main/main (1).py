#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     VisionVoiceAsist  v4.0  —  Ağıllı Görmə Yardımçısı                    ║
║     Founder & Lead Developer : Əliəsgər                                     ║
║     Version                  : 4.0  (Cross-Platform Edition)               ║
║     Fixes: UTF-8, TTS, Audio, YOLO conf, Termux dəstəyi                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

DÜZƏLİŞLƏR (v3 → v4):
  ✦ Windows UTF-8 kodlaması düzəldildi  (???da insan → insanda)
  ✦ TTS: pyttsx3 əlavə edildi (Windows/Mac/Linux offline nitq)
  ✦ Audio: pygame ilə çarpaz platforma mp3 oynatma
  ✦ Temp fayl: /tmp yerinə tempfile modulu (Windows uyğun)
  ✦ YOLO conf: 0.48 → 0.35 (fincan, stəkan, daha çox əşya tanınır)
  ✦ ElevenLabs xəta mesajları daha aydın
  ✦ Termux (Android) dəstəyi
  ✦ --nogui rejimi tam işlək
"""

# ════════════════════════════════════════════════════════════════════════════
#  PLATFORMA / KODLAŞDİRMA DÜZƏLİŞİ — ƏN ƏVVƏL YÜKLƏN
# ════════════════════════════════════════════════════════════════════════════
import sys
import os

# Windows-da terminal UTF-8 dəstəyi
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        os.system("chcp 65001 >nul 2>&1")  # Windows konsolunu UTF-8 et
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════════════════
#  İMPORTLAR
# ════════════════════════════════════════════════════════════════════════════
import cv2
import numpy as np
from ultralytics import YOLO
import time
import requests
import threading
import queue
import logging
import base64
import json
import psutil
import argparse
import tempfile
import platform
from collections import Counter, deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from enum import Enum
from datetime import datetime
from pathlib import Path

# ── İsteğe bağlı kitabxanalar ──────────────────────────────────────────────
try:
    import google.generativeai as genai
    from PIL import Image as PILImage
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False
    print("[!] google-generativeai quraşdırılmayıb. pip install google-generativeai")

try:
    import pytesseract
    OCR_OK = True
except ImportError:
    OCR_OK = False

try:
    import pygame
    pygame.mixer.init()
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

try:
    import pyttsx3
    PYTTSX3_OK = True
except ImportError:
    PYTTSX3_OK = False

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO_OK = True
except Exception:
    GPIO_OK = False

IS_TERMUX = "com.termux" in os.environ.get("PREFIX", "")
IS_WINDOWS = sys.platform == "win32"
IS_LINUX   = sys.platform.startswith("linux")
IS_MAC     = sys.platform == "darwin"

# ════════════════════════════════════════════════════════════════════════════
#  JURNALIZASIYA
# ════════════════════════════════════════════════════════════════════════════
os.makedirs("logs", exist_ok=True)
log_file = f"logs/vva_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("VVA.Main")

# ════════════════════════════════════════════════════════════════════════════
#  KONFİQURASİYA
# ════════════════════════════════════════════════════════════════════════════
class Config:
    """Bütün tənzimləmələr bir yerdə."""

    # ── API Açarları — Buraya öz açarlarınızı yazın ──────────────────────────
    ELEVENLABS_KEY   = "SIZIN_ELEVENLABS_ACARINIZ_BURA"   # elevenlabs.io → Profile → API Key
    ELEVENLABS_VOICE = "21m00Tcm4TlvDq8ikWAM"             # Rachel (multilingual)
    GEMINI_KEY       = "SIZIN_GEMINI_ACARINIZ_BURA"       # aistudio.google.com/apikey

    # ── Kamera ─────────────────────────────────────────────────────────────
    CAM_INDEX = 0
    CAM_W     = 640
    CAM_H     = 480
    CAM_FPS   = 30

    # ── YOLO ──────────────────────────────────────────────────────────────
    YOLO_MODEL = "yolov8n.pt"
    YOLO_CONF  = 0.35   # ← 0.48 deyil! Fincan, stəkan, qaşıq da tanınır

    # ── Təhlükə Hədləri ───────────────────────────────────────────────────
    THR_CRITICAL = 0.55
    THR_WARNING  = 0.24
    THR_APPROACH = 0.035

    # ── Vaxt İntervalları (saniyə) ────────────────────────────────────────
    T_YOLO_SUMMARY = 4.0
    T_GEMINI       = 15.0
    T_OCR          = 22.0
    T_PIT          = 0.40
    T_BATTERY      = 60.0

    # ── Çuxur Aşkarlama ───────────────────────────────────────────────────
    PIT_DARK_THRESH = 36
    PIT_MIN_AREA    = 5500
    PIT_FLOOR_RATIO = 0.52

    # ── GPIO ──────────────────────────────────────────────────────────────
    GPIO_PIN              = 18
    GPIO_CRITICAL_PATTERN = [(0.15, 0.05)] * 4
    GPIO_WARNING_PATTERN  = [(0.3,  0.1)]

    # ── TTS ───────────────────────────────────────────────────────────────
    TTS_STABILITY  = 0.65
    TTS_SIMILARITY = 0.85
    TTS_STYLE      = 0.25

    # ── Yaddaş ───────────────────────────────────────────────────────────
    RECENT_SPEECH_SIZE = 7
    APPROACH_WINDOW    = 8
    SCENE_MEMORY_SIZE  = 4

    # ── Temp fayl ─────────────────────────────────────────────────────────
    TEMP_DIR = tempfile.gettempdir()     # Windows-da C:\Users\...\AppData\Local\Temp


# ════════════════════════════════════════════════════════════════════════════
#  AZƏRBAYCAN DİLİ LÜĞƏTİ — YOLOv8 COCO sinifləri
# ════════════════════════════════════════════════════════════════════════════
AZ: Dict[str, str] = {
    "person":"insan",            "bicycle":"velosiped",       "car":"maşın",
    "motorcycle":"motosiklet",   "airplane":"təyyarə",        "bus":"avtobus",
    "train":"qatar",             "truck":"yük maşını",        "boat":"qayıq",
    "traffic light":"svetofor",  "fire hydrant":"yanğın kranı","stop sign":"dayan nişanı",
    "parking meter":"parkomat",  "bench":"skamya",            "bird":"quş",
    "cat":"pişik",               "dog":"it",                  "horse":"at",
    "sheep":"qoyun",             "cow":"inək",                "elephant":"fil",
    "bear":"ayı",                "zebra":"zebra",             "giraffe":"zürafə",
    "backpack":"çanta",          "umbrella":"çətir",          "handbag":"əl çantası",
    "tie":"qalstuk",             "suitcase":"çamadan",        "frisbee":"frizbi",
    "skis":"xizək",              "snowboard":"snoubord",      "sports ball":"top",
    "kite":"çərpələng",          "baseball bat":"beysbol çubuğu",
    "baseball glove":"beysbol əlcəyi",                        "skateboard":"skeytbord",
    "surfboard":"sörf taxtası",  "tennis racket":"tennis raketi",
    "bottle":"butulka",          "wine glass":"şərab stəkanı","cup":"fincan/stəkan",
    "fork":"çəngəl",             "knife":"bıçaq",             "spoon":"qaşıq",
    "bowl":"kasa",               "banana":"banan",            "apple":"alma",
    "sandwich":"sendviç",        "orange":"portağal",         "broccoli":"brokoli",
    "carrot":"kök",              "hot dog":"hot-dog",         "pizza":"pizza",
    "donut":"donut",             "cake":"tort",               "chair":"stul",
    "couch":"divan",             "potted plant":"çiçək",      "bed":"yataq",
    "dining table":"masa",       "toilet":"tualet",           "tv":"televizor",
    "laptop":"noutbuk",          "mouse":"siçan (kompüter)",  "remote":"pult",
    "keyboard":"klaviatura",     "cell phone":"telefon",      "microwave":"mikrodalğalı soba",
    "oven":"soba",               "toaster":"toster",          "sink":"lavabo",
    "refrigerator":"soyuducu",   "book":"kitab",              "clock":"saat",
    "vase":"vaza",               "scissors":"qayçı",          "teddy bear":"oyuncaq ayı",
    "hair drier":"saç qurutma",  "toothbrush":"diş fırçası",
}

SURFACES: Set[str] = {
    "dining table", "bed", "couch", "chair", "bench", "desk", "shelf"
}

EMERGENCY_LABELS: Set[str] = {
    "person", "car", "motorcycle", "truck", "bus", "bicycle"
}


# ════════════════════════════════════════════════════════════════════════════
#  ENUM — PRİORİTET
# ════════════════════════════════════════════════════════════════════════════
class Priority(Enum):
    CRITICAL = 1
    HIGH     = 2
    NORMAL   = 3
    LOW      = 4


# ════════════════════════════════════════════════════════════════════════════
#  DATACLASS — AŞKARLAMA
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Detection:
    label_eng : str
    label_az  : str
    bbox      : Tuple[int, int, int, int]
    conf      : float
    cx        : int
    cy        : int
    area_pct  : float
    frame_id  : int = 0


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
#  GPIO VİBRASİYA MOTORU
# ════════════════════════════════════════════════════════════════════════════
class VibrationMotor:
    def __init__(self):
        self._ok = False
        if GPIO_OK:
            try:
                GPIO.setup(Config.GPIO_PIN, GPIO.OUT)
                self._ok = True
            except Exception as e:
                log.warning("GPIO xətası: %s", e)

    def pulse(self, pattern: List[Tuple[float, float]]):
        if not self._ok:
            return
        def _run():
            for on_t, off_t in pattern:
                GPIO.output(Config.GPIO_PIN, GPIO.HIGH)
                time.sleep(on_t)
                GPIO.output(Config.GPIO_PIN, GPIO.LOW)
                time.sleep(off_t)
        threading.Thread(target=_run, daemon=True).start()

    def critical(self): self.pulse(Config.GPIO_CRITICAL_PATTERN)
    def warning(self):  self.pulse(Config.GPIO_WARNING_PATTERN)


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


# ════════════════════════════════════════════════════════════════════════════
#  YOLO DETEKTOR
# ════════════════════════════════════════════════════════════════════════════
class YOLODetector:
    def __init__(self):
        log.info("YOLO modeli yüklənir: %s", Config.YOLO_MODEL)
        self._model    = YOLO(Config.YOLO_MODEL)
        self._frame_id = 0
        log.info("YOLO hazırdır.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        self._frame_id += 1
        results = self._model(
            frame,
            conf=Config.YOLO_CONF,
            verbose=False,
            device="cpu",
        )
        dets: List[Detection] = []
        h, w = frame.shape[:2]

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label_eng = self._model.names[cls]
            label_az  = AZ.get(label_eng, label_eng)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area_pct = ((x2 - x1) * (y2 - y1)) / (w * h + 1e-6)
            dets.append(Detection(
                label_eng=label_eng, label_az=label_az,
                bbox=(x1, y1, x2, y2),
                conf=float(box.conf[0]),
                cx=cx, cy=cy, area_pct=area_pct,
                frame_id=self._frame_id,
            ))
        return dets


# ════════════════════════════════════════════════════════════════════════════
#  YAXINLAŞMA İZLƏYİCİSİ
# ════════════════════════════════════════════════════════════════════════════
class ApproachTracker:
    def __init__(self):
        self._history : Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=Config.APPROACH_WINDOW)
        )
        self._alerted: Set[str] = set()

    def update(self, label: str, area_pct: float) -> bool:
        self._history[label].append(area_pct)
        hist = list(self._history[label])
        if len(hist) < 4:
            return False
        xs    = np.arange(len(hist), dtype=float)
        slope = float(np.polyfit(xs, hist, 1)[0])
        approaching = slope > Config.THR_APPROACH
        if approaching and label not in self._alerted:
            self._alerted.add(label)
            return True
        if not approaching:
            self._alerted.discard(label)
        return False


# ════════════════════════════════════════════════════════════════════════════
#  MƏKAN ANALİZATORU
# ════════════════════════════════════════════════════════════════════════════
class SpatialAnalyzer:
    @staticmethod
    def position(cx: int, w: int) -> str:
        if cx < w // 3:      return "Solunuzda"
        if cx > 2 * w // 3:  return "Saginizdə"
        return "Qarsinizda"

    @staticmethod
    def distance_label(area_pct: float) -> str:
        if area_pct > 0.45: return "cox yaxin (< 1 m)"
        if area_pct > 0.18: return "yaxin (1-2 m)"
        if area_pct > 0.06: return "orta (2-4 m)"
        return "uzaqda (4+ m)"

    @staticmethod
    def build_scene_graph(dets: List[Detection]) -> List[str]:
        relations: List[str] = []
        surfaces     = [d for d in dets if d.label_eng in SURFACES]
        non_surfaces = [d for d in dets if d.label_eng not in SURFACES]

        for surf in surfaces:
            sx1, sy1, sx2, sy2 = surf.bbox
            on_surf: List[str] = []
            for obj in non_surfaces:
                ox1, oy1, ox2, oy2 = obj.bbox
                obj_cx = (ox1 + ox2) // 2
                if sx1 < obj_cx < sx2 and sy1 < oy2 < (sy1 + (sy2 - sy1) * 0.42):
                    on_surf.append(obj.label_az)
            if on_surf:
                unique = list(dict.fromkeys(on_surf))
                if len(unique) == 1:
                    relations.append(f"{surf.label_az.capitalize()} uzərindəkitab {unique[0]} var")
                elif len(unique) <= 3:
                    relations.append(
                        f"{surf.label_az.capitalize()} uzərində "
                        f"{', '.join(unique[:-1])} və {unique[-1]} var"
                    )
                else:
                    relations.append(
                        f"{surf.label_az.capitalize()} uzərində {len(unique)} əşya var"
                    )

        people = [d for d in dets if d.label_eng == "person"]
        if len(people) >= 3:
            relations.append(f"Ətrafınızda {len(people)} nəfər var — izdiham")
        elif len(people) == 2:
            relations.append("Ətrafınızda 2 nəfər var")

        if len(dets) >= 6:
            relations.append(f"Mühit çox əşya dolu — {len(dets)} əşya aşkarlandı, ehtiyatlı olun")
        return relations


# ════════════════════════════════════════════════════════════════════════════
#  RƏNG ANALİZATORU
# ════════════════════════════════════════════════════════════════════════════
class ColorAnalyzer:
    @staticmethod
    def from_hsv(h: int, s: int, v: int) -> str:
        if v < 42:                  return "qara"
        if s < 28 and v > 168:      return "ağ"
        if s < 28:                  return "boz"
        if h < 10 or h > 165:      return "qırmızı"
        if 10 <= h < 30:            return "narıncı"
        if 30 <= h < 73:            return "sarı"
        if 73 <= h < 95:            return "açıq yaşıl"
        if 95 <= h < 130:           return "göy"
        if 130 <= h < 150:          return "mavi"
        if 150 <= h <= 165:         return "bənövşəyi"
        return "rəngli"

    @staticmethod
    def traffic_light(hsv_roi: np.ndarray) -> str:
        if hsv_roi.size == 0:        return "aşkarlandı"
        avg_v = float(np.mean(hsv_roi[:, :, 2]))
        avg_h = float(np.mean(hsv_roi[:, :, 0]))
        if avg_v < 50:               return "sönülüdür"
        if avg_h < 15 or avg_h > 160: return "QIRMIZIdır — DAYANIN"
        if 40 < avg_h < 90:          return "YAŞILdır — keçə bilərsiniz"
        if 20 < avg_h < 35:          return "SARIdır — hazırlaşın"
        return "rəngi aydın deyil"


# ════════════════════════════════════════════════════════════════════════════
#  ÇUXUR / PİLLƏKƏN AŞKARLAYICI
# ════════════════════════════════════════════════════════════════════════════
class PitDetector:
    def __init__(self):
        self._prev_mean : Optional[float] = None
        self._kernel_lg = np.ones((9, 9), np.uint8)
        self._kernel_sm = np.ones((5, 5), np.uint8)

    def detect(self, frame: np.ndarray) -> Optional[Tuple[str, Priority]]:
        h, w = frame.shape[:2]
        roi_y = int(h * Config.PIT_FLOOR_RATIO)
        floor = frame[roi_y:, :]
        gray    = cv2.cvtColor(floor, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (13, 13), 0)

        edges = cv2.Canny(blurred, 25, 90)
        edge_density = float(np.sum(edges > 0)) / (edges.size + 1e-6)

        _, dark = cv2.threshold(blurred, Config.PIT_DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
        dark    = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, self._kernel_lg)
        dark    = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  self._kernel_sm)

        combined = cv2.bitwise_or(dark, edges)
        cnts, _  = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_cnts = [c for c in cnts if cv2.contourArea(c) > Config.PIT_MIN_AREA]

        cur_mean = float(np.mean(blurred))

        result = None
        if big_cnts and edge_density > 0.12:
            result = ("DİQQƏT! Qarşınızda pilləkən var! Dərhal dayanın!", Priority.CRITICAL)
        elif big_cnts and len(big_cnts) >= 2:
            result = ("Xəbərdarlıq! Döşəmədə maneə var.", Priority.HIGH)
        elif self._prev_mean is not None:
            delta = abs(cur_mean - self._prev_mean)
            if delta > 22:
                result = ("Diqqət! Döşəmə kəskin dəyişir — astana ola bilər.", Priority.HIGH)

        self._prev_mean = cur_mean
        return result


# ════════════════════════════════════════════════════════════════════════════
#  GEMINI VİSİON AI ANALİZATORU
# ════════════════════════════════════════════════════════════════════════════
class GeminiAnalyzer:
    SYSTEM_PROMPT = (
        "Sən görmə məhdudiyyətli şəxs üçün işləyən AI görmə köməkçisisən. "
        "Şəkildəki mühiti AZƏRBAYCAN DİLİNDƏ qısa, aydın, praktik təsvir et. "
        "Ən vacib məlumatı əvvəl söylə. "
        "Məsafə, rəng, mövqe barədə dəqiq məlumat ver. "
        "Ən çox 3 cümlə. Qısa olsun."
    )

    def __init__(self):
        self._ok     = False
        self._memory : deque = deque(maxlen=Config.SCENE_MEMORY_SIZE)
        self.is_active = False

        if not GEMINI_OK:
            log.warning("Gemini kitabxanası yoxdur. pip install google-generativeai")
            return
        if not Config.GEMINI_KEY or Config.GEMINI_KEY.startswith("SIZIN"):
            log.warning("Gemini API açarı təyin edilməyib. Config.GEMINI_KEY dəyişdirin.")
            return

        try:
            genai.configure(api_key=Config.GEMINI_KEY)
            self._model = genai.GenerativeModel("gemini-1.5-flash")
            self._ok    = True
            self.is_active = True
            log.info("Gemini Vision AI aktiv.")
        except Exception as e:
            log.warning("Gemini init xətası: %s", e)

    def analyze_async(self, frame: np.ndarray, speech: "SpeechEngine"):
        if not self._ok:
            return
        f = frame.copy()
        threading.Thread(
            target=self._analyze_worker, args=(f, speech),
            daemon=True, name="GeminiThread"
        ).start()

    def _analyze_worker(self, frame: np.ndarray, speech: "SpeechEngine"):
        try:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            img_b64 = base64.b64encode(jpeg.tobytes()).decode()

            pil_img = PILImage.frombytes(
                "RGB",
                (frame.shape[1], frame.shape[0]),
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()
            )

            response = self._model.generate_content(
                [self.SYSTEM_PROMPT, pil_img],
                generation_config={"max_output_tokens": 200}
            )
            text = response.text.strip()
            if text:
                self._memory.append(text)
                speech.speak(text, Priority.NORMAL)
                log.info("Gemini analizi: %s", text[:80])
        except Exception as e:
            log.warning("Gemini analiz xətası: %s", e)


# ════════════════════════════════════════════════════════════════════════════
#  OCR MODELİ
# ════════════════════════════════════════════════════════════════════════════
class OCRModule:
    def __init__(self):
        self._ok = OCR_OK
        if not self._ok:
            log.warning("pytesseract quraşdırılmayıb. pip install pytesseract")

    def read(self, frame: np.ndarray) -> Optional[str]:
        if not self._ok:
            return None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, lang="aze+eng", config="--psm 6").strip()
            return text if len(text) > 3 else None
        except Exception as e:
            log.debug("OCR xətası: %s", e)
            return None


# ════════════════════════════════════════════════════════════════════════════
#  SİSTEM MONİTOR
# ════════════════════════════════════════════════════════════════════════════
class SystemMonitor:
    def __init__(self, speech: SpeechEngine):
        self._speech   = speech
        self._fps_buf  : deque = deque(maxlen=20)
        self._t_last   = time.time()
        self._bat_warned = False
        self.fps       = 0.0

    def tick(self):
        now = time.time()
        dt  = now - self._t_last
        self._t_last = now
        if dt > 0:
            self._fps_buf.append(1.0 / dt)
            self.fps = float(np.mean(self._fps_buf))

    def check_battery(self):
        bat = psutil.sensors_battery()
        if bat is None:
            return
        pct = int(bat.percent)
        if pct <= 10 and not self._bat_warned:
            self._speech.speak(
                f"KRİTİK! Batareya yalnız {pct} faizdir! Dərhal şarj edin!",
                Priority.CRITICAL
            )
            self._bat_warned = True
        elif pct <= 20:
            self._speech.speak(
                f"Xəbərdarlıq: Batareya {pct} faizdir. Şarj etməyi unutmayın.",
                Priority.HIGH
            )
        elif pct > 25:
            self._bat_warned = False


# ════════════════════════════════════════════════════════════════════════════
#  EKRAN ÜST QATI (OVERLAY)
# ════════════════════════════════════════════════════════════════════════════
class Overlay:
    COLORS = {
        Priority.CRITICAL : (0,   0, 255),   # qırmızı
        Priority.HIGH     : (0, 128, 255),   # narıncı
        Priority.NORMAL   : (0, 220,  80),   # yaşıl
        Priority.LOW      : (200, 200, 200), # boz
    }

    @classmethod
    def render(cls, frame: np.ndarray, dets: List[Detection],
               pit_msg: Optional[str], fps: float,
               ai_active: bool, bat_pct: Optional[int],
               summary: str, show_gui: bool = True):
        if not show_gui:
            return

        h, w = frame.shape[:2]

        # Üst panel
        cv2.rectangle(frame, (0, 0), (w, 34), (10, 10, 10), -1)
        ai_str  = "AI:ON" if ai_active else "AI:OFF"
        bat_str = f"BAT:{bat_pct}%" if bat_pct is not None else "BAT:--"
        header  = f"VisionVoiceAsist v4.0    FPS:{fps:.1f}    {ai_str}    {bat_str}"
        cv2.putText(frame, header, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 1, cv2.LINE_AA)

        # Aşkarlama qutuları
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            pri = Priority.CRITICAL if det.area_pct > Config.THR_CRITICAL else \
                  Priority.HIGH     if det.area_pct > Config.THR_WARNING  else \
                  Priority.NORMAL
            color = cls.COLORS[pri]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            dist  = SpatialAnalyzer.distance_label(det.area_pct)
            label = f"{det.label_az}  {det.conf:.0%}  {dist}"
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label) * 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

        # Çuxur/pilləkən xəbərdarlığı
        if pit_msg:
            cv2.rectangle(frame, (0, h - 42), (w, h), (0, 0, 200), -1)
            cv2.putText(frame, pit_msg[:80], (8, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Xülasə
        if summary:
            cv2.rectangle(frame, (0, h - 84), (w, h - 42), (18, 18, 18), -1)
            cv2.putText(frame, summary[:90], (8, h - 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 200), 1, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════════════════════
#  ANA KONTROLLER
# ════════════════════════════════════════════════════════════════════════════
class VisionVoiceAsist:
    def __init__(self, show_gui: bool = True):
        self.show_gui = show_gui

        # Alt sistemlər
        self.vib      = VibrationMotor()
        self.speech   = SpeechEngine(self.vib)
        self.yolo     = YOLODetector()
        self.approach = ApproachTracker()
        self.spatial  = SpatialAnalyzer()
        self.color    = ColorAnalyzer()
        self.pit      = PitDetector()
        self.ai       = GeminiAnalyzer()
        self.ocr      = OCRModule()
        self.sysmon   = SystemMonitor(self.speech)

        # Kamera
        self.cap = self._open_camera()

        # Vəziyyət
        self._last_dets    : List[Detection] = []
        self._last_summary = ""
        self._pit_msg      : Optional[str] = None
        self._bat_pct      : Optional[int] = None
        self._t = {k: 0.0 for k in ("yolo", "gemini", "ocr", "pit", "battery")}

        log.info("VisionVoiceAsist v4.0 hazırdır.")
        log.info("Platform: %s | Termux: %s | GPIO: %s | AI: %s",
                 platform.system(), IS_TERMUX, GPIO_OK, self.ai.is_active)

    def _open_camera(self) -> cv2.VideoCapture:
        backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_V4L2]
        for backend in backends:
            cap = cv2.VideoCapture(Config.CAM_INDEX, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAM_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_H)
                cap.set(cv2.CAP_PROP_FPS,          Config.CAM_FPS)
                log.info("Kamera açıldı: indeks=%d, backend=%d", Config.CAM_INDEX, backend)
                return cap
        raise RuntimeError(
            f"Kamera açılmadı! Kamera indeksi {Config.CAM_INDEX} yanlış ola bilər. "
            "--cam 1 ilə cəhd edin."
        )

    def _greet(self):
        bat = psutil.sensors_battery()
        bat_str = f"Batareya {int(bat.percent)} faizdir." if bat else ""
        ai_str  = "Süni intellekt aktiv." if self.ai.is_active else "Offline rejim."
        msg = f"Salam! VisionVoiceAsist aktiv oldu. {bat_str} {ai_str}"
        self.speech.speak(msg, Priority.HIGH, force=True)

    def _process_detections(self, dets: List[Detection], frame: np.ndarray) -> List[str]:
        h, w  = frame.shape[:2]
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        desc  : List[str] = []

        for det in dets:
            eng  = det.label_eng
            az   = det.label_az
            area = det.area_pct
            pos  = SpatialAnalyzer.position(det.cx, w)
            dist = SpatialAnalyzer.distance_label(area)

            # Yaxınlaşma yoxlaması
            if eng in EMERGENCY_LABELS and self.approach.update(eng, area):
                msg = f"DİQQƏT! {az} sürətlə yaxınlaşır! Dayanın!"
                self.speech.speak(msg, Priority.CRITICAL, force=True)

            # Kritik yaxınlıq
            if area > Config.THR_CRITICAL:
                msg = f"KRİTİK TƏHLÜKƏ! {az} çox yaxındır, dərhal dayanın!"
                self.speech.speak(msg, Priority.CRITICAL, force=True)
                desc.append(f"{pos} {az} çox yaxın (< 1 m).")
                continue

            # Svetofor
            if eng == "traffic light":
                x1, y1, x2, y2 = det.bbox
                roi = hsv[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                renk = ColorAnalyzer.traffic_light(roi)
                desc.append(f"Svetofor {renk}")

            # Yıxılmış insan
            elif eng == "person":
                bw = det.bbox[2] - det.bbox[0]
                bh = det.bbox[3] - det.bbox[1]
                if bw > 0 and bh > 0 and (bw / bh) > 2.2 and area > 0.05:
                    desc.append(f"{pos} yıxılmış insan ola bilər — yardım lazım ola bilər")
                    self.speech.speak(
                        f"DİQQƏT! {pos} yıxılmış insan görünür!",
                        Priority.CRITICAL, force=True
                    )
                else:
                    desc.append(f"{pos} insan ({dist})")

            else:
                cy, cx = min(det.cy, h - 1), min(det.cx, w - 1)
                hv = int(hsv[cy, cx][0])
                sv = int(hsv[cy, cx][1])
                vv = int(hsv[cy, cx][2])
                col = ColorAnalyzer.from_hsv(hv, sv, vv)
                desc.append(f"{pos} {col} {az} ({dist})")

        desc.extend(SpatialAnalyzer.build_scene_graph(dets))
        return desc

    @staticmethod
    def _summarize(descriptions: List[str]) -> str:
        if not descriptions:
            return ""
        counts = Counter(descriptions)
        parts  = []
        for text, cnt in counts.most_common(4):
            parts.append(f"{cnt} ədəd {text}" if cnt > 1 else text)
        return ". ".join(parts) + "."

    # ── Əsas Dövrə ────────────────────────────────────────────────────────
    def run(self):
        self._greet()
        log.info("Əsas dövrə başladı. [Q] çıxış, [R] AI analiz, [S] status")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                self.sysmon.tick()
                now = time.time()

                # Çuxur / Pilləkən
                if now - self._t["pit"] >= Config.T_PIT:
                    result = self.pit.detect(frame)
                    self._pit_msg = result[0] if result else None
                    if result:
                        self.speech.speak(result[0], result[1], force=True)
                    self._t["pit"] = now

                # YOLO + Xülasə
                if now - self._t["yolo"] >= Config.T_YOLO_SUMMARY:
                    dets         = self.yolo.detect(frame)
                    self._last_dets = dets
                    descriptions = self._process_detections(dets, frame)
                    summary      = self._summarize(descriptions)

                    if summary and summary != self._last_summary:
                        self.speech.speak(summary, Priority.NORMAL)
                        self._last_summary = summary
                    elif not dets:
                        self.speech.speak(
                            "Qarşınızda açıq yol var, maneə aşkarlanmadı.",
                            Priority.NORMAL
                        )
                    self._t["yolo"] = now

                # Gemini AI
                if now - self._t["gemini"] >= Config.T_GEMINI:
                    self.ai.analyze_async(frame, self.speech)
                    self._t["gemini"] = now

                # OCR
                if now - self._t["ocr"] >= Config.T_OCR:
                    text = self.ocr.read(frame)
                    if text:
                        self.speech.speak(f"Yazı oxuyuram: {text}", Priority.LOW)
                    self._t["ocr"] = now

                # Batareya
                if now - self._t["battery"] >= Config.T_BATTERY:
                    self.sysmon.check_battery()
                    bat = psutil.sensors_battery()
                    self._bat_pct = int(bat.percent) if bat else None
                    self._t["battery"] = now

                # Overlay
                Overlay.render(
                    frame, self._last_dets, self._pit_msg,
                    self.sysmon.fps, self.ai.is_active,
                    self._bat_pct, self._last_summary,
                    show_gui=self.show_gui,
                )

                if self.show_gui:
                    cv2.imshow("VisionVoiceAsist v4.0 | Əliəsgər", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("r"):
                        self.speech.speak("Mühit dərindən analiz edilir.", Priority.HIGH, force=True)
                        self.ai.analyze_async(frame, self.speech)
                        self._t["gemini"] = now
                    elif key == ord("s"):
                        self._status_report()

        except KeyboardInterrupt:
            log.info("Klaviatura ilə dayandırıldı.")
        finally:
            self._shutdown()

    def _status_report(self):
        bat   = psutil.sensors_battery()
        b_msg = f"Batareya {int(bat.percent)} faizdir." if bat else "Qidalanma bloku."
        st    = self.speech.stats()
        msg   = (
            f"Sistem normal işləyir. {b_msg} "
            f"FPS: {self.sysmon.fps:.1f}. "
            f"Cəmi {st['total']} mesaj söylənildi. "
            f"AI {'aktiv' if self.ai.is_active else 'deaktiv'}."
        )
        self.speech.speak(msg, Priority.HIGH, force=True)

    def _shutdown(self):
        self.speech.speak(
            "VisionVoiceAsist bağlanır. Diqqətiniz üçün təşəkkür edirəm. Hər şey yaxşı olsun!",
            Priority.HIGH, force=True
        )
        time.sleep(2)
        self.cap.release()
        if self.show_gui:
            cv2.destroyAllWindows()
        if GPIO_OK:
            GPIO.cleanup()
        log.info("Sistem tam dayandırıldı. Jurnal: %s", log_file)


# ════════════════════════════════════════════════════════════════════════════
#  GİRİŞ NÖQTƏSİ
# ════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="VisionVoiceAsist v4.0 — Görmə Məhdudiyyətli Şəxslər üçün AI"
    )
    p.add_argument("--cam",   type=int,   default=Config.CAM_INDEX,
                   help="Kamera indeksi (defolt: 0)")
    p.add_argument("--conf",  type=float, default=Config.YOLO_CONF,
                   help="YOLO etibarlılıq həddi (defolt: 0.35)")
    p.add_argument("--noai",  action="store_true", help="AI olmadan işlə")
    p.add_argument("--nogui", action="store_true",
                   help="Ekran göstərmə (Termux / SSH üçün)")
    p.add_argument("--gemini-key", type=str, default=None,
                   help="Gemini API açarı (Config-i əvəz edir)")
    p.add_argument("--elevenlabs-key", type=str, default=None,
                   help="ElevenLabs API açarı")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    Config.CAM_INDEX = args.cam
    Config.YOLO_CONF = args.conf

    if args.noai:
        Config.GEMINI_KEY = "AIzaSyCrd1qyGyHyf9XWMZD3_z9d1DzhF56mM20"
    if args.gemini_key:
        Config.GEMINI_KEY = args.gemini_key
    if args.elevenlabs_key:
        Config.ELEVENLABS_KEY = args.elevenlabs_key

    show_gui = not args.nogui

    print("""
╔══════════════════════════════════════════════════════════╗
║    VisionVoiceAsist v4.0  —  Cross-Platform Edition     ║
║    Founder & Developer: Əliəsgər                        ║
╠══════════════════════════════════════════════════════════╣
║  [Q] Çıxış           [R] Dərhal AI analizi             ║
║  [S] Sistem statusu                                     ║
╠══════════════════════════════════════════════════════════╣
║  Termux:  python main_v4.py --nogui                     ║
║  No AI:   python main_v4.py --noai                      ║
║  Custom:  python main_v4.py --gemini-key KEY            ║
╚══════════════════════════════════════════════════════════╝
""")

    VisionVoiceAsist(show_gui=show_gui).run()