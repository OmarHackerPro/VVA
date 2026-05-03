#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import logging
import tempfile
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from enum import Enum
from datetime import datetime
from pathlib import Path
import json

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
