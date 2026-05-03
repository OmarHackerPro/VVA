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
import time
import psutil
import argparse
import platform
from collections import Counter
from typing import Optional, List

from config import Config, Priority, Detection, GPIO_OK, IS_TERMUX, log, log_file
from hardware.glasses import VibrationMotor
from audio.speaker import SpeechEngine
from vision.camera import open_camera
from vision.detector import (
    YOLODetector, ApproachTracker, SpatialAnalyzer,
    ColorAnalyzer, PitDetector, GeminiAnalyzer, OCRModule,
)
from utils import SystemMonitor, Overlay
from config import EMERGENCY_LABELS

if GPIO_OK:
    import RPi.GPIO as GPIO

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
        return open_camera()

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
