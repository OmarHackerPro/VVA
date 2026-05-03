#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import threading
import base64
from collections import deque, defaultdict
from typing import Optional, List, Dict, Tuple, Set

from ultralytics import YOLO

from config import (
    Config, AZ, SURFACES, EMERGENCY_LABELS, Priority, Detection,
    GEMINI_OK, OCR_OK, log,
)

if GEMINI_OK:
    import google.generativeai as genai
    from PIL import Image as PILImage

if OCR_OK:
    import pytesseract

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
