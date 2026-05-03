#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import psutil
import time
from collections import deque
from typing import Optional, List

from config import Config, Priority, Detection, log
from vision.detector import SpatialAnalyzer
from audio.speaker import SpeechEngine

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
