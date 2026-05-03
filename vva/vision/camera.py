#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

from config import Config, log

# ════════════════════════════════════════════════════════════════════════════
#  KAMERA — Çərçivə Tutma
# ════════════════════════════════════════════════════════════════════════════
def open_camera() -> cv2.VideoCapture:
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
