#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from typing import List, Tuple

from config import Config, GPIO_OK, log

if GPIO_OK:
    import RPi.GPIO as GPIO

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
