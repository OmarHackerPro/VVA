"""OpenCV HUD overlay — only redraws when state actually changes."""

from __future__ import annotations

import hashlib
from typing import Optional

import cv2
import numpy as np

from ..i18n import distance_label
from ..settings import ThresholdSettings
from ..types import Detection, Priority


class Overlay:
    """Renders detection boxes, header, pit warning, summary, FPS.

    Optimisation: keeps a hash of the last rendered state and skips redraws
    when nothing changed (frees CPU cycles for vision work).
    """

    COLORS = {
        Priority.CRITICAL: (0, 0, 255),
        Priority.HIGH: (0, 128, 255),
        Priority.NORMAL: (0, 220, 80),
        Priority.LOW: (200, 200, 200),
    }

    def __init__(self, thresholds: ThresholdSettings) -> None:
        self._th = thresholds
        self._last_hash: Optional[str] = None

    def render(
        self,
        frame: np.ndarray,
        *,
        detections: list[Detection],
        pit_message: Optional[str],
        summary: str,
        fps: float,
        ai_active: bool,
        battery_pct: Optional[int],
        mode: str,
    ) -> bool:
        """Mutate *frame* in place. Returns True if anything was redrawn."""
        state_hash = self._compute_hash(detections, pit_message, summary,
                                         ai_active, battery_pct, mode)
        if state_hash == self._last_hash and fps:
            # State unchanged: only refresh the FPS counter (cheap).
            self._draw_header(frame, fps, ai_active, battery_pct, mode)
            return False
        self._last_hash = state_hash

        self._draw_header(frame, fps, ai_active, battery_pct, mode)
        self._draw_detections(frame, detections)
        self._draw_pit(frame, pit_message)
        self._draw_summary(frame, summary)
        return True

    def _draw_header(
        self,
        frame: np.ndarray,
        fps: float,
        ai_active: bool,
        battery_pct: Optional[int],
        mode: str,
    ) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 34), (10, 10, 10), -1)
        ai_str = f"AI:{mode.upper()}" if ai_active else "AI:OFF"
        bat_str = f"BAT:{battery_pct}%" if battery_pct is not None else "BAT:--"
        header = f"VisionVoiceAsist v5  FPS:{fps:.1f}  {ai_str}  {bat_str}"
        cv2.putText(frame, header, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 1, cv2.LINE_AA)

    def _draw_detections(self, frame: np.ndarray, dets: list[Detection]) -> None:
        for det in dets:
            x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
            pri = (Priority.CRITICAL if det.area_pct > self._th.critical_area
                   else Priority.HIGH if det.area_pct > self._th.warning_area
                   else Priority.NORMAL)
            color = self.COLORS[pri]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"{det.label_az} {det.conf:.0%} {distance_label(det.area_pct)}"
            tag_w = min(len(tag) * 8, frame.shape[1])
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + tag_w, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 2, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_pit(self, frame: np.ndarray, message: Optional[str]) -> None:
        if not message:
            return
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 42), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, message[:80], (8, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_summary(self, frame: np.ndarray, summary: str) -> None:
        if not summary:
            return
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 84), (w, h - 42), (18, 18, 18), -1)
        cv2.putText(frame, summary[:90], (8, h - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 200), 1, cv2.LINE_AA)

    @staticmethod
    def _compute_hash(
        detections: list[Detection],
        pit_message: Optional[str],
        summary: str,
        ai_active: bool,
        battery_pct: Optional[int],
        mode: str,
    ) -> str:
        h = hashlib.md5(usedforsecurity=False)
        for d in detections:
            h.update(f"{d.label_eng}:{d.bbox.x1},{d.bbox.y1},"
                     f"{d.bbox.x2},{d.bbox.y2},{d.conf:.2f}|".encode())
        h.update(f"|{pit_message}|{summary}|{ai_active}|{battery_pct}|{mode}".encode())
        return h.hexdigest()
