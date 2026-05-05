"""Runtime orchestrator — wires every subsystem onto the event bus.

This is the only place where high-level coordination lives. Every sub-module
is independently testable; this file is the *only* one that knows about
all of them.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Optional

import cv2
import numpy as np

from .ai import AiRouter, GeminiVlm, LocalVlm, VoiceQueryService
from .audio import SpeechEngine
from .events import EventBus, EventType
from .hardware import HapticMotor, ImuSensor
from .health import HealthMonitor, liveness_probe
from .i18n import EMERGENCY_LABELS, Messages, az_label
from .monitoring import BatteryWatcher, FpsCounter
from .settings import Settings
from .types import (
    Detection,
    Frame,
    HapticEvent,
    HealthReport,
    OperatingMode,
    Priority,
    SpeechEvent,
)
from .ui import Overlay
from .vision import (
    ApproachTracker,
    CameraSource,
    ColorAnalyzer,
    ObjectTracker,
    OcrModule,
    PitDetector,
    SpatialAnalyzer,
    StatefulNarrator,
    YoloDetector,
)
from .audio.spatial import SpatialPanner

log = logging.getLogger(__name__)


class Runtime:
    """Top-level coordinator. Construct with a Settings; call :py:meth:`run`."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._bus = EventBus()

        # Vision
        self._camera = CameraSource(settings.camera)
        self._yolo = YoloDetector(settings.yolo)
        self._approach = ApproachTracker(settings.thresholds)
        self._tracker = ObjectTracker()
        self._narrator = StatefulNarrator()
        self._pit = PitDetector(settings.thresholds)
        self._ocr = OcrModule()
        self._panner = SpatialPanner(settings.camera.width)

        # AI
        cloud = GeminiVlm(settings.ai)
        offline = LocalVlm(settings.ai)
        self._router = AiRouter(settings.ai, self._bus, cloud=cloud, offline=offline)
        self._voice_query = VoiceQueryService(self._bus, self._router)

        # Audio + haptics
        self._speech = SpeechEngine(settings, self._bus)
        self._haptics = HapticMotor(settings.gpio, self._bus)
        self._imu = ImuSensor()

        # Monitoring + UI
        self._fps = FpsCounter()
        self._battery = BatteryWatcher(self._bus)
        self._overlay = Overlay(settings.thresholds)
        self._health = HealthMonitor(self._bus, settings.timing.health_s)

        # Optional dashboard
        self._dashboard: object | None = None

        # State
        self._last_dets: list[Detection] = []
        self._last_summary = ""
        self._pit_msg: Optional[str] = None
        self._battery_pct: Optional[int] = None
        self._beats: dict[str, list[float]] = {
            "yolo": [], "ai": [], "pit": [], "ocr": [], "battery": [],
        }
        self._timers: dict[str, float] = {k: 0.0 for k in self._beats}

        self._wire_bus()
        self._register_health_probes()

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def run(self) -> None:
        log.info("Runtime starting (offline_mode=%s)", self._s.ai.offline_mode)
        self._speech.start()
        self._health.start()
        self._bus.subscribe(EventType.SPEECH, self._speech.enqueue)

        try:
            self._camera.open()
        except Exception:
            log.exception("Camera open failed")
            self._bus.publish(EventType.SPEECH,
                              SpeechEvent(text=Messages.CAMERA_LOST,
                                          priority=Priority.CRITICAL, force=True))
            return

        self._maybe_start_dashboard()
        self._yolo.warm_up()
        self._greet()
        self._main_loop()

    def shutdown(self) -> None:
        log.info("Runtime shutdown")
        self._bus.publish(
            EventType.SPEECH,
            SpeechEvent(text=Messages.SHUTDOWN, priority=Priority.HIGH, force=True),
        )
        time.sleep(1.5)
        self._health.stop()
        self._speech.stop()
        self._camera.close()
        if self._s.show_gui:
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
        self._bus.publish(EventType.SHUTDOWN, None)

    # ── Wiring ────────────────────────────────────────────────────────────
    def _wire_bus(self) -> None:
        # Bridge "approach alert" → speech + haptic.
        self._bus.subscribe(EventType.APPROACH_ALERT, self._on_approach)
        self._bus.subscribe(EventType.PIT_WARNING, self._on_pit)
        self._bus.subscribe(EventType.AI_DESCRIPTION,
                            lambda text: self._bus.publish(
                                EventType.SPEECH,
                                SpeechEvent(text=text, priority=Priority.NORMAL),
                            ))
        self._bus.subscribe(EventType.OCR_TEXT,
                            lambda text: self._bus.publish(
                                EventType.SPEECH,
                                SpeechEvent(
                                    text=Messages.CLEAR_TEXT.format(text=text),
                                    priority=Priority.LOW),
                            ))

    def _register_health_probes(self) -> None:
        self._health.register("speech", self._speech.health)
        self._health.register("haptics", self._haptics.health)
        self._health.register("camera", self._camera_probe)
        self._health.register("yolo",
                              liveness_probe("yolo", self._beats["yolo"], stale_after_s=20))
        self._health.register("ai",
                              liveness_probe("ai", self._beats["ai"], stale_after_s=120))

    def _camera_probe(self) -> HealthReport:
        return HealthReport(
            component="camera", healthy=self._camera.is_alive(),
            detail="alive" if self._camera.is_alive() else "stale",
        )

    # ── Bus handlers ──────────────────────────────────────────────────────
    def _on_approach(self, payload: tuple[str, Priority]) -> None:
        label_eng, priority = payload
        self._bus.publish(
            EventType.SPEECH,
            SpeechEvent(text=Messages.APPROACH.format(label=az_label(label_eng)),
                        priority=priority, force=True),
        )
        self._bus.publish(EventType.HAPTIC, HapticEvent(priority=priority))

    def _on_pit(self, payload: tuple[str, Priority]) -> None:
        message, priority = payload
        self._bus.publish(
            EventType.SPEECH,
            SpeechEvent(text=message, priority=priority, force=True),
        )
        self._bus.publish(EventType.HAPTIC, HapticEvent(priority=priority))

    # ── Greeting ──────────────────────────────────────────────────────────
    def _greet(self) -> None:
        bat = self._battery.check()
        bat_msg = Messages.GREETING_BATTERY.format(pct=bat) if bat is not None else ""
        ai_msg = (Messages.AI_ONLINE if self._router.mode is OperatingMode.ONLINE
                  else Messages.AI_OFFLINE)
        text = " ".join(filter(None, [Messages.GREETING, bat_msg, ai_msg]))
        self._bus.publish(EventType.SPEECH,
                          SpeechEvent(text=text, priority=Priority.HIGH, force=True))

    # ── Main loop ─────────────────────────────────────────────────────────
    def _main_loop(self) -> None:
        log.info("Main loop started — Q to quit, R for AI, S for status")
        while True:
            frame = self._camera.read(timeout_s=2.0)
            if frame is None:
                self._handle_camera_lost()
                break

            self._fps.tick()
            now = time.time()
            self._bus.publish(EventType.FRAME, frame)

            self._stage_pit(frame, now)
            self._stage_yolo(frame, now)
            self._stage_ai(frame, now)
            self._stage_ocr(frame, now)
            self._stage_battery(now)

            if self._s.show_gui:
                self._render_gui(frame)
                if not self._handle_keys(frame, now):
                    break

    def _handle_camera_lost(self) -> None:
        log.error("Camera read timed out — declaring disconnect")
        self._bus.publish(
            EventType.SPEECH,
            SpeechEvent(text=Messages.CAMERA_LOST,
                        priority=Priority.CRITICAL, force=True),
        )
        self._bus.publish(EventType.HAPTIC, HapticEvent(priority=Priority.CRITICAL))

    # ── Pipeline stages ───────────────────────────────────────────────────
    def _stage_pit(self, frame: Frame, now: float) -> None:
        if now - self._timers["pit"] < self._s.timing.pit_s:
            return
        result = self._pit.detect(frame.image)
        self._pit_msg = result[0] if result else None
        if result:
            self._bus.publish(EventType.PIT_WARNING, result)
        self._timers["pit"] = now
        self._beats["pit"].append(now)

    def _stage_yolo(self, frame: Frame, now: float) -> None:
        if now - self._timers["yolo"] < self._s.timing.yolo_summary_s:
            return
        try:
            dets = self._yolo.detect(frame.image)
        except Exception:
            log.exception("YOLO detect failed")
            return
        dets = self._tracker.update(dets, frame.frame_id)
        self._panner.update_frame_width(frame.image.shape[1])
        self._last_dets = dets

        # Approach detection (use raw label_eng).
        for d in dets:
            if d.label_eng in EMERGENCY_LABELS and self._approach.update(
                d.label_eng, d.area_pct
            ):
                self._bus.publish(
                    EventType.APPROACH_ALERT, (d.label_eng, Priority.CRITICAL)
                )

        # Stateful narration → summary.
        narr_dets = self._narrator.filter(dets, frame.image.shape[1])
        descriptions = self._describe_detections(narr_dets, frame.image)
        descriptions.extend(SpatialAnalyzer.build_scene_graph(dets))
        summary = self._summarize(descriptions)
        if summary and summary != self._last_summary:
            self._bus.publish(
                EventType.SPEECH,
                SpeechEvent(text=summary, priority=Priority.NORMAL),
            )
            self._last_summary = summary
        elif not dets:
            self._bus.publish(
                EventType.SPEECH,
                SpeechEvent(text=Messages.OPEN_PATH, priority=Priority.NORMAL),
            )
        self._bus.publish(EventType.DETECTIONS, dets)
        self._timers["yolo"] = now
        self._beats["yolo"].append(now)

    def _stage_ai(self, frame: Frame, now: float) -> None:
        if now - self._timers["ai"] < self._s.timing.gemini_s:
            return
        if not self._router.is_active():
            return
        # Run AI in a worker so the loop never blocks.
        from threading import Thread
        Thread(target=self._ai_worker, args=(frame.image.copy(),),
               daemon=True, name="AiAnalysis").start()
        self._timers["ai"] = now

    def _ai_worker(self, image: np.ndarray) -> None:
        result = self._router.describe(image)
        self._beats["ai"].append(time.time())
        if result.success and result.text:
            self._bus.publish(EventType.AI_DESCRIPTION, result.text)

    def _stage_ocr(self, frame: Frame, now: float) -> None:
        if now - self._timers["ocr"] < self._s.timing.ocr_s:
            return
        text = self._ocr.read(frame.image)
        if text:
            self._bus.publish(EventType.OCR_TEXT, text)
        self._timers["ocr"] = now
        self._beats["ocr"].append(now)

    def _stage_battery(self, now: float) -> None:
        if now - self._timers["battery"] < self._s.timing.battery_s:
            return
        self._battery_pct = self._battery.check()
        self._timers["battery"] = now
        self._beats["battery"].append(now)

    # ── Description helpers ──────────────────────────────────────────────
    def _describe_detections(
        self, dets: list[Detection], image: np.ndarray
    ) -> list[str]:
        from .vision.spatial import SpatialAnalyzer as SA  # local: cycle-safe

        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        out: list[str] = []
        for det in dets:
            position = SA.position(det.bbox.cx, w)
            distance = SA.distance(det.area_pct)

            # Critical proximity.
            if det.area_pct > self._s.thresholds.critical_area:
                self._bus.publish(
                    EventType.SPEECH,
                    SpeechEvent(
                        text=Messages.CRITICAL_PROXIMITY.format(label=det.label_az),
                        priority=Priority.CRITICAL, force=True,
                        pan=self._panner.for_position(det.bbox.cx, det.area_pct).pan,
                    ),
                )
                self._bus.publish(EventType.HAPTIC,
                                  HapticEvent(priority=Priority.CRITICAL))
                out.append(f"{position} {det.label_az} çox yaxın (< 1 m).")
                continue

            if det.label_eng == "traffic light":
                roi = hsv[max(0, det.bbox.y1):min(h, det.bbox.y2),
                           max(0, det.bbox.x1):min(w, det.bbox.x2)]
                out.append(f"Svetofor: {ColorAnalyzer.traffic_light(roi)}")
            elif det.label_eng == "person":
                bw, bh = det.bbox.width, det.bbox.height
                if bw > 0 and bh > 0 and (bw / bh) > 2.2 and det.area_pct > 0.05:
                    out.append(f"{position} yıxılmış insan ola bilər")
                    self._bus.publish(
                        EventType.SPEECH,
                        SpeechEvent(text=Messages.FALLEN_PERSON.format(position=position),
                                    priority=Priority.CRITICAL, force=True),
                    )
                else:
                    out.append(f"{position} insan ({distance})")
            else:
                cy, cx = min(det.bbox.cy, h - 1), min(det.bbox.cx, w - 1)
                col = ColorAnalyzer.from_hsv(
                    int(hsv[cy, cx][0]), int(hsv[cy, cx][1]), int(hsv[cy, cx][2])
                )
                out.append(f"{position} {col} {det.label_az} ({distance})")
        return out

    @staticmethod
    def _summarize(descriptions: list[str]) -> str:
        if not descriptions:
            return ""
        counts = Counter(descriptions)
        parts = [f"{n} ədəd {t}" if n > 1 else t
                 for t, n in counts.most_common(4)]
        return ". ".join(parts) + "."

    # ── GUI ───────────────────────────────────────────────────────────────
    def _render_gui(self, frame: Frame) -> None:
        self._overlay.render(
            frame.image, detections=self._last_dets,
            pit_message=self._pit_msg, summary=self._last_summary,
            fps=self._fps.fps, ai_active=self._router.is_active(),
            battery_pct=self._battery_pct, mode=self._router.mode.value,
        )
        cv2.imshow("VisionVoiceAsist v5", frame.image)

    def _handle_keys(self, frame: Frame, now: float) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord("r"):
            self._bus.publish(
                EventType.SPEECH,
                SpeechEvent(text="Mühit dərindən analiz edilir.",
                            priority=Priority.HIGH, force=True),
            )
            self._stage_ai(frame, now=0.0)  # force immediate AI
            self._timers["ai"] = now
        elif key == ord("s"):
            self._status_report()
        elif key == ord("h"):
            self._health.force_check()
        return True

    def _status_report(self) -> None:
        bat = self._battery_pct
        bat_msg = (Messages.GREETING_BATTERY.format(pct=bat) if bat is not None
                   else "Qidalanma bloku.")
        st = self._speech.stats()
        msg = (
            f"Sistem normal işləyir. {bat_msg} "
            f"FPS: {self._fps.fps:.1f}. "
            f"Cəmi {st.get('total', 0)} mesaj söylənildi. "
            f"Rejim: {self._router.mode.value}."
        )
        self._bus.publish(EventType.SPEECH,
                          SpeechEvent(text=msg, priority=Priority.HIGH, force=True))

    # ── Dashboard ─────────────────────────────────────────────────────────
    def _maybe_start_dashboard(self) -> None:
        if not self._s.dashboard.enabled:
            return
        try:
            from .dashboard.app import DashboardApp  # noqa: PLC0415
        except ImportError:
            log.warning("Dashboard deps not installed; pip install '.[dashboard]'")
            return
        self._dashboard = DashboardApp(self._s.dashboard, self._bus)
        self._dashboard.start()  # type: ignore[attr-defined]
