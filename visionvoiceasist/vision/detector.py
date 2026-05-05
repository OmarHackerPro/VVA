"""YOLOv8 object detector with auto-device selection."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..i18n import az_label
from ..settings import YoloSettings
from ..types import BBox, Detection
from ..utils.device import detect_yolo_device

log = logging.getLogger(__name__)


class YoloDetector:
    """Wraps Ultralytics YOLO, returning typed Detection objects.

    Lazy model load: the heavy `from ultralytics import YOLO` is only run
    on the first call to :py:meth:`detect`, so importing this module is cheap.
    """

    def __init__(self, settings: YoloSettings) -> None:
        self._s = settings
        self._model: Optional[object] = None
        self._device: str = ""
        self._frame_id = 0

    def warm_up(self) -> None:
        """Load the model now (otherwise lazy on first detect call)."""
        if self._model is not None:
            return
        from ultralytics import YOLO  # noqa: PLC0415 — heavy import

        self._device = detect_yolo_device(self._s.device)
        log.info("Loading YOLO model %s on %s", self._s.model, self._device)
        self._model = YOLO(self._s.model)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR image."""
        self.warm_up()
        assert self._model is not None  # noqa: S101
        self._frame_id += 1
        results = self._model(  # type: ignore[operator]
            image,
            conf=self._s.conf,
            iou=self._s.iou,
            verbose=False,
            device=self._device,
        )
        if not results:
            return []
        h, w = image.shape[:2]
        out: list[Detection] = []
        boxes = results[0].boxes
        names = self._model.names  # type: ignore[attr-defined]
        for box in boxes:
            cls = int(box.cls[0])
            label_eng = names[cls]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            bbox = BBox(x1, y1, x2, y2)
            out.append(
                Detection(
                    label_eng=label_eng,
                    label_az=az_label(label_eng),
                    bbox=bbox,
                    conf=float(box.conf[0]),
                    area_pct=bbox.area_pct(w, h),
                    frame_id=self._frame_id,
                )
            )
        return out

    @property
    def device(self) -> str:
        return self._device
