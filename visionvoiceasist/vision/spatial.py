"""Spatial reasoning: position, distance, scene graph, approach tracking."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Optional

import numpy as np

from ..i18n import EMERGENCY_LABELS, SURFACES, Messages, az_label, distance_label, position_label
from ..settings import ThresholdSettings
from ..types import Detection

log = logging.getLogger(__name__)


class SpatialAnalyzer:
    """Pure functions for scene-graph construction. No state."""

    @staticmethod
    def position(cx: int, frame_w: int) -> str:
        return position_label(cx, frame_w)

    @staticmethod
    def distance(area_pct: float) -> str:
        return distance_label(area_pct)

    @staticmethod
    def build_scene_graph(dets: list[Detection]) -> list[str]:
        """Return human-readable spatial relations from a detection list."""
        relations: list[str] = []
        surfaces = [d for d in dets if d.label_eng in SURFACES]
        non_surfaces = [d for d in dets if d.label_eng not in SURFACES]

        for surf in surfaces:
            on_surf = SpatialAnalyzer._objects_on_surface(surf, non_surfaces)
            if not on_surf:
                continue
            unique = list(dict.fromkeys(on_surf))
            cap = surf.label_az.capitalize()
            if len(unique) == 1:
                relations.append(
                    Messages.SCENE_ON_SURFACE_ONE.format(surface=cap, item=unique[0])
                )
            elif len(unique) <= 3:
                relations.append(
                    Messages.SCENE_ON_SURFACE_FEW.format(
                        surface=cap, items=", ".join(unique[:-1]), last=unique[-1]
                    )
                )
            else:
                relations.append(Messages.SCENE_ON_SURFACE_MANY.format(surface=cap, n=len(unique)))

        people = [d for d in dets if d.label_eng == "person"]
        if len(people) >= 3:
            relations.append(Messages.SCENE_CROWD.format(n=len(people)))
        elif len(people) == 2:
            relations.append(Messages.SCENE_TWO_PEOPLE)

        if len(dets) >= 6:
            relations.append(Messages.SCENE_BUSY.format(n=len(dets)))
        return relations

    @staticmethod
    def _objects_on_surface(
        surf: Detection, candidates: list[Detection]
    ) -> list[str]:
        """Return Az labels of objects whose bottom-edge sits on *surf*'s upper region."""
        sx1, sy1, sx2, sy2 = surf.bbox.x1, surf.bbox.y1, surf.bbox.x2, surf.bbox.y2
        upper_band_end = sy1 + (sy2 - sy1) * 0.42
        out: list[str] = []
        for obj in candidates:
            obj_cx = obj.bbox.cx
            obj_y2 = obj.bbox.y2
            if sx1 < obj_cx < sx2 and sy1 < obj_y2 < upper_band_end:
                out.append(obj.label_az)
        return out


class ApproachTracker:
    """Tracks bbox-area trajectories to predict approaching objects.

    Uses linear regression (np.polyfit) on the recent area history. Bounded
    memory: stale labels are pruned after *ttl_s* seconds.
    """

    def __init__(
        self,
        thresholds: ThresholdSettings,
        window: int = 8,
        ttl_s: float = 30.0,
    ) -> None:
        self._th = thresholds
        self._window = window
        self._ttl = ttl_s
        self._history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._last_seen: dict[str, float] = {}
        self._alerted: set[str] = set()

    def update(self, label: str, area_pct: float) -> bool:
        """Push a sample; return True if approach alert fires *this* update."""
        self._prune()
        self._history[label].append(area_pct)
        self._last_seen[label] = time.time()
        hist = list(self._history[label])
        if len(hist) < 4:
            return False
        xs = np.arange(len(hist), dtype=np.float64)
        slope = float(np.polyfit(xs, hist, 1)[0])
        approaching = slope > self._th.approach_slope
        if approaching and label not in self._alerted:
            self._alerted.add(label)
            return True
        if not approaching:
            self._alerted.discard(label)
        return False

    def _prune(self) -> None:
        now = time.time()
        stale = [k for k, t in self._last_seen.items() if now - t > self._ttl]
        for k in stale:
            self._history.pop(k, None)
            self._last_seen.pop(k, None)
            self._alerted.discard(k)

    def reset(self) -> None:
        self._history.clear()
        self._last_seen.clear()
        self._alerted.clear()

    @property
    def tracked_labels(self) -> set[str]:
        return set(self._history.keys())


def is_emergency(label_eng: str) -> bool:
    """Quick predicate for emergency-class labels."""
    return label_eng in EMERGENCY_LABELS


# Re-export for convenience
__all__ = ["ApproachTracker", "SpatialAnalyzer", "is_emergency"]
