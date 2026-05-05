"""Object tracking and stateful narration.

Naive IoU + centroid tracker — gives each detection a stable ID across
frames. The :py:class:`StatefulNarrator` uses these IDs to avoid repeating
``"car, car, car"`` and instead say things like
``"Maşın hələ də solunuzda"`` (the car is still on your left).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..types import Detection


@dataclass
class _Track:
    track_id: int
    label_eng: str
    label_az: str
    last_bbox: tuple[int, int, int, int]
    last_seen_frame: int
    last_area_pct: float
    last_position: str = ""
    first_seen_at: float = field(default_factory=time.time)
    last_announced_at: float = 0.0


def _iou(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


class ObjectTracker:
    """Greedy IoU tracker with class-conditional matching.

    Two detections only match if they share the same class label *and* their
    boxes have IoU > *iou_threshold*. After *max_age* missed frames a track
    is dropped.
    """

    def __init__(self, iou_threshold: float = 0.30, max_age: int = 15) -> None:
        self._iou_t = iou_threshold
        self._max_age = max_age
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(self, dets: list[Detection], frame_id: int) -> list[Detection]:
        """Return *dets* enriched with stable ``track_id`` values."""
        out: list[Detection] = []
        used_track_ids: set[int] = set()

        # Match each detection to the best matching active track of same class.
        for d in dets:
            best_tid: int | None = None
            best_iou = self._iou_t
            for tid, t in self._tracks.items():
                if tid in used_track_ids or t.label_eng != d.label_eng:
                    continue
                bbox = (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)
                score = _iou(t.last_bbox, bbox)
                if score > best_iou:
                    best_iou = score
                    best_tid = tid
            if best_tid is None:
                best_tid = self._next_id
                self._next_id += 1
                self._tracks[best_tid] = _Track(
                    track_id=best_tid,
                    label_eng=d.label_eng,
                    label_az=d.label_az,
                    last_bbox=(d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2),
                    last_seen_frame=frame_id,
                    last_area_pct=d.area_pct,
                )
            else:
                t = self._tracks[best_tid]
                t.last_bbox = (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)
                t.last_seen_frame = frame_id
                t.last_area_pct = d.area_pct
            used_track_ids.add(best_tid)

            # Replace the frozen Detection with one that has a track_id.
            out.append(
                Detection(
                    label_eng=d.label_eng,
                    label_az=d.label_az,
                    bbox=d.bbox,
                    conf=d.conf,
                    area_pct=d.area_pct,
                    frame_id=d.frame_id,
                    track_id=best_tid,
                    timestamp=d.timestamp,
                )
            )

        # Drop stale tracks.
        stale = [
            tid for tid, t in self._tracks.items()
            if frame_id - t.last_seen_frame > self._max_age
        ]
        for tid in stale:
            del self._tracks[tid]
        return out

    def get_track(self, track_id: int) -> _Track | None:
        return self._tracks.get(track_id)

    @property
    def active_count(self) -> int:
        return len(self._tracks)


class StatefulNarrator:
    """Generates non-repetitive descriptions from tracked detections.

    Strategy: announce a track at most once every *cooldown_s* seconds, and
    only if its position bucket changed (left/centre/right) or its distance
    bucket changed.
    """

    def __init__(self, cooldown_s: float = 6.0) -> None:
        self._cooldown = cooldown_s
        self._announced: dict[int, tuple[str, str, float]] = {}
        # tid → (last_position, last_distance, last_announce_ts)

    def filter(
        self, dets: list[Detection], frame_w: int
    ) -> list[Detection]:
        """Return only the detections worth speaking right now."""
        from ..i18n import distance_label, position_label  # local: cycle-safe

        now = time.time()
        keep: list[Detection] = []
        for d in dets:
            if d.track_id is None:
                keep.append(d)
                continue
            pos = position_label(d.bbox.cx, frame_w)
            dist = distance_label(d.area_pct)
            prev = self._announced.get(d.track_id)
            should_speak = (
                prev is None
                or (now - prev[2] > self._cooldown and (pos != prev[0] or dist != prev[1]))
            )
            if should_speak:
                self._announced[d.track_id] = (pos, dist, now)
                keep.append(d)
        # Garbage-collect old track entries so dict can't grow unbounded.
        cutoff = now - 5 * self._cooldown
        stale = [tid for tid, (_, _, ts) in self._announced.items() if ts < cutoff]
        for tid in stale:
            del self._announced[tid]
        return keep

    def reset(self) -> None:
        self._announced.clear()
