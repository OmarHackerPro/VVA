"""Tests for ObjectTracker and StatefulNarrator."""

from __future__ import annotations

import time

import pytest

from tests.conftest import make_det
from visionvoiceasist.vision.tracking import ObjectTracker, StatefulNarrator, _iou


class TestIou:
    def test_identical_boxes(self) -> None:
        assert _iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert _iou((0, 0, 50, 50), (100, 100, 200, 200)) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        score = _iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.0 < score < 1.0

    def test_contained_box(self) -> None:
        score = _iou((0, 0, 100, 100), (25, 25, 75, 75))
        assert score == pytest.approx(0.25)

    def test_inverted_box_zero_area(self) -> None:
        assert _iou((100, 100, 50, 50), (0, 0, 200, 200)) == pytest.approx(0.0)


class TestObjectTracker:
    def test_assigns_track_id(self) -> None:
        tracker = ObjectTracker()
        det = make_det("car")
        result = tracker.update([det], frame_id=1)
        assert result[0].track_id is not None

    def test_consistent_id_across_frames(self) -> None:
        tracker = ObjectTracker()
        d1 = make_det("car", x1=100, y1=100, x2=200, y2=200)
        [r1] = tracker.update([d1], frame_id=1)
        d2 = make_det("car", x1=105, y1=105, x2=205, y2=205)
        [r2] = tracker.update([d2], frame_id=2)
        assert r1.track_id == r2.track_id

    def test_different_class_gets_new_id(self) -> None:
        tracker = ObjectTracker()
        car = make_det("car", x1=100, y1=100, x2=200, y2=200)
        person = make_det("person", x1=100, y1=100, x2=200, y2=200)
        [r_car] = tracker.update([car], frame_id=1)
        [r_person] = tracker.update([person], frame_id=2)
        assert r_car.track_id != r_person.track_id

    def test_non_overlapping_detection_new_id(self) -> None:
        tracker = ObjectTracker()
        d1 = make_det("car", x1=0, y1=0, x2=100, y2=100)
        [r1] = tracker.update([d1], frame_id=1)
        d2 = make_det("car", x1=500, y1=400, x2=600, y2=480)
        [r2] = tracker.update([d2], frame_id=2)
        assert r1.track_id != r2.track_id

    def test_stale_tracks_dropped(self) -> None:
        tracker = ObjectTracker(max_age=3)
        d = make_det("car")
        tracker.update([d], frame_id=1)
        assert tracker.active_count == 1
        # Advance frames without that detection
        tracker.update([], frame_id=5)  # 5 - 1 = 4 > max_age=3
        assert tracker.active_count == 0

    def test_multiple_detections_unique_ids(self) -> None:
        tracker = ObjectTracker()
        dets = [
            make_det("car", x1=0, x2=50),
            make_det("car", x1=200, x2=250),
            make_det("person", x1=400, x2=450),
        ]
        results = tracker.update(dets, frame_id=1)
        ids = [r.track_id for r in results]
        assert len(set(ids)) == 3

    def test_empty_detections_returns_empty(self) -> None:
        tracker = ObjectTracker()
        assert tracker.update([], frame_id=1) == []

    def test_get_track(self) -> None:
        tracker = ObjectTracker()
        d = make_det("car")
        [r] = tracker.update([d], frame_id=1)
        track = tracker.get_track(r.track_id)  # type: ignore[arg-type]
        assert track is not None
        assert track.label_eng == "car"


class TestStatefulNarrator:
    def test_new_detection_always_passes(self) -> None:
        narrator = StatefulNarrator(cooldown_s=10.0)
        det = make_det("car", track_id=1)
        result = narrator.filter([det], frame_w=640)
        assert len(result) == 1

    def test_same_track_suppressed_within_cooldown(self) -> None:
        narrator = StatefulNarrator(cooldown_s=60.0)
        det = make_det("car", track_id=1, x1=100, y1=100, x2=200, y2=200)
        narrator.filter([det], frame_w=640)
        result = narrator.filter([det], frame_w=640)
        assert len(result) == 0

    def test_track_announced_after_cooldown(self) -> None:
        narrator = StatefulNarrator(cooldown_s=0.01)
        det = make_det("car", track_id=1)
        narrator.filter([det], frame_w=640)
        time.sleep(0.05)
        # Change position to trigger re-announcement
        moved = make_det("car", x1=500, x2=600, track_id=1)
        result = narrator.filter([moved], frame_w=640)
        assert len(result) == 1

    def test_no_track_id_always_passes(self) -> None:
        narrator = StatefulNarrator()
        det = make_det("car", track_id=None)
        # Call twice — no suppression without track_id
        narrator.filter([det], frame_w=640)
        result = narrator.filter([det], frame_w=640)
        assert len(result) == 1

    def test_reset_clears_cooldowns(self) -> None:
        narrator = StatefulNarrator(cooldown_s=60.0)
        det = make_det("car", track_id=1)
        narrator.filter([det], frame_w=640)
        narrator.reset()
        result = narrator.filter([det], frame_w=640)
        assert len(result) == 1

    def test_stale_entries_garbage_collected(self) -> None:
        narrator = StatefulNarrator(cooldown_s=0.01)
        det = make_det("car", track_id=42)
        narrator.filter([det], frame_w=640)
        time.sleep(0.1)
        # Trigger GC via filter with a different det
        narrator.filter([make_det("cup", track_id=99)], frame_w=640)
        assert 42 not in narrator._announced  # type: ignore[attr-defined]
