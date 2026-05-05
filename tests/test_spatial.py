"""Tests for SpatialAnalyzer and ApproachTracker."""

from __future__ import annotations

import time

import pytest

from tests.conftest import make_det
from visionvoiceasist.settings import ThresholdSettings
from visionvoiceasist.types import BBox, Detection
from visionvoiceasist.vision.spatial import ApproachTracker, SpatialAnalyzer, is_emergency


class TestSpatialAnalyzerPosition:
    def test_left(self) -> None:
        assert SpatialAnalyzer.position(50, 640) == "Solunuzda"

    def test_right(self) -> None:
        assert SpatialAnalyzer.position(500, 640) == "Sağınızda"

    def test_center(self) -> None:
        assert SpatialAnalyzer.position(320, 640) == "Qarşınızda"


class TestSpatialAnalyzerDistance:
    def test_very_close(self) -> None:
        assert "< 1 m" in SpatialAnalyzer.distance(0.6)

    def test_far(self) -> None:
        assert "4+" in SpatialAnalyzer.distance(0.02)


class TestSpatialAnalyzerBuildSceneGraph:
    def test_empty_returns_empty(self) -> None:
        assert SpatialAnalyzer.build_scene_graph([]) == []

    def test_crowd_three_people(self) -> None:
        people = [make_det("person", x1=i * 50, x2=i * 50 + 40) for i in range(3)]
        graph = SpatialAnalyzer.build_scene_graph(people)
        assert any("nəfər" in r for r in graph)

    def test_two_people(self) -> None:
        people = [make_det("person"), make_det("person", x1=300)]
        graph = SpatialAnalyzer.build_scene_graph(people)
        assert any("2 nəfər" in r for r in graph)

    def test_busy_scene(self) -> None:
        dets = [make_det("car", x1=i * 50, x2=i * 50 + 40) for i in range(6)]
        graph = SpatialAnalyzer.build_scene_graph(dets)
        assert any("əşya" in r for r in graph)

    def test_object_on_table(self) -> None:
        # Table occupies full width; cup on top of it
        table = make_det("dining table", x1=0, y1=200, x2=640, y2=480)
        cup = make_det("cup", x1=200, y1=210, x2=250, y2=260)
        graph = SpatialAnalyzer.build_scene_graph([table, cup])
        assert any("fincan" in r or "üzərində" in r for r in graph)

    def test_no_false_surface_objects(self) -> None:
        # Object not within surface bbox → no surface relation
        table = make_det("dining table", x1=0, y1=300, x2=200, y2=480)
        car = make_det("car", x1=400, y1=100, x2=550, y2=200)
        graph = SpatialAnalyzer.build_scene_graph([table, car])
        assert not any("üzərində" in r for r in graph)


class TestApproachTracker:
    def _make_tracker(self) -> ApproachTracker:
        return ApproachTracker(ThresholdSettings(approach_slope=0.02))

    def test_no_alert_with_fewer_than_4_samples(self) -> None:
        t = self._make_tracker()
        for i in range(3):
            result = t.update("car", 0.1 + i * 0.05)
        assert result is False

    def test_alert_fires_on_steep_slope(self) -> None:
        t = self._make_tracker()
        alerted = False
        # Feed rapidly increasing areas to guarantee slope > threshold
        for i in range(8):
            if t.update("car", 0.05 + i * 0.08):
                alerted = True
        assert alerted

    def test_alert_fires_once_per_approach(self) -> None:
        t = self._make_tracker()
        alerts = 0
        for i in range(12):
            if t.update("car", 0.05 + i * 0.08):
                alerts += 1
        assert alerts == 1

    def test_alert_resets_on_receding(self) -> None:
        t = self._make_tracker()
        # Build up and trigger alert
        for i in range(8):
            t.update("car", 0.05 + i * 0.08)
        # Now recede (slope < threshold)
        for i in range(8):
            t.update("car", 0.70 - i * 0.08)
        # Should be able to alert again
        alerted = False
        for i in range(8):
            if t.update("car", 0.05 + i * 0.08):
                alerted = True
        assert alerted

    def test_tracked_labels(self) -> None:
        t = self._make_tracker()
        t.update("car", 0.1)
        t.update("person", 0.2)
        assert "car" in t.tracked_labels
        assert "person" in t.tracked_labels

    def test_prune_removes_stale(self) -> None:
        t = ApproachTracker(ThresholdSettings(), ttl_s=0.01)
        t.update("car", 0.1)
        time.sleep(0.05)
        t.update("bus", 0.1)  # triggers prune
        assert "car" not in t.tracked_labels

    def test_reset_clears_state(self) -> None:
        t = self._make_tracker()
        t.update("car", 0.5)
        t.reset()
        assert t.tracked_labels == set()

    def test_flat_trajectory_no_alert(self) -> None:
        t = self._make_tracker()
        for _ in range(8):
            assert t.update("car", 0.15) is False


class TestIsEmergency:
    def test_car_is_emergency(self) -> None:
        assert is_emergency("car") is True

    def test_person_is_emergency(self) -> None:
        assert is_emergency("person") is True

    def test_cup_not_emergency(self) -> None:
        assert is_emergency("cup") is False
