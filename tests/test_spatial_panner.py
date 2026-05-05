"""Tests for SpatialPanner stereo positioning."""

from __future__ import annotations

import pytest

from visionvoiceasist.audio.spatial import SpatialCue, SpatialPanner


class TestSpatialPanner:
    def test_center_position_pan_zero(self) -> None:
        p = SpatialPanner(frame_w=640)
        cue = p.for_position(cx=320)
        assert cue.pan == pytest.approx(0.0, abs=0.01)

    def test_left_edge_pan_negative_one(self) -> None:
        p = SpatialPanner(frame_w=640)
        cue = p.for_position(cx=0)
        assert cue.pan == pytest.approx(-1.0, abs=0.01)

    def test_right_edge_pan_positive_one(self) -> None:
        p = SpatialPanner(frame_w=640)
        cue = p.for_position(cx=640)
        assert cue.pan == pytest.approx(1.0, abs=0.01)

    def test_pan_clamped_to_minus_one(self) -> None:
        p = SpatialPanner(frame_w=640)
        cue = p.for_position(cx=-999)
        assert cue.pan == pytest.approx(-1.0)

    def test_pan_clamped_to_plus_one(self) -> None:
        p = SpatialPanner(frame_w=640)
        cue = p.for_position(cx=9999)
        assert cue.pan == pytest.approx(1.0)

    def test_zero_frame_width_returns_center(self) -> None:
        p = SpatialPanner(frame_w=0)
        cue = p.for_position(cx=320)
        assert cue.pan == pytest.approx(0.0)

    def test_large_area_pct_increases_gain(self) -> None:
        p = SpatialPanner(frame_w=640)
        small = p.for_position(cx=320, area_pct=0.0)
        large = p.for_position(cx=320, area_pct=0.5)
        assert large.distance_gain >= small.distance_gain

    def test_gain_clamped_to_unit_range(self) -> None:
        p = SpatialPanner(frame_w=640)
        cue = p.for_position(cx=320, area_pct=999.0)
        assert cue.distance_gain <= 1.0

    def test_gain_minimum_is_0_4(self) -> None:
        p = SpatialPanner(frame_w=640)
        cue = p.for_position(cx=320, area_pct=0.0)
        assert cue.distance_gain >= 0.4 - 1e-9

    def test_update_frame_width(self) -> None:
        p = SpatialPanner(frame_w=640)
        p.update_frame_width(1280)
        cue = p.for_position(cx=640)
        assert cue.pan == pytest.approx(0.0, abs=0.01)

    def test_update_frame_width_zero_ignored(self) -> None:
        p = SpatialPanner(frame_w=640)
        p.update_frame_width(0)
        assert p._w == 640  # type: ignore[attr-defined]

    def test_spatial_cue_is_frozen(self) -> None:
        cue = SpatialCue(pan=0.5)
        with pytest.raises(Exception):
            cue.pan = 0.0  # type: ignore[misc]

    def test_pan_symmetry(self) -> None:
        p = SpatialPanner(frame_w=640)
        left = p.for_position(cx=100)
        right = p.for_position(cx=540)
        assert left.pan == pytest.approx(-right.pan, abs=0.05)
