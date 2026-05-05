"""Tests for HRTF-based spatial audio calculator."""

from __future__ import annotations

import math

import pytest

from visionvoiceasist.audio.hrtf import HrtfCalculator, HrtfCue


class TestHrtfCue:
    def test_frozen(self) -> None:
        cue = HrtfCue(0.5, 0.5, 0.0, 0.0, 0.0)
        with pytest.raises(Exception):
            cue.left_gain = 1.0  # type: ignore[misc]


class TestHrtfCalculator:
    @pytest.fixture
    def calc(self) -> HrtfCalculator:
        return HrtfCalculator()

    # ── Center position ───────────────────────────────────────────────────────

    def test_center_pan_equal_gains(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=0.0)
        assert cue.left_gain == pytest.approx(cue.right_gain, abs=1e-4)

    def test_center_pan_zero_itd(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=0.0)
        assert cue.itd_ms == pytest.approx(0.0, abs=1e-6)

    def test_center_pan_zero_ild(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=0.0)
        assert cue.ild_db == pytest.approx(0.0, abs=1e-4)

    def test_center_azimuth_zero_degrees(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=0.0)
        assert cue.azimuth_deg == pytest.approx(0.0, abs=0.01)

    # ── Left/right dominance ──────────────────────────────────────────────────

    def test_right_pan_right_gain_dominates(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=1.0)
        assert cue.right_gain > cue.left_gain

    def test_left_pan_left_gain_dominates(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=-1.0)
        assert cue.left_gain > cue.right_gain

    # ── ITD sign and magnitude ────────────────────────────────────────────────

    def test_right_source_positive_itd(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=1.0)
        assert cue.itd_ms > 0

    def test_left_source_negative_itd(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=-1.0)
        assert cue.itd_ms < 0

    def test_max_itd_within_physical_range(self, calc: HrtfCalculator) -> None:
        # Blauert (1997): physiological maximum ITD ≤ ~0.7 ms
        cue = calc.compute(pan=1.0)
        assert abs(cue.itd_ms) <= 0.75

    # ── ILD sign ─────────────────────────────────────────────────────────────

    def test_ild_positive_for_right_source(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=0.8)
        assert cue.ild_db > 0

    def test_ild_negative_for_left_source(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=-0.8)
        assert cue.ild_db < 0

    # ── Gain range ────────────────────────────────────────────────────────────

    def test_gains_clamped_to_unit_range(self, calc: HrtfCalculator) -> None:
        for pan in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            cue = calc.compute(pan=pan)
            assert 0.0 <= cue.left_gain <= 1.0, f"left_gain out of range at pan={pan}"
            assert 0.0 <= cue.right_gain <= 1.0, f"right_gain out of range at pan={pan}"

    # ── Pan clamping ──────────────────────────────────────────────────────────

    def test_pan_over_one_clamped(self, calc: HrtfCalculator) -> None:
        cue_over = calc.compute(pan=2.0)
        cue_max = calc.compute(pan=1.0)
        assert cue_over.left_gain == pytest.approx(cue_max.left_gain)
        assert cue_over.right_gain == pytest.approx(cue_max.right_gain)

    def test_pan_under_minus_one_clamped(self, calc: HrtfCalculator) -> None:
        cue_over = calc.compute(pan=-5.0)
        cue_max = calc.compute(pan=-1.0)
        assert cue_over.left_gain == pytest.approx(cue_max.left_gain)

    # ── Distance gain ─────────────────────────────────────────────────────────

    def test_higher_distance_gain_louder(self, calc: HrtfCalculator) -> None:
        cue_close = calc.compute(pan=0.5, distance_gain=1.0)
        cue_far = calc.compute(pan=0.5, distance_gain=0.5)
        assert cue_close.right_gain > cue_far.right_gain

    def test_distance_gain_zero_silent(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=0.0, distance_gain=0.0)
        assert cue.left_gain == pytest.approx(0.0, abs=1e-6)
        assert cue.right_gain == pytest.approx(0.0, abs=1e-6)

    # ── Azimuth ───────────────────────────────────────────────────────────────

    def test_full_right_azimuth_90(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=1.0)
        assert cue.azimuth_deg == pytest.approx(90.0, abs=0.01)

    def test_full_left_azimuth_minus_90(self, calc: HrtfCalculator) -> None:
        cue = calc.compute(pan=-1.0)
        assert cue.azimuth_deg == pytest.approx(-90.0, abs=0.01)

    # ── Symmetry ─────────────────────────────────────────────────────────────

    def test_left_right_gain_symmetry(self, calc: HrtfCalculator) -> None:
        left_cue = calc.compute(pan=-0.6)
        right_cue = calc.compute(pan=0.6)
        assert left_cue.left_gain == pytest.approx(right_cue.right_gain, abs=1e-4)
        assert left_cue.right_gain == pytest.approx(right_cue.left_gain, abs=1e-4)

    def test_itd_antisymmetric(self, calc: HrtfCalculator) -> None:
        left_cue = calc.compute(pan=-0.5)
        right_cue = calc.compute(pan=0.5)
        assert left_cue.itd_ms == pytest.approx(-right_cue.itd_ms, abs=1e-4)

    # ── Woodworth formula verification ────────────────────────────────────────

    def test_woodworth_itd_formula_at_90_degrees(self) -> None:
        """Verify Δt = (a/c) · (sin(π/2) + π/2) at θ = π/2."""
        a, c = 0.0875, 343.0
        calc = HrtfCalculator(head_radius=a, speed_of_sound=c)
        cue = calc.compute(pan=1.0)
        expected_ms = (a / c) * (math.sin(math.pi / 2) + math.pi / 2) * 1000.0
        assert cue.itd_ms == pytest.approx(expected_ms, abs=0.001)

    # ── Custom parameters ─────────────────────────────────────────────────────

    def test_larger_head_radius_larger_itd(self) -> None:
        calc_small = HrtfCalculator(head_radius=0.08)
        calc_large = HrtfCalculator(head_radius=0.10)
        cue_small = calc_small.compute(pan=0.5)
        cue_large = calc_large.compute(pan=0.5)
        assert cue_large.itd_ms > cue_small.itd_ms

    def test_larger_ild_max_wider_separation(self) -> None:
        # pan=0.1: base right_gain ≈ 0.76 — ILD correction fits below the 1.0 ceiling
        calc_narrow = HrtfCalculator(max_ild_db=1.0)
        calc_wide = HrtfCalculator(max_ild_db=9.0)
        cue_narrow = calc_narrow.compute(pan=0.1)
        cue_wide = calc_wide.compute(pan=0.1)
        assert cue_wide.right_gain > cue_narrow.right_gain
