"""Tests for PrivacyFilter PII redaction."""

from __future__ import annotations

import numpy as np
import pytest

from visionvoiceasist.vision.privacy import PrivacyFilter, RedactionStats


class TestRedactionStats:
    def test_total_sums_both_fields(self) -> None:
        s = RedactionStats(faces_blurred=2, plates_blurred=1)
        assert s.total == 3

    def test_zero_total(self) -> None:
        assert RedactionStats(0, 0).total == 0

    def test_frozen(self) -> None:
        s = RedactionStats(1, 2)
        with pytest.raises(Exception):
            s.faces_blurred = 0  # type: ignore[misc]


class TestPrivacyFilter:
    def test_empty_array_returns_original_unchanged(self) -> None:
        pf = PrivacyFilter()
        empty = np.array([])
        out, stats = pf.redact(empty)
        assert out is empty  # zero-copy for empty input
        assert stats.total == 0

    def test_returns_copy_not_original(self) -> None:
        pf = PrivacyFilter()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out, _ = pf.redact(frame)
        assert out is not frame

    def test_output_preserves_shape(self) -> None:
        pf = PrivacyFilter()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out, _ = pf.redact(frame)
        assert out.shape == frame.shape

    def test_output_preserves_dtype(self) -> None:
        pf = PrivacyFilter()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out, _ = pf.redact(frame)
        assert out.dtype == frame.dtype

    def test_blank_frame_zero_detections(self) -> None:
        pf = PrivacyFilter()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, stats = pf.redact(frame)
        assert stats.faces_blurred == 0
        assert stats.plates_blurred == 0

    def test_white_frame_no_crash(self) -> None:
        pf = PrivacyFilter()
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        out, stats = pf.redact(frame)
        assert out.shape == (480, 640, 3)

    def test_odd_ksize_preserved(self) -> None:
        pf = PrivacyFilter(blur_ksize=31)
        assert pf._k == 31

    def test_even_ksize_rounded_up_to_odd(self) -> None:
        pf = PrivacyFilter(blur_ksize=30)
        assert pf._k == 31

    def test_even_ksize_1_rounded_to_1(self) -> None:
        pf = PrivacyFilter(blur_ksize=1)
        assert pf._k == 1

    def test_small_frame_no_crash(self) -> None:
        pf = PrivacyFilter()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        out, _ = pf.redact(frame)
        assert out.shape == (10, 10, 3)

    def test_non_square_frame(self) -> None:
        pf = PrivacyFilter()
        frame = np.zeros((240, 1280, 3), dtype=np.uint8)
        out, _ = pf.redact(frame)
        assert out.shape == (240, 1280, 3)

    def test_original_unmodified_after_redact(self) -> None:
        pf = PrivacyFilter()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        original = frame.copy()
        pf.redact(frame)
        np.testing.assert_array_equal(frame, original)
