"""Tests for the @with_retry exponential-backoff decorator."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from visionvoiceasist.utils.retry import with_retry


class TestWithRetry:
    def test_succeeds_first_try(self) -> None:
        call_count = 0

        @with_retry(attempts=3, base_delay=0.0)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        assert fn() == "ok"
        assert call_count == 1

    def test_retries_on_failure_then_succeeds(self) -> None:
        attempts_log: list[int] = []

        @with_retry(attempts=3, base_delay=0.001, jitter=False)
        def fn() -> str:
            attempts_log.append(1)
            if len(attempts_log) < 3:
                raise ValueError("not yet")
            return "done"

        assert fn() == "done"
        assert len(attempts_log) == 3

    def test_raises_after_all_attempts_exhausted(self) -> None:
        @with_retry(attempts=3, base_delay=0.001, jitter=False)
        def always_fails() -> None:
            raise RuntimeError("bang")

        with pytest.raises(RuntimeError, match="bang"):
            always_fails()

    def test_call_count_matches_attempts(self) -> None:
        count = 0

        @with_retry(attempts=4, base_delay=0.001, jitter=False)
        def fn() -> None:
            nonlocal count
            count += 1
            raise OSError("fail")

        with pytest.raises(OSError):
            fn()
        assert count == 4

    def test_only_retries_specified_exception(self) -> None:
        @with_retry(attempts=3, base_delay=0.001, exceptions=(ValueError,))
        def fn() -> None:
            raise TypeError("not retried")

        with pytest.raises(TypeError):
            fn()

    def test_non_matching_exception_propagates_immediately(self) -> None:
        call_count = 0

        @with_retry(attempts=5, base_delay=0.001, exceptions=(ValueError,))
        def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise KeyError("oops")

        with pytest.raises(KeyError):
            fn()
        assert call_count == 1  # no retries

    def test_preserves_return_value(self) -> None:
        @with_retry(attempts=2)
        def fn() -> dict:
            return {"key": 42}

        assert fn() == {"key": 42}

    def test_preserves_function_metadata(self) -> None:
        @with_retry()
        def my_function() -> None:
            """My docstring."""

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_no_sleep_on_last_attempt(self) -> None:
        sleep_calls: list[float] = []

        @with_retry(attempts=2, base_delay=1.0, jitter=False)
        def fn() -> None:
            raise ValueError("x")

        with patch("visionvoiceasist.utils.retry.time.sleep", side_effect=sleep_calls.append):
            with pytest.raises(ValueError):
                fn()

        # 2 attempts → 1 sleep before attempt 2, none after attempt 2
        assert len(sleep_calls) == 1

    def test_jitter_delay_within_range(self) -> None:
        """With jitter=True the delay should be in [0.75*base, 1.25*base]."""
        delays: list[float] = []

        @with_retry(attempts=2, base_delay=1.0, jitter=True)
        def fn() -> None:
            raise ValueError("x")

        with patch("visionvoiceasist.utils.retry.time.sleep", side_effect=delays.append):
            with pytest.raises(ValueError):
                fn()

        assert len(delays) == 1
        assert 0.74 <= delays[0] <= 1.26
