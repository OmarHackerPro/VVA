"""Retry decorator with exponential backoff.

Wraps the `tenacity` library if available, falls back to a hand-rolled
implementation so the package keeps working in minimal environments.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


def with_retry(
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    jitter: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: retry *attempts* times with exponential backoff + jitter.

    Args:
        attempts: Total tries including the first call.
        base_delay: Initial wait in seconds.
        max_delay: Cap on wait between attempts.
        exceptions: Exception classes that trigger retry.
        jitter: If True, randomise wait by ±25 %.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> T:
            last_exc: BaseException | None = None
            for i in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if i == attempts - 1:
                        break
                    delay = min(base_delay * (2 ** i), max_delay)
                    if jitter:
                        delay *= 0.75 + random.random() * 0.5  # noqa: S311 — non-crypto
                    log.warning(
                        "%s attempt %d/%d failed: %s — retrying in %.2fs",
                        fn.__name__, i + 1, attempts, exc, delay,
                    )
                    time.sleep(delay)
            assert last_exc is not None  # noqa: S101
            raise last_exc

        return wrapper

    return decorator
