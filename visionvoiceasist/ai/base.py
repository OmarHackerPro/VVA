"""Strategy interface for vision-language models (VLMs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class VlmResult:
    """Result of a VLM inference."""

    text: str
    source: str           # "gemini" | "local-moondream2" | etc
    latency_s: float
    success: bool = True
    error: Optional[str] = None


@runtime_checkable
class VisionLanguageModel(Protocol):
    """Protocol implemented by every VLM backend."""

    name: str

    def is_available(self) -> bool:
        """Return True if this model can serve requests right now."""

    def describe(
        self, image_bgr: np.ndarray, prompt: str, *, max_tokens: int = 200
    ) -> VlmResult:
        """Synchronously describe *image_bgr* given *prompt*."""

    def query(
        self,
        image_bgr: np.ndarray,
        question: str,
        *,
        max_tokens: int = 200,
    ) -> VlmResult:
        """Answer a free-form *question* about the image."""
