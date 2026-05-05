"""Local on-device VLM (Moondream2 / SmolVLM) — offline fallback.

Implementation note:
    Loading a local 4-bit-quantised VLM requires `transformers` + `torch`
    plus model weights pulled at runtime. To keep the package shipping
    cleanly (and CI tests fast) the heavy import is lazy and gated by
    availability checks.

    The class implements the full :py:class:`VisionLanguageModel` protocol
    and falls back to a deterministic descriptive stub when weights are
    not yet downloaded — this preserves "always-on" behaviour for end users
    while a model engineer drops in real weights.
"""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Optional

import numpy as np

from ..settings import AiSettings
from .base import VlmResult

log = logging.getLogger(__name__)

_MOONDREAM_PROMPT = (
    "Describe this scene briefly for a visually-impaired user. "
    "Most important information first. Maximum 2 sentences."
)


class LocalVlm:
    """Lazy-loaded local VLM. Falls back to descriptive stub if weights missing."""

    def __init__(self, settings: AiSettings) -> None:
        self._s = settings
        self._model_name = settings.local_vlm
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._processor: object | None = None
        self._load_lock = Lock()
        self._load_attempted = False
        self._load_failed = False

    @property
    def name(self) -> str:
        return f"local-{self._model_name}"

    def is_available(self) -> bool:
        # Always "available" — we'll either run the model or fall back to stub.
        return True

    def _try_load(self) -> None:
        with self._load_lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                import torch  # noqa: PLC0415, F401
                from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

                model_id = {
                    "moondream2": "vikhyatk/moondream2",
                    "smolvlm-256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
                }.get(self._model_name, self._model_name)
                log.info("Loading local VLM: %s", model_id)
                self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True
                )
                log.info("Local VLM ready.")
            except Exception:  # noqa: BLE001
                self._load_failed = True
                log.warning(
                    "Local VLM weights unavailable — using descriptive stub. "
                    "To enable, install: pip install 'visionvoiceasist[local-vlm]' "
                    "and ensure model weights are cached."
                )

    def describe(
        self, image_bgr: np.ndarray, prompt: str = _MOONDREAM_PROMPT,
        *, max_tokens: int = 100
    ) -> VlmResult:
        self._try_load()
        t0 = time.time()
        if self._load_failed or self._model is None:
            return self._stub_describe(image_bgr, t0)
        return self._real_describe(image_bgr, prompt, max_tokens, t0)

    def query(
        self, image_bgr: np.ndarray, question: str, *, max_tokens: int = 100
    ) -> VlmResult:
        return self.describe(image_bgr, question, max_tokens=max_tokens)

    # ── Real model path ───────────────────────────────────────────────────
    def _real_describe(
        self, image_bgr: np.ndarray, prompt: str, max_tokens: int, t0: float
    ) -> VlmResult:
        try:
            import cv2  # noqa: PLC0415
            from PIL import Image as PILImage  # noqa: PLC0415

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            assert self._model is not None  # noqa: S101
            assert self._tokenizer is not None  # noqa: S101

            # Moondream2-style API
            if hasattr(self._model, "answer_question"):
                enc_image = self._model.encode_image(pil_img)  # type: ignore[attr-defined]
                text = self._model.answer_question(  # type: ignore[attr-defined]
                    enc_image, prompt, self._tokenizer
                )
            else:  # pragma: no cover — alternative VLM
                inputs = self._tokenizer(prompt, return_tensors="pt")  # type: ignore[operator]
                out = self._model.generate(**inputs, max_new_tokens=max_tokens)  # type: ignore[attr-defined]
                text = self._tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore[attr-defined]

            return VlmResult(
                text=str(text).strip(),
                source=self.name,
                latency_s=time.time() - t0,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Local VLM inference failed; falling back to stub")
            return self._stub_describe(image_bgr, t0, error=str(exc))

    # ── Stub path: descriptive fallback ───────────────────────────────────
    @staticmethod
    def _stub_describe(
        image_bgr: np.ndarray, t0: float, *, error: Optional[str] = None
    ) -> VlmResult:
        """Deterministic, fast description from frame statistics.

        Not a substitute for a real VLM, but ensures the system **always**
        answers something in offline mode rather than failing silently.
        """
        h, w = image_bgr.shape[:2]
        mean = float(image_bgr.mean())
        if mean < 60:
            tone = "qaranlıq"
        elif mean > 180:
            tone = "işıqlı"
        else:
            tone = "orta işıqlı"
        text = (
            f"Yerli süni intellekt aktivdir. Mühit {tone} görünür. "
            f"Daha dəqiq təsvir üçün YOLO və pit detektoruna güvənin."
        )
        return VlmResult(
            text=text,
            source="local-stub",
            latency_s=time.time() - t0,
            success=error is None,
            error=error,
        )
