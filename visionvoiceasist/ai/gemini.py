"""Google Gemini Vision backend with retry / circuit-breaker semantics."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from ..settings import AiSettings
from ..utils.retry import with_retry
from .base import VlmResult

log = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Sən görmə məhdudiyyətli şəxs üçün işləyən AI görmə köməkçisisən. "
    "Şəkildəki mühiti AZƏRBAYCAN DİLİNDƏ qısa, aydın, praktik təsvir et. "
    "Ən vacib məlumatı əvvəl söylə. "
    "Məsafə, rəng, mövqe barədə dəqiq məlumat ver. "
    "Ən çox 3 cümlə. Qısa olsun."
)


class GeminiVlm:
    """Google Gemini 1.5 Flash vision model."""

    name = "gemini-1.5-flash"

    def __init__(self, settings: AiSettings) -> None:
        self._s = settings
        self._model: object | None = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._init()

    def _init(self) -> None:
        if not self._s.gemini_key:
            log.warning("GEMINI_KEY not set; Gemini disabled.")
            return
        try:
            import google.generativeai as genai  # noqa: PLC0415

            genai.configure(api_key=self._s.gemini_key)
            self._model = genai.GenerativeModel("gemini-1.5-flash")
            log.info("Gemini VLM ready.")
        except Exception:  # noqa: BLE001
            log.exception("Failed to init Gemini")

    def is_available(self) -> bool:
        if self._model is None:
            return False
        if time.time() < self._circuit_open_until:
            return False
        return True

    def describe(
        self, image_bgr: np.ndarray, prompt: str = DEFAULT_PROMPT, *, max_tokens: int = 200
    ) -> VlmResult:
        return self._call(image_bgr, prompt, max_tokens)

    def query(
        self, image_bgr: np.ndarray, question: str, *, max_tokens: int = 200
    ) -> VlmResult:
        prompt = (
            f"{DEFAULT_PROMPT}\n\nİstifadəçinin sualı: {question}\n"
            "Sualı qısa və konkret cavablandır."
        )
        return self._call(image_bgr, prompt, max_tokens)

    @with_retry(attempts=3, base_delay=0.6, max_delay=4.0)
    def _generate(self, pil_img: object, prompt: str, max_tokens: int) -> str:
        assert self._model is not None  # noqa: S101
        response = self._model.generate_content(  # type: ignore[attr-defined]
            [prompt, pil_img],
            generation_config={"max_output_tokens": max_tokens},
        )
        return str(response.text).strip()

    def _call(
        self, image_bgr: np.ndarray, prompt: str, max_tokens: int
    ) -> VlmResult:
        if not self.is_available():
            return VlmResult(text="", source=self.name, latency_s=0.0,
                             success=False, error="unavailable")
        from PIL import Image as PILImage  # noqa: PLC0415

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        t0 = time.time()
        try:
            text = self._generate(pil_img, prompt, max_tokens)
        except Exception as exc:  # noqa: BLE001
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                # Open circuit for 60 seconds.
                self._circuit_open_until = time.time() + 60
                log.warning("Gemini circuit opened for 60s after %d failures",
                            self._consecutive_failures)
            return VlmResult(
                text="", source=self.name, latency_s=time.time() - t0,
                success=False, error=str(exc),
            )
        self._consecutive_failures = 0
        return VlmResult(text=text, source=self.name, latency_s=time.time() - t0)
