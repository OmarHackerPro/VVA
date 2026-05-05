"""Auto-detect best available device for ML inference.

Order of preference:
    1. NVIDIA CUDA  (`torch.cuda.is_available()`)
    2. Apple MPS    (`torch.backends.mps.is_available()`)
    3. OpenVINO     (Raspberry Pi 5 + Intel) — heuristic
    4. CPU
"""

from __future__ import annotations

import logging
import os
import platform

log = logging.getLogger(__name__)


def detect_yolo_device(preference: str = "auto") -> str:
    """Return device string suitable for `YOLO(...)(device=...)`.

    Args:
        preference: One of "auto", "cpu", "cuda", "mps", "openvino".
    """
    if preference != "auto":
        return preference

    # NVIDIA
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            log.info("CUDA detected: %s", torch.cuda.get_device_name(0))
            return "cuda:0"
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and platform.machine() in {"arm64", "aarch64"}
        ):
            log.info("Apple MPS detected")
            return "mps"
    except ImportError:
        log.debug("torch not available")

    # OpenVINO heuristic — Raspberry Pi 5 with Bookworm
    if _is_raspberry_pi() and os.environ.get("VVA_USE_OPENVINO"):
        log.info("Raspberry Pi + OpenVINO opt-in")
        return "openvino"

    log.info("Falling back to CPU")
    return "cpu"


def _is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model", encoding="utf-8") as f:  # noqa: S108
            return "Raspberry Pi" in f.read()
    except OSError:
        return False
