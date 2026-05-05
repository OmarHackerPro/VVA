"""Command-line entry point: ``vva`` or ``python -m visionvoiceasist``."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .settings import Settings
from .utils.logging import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vva",
        description="VisionVoiceAsist — AI assistive vision system.",
    )
    p.add_argument("--version", action="version", version=f"vva {__version__}")
    p.add_argument("--cam", type=int, help="Camera index override")
    p.add_argument("--conf", type=float, help="YOLO confidence override")
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps", "openvino"],
        help="YOLO device override",
    )
    p.add_argument("--noai", action="store_true", help="Disable cloud AI")
    p.add_argument("--nogui", action="store_true", help="Headless mode (no OpenCV window)")
    p.add_argument("--dashboard", action="store_true", help="Run web dashboard alongside")
    p.add_argument(
        "--offline",
        choices=["auto", "always", "never"],
        help="Force offline mode policy",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level override",
    )
    return p


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    overrides: dict[str, object] = {}
    if args.cam is not None:
        overrides["camera"] = settings.camera.__class__(
            **{**settings.camera.__dict__, "index": args.cam}
        )
    if args.conf is not None or args.device is not None:
        overrides["yolo"] = settings.yolo.__class__(
            **{
                **settings.yolo.__dict__,
                **({"conf": args.conf} if args.conf is not None else {}),
                **({"device": args.device} if args.device is not None else {}),
            }
        )
    if args.noai:
        overrides["ai"] = settings.ai.__class__(
            **{**settings.ai.__dict__, "gemini_key": ""}
        )
    if args.offline:
        overrides["ai"] = (overrides.get("ai") or settings.ai).__class__(
            **{**(overrides.get("ai") or settings.ai).__dict__, "offline_mode": args.offline}
        )
    if args.dashboard:
        overrides["dashboard"] = settings.dashboard.__class__(
            **{**settings.dashboard.__dict__, "enabled": True}
        )
    if args.nogui:
        overrides["show_gui"] = False
    if args.log_level:
        overrides["log_level"] = args.log_level
    return settings.with_overrides(**overrides) if overrides else settings


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = _apply_overrides(Settings.from_env(), args)
    configure_logging(level=settings.log_level, log_dir=settings.log_dir)
    log = logging.getLogger("vva")

    log.info("VisionVoiceAsist v%s starting (gui=%s, dashboard=%s, offline=%s)",
             __version__, settings.show_gui,
             settings.dashboard.enabled, settings.ai.offline_mode)

    # Lazy import — keeps `vva --version` fast and avoids loading cv2/torch
    # for users who only want help text.
    from .runtime import Runtime

    rt = Runtime(settings)
    try:
        rt.run()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception:  # noqa: BLE001
        log.exception("Fatal error in runtime")
        return 1
    finally:
        rt.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
