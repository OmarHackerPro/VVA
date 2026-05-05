"""VisionVoiceAsist — AI-powered wearable assistive system for visually impaired users.

Public API: import the high-level entry points from this package root.

Example:
    >>> from visionvoiceasist import Settings, Runtime
    >>> settings = Settings.from_env()
    >>> Runtime(settings).run()
"""

from __future__ import annotations

__version__ = "5.0.0"

from .events import EventBus, EventType
from .settings import Settings
from .types import Detection, Priority, SpeechEvent

__all__ = [
    "Detection",
    "EventBus",
    "EventType",
    "Priority",
    "Settings",
    "SpeechEvent",
    "__version__",
]
