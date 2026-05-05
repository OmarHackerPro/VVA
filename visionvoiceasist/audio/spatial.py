"""Spatial-audio panner — maps detection horizontal position to stereo pan.

Real HRTF requires a convolution against a measured impulse-response set
(e.g. CIPIC, MIT KEMAR). This module implements a *placeholder* panner
suitable for headphones / bone-conduction without external dependencies;
the runtime can drop in a true HRTF backend later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialCue:
    pan: float                  # [-1, 1]; -1 = full left
    elevation: float = 0.0      # reserved for future HRTF use
    distance_gain: float = 1.0  # 0..1; nearer = louder


class SpatialPanner:
    """Maps a detection's pixel position + area to a SpatialCue."""

    def __init__(self, frame_w: int = 640) -> None:
        self._w = frame_w

    def for_position(self, cx: int, area_pct: float = 0.0) -> SpatialCue:
        """Return a SpatialCue from horizontal centre and bbox area."""
        if self._w <= 0:
            return SpatialCue(pan=0.0)
        normalised_x = (cx - self._w / 2) / (self._w / 2)
        pan = max(-1.0, min(1.0, normalised_x))
        # Larger area → closer object → louder.
        gain = max(0.4, min(1.0, 0.4 + area_pct * 1.2))
        return SpatialCue(pan=pan, distance_gain=gain)

    def update_frame_width(self, w: int) -> None:
        if w > 0:
            self._w = w
