"""HRTF-inspired stereo audio cue computation.

True Head-Related Transfer Function (HRTF) rendering requires convolving
an audio signal with a set of direction-dependent impulse responses measured
on an acoustic mannequin (MIT KEMAR, CIPIC database).  That approach demands
~700 KB of IR data per subject and a real-time convolution engine.

This module implements a *physics-derived approximation* suitable for
bone-conduction headphones and earphones used in assistive wearables:

  ITD  (Inter-aural Time Delay) — Woodworth & Schlosberg (1954)
  ─────────────────────────────────────────────────────────────
      Δt = (a / c) · (sin θ + θ)      for |θ| ≤ π/2

  where
      a = head radius  ≈ 0.0875 m  (ISO 7250-1 adult average)
      c = speed of sound ≈ 343 m/s  (20 °C, 1 atm)
      θ = azimuth angle in radians  (positive → source to the right)

  Maximum Δt ≈ 0.65 ms at θ = π/2.

  ILD  (Inter-aural Level Difference) — Feddersen et al. (1957)
  ─────────────────────────────────────────────────────────────
      ΔL = L_max · sin θ    (dB),   L_max ≈ 6.5 dB broadband empirical peak

  Per-channel gains — equal-power panning law
  ───────────────────────────────────────────
      φ = (pan + 1) / 2 · (π/2)      φ ∈ [0, π/2]
      g_L = cos(φ) · G_dist
      g_R = sin(φ) · G_dist

  ILD correction applied on top:
      source right:  g_R ×= 10^(|ΔL|/20),  g_L /= 10^(|ΔL|/20)
      source left:   g_L ×= 10^(|ΔL|/20),  g_R /= 10^(|ΔL|/20)

References:
    Woodworth, R. S., & Schlosberg, H. (1954). Experimental Psychology.
    Blauert, J. (1997). Spatial Hearing. MIT Press.
    Feddersen, W. E. et al. (1957). JASA 29(9), 988–991.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ── Physical constants ────────────────────────────────────────────────────────
_HEAD_RADIUS_M: float = 0.0875   # ISO 7250-1 adult average (metres)
_SPEED_OF_SOUND: float = 343.0   # m/s at 20 °C, 1 atm
_MAX_ILD_DB: float = 6.5         # broadband empirical peak (Feddersen 1957)


# ── Public types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HrtfCue:
    """Stereo rendering parameters derived from source azimuth + distance.

    Attributes:
        left_gain:    Linear amplitude scale for the left channel [0, 1].
        right_gain:   Linear amplitude scale for the right channel [0, 1].
        itd_ms:       Inter-aural time delay in milliseconds.
                      Positive → right ear leads (source to the right).
        ild_db:       Inter-aural level difference in dB.
                      Positive → right ear louder.
        azimuth_deg:  Source azimuth in degrees (−90 left, +90 right).
    """

    left_gain: float
    right_gain: float
    itd_ms: float
    ild_db: float
    azimuth_deg: float


# ── Calculator ────────────────────────────────────────────────────────────────

class HrtfCalculator:
    """Compute HrtfCue from a normalised stereo-pan value.

    The *pan* input matches the ``SpatialCue.pan`` produced by
    ``SpatialPanner``, so the two classes are drop-in alternatives.

    Args:
        head_radius:    Head radius in metres (default ISO 7250-1: 0.0875).
        speed_of_sound: Speed of sound in m/s (default 343.0 at 20 °C).
        max_ild_db:     Peak ILD in dB (default 6.5 broadband empirical).
    """

    def __init__(
        self,
        head_radius: float = _HEAD_RADIUS_M,
        speed_of_sound: float = _SPEED_OF_SOUND,
        max_ild_db: float = _MAX_ILD_DB,
    ) -> None:
        self._a = head_radius
        self._c = speed_of_sound
        self._l_max = max_ild_db

    def compute(self, pan: float, distance_gain: float = 1.0) -> HrtfCue:
        """Return stereo rendering parameters for a given horizontal position.

        Args:
            pan:           Normalised horizontal position [−1, +1].
                           −1 = far left, 0 = centre, +1 = far right.
            distance_gain: Overall amplitude scale [0, 1]; use
                           ``SpatialPanner.for_position().distance_gain``.

        Returns:
            ``HrtfCue`` with left/right gains, ITD, ILD, and azimuth.
        """
        pan_c = max(-1.0, min(1.0, pan))
        dist_c = max(0.0, min(1.0, distance_gain))

        # Azimuth: pan ±1 → ±90°
        azimuth_rad = pan_c * (math.pi / 2.0)
        azimuth_deg = math.degrees(azimuth_rad)

        # ── ITD (Woodworth & Schlosberg 1954) ──────────────────────────────
        itd_s = (self._a / self._c) * (math.sin(azimuth_rad) + azimuth_rad)
        itd_ms = itd_s * 1_000.0

        # ── ILD (Feddersen et al. 1957, simplified sinusoidal model) ──────
        ild_db = self._l_max * math.sin(azimuth_rad)

        # ── Per-channel gains: equal-power panning law ─────────────────────
        # φ ∈ [0, π/2]:  φ=0 → full left,  φ=π/2 → full right
        phi = (pan_c + 1.0) / 2.0 * (math.pi / 2.0)
        left_gain = math.cos(phi) * dist_c
        right_gain = math.sin(phi) * dist_c

        # ── ILD correction on top of equal-power base ──────────────────────
        ild_linear = 10.0 ** (abs(ild_db) / 20.0)
        if ild_db > 0.0:          # source right: boost right, attenuate left
            right_gain = min(1.0, right_gain * ild_linear)
            left_gain = max(0.0, left_gain / ild_linear)
        elif ild_db < 0.0:        # source left: boost left, attenuate right
            left_gain = min(1.0, left_gain * ild_linear)
            right_gain = max(0.0, right_gain / ild_linear)

        return HrtfCue(
            left_gain=round(left_gain, 6),
            right_gain=round(right_gain, 6),
            itd_ms=round(itd_ms, 4),
            ild_db=round(ild_db, 4),
            azimuth_deg=round(azimuth_deg, 2),
        )
