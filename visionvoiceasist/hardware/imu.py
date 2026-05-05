"""Optional MPU-6050 IMU for head-tracking spatial audio.

Stub implementation — exposes the interface so the runtime can use IMU
data when present, and silently returns identity when absent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeadOrientation:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    timestamp: float = 0.0


class ImuSensor:
    """MPU-6050 IMU wrapper. Returns zeros if hardware not present."""

    def __init__(self, i2c_bus: int = 1, address: int = 0x68) -> None:
        self._bus_num = i2c_bus
        self._addr = address
        self._bus: object | None = None
        self._init()

    @property
    def is_present(self) -> bool:
        return self._bus is not None

    def _init(self) -> None:
        try:
            import smbus  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError:
            log.debug("smbus not available — IMU stub")
            return
        try:
            bus = smbus.SMBus(self._bus_num)
            # Wake up MPU-6050 from sleep mode.
            bus.write_byte_data(self._addr, 0x6B, 0)
            self._bus = bus
            log.info("IMU MPU-6050 ready on i2c-%d 0x%X", self._bus_num, self._addr)
        except Exception as exc:  # noqa: BLE001
            log.info("IMU not present: %s", exc)

    def read(self) -> HeadOrientation:
        """Read current head orientation. Returns zeros if no hardware."""
        if self._bus is None:
            return HeadOrientation(timestamp=time.time())
        try:
            ax = self._read_word(0x3B) / 16384.0
            ay = self._read_word(0x3D) / 16384.0
            az = self._read_word(0x3F) / 16384.0
        except Exception:  # noqa: BLE001
            return HeadOrientation(timestamp=time.time())
        # Tiny-angle approximation good enough for audio panning.
        import math  # noqa: PLC0415

        roll = math.degrees(math.atan2(ay, az))
        pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        return HeadOrientation(
            yaw_deg=0.0, pitch_deg=pitch, roll_deg=roll, timestamp=time.time()
        )

    def _read_word(self, reg: int) -> int:
        assert self._bus is not None  # noqa: S101
        h = self._bus.read_byte_data(self._addr, reg)  # type: ignore[attr-defined]
        l = self._bus.read_byte_data(self._addr, reg + 1)  # type: ignore[attr-defined]
        v = (h << 8) + l
        return v - 65536 if v >= 32768 else v
