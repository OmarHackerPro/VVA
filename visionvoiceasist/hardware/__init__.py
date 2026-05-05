"""Hardware abstraction: GPIO haptics, IMU."""

from .haptics import HapticMotor
from .imu import ImuSensor

__all__ = ["HapticMotor", "ImuSensor"]
