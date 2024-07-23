"""Rotation conversion utils module."""

from __future__ import annotations

from math import asin, atan2
from typing import TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", float, npt.NDArray[np.floating])


def euler_from_quaternion(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Convert a quaternion into euler angles (roll, pitch, yaw).

    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def map2pi(angle: T) -> T:
    """Map an angle or array of angles to the interval of [-pi, pi].

    Args:
        angle: Number or array of numbers.

    Returns:
        The remapped angles.
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi
