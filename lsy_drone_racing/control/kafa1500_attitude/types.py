"""Shared types for the KaFa1500 attitude controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scipy.interpolate import PPoly

Observation = dict[str, NDArray[np.floating]]
Vec3 = NDArray[np.float32]


class FlightPhase(Enum):
    """High-level controller phases."""

    TAKEOFF = auto()
    TRACK = auto()
    FINISH = auto()


@dataclass(frozen=True, slots=True)
class GateFrame:
    """World-frame gate axes and safe aperture dimensions."""

    index: int
    position: Vec3
    forward: Vec3
    lateral: Vec3
    up: Vec3
    traversal: Vec3
    safe_half_width: float
    safe_half_height: float


@dataclass(frozen=True, slots=True)
class CubicPath:
    """Strictly cubic path plus samples used by the reference manager."""

    spline: PPoly
    velocity_spline: PPoly
    acceleration_spline: PPoly
    params: NDArray[np.float32]
    points: NDArray[np.float32]
    lengths: NDArray[np.float32]
    gate_indices: NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class Reference:
    """Active target sampled from the cubic path."""

    position: Vec3
    velocity: Vec3
    acceleration: Vec3
    yaw: float
    index: int
    distance: float
    done: bool
