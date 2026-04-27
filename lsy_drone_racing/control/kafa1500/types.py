"""Shared types for the KaFa1500 controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scipy.interpolate import BSpline

Observation = dict[str, NDArray[np.floating]]
Vec3 = NDArray[np.float32]


class KaFa1500State(Enum):
    """High-level states for gate-by-gate navigation."""

    TAKEOFF = auto()
    SCAN = auto()
    APPROACH = auto()
    PASS_GATE = auto()
    FINISH = auto()


@dataclass(frozen=True, slots=True)
class GateFrame:
    """World-frame pose description for a gate."""

    position: Vec3
    forward: Vec3
    lateral: Vec3


@dataclass(slots=True)
class GatePlan:
    """Local route and path data for the currently targeted gate."""

    gate_idx: int
    gate_pos: Vec3
    gate_x: Vec3
    pass_target: Vec3
    path_spline: BSpline
    path_spline_d1: BSpline
    path_params: NDArray[np.float32]
    path_points: NDArray[np.float32]
    path_lengths: NDArray[np.float32]
    route_line: NDArray[np.float32]
    progress: float = 0.0


@dataclass(frozen=True, slots=True)
class PathTarget:
    """Lookahead target sampled from the current smooth path."""

    target: Vec3
    yaw_dir: Vec3
    remaining: float
