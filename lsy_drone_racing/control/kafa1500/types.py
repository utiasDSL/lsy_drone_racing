"""Shared types for the KaFa1500 controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

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
