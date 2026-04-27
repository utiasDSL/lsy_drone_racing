"""Tunable parameters for KaFa1500 attitude control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _vec(values: tuple[float, float, float]) -> NDArray[np.float32]:
    """Create a float32 vector dataclass default."""
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class PathConfig:
    """Gate-aware cubic path generation parameters."""

    takeoff_height: float = 0.82
    takeoff_reached_distance: float = 0.14
    gate_inner_width: float = 0.40
    gate_inner_height: float = 0.40
    gate_safety_margin: float = 0.075
    vertical_reference_offset: float = -0.035
    d_pre: float = 0.42
    d_post: float = 0.32
    d_pass: float = 0.56
    d_pre_per_gate: tuple[float, ...] = (0.50, 0.42, 0.38, 0.36)
    d_post_per_gate: tuple[float, ...] = (0.32, 0.34, 0.36, 0.38)
    previous_gate_escape_distance: float = 0.28
    previous_gate_escape_window: float = 0.24
    previous_gate_escape_lateral_offsets: tuple[float, ...] = (
        0.0,
        0.18,
        -0.18,
        0.30,
        -0.30,
        0.42,
        -0.42,
    )
    obstacle_radius: float = 0.015
    drone_radius: float = 0.10
    obstacle_tracking_margin: float = 0.24
    bypass_extra: float = 0.22
    max_bypass_points: int = 3
    sample_spacing: float = 0.055
    control_smoothing_passes: int = 2
    control_smoothing_weight: float = 0.35

    @property
    def obstacle_clearance(self) -> float:
        """XY clearance used around obstacle poles."""
        return self.drone_radius + self.obstacle_radius + self.obstacle_tracking_margin


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Closed-loop reference advancement parameters."""

    target_reached_distance: float = 0.16
    target_hysteresis: float = 0.03
    min_ticks_between_advances: int = 1
    start_index: int = 1
    max_advance_per_step: int = 3
    nearest_forward_search: int = 6
    nominal_speed: float = 0.38
    gate_speed: float = 0.30
    final_speed: float = 0.25
    gate_window_samples: int = 5
    follow_path_yaw: bool = False


@dataclass(frozen=True, slots=True)
class FeedbackConfig:
    """Attitude feedback gains and output limits."""

    kp: NDArray[np.float32] = field(default_factory=lambda: _vec((0.40, 0.40, 1.25)))
    ki: NDArray[np.float32] = field(default_factory=lambda: _vec((0.04, 0.04, 0.045)))
    kd: NDArray[np.float32] = field(default_factory=lambda: _vec((0.22, 0.22, 0.45)))
    integral_limit: NDArray[np.float32] = field(default_factory=lambda: _vec((1.50, 1.50, 0.40)))
    feedforward_acc_scale: float = 0.0
    max_feedforward_acc: float = 4.5
    max_tilt: float = 0.46
    max_yaw_rate_step: float = 0.10
    attitude_smoothing: float = 0.50
    thrust_smoothing: float = 0.45
    hover_thrust_scale: float = 1.00
    gravity: float = 9.81
