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

    takeoff_height: float = 0.2
    takeoff_reached_distance: float = 0.1
    gate_inner_width: float = 0.40
    gate_inner_height: float = 0.40
    gate_safety_margin: float = 0.075
    d_pre: float = 0.55
    d_post: float = 0.55
    saturation_radius: float = 0.6
    obstacle_detour_margin: float = 0.25
    gate_avoidance_radius: float = 0.45
    max_detour_iterations: int = 5
    min_waypoint_spacing: float = 0.20
    collinear_angle_threshold_deg: float = 8.0
    sample_spacing: float = 0.04


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Closed-loop reference advancement parameters."""

    target_reached_distance: float = 0.16
    target_hysteresis: float = 0.03
    min_ticks_between_advances: int = 1
    start_index: int = 1
    max_advance_per_step: int = 3
    nearest_forward_search: int = 6
    nominal_speed: float = 0.48
    gate_speed: float = 0.34
    final_speed: float = 0.26
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
