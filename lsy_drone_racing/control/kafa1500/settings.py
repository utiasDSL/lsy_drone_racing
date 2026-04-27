"""Tunable parameters for the KaFa1500 controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlannerSettings:
    """Navigation and path-planning parameters."""

    scan_hold_steps: int = 1
    align_distance: float = 0.95
    approach_distance: float = 0.62
    pass_distance: float = 0.34
    takeoff_tol: float = 0.12
    approach_tol: float = 0.12
    replan_margin: float = 0.10
    route_progress: float = 0.22
    max_route_points: int = 4
    path_spacing: float = 0.06
    spline_sample_spacing: float = 0.01
    path_lookahead: float = 0.55
    pre_gate_lookahead: float = 0.36
    path_smoothing_iterations: int = 4
    control_relaxation_iterations: int = 4
    control_relaxation_weight: float = 0.55
    visibility_node_samples: int = 16
    visibility_clearance_margin: float = 0.12
    corner_radius: float = 0.36
    corner_samples: int = 24
    final_straight_length: float = 0.16
    drone_radius: float = 0.08
    obstacle_radius: float = 0.015
    tracking_margin: float = 0.16

    @property
    def obstacle_clearance(self) -> float:
        """Combined XY clearance around each obstacle pole."""

        return self.drone_radius + self.obstacle_radius + self.tracking_margin


@dataclass(frozen=True, slots=True)
class ActionSettings:
    """Tracking and command shaping parameters."""

    takeoff_speed: float = 0.40
    route_speed: float = 0.58
    approach_speed: float = 0.44
    pass_speed: float = 0.52
    pos_gain: float = 0.82
    vel_damping: float = 0.45
    acc_gain: float = 1.00
    vertical_acc_gain: float = 0.80
    max_vertical_acc: float = 2.2
    max_takeoff_lateral_acc: float = 0.22
    max_scan_lateral_acc: float = 0.38
    max_route_lateral_acc: float = 0.70
    max_approach_lateral_acc: float = 0.55
    max_pass_lateral_acc: float = 0.60
    velocity_filter_gain: float = 0.10
    accel_filter_gain: float = 0.13
