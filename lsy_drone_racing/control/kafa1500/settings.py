"""Tunable parameters for the KaFa1500 controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlannerSettings:
    """Navigation and path-planning parameters."""

    scan_hold_steps: int = 6
    max_scan_hold_steps: int = 18
    scan_release_speed: float = 0.65
    align_distance: float = 0.95
    approach_distance: float = 0.62
    pass_distance: float = 0.44
    takeoff_tol: float = 0.12
    approach_tol: float = 0.12
    replan_margin: float = 0.10
    max_pass_overshoot: float = 0.32
    route_progress: float = 0.26
    max_route_points: int = 4
    path_spacing: float = 0.06
    spline_sample_spacing: float = 0.01
    path_lookahead: float = 0.46
    pre_gate_lookahead: float = 0.28
    final_corridor_distance: float = 0.34
    path_smoothing_iterations: int = 5
    control_relaxation_iterations: int = 5
    control_relaxation_weight: float = 0.62
    obstacle_route_relaxation_iterations: int = 3
    visibility_node_samples: int = 24
    visibility_clearance_margin: float = 0.18
    visibility_turn_weight: float = 0.40
    corner_radius: float = 0.42
    corner_samples: int = 28
    final_straight_length: float = 0.14
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

    takeoff_speed: float = 0.38
    route_speed: float = 0.48
    approach_speed: float = 0.36
    pass_speed: float = 0.34
    pos_gain: float = 0.82
    vel_damping: float = 0.45
    tangent_correction_gain: float = 0.42
    tangent_velocity_damping: float = 0.22
    acc_gain: float = 1.00
    vertical_acc_gain: float = 0.80
    max_vertical_acc: float = 2.2
    max_takeoff_lateral_acc: float = 0.22
    max_scan_lateral_acc: float = 0.85
    max_route_lateral_acc: float = 0.72
    max_approach_lateral_acc: float = 0.62
    max_pass_lateral_acc: float = 0.48
    velocity_filter_gain: float = 0.10
    accel_filter_gain: float = 0.13
