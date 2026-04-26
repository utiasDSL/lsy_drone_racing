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
    approach_tol: float = 0.10
    replan_margin: float = 0.10
    route_progress: float = 0.16
    max_route_points: int = 4
    path_spacing: float = 0.06
    path_lookahead: float = 0.34
    pre_gate_lookahead: float = 0.20
    path_smoothing_iterations: int = 2
    corner_radius: float = 0.24
    corner_samples: int = 6
    final_straight_length: float = 0.26
    tangent_smoothing_distance: float = 0.14
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

    takeoff_speed: float = 0.55
    route_speed: float = 0.90
    approach_speed: float = 0.65
    pass_speed: float = 0.95
    pos_gain: float = 1.25
    vel_damping: float = 0.28
    acc_gain: float = 2.0
    vertical_acc_gain: float = 1.2
    max_vertical_acc: float = 4.0
    max_takeoff_lateral_acc: float = 0.25
    max_scan_lateral_acc: float = 0.6
    max_route_lateral_acc: float = 2.0
    max_approach_lateral_acc: float = 1.4
    max_pass_lateral_acc: float = 1.8
    velocity_filter_gain: float = 0.22
    accel_filter_gain: float = 0.28
