from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class PlannerConfig:
    d_pre: float = 0.45
    d_post: float = 0.30
    d_stop: float = 0.30
    v_cruise: float = 0.8
    v_cruise_inter: float = 0.0  # inter-gate cruise (>v_cruise to speed up). 0 disables.
    t_min_seg: float = 0.4
    r_obs: float = 0.28
    # Per-gate overrides (indexed by original gate index, 0-based). Entries
    # missing or NaN fall back to the global ``d_pre`` / ``d_post`` /
    # ``v_cruise``. Lets us shrink approach on tight gates while keeping a
    # longer runway on others, or run a specific gate slower without
    # penalising the whole track.
    d_pre_per_gate: tuple[float, ...] = ()
    d_post_per_gate: tuple[float, ...] = ()
    v_peri_per_gate: tuple[float, ...] = ()

    def d_pre_for(self, gate_index: int) -> float:
        """Per-gate approach distance, falling back to the global ``d_pre``."""
        if 0 <= gate_index < len(self.d_pre_per_gate):
            gate_value = self.d_pre_per_gate[gate_index]
            if np.isfinite(gate_value) and gate_value > 0:
                return float(gate_value)
        return self.d_pre

    def d_post_for(self, gate_index: int) -> float:
        """Per-gate exit distance, falling back to the global ``d_post``."""
        if 0 <= gate_index < len(self.d_post_per_gate):
            gate_value = self.d_post_per_gate[gate_index]
            if np.isfinite(gate_value) and gate_value > 0:
                return float(gate_value)
        return self.d_post

    def v_peri_for(self, gate_index: int) -> float:
        """Per-gate peri-cruise speed, falling back to the global ``v_cruise``."""
        if 0 <= gate_index < len(self.v_peri_per_gate):
            gate_value = self.v_peri_per_gate[gate_index]
            if np.isfinite(gate_value) and gate_value > 0:
                return float(gate_value)
        return self.v_cruise

    # Time-optimal refinement caps (0 to disable). The heuristic refiner
    # iterates per-segment toward target utilization; the slsqp refiner
    # solves a small NLP (min sum(seg_t) s.t. peak vel/accel caps).
    max_vel: float = 0.0
    max_accel: float = 0.0
    use_slsqp: bool = False
    # M3-lite: replace the cubic interpolator with a quintic (k=5) spline.
    # Quintic has continuous jerk (C⁴) across knots — removing the cubic's
    # discontinuous-snap spikes that force the MPC to over-brake at waypoint
    # transitions. Time allocation and waypoint sequence are unchanged.
    use_quintic: bool = False
    # Path 1: drop conservative clearance + turn_apex waypoints between gates.
    # The MPC's wing soft constraints (current-gate L/R/T/B) prevent frame
    # clipping on the entry side; if we also trust them on the exit side,
    # the ~0.5–0.8 s per transition spent traversing clearance points vanishes.
    skip_clearance: bool = False


@dataclass
class Plan:
    waypoints: np.ndarray  # (n, 3)
    t_knots: np.ndarray  # (n,)
    pos_spline: CubicSpline
    vel_spline: CubicSpline
    t_total: float


def build_plan(
    start_pos: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    gates_quat: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    target_gate: int,
    cfg: PlannerConfig = PlannerConfig(),
) -> Plan:
    """Build a clamped cubic spline through gates ``[target_gate:]`` from ``start_pos``."""
    start_pos = np.asarray(start_pos, dtype=np.float64)
    start_vel = np.asarray(start_vel, dtype=np.float64)
    gates_pos = np.asarray(gates_pos, dtype=np.float64)
    gates_quat = np.asarray(gates_quat, dtype=np.float64)
    obstacles_pos = np.asarray(obstacles_pos, dtype=np.float64)

    remaining_gate_positions = gates_pos[target_gate:]
    remaining_gate_quaternions = gates_quat[target_gate:]
    waypoints = _build_waypoints(
        start_pos,
        remaining_gate_positions,
        remaining_gate_quaternions,
        obstacles_pos,
        cfg,
        target_gate=target_gate,
    )
    if target_gate > 0 and target_gate < len(gates_pos) and not cfg.skip_clearance:
        previous_gate_position = gates_pos[target_gate - 1]
        if float(np.linalg.norm(start_pos - previous_gate_position)) < 0.6:
            clearance_waypoints = _exited_gate_clearance(
                previous_gate_position,
                gates_quat[target_gate - 1],
                gates_pos[target_gate],
                gates_quat[target_gate],
                obstacles_pos,
                cfg,
                prev_gi_abs=target_gate - 1,
                next_gi_abs=target_gate,
            )
            if clearance_waypoints is not None:
                waypoints = np.vstack([waypoints[:1], clearance_waypoints, waypoints[1:]])
    knot_times, spline = _build_spline(
        waypoints, start_vel, obstacles_pos, cfg, gates_pos=gates_pos
    )
    return Plan(
        waypoints=waypoints,
        t_knots=knot_times,
        pos_spline=spline,
        vel_spline=spline.derivative(1),
        t_total=float(knot_times[-1]),
    )


def _exit_axis_obstructed(
    gp: NDArray[np.floating],
    x_axis: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    r_obs: float,
    max_dist: float,
) -> bool:
    """Check whether an obstacle sits along the gate's exit axis.

    Returns ``True`` if any obstacle is within ``r_obs`` of the axis up to
    ``max_dist`` m past the gate — i.e. a straight clearance detour along the
    axis would graze the obstacle and need to be routed further out.
    """
    if len(obstacles_pos) == 0:
        return False
    gate_axis_xy = x_axis[:2]
    gate_axis_norm = float(np.linalg.norm(gate_axis_xy))
    if gate_axis_norm < 1e-6:
        return False
    gate_axis_unit = gate_axis_xy / gate_axis_norm
    for obstacle in obstacles_pos:
        relative_obstacle = obstacle[:2] - gp[:2]
        along = float(np.dot(relative_obstacle, gate_axis_unit))
        if along < 0.0 or along > max_dist:
            continue
        lateral = float(np.linalg.norm(relative_obstacle - along * gate_axis_unit))
        if lateral < r_obs:
            return True
    return False


def _clearance_distance(
    gp: NDArray[np.floating],
    x_axis: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    r_obs: float,
) -> float:
    """Return the past-exit clearance distance.

    Short (0.35) when the exit axis is obstacle-free, long (0.60) when an
    obstacle sits along the axis and the clearance must push past it.
    """
    if _exit_axis_obstructed(gp, x_axis, obstacles_pos, r_obs + 0.08, 1.0):
        return 0.60
    return 0.35


def _exited_gate_clearance(
    prev_gp: NDArray[np.floating],
    prev_quat: NDArray[np.floating],
    next_gp: NDArray[np.floating],
    next_quat: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    cfg: PlannerConfig,
    prev_gi_abs: int = -1,
    next_gi_abs: int = -1,
) -> NDArray[np.floating] | None:
    """Return clearance + turn_apex waypoints for a recently-exited gate.

    Mirrors the logic in ``_build_waypoints`` that runs between gate i's exit
    and gate i+1's approach, but callable standalone for use on replans where
    the start position is near the just-exited gate.
    """
    if abs(float(next_gp[2]) - float(prev_gp[2])) <= 0.15:
        return None
    previous_gate_rotation = R.from_quat(prev_quat).as_matrix()
    previous_gate_forward = previous_gate_rotation[:, 0]
    next_gate_rotation = R.from_quat(next_quat).as_matrix()
    next_gate_forward = next_gate_rotation[:, 0]
    previous_post_distance = cfg.d_post_for(prev_gi_abs)
    next_pre_distance = cfg.d_pre_for(next_gi_abs)
    next_approach = next_gp - next_pre_distance * next_gate_forward
    clearance_xy = (prev_gp + (previous_post_distance + 0.60) * previous_gate_forward)[:2]
    if float(next_gp[2]) > float(prev_gp[2]):
        clearance_z = max(float(prev_gp[2]) + 0.55, float(next_gp[2]) - 0.05)
        apex_z = float(next_gp[2]) - 0.05
    else:
        clearance_z = max(float(prev_gp[2]) - 0.30, float(next_gp[2]) + 0.15)
        apex_z = float(next_gp[2]) + 0.05
    clearance = np.array([clearance_xy[0], clearance_xy[1], clearance_z])
    mid_xy = 0.5 * (clearance_xy + next_approach[:2])
    away_from_previous_gate = mid_xy - prev_gp[:2]
    away_norm = float(np.linalg.norm(away_from_previous_gate))
    if away_norm > 1e-6:
        mid_xy = mid_xy + (away_from_previous_gate / away_norm) * 0.10
    turn_apex = np.array([mid_xy[0], mid_xy[1], apex_z])
    return np.stack([clearance, turn_apex])


def _build_waypoints(
    start_pos: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    gates_quat: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    cfg: PlannerConfig,
    target_gate: int = 0,
) -> NDArray[np.floating]:
    waypoints: list[NDArray[np.floating]] = [start_pos.copy()]
    if start_pos[2] < 0.15 and len(gates_pos) > 0:
        first_gate_z = float(gates_pos[0][2])
        target_z = max(0.55, 0.85 * first_gate_z)
        toward_first = gates_pos[0][:2] - start_pos[:2]
        distance_to_first_gate = float(np.linalg.norm(toward_first))
        offset_xy = (
            toward_first / distance_to_first_gate * 0.40
            if distance_to_first_gate > 1e-6
            else np.zeros(2)
        )
        liftoff = np.array([start_pos[0] + offset_xy[0], start_pos[1] + offset_xy[1], target_z])
        waypoints.append(liftoff)
    n_gates = len(gates_pos)
    for local_gate_index, (gate_position, gate_quat) in enumerate(zip(gates_pos, gates_quat)):
        absolute_gate_index = target_gate + local_gate_index
        pre_distance = cfg.d_pre_for(absolute_gate_index)
        post_distance = cfg.d_post_for(absolute_gate_index)
        gate_rotation = R.from_quat(gate_quat).as_matrix()
        gate_forward, gate_lateral = gate_rotation[:, 0], gate_rotation[:, 1]
        approach_raw = gate_position - pre_distance * gate_forward
        exit_raw = gate_position + post_distance * gate_forward
        previous_waypoint = waypoints[-1]
        lateral_bias = float(np.dot((previous_waypoint - approach_raw)[:2], gate_lateral[:2]))
        bias_sign = np.sign(lateral_bias) if abs(lateral_bias) > 1e-3 else 0.0
        approach = _nudge_lateral(
            approach_raw, gate_lateral, obstacles_pos, cfg.r_obs, bias_sign=bias_sign
        )
        exit_waypoint = _nudge_lateral(exit_raw, gate_lateral, obstacles_pos, cfg.r_obs)
        gap_to_approach = float(np.linalg.norm((approach - previous_waypoint)[:2]))
        lateral_offset = float(abs(np.dot((previous_waypoint - approach)[:2], gate_lateral[:2])))
        if gap_to_approach > 0.55 and lateral_offset > 0.12:
            far_approach = _nudge_lateral(
                gate_position - (pre_distance + 0.50) * gate_forward,
                gate_lateral,
                obstacles_pos,
                cfg.r_obs,
                bias_sign=bias_sign,
            )
            waypoints.append(far_approach)
        nudge_dist = float(np.linalg.norm((approach - approach_raw)[:2]))
        if nudge_dist > 0.05:
            near_gate = gate_position - 0.15 * gate_forward
            waypoints.extend([approach, near_gate, gate_position.copy(), exit_waypoint])
        else:
            waypoints.extend([approach, gate_position.copy(), exit_waypoint])
        if local_gate_index + 1 < n_gates and not cfg.skip_clearance:
            next_gate_position = gates_pos[local_gate_index + 1]
            next_gate_rotation = R.from_quat(gates_quat[local_gate_index + 1]).as_matrix()
            next_gate_forward = next_gate_rotation[:, 0]
            next_pre_distance = cfg.d_pre_for(absolute_gate_index + 1)
            next_approach_raw = next_gate_position - next_pre_distance * next_gate_forward
            next_z = float(next_approach_raw[2])
            if abs(next_z - gate_position[2]) > 0.15:
                clearance_xy = (gate_position + (post_distance + 0.60) * gate_forward)[:2]
                if next_z > gate_position[2]:
                    clearance_z = max(gate_position[2] + 0.55, next_z - 0.05)
                else:
                    # Going down: keep clearance ABOVE next gate so cubic spline
                    # doesn't undershoot below it on the way in.
                    clearance_z = max(gate_position[2] - 0.30, next_z + 0.15)
                clearance = np.array([clearance_xy[0], clearance_xy[1], clearance_z])
                waypoints.append(clearance)
                next_approach_xy = next_approach_raw[:2]
                mid_xy = 0.5 * (clearance_xy + next_approach_xy)
                away_from_gate = mid_xy - gate_position[:2]
                away_norm = float(np.linalg.norm(away_from_gate))
                if away_norm > 1e-6:
                    mid_xy = mid_xy + (away_from_gate / away_norm) * 0.10
                if next_z > gate_position[2]:
                    apex_z = next_z - 0.05
                else:
                    apex_z = next_z + 0.05
                turn_apex = np.array([mid_xy[0], mid_xy[1], apex_z])
                waypoints.append(turn_apex)
    final_gate_forward = R.from_quat(gates_quat[-1]).as_matrix()[:, 0]
    final_gate_index = target_gate + n_gates - 1
    final_post_distance = cfg.d_post_for(final_gate_index)
    waypoints.append(gates_pos[-1] + (final_post_distance + cfg.d_stop) * final_gate_forward)
    waypoints_array = np.asarray(waypoints)
    return _insert_obstacle_midpoints(waypoints_array, obstacles_pos, cfg.r_obs)


def _approach_swing(
    gate_pos: NDArray[np.floating],
    x_axis: NDArray[np.floating],
    y_axis: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    d_pre: float,
    r_obs: float,
) -> NDArray[np.floating] | None:
    """Return a swing waypoint around an obstacle blocking the gate's approach corridor.

    Returns ``None`` if no obstacle sits in the corridor. The corridor spans
    ``[-(d_pre + r_obs), r_obs]`` along the gate x-axis (from slightly past the
    approach point to slightly past the gate center) with width
    ``r_obs + 0.15`` in the gate y-axis. If blocked, the swing is placed at the
    blocker's x-position (along gate x-axis) with a lateral offset of
    ``r_obs + 0.15`` on the side away from the blocker, at gate z-height.
    """
    gate_forward_xy = x_axis[:2]
    gate_lateral_xy = y_axis[:2]
    forward_norm = float(np.linalg.norm(gate_forward_xy))
    lateral_norm = float(np.linalg.norm(gate_lateral_xy))
    if forward_norm < 1e-6 or lateral_norm < 1e-6:
        return None
    forward_unit = gate_forward_xy / forward_norm
    lateral_unit = gate_lateral_xy / lateral_norm
    blocker = None
    best_lateral = 0.0
    best_along = 0.0
    smallest_lateral_abs = r_obs
    for obstacle in obstacles_pos:
        obstacle_delta = obstacle[:2] - gate_pos[:2]
        along = float(np.dot(obstacle_delta, forward_unit))
        lateral = float(np.dot(obstacle_delta, lateral_unit))
        # Obstacle blocks the approach if it sits between the approach waypoint
        # (along = -d_pre) and the gate center, roughly on the ray.
        if -(d_pre + 0.1) < along < 0.05 and abs(lateral) < smallest_lateral_abs:
            blocker = obstacle
            best_along = along
            best_lateral = lateral
            smallest_lateral_abs = abs(lateral)
    if blocker is None:
        return None
    side = -1.0 if best_lateral > 0 else 1.0
    swing_xy = gate_pos[:2] + best_along * forward_unit + side * (r_obs + 0.15) * lateral_unit
    swing = np.array([swing_xy[0], swing_xy[1], gate_pos[2]])
    return swing


def _nudge_lateral(
    point: NDArray[np.floating],
    lateral: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    r_obs: float,
    bias_sign: float = 0.0,
) -> NDArray[np.floating]:
    lateral_xy = lateral[:2]
    lateral_norm = float(np.linalg.norm(lateral_xy))
    if lateral_norm < 1e-6:
        return _nudge(point, lateral, obstacles_pos, r_obs)
    lateral_unit = lateral_xy / lateral_norm
    nudged_point = point.copy()
    margin = r_obs + 0.02
    for _ in range(4):
        offenders = [
            obstacle
            for obstacle in obstacles_pos
            if np.linalg.norm(nudged_point[:2] - obstacle[:2]) < margin
        ]
        if not offenders:
            break
        closest_obstacle = min(
            offenders, key=lambda obstacle: np.linalg.norm(nudged_point[:2] - obstacle[:2])
        )
        obstacle_delta = nudged_point[:2] - closest_obstacle[:2]
        delta_dot_lateral = float(np.dot(obstacle_delta, lateral_unit))
        delta_dot_delta = float(np.dot(obstacle_delta, obstacle_delta))
        discriminant = delta_dot_lateral**2 + margin**2 - delta_dot_delta
        if discriminant < 0:
            # lateral direction is exactly perpendicular to d; need radial fallback
            return _nudge(point, lateral, obstacles_pos, r_obs)
        root = np.sqrt(discriminant)
        positive_offset = -delta_dot_lateral + root
        negative_offset = -delta_dot_lateral - root
        if (
            bias_sign > 0
            and positive_offset > 0
            and abs(positive_offset) <= 1.5 * abs(negative_offset)
        ):
            lateral_offset = positive_offset
        elif (
            bias_sign < 0
            and negative_offset < 0
            and abs(negative_offset) <= 1.5 * abs(positive_offset)
        ):
            lateral_offset = negative_offset
        else:
            lateral_offset = (
                positive_offset if abs(positive_offset) < abs(negative_offset) else negative_offset
            )
        nudged_point[:2] = nudged_point[:2] + lateral_offset * lateral_unit
    return nudged_point


def _nudge(
    point: NDArray[np.floating],
    lateral: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    r_obs: float,
) -> NDArray[np.floating]:
    nudged_point = point.copy()
    for _ in range(4):
        distances = [
            float(np.linalg.norm(nudged_point[:2] - obstacle[:2])) for obstacle in obstacles_pos
        ]
        if not distances or min(distances) >= r_obs:
            break
        nearest_obstacle_index = int(np.argmin(distances))
        nearest_obstacle = obstacles_pos[nearest_obstacle_index]
        delta = nudged_point[:2] - nearest_obstacle[:2]
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm < 1e-6:
            fallback = lateral[:2]
            fallback_norm = float(np.linalg.norm(fallback))
            direction_xy = (
                fallback / fallback_norm if fallback_norm > 1e-6 else np.array([1.0, 0.0])
            )
        else:
            direction_xy = delta / delta_norm
        nudged_point[:2] = nearest_obstacle[:2] + direction_xy * (r_obs + 0.05)
    return nudged_point


def _insert_obstacle_midpoints(
    waypoints: NDArray[np.floating], obstacles_pos: NDArray[np.floating], r_obs: float
) -> NDArray[np.floating]:
    if len(obstacles_pos) == 0:
        return waypoints
    routed_waypoints: list[NDArray[np.floating]] = [waypoints[0]]
    for segment_index in range(len(waypoints) - 1):
        segment_start = waypoints[segment_index]
        segment_end = waypoints[segment_index + 1]
        segment_xy = (segment_end - segment_start)[:2]
        segment_norm_sq = float(np.dot(segment_xy, segment_xy))
        if segment_norm_sq < 1e-9:
            routed_waypoints.append(segment_end)
            continue
        worst_distance = r_obs
        worst_ratio: float | None = None
        worst_obstacle: NDArray[np.floating] | None = None
        for obstacle in obstacles_pos:
            obstacle_from_start = obstacle[:2] - segment_start[:2]
            ratio = float(
                np.clip(np.dot(obstacle_from_start, segment_xy) / segment_norm_sq, 0.0, 1.0)
            )
            closest_xy = segment_start[:2] + ratio * segment_xy
            distance = float(np.linalg.norm(obstacle[:2] - closest_xy))
            if distance < worst_distance:
                worst_distance, worst_ratio, worst_obstacle = distance, ratio, obstacle
        if worst_obstacle is not None and worst_ratio is not None:
            closest_xy = segment_start[:2] + worst_ratio * segment_xy
            push_delta = closest_xy - worst_obstacle[:2]
            push_norm = float(np.linalg.norm(push_delta))
            if push_norm < 1e-6:
                perpendicular = np.array([-segment_xy[1], segment_xy[0]])
                perpendicular_norm = float(np.linalg.norm(perpendicular))
                push_direction = (
                    perpendicular / perpendicular_norm
                    if perpendicular_norm > 1e-6
                    else np.array([1.0, 0.0])
                )
            else:
                push_direction = push_delta / push_norm
            midpoint_z = float(segment_start[2] + worst_ratio * (segment_end[2] - segment_start[2]))
            midpoint = np.array(
                [
                    worst_obstacle[0] + push_direction[0] * (r_obs + 0.08),
                    worst_obstacle[1] + push_direction[1] * (r_obs + 0.08),
                    midpoint_z,
                ]
            )
            routed_waypoints.append(midpoint)
        routed_waypoints.append(segment_end)
    return np.asarray(routed_waypoints)


def _build_spline(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    cfg: PlannerConfig,
    gates_pos: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], CubicSpline]:
    segment_vectors = np.diff(waypoints, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    # Per-segment cruise speed: peri-gate segments (either endpoint within
    # 0.55 m of a gate center) use the conservative v_cruise; inter-gate
    # segments run at v_cruise_inter for faster transitions.
    inter_gate_speed = cfg.v_cruise_inter if cfg.v_cruise_inter > 0 else cfg.v_cruise
    segment_times = np.empty(len(segment_lengths))
    for segment_index in range(len(segment_lengths)):
        peri_gate_index = -1
        if gates_pos is not None:
            for gate_index, gate_position in enumerate(gates_pos):
                if float(np.linalg.norm(waypoints[segment_index, :2] - gate_position[:2])) < 0.55:
                    peri_gate_index = gate_index
                    break
                if (
                    float(np.linalg.norm(waypoints[segment_index + 1, :2] - gate_position[:2]))
                    < 0.55
                ):
                    peri_gate_index = gate_index
                    break
        if peri_gate_index >= 0:
            segment_speed = cfg.v_peri_for(peri_gate_index)
        else:
            segment_speed = inter_gate_speed
        segment_times[segment_index] = max(
            segment_lengths[segment_index] / segment_speed, cfg.t_min_seg
        )
        # Cold-start segments (i==0 with zero start_vel) need extra runway:
        # spline accel from 0 → v_inter over a short t_min_seg exceeds the
        # drone's horizontal authority. Floor the first segment higher.
        if float(np.linalg.norm(start_vel)) < 0.3 and segment_index < 2:
            segment_times[segment_index] = max(segment_times[segment_index], 0.22)
    if len(obstacles_pos) > 0:
        slow_radius = 0.32
        for segment_index in range(len(waypoints) - 1):
            segment_start_xy = waypoints[segment_index, :2]
            segment_end_xy = waypoints[segment_index + 1, :2]
            segment_xy = segment_end_xy - segment_start_xy
            segment_norm_sq = float(np.dot(segment_xy, segment_xy))
            if segment_norm_sq < 1e-9:
                continue
            min_obstacle_distance = np.inf
            for obstacle in obstacles_pos:
                ratio = float(
                    np.clip(
                        np.dot(obstacle[:2] - segment_start_xy, segment_xy) / segment_norm_sq,
                        0.0,
                        1.0,
                    )
                )
                closest_xy = segment_start_xy + ratio * segment_xy
                distance = float(np.linalg.norm(obstacle[:2] - closest_xy))
                if distance < min_obstacle_distance:
                    min_obstacle_distance = distance
            if min_obstacle_distance < slow_radius:
                stretch = 1.0 + 0.6 * (slow_radius - min_obstacle_distance) / slow_radius
                segment_times[segment_index] *= stretch
    if cfg.max_vel > 0 and cfg.max_accel > 0:
        if cfg.use_slsqp:
            segment_times = _slsqp_time_optimal(waypoints, start_vel, segment_times, cfg)
        else:
            segment_times = _time_optimal_refine(waypoints, start_vel, segment_times, cfg)
    knot_times = np.concatenate([[0.0], np.cumsum(segment_times)])
    if cfg.use_quintic and len(waypoints) >= 6:
        # Quintic spline with clamped vel + zero-accel BCs at both ends.
        # make_interp_spline(k=5) needs 4 extra conditions (2 at each end).
        start_velocity = np.asarray(start_vel, dtype=np.float64)
        bc_type = ([(1, start_velocity), (2, np.zeros(3))], [(1, np.zeros(3)), (2, np.zeros(3))])
        try:
            spline = make_interp_spline(knot_times, waypoints, k=5, bc_type=bc_type)
            return knot_times, spline
        except Exception:
            # Fall through to cubic on degenerate knot sequences.
            pass
    bc = ((1, np.asarray(start_vel, dtype=np.float64)), (1, np.zeros(3)))
    spline = CubicSpline(knot_times, waypoints, bc_type=bc)
    return knot_times, spline


def _slsqp_time_optimal(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    seg_t0: NDArray[np.floating],
    cfg: PlannerConfig,
) -> NDArray[np.floating]:
    """Minimize total time subject to peak vel/accel caps via scipy SLSQP.

    Variables: segment times (one per segment). Objective: sum(seg_t).
    Constraints: seg_t >= t_min_seg (via bounds), peak vel <= max_vel,
    peak accel <= max_accel (sampled at 12 points per segment). Uses
    finite-difference gradients; fine for a small number of segments.
    """
    segment_count = len(seg_t0)
    bc = ((1, np.asarray(start_vel, dtype=np.float64)), (1, np.zeros(3)))

    def _peaks(segment_times: np.ndarray) -> tuple[float, float]:
        knot_times = np.concatenate([[0.0], np.cumsum(np.maximum(segment_times, 1e-3))])
        spline = CubicSpline(knot_times, waypoints, bc_type=bc)
        sample_times = np.linspace(0.0, float(knot_times[-1]), 12 * segment_count)
        peak_velocity = float(np.max(np.linalg.norm(spline.derivative(1)(sample_times), axis=1)))
        peak_acceleration = float(
            np.max(np.linalg.norm(spline.derivative(2)(sample_times), axis=1))
        )
        return peak_velocity, peak_acceleration

    def _objective(segment_times: np.ndarray) -> float:
        return float(np.sum(segment_times))

    def _velocity_constraint(segment_times: np.ndarray) -> float:
        return cfg.max_vel - _peaks(segment_times)[0]

    def _acceleration_constraint(segment_times: np.ndarray) -> float:
        return cfg.max_accel - _peaks(segment_times)[1]

    bounds = [(cfg.t_min_seg, None)] * segment_count
    constraints = [
        {"type": "ineq", "fun": _velocity_constraint},
        {"type": "ineq", "fun": _acceleration_constraint},
    ]
    try:
        result = minimize(
            _objective,
            seg_t0.copy(),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 25, "ftol": 1e-4},
        )
        if (
            result.success
            and _peaks(result.x)[0] <= cfg.max_vel * 1.05
            and _peaks(result.x)[1] <= cfg.max_accel * 1.1
        ):
            return np.asarray(np.maximum(result.x, cfg.t_min_seg))
    except Exception:
        pass
    # Fall back to the heuristic refiner on any failure.
    return _time_optimal_refine(waypoints, start_vel, seg_t0, cfg)


def _time_optimal_refine(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    segment_times: NDArray[np.floating],
    cfg: PlannerConfig,
) -> NDArray[np.floating]:
    segment_times = np.asarray(segment_times, dtype=np.float64).copy()
    bc = ((1, np.asarray(start_vel, dtype=np.float64)), (1, np.zeros(3)))
    for _ in range(8):
        knot_times = np.concatenate([[0.0], np.cumsum(segment_times)])
        spline = CubicSpline(knot_times, waypoints, bc_type=bc)
        velocity_spline = spline.derivative(1)
        acceleration_spline = spline.derivative(2)
        for segment_index in range(len(segment_times)):
            sample_times = np.linspace(knot_times[segment_index], knot_times[segment_index + 1], 24)
            peak_velocity = float(np.max(np.linalg.norm(velocity_spline(sample_times), axis=1)))
            peak_acceleration = float(
                np.max(np.linalg.norm(acceleration_spline(sample_times), axis=1))
            )
            utilization = max(
                peak_velocity / cfg.max_vel, np.sqrt(peak_acceleration / cfg.max_accel)
            )
            # Scale toward utilization = 0.75 (leave 25% margin for tracking lag).
            target = 0.75
            scale = float(np.clip(utilization / target, 0.85, 1.15))
            new_segment_time = segment_times[segment_index] * scale
            if cfg.t_min_seg > 0:
                new_segment_time = max(new_segment_time, cfg.t_min_seg)
            segment_times[segment_index] = new_segment_time
    return segment_times
