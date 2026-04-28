"""Hand-tuned racing line for level2.

A sequence of 21 waypoints designed against the level2 nominal gate/obstacle
layout. Each waypoint is tagged with the gate it "belongs to" (-1 = start
segment, not tied to any gate). At runtime we compute the per-gate observed
position and yaw deltas vs nominal and apply an affine warp (translation +
in-plane rotation about the gate center) to every waypoint tagged with that
gate. Obstacle randomization is absorbed by a final obstacle-nudge pass.

The line deliberately swings **north** of obstacle 3 (at nominal xy
``(-0.5, -0.75)``) when threading from gate 2 to gate 3, so the drone avoids
the dead-on approach-corridor geometry that kills naïve gate-aware planners.

Gates (nominal level2):
    g0  (0.50,  0.25, 0.70)  yaw = -0.78
    g1  (1.05,  0.75, 1.20)  yaw =  2.35
    g2  (-1.00, -0.25, 0.70) yaw =  3.14
    g3  (0.00, -0.75, 1.20)  yaw =  0.00

Obstacles (nominal level2, top z):
    o0  (0.00,  0.75, 1.55)
    o1  (1.00,  0.25, 1.55)
    o2  (-1.50, -0.25, 1.55)
    o3  (-0.50, -0.75, 1.55)   <-- the killer; our line hooks north of it
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.utils.planner import Plan, _insert_obstacle_midpoints, _nudge_lateral

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Nominal level2 gate poses (xyz + yaw) — used only to compute warp deltas.
NOMINAL_GATES_POS = np.array(
    [[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2]]
)
NOMINAL_GATES_YAW = np.array([-0.78, 2.35, 3.14, 0.0])

# Hand-tuned waypoints with gate-ownership tags (-1 = no gate / transition).
#
# Design principles for this v2 line:
#   * Takeoff: 3-waypoint gentle climb (avoid MPC infeasibility from standstill).
#   * Gate entries: one approach waypoint on the gate x-axis at d_pre = 0.35 m
#     so the drone enters near-perpendicular without fighting lateral nudge.
#   * Gate 0 → 1: swing EAST of obstacle 1 (at x=1.0) then climb NW into gate 1.
#   * Gate 1 → 2: descend gradually through open airspace (no obstacles).
#   * Gate 2 → 3: critical transition. Swing NORTH of obstacle 3 (at y=-0.75)
#     with y>=-0.40 until well past the obstacle's x, keeping distance >= 0.37 m
#     (tolerates worst-case ±0.15 m obstacle randomization while staying
#     >= 0.22 m nominal r_obs).
#   * Each consecutive triplet of waypoints makes an obtuse angle (turn ≤ 45°)
#     so cubic-spline curvature stays low enough for MPC to track at ~2 m/s.
LEVEL2_WAYPOINTS = np.array(
    [
        [-1.50, 0.75, 0.01],  # 0   start
        [-1.48, 0.74, 0.08],  # 1   micro-takeoff (gentle z-ramp prevents cubic undershoot)
        [-1.42, 0.72, 0.22],  # 2   climb
        [-1.25, 0.68, 0.40],  # 3   climb + east
        [-0.95, 0.62, 0.55],  # 4   continue east-south
        [-0.55, 0.55, 0.65],  # 5   approaching g0 altitude
        [-0.15, 0.52, 0.70],  # 6   pre-approach
        [0.15, 0.55, 0.70],  # 7   g0 approach
        [0.50, 0.25, 0.70],  # 8   g0 center
        [0.72, 0.03, 0.75],  # 9   g0 exit + slight climb
        [1.15, -0.05, 0.95],  # 10  swing EAST of obstacle 1
        [1.40, 0.20, 1.12],  # 11  climb to gate 1 altitude
        [1.40, 0.40, 1.20],  # 12  g1 approach
        [1.05, 0.75, 1.20],  # 13  g1 center
        [0.84, 0.96, 1.20],  # 14  g1 exit
        [0.55, 0.70, 1.20],  # 15  head W at gate-1 altitude
        [0.05, 0.45, 1.20],  # 16  continue W
        [-0.40, 0.20, 1.05],  # 17  descend SW
        [-0.75, -0.10, 0.85],  # 18  continue SW toward g2
        [-0.55, -0.25, 0.70],  # 19  g2 approach
        [-1.00, -0.25, 0.70],  # 20  g2 center
        [-1.18, -0.30, 0.90],  # 21  g2 exit + start climb
        [-0.80, -0.35, 1.05],  # 22  begin NORTH arc around obstacle 3
        [-0.40, -0.35, 1.18],  # 23  apex north of obstacle 3
        [-0.30, -0.55, 1.20],  # 24  heading SE, past obstacle
        [-0.10, -0.70, 1.20],  # 25  g3 approach
        [0.00, -0.75, 1.20],  # 26  g3 center
        [0.30, -0.75, 1.20],  # 27  g3 exit
        [0.55, -0.75, 1.20],  # 28  stop
    ]
)

# Gate tags. Transitions (-1) stay in world frame; gate-tagged waypoints warp
# with the observed position/yaw of their gate.
WAYPOINT_GATE_TAG = np.array(
    [
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        0,
        0,
        -1,
        -1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        2,
        2,
        2,
        -1,
        -1,
        3,
        3,
        3,
        3,
        3,
    ]
)

# First segment's timing override — gentle takeoff from standstill.
TAKEOFF_SEG_MIN_T = 0.9


@dataclass(frozen=True)
class RacingLineConfig:
    v_cruise: float = 2.5
    t_min_seg: float = 0.15
    max_accel: float = 6.0  # m/s² — cap peak spline acceleration
    max_vel: float = 3.5  # m/s — cap peak spline speed
    r_obs: float = 0.22


def _warp_waypoint(
    waypoint: NDArray[np.floating],
    gate_tag: int,
    nominal_gates_pos: NDArray[np.floating],
    nominal_gates_yaw: NDArray[np.floating],
    observed_gates_pos: NDArray[np.floating],
    observed_gates_yaw: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Warp a gate-tagged waypoint by the observed vs nominal delta for that gate.

    For tag < 0 the waypoint is returned unchanged. Otherwise we translate
    the waypoint relative to the nominal gate center, rotate it in xy by the
    observed yaw delta, then translate to the observed gate center. Z delta
    is applied additively. This preserves the waypoint's role (approach /
    center / exit) under track randomization.
    """
    if gate_tag < 0:
        return waypoint.copy()
    nominal_gate_position = nominal_gates_pos[gate_tag]
    observed_gate_position = observed_gates_pos[gate_tag]
    yaw_delta = float(observed_gates_yaw[gate_tag] - nominal_gates_yaw[gate_tag])
    cos_yaw = float(np.cos(yaw_delta))
    sin_yaw = float(np.sin(yaw_delta))
    relative_waypoint = waypoint - nominal_gate_position
    rotated_relative_waypoint = np.array(
        [
            cos_yaw * relative_waypoint[0] - sin_yaw * relative_waypoint[1],
            sin_yaw * relative_waypoint[0] + cos_yaw * relative_waypoint[1],
            relative_waypoint[2],
        ]
    )
    return observed_gate_position + rotated_relative_waypoint


def warp_waypoints(
    waypoints: NDArray[np.floating],
    gate_tags: NDArray[np.integer],
    observed_gates_pos: NDArray[np.floating],
    observed_gates_quat: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply per-gate affine warp to every tagged waypoint."""
    observed_yaw = R.from_quat(observed_gates_quat).as_euler("xyz")[:, 2]
    warped_waypoints = np.zeros_like(waypoints)
    for waypoint_index, (waypoint, gate_tag) in enumerate(zip(waypoints, gate_tags)):
        warped_waypoints[waypoint_index] = _warp_waypoint(
            waypoint,
            int(gate_tag),
            NOMINAL_GATES_POS,
            NOMINAL_GATES_YAW,
            observed_gates_pos,
            observed_yaw,
        )
    return warped_waypoints


def _first_reachable_index(
    warped: NDArray[np.floating], gate_tags: NDArray[np.integer], target_gate: int
) -> int:
    """Return index of the first waypoint to use when heading for ``target_gate``.

    When ``target_gate`` is 0 we start from the very beginning. Otherwise we
    skip everything up to and including the last waypoint owned by the
    previous gate, but *keep* any transition waypoints (tag = -1) that follow
    — those route the drone safely toward ``target_gate``.
    """
    if target_gate <= 0:
        return 0
    last_previous_gate_index = -1
    for waypoint_index, gate_tag in enumerate(gate_tags):
        if gate_tag == target_gate - 1:
            last_previous_gate_index = waypoint_index
    if last_previous_gate_index < 0:
        return 0
    return min(last_previous_gate_index + 1, len(warped) - 1)


def _feasible_segment_times(
    waypoints: NDArray[np.floating], cfg: RacingLineConfig
) -> NDArray[np.floating]:
    """Pick per-segment times so peak spline speed/accel stays within caps.

    Uses the same PCHIP interpolator the final plan will use, and adjusts
    each segment's time independently based on its local peak velocity and
    acceleration. Segments with headroom shrink; overloaded ones stretch.
    """
    segment_vectors = np.diff(waypoints, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    segment_times = np.maximum(segment_lengths / cfg.v_cruise, cfg.t_min_seg)
    segment_times[0] = max(segment_times[0], TAKEOFF_SEG_MIN_T)
    for _ in range(8):
        knot_times = np.concatenate([[0.0], np.cumsum(segment_times)])
        spline = PchipInterpolator(knot_times, waypoints, axis=0)
        velocity_spline = spline.derivative(1)
        acceleration_spline = spline.derivative(2)
        any_changed = False
        for segment_index in range(len(segment_times)):
            sample_times = np.linspace(knot_times[segment_index], knot_times[segment_index + 1], 20)
            peak_velocity = float(np.max(np.linalg.norm(velocity_spline(sample_times), axis=1)))
            peak_acceleration = float(
                np.max(np.linalg.norm(acceleration_spline(sample_times), axis=1))
            )
            velocity_utilization = peak_velocity / cfg.max_vel
            acceleration_utilization = np.sqrt(peak_acceleration / cfg.max_accel)
            utilization = max(velocity_utilization, acceleration_utilization)
            target = 0.85  # leave ~15% margin for MPC tracking
            scale = float(np.clip(utilization / target, 0.85, 1.25))
            new_segment_time = max(segment_times[segment_index] * scale, cfg.t_min_seg)
            if segment_index == 0:
                new_segment_time = max(new_segment_time, TAKEOFF_SEG_MIN_T)
            if (
                abs(new_segment_time - segment_times[segment_index])
                / max(segment_times[segment_index], 1e-6)
                > 0.02
            ):
                any_changed = True
            segment_times[segment_index] = new_segment_time
        if not any_changed:
            break
    return segment_times


def build_racing_line_plan(
    start_pos: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    gates_quat: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    target_gate: int,
    cfg: RacingLineConfig = RacingLineConfig(),
) -> Plan:
    """Return a :class:`Plan` built from the hand-tuned, per-gate warped racing line.

    Only the tail of the line from the current target gate onward is used;
    the current drone state replaces whatever waypoint came before.
    """
    start_pos = np.asarray(start_pos, dtype=np.float64)
    start_vel = np.asarray(start_vel, dtype=np.float64)
    gates_pos = np.asarray(gates_pos, dtype=np.float64)
    gates_quat = np.asarray(gates_quat, dtype=np.float64)
    obstacles_pos = np.asarray(obstacles_pos, dtype=np.float64)

    warped_waypoints = warp_waypoints(LEVEL2_WAYPOINTS, WAYPOINT_GATE_TAG, gates_pos, gates_quat)
    first_reachable_index = _first_reachable_index(
        warped_waypoints, WAYPOINT_GATE_TAG, max(target_gate, 0)
    )
    # The first nominal waypoint (WP0 at target_gate=0, or the wp right after
    # the previous gate) becomes redundant once we anchor at ``start_pos``.
    # Skip it only if ``start_pos`` is close enough to avoid a degenerate
    # near-zero-length first segment in the cubic spline.
    skip_first_waypoint = (
        float(np.linalg.norm(warped_waypoints[first_reachable_index] - start_pos)) < 0.15
    )
    waypoint_tail = warped_waypoints[first_reachable_index + (1 if skip_first_waypoint else 0) :]
    plan_waypoints = np.vstack([start_pos[None, :], waypoint_tail])

    # Nudge any waypoints that sit within r_obs of an observed obstacle
    # (randomization can push obstacles into the nominal line). Preserve
    # the approach/exit direction by using each waypoint's direction-to-next
    # as the "along" axis and nudging perpendicular to it.
    if len(plan_waypoints) >= 2:
        for waypoint_index in range(1, len(plan_waypoints) - 1):
            tangent = plan_waypoints[waypoint_index + 1] - plan_waypoints[waypoint_index - 1]
            tangent_norm_xy = float(np.linalg.norm(tangent[:2]))
            if tangent_norm_xy < 1e-6:
                continue
            lateral_axis = np.array(
                [-tangent[1] / tangent_norm_xy, tangent[0] / tangent_norm_xy, 0.0]
            )
            plan_waypoints[waypoint_index] = _nudge_lateral(
                plan_waypoints[waypoint_index], lateral_axis, obstacles_pos, cfg.r_obs
            )
    # Add midpoint detours for any segment that still grazes an obstacle
    plan_waypoints = _insert_obstacle_midpoints(plan_waypoints, obstacles_pos, cfg.r_obs)

    segment_times = _feasible_segment_times(plan_waypoints, cfg)
    knot_times = np.concatenate([[0.0], np.cumsum(segment_times)])
    # PchipInterpolator preserves per-segment monotonicity of each coordinate,
    # preventing the cubic spline from dipping below z=0 during takeoff or
    # undershooting lateral waypoints. It's only C1 (not C2) but that's fine
    # for MPC tracking.
    if float(np.linalg.norm(start_vel)) < 0.05:
        spline = PchipInterpolator(knot_times, plan_waypoints, axis=0)
    else:
        # When we have a non-trivial start velocity (mid-flight replan), fall
        # back to CubicSpline with a clamped start BC to preserve C1 continuity.
        spline = CubicSpline(knot_times, plan_waypoints, bc_type=((1, start_vel), (1, np.zeros(3))))
    return Plan(
        waypoints=plan_waypoints,
        t_knots=knot_times,
        pos_spline=spline,
        vel_spline=spline.derivative(1),
        t_total=float(knot_times[-1]),
    )
