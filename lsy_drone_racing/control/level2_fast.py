"""Fast trajectory controller for Level 2 with Graphical Debugging.

Uses a cubic spline with dynamic time allocation (Progress Variable).
- Gate Threading: Brackets gates with 'pre' and 'post' waypoints.
- Obstacle Repulsion: Radially pushes waypoints away from obstacles.
- Pruning: Strips out overlapping waypoints to prevent "W" curve overshoots.
- Dynamic Tracking: Slows down the reference point based on inertia, speed, and tracking error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

_REPLAN_THRESHOLD = 0.05
_GATE_MARGIN = 0.160
_OBSTACLE_MARGIN = 0.250

# Nominal track layout
_NOMINAL_GATE_POS = np.array(
    [[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2]], dtype=np.float64
)
_NOMINAL_GATE_YAW = np.array([-0.78, 2.35, 3.14, 0.0], dtype=np.float64)

_NOMINAL_OBSTACLE_POS = np.array(
    [[0.0, 0.75, 1.55], [1.0, 0.25, 1.55], [-1.5, -0.25, 1.55], [-0.5, -0.75, 1.55]],
    dtype=np.float64,
)


class StateController(Controller):
    """Fast state controller with dynamic gate threading, obstacle repulsion, and visual debug."""

    MAX_VELOCITY = 100
    MAX_ACCELERATION = 100

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._t_total = 6

        self._waypoints_list = []
        self._gate_indices = {}

        self._waypoints_list.append([-1.5, 0.75, 0.05])  # Start
        self._waypoints_list.append([-1.0, 0.55, 0.4])  # Intermediate

        self._add_gate_waypoints(gate_id=0)
        self._add_gate_waypoints(gate_id=1, intermediate_point=[1.3, -0.15, 0.9])
        self._add_gate_waypoints(gate_id=2, intermediate_point=[-0.5, -0.05, 0.7])
        self._waypoints_list.append([-1.2, -0.2, 1.2])  # Intermediate

        self._add_gate_waypoints(gate_id=3, intermediate_point=[-0.6, -0.2, 1.2])
        self._waypoints_list.append([0.5, -0.75, 1.2])  # End

        self._base_waypoints = np.array(self._waypoints_list, dtype=np.float64)
        self._waypoints = self._base_waypoints.copy()

        self._planned_gates_pos = np.array(
            obs.get("gates_pos", _NOMINAL_GATE_POS), dtype=np.float64
        )
        self._planned_obstacles_pos = np.array(
            obs.get("obstacles_pos", _NOMINAL_OBSTACLE_POS), dtype=np.float64
        )
        self._replanned_gates: set[int] = set()

        self._dt = 1.0 / self._freq
        self._z_error_integral = 0.0
        self._ki_z = 0.8

        self._current_pos = np.zeros(3)
        self._path_history = []

        self._des_pos_spline: CubicSpline | None = None
        self._des_vel_spline: CubicSpline | None = None
        self._des_acc_spline: CubicSpline | None = None

        self._t_track = 0.0
        self._finished = False

        self._build_spline()

    def _add_gate_waypoints(self, gate_id: int, intermediate_point: list[float] | None = None):
        if intermediate_point:
            self._waypoints_list.append(intermediate_point)

        pos = _NOMINAL_GATE_POS[gate_id]
        yaw = _NOMINAL_GATE_YAW[gate_id]

        normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])

        prev_wp = np.array(self._waypoints_list[-1])
        travel_vec = pos - prev_wp
        if np.dot(travel_vec, normal) < 0:
            normal = -normal

        pre_wp = pos - _GATE_MARGIN * normal
        post_wp = pos + _GATE_MARGIN * normal

        pre_idx = len(self._waypoints_list)
        self._waypoints_list.append(pre_wp.tolist())
        post_idx = len(self._waypoints_list)
        self._waypoints_list.append(post_wp.tolist())

        self._gate_indices[gate_id] = (pre_idx, post_idx)

    def _build_spline(self) -> None:
        """Build spline with collision avoidance and acceleration limits."""
        # Prune tight waypoint clusters
        wps = [self._waypoints[0].copy()]
        for i in range(1, len(self._waypoints)):
            if np.linalg.norm(self._waypoints[i] - wps[-1]) > 0.15 or i == len(self._waypoints) - 1:
                wps.append(self._waypoints[i].copy())

        # Iteratively nudge waypoints away from obstacles
        for iteration in range(4):
            wps_arr = np.array(wps)
            distances = np.linalg.norm(np.diff(wps_arr, axis=0), axis=1)
            cum_distances = np.concatenate(([0], np.cumsum(distances)))
            total_distance = cum_distances[-1]

            if total_distance == 0:
                break

            t_wps = (cum_distances / total_distance) * self._t_total
            temp_spline = CubicSpline(t_wps, wps_arr)

            t_samples = np.linspace(0, self._t_total, 200)
            spline_pts = temp_spline(t_samples)
            collision_found = False

            for obs_pos in self._planned_obstacles_pos:
                dist_xy = np.linalg.norm(spline_pts[:, :2] - obs_pos[:2], axis=1)
                min_idx = np.argmin(dist_xy)

                if dist_xy[min_idx] < _OBSTACLE_MARGIN:
                    p_coll = spline_pts[min_idx]
                    t_coll = t_samples[min_idx]

                    push_vec = p_coll[:2] - obs_pos[:2]
                    if np.linalg.norm(push_vec) < 1e-3:
                        push_vec = np.array([1.0, 0.0])
                    push_vec = push_vec / np.linalg.norm(push_vec)

                    nudged_wp = p_coll.copy()
                    nudged_wp[:2] = obs_pos[:2] + push_vec * (_OBSTACLE_MARGIN + 0.05)

                    insert_idx = np.searchsorted(t_wps, t_coll)
                    wps.insert(insert_idx, nudged_wp)

                    collision_found = True
                    break

            if not collision_found:
                break

        # Remove clustered waypoints to prevent "W" shapes
        final_wps = [wps[0]]
        for wp in wps[1:-1]:
            if np.linalg.norm(wp - final_wps[-1]) > 0.3:
                final_wps.append(wp)

        if np.linalg.norm(final_wps[-1] - wps[-1]) > 0.05:
            final_wps.append(wps[-1])

        self._active_path_wps = np.array(final_wps)

        distances = np.linalg.norm(np.diff(self._active_path_wps, axis=0), axis=1)
        cum_distances = np.concatenate(([0], np.cumsum(distances)))
        total_distance = cum_distances[-1]

        # Enforce max acceleration by extending trajectory time if needed
        max_retries = 10
        for attempt in range(max_retries):
            t_wps = (cum_distances / total_distance) * self._t_total

            self._des_pos_spline = CubicSpline(t_wps, self._active_path_wps)
            self._des_vel_spline = self._des_pos_spline.derivative(nu=1)
            self._des_acc_spline = self._des_pos_spline.derivative(nu=2)

            t_samples = np.linspace(0, self._t_total, 200)
            acc_samples = self._des_acc_spline(t_samples)
            max_acc = np.max(np.linalg.norm(acc_samples, axis=1))

            if max_acc > 3.0:
                self._t_total += 0.2
            else:
                break

    def _check_and_replan(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Replan trajectory when gates/obstacles appear."""
        needs_rebuild = False

        target_gate = int(obs["target_gate"])
        if target_gate >= 0 and target_gate not in self._replanned_gates:
            new_gate_pos = np.asarray(obs["gates_pos"][target_gate], dtype=np.float64)
            new_yaw = (
                obs["gates_rpy"][target_gate][2]
                if "gates_rpy" in obs
                else _NOMINAL_GATE_YAW[target_gate]
            )

            delta_pos = new_gate_pos - self._planned_gates_pos[target_gate]

            if np.linalg.norm(delta_pos) > _REPLAN_THRESHOLD:
                normal = np.array([np.cos(new_yaw), np.sin(new_yaw), 0.0])

                pre_idx, post_idx = self._gate_indices[target_gate]
                prev_wp = self._waypoints[pre_idx - 1]
                if np.dot(new_gate_pos - prev_wp, normal) < 0:
                    normal = -normal

                self._waypoints[pre_idx] = new_gate_pos - _GATE_MARGIN * normal
                self._waypoints[post_idx] = new_gate_pos + _GATE_MARGIN * normal

                self._planned_gates_pos[target_gate] = new_gate_pos
                self._replanned_gates.add(target_gate)
                needs_rebuild = True

        if "obstacles_pos" in obs:
            current_obs_pos = np.asarray(obs["obstacles_pos"], dtype=np.float64)
            for i in range(len(current_obs_pos)):
                delta = current_obs_pos[i] - self._planned_obstacles_pos[i]
                if np.linalg.norm(delta) > _REPLAN_THRESHOLD:
                    self._planned_obstacles_pos[i] = current_obs_pos[i]
                    needs_rebuild = True

        if needs_rebuild:
            if self._des_pos_spline is not None:
                old_des_pos = self._des_pos_spline(self._t_track)
            else:
                old_des_pos = obs["pos"]

            self._build_spline()

            if "pos" in obs:
                t_start = max(0.0, self._t_track - 1.0)
                t_end = min(self._t_total, self._t_track + 1.0)

                t_samples = np.linspace(t_start, t_end, 200)
                path_pts = self._des_pos_spline(t_samples)

                closest_idx = np.argmin(np.linalg.norm(path_pts - old_des_pos, axis=1))
                self._t_track = t_samples[closest_idx]

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute desired state with dynamic tracking and altitude compensation."""
        if "pos" in obs:
            self._path_history.append(obs["pos"].copy())
            if len(self._path_history) > 500:
                self._path_history.pop(0)

        if self._t_track >= self._t_total:
            self._finished = True

        self._check_and_replan(obs)

        des_pos = self._des_pos_spline(self._t_track)
        des_vel = self._des_vel_spline(self._t_track)
        des_acc = self._des_acc_spline(self._t_track)

        self._current_pos = obs["pos"].copy()

        # Evaluate position error and current speed
        pos_error = np.linalg.norm(self._current_pos - des_pos)
        current_speed = np.linalg.norm(obs.get("vel", np.zeros(3)))

        # Compute dynamic lookahead for upcoming acceleration
        dynamic_lookahead = np.clip(0.3 + (0.04 * current_speed), 0.3, 0.8)
        lookahead_t = min(self._t_track + dynamic_lookahead, self._t_total)
        upcoming_acc = np.linalg.norm(self._des_acc_spline(lookahead_t))

        # Boost progress on straightaways, slow on curves
        error_factor = np.clip(1.0 - (pos_error / 1.5), 0.2, 1.0)
        accel_penalty = 1.0 + (0.015 * upcoming_acc * current_speed)

        straight_boost = 1.0
        if upcoming_acc < 3.0:
            straight_boost = 1.0 + 1.5 * (1.0 - (upcoming_acc / 4.0))

        accel_factor = straight_boost / accel_penalty

        # Advance trajectory progress
        self._t_track += self._dt * error_factor * accel_factor
        self._t_track = min(self._t_track, self._t_total)

        # Compensate altitude drift with integral feedback
        z_error = des_pos[2] - self._current_pos[2]
        self._z_error_integral += z_error * self._dt
        self._z_error_integral = np.clip(self._z_error_integral, -0.2, 0.2)

        compensated_des_pos = des_pos.copy()
        compensated_des_pos[2] += self._ki_z * self._z_error_integral

        return np.concatenate(
            (compensated_des_pos, des_vel, des_acc, np.zeros(4)), dtype=np.float32
        )

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Check if trajectory is complete."""
        return self._finished

    def episode_callback(self) -> None:
        """Reset trajectory state for new episode."""
        self._t_track = 0.0
        self._finished = False
        self._waypoints = self._base_waypoints.copy()
        self._replanned_gates = set()
        self._path_history = []
        self._z_error_integral = 0.0
        self._build_spline()

    def render_callback(self, sim: Sim) -> None:
        """Visualize trajectory, obstacles, waypoints, and velocity vectors."""
        # Actual flight path (orange, downsampled)
        if len(self._path_history) > 1:
            path_array = np.array(self._path_history[::5])
            draw_line(sim, path_array, rgba=(1.0, 0.5, 0.0, 1.0))

        # Target waypoints (magenta dots)
        draw_points(sim, self._waypoints, rgba=(1.0, 0.0, 1.0, 1.0), size=0.03)

        # Obstacle danger zones (red circles + poles)
        theta = np.linspace(0, 2 * np.pi, 21)
        for obs_pos in self._planned_obstacles_pos:
            z_height = 1.0
            circle_points = []
            for th in theta:
                circle_points.append(
                    [
                        obs_pos[0] + _OBSTACLE_MARGIN * np.cos(th),
                        obs_pos[1] + _OBSTACLE_MARGIN * np.sin(th),
                        z_height,
                    ]
                )
            draw_line(sim, np.array(circle_points), rgba=(1.0, 0.0, 0.0, 0.8))

            pole_line = np.array([[obs_pos[0], obs_pos[1], 0.0], [obs_pos[0], obs_pos[1], 1.55]])
            draw_line(sim, pole_line, rgba=(1.0, 0.0, 0.0, 0.4))

        # Ideal trajectory (green line)
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

        # Current setpoint (red dot)
        setpoint = self._des_pos_spline(self._t_track).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)

        # Velocity vectors at trajectory points (blue ticks)
        time_samples = np.linspace(0, self._t_total, 15)
        pos_samples = self._des_pos_spline(time_samples)
        vel_samples = self._des_vel_spline(time_samples)

        vel_magnitude_array = np.linalg.norm(vel_samples, axis=1, keepdims=True)
        vel_magnitude_array[vel_magnitude_array == 0] = 1.0
        vel_normalized = vel_samples / vel_magnitude_array * 0.2

        for pos, vel in zip(pos_samples, vel_normalized):
            end_point = pos + vel
            draw_line(sim, np.array([pos, end_point]), rgba=(0.0, 0.5, 1.0, 0.7))
