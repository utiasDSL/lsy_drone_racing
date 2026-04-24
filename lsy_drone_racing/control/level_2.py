"""Level 2 Controller with TARGET_GATE TRACKING FIX.

KEY INSIGHT: obs['target_gate'] tells you which gate (0,1,2,3 or -1) to pass NEXT.
When it changes, REPLAN immediately!
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class Level2Controller(Controller):
    """Level 2 Controller - NOW WITH TARGET GATE TRACKING!"""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._gates_pos = obs["gates_pos"].copy()
        self._obstacles_pos = obs["obstacles_pos"].copy()

        # ★★★ TARGET GATE TRACKING (THE FIX!) ★★★
        self._current_target_gate = obs.get("target_gate", -1)

        self._total_time_per_gate = 5.0
        self._waypoint_offset = 0.15
        self._tick = 0
        self._finished = False
        self._segment_start_tick = 0
        self._trajectory_spline = None
        self._current_waypoints = None

        self._plan_to_target_gate(obs)

    def _apply_obstacle_avoidance(self, waypoint: np.ndarray) -> np.ndarray:
        """Push waypoint away from obstacles."""
        for obs_pos in self._obstacles_pos:
            delta = waypoint - obs_pos
            dist_xy = np.linalg.norm(delta[:2])
            if dist_xy < 0.3 and dist_xy > 1e-6:
                direction = delta[:2] / dist_xy
                strength = max(0, (0.3 - dist_xy) / 0.3) * 0.15
                waypoint[:2] += direction * strength
        return waypoint

    def _plan_to_target_gate(self, obs: dict) -> None:
        """Plan to the current target gate only."""
        target_gate = obs.get("target_gate", -1)
        current_pos = obs["pos"].copy()

        waypoints = [current_pos]

        if 0 <= target_gate < len(self._gates_pos):
            # Target gate
            target_pos = self._gates_pos[target_gate].copy()
            target_pos = self._apply_obstacle_avoidance(target_pos)
            waypoints.append(target_pos)

            # Next gate (look-ahead)
            if target_gate + 1 < len(self._gates_pos):
                next_pos = self._gates_pos[target_gate + 1].copy()
                next_pos = self._apply_obstacle_avoidance(next_pos)
                waypoints.append(next_pos)

        waypoints_arr = np.array(waypoints, dtype=np.float32)
        t = np.linspace(0, self._total_time_per_gate, len(waypoints_arr))

        try:
            self._trajectory_spline = CubicSpline(t, waypoints_arr, bc_type="natural")
            self._current_waypoints = waypoints_arr
        except Exception:
            self._trajectory_spline = None
            self._current_waypoints = waypoints_arr

        self._segment_start_tick = self._tick

    def _get_desired_position(self, t: float) -> np.ndarray:
        """Get position at time t."""
        if self._trajectory_spline is None:
            return self._current_waypoints[-1].copy()
        t_clamped = np.clip(t, 0, self._total_time_per_gate)
        try:
            return np.array(self._trajectory_spline(t_clamped), dtype=np.float32)
        except Exception:
            return self._current_waypoints[-1].copy()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute control."""
        target_gate = obs.get("target_gate", -1)

        # ★★★ CRITICAL: Detect gate change and REPLAN ★★★
        if target_gate != self._current_target_gate:
            self._current_target_gate = target_gate
            self._plan_to_target_gate(obs)

        if target_gate == -1:
            self._finished = True
            des_pos = obs["pos"].copy()
        else:
            t = (self._tick - self._segment_start_tick) / self._freq
            if t > self._total_time_per_gate:
                des_pos = self._current_waypoints[-1].copy()
            else:
                des_pos = self._get_desired_position(t)

        action = np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Step callback."""
        self._tick += 1
        return self._finished or terminated

    def episode_callback(self):
        """Reset."""
        self._tick = 0
        self._finished = False
        self._segment_start_tick = 0
        self._current_target_gate = -1

    def render_callback(self, sim: Sim):
        """Visualize."""
        from crazyflow.sim.visualize import draw_line, draw_points

        if self._current_waypoints is not None:
            draw_points(sim, self._current_waypoints, rgba=(0, 1, 0, 1), size=0.02)
        if self._trajectory_spline is not None:
            t_vals = np.linspace(0, self._total_time_per_gate, 50)
            trajectory = np.array([self._trajectory_spline(t) for t in t_vals])
            draw_line(sim, trajectory, rgba=(1, 0, 0, 1))
