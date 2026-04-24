"""Advanced Level 2 controller with velocity and acceleration commands.

This is an enhanced version of level2_controller.py that additionally computes
desired velocities and accelerations by taking derivatives of the planned trajectory.

This provides more information to the lower-level attitude controller, potentially
improving tracking performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class Level2AdvancedController(Controller):
    """Advanced state controller with velocity and acceleration estimation."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the advanced Level 2 controller.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq

        # Extract gate and obstacle information
        self._gates_pos = obs["gates_pos"].copy()
        self._gates_quat = obs["gates_quat"].copy()
        self._obstacles_pos = obs["obstacles_pos"].copy()

        # Planning parameters
        self._total_time = 18.0  # seconds (slightly shorter for aggressive trajectory)
        self._waypoint_offset = 0.15
        self._replan_threshold = 0.4
        self._replan_cooldown = 40

        # Trajectory tracking
        self._tick = 0
        self._finished = False
        self._current_waypoints = None
        self._trajectory_spline = None
        self._velocity_spline = None
        self._acceleration_spline = None
        self._last_replan_tick = -self._replan_cooldown

        # Yaw control
        self._use_yaw_tracking = True
        self._next_waypoint_idx = 0

        self._plan_trajectory()

    def _compute_waypoints(self) -> np.ndarray:
        """Generate waypoints through gates with obstacle avoidance."""
        waypoints = []

        # Start position
        start_pos = np.array([-1.5, 0.75, 0.05])
        waypoints.append(start_pos)

        # Add intermediate waypoint for smooth start
        if len(self._gates_pos) > 0:
            first_gate = self._gates_pos[0]
            mid_waypoint = 0.7 * start_pos + 0.3 * first_gate
            waypoints.append(mid_waypoint)

        # Process each gate
        for gate_idx in range(len(self._gates_pos)):
            gate_pos = self._gates_pos[gate_idx]
            waypoint = gate_pos.copy()
            waypoint = self._apply_obstacle_avoidance(waypoint)
            waypoints.append(waypoint)

        # Final waypoint above last gate
        if len(self._gates_pos) > 0:
            final_pos = self._gates_pos[-1].copy()
            final_pos[2] += 0.3
            final_pos = self._apply_obstacle_avoidance(final_pos)
            waypoints.append(final_pos)

        return np.array(waypoints, dtype=np.float32)

    def _apply_obstacle_avoidance(self, waypoint: np.ndarray) -> np.ndarray:
        """Apply obstacle avoidance with enhanced collision checking.

        Args:
            waypoint: The proposed waypoint position.

        Returns:
            Adjusted waypoint position.
        """
        obstacle_radius = 0.08  # Safety radius around obstacles
        interaction_distance = 0.4  # Start pushing away at this distance
        max_offset = self._waypoint_offset

        for obs_pos in self._obstacles_pos:
            delta = waypoint - obs_pos
            dist = np.linalg.norm(delta)

            if dist < interaction_distance and dist > 1e-6:
                # Push waypoint away from obstacle
                direction = delta / dist
                # Stronger push when very close
                push_strength = max_offset * max(0, 1.0 - dist / interaction_distance)
                waypoint += direction * push_strength

        return waypoint

    def _plan_trajectory(self):
        """Plan trajectory with velocity and acceleration derivatives."""
        self._current_waypoints = self._compute_waypoints()
        n_waypoints = len(self._current_waypoints)

        # Create time points with variable spacing
        # Spend more time at difficult sections (near obstacles)
        t_points = self._compute_time_allocation()

        try:
            # Create splines with natural boundary conditions
            self._trajectory_spline = CubicSpline(
                t_points, self._current_waypoints, bc_type="natural"
            )

            # Velocity is the first derivative
            # We'll compute this numerically when needed
        except Exception as e:
            print(f"Warning: Failed to create spline. Error: {e}")
            self._trajectory_spline = None

    def _compute_time_allocation(self) -> np.ndarray:
        """Allocate time to each waypoint based on difficulty.

        Segments with obstacles nearby get more time for careful navigation.

        Returns:
            Array of time points corresponding to waypoints.
        """
        waypoints = self._current_waypoints
        n_waypoints = len(waypoints)

        # Start with uniform spacing
        times = [0.0]

        for i in range(1, n_waypoints):
            prev_wp = waypoints[i - 1]
            curr_wp = waypoints[i]
            distance = np.linalg.norm(curr_wp - prev_wp)

            # Check if this segment has obstacles nearby
            min_obs_dist = float("inf")
            for obs_pos in self._obstacles_pos:
                # Distance from segment to obstacle
                t = max(
                    0,
                    min(
                        1,
                        np.dot(obs_pos - prev_wp, curr_wp - prev_wp)
                        / (distance**2 + 1e-6),
                    ),
                )
                closest_point = prev_wp + t * (curr_wp - prev_wp)
                obs_dist = np.linalg.norm(obs_pos - closest_point)
                min_obs_dist = min(min_obs_dist, obs_dist)

            # Allocate time based on distance and obstacle proximity
            base_time = 1.5 * distance  # Base time per unit distance
            if min_obs_dist < 0.4:
                # More time for close obstacles
                multiplier = 2.0 - min_obs_dist
            else:
                multiplier = 1.0

            time_for_segment = base_time * multiplier / 0.6  # Scale to ~18s total
            times.append(times[-1] + time_for_segment)

        # Normalize to total time
        if times[-1] > 0:
            times = np.array(times) * self._total_time / times[-1]

        return np.array(times)

    def _get_desired_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get desired position, velocity, and acceleration at time t.

        Args:
            t: Time in seconds along trajectory.

        Returns:
            Tuple of (position, velocity, acceleration) arrays.
        """
        if self._trajectory_spline is None:
            des_pos = self._current_waypoints[-1]
            des_vel = np.zeros(3)
            des_acc = np.zeros(3)
            return des_pos, des_vel, des_acc

        t_clamped = np.clip(t, 0, self._total_time)

        try:
            # Position
            des_pos = np.array(self._trajectory_spline(t_clamped), dtype=np.float32)

            # Velocity (first derivative of spline)
            # CubicSpline's derivative can be computed via calling with derivative kwarg
            des_vel = np.array(self._trajectory_spline(t_clamped, 1), dtype=np.float32)

            # Acceleration (second derivative)
            des_acc = np.array(self._trajectory_spline(t_clamped, 2), dtype=np.float32)

            return des_pos, des_vel, des_acc

        except Exception as e:
            print(f"Warning: Failed to compute trajectory derivatives. Error: {e}")
            return des_pos, np.zeros(3), np.zeros(3)

    def _compute_desired_yaw(self) -> float:
        """Compute desired yaw angle pointing towards next waypoint.

        Returns:
            Desired yaw angle in radians.
        """
        if self._next_waypoint_idx >= len(self._current_waypoints) - 1:
            return 0.0

        current_waypoint = self._current_waypoints[self._next_waypoint_idx]
        next_waypoint = self._current_waypoints[self._next_waypoint_idx + 1]

        direction = next_waypoint[:2] - current_waypoint[:2]
        desired_yaw = np.arctan2(direction[1], direction[0])

        return float(desired_yaw)

    def _should_replan(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Check if trajectory should be replanned."""
        if self._tick - self._last_replan_tick < self._replan_cooldown:
            return False

        # Check if gates changed
        new_gates_pos = obs["gates_pos"]
        gate_distance = np.linalg.norm(new_gates_pos - self._gates_pos)

        if gate_distance > 0.01:
            self._gates_pos = new_gates_pos.copy()
            self._gates_quat = obs["gates_quat"].copy()
            self._obstacles_pos = obs["obstacles_pos"].copy()
            return True

        # Check if drone is far from trajectory
        current_pos = obs["pos"]
        t = min(self._tick / self._freq, self._total_time)
        desired_pos, _, _ = self._get_desired_state(t)
        distance_to_trajectory = np.linalg.norm(current_pos - desired_pos)

        if distance_to_trajectory > self._replan_threshold:
            return True

        return False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute desired state with position, velocity, and acceleration.

        Args:
            obs: Current observation.
            info: Optional information.

        Returns:
            State command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate].
        """
        # Replan if necessary
        if self._should_replan(obs):
            self._plan_trajectory()
            self._last_replan_tick = self._tick

        # Get time in trajectory
        t = min(self._tick / self._freq, self._total_time)

        # Check if finished
        if t >= self._total_time:
            self._finished = True

        # Get desired state
        des_pos, des_vel, des_acc = self._get_desired_state(t)

        # Compute desired yaw
        des_yaw = self._compute_desired_yaw() if self._use_yaw_tracking else 0.0

        # Assemble action: [pos(3), vel(3), acc(3), yaw(1), ang_vel_rates(3)]
        action = np.concatenate(
            (des_pos, des_vel, des_acc, [des_yaw], [0.0, 0.0, 0.0]), dtype=np.float32
        )

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
        """Update controller state."""
        self._tick += 1

        # Update next waypoint index based on current target gate
        target_gate = obs.get("target_gate", -1)
        if target_gate >= 0:
            # Waypoint index: +1 for start waypoint, +1 for gate index
            self._next_waypoint_idx = min(target_gate + 2, len(self._current_waypoints) - 1)

        return self._finished or terminated

    def episode_callback(self):
        """Reset after episode."""
        self._tick = 0
        self._finished = False
        self._next_waypoint_idx = 0
        self._last_replan_tick = -self._replan_cooldown

    def render_callback(self, sim: Sim):
        """Visualize trajectory, waypoints, and target."""
        from crazyflow.sim.visualize import draw_line, draw_points

        if self._current_waypoints is not None:
            # Draw waypoints in green
            draw_points(
                sim, self._current_waypoints, rgba=(0.0, 1.0, 0.0, 1.0), size=0.02
            )

        if self._trajectory_spline is not None:
            # Draw trajectory in red
            t_points = np.linspace(0, self._total_time, 150)
            trajectory_points = np.array(
                [self._trajectory_spline(t) for t in t_points], dtype=np.float32
            )
            draw_line(sim, trajectory_points, rgba=(1.0, 0.0, 0.0, 1.0))

        # Draw current target in blue
        t = min(self._tick / self._freq, self._total_time)
        des_pos, _, _ = self._get_desired_state(t)
        draw_points(
            sim, des_pos.reshape(1, -1), rgba=(0.0, 0.0, 1.0, 1.0), size=0.03
        )
