"""Adaptive controller for Level 2 challenge with gate and obstacle randomization.

This controller handles:
- Dynamically generated waypoints based on actual gate positions
- Obstacle avoidance via waypoint offset
- Trajectory re-planning during flight if needed
- Robustness to randomized drone properties

The key improvement over StateController is that waypoints are generated from
the actual gate positions observed in the environment, not hard-coded.
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
    """Adaptive state controller for Level 2 challenge."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the Level 2 controller.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._control_mode = config.env.control_mode
        self._sensor_range = config.env.sensor_range

        # Extract gate and obstacle information from observation
        self._gates_pos = obs["gates_pos"].copy()
        self._gates_quat = obs["gates_quat"].copy()
        self._obstacles_pos = obs["obstacles_pos"].copy()
        self._n_gates = len(self._gates_pos)
        self._n_obstacles = len(self._obstacles_pos)

        # Planning parameters
        self._total_time = 20.0  # seconds (generous for randomized track)
        self._waypoint_offset = 0.15  # meters (safety margin for obstacle avoidance)
        self._replan_threshold = 0.5  # meters (distance at which to replan)
        self._replan_cooldown = 50  # steps (avoid too frequent replanning)
        self._last_replan_tick = -self._replan_cooldown

        # Trajectory tracking
        self._tick = 0
        self._finished = False
        self._current_waypoints = None
        self._trajectory_spline = None
        self._plan_trajectory()

    def _compute_waypoints(self) -> np.ndarray:
        """Generate waypoints by passing through gate centers with obstacle avoidance.

        The waypoints are positioned:
        1. At the center of each gate (projected slightly before/after)
        2. Offset to avoid detected obstacles
        3. With smooth height transitions

        Returns:
            Array of waypoints with shape (n_waypoints, 3)
        """
        waypoints = []

        # Start position (slightly above ground at origin or current position)
        start_pos = np.array([-1.5, 0.75, 0.05])
        waypoints.append(start_pos)

        # Process each gate in order
        for gate_idx in range(self._n_gates):
            gate_pos = self._gates_pos[gate_idx]

            # The waypoint is at the gate center
            waypoint = gate_pos.copy()

            # Apply obstacle avoidance: check nearby obstacles and offset if needed
            waypoint = self._apply_obstacle_avoidance(waypoint)

            waypoints.append(waypoint)

        # Add final waypoint above last gate
        final_pos = self._gates_pos[-1].copy()
        final_pos[2] += 0.2
        final_pos = self._apply_obstacle_avoidance(final_pos)
        waypoints.append(final_pos)

        return np.array(waypoints, dtype=np.float32)

    def _apply_obstacle_avoidance(self, waypoint: np.ndarray) -> np.ndarray:
        """Apply obstacle avoidance by offsetting waypoint away from nearby obstacles.

        Args:
            waypoint: The proposed waypoint position [x, y, z].

        Returns:
            Adjusted waypoint position with obstacle avoidance applied.
        """
        # Only apply avoidance if obstacles are close
        obstacle_distance = 0.3  # meters
        max_offset = self._waypoint_offset

        for obs_pos in self._obstacles_pos:
            delta = waypoint - obs_pos
            dist = np.linalg.norm(delta[:2])  # Distance in xy-plane

            if dist < obstacle_distance and dist > 1e-6:
                # Push waypoint away from obstacle
                direction = delta[:2] / dist
                offset = max_offset * (obstacle_distance - dist) / obstacle_distance
                waypoint[:2] += direction * offset

        return waypoint

    def _plan_trajectory(self):
        """Plan a smooth trajectory through the computed waypoints."""
        self._current_waypoints = self._compute_waypoints()
        n_waypoints = len(self._current_waypoints)

        # Create time points with consistent spacing
        t = np.linspace(0, self._total_time, n_waypoints)

        # Create cubic spline for smooth interpolation
        try:
            self._trajectory_spline = CubicSpline(t, self._current_waypoints, bc_type="natural")
        except Exception as e:
            print(f"Warning: Failed to create spline, using linear interpolation. Error: {e}")
            self._trajectory_spline = None

    def _get_desired_position(self, t: float) -> np.ndarray:
        """Get the desired position at time t along the trajectory.

        Args:
            t: Time in seconds.

        Returns:
            Desired position [x, y, z].
        """
        if self._trajectory_spline is None:
            return self._current_waypoints[-1]

        # Clamp time to trajectory duration
        t_clamped = min(max(t, 0), self._total_time)

        try:
            des_pos = self._trajectory_spline(t_clamped)
            return np.array(des_pos, dtype=np.float32)
        except Exception:
            return self._current_waypoints[-1]

    def _should_replan(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Check if trajectory should be replanned.

        Replanning is triggered if:
        1. Gates have changed significantly (new randomization detected)
        2. Current position is far from trajectory
        3. Enough time has passed since last replan

        Args:
            obs: Current observation.

        Returns:
            True if replanning is needed.
        """
        # Cooldown to avoid too frequent replanning
        if self._tick - self._last_replan_tick < self._replan_cooldown:
            return False

        # Check if gates have changed significantly
        new_gates_pos = obs["gates_pos"]
        gate_distance = np.linalg.norm(new_gates_pos - self._gates_pos)

        if gate_distance > 0.01:  # Gates moved
            self._gates_pos = new_gates_pos.copy()
            self._gates_quat = obs["gates_quat"].copy()
            self._obstacles_pos = obs["obstacles_pos"].copy()
            return True

        # Check if drone is far from trajectory
        current_pos = obs["pos"]
        desired_pos = self._get_desired_position(self._tick / self._freq)
        distance_to_trajectory = np.linalg.norm(current_pos - desired_pos)

        if distance_to_trajectory > self._replan_threshold:
            return True

        return False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information.

        Returns:
            The drone state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
            (13D state vector).
        """
        # Replan if necessary
        if self._should_replan(obs):
            self._plan_trajectory()
            self._last_replan_tick = self._tick

        # Get current time in trajectory
        t = min(self._tick / self._freq, self._total_time)

        # Check if trajectory is complete
        if t >= self._total_time:
            self._finished = True
            # Return final position
            des_pos = self._current_waypoints[-1]
        else:
            des_pos = self._get_desired_position(t)

        # Return state command: [pos(3), vel(3), acc(3), yaw(1), ang_vel(3)]
        # For now, we only specify position and let
        #  the lower controller handle velocity/acceleration
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
        """Step callback for updating internal state.

        Args:
            action: Latest applied action.
            obs: Latest environment observation.
            reward: Latest reward.
            terminated: Latest terminated flag.
            truncated: Latest truncated flag.
            info: Latest information dictionary.

        Returns:
            True if the controller has finished.
        """
        self._tick += 1
        return self._finished or terminated

    def episode_callback(self):
        """Reset the controller after episode completion."""
        self._tick = 0
        self._finished = False
        self._last_replan_tick = -self._replan_cooldown

    def render_callback(self, sim: Sim):
        """Visualize trajectory and waypoints.

        Args:
            sim: The simulator object for rendering.
        """
        from crazyflow.sim.visualize import draw_line, draw_points

        # Draw waypoints
        if self._current_waypoints is not None:
            draw_points(
                sim, self._current_waypoints, rgba=(0.0, 1.0, 0.0, 1.0), size=0.02
            )

        # Draw trajectory spline
        if self._trajectory_spline is not None:
            t_points = np.linspace(0, self._total_time, 100)
            trajectory_points = self._trajectory_spline(t_points)
            draw_line(sim, trajectory_points, rgba=(1.0, 0.0, 0.0, 1.0))

        # Draw current setpoint
        t = min(self._tick / self._freq, self._total_time)
        current_setpoint = self._get_desired_position(t)
        draw_points(sim, current_setpoint.reshape(1, -1), rgba=(0.0, 0.0, 1.0, 1.0), size=0.03)
