"""Fast trajectory controller with velocity and acceleration from spline derivatives.

This controller extends the basic StateController by utilizing the derivatives of the
precalculated cubic splines to provide smooth velocity and acceleration references.
The velocity and acceleration are clamped to respect the drone's physical limits.

The drone state action format is:
    [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]

The controller computes these values from spline derivatives while respecting limits:
    - Position: determined by environment boundaries
    - Velocity: clamped to physically achievable limits (~2-3 m/s for Crazyflie)
    - Acceleration: clamped to achievable values (~5-10 m/s^2 for Crazyflie)
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


class Level1Fast(Controller):
    """Fast state controller following a pre-defined trajectory with velocity and acceleration."""

    # Drone physical limits for Crazyflie 2.1
    # These are conservative estimates based on the drone's physical capabilities
    MAX_VELOCITY = 12.5  # m/s - maximum achievable velocity
    MAX_ACCELERATION = 18.0  # m/s^2 - maximum achievable acceleration
    MAX_ANGULAR_RATE = 15.0  # rad/s - maximum angular velocity rate

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration containing
                     disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # Same waypoints as in the state and attitude controllers
        waypoints = np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )

        # 1. Lowered the total time for a much faster run!
        self._t_total = 5.0  # s

        # 2. Distance-Based Time Allocation
        # Calculate the Euclidean distance between each consecutive waypoint
        differences = np.diff(waypoints, axis=0)
        distances = np.linalg.norm(differences, axis=1)

        # Create an array of cumulative distances (start at 0)
        cum_distances = np.concatenate(([0], np.cumsum(distances)))
        total_distance = cum_distances[-1]

        # Map those cumulative distances proportionally to our total time
        t = (cum_distances / total_distance) * self._t_total

        # Create position spline and its derivatives
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative(nu=1)
        self._des_acc_spline = self._des_pos_spline.derivative(nu=2)

        self._tick = 0
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        This method evaluates the position, velocity, and acceleration splines at the current
        time step and returns them as the desired drone state. Values are clamped to respect
        the drone's physical limits.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
            as a numpy array (dtype=float32).
        """
        # Current time in trajectory
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:
            self._finished = True

        # Evaluate splines at current time
        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_acc = self._des_acc_spline(t)

        # Clamp velocity to physical limits
        vel_magnitude = np.linalg.norm(des_vel)
        if vel_magnitude > self.MAX_VELOCITY:
            des_vel = des_vel * (self.MAX_VELOCITY / vel_magnitude)

        # Clamp acceleration to physical limits
        acc_magnitude = np.linalg.norm(des_acc)
        if acc_magnitude > self.MAX_ACCELERATION:
            des_acc = des_acc * (self.MAX_ACCELERATION / acc_magnitude)

        # Construct the full 13-element state action
        # [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
        action = np.concatenate(
            (
                des_pos,  # position (3)
                des_vel,  # velocity (3)
                des_acc,  # acceleration (3)
                np.zeros(4),  # yaw, rrate, prate, yrate (4) - not used by state controller
            ),
            dtype=np.float32,
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
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint.

        This method draws:
        - Red points: current setpoint position
        - Green line: full desired trajectory
        - Blue line: velocity vectors at sample points
        """
        # Get current time and desired state
        t = min(self._tick / self._freq, self._t_total)
        des_pos = self._des_pos_spline(t)
        # des_vel = self._des_vel_spline(t)

        # Draw current setpoint
        setpoint = des_pos.reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)

        # Draw full trajectory
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

        # Draw velocity vectors at sample points for visualization
        time_samples = np.linspace(0, self._t_total, 15)
        pos_samples = self._des_pos_spline(time_samples)
        vel_samples = self._des_vel_spline(time_samples)

        # Normalize and scale velocity vectors for visibility
        vel_magnitude_array = np.linalg.norm(vel_samples, axis=1, keepdims=True)
        vel_magnitude_array[vel_magnitude_array == 0] = 1.0  # Avoid division by zero
        vel_normalized = vel_samples / vel_magnitude_array * 0.2  # Scale for visibility

        # Draw velocity vectors
        for pos, vel in zip(pos_samples, vel_normalized):
            end_point = pos + vel
            draw_line(sim, np.array([pos, end_point]), rgba=(0.0, 0.5, 1.0, 0.7))
