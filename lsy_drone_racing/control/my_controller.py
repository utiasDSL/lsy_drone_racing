"""Working PID-based attitude controller for drone racing.

This controller uses a cubic spline trajectory and PID control to track
a pre-defined set of waypoints. It outputs attitude commands [roll, pitch, yaw, thrust].
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MyController(Controller):
    """Working attitude controller with PID position control."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller with PID gains and trajectory."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # Drone mass for gravity compensation
        self.drone_mass = 0.032
        # config.sim.drone_model.get("mass", 0.032)

        # PID gains
        self.kp = np.array([0.5, 0.5, 1.5])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.25, 0.25, 0.5])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81

        # Define waypoints from the initial position
        start_pos = obs["pos"]
        waypoints = np.array([
            start_pos,
            [start_pos[0] + 0.5, start_pos[1] + 0.3, start_pos[2] + 0.2],
            [start_pos[0] + 1.0, start_pos[1] + 0.5, start_pos[2] + 0.4],
            [start_pos[0] + 1.5, start_pos[1] + 0.3, start_pos[2] + 0.6],
            [start_pos[0] + 2.0, start_pos[1], start_pos[2] + 0.8],
            [start_pos[0] + 2.5, start_pos[1] - 0.3, start_pos[2] + 1.0],
            [start_pos[0] + 2.0, start_pos[1] - 0.5, start_pos[2] + 0.8],
            [start_pos[0] + 1.5, start_pos[1] - 0.3, start_pos[2] + 0.6],
            [start_pos[0] + 1.0, start_pos[1], start_pos[2] + 0.4],
            [start_pos[0] + 0.5, start_pos[1] + 0.2, start_pos[2] + 0.2],
            start_pos,  # Return to start
        ])

        self._t_total = 20  # seconds for complete trajectory
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

        self._tick = 0
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute attitude command using PID control.

        Args:
            obs: Current observation with 'pos', 'vel', 'quat' keys.
            info: Optional additional information.

        Returns:
            Attitude command [roll, pitch, yaw, thrust] as numpy array.
        """
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:
            self._finished = True

        # Get desired position and velocity from spline
        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_yaw = 0.0

        # Calculate position and velocity errors
        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        # Update integral error with anti-windup
        self.i_error += pos_error * (1 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust vector
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust[2] += self.drone_mass * self.g  # Gravity compensation

        # Get current drone orientation
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]

        # Compute desired collective thrust
        thrust_desired = max(target_thrust.dot(z_axis), 0.1)  # Minimum thrust to hover

        # Compute desired orientation from thrust direction
        z_axis_desired = target_thrust / (np.linalg.norm(target_thrust) + 1e-8)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired) + 1e-8
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        action = np.array([
            obs["pos"][0], obs["pos"][1], obs["pos"][2],  # position
            0.0, 0.0, 0.0,                                # velocity
            0.0, 0.0, thrust_desired,                     # acceleration (z only)
            euler_desired[2],                             # yaw
            0.0, 0.0, 0.0                                 # angular rates
            ], dtype=np.float32)
        
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
        """Update tick counter."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset controller state for new episode."""
        self._tick = 0
        self.i_error = np.zeros(3)
        self._finished = False