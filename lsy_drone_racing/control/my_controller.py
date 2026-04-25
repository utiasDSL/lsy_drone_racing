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
        t = self._tick / self._freq
        pos = obs["pos"]
        vel = obs["vel"]

        # --- target height profile ---
        if t < 2.0:
            target_z   =  0.5 * (t / 2.0)
            target_vz  =  0.5 / 2.0          # 0.25 m/s upward
        elif t < 4.0:
            target_z   =  0.5
            target_vz  =  0.0
        elif t < 6.0:
            target_z   =  0.5 * (1 - (t - 4) / 2.0)
            target_vz  = -0.5 / 2.0          # 0.25 m/s downward
        else:
            target_z   =  0.0
            target_vz  =  0.0

        # PD on position error → desired acceleration
        kp_z, kd_z = 3.0, 1.5
        acc_z = kp_z * (target_z - pos[2]) + kd_z * (target_vz - vel[2])
        acc_z = np.clip(acc_z, -15.0, 15.0)

        # --- action ---
        action = np.array([
            pos[0], pos[1], target_z,    # desired position (x, y, z)
            0.0,    0.0,    target_vz,   # desired velocity (vx, vy, vz)  ← THIS was the bug
            0.0,    0.0,    acc_z,       # desired acceleration
            0.0,                         # yaw
            0.0,    0.0,    0.0          # angular rates
            ], dtype=np.float32)

        # debug
        if self._tick % 50 == 0:
            print(f"\n t={t:.2f}")
            print("z:", pos[2], "target_z:", target_z)
            print("vel_z:", vel[2], "acc_z:", acc_z)

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