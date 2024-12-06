"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import pybullet as p
from scipy.interpolate import CubicHermiteSpline

from lsy_drone_racing.control import BaseController

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryController(BaseController):
    """Controller that follows a pre-defined trajectory."""

    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], initial_info: dict):
        """Initialization of the controller.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """
        super().__init__(initial_obs, initial_info)
        waypoints = np.array(
            [
                [2.9, 1.0, 0.05],
                [1.0, 1.0, 0.6],
                [-1.0, 1.0, 1.11],
                [-5.0, 1.0, 0.05], #[-7.0, 1.0, 1.11],
            ]
        )
        waypoints_v = np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.5],
                [-0.2, 0.0, 0.0], # [-4.0, 0.0, -4.0],
                [0.0, 0.0, 0.0],
            ]
        )
        self.t_total = 8 # time to complete the track
        t = np.linspace(0, self.t_total, len(waypoints))
        # self.trajectory = CubicSpline(t, waypoints)
        self.trajectory = CubicHermiteSpline(t, waypoints, waypoints_v)
        self._tick = 0
        self._freq = initial_info["env_freq"]

        # Generate points along the spline for visualization
        t_vis = np.linspace(0, self.t_total, 100)
        spline_points = self.trajectory(t_vis)
        try:
            # Plot the spline as a line in PyBullet
            for i in range(len(spline_points) - 1):
                p.addUserDebugLine(
                    spline_points[i],
                    spline_points[i + 1],
                    lineColorRGB=[1, 0, 0],  # Red color
                    lineWidth=2,
                    lifeTime=0,  # 0 means the line persists indefinitely
                    physicsClientId=0,
                )
        except p.error:
            ...  # Ignore errors if PyBullet is not available

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        target_pos = self.trajectory(min(self._tick / self._freq, self.t_total))
        target_vel = self.trajectory.derivative()
        target_vel = target_vel(min(self._tick / self._freq, self.t_total))
        target_acc = self.trajectory.derivative()
        target_acc = target_acc(min(self._tick / self._freq, self.t_total))
        return np.concatenate((target_pos, np.zeros(10)))
        # return np.concatenate((target_pos, target_vel, target_acc, np.zeros(4)))

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        """Increment the time step counter."""
        self._tick += 1

    def episode_reset(self):
        """Reset the time step counter."""
        self._tick = 0
