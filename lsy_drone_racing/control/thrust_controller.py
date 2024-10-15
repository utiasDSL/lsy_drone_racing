from __future__ import annotations  # Python 3.10 type hints

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pybullet as p
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import BaseController

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ThrustController(BaseController):
    """Example of a controller using the collective thrust interface.

    Modified from https://github.com/utiasDSL/crazyswarm-import/blob/ad2f7ea987f458a504248a1754b124ba39fc2f21/ros_ws/src/crazyswarm/scripts/position_ctl_m.py
    """

    def __init__(self, initial_obs: npt.NDArray[np.floating], initial_info: dict):
        """Initialization of the controller.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """
        super().__init__(initial_obs, initial_info)
        self.low_level_ctrl_freq = initial_info["sim.ctrl_freq"]
        self.drone_mass = initial_info["sim.drone.mass"]
        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81
        self._tick = 0

        # Same waypoints as in the trajectory controller. Determined by trial and error.
        waypoints = np.array(
            [
                [1.0, 1.0, 0.05],
                [0.55, -0.8, 0.4],
                [0.2, -1.8, 0.65],
                [1.1, -1.35, 1.0],
                [0.2, 0.0, 0.65],
                [0.0, 0.75, 0.525],
                [-0.2, 0.75, 1.2],
                [-0.5, -0.5, 1.0],
                [-0.5, -1.0, 0.8],
            ]
        )
        # Scale trajectory between 0 and 1
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])

        des_completion_time = 10
        ts = np.linspace(0, 1, int(initial_info["env.freq"] * des_completion_time))

        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)

        try:
            # Draw interpolated Trajectory
            trajectory = np.vstack([self.x_des, self.y_des, self.z_des]).T
            for i in range(len(trajectory) - 1):
                p.addUserDebugLine(
                    trajectory[i],
                    trajectory[i + 1],
                    lineColorRGB=[1, 0, 0],
                    lineWidth=2,
                    lifeTime=0,
                    physicsClientId=0,
                )
        except p.error:
            ...  # Ignore pybullet errors if not running in the pybullet GUI

    def compute_control(
        self, obs: npt.NDArray[np.floating], info: dict | None = None
    ) -> npt.NDarray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        des_pos = np.array([self.x_des[self._tick], self.y_des[self._tick], self.z_des[self._tick]])
        des_vel = np.zeros(3)
        des_yaw = 0.0

        # Calculate the deviations from the desired trajectory
        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        # Update integral error
        self.i_error += pos_error * (1 / self.low_level_ctrl_freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        # target_thrust += params.quad.m * desired_acc
        target_thrust[2] += self.drone_mass * self.g

        # Update z_axis to the current orientation of the drone
        rpy = obs["rpy"]
        z_axis = R.from_euler("xyz", [rpy[0], rpy[1], rpy[2]]).as_matrix()[:, 2]

        # update current thrust
        thrust_desired = target_thrust.dot(z_axis)
        thrust_desired = max(thrust_desired, 0.3 * self.drone_mass * self.g)
        thrust_desired = min(thrust_desired, 1.8 * self.drone_mass * self.g)

        # update z_axis_desired
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)
        thrust_desired, euler_desired

        # Invert the pitch because of the legacy Crazyflie firmware coordinate system
        euler_desired[1] = -euler_desired[1]
        return np.concatenate([[thrust_desired], euler_desired])

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: NDArray[np.floating],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        """Increment the tick counter."""
        self._tick += 1

    def episode_callback(self):
        """Reset the integral error."""
        self.i_error[:] = 0
