"""This controller is only called when the drone passes the last gate and the environment gets closed.

The controller works in two steps: First, is slows down from any initial speed. Second, it returns to the starting position -> return to home (RTH)

The waypoints for the stopping part are generated using a quintic spline, to be able to set the start and end velocity and acceleration.
The waypoints for the RTH trajectory are created by a cubic spline with waypoints in a predefined height. 
"""

from __future__ import annotations  # Python 3.10 type hints

import logging
from enum import Enum
from typing import TYPE_CHECKING, Union

import numpy as np
import pybullet as p
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import BaseController

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger("rosout." + __name__)  # Get ROS compatible logger


class Mode(Enum):
    """Enum class for the different modes of the controller."""

    STOP = 0
    HOVER = 1
    RETURN = 2
    LAND = 3


class ClosingController(BaseController):
    """Controller that creates and follows a braking and homing trajectory."""

    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], initial_info: dict):
        """Initialization of the controller.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """
        super().__init__(initial_obs, initial_info)

        self._tick = 0
        self.info = initial_info
        self._freq = self.info["env_freq"]
        self._t_step_ctrl = 1 / self._freq  # control step

        self.debug = False  # Plot the trajectories in sim. TODO: Make configurable
        self._return_height = 2.0  # height of the return path

        self._target_pos = initial_obs["pos"]
        self._brake_trajectory, self._t_brake = self._generate_brake_trajectory(initial_obs)
        self._return_trajectory = None  # Gets generated on the fly based on the current state

        self._t_hover = 2.0  # time the drone hovers before returning home and before landing
        self._t_return = 9.0  # time it takes to get back to start (return to home)
        self.t_total = self._t_brake + self._t_hover + self._t_return + self._t_hover

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
        t = self._tick * self._t_step_ctrl
        if self.mode == Mode.STOP:
            target_pos = self._brake_trajectory(t, order=0)
            target_vel = self._brake_trajectory(t, order=1)
            target_acc = self._brake_trajectory(t, order=2)
            self._target_pos = obs["pos"]  # store for hover mode
        elif self.mode == Mode.HOVER:
            target_pos = self._target_pos
            target_vel = [0, 0, 0]
            target_acc = [0, 0, 0]
        elif self.mode == Mode.RETURN:
            if self._return_trajectory is None:
                self._return_trajectory = self._generate_return_trajectory(obs, t)
            target_pos = self._return_trajectory(t)
            trajectory_v = self._return_trajectory.derivative()
            trajectory_a = trajectory_v.derivative()
            target_vel = trajectory_v(t) * 0
            target_acc = trajectory_a(t) * 0
        elif self.mode == Mode.LAND:
            target_pos = np.array(self.info["drone_start_pos"]) + np.array([0, 0, 0.25])
            target_vel = [0, 0, 0]
            target_acc = [0, 0, 0]
        return np.concatenate((target_pos, target_vel, target_acc, np.zeros(4)))

    @property
    def mode(self) -> Mode:
        """Return the current mode of the controller."""
        t = self._tick * self._t_step_ctrl
        if t <= self._t_brake:
            return Mode.STOP
        if self._t_brake < t <= self._t_brake + self._t_hover:
            return Mode.HOVER
        if self._t_brake + self._t_hover < t <= self._t_brake + self._t_hover + self._t_return:
            return Mode.RETURN
        return Mode.LAND

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

    def _generate_brake_trajectory(
        self, obs: dict[str, NDArray[np.floating]]
    ) -> Union[QuinticSpline, np.floating]:
        start_pos = obs["pos"]
        start_vel = obs["vel"]
        start_acc = obs["acc"]

        direction = start_vel / np.linalg.norm(start_vel)  # Unit vector in the direction of travel
        # Angle to the floor => negative means v_z < 0
        direction_angle = -(np.arccos((np.dot(direction, [0, 0, 1]))) - np.pi / 2)
        # Calculate the distance the drone can travel before stopping
        dist = np.clip(np.linalg.norm(start_vel), 0.5, 2.0) / np.cos(direction_angle)
        # Check if drone would crash into floor or ceiling, and clip the values if necessary
        pos_z = start_pos[2] + dist * np.sin(direction_angle)
        dist = (np.clip(pos_z, 0.2, 2.5) - start_pos[2]) / np.sin(direction_angle)

        pos_z = start_pos[2] + dist * np.sin(direction_angle)  # Recalculate pos_z after clipping
        logger.debug(f"Start height: {start_pos[2]:.2f}, End height: {pos_z:.2f}, Dist: {dist:.2f}")
        logger.debug(f"Angle: {direction_angle*180/np.pi:.0f}°")

        # Estimate the constant deceleration that is necessary for braking
        const_acc = np.linalg.norm(start_vel) ** 2 / (dist)
        # The time it takes to brake completely. Clip to 1s minimum to avoid short braking times
        t_brake_max = max(1.0, np.sqrt(4 * dist / const_acc))

        logger.debug(f"Gate velocity: {start_vel}, Braking time: {t_brake_max:.2f}s")

        brake_trajectory = QuinticSpline(
            0,
            t_brake_max,
            start_pos,
            start_vel,
            start_acc,
            start_pos + direction * dist,
            start_vel * 0,
            start_acc * 0,
        )

        if self.debug:
            plot_trajectory(brake_trajectory, color=[0, 1, 0])
        return brake_trajectory, t_brake_max

    def _generate_return_trajectory(
        self, obs: dict[str, NDArray[np.floating]], t: np.floating
    ) -> CubicSpline:
        start_pos = obs["pos"]
        # Landing position is 0.25m above the ground
        landing_pos = np.array(self.info["drone_start_pos"]) + np.array([0, 0, 0.25])

        delta_pos = landing_pos - start_pos
        intermed_pos1 = start_pos + delta_pos / 4
        intermed_pos1[2] = self._return_height

        intermed_pos2 = intermed_pos1 + delta_pos / 2
        intermed_pos2[2] = self._return_height

        intermed_pos3 = 5 * landing_pos / 6 + intermed_pos2 / 6
        intermed_pos3[2] = landing_pos[2] / 2 + intermed_pos2[2] / 2

        waypoints = np.array([start_pos, intermed_pos1, intermed_pos2, intermed_pos3, landing_pos])
        bc_type = ((1, [0, 0, 0]), (1, [0, 0, 0]))  # Set boundary conditions for the derivative (1)
        t = np.linspace(t, t + self._t_return, len(waypoints))
        return_trajectory = CubicSpline(t, waypoints, bc_type=bc_type)

        if self.debug:
            plot_trajectory(return_trajectory)
        return return_trajectory


def plot_trajectory(spline: CubicSpline | QuinticSpline, color: list[float] = [0, 0, 1]):
    """Plot the drone's and the controller's trajectories."""
    n_segments = 20
    t = np.linspace(spline.x[0], spline.x[-1], n_segments)
    pos = spline(t)
    try:
        for i in range(len(pos)):
            p.addUserDebugLine(
                pos[max(i - 1, 0)],
                pos[i],
                lineColorRGB=color,
                lineWidth=2,
                lifeTime=0,  # 0 means the line persists indefinitely
                physicsClientId=0,
            )
        for x in spline(spline.x):
            p.addUserDebugText("x", x, textColorRGB=color)
    except p.error:
        logger.warning("PyBullet not available")  # Ignore errors if PyBullet is not available


class QuinticSpline:
    """This class and the code below is mostly written by ChatGPT 4.0."""

    def __init__(
        self,
        t0: np.floating,
        t1: np.floating,
        x0: np.floating,
        v0: np.floating,
        a0: np.floating,
        x1: np.floating,
        v1: np.floating,
        a1: np.floating,
    ):
        """Initialize the quintic spline for multidimensional conditions.

        Params:
            t0: Start time.
            t1: End time.
            x0: Initial position.
            v0: Initial velocity.
            a0: Initial acceleration.
            x1: Final position.
            v1: Final velocity.
            a1: Final acceleration.
        """
        self.t_points = (t0, t1)  # Start and end time points
        self.dimensions = len(x0)  # Number of dimensions
        # Boundary conditions per dimension
        self.boundary_conditions = np.array([x0, v0, a0, x1, v1, a1])
        self.splines = np.array([self._compute_spline(i) for i in range(self.dimensions)])
        self.x = np.array([t0, t1])  # Make compatible to CubicSpline API

    def _compute_spline(self, dim: int) -> NDArray:
        t0, t1 = self.t_points
        x0, v0, a0, x1, v1, a1 = self.boundary_conditions[:, dim]

        # Constructing the coefficient matrix for the quintic polynomial
        M = np.array(
            [
                [1, t0, t0**2, t0**3, t0**4, t0**5],  # position @ t0
                [0, 1, 2 * t0, 3 * t0**2, 4 * t0**3, 5 * t0**4],  # velocity @ t0
                [0, 0, 2, 6 * t0, 12 * t0**2, 20 * t0**3],  # acceleration @ t0
                [1, t1, t1**2, t1**3, t1**4, t1**5],  # position @ t1
                [0, 1, 2 * t1, 3 * t1**2, 4 * t1**3, 5 * t1**4],  # velocity @ t1
                [0, 0, 2, 6 * t1, 12 * t1**2, 20 * t1**3],  # acceleration @ t1
            ]
        )
        # Construct the boundary condition vector
        b = np.array([x0, v0, a0, x1, v1, a1])
        # Solve for coefficients of the quintic polynomial
        coefficients = np.linalg.solve(M, b)
        return coefficients

    def __call__(self, t: np.floating, order: int = 0) -> NDArray:
        """Evaluate the quintic spline or its derivatives at a given time t for all dimensions.

        Params:
            t: Time at which to evaluate
            order: Derivative order (0=position, 1=velocity, 2=acceleration)

        Examples:
            >>> t0, t1 = 0, 1
            >>> x0, v0, a0 = [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]  # Initial conditions for x and y
            >>> x1, v1, a1 = [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]  # Final conditions for x and y
            >>> spline = QuinticSpline(t0, t1, x0, v0, a0, x1, v1, a1)
            >>> t_values = np.linspace(t0, t1, 100)
            >>> pos_values = spline(t_values, order=0)  # Position
            >>> vel_values = np.array([spline(t, order=1) for t in t_values])  # Velocity
            >>> acc_values = np.array([spline(t, order=2) for t in t_values])  # Acceleration

        Returns:
            A list of evaluated values for each dimension
        """
        t = np.atleast_1d(t)
        if order == 0:
            return self._position(t).squeeze()
        elif order == 1:
            return self._velocity(t).squeeze()
        elif order == 2:
            return self._acceleration(t).squeeze()
        raise ValueError(f"Spline orders (0, 1, 2) are supported, got {order}")

    def _position(self, t: NDArray[np.floating]) -> NDArray[np.floating]:
        powers = t[:, None] ** np.arange(len(self.splines[0]))
        return np.dot(powers, self.splines.T)

    def _velocity(self, t: NDArray[np.floating]) -> NDArray[np.floating]:
        mult = np.arange(len(self.splines[0]))
        powers = np.zeros((len(t), len(self.splines[0])))
        powers[1:, :] = t[1:, None] ** (mult - 1)
        coeffs = self.splines * mult[None, :]  # Multiply by i
        return np.dot(powers, coeffs.T)

    def _acceleration(self, t: NDArray[np.floating]) -> NDArray[np.floating]:
        mult = np.arange(len(self.splines[0]))
        powers = np.zeros((len(t), len(self.splines[0])))
        powers[1:, :] = t[1:, None] ** (mult - 2)
        coeffs = self.splines * (mult * (mult - 1))[None, :]  # Multiply by i*(i-1)
        return np.dot(powers, coeffs.T)
