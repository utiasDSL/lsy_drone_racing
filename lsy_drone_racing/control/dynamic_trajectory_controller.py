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
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DynamicTrajectoryController(Controller):
    """Trajectory controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)

        self.start_pos = obs["pos"]

        self.t_total = 12
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False
        
      
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
        
        # Generate the trajectory based on the new waypoints
        trajectory = self.create_trajectory(obs)

        tau = min(self._tick / self._freq, self.t_total)
        target_pos = trajectory(tau)
        if tau == self.t_total:  # Maximum duration reached
            self._finished = True
        
        return np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)
    
    def create_trajectory(self, obs: NDArray[np.floating]) -> CubicSpline:
        """Create a cubic spline trajectory based on the current observation.

        Args:
            obs (NDArray[np.floating]): The current observation of the environment
                observation space for details.

        Returns:
            CubicSpline: A cubic spline object representing the trajectory.
        """
        waypoints = np.array([
            self.start_pos,
            obs["obstacles_pos"][0] + [0.2, 0.5, -0.7],
            obs["obstacles_pos"][0] + [0.2, -0.3, -0.7],
            obs["gates_pos"][0] + 0.5 * (obs["obstacles_pos"][0] - [0, 0, 0.6] - obs["gates_pos"][0]),
            obs["gates_pos"][0] + [-0.1, 0.1, 0],
            obs["gates_pos"][0] + [-0.6, -0.2, 0],
            obs["obstacles_pos"][1] + [-0.3, -0.3, -0.7],
            obs["gates_pos"][1] + [-0.1, -0.2, 0],
            obs["gates_pos"][1],
            obs["gates_pos"][1] + [0.2, 0.5, 0],
            obs["obstacles_pos"][0] + [-0.3, 0, -0.7],
            obs["gates_pos"][2] + [0.2, -0.5, 0],
            obs["gates_pos"][2] + [0.1, 0, 0],
            obs["gates_pos"][2] + [0.1, 0.15, 0],
            obs["gates_pos"][2] + [0.1, 0.15, 1],
            obs["obstacles_pos"][3] + [0.4, 0.3, -0.2],
            obs["obstacles_pos"][3] + [0.4, 0, -0.2],
            obs["gates_pos"][3],
            obs["gates_pos"][3] + [0, -0.5, 0],
        ])

        t = np.linspace(0, self.t_total, len(waypoints))
        return CubicSpline(t, waypoints)


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
        """Reset the time step counter."""
        self._tick = 0
