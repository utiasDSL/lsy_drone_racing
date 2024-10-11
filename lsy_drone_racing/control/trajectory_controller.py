"""Write your control strategy.

Then run:

    $ python scripts/sim --config config/getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) compute_control
        3) step_learn (optional)
        4) episode_learn (optional)

"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import pybullet as p
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import BaseController

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Controller(BaseController):
    """Template controller class."""

    def __init__(self, initial_obs: NDArray[np.floating], initial_info: dict):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """
        super().__init__(initial_obs, initial_info)
        waypoints = np.array(
            [
                [1.0, 1.0, 0.0],
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
        t = np.arange(len(waypoints))
        self.trajectory = CubicSpline(t, waypoints)
        self._tick = 0

        # Generate points along the spline for visualization
        t_vis = np.linspace(0, len(waypoints) - 1, 100)
        spline_points = self.trajectory(t_vis)
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

    def compute_control(
        self, obs: NDArray[np.floating], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired position and orientation of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone pose [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        target_pos = self.trajectory(self._tick / 30)
        return np.concatenate((target_pos, np.zeros(10)))

    def step_learn(self, *args, **kwargs):
        self._tick += 1
        return super().step_learn(*args, **kwargs)

    def episode_reset(self):
        self._tick = 0
