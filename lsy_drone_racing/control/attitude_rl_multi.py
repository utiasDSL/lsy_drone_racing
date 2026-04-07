"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

from lsy_drone_racing.control.attitude_rl import AttitudeRL as SingleAttitudeRL

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class AttitudeRL(SingleAttitudeRL):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self.rank = info["rank"]

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        obs["pos"] = obs["pos"][self.rank]
        obs["quat"] = obs["quat"][self.rank]
        obs["vel"] = obs["vel"][self.rank]
        obs["ang_vel"] = obs["ang_vel"][self.rank]
        obs["target_gate"] = obs["target_gate"][self.rank]
        obs["gates_visited"] = obs["gates_visited"][self.rank]
        obs["obstacles_visited"] = obs["obstacles_visited"][self.rank]
        return super().compute_control(obs, info)
