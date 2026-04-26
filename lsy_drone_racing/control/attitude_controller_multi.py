"""This module implements a wrapper for the attitude controller for a multi-agent environment.

It extracts the relevant information for the current agent from the observation,
 and passes it to the single-agent attitude controller.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

from lsy_drone_racing.control.attitude_controller import (
    AttitudeController as SingleAttitudeController,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class AttitudeController(SingleAttitudeController):
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
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        assert obs["pos"].ndim == 2, (
            f"Observation should have 2 dimensions but now it has {obs['pos'].ndim} dimensions. "
            "Are you sure you are running the multi-agent environment?"
        )
        obs = {
            "pos": obs["pos"][self.rank],
            "vel": obs["vel"][self.rank],
            "quat": obs["quat"][self.rank],
            "ang_vel": obs["ang_vel"][self.rank],
        }
        return super().compute_control(obs, info)
