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

from pathlib import Path

import numpy as np
import numpy.typing as npt
from stable_baselines3 import PPO

from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.wrapper import ObsWrapper


class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self, initial_obs: npt.NDArray[np.floating], initial_info: dict, buffer_size: int = 100
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
            buffer_size: Size of the data buffers used in method `learn()`.
        """
        super().__init__(initial_obs, initial_info, buffer_size)
        self.policy = PPO.load(Path(__file__).resolve().parents[1] / "models/ppo/model.zip")
        self._last_action = np.zeros(3)

    def compute_control(
        self, obs: npt.NDArray[np.floating], info: dict | None = None
    ) -> npt.NDarray[np.floating]:
        """Compute the next desired position and orientation of the drone.

        INSTRUCTIONS:
            Re-implement this method to return the target pose to be sent from Crazyswarm to the
            Crazyflie using the `cmdFullState` call.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone pose [x_des, y_des, z_des, yaw_des] as a numpy array.
        """
        obs_tf = ObsWrapper.observation_transform(obs, info, self._last_action)
        action, _ = self.policy.predict(obs_tf, deterministic=True)
        self._last_action[:] = action
        target_pos = self.action_transform(action, obs)
        action = np.zeros(4)
        action[:3] = target_pos
        return action

    @staticmethod
    def action_transform(
        action: npt.NDArray[np.floating], obs: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        drone_pos = obs[:3]
        return drone_pos + action

    def episode_reset(self):
        self._last_action = np.zeros(3)
