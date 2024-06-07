"""Controller using a PPO agent trained with stable-baselines3."""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from safe_control_gym.controllers.firmware.firmware_wrapper import logging
from stable_baselines3 import PPO

from controllers.ppo.ppo_deploy import StateMachine
from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.env_modifiers import ObservationParser, transform_action

logger = logging.getLogger(__name__)


class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = False,
    ):
        """Initialization of the controller."""
        super().__init__(initial_obs, initial_info, buffer_size, verbose)

        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        self.reset()
        self.episode_reset()

        self.model_name = "models/best_model_minimal_train0"
        self.model = PPO.load(self.model_name)

        self._goal = np.array(
            [
                initial_info["x_reference"][0],
                initial_info["x_reference"][2],
                initial_info["x_reference"][4],
            ]
        )
        self.state_machine = StateMachine(self._goal)

    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface."""
        if ep_time - 2 > 0 and info["current_gate_id"] != -1:
            action, next_predicted_state = self.model.predict(obs, deterministic=True)
            action = transform_action(action, drone_pos=obs[:3])

            x = float(action[0])
            y = float(action[1])
            z = float(action[2])
            target_pos = np.array([x, y, z])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = float(action[3])
            target_rpy_rates = np.zeros(3)
            command_type = Command.FULLSTATE
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
            return command_type, args

        return self.state_machine.transition(ep_time, info)
