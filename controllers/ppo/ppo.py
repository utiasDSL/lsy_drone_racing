"""Controller using a PPO agent trained with stable-baselines3."""

from __future__ import annotations

from typing import Any  # Python 3.10 type hints

import numpy as np
from safe_control_gym.controllers.firmware.firmware_wrapper import logging
from stable_baselines3 import PPO

from controllers.ppo.ppo_deploy import DroneStateMachine
from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.env_modifiers import ActionTransformer, ObservationParser, RelativeActionTransformer

logger = logging.getLogger(__name__)


class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = False,
        model_name: str | None = None,
        action_transformer: str | None = None,
        **kwargs: Any,
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. Consists of
                [drone_xyz_yaw, gates_xyz_yaw, gates_in_range, obstacles_xyz, obstacles_in_range,
                gate_id]
            initial_info: The a priori information as a dictionary with keys 'symbolic_model',
                'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            buffer_size: Size of the data buffers used in method `learn()`.
            verbose: Turn on and off additional printouts and plots.
            model_name: The path to the trained model.
            action_transformer: The action transformer object.
            **kwargs: Additional keyword arguments.
        """
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

        self.model_name = model_name if model_name else "models/working_model"
        self.model = PPO.load(self.model_name)
        self.action_transformer = (
            ActionTransformer.from_yaml(action_transformer) if action_transformer else RelativeActionTransformer()
        )

        self._goal = np.array(
            [
                initial_info["x_reference"][0],
                initial_info["x_reference"][2],
                initial_info["x_reference"][4],
            ]
        )
        self.state_machine = DroneStateMachine(self._goal, self.model, self.action_transformer) 

    def compute_control(
        self,
        ep_time: float,
        obs: ObservationParser | np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration,
            attitude, and attitude rates to be sent from Crazyswarm to the Crazyflie using, e.g., a
            `cmdFullState` call.

        Args:
            ep_time: Episode's elapsed time, in seconds.
            obs: The environment's observation [drone_xyz_yaw, gates_xyz_yaw, gates_in_range,
                obstacles_xyz, obstacles_in_range, gate_id].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        command, args = self.state_machine.transition(ep_time, obs, info)
        return command, args
