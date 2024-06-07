"""Controller using a PPO agent trained with stable-baselines3."""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from safe_control_gym.controllers.firmware.firmware_wrapper import logging
from stable_baselines3 import PPO

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
        """
        super().__init__(initial_obs, initial_info, buffer_size, verbose)

        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        # self.model_name = "models/best_model_morning_2024-05-31"
        # self.model_name = "models/ppo_2024-06-05_23-34-37"
        # self.model_name = "models/ppo_2024-06-06_08-08-15"
        # self.model_name = "models/ppo_2024-06-05_09-43-45"
        self.model_name = "models/best_model"
        self.model = PPO.load(self.model_name)

        self._take_off = False
        self._setpoint_land = False
        self._land = False
        self._goto = False
        self.stamp = 0
        self._goal = np.array(
            [
                initial_info["x_reference"][0],
                initial_info["x_reference"][2],
                initial_info["x_reference"][4],
            ]
        )

    # TODO testen State-Machine

    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
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
        # take off
        if not self._take_off:
            command_type = Command.TAKEOFF
            args = [0.1, 2]  # height, duration
            self._take_off = True  # only send takeoff command once

        else:
            # use policy until reaching last gate
            if (
                ep_time - 2 > 0 and info["current_gate_id"] != -1
            ):  # Account for 2s delay due to takeoff
                action, next_predicted_state = self.model.predict(obs, deterministic=True)
                action = transform_action(action, drone_pos=obs[:3])

                # Prepare the command to be sent to the quadrotor.
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

            # Notify set point stop has to be called every time we transition from low-level
            # commands to high-level ones. Prepares for landing
            elif (
                info["current_gate_id"] == -1 and not self._setpoint_land
            ):  # TODO testen zum punkt fliegen wo er landen soll
                command_type = Command.NOTIFYSETPOINTSTOP
                args = []
                self.time_stamp = ep_time
                self._setpoint_land = True

            elif (
                info["current_gate_id"] == -1 and not self._goto
            ):  # TODO testen zum punkt fliegen wo er landen soll
                command_type = Command.GOTO
                # Prepare the command to be sent to the quadrotor.

                # target_pos = np.array([x, y, z])
                args = [self._goal, 0.0, 3.0, False]
                self.time_stamp = ep_time
                self._goto = True
                self.stamp = ep_time

            elif (
                info["current_gate_id"] == -1 and info["at_goal_position"] and not self._land
            ):  # TODO testen landen
                command_type = Command.LAND
                args = [0.0, 10]  # Height, duration
                self._land = True  # Send landing command only once

            elif self._land:
                command_type = Command.FINISHED
                args = []

            else:
                command_type = Command.NONE
                args = []

        return command_type, args
