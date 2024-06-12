"""Wrapper to make the environment compatible with the gymnasium API.

The drone simulator does not conform to the gymnasium API, which is used by most RL frameworks. This
wrapper can be used as a translation layer between these modules and the simulation.

RL environments are expected to have a uniform action interface. However, the Crazyflie commands are
highly heterogeneous. Users have to make a discrete action choice, each of which comes with varying
additional arguments. Such an interface is impractical for most standard RL algorithms. Therefore,
we restrict the action space to only include FullStateCommands.

We also include the gate pose and range in the observation space. This information is usually
available in the info dict, but since it is vital information for the agent, we include it directly
in the observation space.

Warning:
    The RL wrapper uses a reduced action space and a transformed observation space!
"""

from __future__ import annotations

import logging
import pprint
from typing import Any

import numpy as np
from gymnasium import Wrapper
from gymnasium.error import InvalidAction
from gymnasium.spaces import Box
from rich.logging import RichHandler
from safe_control_gym.controllers.firmware.firmware_wrapper import FirmwareWrapper
from termcolor import colored

from lsy_drone_racing.env_modifiers import (
    ActionTransformer,
    MinimalObservationParser,
    ObservationParser,
    RelativeActionTransformer,
    RelativePositionObservationParser,
    Rewarder,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("wrapper")
logger.setLevel(logging.INFO)


class DroneRacingWrapper(Wrapper):
    """Drone racing firmware wrapper to make the environment compatible with the gymnasium API.

    In contrast to the underlying environment, this wrapper only accepts FullState commands as
    actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env: FirmwareWrapper,
        terminate_on_lap: bool = False,
        observation_parser_path: str = None,
        rewarder_path: str = None,
        action_transformer_path: str = None,
    ):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
            terminate_on_lap: Stop the simulation early when the drone has passed the last gate.
            observation_parser_path: The path to the observation parser configuration file.
            rewarder_path: The path to the rewarder configuration file.
            action_transformer_path: The path to the action transformer configuration file.
        """
        super().__init__(env)
        # Patch the FirmwareWrapper to add any missing attributes required by the gymnasium API.
        self.env = env
        self.env.unwrapped = None  # Add an (empty) unwrapped attribute
        self.env.render_mode = None

        # Action space for the model. The action space is later transformed and finally converted to
        # the firmware action.
        action_limits = np.ones(4)
        self.action_space = Box(-action_limits, action_limits, dtype=np.float32)

        # Observation space:
        if observation_parser_path:
            try:
                self.observation_parser = ObservationParser.from_yaml(
                    n_gates=env.env.NUM_GATES,
                    n_obstacles=env.env.n_obstacles,
                    file_path=observation_parser_path,
                )
            except Exception as e:
                logger.error(f"Failed to load observation parser from YAML: {e}")
                logger.error("Using default minimal observation parser.")
                self.observation_parser = MinimalObservationParser(
                    n_gates=env.env.NUM_GATES, n_obstacles=env.env.n_obstacles
                )
        else:
            self.observation_parser = MinimalObservationParser(
                n_gates=env.env.NUM_GATES, n_obstacles=env.env.n_obstacles
            )
        self.observation_space = self.observation_parser.observation_space

        # Loading the rewarder
        if rewarder_path:
            try:
                self.rewarder = Rewarder.from_yaml(rewarder_path)
            except Exception as e:
                logger.error(f"Failed to load rewarder from YAML: {e}")
                logger.error("Using default rewarder.")
                self.rewarder = Rewarder()
        else:
            self.rewarder = Rewarder()

        # Loading the action transformer
        if action_transformer_path:
            try:
                self.action_transformer = ActionTransformer.from_yaml(file_path=action_transformer_path)
            except Exception as e:
                logger.error(f"Failed to load action transformer from YAML: {e}")
                logger.error("Using action transformer with relative actions.")
                self.action_transformer = RelativeActionTransformer()

        self.pyb_client_id: int = env.env.PYB_CLIENT
        # Config and helper flags
        self.terminate_on_lap = terminate_on_lap
        self._reset_required = False
        # The original firmware wrapper requires a sim time as input to the step function. This
        # The firmware quadrotor env requires the rotor forces as input to the step function. These
        # are zero initially and updated by the step function. We automatically insert them to
        # ensure compatibility with the gymnasium interface.
        # TODO: It is not clear if the rotor forces are even used in the firmware env. Initial tests
        #       suggest otherwise.
        self._f_rotors = np.zeros(4)

    @property
    def time(self) -> float:
        """Return the current simulation time in seconds."""
        return self._sim_time

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            seed: The random seed to use for the environment. Not used in this wrapper.
            options: Additional options to pass to the environment. Not used in this wrapper.

        Returns:
            The initial observation and info dict of the next episode.
        """
        self._reset_required = False
        self._sim_time = 0.0
        self._f_rotors[:] = 0.0
        obs, info = self.env.reset()

        logger.debug("===========RESET===========")
        logger.debug(f"Collision: {pprint.pformat(info['collision'])}")
        logger.debug(f"Finished: {pprint.pformat(info['task_completed'])}")

        # Store obstacle height for observation expansion during env steps.
        self.observation_parser.update(obs, info, initial=True)
        obs = self.observation_parser.get_observation().astype(np.float32)

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        assert not self._reset_required, "Environment must be reset before taking a step"
        if action not in self.action_space:
            # Wrapper has a reduced action space compared to the firmware env to make it compatible
            # with the gymnasium interface and popular RL libraries.
            raise InvalidAction(f"Invalid action: {action}")

        # Transform the action using a custom action transformer and then adapt it to the firmware
        action = self.action_transformer.transform(raw_action=action, drone_pos=self.observation_parser.drone_pos)
        firmware_action = self.action_transformer.create_firmware_action(action, sim_time=self._sim_time)
        self.env.sendFullStateCmd(*firmware_action)

        # The firmware quadrotor env requires the sim time as input to the step function. It also
        # returns the desired rotor forces. Both modifications are not part of the gymnasium
        # interface. We automatically insert the sim time and reuse the last rotor forces.
        obs, reward, done, info, f_rotors = self.env.step(self._sim_time, action=self._f_rotors)
        self._f_rotors[:] = f_rotors

        # We set truncated to True if the task is completed but the drone has not yet passed the
        # final gate. We set terminated to True if the task is completed and the drone has passed
        # the final gate.
        terminated, truncated = False, False
        if info["task_completed"] and info["current_gate_id"] != -1:
            truncated = True
        elif self.terminate_on_lap and info["current_gate_id"] == -1:
            info["task_completed"] = True
            terminated = True
        elif done:  # Done, but last gate not passed -> terminate
            terminated = True

        # Update the observation parser and get the observation.
        self.observation_parser.update(obs, info)
        obs = self.observation_parser.get_observation().astype(np.float32)

        if self.observation_parser.out_of_bounds():
            terminated = True

        # Increment the sim time after the step if we are not yet done.
        if not terminated and not truncated:
            self._sim_time += self.env.ctrl_dt

        # Compute the custom reward
        reward = self.rewarder.get_custom_reward(self.observation_parser, info)

        logger.debug("===Step===")
        logger.debug(f"Collision: {pprint.pformat(info['collision'])}")
        logger.debug(f"Finished: {pprint.pformat(info['task_completed'])}")
        logger.debug(f"Drone position: {self.observation_parser.drone_pos}")
        logger.debug(f"Action: {action}")
        logger.debug(f"Available keys in info: {pprint.pformat(info.keys())}")
        logger.debug(f"Reward: {reward}")

        self._reset_required = terminated or truncated
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment.

        Used for compatibility with the gymnasium API. Checks if PyBullet was launched with an
        active GUI.

        Raises:
            AssertionError: If PyBullet was not launched with an active GUI.
        """
        assert self.pyb_client_id != -1, "PyBullet not initialized with active GUI"


class DroneRacingObservationWrapper:
    """Wrapper to transform the observation space the firmware wrapper.

    This wrapper matches the observation space of the DroneRacingWrapper. See its definition for
    more details. While we want to transform the observation space, we do not want to change the API
    of the firmware wrapper. Therefore, we create a separate wrapper for the observation space.

    Note:
        This wrapper is not a subclass of the gymnasium ObservationWrapper because the firmware is
        not compatible with the gymnasium API.
    """

    def __init__(
        self,
        env: FirmwareWrapper,
        observation_parser: ObservationParser = None,
        rewarder: Rewarder = None,
        action_transformer: ActionTransformer = None,
    ):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
            observation_parser: The observation parser to use. If None, a default parser is used.
            rewarder: The rewarder to use. If None, a default rewarder is used.
            action_transformer: The action transformer to use. If None, a default transformer is used.
        """
        if not isinstance(env, FirmwareWrapper):
            raise TypeError(f"`env` must be an instance of `FirmwareWrapper`, is {type(env)}")
        self.env = env
        self.pyb_client_id: int = env.env.PYB_CLIENT
        self.observation_parser = observation_parser if observation_parser else RelativePositionObservationParser(
            n_gates=env.env.NUM_GATES, n_obstacles=env.env.n_obstacles
        )
        self.rewarder = rewarder if rewarder else Rewarder()
        self.action_transformer = action_transformer if action_transformer else RelativeActionTransformer()

    def __getattribute__(self, name: str) -> Any:
        """Get an attribute from the object.

        If the attribute is not found in the wrapper, it is fetched from the firmware wrapper.

        Args:
        If the attribute is not found in the wrapper, it is fetched from the firmware wrapper.

        Args:
            name: The name of the attribute.

        Returns:
            The attribute value.
        """
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The transformed observation and the info dict.
        """
        obs, info = self.env.reset(*args, **kwargs)
        self.observation_parser.update(obs, info, initial=True)
        obs = self.observation_parser.get_observation()

        logger.debug(f"{colored('===Reset===', 'green')}")
        logger.debug(f"Available keys in info: {pprint.pformat(info.keys())}")
        logger.debug(f"Collision: {pprint.pformat(info['collision'])}")
        logger.debug(f"Finished: {pprint.pformat(info['task_completed'])}")

        return obs, info

    def step(self, *args: Any, **kwargs: dict[str, Any]) -> tuple[np.ndarray, float, bool, dict, np.ndarray]:
        """Take a step in the current environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            obs: The transformed observation.
            reward: The reward signal.
            done: Whether the episode has terminated.
            info: The info dictionary.
            action: The action taken by the agent in the firmware format.
        """
        obs, reward, done, info, action = self.env.step(*args, **kwargs)
        self.observation_parser.update(obs, info)
        obs = self.observation_parser.get_observation().astype(np.float32)

        reward = self.rewarder.get_custom_reward(self.observation_parser, info)

        logger.debug("===Step===")
        logger.debug(f"Collision: {pprint.pformat(info['collision'])}")
        logger.debug(f"Finished: {pprint.pformat(info['task_completed'])}")
        logger.debug(f"Reward: {reward}")
        logger.debug(f"Drone position: {self.observation_parser.drone_pos}")
        logger.debug(f"Gate ID: {self.observation_parser.gate_id}")
        logger.debug(f"Action: {action}")

        return obs, reward, done, info, action
