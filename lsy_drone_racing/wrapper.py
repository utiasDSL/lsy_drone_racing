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
from safe_control_gym.controllers.firmware.firmware_wrapper import FirmwareWrapper
from termcolor import colored

from lsy_drone_racing.constants import Z_HIGH, Z_LOW
from lsy_drone_racing.env_modifiers import ObservationParser, Rewarder, map_reward_to_color

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DroneRacingWrapper(Wrapper):
    """Drone racing firmware wrapper to make the environment compatible with the gymnasium API.

    In contrast to the underlying environment, this wrapper only accepts FullState commands as
    actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: FirmwareWrapper, terminate_on_lap: bool = True):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
            terminate_on_lap: Stop the simulation early when the drone has passed the last gate.
        """
        super().__init__(env)
        # Patch the FirmwareWrapper to add any missing attributes required by the gymnasium API.
        self.env = env
        self.env.unwrapped = None  # Add an (empty) unwrapped attribute
        self.env.render_mode = None

        # Gymnasium env required attributes
        # Action space:
        # [x, y, z, yaw]
        # x, y, z)  The desired position of the drone in the world frame.
        # yaw)      The desired yaw angle.
        # All values are scaled to [-1, 1]. Transformed back, x, y, z values of 1 correspond to 5m.
        # The yaw value of 1 corresponds to pi radians.
        action_limits = np.ones(4)
        self.action_scale = np.array([5, 5, 5, np.pi])
        self.action_space = Box(-action_limits, action_limits, dtype=np.float32)

        # Observation space:
        # [drone_xyz_yaw, gates_xyz_yaw, gates_in_range, obstacles_xyz, obstacles_in_range, gate_id]
        # drone_xyz_yaw)  x, y, z, yaw are the drone pose of the drone in the world frame. Position
        #       is in meters and yaw is in radians.
        # gates_xyz_yaw)  The pose of the gates. Positions are in meters and yaw in radians. The
        #       length is dependent on the number of gates. Ordering is [x0, y0, z0, yaw0, x1,...].
        # gates_in_range)  A boolean array indicating if the drone is within the gates' range. The
        #       length is dependent on the number of gates.
        # obstacles_xyz)  The pose of the obstacles. Positions are in meters. The length is
        #       dependent on the number of obstacles. Ordering is [x0, y0, z0, x1,...].
        # obstacles_in_range)  A boolean array indicating if the drone is within the obstacles'
        #       range. The length is dependent on the number of obstacles.
        # gate_id)  The ID of the current target gate. -1 if the task is completed.
        n_gates = env.env.NUM_GATES
        n_obstacles = env.env.n_obstacles
        drone_limits = [5, 5, 5, np.pi]
        gate_limits = [5, 5, 5, np.pi] * n_gates + [1] * n_gates  # Gate poses and range mask
        obstacle_limits = [5, 5, 5] * n_obstacles + [1] * n_obstacles  # Obstacle pos and range mask
        obs_limits = drone_limits + gate_limits + obstacle_limits + [n_gates]  # [1] for gate_id
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)
        self.observation_parser = None

        # Reward values TODO: Load from YAML
        self.rewarder = Rewarder()

        self.pyb_client_id: int = env.env.PYB_CLIENT
        # Config and helper flags
        self.terminate_on_lap = terminate_on_lap
        self._reset_required = False
        # The original firmware wrapper requires a sim time as input to the step function. This
        # breaks the gymnasium interface. Instead, we keep track of the sim time here. On each step,
        # it is incremented by the control time step. On env reset, it is reset to 0.
        self._sim_time = 0.0
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

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
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
        # Store obstacle height for observation expansion during env steps.
        self._obstacle_height = info["obstacle_dimensions"]["height"]
        self.observation_parser = ObservationParser.from_initial_observation(obs, info)
        obs = self.observation_parser.get_observation().astype(np.float32)

        logger.debug(f"{colored('===Reset===', 'green')}")
        logger.debug(f"Available keys in info: {pprint.pformat(info.keys())}")
        logger.debug(f"Collision: {pprint.pformat(info['collision'])}")
        logger.debug(f"Finished: {pprint.pformat(info['task_completed'])}")

        # assert obs in self.observation_space, f"Invalid observation: {obs}"
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
        action = self._action_transform(action)
        # The firmware does not use the action input in the step function
        zeros = np.zeros(3)
        self.env.sendFullStateCmd(action[:3], zeros, zeros, action[3], zeros, self._sim_time)
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
        elif self.terminate_on_lap and done:  # Done, but last gate not passed -> terminate
            terminated = True
        # Increment the sim time after the step if we are not yet done.
        if not terminated and not truncated:
            self._sim_time += self.env.ctrl_dt
        self.observation_parser.update(obs, info)
        obs = self.observation_parser.get_observation().astype(np.float32)

        # Compute the custom reward since we cannot modify the firmware environment for the
        # competion.
        reward = self.rewarder.get_custon_reward(self.observation_parser, info)
        if obs not in self.observation_space:
            terminated = True
            reward = self.rewarder.collision

        logger.debug(f"{colored('===Step===', 'white')}")
        logger.debug(f"Collision: {pprint.pformat(info['collision'])}")
        logger.debug(f"Finished: {pprint.pformat(info['task_completed'])}")
        logger.debug(f"Reward: {colored(pprint.pformat(reward), map_reward_to_color(reward))}")

        self._reset_required = terminated or truncated
        return obs, reward, terminated, truncated, info

    def _action_transform(self, action: np.ndarray) -> np.ndarray:
        """Transform the action to the format expected by the firmware env.

        Scale the action from [-1, 1] to [-5, 5] for the position and [-pi, pi] for the yaw.

        Args:
            action: The action to transform.

        Returns:
            The transformed action.
        """
        # action_transform = np.zeros(14)
        # scaled_action = action * self.action_scale
        # action_transform[:3] = scaled_action[:3]
        # action_transform[9] = scaled_action[3]
        # return action_transform

        action = np.clip(action, -1, 1)  # Clip the action to be within [-1, 1]

        # Sample action from a Gaussian distribution centered on the drone's current position
        mean = self.observation_parser.drone_pos  # The current position of the drone
        std_dev = self.action_scale[:3] / 2  # Standard deviation for x, y, z. Adjust as needed.
        sampled_action = np.random.normal(mean[:3], std_dev)

        # Clip sampled_action to ensure it stays within bounds
        sampled_action = np.clip(sampled_action, -self.action_scale[:3], self.action_scale[:3])

        centered_action = np.zeros(4)
        centered_action[:3] = sampled_action
        centered_action[3] = action[3] * self.action_scale[3]  # Use the original yaw action

        return centered_action

    def render(self):
        """Render the environment.

        Used for compatibility with the gymnasium API. Checks if PyBullet was launched with an
        active GUI.

        Raises:
            AssertionError: If PyBullet was not launched with an active GUI.
        """
        assert self.pyb_client_id != -1, "PyBullet not initialized with active GUI"

    def observation_transform(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Transform the observation to include additional information.

        Args:
            obs: The observation to transform.
            info: Additional information to include in the observation.

        Returns:
            The transformed observation.
        """
        drone_pos = obs[0:6:2]
        drone_yaw = obs[8]
        # The initial info dict does not include the gate pose and range, but it does include the
        # nominal gate positions and types, which we can use as a fallback for the first step.
        initial = "nominal_gates_pos_and_type" in info
        gate_type = info["gate_types"]
        gate_pos = info["gates_pos"][:, :2]
        gate_yaw = info["gates_pos"][:, 5]
        if initial:
            gate_pos = info["nominal_gates_pos_and_type"][:, :2]
            gate_yaw = info["nominal_gates_pos_and_type"][:, 5]

        z = np.where(gate_type == 1, Z_LOW, Z_HIGH)  # Infer gate z position based on type
        gate_poses = np.concatenate([gate_pos, z[:, None], gate_yaw[:, None]], axis=1)
        obstacle_pos = info["obstacles_pos"][:, :3]
        obstacle_pos[:, 2] = self._obstacle_height
        gate_id = info["current_gate_id"]
        gates_in_range = info["gates_in_range"]
        obstacles_in_range = info["obstacles_in_range"]
        obs = np.concatenate(
            [
                drone_pos,
                [drone_yaw],
                gate_poses.flatten(),
                gates_in_range,
                obstacle_pos.flatten(),
                obstacles_in_range,
                [gate_id],
            ]
        )
        return obs


class DroneRacingObservationWrapper:
    """Wrapper to transform the observation space the firmware wrapper.

    This wrapper matches the observation space of the DroneRacingWrapper. See its definition for
    more details. While we want to transform the observation space, we do not want to change the API
    of the firmware wrapper. Therefore, we create a separate wrapper for the observation space.

    Note:
        This wrapper is not a subclass of the gymnasium ObservationWrapper because the firmware is
        not compatible with the gymnasium API.
    """

    def __init__(self, env: FirmwareWrapper):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        if not isinstance(env, FirmwareWrapper):
            raise TypeError(f"`env` must be an instance of `FirmwareWrapper`, is {type(env)}")
        self.env = env
        self.pyb_client_id: int = env.env.PYB_CLIENT
        self.observation_parser = None
        self.rewarder = Rewarder()  # TODO: Load from YAML

    def __getattribute__(self, name: str) -> Any:
        """Get an attribute from the object.

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
        self.observation_parser = ObservationParser.from_initial_observation(obs, info)
        obs = self.observation_parser.get_observation()

        logger.debug(f"{colored('===Reset===', 'green')}")
        logger.debug(f"Available keys in info: {pprint.pformat(info.keys())}")
        logger.debug(f"Collision: {pprint.pformat(info['collision'])}")
        logger.debug(f"Finished: {pprint.pformat(info['task_completed'])}")

        return obs, info

    def step(
        self, *args: Any, **kwargs: dict[str, Any]
    ) -> tuple[np.ndarray, float, bool, dict, np.ndarray]:
        """Take a step in the current environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The transformed observation and the info dict.
        """
        obs, reward, done, info, action = self.env.step(*args, **kwargs)
        self.observation_parser.update(obs, info)
        obs = self.observation_parser.get_observation()
        reward = self.rewarder.get_custom_reward(self.observation_parser, info)

        logger.debug(f"{colored('===Step===', 'white')}")
        logger.debug(f"Collision: {pprint.pformat(info['collision'])}")
        logger.debug(f"Finished: {pprint.pformat(info['task_completed'])}")
        logger.debug(f"Reward: {colored(pprint.pformat(reward), map_reward_to_color(reward))}")

        return obs, reward, done, info, action
