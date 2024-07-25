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
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import Env, Wrapper
from gymnasium.error import InvalidAction
from gymnasium.spaces import Box

if TYPE_CHECKING:
    from lsy_drone_racing.envs.drone_racing_env import DroneRacingEnv

logger = logging.getLogger(__name__)


class DroneRacingWrapper(Wrapper):
    """Drone racing firmware wrapper to make the environment compatible with the gymnasium API.

    In contrast to the underlying environment, this wrapper only accepts FullState commands as
    actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: DroneRacingEnv, terminate_on_lap: bool = True):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
            terminate_on_lap: Stop the simulation early when the drone has passed the last gate.
        """
        super().__init__(env)
        # Gymnasium env required attributes
        # Action space:
        # [x, y, z, yaw]
        # x, y, z)  The desired position of the drone in the world frame.
        # yaw)      The desired yaw angle.
        # All values are scaled to [-1, 1]. Transformed back, x, y, z values of 1 correspond to 5m.
        # The yaw value of 1 corresponds to pi radians.
        self.action_scale = np.array([1, 1, 1])
        self.action_space = Box(-1, 1, shape=(3,), dtype=np.float32)

        # Observation space:
        # [drone_xyz, drone_rpy, drone_vxyz, drone vrpy, gates_xyz_yaw, gates_in_range,
        # obstacles_xyz, obstacles_in_range, gate_id]
        # drone_xyz)  Drone position in meters.
        # drone_rpy)  Drone orientation in radians.
        # drone_vxyz)  Drone velocity in m/s.
        # drone_vrpy)  Drone angular velocity in rad/s.
        # gates_xyz_yaw)  The pose of the gates. Positions are in meters and yaw in radians. The
        #       length is dependent on the number of gates. Ordering is [x0, y0, z0, yaw0, x1,...].
        # gates_in_range)  A boolean array indicating if the drone is within the gates' range. The
        #       length is dependent on the number of gates.
        # obstacles_xyz)  The pose of the obstacles. Positions are in meters. The length is
        #       dependent on the number of obstacles. Ordering is [x0, y0, z0, x1,...].
        # obstacles_in_range)  A boolean array indicating if the drone is within the obstacles'
        #       range. The length is dependent on the number of obstacles.
        # gate_id)  The ID of the current target gate. -1 if the task is completed.
        n_gates = self.env.unwrapped.sim.n_gates
        n_obstacles = self.env.unwrapped.sim.n_obstacles
        # Velocity limits are set to 10 m/s for the drone and 10 rad/s for the angular velocity.
        # While drones could go faster in theory, it's not safe in practice and we don't allow it in
        # sim either.
        drone_limits = [5, 5, 5, np.pi, np.pi, np.pi, 10, 10, 10, 10, 10, 10]
        gate_limits = [5, 5, 5, np.pi] * n_gates + [1] * n_gates  # Gate poses and range mask
        obstacle_limits = [5, 5, 5] * n_obstacles + [1] * n_obstacles  # Obstacle pos and range mask
        obs_limits = drone_limits + gate_limits + obstacle_limits + [n_gates]  # [1] for gate_id
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)
        self.pyb_client: int = self.env.unwrapped.sim.pyb_client
        self._drone_pose = None

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
        obs, info = self.env.reset()
        self._drone_pos = obs[:3]
        obs = self.observation_transform(obs, info).astype(np.float32)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        if action not in self.action_space:
            raise InvalidAction(f"Invalid action: {action}")
        action = self._action_transform(action)
        assert action.shape[-1] == 3, "Action must have shape (..., 3)"
        obs, reward, terminated, truncated, info = self.env.step(np.concatenate([action, [0]]))
        obs = self.observation_transform(obs, info).astype(np.float32)
        self._drone_pos = obs[:3]
        return obs, reward, terminated, truncated, info

    def _action_transform(self, action: np.ndarray) -> np.ndarray:
        """Transform the action to the format expected by the firmware env.

        Args:
            action: The action to transform.

        Returns:
            The transformed action.
        """
        return self._drone_pos + (action * self.action_scale)

    @staticmethod
    def observation_transform(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Transform the observation to include additional information.

        Args:
            obs: The observation to transform.
            info: Additional information to include in the observation.

        Returns:
            The transformed observation.
        """
        drone_pos = obs[0:6:2]
        drone_vel = obs[1:6:2]
        drone_rpy = obs[6:9]
        drone_ang_vel = obs[9:12]
        gates_pos, gates_yaw = info["gates.pos"], info["gates.rpy"][:, 2]
        gates_pose = np.concatenate(
            [np.concatenate([p, [y]]) for p, y in zip(gates_pos, gates_yaw)]
        )
        obs = np.concatenate(
            [
                drone_pos,
                drone_rpy,
                drone_vel,
                drone_ang_vel,
                gates_pose,
                info["gates.in_range"],
                info["obstacles.pos"].flatten(),
                info["obstacles.in_range"],
                [info["target_gate"]],
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

    def __init__(self, env: DroneRacingEnv):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        # if not isinstance(env, DroneRacingEnv):
        #     raise TypeError(f"`env` must be an instance of `DroneRacingEnv`, is {type(env)}")
        self.env = env.unwrapped
        self.pyb_client: int = self.env.unwrapped.sim.pyb_client

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
        obs = DroneRacingWrapper.observation_transform(obs, info)
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
        obs, reward, terminated, truncated, info = self.env.step(*args, **kwargs)
        obs = DroneRacingWrapper.observation_transform(obs, info)
        return obs, reward, terminated, truncated, info


class MultiProcessingWrapper(Wrapper):
    """Wrapper to enable multiprocessing for vectorized environments.

    The info dict returned by the environment may contain CasADi models. These models cannot be
    pickled and therefore cannot be passed between processes. This wrapper overrides the environment
    settings to enfoce the removal of the CasADi models to enable multiprocessing.

    Alternatively, the symbolic models can be removed in the config file by setting
    `env.symbolic = False`.
    """

    def __init__(self, env: Env):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        super().__init__(env)
        self.env.unwrapped.symbolic = False


class RewardWrapper(Wrapper):
    """Wrapper to alter the default reward function from the environment for RL training."""

    def __init__(self, env: Env):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        super().__init__(env)
        self._last_gate = None
        self._last_action = None

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> np.ndarray:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The initial observation of the next episode.
        """
        obs, info = self.env.reset(*args, **kwargs)
        self._last_gate = info["target_gate"]
        self._last_pos = info["drone.pos"]
        self._last_action = None
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(obs, action, reward, terminated, truncated, info)
        self._last_gate = info["target_gate"]
        self._last_pos = info["drone.pos"]
        self._last_action = action
        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> float:
        """Compute the reward for the current step.

        Args:
            obs: The current observation.
            action: The action taken in the current step.
            reward: The reward from the environment.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: Additional information from the environment.

        Returns:
            The computed reward.
        """
        gate_id = info["target_gate"]
        drone_pos = info["drone.pos"]
        if self._last_action is not None:
            action_penalty = -0.005 * np.linalg.norm(action - self._last_action)
        else:
            action_penalty = 0
        gate_distance = np.linalg.norm(info["gates.pos"][gate_id] - drone_pos)
        old_gate_distance = np.linalg.norm(info["gates.pos"][gate_id] - self._last_pos)
        gate_reward = (old_gate_distance - gate_distance) * 1.0
        gate_reward += (gate_id != self._last_gate) * 1.0
        crash_penalty = -2.0 if len(info["collisions"]) > 0 else 0.0
        reward = gate_reward + crash_penalty + action_penalty
        return reward


class ObsWrapper(Wrapper):
    def __init__(self, env: DroneRacingWrapper):
        super().__init__(env)
        n_gates, n_obstacles = 4, 4
        self.n_gates = n_gates
        drone_limits = [5, 5, 5, np.pi, np.pi, np.pi, 10, 10, 10, 10, 10, 10]
        gate_limits = [5, 5, 5, np.pi] * n_gates + [1] * n_gates  # Gate poses and range mask
        obstacle_limits = [5, 5, 5] * n_obstacles + [1] * n_obstacles  # Obstacle pos and range mask

        to_gate_limits = [10, 10, 10] * n_gates
        to_obstacles_limits = [10, 10, 10] * n_obstacles
        obs_limits = (
            drone_limits
            + gate_limits
            + obstacle_limits
            + [1] * n_gates
            + to_gate_limits
            + to_obstacles_limits
            + [10, 10, 10]
            + [1, 1]
            + [1, 1, 1]
        )
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> np.ndarray:
        obs, info = self.env.reset(*args, **kwargs)
        return self.observation_transform(obs, info, None), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation_transform(obs, info, action), reward, terminated, truncated, info

    @staticmethod
    def _onehot(idx: int, n_classes: int) -> np.ndarray:
        onehot = np.zeros(n_classes)
        if idx != -1:
            onehot[idx] = 1
        return onehot

    @staticmethod
    def observation_transform(obs: np.ndarray, info: dict, action: np.ndarray | None) -> np.ndarray:
        to_gates = info["gates.pos"] - obs[:3]
        gate_vec = info["gates.pos"][info["target_gate"]]  # TODO: BUG, Change to diff vector
        gate_angle = info["gates.rpy"][info["target_gate"], 2]
        gate_vec /= np.linalg.norm(gate_vec)
        gate_direction = np.array([np.cos(gate_angle), np.sin(gate_angle)])
        to_obstacles = info["obstacles.pos"] - obs[:3]
        gate_id_onehot = ObsWrapper._onehot(info["target_gate"], info["gates.pos"].shape[0])
        obs = np.concatenate(
            [
                obs[:-1],
                gate_id_onehot,
                to_gates.flatten(),
                to_obstacles.flatten(),
                gate_vec,
                gate_direction,
                np.zeros(3) if action is None else action,
            ]
        )
        return obs
