"""Single drone racing environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from gymnasium import Env
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from lsy_drone_racing.envs.race_core import RaceCoreEnv, build_action_space, build_observation_space

if TYPE_CHECKING:
    from jax import Array
    from ml_collections import ConfigDict


class DroneRaceEnv(RaceCoreEnv, Env):
    """Single-agent drone racing environment."""

    def __init__(
        self,
        freq: int,
        sim_config: ConfigDict,
        sensor_range: float = 0.5,
        action_space: Literal["state", "attitude"] = "state",
        track: ConfigDict | None = None,
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        random_resets: bool = False,
        seed: int = 1337,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """Initialize the single-agent drone racing environment.

        Args:
            freq: Environment step frequency.
            sim_config: Simulation configuration.
            sensor_range: Sensor range.
            action_space: Control mode for the drones. See `build_action_space` for details.
            track: Track configuration.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            random_resets: Flag to reset the environment randomly.
            seed: Random seed.
            max_episode_steps: Maximum number of steps per episode.
            device: Device used for the environment and the simulation.
        """
        super().__init__(
            n_envs=1,
            n_drones=1,
            freq=freq,
            sim_config=sim_config,
            sensor_range=sensor_range,
            action_space=action_space,
            track=track,
            disturbances=disturbances,
            randomizations=randomizations,
            random_resets=random_resets,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.action_space = build_action_space(action_space)
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        self.observation_space = build_observation_space(n_gates, n_obstacles)
        self.autoreset = False

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional reset options. Not used.

        Returns:
            The initial observation and info.
        """
        obs, info = self._reset(seed=seed, options=options)
        obs = {k: v[0, 0] for k, v in obs.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        return obs, info

    def step(self, action: Array) -> tuple[dict, float, bool, bool, dict]:
        """Step the environment.

        Args:
            action: Action for the drone.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        obs, reward, terminated, truncated, info = self._step(action)
        obs = {k: v[0, 0] for k, v in obs.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        return obs, float(reward[0, 0]), bool(terminated[0, 0]), bool(truncated[0, 0]), info


class VecDroneRaceEnv(RaceCoreEnv, VectorEnv):
    """Vectorized single-agent drone racing environment."""

    def __init__(
        self,
        num_envs: int,
        freq: int,
        sim_config: ConfigDict,
        sensor_range: float = 0.5,
        action_space: Literal["state", "attitude"] = "state",
        track: ConfigDict | None = None,
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        random_resets: bool = False,
        seed: int = 1337,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """Initialize the vectorized single-agent drone racing environment.

        Args:
            num_envs: Number of worlds in the vectorized environment.
            freq: Environment step frequency.
            sim_config: Simulation configuration.
            sensor_range: Sensor range.
            action_space: Control mode for the drones. See `build_action_space` for details.
            track: Track configuration.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            random_resets: Flag to reset the environment randomly.
            seed: Random seed.
            max_episode_steps: Maximum number of steps per episode.
            device: Device used for the environment and the simulation.
        """
        super().__init__(
            n_envs=num_envs,
            n_drones=1,
            freq=freq,
            sim_config=sim_config,
            sensor_range=sensor_range,
            action_space=action_space,
            track=track,
            disturbances=disturbances,
            randomizations=randomizations,
            random_resets=random_resets,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.single_action_space = build_action_space(action_space)
        self.action_space = batch_space(self.single_action_space, num_envs)
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        self.single_observation_space = build_observation_space(n_gates, n_obstacles)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment in all worlds.

        Args:
            seed: Random seed.
            options: Additional reset options. Not used.

        Returns:
            The initial observation and info.
        """
        obs, info = self._reset(seed=seed, options=options)
        obs = {k: v[:, 0] for k, v in obs.items()}
        info = {k: v[:, 0] for k, v in info.items()}
        return obs, info

    def step(self, action: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Step the environment in all worlds.

        Args:
            action: Action for all worlds, i.e., a batch of (n_envs, action_dim) arrays.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        obs, reward, terminated, truncated, info = self._step(action)
        obs = {k: v[:, 0] for k, v in obs.items()}
        info = {k: v[:, 0] for k, v in info.items()}
        return obs, reward[:, 0], terminated[:, 0], truncated[:, 0], info
