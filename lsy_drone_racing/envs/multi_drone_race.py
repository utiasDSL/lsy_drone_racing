"""Multi-agent drone racing environments."""

from typing import Literal

import numpy as np
from gymnasium import Env
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from ml_collections import ConfigDict
from numpy.typing import NDArray

from lsy_drone_racing.envs.race_core import RaceCoreEnv, build_action_space, build_observation_space


class MultiDroneRaceEnv(RaceCoreEnv, Env):
    """Multi-agent drone racing environment.

    This environment enables multiple agents to simultaneously compete with each other on the same
    track.
    """

    def __init__(
        self,
        n_drones: int,
        freq: int,
        sim_config: ConfigDict,
        sensor_range: float,
        action_space: Literal["state", "attitude"] = "state",
        track: ConfigDict | None = None,
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        random_resets: bool = False,
        seed: int = 1337,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """Initialize the multi-agent drone racing environment.

        Args:
            n_drones: Number of drones.
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
            n_drones=n_drones,
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
        self.action_space = batch_space(build_action_space(action_space), n_drones)
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        self.observation_space = batch_space(
            build_observation_space(n_gates, n_obstacles), n_drones
        )
        self.autoreset = False

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment for all drones.

        Args:
            seed: Random seed.
            options: Additional reset options. Not used.

        Returns:
            Observation and info for all drones.
        """
        obs, info = self._reset(seed=seed, options=options)
        obs = {k: v[0] for k, v in obs.items()}
        info = {k: v[0] for k, v in info.items()}
        return obs, info

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict, NDArray[np.floating], NDArray[np.bool_], NDArray[np.bool_], dict]:
        """Step the environment for all drones.

        Args:
            action: Action for all drones, i.e., a batch of (n_drones, action_dim) arrays.

        Returns:
            Observation, reward, terminated, truncated, and info for all drones.
        """
        obs, reward, terminated, truncated, info = self._step(action)
        obs = {k: v[0] for k, v in obs.items()}
        info = {k: v[0] for k, v in info.items()}
        return obs, reward[0], terminated[0], truncated[0], info


class VecMultiDroneRaceEnv(RaceCoreEnv, VectorEnv):
    """Vectorized multi-agent drone racing environment.

    This environment enables vectorized training of multi-agent drone racing agents.
    """

    def __init__(
        self,
        num_envs: int,
        n_drones: int,
        freq: int,
        sim_config: ConfigDict,
        sensor_range: float,
        action_space: Literal["state", "attitude"] = "state",
        track: ConfigDict | None = None,
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        random_resets: bool = False,
        seed: int = 1337,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """Vectorized multi-agent drone racing environment.

        Args:
            num_envs: Number of worlds in the vectorized environment.
            n_drones: Number of drones in each world.
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
            n_drones=n_drones,
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
        self.single_action_space = batch_space(build_action_space(action_space), n_drones)
        self.action_space = batch_space(batch_space(self.single_action_space), num_envs)
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        self.single_observation_space = batch_space(
            build_observation_space(n_gates, n_obstacles), n_drones
        )
        self.observation_space = batch_space(batch_space(self.single_observation_space), num_envs)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment for all drones.

        Args:
            seed: Random seed.
            options: Additional reset options. Not used.

        Returns:
            Observation and info for all drones.
        """
        return self._reset(seed=seed, options=options)

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict, NDArray[np.floating], NDArray[np.bool_], NDArray[np.bool_], dict]:
        """Step the environment for all drones.

        Args:
            action: Action for all drones, i.e., a batch of (n_drones, action_dim) arrays.
        """
        return self._step(action)
