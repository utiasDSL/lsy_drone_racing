import warnings
from pathlib import Path

import gymnasium
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config


@pytest.mark.unit
@pytest.mark.parametrize("control_mode", ["state", "attitude"])
def test_passive_checker_wrapper_warnings(control_mode: str):
    """Check passive env checker wrapper warnings.

    We disable the passive env checker by default. This test ensures that unexpected warnings are
    still seen, and even raises them to an exception.
    """
    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    with warnings.catch_warnings(record=True):  # Catch unnecessary warnings from gymnasium
        env = gymnasium.make(
            "DroneRacing-v0",
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            control_mode=control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=config.env.seed,
            disable_env_checker=False,
        )
        check_env(JaxToNumpy(env.unwrapped))
    env.close()

    config = load_config(Path(__file__).parents[3] / "config/multi_level0.toml")
    with warnings.catch_warnings(record=True):
        env = gymnasium.make(
            "MultiDroneRacing-v0",
            freq=config.env.kwargs[0]["freq"],
            sim_config=config.sim,
            sensor_range=config.env.kwargs[0]["sensor_range"],
            control_mode=control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=config.env.seed,
            disable_env_checker=False,
        )
        check_env(JaxToNumpy(env.unwrapped))
    env.close()

    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    with warnings.catch_warnings(record=True):  # Catch unnecessary warnings from gymnasium
        env = gymnasium.make_vec(
            "DroneRacing-v0",
            num_envs=2,
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            control_mode=control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=config.env.seed,
        )
        env = gymnasium.wrappers.vector.JaxToNumpy(env)
        # Check vector env specific attributes
        assert hasattr(env, "num_envs")
        assert isinstance(env.num_envs, int)
        assert env.num_envs == 2
        assert hasattr(env, "single_observation_space")
        assert hasattr(env, "single_action_space")
        obs, info = env.reset()
        assert obs in env.observation_space
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs in env.observation_space
        assert reward.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
    env.close()

    config = load_config(Path(__file__).parents[3] / "config/multi_level0.toml")
    with warnings.catch_warnings(record=True):  # Catch unnecessary warnings from gymnasium
        env = gymnasium.make_vec(
            "MultiDroneRacing-v0",
            num_envs=3,
            freq=config.env.kwargs[0]["freq"],
            sim_config=config.sim,
            sensor_range=config.env.kwargs[0]["sensor_range"],
            control_mode=control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=config.env.seed,
        )
        env = gymnasium.wrappers.vector.JaxToNumpy(env)
        # Check vector env specific attributes
        assert hasattr(env, "num_envs")
        assert isinstance(env.num_envs, int)
        assert env.num_envs == 3
        assert hasattr(env, "single_observation_space")
        assert hasattr(env, "single_action_space")
        obs, info = env.reset()
        assert obs in env.observation_space
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs in env.observation_space
        assert reward.shape == (3, 2)  # 2 envs, 3 drones
        assert terminated.shape == (3, 2)  # 2 envs, 3 drones
        assert truncated.shape == (3, 2)  # 2 envs, 3 drones
    env.close()
