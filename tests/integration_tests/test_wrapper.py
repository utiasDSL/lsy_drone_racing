from pathlib import Path
from typing import Generator

import gymnasium
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from lsy_drone_racing.envs.drone_racing_env import DroneRacingEnv
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.wrapper import (
    DroneRacingObservationWrapper,
    DroneRacingWrapper,
    MultiProcessingWrapper,
    RewardWrapper,
)


@pytest.fixture(scope="session")
def env() -> Generator[DroneRacingEnv, None, None]:
    """Create the drone racing environment."""
    config = load_config(Path(__file__).parents[1] / "config/test.toml")
    yield gymnasium.make("DroneRacing-v0", config=config)


@pytest.mark.parametrize("terminate_on_lap", [True, False])
@pytest.mark.integration
def test_drone_racing_wrapper(env: DroneRacingEnv, terminate_on_lap: bool):
    """Test the DroneRacingWrapper."""
    env = DroneRacingWrapper(env, terminate_on_lap=terminate_on_lap)
    env.reset()
    env.step(env.action_space.sample())


@pytest.mark.parametrize("terminate_on_lap", [True, False])
@pytest.mark.integration
def test_drone_racing_wrapper_sb3(env: DroneRacingEnv, terminate_on_lap: bool):
    """Test the DroneRacingWrapper for compatibility with sb3's API."""
    check_env(DroneRacingWrapper(env, terminate_on_lap=terminate_on_lap))


@pytest.mark.integration
def test_obs_wrapper(env: DroneRacingEnv):
    """Test the DroneRacingObservationWrapper."""
    DroneRacingObservationWrapper(env)


@pytest.mark.integration
def test_reward_wrapper(env: DroneRacingEnv):
    """Test the DroneRacingRewardWrapper."""
    env = RewardWrapper(DroneRacingWrapper(env))
    env.reset()
    env.step(env.action_space.sample())


@pytest.mark.integration
def test_reward_wrapper_sb3(env: DroneRacingEnv):
    """Test the DroneRacingRewardWrapper for compatibility with sb3's API."""
    check_env(RewardWrapper(DroneRacingWrapper(env)))


@pytest.mark.integration
def test_multiprocessing_wrapper():
    """Test the MultiProcessingWrapper."""
    config = load_config(Path(__file__).parents[1] / "config/test.toml")
    env = make_vec_env(
        lambda: MultiProcessingWrapper(
            DroneRacingWrapper(gymnasium.make("DroneRacing-v0", config=config))
        ),
        n_envs=2,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    env.reset()
    env.step(np.array([env.action_space.sample()] * 2))


@pytest.mark.integration
def test_multiprocessing_wrapper_sb3():
    """Test if the multiprocessing wrapper can be used for sb3's vecenv."""
    config = load_config(Path(__file__).parents[1] / "config/test.toml")
    config.env.symbolic = True  # Enforce symbolic models to check if the wrapper can handle them
    env = make_vec_env(
        lambda: MultiProcessingWrapper(
            DroneRacingWrapper(gymnasium.make("DroneRacing-v0", config=config))
        ),
        n_envs=2,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    env.reset()
    env.step(np.array([env.action_space.sample()] * 2))


@pytest.mark.integration
def test_sb3_dummy_vec():
    """Test if the environment can be used for sb3's DummyVecEnv."""
    config = load_config(Path(__file__).parents[1] / "config/test.toml")
    env = make_vec_env(
        lambda: DroneRacingWrapper(gymnasium.make("DroneRacing-v0", config=config)),
        n_envs=2,
        vec_env_cls=DummyVecEnv,
    )
    env.reset()
    env.step(np.array([env.action_space.sample()] * 2))
