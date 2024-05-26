from typing import Generator

import numpy as np
import pytest
from safe_control_gym.controllers.firmware.firmware_wrapper import FirmwareWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from lsy_drone_racing.wrapper import (
    DroneRacingObservationWrapper,
    DroneRacingWrapper,
    MultiProcessingWrapper,
    RewardWrapper,
)
from tests.utils import make_env


@pytest.fixture(scope="session")
def env() -> Generator[FirmwareWrapper, None, None]:
    """Create the drone racing environment."""
    yield make_env()


@pytest.mark.parametrize("terminate_on_lap", [True, False])
def test_drone_racing_wrapper(env: FirmwareWrapper, terminate_on_lap: bool):
    """Test the DroneRacingWrapper."""
    env = DroneRacingWrapper(env, terminate_on_lap=terminate_on_lap)
    env.reset()
    env.step(env.action_space.sample())


@pytest.mark.parametrize("terminate_on_lap", [True, False])
def test_drone_racing_wrapper_sb3(env: FirmwareWrapper, terminate_on_lap: bool):
    """Test the DroneRacingWrapper for compatibility with sb3's API."""
    check_env(DroneRacingWrapper(env, terminate_on_lap=terminate_on_lap))


def test_obs_wrapper(env: FirmwareWrapper):
    """Test the DroneRacingObservationWrapper."""
    DroneRacingObservationWrapper(env)


def test_reward_wrapper(env: FirmwareWrapper):
    """Test the DroneRacingRewardWrapper."""
    env = RewardWrapper(DroneRacingWrapper(env))
    env.reset()
    env.step(env.action_space.sample())


def test_reward_wrapper_sb3(env: FirmwareWrapper):
    """Test the DroneRacingRewardWrapper for compatibility with sb3's API."""
    check_env(RewardWrapper(DroneRacingWrapper(env)))


def test_multiprocessing_wrapper():
    """Test the MultiProcessingWrapper."""
    env = make_vec_env(
        lambda: MultiProcessingWrapper(DroneRacingWrapper(make_env())),
        n_envs=2,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    env.reset()
    env.step(np.array([env.action_space.sample()] * 2))


def test_multiprocessing_wrapper_sb3():
    """Test if the multiprocessing wrapper can be used for sb3's vecenv."""
    env = make_vec_env(
        lambda: MultiProcessingWrapper(DroneRacingWrapper(make_env())),
        n_envs=2,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    env.reset()
    env.step(np.array([env.action_space.sample()] * 2))


def test_sb3_dummy_vec():
    """Test if the environment can be used for sb3's DummyVecEnv."""
    env = make_vec_env(lambda: DroneRacingWrapper(make_env()), n_envs=2, vec_env_cls=DummyVecEnv)
    env.reset()
    env.step(np.array([env.action_space.sample()] * 2))
