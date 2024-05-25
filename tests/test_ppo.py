from typing import Generator

import pytest
from safe_control_gym.controllers.firmware.firmware_wrapper import FirmwareWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO

from lsy_drone_racing.wrapper import DroneRacingWrapper, MultiProcessingWrapper
from tests.utils import make_env


@pytest.fixture(scope="session")
def env() -> Generator[FirmwareWrapper, None, None]:
    """Create the drone racing environment."""
    yield make_env()


@pytest.mark.parametrize("terminate_on_lap", [True, False])
def test_sb3_ppo(env: FirmwareWrapper, terminate_on_lap: bool):
    """Test training with sb3 PPO."""
    env = DroneRacingWrapper(env, terminate_on_lap=terminate_on_lap)
    model = PPO("MlpPolicy", env, verbose=1, n_epochs=2, n_steps=2, batch_size=2)
    model.learn(total_timesteps=1)


@pytest.mark.parametrize("terminate_on_lap", [True, False])
def test_sb3_ppo_vec(env: FirmwareWrapper, terminate_on_lap: bool):
    """Test training with sb3 PPO and subprocess vecenvs."""
    env = make_vec_env(
        lambda: MultiProcessingWrapper(DroneRacingWrapper(make_env())),
        n_envs=2,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    model = PPO("MlpPolicy", env, verbose=1, n_epochs=2, n_steps=2, batch_size=2)
    model.learn(total_timesteps=1)
