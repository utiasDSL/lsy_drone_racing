from pathlib import Path
from typing import Generator

import gymnasium
import pytest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO

from lsy_drone_racing.envs.drone_racing_env import DroneRacingEnv
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.wrapper import DroneRacingWrapper, MultiProcessingWrapper


@pytest.fixture(scope="session")
def env() -> Generator[DroneRacingEnv, None, None]:
    """Create the drone racing environment."""
    config = load_config(Path(__file__).parents[1] / "config/test.toml")
    yield gymnasium.make("DroneRacing-v0", config=config)


@pytest.mark.integration
def test_sb3_ppo(env: DroneRacingEnv):
    """Test training with sb3 PPO."""
    env = DroneRacingWrapper(env)
    model = PPO("MlpPolicy", env, verbose=1, n_epochs=2, n_steps=2, batch_size=2)
    model.learn(total_timesteps=1)


@pytest.mark.integration
def test_sb3_ppo_vec(env: DroneRacingEnv):
    """Test training with sb3 PPO and subprocess vecenvs."""
    config = load_config(Path(__file__).parents[1] / "config/test.toml")
    env = make_vec_env(
        lambda: MultiProcessingWrapper(
            DroneRacingWrapper(gymnasium.make("DroneRacing-v0", config=config))
        ),
        n_envs=2,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    model = PPO("MlpPolicy", env, verbose=1, n_epochs=2, n_steps=2, batch_size=2)
    model.learn(total_timesteps=1)
