import importlib
from pathlib import Path

import gymnasium
import pytest

from lsy_drone_racing.envs.drone_racing_env import DroneRacingEnv
from lsy_drone_racing.sim.physics import PhysicsMode
from lsy_drone_racing.utils import load_config


@pytest.mark.integration
@pytest.mark.skipif(
    not importlib.util.find_spec("stable_baselines3"),
    reason="requires the stable baselines3 library",
)
def test_sb3_ppo():
    """Test training with sb3 PPO."""
    from stable_baselines3.ppo import PPO

    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.physics = PhysicsMode.DEFAULT
    env = gymnasium.make("DroneRacing-v0", config=config)
    model = PPO("MultiInputPolicy", env, verbose=1, n_epochs=2, n_steps=2, batch_size=2)
    model.learn(total_timesteps=1)


@pytest.mark.integration
@pytest.mark.skipif(
    not importlib.util.find_spec("stable_baselines3"),
    reason="requires the stable baselines3 library",
)
def test_sb3_ppo_vec():
    """Test training with sb3 PPO and subprocess vecenvs."""
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.ppo import PPO

    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.physics = PhysicsMode.DEFAULT

    def _make_env() -> DroneRacingEnv:
        import lsy_drone_racing  # noqa: F401, register the env with gymnasium

        return gymnasium.make("DroneRacing-v0", config=config)

    env = make_vec_env(
        _make_env, n_envs=2, vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "spawn"}
    )
    model = PPO("MultiInputPolicy", env, verbose=1, n_epochs=2, n_steps=2, batch_size=2)
    model.learn(total_timesteps=1)
