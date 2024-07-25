"""Example training script for deep RL using the stable-baselines3 library.

Note:
    This script requires you to install the stable-baselines3 library.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)


def main(config: str = "config/getting_started.toml", gui: bool = False):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=logging.INFO)
    config = load_config(Path(__file__).resolve().parents[1] / config)
    config.sim.gui = gui
    env = gymnasium.make("DroneRacing-v0", config=config)
    check_env(env)  # Sanity check to ensure the environment conforms to the sb3 API
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=4096)


if __name__ == "__main__":
    fire.Fire(main)
